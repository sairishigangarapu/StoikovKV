"""
AtlasKV-Inspired Persistent External KV Memory Bank (`Doc2KV` / `KG2KV`) and
Rectangular Attention Adapter for StoikovKV (`stoikov_kv/atlas_memory.py`).

Design Highlights:
1. FlashAttention / SDPA-Compatible Pre-RoPE Indexing (`Doc2KV` & `KG2KV`):
   - Indexes arbitrary document corpora or Knowledge Graph `(subject, relation, object)`
     triples into a persistent hierarchical `HiKVPStore` via lightweight forward hooks on
     each decoder layer's `self_attn.k_proj` and `self_attn.v_proj`.
   - Runs model forward passes with `use_cache=False, output_attentions=False`, so full
     `T x T` attention matrices are NEVER materialized.
   - Captures keys directly from `k_proj` output in the position-free **pre-RoPE frame**
     prior to rotary embedding, enabling position-decoupled retrieval and clean RoPE
     re-indexing during Rectangular Attention.
2. Architecture Compatibility:
   - Works identically on CPU (`CUDA_VISIBLE_DEVICES=""`) and CUDA GPUs.
   - Works with both HuggingFace Qwen2 / Llama CausalLM models and the standalone
     `FallbackQwen2ForCausalLM` model from `stoikov_kv.calibration`.
3. Rectangular Attention Adapter:
   - `AtlasMemoryBank.retrieve_for_layer(...)` fetches per-head top-k pre-RoPE `(K, V)`
     leaves from `HiKVPStore` and aligns head lengths with zero-padded tails into uniform
     `[B, H_kv, T_atlas, D]` tensors ready for SDPA Rectangular Attention.
   - `apply_rope_to_atlas_keys(...)` and `rectangular_attention(...)` re-index retrieved
     pre-RoPE Atlas keys to virtual prefix positions and compute GQA-aware SDPA over
     concatenated `[Atlas prefix | Live context]` KV states.
4. Persistence:
   - `save(directory)` / `load(directory, U_Q=None)` persist both the underlying
     `HiKVPStore` (`hikvp_store.pt`) and chunk/RoPE provenance (`atlas_meta.json`).
"""

from __future__ import annotations

import json
import math
import os
import stat
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._security import (
    safe_open_regular_read,
    secure_atomic_write_bytes,
    secure_makedirs,
    validate_safe_path,
)
from .hikvp import HiKVPConfig, HiKVPStore

ATLAS_BANK_VERSION: str = "1.0.0"


@dataclass
class AtlasTriple:
    """Knowledge-graph triple `(subject, relation, obj)` for KG2KV indexing."""

    subject: str
    relation: str
    obj: str


@dataclass
class AtlasDocumentChunk:
    """Metadata describing a single indexed token chunk inside an `AtlasMemoryBank`."""

    chunk_id: int
    source: str
    text: str
    start_pos: int
    num_tokens: int


def format_kg_triples(
    triples: Sequence[Union[AtlasTriple, Tuple[str, str, str]]],
) -> List[str]:
    """
    Converts `(subject, relation, object)` triples into compact, attention-native
    factual sentences of the form `"{subject} | {relation}: {obj}."`.
    """
    formatted: List[str] = []
    for item in triples:
        if isinstance(item, AtlasTriple):
            subj, rel, obj = item.subject, item.relation, item.obj
        elif isinstance(item, (tuple, list)) and len(item) == 3:
            subj, rel, obj = item[0], item[1], item[2]
        else:
            raise ValueError(
                f"Expected AtlasTriple or 3-tuple (subject, relation, obj), got: {item!r}"
            )
        s_clean = " ".join(str(subj).strip().split())
        r_clean = " ".join(str(rel).strip().split())
        o_clean = " ".join(str(obj).strip().split()).rstrip(".")
        formatted.append(f"{s_clean} | {r_clean}: {o_clean}.")
    return formatted


def consolidate_kg_triples(
    triples: Sequence[Union[AtlasTriple, Tuple[str, str, str]]],
    max_inverse_subjects: int = 16,
) -> List[str]:
    """
    MeMo-inspired (`arXiv:2605.15156`) zero-SFT CPU entity consolidation & bidirectional
    surfacing for `KG2KV`:
    1. Deduplicates and groups outgoing attributes per subject (`"{S} | {r1}: {o1}; {r2}: {o2}."`)
       so related facts co-locate in sharp-centroid HiKVP leaf pages.
    2. Emits bounded inverse entity records (`"{O} | inverse_{r}: {s1}, {s2}."`) so `{s_i}` tokens
       follow `{O}` in causal order, curing forward-only `KG2KV` causal value blindness in O(N) CPU.
    """
    fwd: Dict[str, List[str]] = {}
    inv: Dict[Tuple[str, str], List[str]] = {}
    cap = max(1, int(max_inverse_subjects))
    for item in triples:
        if isinstance(item, AtlasTriple):
            subj, rel, obj = item.subject, item.relation, item.obj
        elif isinstance(item, (tuple, list)) and len(item) == 3:
            subj, rel, obj = item[0], item[1], item[2]
        else:
            raise ValueError(
                f"Expected AtlasTriple or 3-tuple (subject, relation, obj), got: {item!r}"
            )
        s_c = " ".join(str(subj).strip().split())
        r_c = " ".join(str(rel).strip().split())
        o_c = " ".join(str(obj).strip().split()).rstrip(".")
        fwd.setdefault(s_c, []).append(f"{r_c}: {o_c}")
        if o_c:
            bucket = inv.setdefault((o_c, r_c), [])
            if len(bucket) < cap and s_c not in bucket:
                bucket.append(s_c)
    out = [f"{s_c} | {'; '.join(rels)}." for s_c, rels in fwd.items()]
    out.extend(
        f"{o_c} | inverse_{r_c}: {', '.join(subs)}."
        for (o_c, r_c), subs in inv.items()
        if subs
    )
    return out


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input for Rotary Position Embeddings (RoPE)."""
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope_to_atlas_keys(
    k_pre: torch.Tensor,
    positions: torch.Tensor,
    rope_theta: float = 1000000.0,
) -> torch.Tensor:
    """
    Applies Rotary Position Embeddings (RoPE) to pre-RoPE Atlas keys at arbitrary
    integer/float positions (including contiguous virtual prefix offsets).

    Args:
        k_pre: Pre-RoPE key tensor of shape `[B, H_kv, T, D]` or `[B, T, D]`.
        positions: 1D tensor of shape `[T]` specifying target RoPE positions.
        rope_theta: RoPE base frequency (default `1000000.0` for Qwen2.5).

    Returns:
        RoPE-rotated key tensor matching `k_pre.shape`, `k_pre.device`, and `k_pre.dtype`.
    """
    if k_pre.shape[-2] == 0:
        return k_pre

    head_dim = int(k_pre.shape[-1])
    t_len = int(k_pre.shape[-2])
    device = k_pre.device
    dtype = k_pre.dtype

    pos = positions.to(device=device, dtype=torch.float32).reshape(-1)
    if pos.shape[0] != t_len:
        raise ValueError(
            f"positions length ({pos.shape[0]}) must match key sequence length ({t_len})."
        )

    inv_freq = 1.0 / (
        float(rope_theta)
        ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / float(head_dim))
    )
    freqs = torch.outer(pos, inv_freq)  # [T, D // 2]
    emb = torch.cat((freqs, freqs), dim=-1)  # [T, D]
    cos = emb.cos().to(dtype=dtype)
    sin = emb.sin().to(dtype=dtype)

    while cos.dim() < k_pre.dim():
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)

    return (k_pre * cos) + (_rotate_half(k_pre) * sin)


def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expands GQA KV heads `[B, H_kv, T, D]` to query-head granularity `[B, H_q, T, D]`."""
    if n_rep == 1:
        return x
    bsz, n_kv, slen, head_dim = x.shape
    return (
        x[:, :, None, :, :]
        .expand(bsz, n_kv, n_rep, slen, head_dim)
        .reshape(bsz, n_kv * n_rep, slen, head_dim)
    )


def rectangular_attention(
    q_rope: torch.Tensor,
    k_live_rope: torch.Tensor,
    v_live: torch.Tensor,
    k_atlas_rope: Optional[torch.Tensor] = None,
    v_atlas: Optional[torch.Tensor] = None,
    atlas_valid_mask: Optional[torch.Tensor] = None,
    is_causal_live: bool = True,
) -> torch.Tensor:
    """
    SDPA / FlashAttention-compatible Rectangular Attention adapter combining retrieved
    external Atlas KV prefix tokens `[B, H_kv, T_atlas, D]` with live causal context KV
    tokens `[B, H_kv, T_live, D]`.

    Every query token in `q_rope` `[B, H_q, Q, D]` attends unconditionally to all valid
    Atlas prefix tokens `[0 .. T_atlas - 1]` and causally (when `is_causal_live=True` and
    `Q > 1`) to live KV tokens `[T_atlas .. T_atlas + T_live - 1]`.

    Args:
        q_rope: Query tensor `[B, H_q, Q, D]` (already RoPE-rotated).
        k_live_rope: Live context keys `[B, H_kv, T_live, D]` (already RoPE-rotated).
        v_live: Live context values `[B, H_kv, T_live, D]`.
        k_atlas_rope: Optional retrieved Atlas keys `[B, H_kv, T_atlas, D]` (RoPE-rotated).
        v_atlas: Optional retrieved Atlas values `[B, H_kv, T_atlas, D]`.
        atlas_valid_mask: Optional boolean mask `[T_atlas]` or `[B, H_kv, T_atlas]`
            where `True` indicates valid (non-padded) Atlas tokens.
        is_causal_live: Whether live tokens follow causal masking for `Q > 1`.

    Returns:
        Attention output tensor of shape `[B, H_q, Q, D]`.
    """
    bsz, num_q_heads, q_len, head_dim = q_rope.shape
    num_kv_heads = int(k_live_rope.shape[1])
    if num_q_heads % num_kv_heads != 0:
        raise ValueError(
            f"num_q_heads ({num_q_heads}) must be divisible by num_kv_heads ({num_kv_heads})."
        )
    q_per_kv = num_q_heads // num_kv_heads
    t_live = int(k_live_rope.shape[2])

    has_atlas = (
        k_atlas_rope is not None
        and v_atlas is not None
        and int(k_atlas_rope.shape[2]) > 0
    )

    if has_atlas:
        assert k_atlas_rope is not None and v_atlas is not None
        t_atlas = int(k_atlas_rope.shape[2])
        k_cat = torch.cat(
            [k_atlas_rope.to(device=q_rope.device, dtype=q_rope.dtype), k_live_rope],
            dim=2,
        )
        v_cat = torch.cat(
            [v_atlas.to(device=q_rope.device, dtype=q_rope.dtype), v_live],
            dim=2,
        )
    else:
        t_atlas = 0
        k_cat = k_live_rope
        v_cat = v_live

    t_total = t_atlas + t_live
    use_native_gqa = q_per_kv > 1 and head_dim <= 256

    # Fast path: single-token decode (Q == 1) without custom atlas mask needs no mask tensor.
    if q_len == 1 and atlas_valid_mask is None:
        if use_native_gqa:
            return F.scaled_dot_product_attention(
                q_rope, k_cat, v_cat, attn_mask=None, dropout_p=0.0, is_causal=False, enable_gqa=True
            )
        k_rep = _repeat_kv(k_cat, q_per_kv)
        v_rep = _repeat_kv(v_cat, q_per_kv)
        return F.scaled_dot_product_attention(
            q_rope, k_rep, v_rep, attn_mask=None, dropout_p=0.0, is_causal=False
        )

    # Fast path: no Atlas prefix, pure causal self-attention with Q == T_live.
    if not has_atlas and is_causal_live and q_len == t_live and atlas_valid_mask is None:
        if use_native_gqa:
            return F.scaled_dot_product_attention(
                q_rope, k_cat, v_cat, attn_mask=None, dropout_p=0.0, is_causal=True, enable_gqa=True
            )
        k_rep = _repeat_kv(k_cat, q_per_kv)
        v_rep = _repeat_kv(v_cat, q_per_kv)
        return F.scaled_dot_product_attention(
            q_rope, k_rep, v_rep, attn_mask=None, dropout_p=0.0, is_causal=True
        )

    k_rep = _repeat_kv(k_cat, q_per_kv)
    v_rep = _repeat_kv(v_cat, q_per_kv)

    # Rectangular attention boolean mask [1 or B, 1 or H_q, Q, T_atlas + T_live]:
    # All queries may attend to Atlas prefix [0 .. T_atlas - 1]; live keys follow causality.
    device = q_rope.device
    if is_causal_live and q_len > 1:
        q_idx = torch.arange(q_len, device=device).unsqueeze(1)  # [Q, 1]
        k_live_idx = torch.arange(t_live, device=device).unsqueeze(0)  # [1, T_live]
        live_allow = k_live_idx <= ((t_live - q_len) + q_idx)  # [Q, T_live]
    else:
        live_allow = torch.ones((q_len, t_live), dtype=torch.bool, device=device)

    if t_atlas > 0:
        atlas_allow = torch.ones((q_len, t_atlas), dtype=torch.bool, device=device)
        base_mask = torch.cat([atlas_allow, live_allow], dim=1)  # [Q, T_total]
    else:
        base_mask = live_allow

    attn_mask = base_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, Q, T_total]

    if has_atlas and atlas_valid_mask is not None:
        vm = atlas_valid_mask.to(device=device, dtype=torch.bool)
        if vm.dim() == 1:
            # [T_atlas] -> [1, 1, 1, T_atlas]
            vm_full = torch.ones((1, 1, 1, t_total), dtype=torch.bool, device=device)
            vm_full[:, :, :, :t_atlas] = vm.view(1, 1, 1, t_atlas)
        elif vm.dim() == 3:
            # [B, H_kv, T_atlas] -> repeat to [B, H_q, 1, T_atlas]
            vm_q = (
                vm[:, :, None, :]
                .expand(bsz, num_kv_heads, q_per_kv, t_atlas)
                .reshape(bsz, num_q_heads, 1, t_atlas)
            )
            vm_full = torch.ones((bsz, num_q_heads, 1, t_total), dtype=torch.bool, device=device)
            vm_full[:, :, :, :t_atlas] = vm_q
        else:
            raise ValueError(
                f"atlas_valid_mask must have shape [T_atlas] or [B, H_kv, T_atlas], got {tuple(vm.shape)}."
            )
        attn_mask = attn_mask & vm_full

    return F.scaled_dot_product_attention(
        q_rope, k_rep, v_rep, attn_mask=attn_mask, dropout_p=0.0, is_causal=False
    )


def _inspect_model_for_atlas(model: nn.Module) -> Dict[str, Any]:
    """
    Inspects HF Qwen2/Llama models and `FallbackQwen2ForCausalLM` to extract layer modules,
    GQA head dimensions, `rope_theta`, and input device cleanly.
    """
    if not hasattr(model, "config"):
        raise ValueError("Model must expose a `.config` attribute with transformer dimensions.")

    cfg_root = model.config
    cfg = (
        getattr(cfg_root, "text_config", None)
        if getattr(cfg_root, "num_hidden_layers", 0) == 0 and getattr(cfg_root, "text_config", None) is not None
        else cfg_root
    )
    num_layers = int(getattr(cfg, "num_hidden_layers", 0))
    num_q_heads = int(getattr(cfg, "num_attention_heads", 0))
    num_kv_heads = int(getattr(cfg, "num_key_value_heads", num_q_heads))
    hidden_size = int(getattr(cfg, "hidden_size", 0))

    if num_layers <= 0 or num_q_heads <= 0 or num_kv_heads <= 0:
        raise ValueError(
            f"Invalid model config dimensions: num_layers={num_layers}, "
            f"num_q_heads={num_q_heads}, num_kv_heads={num_kv_heads}."
        )
    if num_q_heads % num_kv_heads != 0:
        raise ValueError(
            f"GQA requires num_attention_heads ({num_q_heads}) divisible by "
            f"num_key_value_heads ({num_kv_heads})."
        )

    head_dim = int(getattr(cfg, "head_dim", None) or (hidden_size // num_q_heads))
    if head_dim <= 0:
        raise ValueError(f"Invalid head_dim={head_dim}.")

    layers_module = None
    for candidate_path in (
        ("model", "layers"),
        ("model", "llm", "layers"),
        ("model", "language_model", "layers"),
        ("layers",),
        ("transformer", "h"),
        ("gpt_neox", "layers"),
    ):
        cur: Any = model
        for attr in candidate_path:
            cur = getattr(cur, attr, None)
            if cur is None:
                break
        if cur is not None and len(cur) == num_layers:
            layers_module = cur
            break

    if layers_module is None:
        raise ValueError(
            "Could not locate transformer decoder layers matching config.num_hidden_layers."
        )

    for l_idx in range(num_layers):
        attn = getattr(layers_module[l_idx], "self_attn", None) or getattr(
            layers_module[l_idx], "attn", None
        )
        if attn is None or not (hasattr(attn, "k_proj") and hasattr(attn, "v_proj")):
            raise ValueError(
                f"Layer {l_idx} does not expose self_attn.{{k_proj, v_proj}} modules."
            )

    # Prefer embed_tokens device (handles device_map="auto" sharded HF models cleanly).
    input_device: Optional[torch.device] = None
    inner = getattr(model, "model", None)
    embed = getattr(inner, "embed_tokens", None) if inner is not None else None
    if embed is not None and hasattr(embed, "weight") and isinstance(embed.weight, torch.Tensor):
        input_device = embed.weight.device
    if input_device is None:
        param = next(model.parameters(), None)
        input_device = param.device if param is not None else torch.device("cpu")

    rope_theta = float(getattr(cfg, "rope_theta", 1000000.0))
    model_id = str(getattr(cfg, "_name_or_path", model.__class__.__name__))

    return {
        "num_layers": num_layers,
        "num_q_heads": num_q_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "rope_theta": rope_theta,
        "layers": layers_module,
        "input_device": input_device,
        "model_id": model_id,
    }


def _encode_text_to_ids(tokenizer: Any, text: str) -> List[int]:
    """
    Tokenizes `text` into a flat list of integer token IDs without truncation.
    Compatible with both HuggingFace tokenizers and `FallbackQwen2Tokenizer`.
    """
    if hasattr(tokenizer, "encode"):
        try:
            ids = tokenizer.encode(text, add_special_tokens=False)
        except TypeError:
            ids = tokenizer.encode(text)
        if isinstance(ids, torch.Tensor):
            return [int(x) for x in ids.reshape(-1).tolist()]
        return [int(x) for x in ids]

    batch = tokenizer(text, return_tensors="pt")
    input_ids = batch["input_ids"]
    if isinstance(input_ids, torch.Tensor):
        return [int(x) for x in input_ids.reshape(-1).tolist()]
    return [int(x) for x in input_ids[0]]


def _decode_chunk_text(
    tokenizer: Any,
    chunk_ids: List[int],
    full_text: str,
    is_single_chunk: bool,
) -> str:
    """Returns clean display text for an indexed document chunk."""
    if is_single_chunk:
        return full_text
    if hasattr(tokenizer, "decode"):
        try:
            return str(tokenizer.decode(chunk_ids, skip_special_tokens=True)).strip()
        except TypeError:
            return str(tokenizer.decode(chunk_ids)).strip()
    return full_text


def _json_safe(obj: Any) -> Any:
    """Converts nested metadata into JSON-serializable Python primitives."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, torch.Tensor):
        if obj.numel() == 1:
            return obj.item()
        return obj.detach().cpu().tolist()
    return str(obj)


class AtlasMemoryBank:
    """
    AtlasKV-inspired Persistent External KV Memory Bank (`Doc2KV` / `KG2KV`) backed by
    `HiKVPStore` in the position-free pre-RoPE frame, with Rectangular Attention retrieval.
    """

    def __init__(
        self,
        store: HiKVPStore,
        chunks: List[AtlasDocumentChunk],
        rope_theta: float = 1000000.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.store: HiKVPStore = store
        self.chunks: List[AtlasDocumentChunk] = list(chunks)
        self.rope_theta: float = float(rope_theta)
        self.metadata: Dict[str, Any] = dict(metadata) if metadata is not None else {}

    @property
    def num_layers(self) -> int:
        return int(getattr(self.store, "num_layers", self.metadata.get("num_layers", 0)))

    @property
    def num_kv_heads(self) -> int:
        return int(getattr(self.store, "num_kv_heads", self.metadata.get("num_kv_heads", 0)))

    @property
    def head_dim(self) -> int:
        return int(getattr(self.store, "head_dim", self.metadata.get("head_dim", 0)))

    @classmethod
    @torch.no_grad()
    def build_from_texts(
        cls,
        model: nn.Module,
        tokenizer: Any,
        texts: Sequence[str],
        artifacts: Dict[str, Any],
        hikvp_config: Optional[HiKVPConfig] = None,
        sources: Optional[Sequence[str]] = None,
        chunk_tokens: int = 256,
        max_chunks: Optional[int] = None,
        all_heads: bool = True,
    ) -> "AtlasMemoryBank":
        """
        Indexes a sequence of raw text documents (`Doc2KV`) into a persistent pre-RoPE
        `HiKVPStore` using forward hooks on `self_attn.k_proj` and `self_attn.v_proj`.

        Runs strictly with `use_cache=False, output_attentions=False` so FlashAttention /
        SDPA is preserved and `T x T` attention matrices are never materialized.
        """
        arch = _inspect_model_for_atlas(model)
        num_layers: int = arch["num_layers"]
        num_kv_heads: int = arch["num_kv_heads"]
        head_dim: int = arch["head_dim"]
        layers = arch["layers"]
        input_device: torch.device = arch["input_device"]

        art = artifacts if artifacts is not None else {}
        rope_theta = float(arch.get("rope_theta", art.get("rope_theta", 1000000.0)))
        U_Q = art.get("U_Q", None)

        if all_heads:
            head_roles = torch.zeros((num_layers, num_kv_heads), dtype=torch.long)
        else:
            raw_roles = art.get("head_roles", None)
            if raw_roles is None:
                head_roles = torch.zeros((num_layers, num_kv_heads), dtype=torch.long)
            else:
                head_roles = torch.as_tensor(raw_roles, dtype=torch.long, device="cpu")
                if head_roles.shape != (num_layers, num_kv_heads):
                    raise ValueError(
                        f"artifacts['head_roles'] shape {tuple(head_roles.shape)} does not match "
                        f"model (num_layers={num_layers}, num_kv_heads={num_kv_heads})."
                    )

        if hikvp_config is None:
            cfg_hikvp = HiKVPConfig(pin_memory=torch.cuda.is_available())
        else:
            cfg_hikvp = hikvp_config

        store = HiKVPStore(
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            head_roles=head_roles,
            config=cfg_hikvp,
            U_Q=U_Q,
        )

        text_list = list(texts)
        if sources is None:
            source_list = [f"doc_{i}" for i in range(len(text_list))]
        else:
            source_list = list(sources)
            if len(source_list) != len(text_list):
                raise ValueError(
                    f"sources length ({len(source_list)}) must match texts length ({len(text_list)})."
                )

        chunk_size = max(1, int(chunk_tokens))
        chunk_limit = None if max_chunks is None else max(0, int(max_chunks))

        # Prepare token chunks upfront.
        pending_chunks: List[Tuple[str, str, List[int]]] = []
        for doc_text, doc_src in zip(text_list, source_list):
            if chunk_limit is not None and len(pending_chunks) >= chunk_limit:
                break
            if not doc_text or not str(doc_text).strip():
                continue
            token_ids = _encode_text_to_ids(tokenizer, str(doc_text))
            if not token_ids:
                continue
            num_slices = (len(token_ids) + chunk_size - 1) // chunk_size
            for start_idx in range(0, len(token_ids), chunk_size):
                if chunk_limit is not None and len(pending_chunks) >= chunk_limit:
                    break
                sub_ids = token_ids[start_idx : start_idx + chunk_size]
                if not sub_ids:
                    continue
                sub_text = _decode_chunk_text(
                    tokenizer=tokenizer,
                    chunk_ids=sub_ids,
                    full_text=str(doc_text),
                    is_single_chunk=(num_slices == 1),
                )
                pending_chunks.append((str(doc_src), sub_text, sub_ids))

        chunks: List[AtlasDocumentChunk] = []
        captured_k: Dict[int, torch.Tensor] = {}
        captured_v: Dict[int, torch.Tensor] = {}
        hooks: List[Any] = []

        def _make_kv_hook(
            layer_idx: int,
            target_dict: Dict[int, torch.Tensor],
            norm_module: Optional[nn.Module] = None,
        ):
            def _hook(_module: nn.Module, _inputs: Tuple[Any, ...], output: Any) -> None:
                out = output[0] if isinstance(output, tuple) else output
                out = out.detach()
                bsz, seq_len, _ = out.shape
                reshaped = out.view(bsz, seq_len, num_kv_heads, head_dim)
                if norm_module is not None:
                    reshaped = norm_module(reshaped)
                # [B, H_kv, T, D] in position-free pre-RoPE frame
                target_dict[layer_idx] = reshaped.transpose(1, 2).contiguous()

            return _hook

        was_training = model.training
        try:
            model.eval()
            for l_idx in range(num_layers):
                attn = getattr(layers[l_idx], "self_attn", None) or getattr(
                    layers[l_idx], "attn", None
                )
                k_norm = getattr(attn, "k_norm", None)
                if not callable(k_norm):
                    k_norm = None
                hooks.append(
                    attn.k_proj.register_forward_hook(_make_kv_hook(l_idx, captured_k, k_norm))
                )
                hooks.append(
                    attn.v_proj.register_forward_hook(_make_kv_hook(l_idx, captured_v, None))
                )

            global_pos = 0
            for chunk_id, (src, chunk_text, chunk_ids) in enumerate(pending_chunks):
                captured_k.clear()
                captured_v.clear()

                t_len = len(chunk_ids)
                input_ids = torch.tensor([chunk_ids], dtype=torch.long, device=input_device)
                attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=input_device)

                # FlashAttention / SDPA-compatible forward pass (never materializes T x T weights).
                model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    output_attentions=False,
                )

                positions = torch.arange(global_pos, global_pos + t_len, dtype=torch.long)

                for l_idx in range(num_layers):
                    if l_idx not in captured_k or l_idx not in captured_v:
                        raise RuntimeError(
                            f"Forward hooks failed to capture k_proj/v_proj outputs for layer {l_idx}."
                        )
                    k_layer = captured_k[l_idx]  # [B, H_kv, T, D]
                    v_layer = captured_v[l_idx]  # [B, H_kv, T, D]
                    for h_idx in range(num_kv_heads):
                        if store.is_offloaded(l_idx, h_idx):
                            store.append_pre_rope(
                                l_idx,
                                h_idx,
                                k_layer[:, h_idx, :, :].contiguous(),
                                v_layer[:, h_idx, :, :].contiguous(),
                                positions,
                            )

                captured_k.clear()
                captured_v.clear()

                chunks.append(
                    AtlasDocumentChunk(
                        chunk_id=chunk_id,
                        source=src,
                        text=chunk_text,
                        start_pos=global_pos,
                        num_tokens=t_len,
                    )
                )
                global_pos += t_len
        finally:
            for handle in hooks:
                handle.remove()
            if was_training:
                model.train()

        metadata: Dict[str, Any] = {
            "source_type": "texts",
            "model_id": arch["model_id"],
            "num_layers": num_layers,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
            "chunk_tokens": chunk_size,
            "all_heads": bool(all_heads),
            "num_input_texts": len(text_list),
            "total_indexed_tokens": sum(c.num_tokens for c in chunks),
        }
        return cls(store=store, chunks=chunks, rope_theta=rope_theta, metadata=metadata)

    @classmethod
    @torch.no_grad()
    def build_from_kg_triples(
        cls,
        model: nn.Module,
        tokenizer: Any,
        triples: Sequence[Union[AtlasTriple, Tuple[str, str, str]]],
        artifacts: Dict[str, Any],
        hikvp_config: Optional[HiKVPConfig] = None,
        triples_per_chunk: int = 16,
        all_heads: bool = True,
        consolidate_bidirectional: bool = False,
        max_inverse_subjects: int = 16,
    ) -> "AtlasMemoryBank":
        """
        Formats Knowledge Graph triples (`KG2KV`) into attention-native factual sentences,
        batches them into chunks of `triples_per_chunk` joined by `\\n`, and builds a
        pre-RoPE `AtlasMemoryBank`. When `consolidate_bidirectional=True`, applies MeMo-inspired
        (`arXiv:2605.15156`) zero-SFT entity consolidation and inverse relation surfacing.
        """
        sentences = (
            consolidate_kg_triples(triples, max_inverse_subjects=max_inverse_subjects)
            if consolidate_bidirectional
            else format_kg_triples(triples)
        )
        per_chunk = max(1, int(triples_per_chunk))

        batched_texts: List[str] = []
        batched_sources: List[str] = []
        for idx, start in enumerate(range(0, len(sentences), per_chunk)):
            batch = sentences[start : start + per_chunk]
            if batch:
                batched_texts.append("\n".join(batch))
                batched_sources.append(f"kg_chunk_{idx}")

        # Choose a generous chunk_tokens ceiling so each triple batch stays intact unless huge.
        chunk_tokens = max(256, per_chunk * 48)
        bank = cls.build_from_texts(
            model=model,
            tokenizer=tokenizer,
            texts=batched_texts,
            artifacts=artifacts,
            hikvp_config=hikvp_config,
            sources=batched_sources,
            chunk_tokens=chunk_tokens,
            all_heads=all_heads,
        )
        bank.metadata["source_type"] = "kg_triples"
        bank.metadata["num_triples"] = len(sentences)
        bank.metadata["triples_per_chunk"] = per_chunk
        return bank

    def retrieve_for_layer(
        self,
        layer_idx: int,
        query_pre_rope: torch.Tensor,
        device: Union[torch.device, str],
        dtype: torch.dtype,
        top_k_pages: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, int]]:
        """
        Retrieves top-k pre-RoPE `(K, V)` leaves per KV head from `self.store` for a given
        decoder layer, aligning differing head lengths with zero-padded tails into uniform
        `[B, H_kv, T_atlas, D]` tensors for SDPA Rectangular Attention.

        Args:
            layer_idx: Decoder layer index.
            query_pre_rope: Pre-RoPE query tensor of shape `[B, H_kv, Q, D]`, `[B, Q, D]`
                (broadcast across all KV heads), or `[B, H_q, Q, D]` (GQA group-averaged).
            device: Target compute device for returned tensors.
            dtype: Target dtype for returned key/value tensors.
            top_k_pages: Optional override for number of leaf pages retrieved per head.

        Returns:
            `(k_atlas_pre [B, H_kv, T_atlas, D], v_atlas [B, H_kv, T_atlas, D],
              pos_atlas [T_atlas], agg_stats)`
        """
        target_device = torch.device(device)
        num_kv_heads = self.num_kv_heads
        head_dim = self.head_dim

        if query_pre_rope.dim() == 3:
            # [B, Q, D] -> broadcast across KV heads
            bsz, _, q_dim = query_pre_rope.shape
            if q_dim != head_dim:
                raise ValueError(
                    f"query_pre_rope head_dim ({q_dim}) != AtlasMemoryBank head_dim ({head_dim})."
                )
            per_head_queries = [query_pre_rope for _ in range(num_kv_heads)]
        elif query_pre_rope.dim() == 4:
            bsz, n_heads, q_len, q_dim = query_pre_rope.shape
            if q_dim != head_dim:
                raise ValueError(
                    f"query_pre_rope head_dim ({q_dim}) != AtlasMemoryBank head_dim ({head_dim})."
                )
            if n_heads == num_kv_heads:
                per_head_queries = [query_pre_rope[:, h, :, :] for h in range(num_kv_heads)]
            elif n_heads == 1:
                q0 = query_pre_rope[:, 0, :, :]
                per_head_queries = [q0 for _ in range(num_kv_heads)]
            elif n_heads > num_kv_heads and (n_heads % num_kv_heads == 0):
                q_per_kv = n_heads // num_kv_heads
                q_grouped = query_pre_rope.view(bsz, num_kv_heads, q_per_kv, q_len, q_dim).mean(
                    dim=2
                )
                per_head_queries = [q_grouped[:, h, :, :] for h in range(num_kv_heads)]
            else:
                raise ValueError(
                    f"query_pre_rope has {n_heads} heads, incompatible with num_kv_heads={num_kv_heads}."
                )
        else:
            raise ValueError(
                f"query_pre_rope must be 3D [B, Q, D] or 4D [B, H_kv, Q, D], got {tuple(query_pre_rope.shape)}."
            )

        k_list: List[torch.Tensor] = []
        v_list: List[torch.Tensor] = []
        pos_list: List[torch.Tensor] = []
        agg_stats: Dict[str, int] = {
            "heads_queried": 0,
            "heads_with_hits": 0,
            "max_retrieved_tokens": 0,
        }

        for h_idx in range(num_kv_heads):
            if self.store.is_offloaded(layer_idx, h_idx):
                agg_stats["heads_queried"] += 1
                k_h, v_h, pos_h, h_stats = self.store.retrieve(
                    layer_idx=layer_idx,
                    head_idx=h_idx,
                    query_pre_rope=per_head_queries[h_idx],
                    device=target_device,
                    dtype=dtype,
                    top_k_pages=top_k_pages,
                )
                if k_h.shape[1] > 0:
                    agg_stats["heads_with_hits"] += 1
                if isinstance(h_stats, dict):
                    for key, val in h_stats.items():
                        if isinstance(val, bool):
                            agg_stats[key] = agg_stats.get(key, 0) + int(val)
                        elif isinstance(val, (int, float)):
                            agg_stats[key] = agg_stats.get(key, 0) + int(val)
            else:
                k_h = torch.empty((bsz, 0, head_dim), device=target_device, dtype=dtype)
                v_h = torch.empty((bsz, 0, head_dim), device=target_device, dtype=dtype)
                pos_h = torch.empty((0,), device=target_device, dtype=torch.long)

            k_list.append(k_h)
            v_list.append(v_h)
            pos_list.append(pos_h)

        max_len = max((int(k_h.shape[1]) for k_h in k_list), default=0)
        agg_stats["max_retrieved_tokens"] = max_len

        if max_len == 0:
            empty_k = torch.empty((bsz, num_kv_heads, 0, head_dim), device=target_device, dtype=dtype)
            empty_v = torch.empty((bsz, num_kv_heads, 0, head_dim), device=target_device, dtype=dtype)
            empty_pos = torch.empty((0,), device=target_device, dtype=torch.long)
            return empty_k, empty_v, empty_pos, agg_stats

        # Uniform zero-padded tails across heads for clean SDPA Rectangular Attention.
        k_atlas_pre = torch.zeros(
            (bsz, num_kv_heads, max_len, head_dim), device=target_device, dtype=dtype
        )
        v_atlas = torch.zeros(
            (bsz, num_kv_heads, max_len, head_dim), device=target_device, dtype=dtype
        )
        pos_atlas: Optional[torch.Tensor] = None

        for h_idx in range(num_kv_heads):
            t_h = int(k_list[h_idx].shape[1])
            if t_h > 0:
                k_atlas_pre[:, h_idx, :t_h, :] = k_list[h_idx].to(device=target_device, dtype=dtype)
                v_atlas[:, h_idx, :t_h, :] = v_list[h_idx].to(device=target_device, dtype=dtype)
                if pos_atlas is None and t_h == max_len:
                    pos_atlas = pos_list[h_idx].to(device=target_device, dtype=torch.long).reshape(-1)

        if pos_atlas is None or pos_atlas.shape[0] != max_len:
            pos_atlas = torch.arange(max_len, device=target_device, dtype=torch.long)

        return k_atlas_pre, v_atlas, pos_atlas, agg_stats

    def apply_rope(
        self,
        k_atlas_pre: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        position_offset: int = 0,
    ) -> torch.Tensor:
        """
        Applies RoPE to retrieved pre-RoPE Atlas keys using either explicit `positions`
        or contiguous virtual prefix positions `[position_offset .. position_offset + T - 1]`.
        """
        t_len = int(k_atlas_pre.shape[-2])
        if t_len == 0:
            return k_atlas_pre
        if positions is None:
            positions = torch.arange(
                int(position_offset),
                int(position_offset) + t_len,
                device=k_atlas_pre.device,
                dtype=torch.long,
            )
        return apply_rope_to_atlas_keys(
            k_pre=k_atlas_pre,
            positions=positions,
            rope_theta=self.rope_theta,
        )

    def retrieve_and_rope_for_layer(
        self,
        layer_idx: int,
        query_pre_rope: torch.Tensor,
        device: Union[torch.device, str],
        dtype: torch.dtype,
        position_offset: int = 0,
        top_k_pages: Optional[int] = None,
        use_original_positions: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, int]]:
        """
        Convenience Rectangular Attention adapter combining `retrieve_for_layer(...)` with
        RoPE rotation at contiguous virtual prefix positions (or original corpus positions).
        """
        k_pre, v_atlas, pos_orig, agg_stats = self.retrieve_for_layer(
            layer_idx=layer_idx,
            query_pre_rope=query_pre_rope,
            device=device,
            dtype=dtype,
            top_k_pages=top_k_pages,
        )
        t_atlas = int(k_pre.shape[2])
        if t_atlas == 0:
            return k_pre, v_atlas, pos_orig, agg_stats

        if use_original_positions:
            target_pos = pos_orig
        else:
            target_pos = torch.arange(
                int(position_offset),
                int(position_offset) + t_atlas,
                device=k_pre.device,
                dtype=torch.long,
            )
        k_rope = apply_rope_to_atlas_keys(k_pre, target_pos, rope_theta=self.rope_theta)
        return k_rope, v_atlas, target_pos, agg_stats

    def attach_to_cache(self, cache: Any) -> None:
        """Attaches this `AtlasMemoryBank` onto a cache instance (`cache.atlas_bank`)."""
        if hasattr(cache, "set_atlas_bank") and callable(cache.set_atlas_bank):
            cache.set_atlas_bank(self)
        else:
            cache.atlas_bank = self

    def detach_from_cache(self, cache: Any) -> None:
        """Detaches any attached `AtlasMemoryBank` from a cache instance."""
        if hasattr(cache, "set_atlas_bank") and callable(cache.set_atlas_bank):
            cache.set_atlas_bank(None)
        elif hasattr(cache, "atlas_bank"):
            cache.atlas_bank = None

    def stats(self) -> Dict[str, Any]:
        """Combines `HiKVPStore.telemetry()` with Atlas corpus/chunk statistics."""
        out: Dict[str, Any] = dict(self.store.telemetry())
        total_indexed = sum(int(c.num_tokens) for c in self.chunks)
        out.update(
            {
                "num_chunks": len(self.chunks),
                "total_indexed_tokens": total_indexed,
                "total_tokens": total_indexed,
                "num_tokens": total_indexed,
                "sources_count": len({c.source for c in self.chunks}),
                "rope_theta": float(self.rope_theta),
                "metadata": dict(self.metadata),
            }
        )
        return out

    def save(self, directory: str) -> str:
        """
        Persists the `AtlasMemoryBank` to `directory`:
        - `hikvp_store.pt`: serialized `HiKVPStore` via `self.store.save(...)`
        - `atlas_meta.json`: document chunks, `rope_theta`, and metadata
        """
        safe_dir = secure_makedirs(directory)
        store_path = os.path.join(safe_dir, "hikvp_store.pt")
        meta_path = os.path.join(safe_dir, "atlas_meta.json")

        self.store.save(store_path)

        payload = {
            "version": ATLAS_BANK_VERSION,
            "rope_theta": float(self.rope_theta),
            "metadata": _json_safe(self.metadata),
            "chunks": [asdict(c) for c in self.chunks],
        }
        secure_atomic_write_bytes(
            meta_path,
            json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8"),
        )

        return directory

    @classmethod
    def load(
        cls,
        directory: str,
        U_Q: Optional[torch.Tensor] = None,
    ) -> "AtlasMemoryBank":
        """
        Loads a previously saved `AtlasMemoryBank` from `directory`.
        """
        resolved_dir = validate_safe_path(directory)
        store_path = os.path.join(str(resolved_dir), "hikvp_store.pt")
        meta_path = os.path.join(str(resolved_dir), "atlas_meta.json")

        if not os.path.exists(store_path):
            raise FileNotFoundError(f"Missing HiKVPStore checkpoint at: {store_path}")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Missing Atlas metadata JSON at: {meta_path}")

        store = HiKVPStore.load(store_path, U_Q=U_Q)

        with safe_open_regular_read(meta_path, max_bytes=128 * 1024 * 1024) as fh:
            payload = json.loads(fh.read().decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Atlas metadata JSON root must be an object.")
        raw_chunks = payload.get("chunks", [])
        if not isinstance(raw_chunks, list) or len(raw_chunks) > 1_000_000:
            raise ValueError("Invalid or excessive chunk list in Atlas metadata JSON.")

        chunks = [
            AtlasDocumentChunk(
                chunk_id=int(c["chunk_id"]),
                source=str(c["source"]),
                text=str(c["text"]),
                start_pos=int(c["start_pos"]),
                num_tokens=int(c["num_tokens"]),
            )
            for c in raw_chunks
        ]
        rope_theta = float(payload.get("rope_theta", 1000000.0))
        if not math.isfinite(rope_theta) or rope_theta <= 0.0 or rope_theta > 1e10:
            raise ValueError(f"Invalid rope_theta {rope_theta!r} in Atlas metadata.")
        metadata = dict(payload.get("metadata", {}))
        return cls(store=store, chunks=chunks, rope_theta=rope_theta, metadata=metadata)


__all__ = [
    "ATLAS_BANK_VERSION",
    "AtlasTriple",
    "AtlasDocumentChunk",
    "format_kg_triples",
    "apply_rope_to_atlas_keys",
    "rectangular_attention",
    "AtlasMemoryBank",
]
