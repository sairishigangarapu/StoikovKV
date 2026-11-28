"""
Live Inference Engine for StoikovKV (`stoikov_kv/cache.py`).

Implements:
- `DynamicCache` compatibility base class (imports `transformers.cache_utils.DynamicCache`
  when `transformers` is installed, or registers a 100% compatible `DynamicCache` in
  `sys.modules["transformers.cache_utils"]` for standalone operation).
- `create_mock_artifacts(model, save_path=None, seed=42)`: Fast single-pass artifact
  initializer (`U_Q`, `head_roles`, `D`, `W_ridge`, `head_entropy`) using only forward
  projection hooks (`output_attentions=False`), enabling immediate standalone execution
  when `calibrated_artifacts.pt` does not exist yet.
- `OnlineRegimeChangeController` (legacy alias `HMMTopicShiftController`): Online non-stationary
  regime-change detector tracking EMA centroid and deviation of subspace/value features; triggers
  rapid H2O score decay (`beta_shift = 0.65` vs stationary `beta_stationary = 0.99`) upon topic
  shifts (`z_score > 1.4`).
- `StoikovKVCache(DynamicCache)`: FlashAttention/SDPA-compatible hybrid KV cache with:
  1. Uncompressed FP16 sliding window of `n_b = 64` recent tokens across all KV heads.
  2. Write-time directional query-subspace importance probe (softplus-gated on pretrained
     HuggingFace models; sigmoid-gated at unit scale) + sink gate (`+100.0` for token
     indices `0..3`) and decayed heavy-hitter value `V_i(t) = beta_t * V_i(t-1) + I_t`.
  3. Per-KV-head routing for tokens older than `n_b = 64`:
     - `ROLE_RETRIEVAL_FP16 (0)`: Retain 100% of tokens in uncompressed FP16.
     - `ROLE_PEAKED_H2O (1)`: Evict lowest `V_i(t)` tokens, keeping only
       `k = int(seq_len * keep_ratio)` Heavy Hitters + `n_b` recent tokens.
     - `ROLE_DIFFUSE_LEXICO (2)`: Compress tokens via 1-step ridge-preconditioned
       projection (`W_ridge`) into `s_budget` magnitude-sorted dictionary atoms
       (`|y_(1)| >= ... >= |y_(s)|`).
  4. `set_matryoshka_mode(active_atoms=m)` for nested partial reconstruction (`3*m + 2`
     bytes read/vector during speculative drafting vs `3*s + 2` during parallel verification).
  5. `crop(max_length)` for exact speculative decoding rollback of rejected draft tokens.
  6. Exact byte telemetry: `fp16_bytes`, `compressed_bytes`, `bytes_read_last_step`,
     `cumulative_bytes_read`, and `per_tier_token_counts`.
"""

from __future__ import annotations

import math
import os
import sys
import types
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from ._security import (
    ensure_secure_triton_cache_dir,
    safe_torch_load,
    secure_torch_save,
    validate_safe_path,
)

ensure_secure_triton_cache_dir()

import torch
import torch.nn as nn
import torch.nn.functional as F

from .calibration import (
    ROLE_DIFFUSE_LEXICO,
    ROLE_PEAKED_H2O,
    ROLE_RETRIEVAL_FP16,
    _apply_rotary_pos_emb,
    _compute_rope_cos_sin,
    _inspect_model_architecture,
    score_write_time,
)
from .elastic_lexico import (
    ElasticLexicoCodec,
    QuantizedInt8Values,
    quantize_sorted_atoms,
)
from .hikvp import (
    HiKVPConfig,
    HiKVPStore,
)
from .triton_kernels import (
    can_use_triton,
    triton_fused_kv_lexico_materialize,
    triton_fused_write_probe,
)

# ============================================================================
# DynamicCache Base Class (HuggingFace Transformers + Standalone Compatibility)
# ============================================================================

try:
    from transformers.cache_utils import Cache, DynamicCache  # type: ignore
except Exception:

    class Cache:
        """Minimal HuggingFace-compatible Cache base class."""

        def update(
            self,
            key_states: torch.Tensor,
            value_states: torch.Tensor,
            layer_idx: int,
            cache_kwargs: Optional[Dict[str, Any]] = None,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            raise NotImplementedError

        def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
            raise NotImplementedError

        def get_max_length(self) -> Optional[int]:
            return None

        def get_usable_length(
            self, new_seq_length: int, layer_idx: Optional[int] = 0
        ) -> int:
            return self.get_seq_length(layer_idx)

    class DynamicCache(Cache):
        """
        Standalone implementation of `transformers.cache_utils.DynamicCache`
        preserving `key_cache`, `value_cache`, `_seen_tokens`, `update`,
        `get_seq_length`, `crop`, and indexing semantics.
        """

        def __init__(self, num_hidden_layers: Optional[int] = None) -> None:
            super().__init__()
            self.key_cache: List[torch.Tensor] = []
            self.value_cache: List[torch.Tensor] = []
            self._seen_tokens: int = 0

        def __getitem__(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
            if layer_idx < len(self):
                return (self.key_cache[layer_idx], self.value_cache[layer_idx])
            raise KeyError(f"Cache only has {len(self)} layers, accessed {layer_idx}")

        def __iter__(self):
            for layer_idx in range(len(self)):
                yield (self.key_cache[layer_idx], self.value_cache[layer_idx])

        def __len__(self) -> int:
            return len(self.key_cache)

        def update(
            self,
            key_states: torch.Tensor,
            value_states: torch.Tensor,
            layer_idx: int,
            cache_kwargs: Optional[Dict[str, Any]] = None,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            if layer_idx == 0:
                self._seen_tokens += int(key_states.shape[-2])
            while len(self.key_cache) <= layer_idx:
                self.key_cache.append(torch.empty(0, device=key_states.device, dtype=key_states.dtype))
                self.value_cache.append(torch.empty(0, device=value_states.device, dtype=value_states.dtype))
            if self.key_cache[layer_idx].numel() == 0:
                self.key_cache[layer_idx] = key_states
                self.value_cache[layer_idx] = value_states
            else:
                self.key_cache[layer_idx] = torch.cat(
                    [self.key_cache[layer_idx], key_states], dim=-2
                )
                self.value_cache[layer_idx] = torch.cat(
                    [self.value_cache[layer_idx], value_states], dim=-2
                )
            return self.key_cache[layer_idx], self.value_cache[layer_idx]

        def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
            idx = 0 if layer_idx is None else int(layer_idx)
            if len(self.key_cache) <= idx or self.key_cache[idx].numel() == 0:
                return 0
            return int(self.key_cache[idx].shape[-2])

        def get_max_length(self) -> Optional[int]:
            return None

        def crop(self, max_length: int) -> None:
            if max_length < 0:
                max_length = self.get_seq_length() - abs(max_length)
            if self.get_seq_length() <= max_length:
                return
            self._seen_tokens = max_length
            for idx in range(len(self.key_cache)):
                if self.key_cache[idx].numel() > 0:
                    self.key_cache[idx] = self.key_cache[idx][..., :max_length, :]
                    self.value_cache[idx] = self.value_cache[idx][..., :max_length, :]

    # Register shim in sys.modules so `from transformers.cache_utils import DynamicCache` works
    if "transformers" not in sys.modules:
        tf_mod = types.ModuleType("transformers")
        sys.modules["transformers"] = tf_mod
    if "transformers.cache_utils" not in sys.modules:
        cu_mod = types.ModuleType("transformers.cache_utils")
        cu_mod.Cache = Cache  # type: ignore[attr-defined]
        cu_mod.DynamicCache = DynamicCache  # type: ignore[attr-defined]
        sys.modules["transformers.cache_utils"] = cu_mod
        setattr(sys.modules["transformers"], "cache_utils", cu_mod)


# ============================================================================
# Standalone Artifact Initialization (`create_mock_artifacts`)
# ============================================================================


@torch.no_grad()
def create_mock_artifacts(
    model: nn.Module,
    tokenizer: Optional[Any] = None,
    save_path: Optional[str] = None,
    subspace_rank: int = 16,
    dict_atoms: int = 1024,
    ridge_lambda2: float = 0.1,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Initializes `U_Q`, `head_roles`, `D`, `W_ridge`, and `head_entropy` from a single fast
    forward pass (`output_attentions=False`) using forward hooks on `q_proj`, `k_proj`,
    and `v_proj`. Never requires `output_attentions=True`.

    Target Shapes:
    - `U_Q`: `[num_layers, num_kv_heads, head_dim, r]` (`r = min(subspace_rank, head_dim)`)
    - `head_roles`: `[num_layers, num_kv_heads]` (`torch.long` in `{0, 1, 2}`)
    - `D`: `[num_layers, head_dim, dict_atoms]`
    - `W_ridge`: `[num_layers, dict_atoms, head_dim]`
    - `head_entropy`: `[num_layers, num_kv_heads]`
    """
    cfg_eff = (
        getattr(model.config, "text_config", None)
        if getattr(model.config, "hidden_size", None) is None
        and getattr(model.config, "text_config", None) is not None
        else model.config
    )
    head_dim = int(
        getattr(cfg_eff, "head_dim", None)
        or (int(cfg_eff.hidden_size) // int(cfg_eff.num_attention_heads))
    )
    eff_rank = min(int(subspace_rank), head_dim)
    dims = _inspect_model_architecture(model, r_subspace=eff_rank, dict_atoms=dict_atoms)
    num_layers = dims["num_layers"]
    num_q_heads = dims["num_q_heads"]
    num_kv_heads = dims["num_kv_heads"]
    q_per_kv = dims["q_per_kv"]
    rope_theta = float(dims["rope_theta"])
    device = dims["device"]
    model_dtype = dims["dtype"]
    layers = dims["layers_module"]

    q_captured: Dict[int, torch.Tensor] = {}
    k_captured: Dict[int, torch.Tensor] = {}
    v_captured: Dict[int, torch.Tensor] = {}
    handles: List[Any] = []

    def _make_hook(store: Dict[int, torch.Tensor], l_idx: int, n_heads: int, apply_rope: bool):
        def _hook(_mod: nn.Module, _inp: Any, out: torch.Tensor) -> None:
            t = out[0] if isinstance(out, tuple) else out
            bsz, seq_len, _ = t.shape
            reshaped = t.detach().view(bsz, seq_len, n_heads, head_dim).transpose(1, 2)
            if apply_rope:
                cos, sin = _compute_rope_cos_sin(
                    seq_len=seq_len,
                    head_dim=head_dim,
                    device=reshaped.device,
                    dtype=reshaped.dtype,
                    rope_theta=rope_theta,
                )
                reshaped = _apply_rotary_pos_emb(reshaped, cos, sin)
            store[l_idx] = reshaped

        return _hook

    try:
        for l_idx in range(num_layers):
            attn = getattr(layers[l_idx], "self_attn", None) or getattr(layers[l_idx], "attn", None)
            if hasattr(attn, "q_proj") and hasattr(attn, "k_proj") and hasattr(attn, "v_proj"):
                handles.append(
                    attn.q_proj.register_forward_hook(_make_hook(q_captured, l_idx, num_q_heads, True))
                )
                handles.append(
                    attn.k_proj.register_forward_hook(_make_hook(k_captured, l_idx, num_kv_heads, True))
                )
                handles.append(
                    attn.v_proj.register_forward_hook(_make_hook(v_captured, l_idx, num_kv_heads, False))
                )

        # Build a single fast batch of diverse synthetic/tokenized sequences (no output_attentions)
        vocab_sz = int(getattr(cfg_eff, "vocab_size", 32768))
        gen = torch.Generator(device="cpu").manual_seed(int(seed))
        input_ids = torch.randint(16, min(8192, vocab_sz), (4, 48), generator=gen).to(device)
        if tokenizer is not None:
            sample_texts = [
                "System configuration: API_PORT = 8443. The secret passkey is 94812. Maintain low latency.",
                "def forward_attention(q, k, v): return torch.nn.functional.scaled_dot_product_attention(q, k, v)",
                ".container { display: flex; backdrop-filter: blur(12px); padding: 1.5rem; border-radius: 8px; }",
                "StoikovKV routes retrieval heads to FP16, peaked heads to H2O, and diffuse heads to Lexico.",
            ]
            for b_i, txt in enumerate(sample_texts):
                enc = tokenizer(txt, return_tensors="pt")
                ids = enc["input_ids"] if isinstance(enc, dict) else enc.input_ids
                take = min(48, int(ids.shape[-1]))
                input_ids[b_i, :take] = ids[0, :take].to(device)

        was_training = model.training
        model.eval()
        try:
            _ = model(input_ids=input_ids, output_attentions=False, use_cache=False)
        except TypeError:
            _ = model(input_ids=input_ids)
        if was_training:
            model.train()
    finally:
        for h in handles:
            h.remove()

    U_Q = torch.zeros(
        (num_layers, num_kv_heads, head_dim, eff_rank),
        dtype=model_dtype,
        device=device,
    )
    D = torch.zeros(
        (num_layers, head_dim, dict_atoms),
        dtype=model_dtype,
        device=device,
    )
    W_ridge = torch.zeros(
        (num_layers, dict_atoms, head_dim),
        dtype=model_dtype,
        device=device,
    )
    head_entropy = torch.zeros((num_layers, num_kv_heads), dtype=torch.float32, device=device)
    retrieval_Proxy = torch.zeros((num_layers, num_kv_heads), dtype=torch.float32, device=device)

    eye_atoms = torch.eye(dict_atoms, dtype=torch.float32, device=device)
    gen_dev = torch.Generator(device="cpu").manual_seed(int(seed) + 101)

    for l_idx in range(num_layers):
        q_t = q_captured.get(l_idx)
        k_t = k_captured.get(l_idx)
        v_t = v_captured.get(l_idx)

        if q_t is not None:
            # q_t: [bsz, num_q_heads, seq_len, head_dim]
            bsz, _, slen, _ = q_t.shape
            q_grouped = (
                q_t.view(bsz, num_kv_heads, q_per_kv, slen, head_dim)
                .permute(1, 0, 3, 2, 4)
                .reshape(num_kv_heads, bsz * slen * q_per_kv, head_dim)
                .float()
            )
            for h_kv in range(num_kv_heads):
                mat_q = q_grouped[h_kv]
                mat_q = torch.nan_to_num(mat_q, nan=0.0, posinf=0.0, neginf=0.0)
                try:
                    _, s_vals, Vh = torch.linalg.svd(mat_q, full_matrices=False)
                except RuntimeError:
                    _, s_vals, Vh = torch.linalg.svd(mat_q.cpu(), full_matrices=False)
                    s_vals = s_vals.to(device)
                    Vh = Vh.to(device)
                basis = Vh[:eff_rank, :].T
                # Deterministic sign canonicalization
                max_idx = torch.argmax(basis.abs(), dim=0, keepdim=True)
                signs = torch.sign(torch.gather(basis, 0, max_idx))
                signs = torch.where(signs == 0, torch.ones_like(signs), signs)
                basis = basis * signs
                U_Q[l_idx, h_kv] = basis.to(dtype=model_dtype)

                # Proxy spectral entropy from singular value concentration
                p_sv = (s_vals ** 2) / (torch.sum(s_vals ** 2) + 1e-8)
                ent = -torch.sum(p_sv * torch.log(p_sv.clamp_min(1e-12)))
                head_entropy[l_idx, h_kv] = ent
                retrieval_Proxy[l_idx, h_kv] = float(s_vals[0].item())

        if k_t is not None and v_t is not None:
            kv_samples = torch.cat(
                [
                    k_t.reshape(-1, head_dim).float(),
                    v_t.reshape(-1, head_dim).float(),
                ],
                dim=0,
            )
            kv_samples = torch.nan_to_num(kv_samples, nan=0.0, posinf=0.0, neginf=0.0)
            try:
                _, _, Vh_kv = torch.linalg.svd(kv_samples, full_matrices=False)
            except RuntimeError:
                _, _, Vh_kv = torch.linalg.svd(kv_samples.cpu(), full_matrices=False)
                Vh_kv = Vh_kv.to(device)
            r_kv = Vh_kv.shape[0]
            rand_atoms = torch.randn(
                (head_dim, dict_atoms), generator=gen_dev, dtype=torch.float32
            ).to(device)
            rand_atoms[:, :r_kv] = Vh_kv.T
            # Mix principal components to form an overcomplete normalized dictionary
            mix = torch.randn((r_kv, dict_atoms - r_kv), generator=gen_dev, dtype=torch.float32).to(device)
            rand_atoms[:, r_kv:] = Vh_kv.T @ mix + 0.10 * rand_atoms[:, r_kv:]
            D_l = F.normalize(rand_atoms, p=2.0, dim=0, eps=1e-8)
        else:
            rand_atoms = torch.randn(
                (head_dim, dict_atoms), generator=gen_dev, dtype=torch.float32
            ).to(device)
            D_l = F.normalize(rand_atoms, p=2.0, dim=0, eps=1e-8)

        gram = D_l.T @ D_l + float(ridge_lambda2) * eye_atoms
        W_l = torch.linalg.solve(gram, D_l.T)
        D[l_idx] = D_l.to(dtype=model_dtype)
        W_ridge[l_idx] = W_l.to(dtype=model_dtype)

    # Assign head_roles: top 5% retrieval -> 0, remaining < 90th percentile entropy -> 1, >= 90th -> 2
    # Using 90th percentile (was median) to minimize Lexico heads (diffuse = lossy, CSS ~0.46).
    # H2O heads have high fidelity (CSS ~0.96). With 90th percentile: ~5% Retrieval, ~85% H2O, ~10% Lexico.
    total_heads = num_layers * num_kv_heads
    num_ret = max(1, int(round(0.05 * total_heads)))
    flat_ret = retrieval_Proxy.view(-1)
    top_ret_idx = torch.topk(flat_ret, k=num_ret, largest=True).indices
    head_roles_flat = torch.full((total_heads,), ROLE_DIFFUSE_LEXICO, dtype=torch.long, device=device)
    is_ret_mask = torch.zeros(total_heads, dtype=torch.bool, device=device)
    is_ret_mask[top_ret_idx] = True
    head_roles_flat[is_ret_mask] = ROLE_RETRIEVAL_FP16

    rem_idx = (~is_ret_mask).nonzero(as_tuple=False).squeeze(-1)
    if rem_idx.numel() > 0:
        rem_ent = head_entropy.view(-1)[rem_idx]
        entropy_threshold = torch.quantile(rem_ent, 0.90)
        peaked_mask = rem_ent < entropy_threshold
        if not torch.any(peaked_mask) and rem_idx.numel() > 1:
            peaked_mask[: rem_idx.numel() // 2] = True
        head_roles_flat[rem_idx[peaked_mask]] = ROLE_PEAKED_H2O
        head_roles_flat[rem_idx[~peaked_mask]] = ROLE_DIFFUSE_LEXICO

    # Co-locate ROLE_DIFFUSE_LEXICO with ROLE_RETRIEVAL_FP16 layers so every
    # ROLE_PEAKED_H2O layer is layer-pure (`all_h2o == True`) and never needs dummy padding.
    head_roles = head_roles_flat.view(num_layers, num_kv_heads)
    n_lex_target = int((head_roles == ROLE_DIFFUSE_LEXICO).sum().item())
    ret_layers = [
        l_idx for l_idx in range(num_layers) if bool((head_roles[l_idx] == ROLE_RETRIEVAL_FP16).any().item())
    ]
    non_ret_layers = [l_idx for l_idx in range(num_layers) if l_idx not in ret_layers]
    for l_idx in range(num_layers):
        for h_idx in range(num_kv_heads):
            if head_roles[l_idx, h_idx] != ROLE_RETRIEVAL_FP16:
                head_roles[l_idx, h_idx] = ROLE_PEAKED_H2O
    lex_assigned = 0
    for l_idx in ret_layers:
        for h_idx in range(num_kv_heads):
            if head_roles[l_idx, h_idx] != ROLE_RETRIEVAL_FP16:
                head_roles[l_idx, h_idx] = ROLE_DIFFUSE_LEXICO
                lex_assigned += 1
    if lex_assigned < n_lex_target and non_ret_layers:
        layer_ent = [(float(head_entropy[l_idx].mean().item()), l_idx) for l_idx in non_ret_layers]
        layer_ent.sort(key=lambda x: x[0], reverse=True)
        for _, l_idx in layer_ent:
            if lex_assigned >= n_lex_target:
                break
            head_roles[l_idx, :] = ROLE_DIFFUSE_LEXICO
            lex_assigned += num_kv_heads

    artifacts = {
        "U_Q": U_Q,
        "head_roles": head_roles,
        "D": D,
        "W_ridge": W_ridge,
        "head_entropy": head_entropy,
        "is_mock": True,
    }
    if save_path is not None:
        secure_torch_save(artifacts, save_path)
    return artifacts


def load_or_create_artifacts(
    model: nn.Module,
    artifacts_path: str = "calibrated_artifacts.pt",
    tokenizer: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Loads `calibrated_artifacts.pt` if it exists on disk and matches `model` dimensions;
    otherwise initializes artifacts on the fly via `create_mock_artifacts(model)`.
    """
    cfg_eff = (
        getattr(model.config, "text_config", None)
        if getattr(model.config, "hidden_size", None) is None
        and getattr(model.config, "text_config", None) is not None
        else model.config
    )
    head_dim = int(
        getattr(cfg_eff, "head_dim", None)
        or (int(cfg_eff.hidden_size) // int(cfg_eff.num_attention_heads))
    )
    dims = _inspect_model_architecture(model, r_subspace=min(16, head_dim), dict_atoms=1024)
    device = dims["device"]
    dtype = dims["dtype"]

    resolved_path = artifacts_path
    if resolved_path:
        validate_safe_path(resolved_path)
        if not os.path.exists(resolved_path):
            alt = os.path.join("artifacts", os.path.basename(resolved_path))
            if os.path.exists(alt):
                validate_safe_path(alt)
                resolved_path = alt

    if resolved_path and os.path.exists(resolved_path):
        loaded = safe_torch_load(resolved_path, map_location=device)
        if not isinstance(loaded, dict):
            raise ValueError("Invalid calibrated artifacts format: expected dictionary.")
        u_q = loaded.get("U_Q")
        d_mat = loaded.get("D")
        w_mat = loaded.get("W_ridge")
        roles_t = loaded.get("head_roles")
        if (
            isinstance(u_q, torch.Tensor)
            and isinstance(d_mat, torch.Tensor)
            and isinstance(w_mat, torch.Tensor)
            and isinstance(roles_t, torch.Tensor)
            and u_q.ndim == 4
            and u_q.shape[0] == dims["num_layers"]
            and u_q.shape[1] == dims["num_kv_heads"]
            and u_q.shape[2] == dims["head_dim"]
            and roles_t.shape == (dims["num_layers"], dims["num_kv_heads"])
        ):
            if not (torch.isfinite(u_q).all() and torch.isfinite(d_mat).all() and torch.isfinite(w_mat).all()):
                raise ValueError("Calibrated artifacts contain non-finite NaN/Inf tensor values.")
            roles_long = roles_t.to(dtype=torch.long)
            if int(roles_long.min().item()) < 0 or int(roles_long.max().item()) > 3:
                raise ValueError("Calibrated head_roles contain invalid tier role IDs outside [0, 3].")
            frame = str(loaded.get("dict_key_frame", "post_rope"))
            if frame not in ("pre_rope", "post_rope"):
                raise ValueError(f"Invalid dict_key_frame {frame!r}; expected 'pre_rope' or 'post_rope'.")
            rope_theta = float(
                loaded.get("rope_theta", getattr(cfg_eff, "rope_theta", 1000000.0))
            )
            if not math.isfinite(rope_theta) or rope_theta <= 0.0 or rope_theta > 1e10:
                raise ValueError(f"Invalid rope_theta {rope_theta!r} in calibrated artifacts.")
            return {
                "U_Q": u_q.to(device=device, dtype=dtype),
                "head_roles": roles_long.to(device=device),
                "D": d_mat.to(device=device, dtype=dtype),
                "W_ridge": w_mat.to(device=device, dtype=dtype),
                "head_entropy": loaded.get(
                    "head_entropy",
                    torch.zeros((dims["num_layers"], dims["num_kv_heads"]), device=device),
                ).to(device=device),
                "is_mock": bool(loaded.get("is_mock", False)),
                "dict_key_frame": frame,
                "rope_theta": rope_theta,
            }

    return create_mock_artifacts(model=model, tokenizer=tokenizer, save_path=None)


# ============================================================================
# Online Regime-Change Decay Controller (EMA Centroid/Deviation Detector)
# ============================================================================


def stoikov_control_step(
    inventory_q: float,
    sigma_sq: float,
    gamma_risk: float = 0.5,
    kappa_arrival: float = 1.5,
) -> Tuple[float, float]:
    """
    Exact Avellaneda-Stoikov (2008) reservation skew & optimal half-spread:
        skew = q_t * gamma * sigma_t^2
        delta* = 0.5 * gamma * sigma_t^2 + (1 / gamma) * ln(1 + gamma / kappa)
    """
    g = max(float(gamma_risk), 1e-4)
    k = max(float(kappa_arrival), 1e-4)
    s2 = max(float(sigma_sq), 0.0)
    skew = float(inventory_q) * g * s2
    half_spread = 0.5 * g * s2 + (1.0 / g) * math.log(1.0 + g / k)
    return skew, half_spread


@dataclass
class OnlineRegimeChangeController:
    """
    Online Avellaneda-Stoikov Inventory & Regime-Change Controller for StoikovKV
    (exported with legacy alias `HMMTopicShiftController`):
    - Tracks EMA centroid and instantaneous drift volatility `sigma_t^2 = ema_dev^2`.
    - Computes normalized KV inventory pressure `q_t = clip((L_t - L_target) / L_target, -1, 1)`
      and applies the Avellaneda-Stoikov reservation skew `q_t * gamma * sigma_t^2` and optimal
      half-spread `delta*` to adapt `keep_ratio` smoothly (retaining more heavy hitters when
      VRAM headroom `q_t < 0` is ample, and tightening back to `keep_ratio` as `q_t -> +1`).
    - Triggers a single-step shift decay (`beta_t = beta_shift = 0.65`) on regime shifts (`z > 1.4`).
    """

    beta_stationary: float = 1.0
    beta_shift: float = 0.65
    keep_ratio: float = 0.20
    s_budget: int = 16
    shift_z_threshold: float = 1.4
    ema_alpha: float = 0.15
    enable_hmm_decay: bool = True
    enable_stoikov_control: bool = True
    gamma_risk: float = 0.5
    kappa_arrival: float = 1.5
    inventory_target_tokens: int = 2048
    hawkes_alpha: float = 0.6
    hawkes_beta: float = 0.25
    vpin_threshold: float = 0.42
    max_h2o_tokens: int = 0

    # Telemetry state
    ema_centroid: Optional[torch.Tensor] = None
    ema_signed_drift: Optional[torch.Tensor] = None
    ema_abs_step: float = 0.0
    ema_dev: float = 0.05
    last_z_score: float = 0.0
    last_vpin: float = 0.0
    last_beta_t: float = 1.0
    last_kappa_t: float = 1.5
    last_liquidity_rebate: float = 0.0
    last_inventory_q: float = 0.0
    last_reservation_skew: float = 0.0
    last_half_spread: float = 0.0
    last_keep_ratio: float = 0.20
    shift_events: int = 0
    step_history: List[Dict[str, float]] = field(default_factory=list)

    def reset(self) -> None:
        self.ema_centroid = None
        self.ema_signed_drift = None
        self.ema_abs_step = 0.0
        self.ema_dev = 0.05
        self.last_z_score = 0.0
        self.last_vpin = 0.0
        self.last_beta_t = self.beta_stationary if self.enable_hmm_decay else 1.0
        self.last_kappa_t = self.kappa_arrival
        self.last_liquidity_rebate = 0.0
        self.last_inventory_q = 0.0
        self.last_reservation_skew = 0.0
        self.last_half_spread = 0.0
        self.last_keep_ratio = self.keep_ratio
        self.shift_events = 0
        self.step_history.clear()

    def compute_dynamic_keep_ratio(self, seq_len: int) -> float:
        """
        Evaluates the Avellaneda-Stoikov + Hawkes + GLFT/Almgren-Chriss retention ratio `keep_ratio_t`:
        - Hawkes (1971) self-exciting arrival `kappa_t >= kappa_0` narrows `delta_t*`, yielding a
          non-negative liquidity rebate `Delta_delta_t = delta*(kappa_0) - delta*(kappa_t) >= 0`
          that expands admission during dense entity/topic bursts.
        - Almgren-Chriss (2001) convex overload urgency `sinh(2*q_t)/sinh(2)` + optional GLFT (2013)
          terminal barrier `max_h2o_tokens` enforce VRAM tightening when `q_t > 0` even as `sigma_t^2 -> 0`.
        """
        if not self.enable_stoikov_control:
            self.last_keep_ratio = self.keep_ratio
            return self.keep_ratio
        target = float(max(128, self.inventory_target_tokens))
        q_t = max(-1.0, min(1.0, (float(seq_len) - target) / target))
        sigma_sq = float(self.ema_dev * self.ema_dev)
        eff_kappa = max(float(self.kappa_arrival), float(self.last_kappa_t))
        skew, half_spread = stoikov_control_step(
            q_t, sigma_sq, gamma_risk=self.gamma_risk, kappa_arrival=eff_kappa
        )
        _, half_spread_0 = stoikov_control_step(
            q_t, sigma_sq, gamma_risk=self.gamma_risk, kappa_arrival=self.kappa_arrival
        )
        liq_rebate = max(0.0, half_spread_0 - half_spread)
        self.last_inventory_q = q_t
        self.last_reservation_skew = skew
        self.last_half_spread = half_spread
        self.last_liquidity_rebate = liq_rebate

        k0 = float(self.keep_ratio)
        if q_t < 0.0:
            # Below target inventory (excess VRAM headroom): relax admission smoothly up to min(0.55, 3*k0)
            bonus = -0.35 * q_t * math.tanh(half_spread_0) - skew + 0.12 * liq_rebate
            k_dyn = max(k0, min(min(0.55, 3.0 * k0), k0 + bonus))
        else:
            # At/above target inventory: Almgren-Chriss convex carrying skew + reservation skew
            q_ac = (math.sinh(2.0 * q_t) / math.sinh(2.0)) if q_t > 0.0 else 0.0
            carry_skew = skew + 0.35 * k0 * q_ac
            k_dyn = max(max(0.08, 0.65 * k0), min(k0, k0 - carry_skew + 0.12 * liq_rebate))
            if self.max_h2o_tokens > 0 and seq_len > 0:
                k_dyn = min(k_dyn, float(self.max_h2o_tokens) / float(seq_len))
        self.last_keep_ratio = float(k_dyn)
        return self.last_keep_ratio

    def _update_drift_and_microstructure(self, feat: torch.Tensor, min_dev_floor: float = 0.04) -> float:
        """Updates spherical EMA centroid, directional VPIN toxicity, and Hawkes arrival `kappa_t`."""
        feat_cpu = F.normalize(feat.detach().to(device="cpu", dtype=torch.float32).flatten(), p=2.0, dim=0, eps=1e-8)
        if self.ema_centroid is None or self.ema_centroid.shape != feat_cpu.shape or self.ema_centroid.device.type != "cpu":
            self.ema_centroid = feat_cpu.clone()
            self.ema_signed_drift = None
            self.ema_abs_step = 0.0
            self.last_z_score = 0.0
            self.last_vpin = 0.0
            self.last_kappa_t = self.kappa_arrival
            return self.beta_stationary

        delta_vec = feat_cpu - self.ema_centroid
        diff = float(torch.linalg.vector_norm(delta_vec, ord=2).item())
        z_score = diff / max(self.ema_dev, 1e-4)
        self.last_z_score = z_score

        if self.ema_signed_drift is None or self.ema_signed_drift.shape != delta_vec.shape:
            self.ema_signed_drift = delta_vec.clone()
            self.ema_abs_step = max(diff, 1e-6)
        else:
            self.ema_signed_drift = (
                (1.0 - self.ema_alpha) * self.ema_signed_drift + self.ema_alpha * delta_vec
            )
            self.ema_abs_step = (1.0 - self.ema_alpha) * self.ema_abs_step + self.ema_alpha * diff

        vpin_t = float(torch.linalg.vector_norm(self.ema_signed_drift, ord=2).item()) / max(
            self.ema_abs_step, 1e-6
        )
        self.last_vpin = min(1.0, vpin_t)

        # Hawkes (1971) self-exciting arrival intensity kappa_t driven by VPIN-weighted shock
        shock_t = min(3.0, max(0.0, z_score - 1.0)) * max(0.25, self.last_vpin)
        self.last_kappa_t = (
            self.kappa_arrival
            + (self.last_kappa_t - self.kappa_arrival) * math.exp(-self.hawkes_beta)
            + self.hawkes_alpha * shock_t
        )

        # Update running EMA centroid and volatility
        self.ema_centroid = F.normalize(
            (1.0 - self.ema_alpha) * self.ema_centroid + self.ema_alpha * feat_cpu,
            p=2.0,
            dim=0,
            eps=1e-8,
        )
        self.ema_dev = (1.0 - self.ema_alpha) * self.ema_dev + self.ema_alpha * max(diff, min_dev_floor)

        is_toxic_shift = (z_score > self.shift_z_threshold) and (
            self.last_vpin >= self.vpin_threshold or z_score > 2.2 * self.shift_z_threshold
        )
        if is_toxic_shift:
            self.shift_events += 1
            return self.beta_shift
        return self.beta_stationary

    @torch.no_grad()
    def observe_and_query(
        self,
        proj_energy: torch.Tensor,
        seq_len: int,
    ) -> Tuple[float, float, int]:
        """
        Observes normalized query-subspace projection energy `proj_energy` (`[r]` or `[H, r]`)
        and returns `(beta_t, keep_ratio_t, s_budget)`.
        """
        if not self.enable_hmm_decay:
            self.last_beta_t = 1.0
            k_dyn = self.compute_dynamic_keep_ratio(seq_len)
            return 1.0, k_dyn, self.s_budget

        feat = F.normalize(proj_energy.detach().float().flatten(), p=2.0, dim=0, eps=1e-8)
        was_init = self.ema_centroid is not None and self.ema_centroid.shape == feat.shape
        beta_t = self._update_drift_and_microstructure(feat, min_dev_floor=0.04)
        self.last_beta_t = beta_t
        k_dyn = self.compute_dynamic_keep_ratio(seq_len)
        if was_init:
            self.step_history.append(
                {
                    "seq_len": float(seq_len),
                    "z_score": float(self.last_z_score),
                    "vpin": float(self.last_vpin),
                    "kappa_t": float(self.last_kappa_t),
                    "beta_t": float(beta_t),
                    "keep_ratio_t": float(k_dyn),
                }
            )
        return beta_t, k_dyn, self.s_budget


# Backward-compatible alias for legacy callers and unit tests
HMMTopicShiftController = OnlineRegimeChangeController


# ============================================================================
# Per-Head Tier Storage Structures
# ============================================================================

ROLE_QUANT_LOWBIT: int = 3


def _quantize_lowbit_2bit(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Packs a `[B, T, 128]` FP16/FP32 vector into KIVI-style 2-bit asymmetric group-quantized format
    (`G = 3` groups covering 128 channels -> `32` packed `uint8` bytes + `3` FP16 mins + `3` FP16 scales
    = `44` physical bytes/vector, i.e., `17.19%` of FP16's `256` bytes/vector).
    """
    bsz, seq_len, d_head = x.shape
    xf = x.float()
    # Partition 128 channels into 3 groups: [0:44], [44:88], [88:128]
    g_slices = (slice(0, 44), slice(44, 88), slice(88, d_head))
    mins = []
    scales = []
    codes = torch.empty((bsz, seq_len, d_head), dtype=torch.uint8, device=x.device)
    for g_s in g_slices:
        xg = xf[..., g_s]
        g_min = xg.amin(dim=-1, keepdim=True)
        g_max = xg.amax(dim=-1, keepdim=True)
        g_scale = ((g_max - g_min) / 3.0).clamp(min=1e-6)
        q = torch.round((xg - g_min) / g_scale).clamp(0.0, 3.0).to(torch.uint8)
        codes[..., g_s] = q
        mins.append(g_min.to(torch.float16))
        scales.append(g_scale.to(torch.float16))
    q_min = torch.cat(mins, dim=-1)      # [B, T, 3] float16 (6 B)
    q_scale = torch.cat(scales, dim=-1)  # [B, T, 3] float16 (6 B)
    # Pack 4 x 2-bit codes into 1 uint8 byte -> [B, T, 32] uint8 (32 B)
    c4 = codes.view(bsz, seq_len, d_head // 4, 4)
    packed = c4[..., 0] | (c4[..., 1] << 2) | (c4[..., 2] << 4) | (c4[..., 3] << 6)
    return packed, q_min, q_scale


def _dequantize_lowbit_2bit(
    packed: torch.Tensor,
    q_min: torch.Tensor,
    q_scale: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Unpacks `[B, T, 32]` 2-bit codes + `[B, T, 3]` FP16 `(q_min, q_scale)` back to `[B, T, 128]`."""
    bsz, seq_len, n_pack = packed.shape
    d_head = n_pack * 4
    c0 = packed & 0x03
    c1 = (packed >> 2) & 0x03
    c2 = (packed >> 4) & 0x03
    c3 = (packed >> 6) & 0x03
    codes = torch.stack([c0, c1, c2, c3], dim=-1).view(bsz, seq_len, d_head).to(dtype)
    out = torch.empty((bsz, seq_len, d_head), dtype=dtype, device=packed.device)
    g_slices = (slice(0, 44), slice(44, 88), slice(88, d_head))
    for g_idx, g_s in enumerate(g_slices):
        m_g = q_min[..., g_idx : g_idx + 1].to(dtype)
        s_g = q_scale[..., g_idx : g_idx + 1].to(dtype)
        out[..., g_s] = codes[..., g_s] * s_g + m_g
    return out


@dataclass
class _HeadTierState:
    """Physical storage for a single `(layer_idx, kv_head_idx)` historical tier (tokens older than `n_b`)."""

    role: int
    # ROLE_RETRIEVAL_FP16 (0): Full uncompressed FP16 [batch, hist_len, head_dim]
    fp16_k: Optional[torch.Tensor] = None
    fp16_v: Optional[torch.Tensor] = None

    # ROLE_PEAKED_H2O (1): Kept heavy-hitter FP16 vectors [batch, k_kept, head_dim] & global token indices [batch, k_kept]
    h2o_k: Optional[torch.Tensor] = None
    h2o_v: Optional[torch.Tensor] = None
    h2o_indices: Optional[torch.Tensor] = None
    h2o_scores: Optional[torch.Tensor] = None

    # ROLE_DIFFUSE_LEXICO (2): Sorted sparse dictionary atoms [batch, hist_len, s_budget]
    lex_k_idx: Optional[torch.Tensor] = None
    lex_k_val: Optional[QuantizedInt8Values] = None
    lex_v_idx: Optional[torch.Tensor] = None
    lex_v_val: Optional[QuantizedInt8Values] = None
    # Cached full-s and active-m reconstructions for fast autoregressive decoding
    lex_recon_k: Optional[torch.Tensor] = None
    lex_recon_v: Optional[torch.Tensor] = None
    lex_recon_atoms: int = -1
    lex_recon_len: int = 0

    # ROLE_QUANT_LOWBIT (3): KIVI-style 2-bit asymmetric group quantization (44 B/vec)
    quant_k_packed: Optional[torch.Tensor] = None
    quant_k_min: Optional[torch.Tensor] = None
    quant_k_scale: Optional[torch.Tensor] = None
    quant_v_packed: Optional[torch.Tensor] = None
    quant_v_min: Optional[torch.Tensor] = None
    quant_v_scale: Optional[torch.Tensor] = None


# ============================================================================
# StoikovKVCache(DynamicCache)
# ============================================================================


_HF_SDPA_PATCHED: bool = False


def _ensure_hf_sdpa_compact_mask_support() -> bool:
    """
    Wraps HuggingFace's `sdpa_attention_forward` in `ALL_ATTENTION_FUNCTIONS["sdpa"]`
    so compact `ROLE_PEAKED_H2O` KV tensors (`kv_len = k_kept + win_len < total_seq_len`)
    work seamlessly for both single-token decoding (`q_len == 1`) and multi-token
    parallel verification (`1 < q_len <= win_len`):
    1. If a 4D causal `attention_mask` has last dim `total_seq_len > kv_len`, slices it to
       `attention_mask[..., -kv_len:]` (exact lower-right causal alignment).
    2. If `attention_mask is None` and `1 < q_len < kv_len`, provides an explicit lower-right
       causal boolean mask so HF `sdpa_attention_forward` does not truncate `key[:, :, :q_len]`.
    """
    global _HF_SDPA_PATCHED
    if _HF_SDPA_PATCHED:
        return True
    try:
        import transformers.integrations.sdpa_attention as hf_sdpa
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        orig_sdpa = hf_sdpa.sdpa_attention_forward

        def _stoikov_sdpa_attention_forward(
            module: Any,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            attention_mask: Optional[torch.Tensor],
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            q_len = int(query.shape[-2])
            kv_len = int(key.shape[-2])
            if isinstance(attention_mask, torch.Tensor) and attention_mask.dim() == 4:
                if attention_mask.shape[-1] > kv_len:
                    attention_mask = attention_mask[..., -kv_len:]
                elif attention_mask.shape[-1] < kv_len:
                    pad_prefix = kv_len - int(attention_mask.shape[-1])
                    if attention_mask.dtype == torch.bool:
                        prefix_mask = torch.ones(
                            (*attention_mask.shape[:-1], pad_prefix),
                            dtype=torch.bool,
                            device=attention_mask.device,
                        )
                    else:
                        prefix_mask = torch.zeros(
                            (*attention_mask.shape[:-1], pad_prefix),
                            dtype=attention_mask.dtype,
                            device=attention_mask.device,
                        )
                    attention_mask = torch.cat([prefix_mask, attention_mask], dim=-1)
            elif attention_mask is None and q_len > 1 and kv_len > q_len:
                q_idx = torch.arange(q_len, device=query.device).unsqueeze(1)
                k_idx = torch.arange(kv_len, device=query.device).unsqueeze(0)
                attention_mask = (k_idx <= ((kv_len - q_len) + q_idx)).unsqueeze(0).unsqueeze(0)
            return orig_sdpa(module, query, key, value, attention_mask, *args, **kwargs)

        hf_sdpa.sdpa_attention_forward = _stoikov_sdpa_attention_forward
        ALL_ATTENTION_FUNCTIONS["sdpa"] = _stoikov_sdpa_attention_forward
        _HF_SDPA_PATCHED = True
        return True
    except Exception:
        return False


class StoikovKVCache(DynamicCache):
    """
    FlashAttention/SDPA-Compatible Hybrid KV Cache (`StoikovKVCache(DynamicCache)`).

    Features:
    1. Recent uncompressed FP16 sliding window of `n_b = 64` tokens across all KV heads.
    2. FlashAttention-compatible write-time importance score:
       `I_t = ||K_t @ U_Q||_2 * ||V_t||_2` + `+100.0` sink protection for indices `0..3`.
    3. Running decayed importance scores `V_i(t) = beta_t * V_i(t-1) + I_t` controlled by
       `HMMTopicShiftController` (or a custom `controller_fn` hook).
    4. Per-KV-head routing for historical tokens (`pos < seq_len - n_b`):
       - `ROLE_RETRIEVAL_FP16 (0)`: Retain 100% of tokens in uncompressed FP16.
       - `ROLE_PEAKED_H2O (1)`: Evict tokens with lowest `V_i(t)`, keeping only
         `k = int(seq_len * keep_ratio)` Heavy Hitters + `n_b` recent tokens.
       - `ROLE_DIFFUSE_LEXICO (2)`: Compress tokens via 1-step ElasticNet projection (`W_ridge`)
         into `s_budget` sorted dictionary atoms (`|y_(1)| >= ... >= |y_(s)|`).
    5. `set_matryoshka_mode(active_atoms=m)` reconstructs `ROLE_DIFFUSE_LEXICO` heads using
       only the top `m` atoms (`3*m + 2` bytes read/vector).
    6. `crop(max_length)` rolls back rejected speculative draft tokens cleanly.
    7. Exact byte telemetry: `fp16_bytes`, `compressed_bytes`, `bytes_read_last_step`,
       `cumulative_bytes_read`, and `per_tier_token_counts`.
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        artifacts: Optional[Dict[str, Any]] = None,
        artifacts_path: str = "calibrated_artifacts.pt",
        n_b: int = 64,
        keep_ratio: float = 0.20,
        s_budget: int = 16,
        lambda1: float = 0.04,
        sink_tokens: int = 4,
        sink_bonus: float = 100.0,
        policy_mode: str = "hybrid",
        controller: Optional[OnlineRegimeChangeController] = None,
        controller_fn: Optional[Callable[[int, int], Tuple[float, float, int]]] = None,
        enable_hmm_decay: bool = True,
        max_rollback_window: int = 32,
        h2o_context_window: int = 0,
        cache_lexico_reconstructions: bool = False,
        hikvp_config: Optional[HiKVPConfig] = None,
        atlas_bank: Optional[Any] = None,
    ) -> None:
        # Initialize HF DynamicCache with config=None so DynamicCache does not allocate
        # duplicate uncompressed DynamicLayer objects across all layers.
        super().__init__(config=None)
        self.layers = []
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        self._seen_tokens: int = 0
        self.cache_lexico_reconstructions: bool = bool(cache_lexico_reconstructions)
        self._is_hf_model: bool = (
            model is not None and model.__class__.__name__ != "FallbackQwen2ForCausalLM"
        )
        if self._is_hf_model:
            _ensure_hf_sdpa_compact_mask_support()
        self._hf_eager_mode: bool = (
            self._is_hf_model
            and getattr(getattr(model, "config", None), "_attn_implementation", "sdpa") == "eager"
        )

        if artifacts is None:
            if model is None:
                raise ValueError("Either `model` or `artifacts` must be provided to StoikovKVCache.")
            artifacts = load_or_create_artifacts(model, artifacts_path=artifacts_path)

        model_dev: Optional[torch.device] = None
        if model is not None:
            try:
                model_dev = next(model.parameters()).device
            except StopIteration:
                model_dev = None

        self.U_Q: torch.Tensor = artifacts["U_Q"]
        self.D: torch.Tensor = artifacts["D"]
        self.W_ridge: torch.Tensor = artifacts["W_ridge"]
        if model_dev is not None and model_dev.type in ("cuda", "xpu", "mps"):
            self.U_Q = self.U_Q.to(device=model_dev)
            self.D = self.D.to(device=model_dev)
            self.W_ridge = self.W_ridge.to(device=model_dev)
        self._u_q_fp32: torch.Tensor = self.U_Q.to(dtype=torch.float32).contiguous()
        base_roles: torch.Tensor = artifacts["head_roles"].clone().long()

        self.num_layers: int = int(self.U_Q.shape[0])
        self.num_kv_heads: int = int(self.U_Q.shape[1])
        self.head_dim: int = int(self.U_Q.shape[2])
        self.subspace_rank: int = int(self.U_Q.shape[3])

        self.policy_mode: str = str(policy_mode).lower().strip()
        if self.policy_mode == "full_fp16":
            self.head_roles = torch.full_like(base_roles, ROLE_RETRIEVAL_FP16)
        elif self.policy_mode in ("standalone_h2o", "h2o"):
            self.head_roles = torch.full_like(base_roles, ROLE_PEAKED_H2O)
        elif self.policy_mode in ("snapkv",):
            self.head_roles = torch.full_like(base_roles, ROLE_PEAKED_H2O)
            enable_hmm_decay = False
            sink_tokens = 0
        elif self.policy_mode in ("quant_int4", "kivi_lowbit", "uniform_quant"):
            self.head_roles = torch.full_like(base_roles, ROLE_QUANT_LOWBIT)
            enable_hmm_decay = False
        elif self.policy_mode in ("standalone_lexico", "lexico"):
            self.head_roles = torch.full_like(base_roles, ROLE_DIFFUSE_LEXICO)
        elif self.policy_mode in ("hybrid", "stoikov", "stoikov_hybrid"):
            self.head_roles = base_roles
        else:
            raise ValueError(
                f"Unknown policy_mode={policy_mode!r}. Expected 'hybrid', 'full_fp16', "
                "'standalone_h2o', 'snapkv', 'quant_int4', or 'standalone_lexico'."
            )

        self.n_b: int = max(1, int(n_b))
        self.keep_ratio: float = float(keep_ratio)
        self.s_budget: int = max(1, int(s_budget))
        self.active_atoms: int = self.s_budget
        self.lambda1: float = float(lambda1)
        self.sink_tokens: int = max(0, int(sink_tokens))
        self.sink_bonus: float = float(sink_bonus)
        self.max_rollback_window: int = max(0, int(max_rollback_window))
        # Retain a small neighborhood around selected heavy hitters.  This
        # keeps a salient token's immediate syntax/context from being split
        # across an eviction boundary while preserving the same H2O budget.
        self.h2o_context_window: int = max(0, int(h2o_context_window))

        self.codec = ElasticLexicoCodec(self.D, self.W_ridge)
        # Key frame used by the dictionary. "pre_rope": keys are un-rotated (inverse RoPE at
        # their absolute position) before sparse coding and re-rotated after decoding. This is
        # required because post-RoPE keys rotate with position, which a static dictionary cannot
        # code sparsely. Legacy artifacts without this key keep the old post-RoPE behaviour.
        self.dict_key_frame: str = str(artifacts.get("dict_key_frame", "post_rope"))
        self.rope_theta: float = float(
            artifacts.get(
                "rope_theta",
                getattr(getattr(model, "config", None), "rope_theta", 1000000.0)
                if model is not None
                else 1000000.0,
            )
        )
        half = self.head_dim // 2
        self._rope_inv_freq = 1.0 / (
            self.rope_theta ** (torch.arange(0, half, dtype=torch.float32) * 2.0 / float(self.head_dim))
        )
        self._rope_cos_cache: Optional[torch.Tensor] = None
        self._rope_sin_cache: Optional[torch.Tensor] = None
        self._rope_cos_half: Optional[torch.Tensor] = None
        self._rope_sin_half: Optional[torch.Tensor] = None
        self._rope_cache_len: int = 0
        self._rope_cache_dev: Optional[torch.device] = None
        is_hybrid = self.policy_mode in ("hybrid", "stoikov", "stoikov_hybrid")
        self.controller = controller or OnlineRegimeChangeController(
            keep_ratio=self.keep_ratio,
            s_budget=self.s_budget,
            enable_hmm_decay=enable_hmm_decay,
            enable_stoikov_control=(is_hybrid and bool(enable_hmm_decay)),
        )
        self.controller_fn = controller_fn

        # Precomputed per-layer role homogeneity flags (avoids per-token GPU/CPU .item() syncs)
        self._layer_all_fp16: List[bool] = [
            bool(torch.all(self.head_roles[l] == ROLE_RETRIEVAL_FP16).item())
            for l in range(self.num_layers)
        ]
        self._layer_all_h2o: List[bool] = [
            bool(torch.all(self.head_roles[l] == ROLE_PEAKED_H2O).item())
            for l in range(self.num_layers)
        ]
        self._layer_all_lexico: List[bool] = [
            bool(torch.all(self.head_roles[l] == ROLE_DIFFUSE_LEXICO).item())
            for l in range(self.num_layers)
        ]
        self._layer_has_h2o: List[bool] = [
            bool(torch.any(self.head_roles[l] == ROLE_PEAKED_H2O).item())
            for l in range(self.num_layers)
        ]
        self._layer_lex_heads: List[List[int]] = [
            [h for h in range(self.num_kv_heads) if int(self.head_roles[l, h].item()) == ROLE_DIFFUSE_LEXICO]
            for l in range(self.num_layers)
        ]
        self._layer_ret_heads: List[List[int]] = [
            [h for h in range(self.num_kv_heads) if int(self.head_roles[l, h].item()) == ROLE_RETRIEVAL_FP16]
            for l in range(self.num_layers)
        ]
        self._layer_pure_ret_lex: List[bool] = [
            (len(self._layer_lex_heads[l]) > 0)
            and (len(self._layer_lex_heads[l]) + len(self._layer_ret_heads[l]) == self.num_kv_heads)
            for l in range(self.num_layers)
        ]

        # Batch-flush stride during 1-token decode: stage up to `n_b + _decode_flush_stride`
        # local tokens and flush `_decode_flush_stride` tokens in one vectorized batch.
        self._decode_flush_stride: int = max(1, min(64, self.n_b))

        # Per-layer state
        self._layer_seq_len: List[int] = [0 for _ in range(self.num_layers)]
        # Recent uncompressed FP16 sliding window [batch, num_kv_heads, win_len, head_dim]
        self._local_k: List[Optional[torch.Tensor]] = [None for _ in range(self.num_layers)]
        self._local_v: List[Optional[torch.Tensor]] = [None for _ in range(self.num_layers)]
        self._local_scores: List[Optional[torch.Tensor]] = [None for _ in range(self.num_layers)]
        self._local_positions: List[Optional[torch.Tensor]] = [None for _ in range(self.num_layers)]

        # Pre-allocated in-place slab backing `_local_*` during 1-token decode (zero torch.cat)
        self._local_buf_k: List[Optional[torch.Tensor]] = [None for _ in range(self.num_layers)]
        self._local_buf_v: List[Optional[torch.Tensor]] = [None for _ in range(self.num_layers)]
        self._local_buf_sc: List[Optional[torch.Tensor]] = [None for _ in range(self.num_layers)]
        self._local_buf_pos: List[Optional[torch.Tensor]] = [None for _ in range(self.num_layers)]
        self._slab_next_pos: List[int] = [-1 for _ in range(self.num_layers)]
        self._slab_scored_len: List[int] = [0 for _ in range(self.num_layers)]
        self._zero_score_buf: Optional[torch.Tensor] = None

        # Cached stacked history views & unified zero-copy [B, H_kv, hist_kept + cap, D] slabs
        # for pure-FP16 and pure-H2O layers, plus shared-storage 4D Lexico packs (zero extra VRAM)
        self._layer_compact_k: List[Optional[torch.Tensor]] = [None for _ in range(self.num_layers)]
        self._layer_compact_v: List[Optional[torch.Tensor]] = [None for _ in range(self.num_layers)]
        self._layer_compact_bytes: List[int] = [0 for _ in range(self.num_layers)]
        self._unified_k: List[Optional[torch.Tensor]] = [None for _ in range(self.num_layers)]
        self._unified_v: List[Optional[torch.Tensor]] = [None for _ in range(self.num_layers)]
        self._unified_hist_len: List[int] = [0 for _ in range(self.num_layers)]
        self._layer_lex_pack: List[Optional[Tuple[torch.Tensor, ...]]] = [None for _ in range(self.num_layers)]

        # Rollback ring buffer storing recently flushed tokens from the local window so
        # speculative `crop(max_length)` is 100% lossless even across local window boundaries
        self._flushed_ring_k: List[Optional[torch.Tensor]] = [None for _ in range(self.num_layers)]
        self._flushed_ring_v: List[Optional[torch.Tensor]] = [None for _ in range(self.num_layers)]
        self._flushed_ring_scores: List[Optional[torch.Tensor]] = [None for _ in range(self.num_layers)]
        self._flushed_ring_pos: List[Optional[torch.Tensor]] = [None for _ in range(self.num_layers)]

        # Historical routed tier per (layer_idx, kv_head_idx)
        self._tiers: List[List[_HeadTierState]] = [
            [
                _HeadTierState(role=int(self.head_roles[l_idx, h_idx].item()))
                for h_idx in range(self.num_kv_heads)
            ]
            for l_idx in range(self.num_layers)
        ]

        # Byte telemetry
        self.fp16_bytes: int = 0
        self.compressed_bytes: int = 0
        self.physical_resident_bytes: int = 0
        self.bytes_read_last_step: int = 0
        self.cumulative_bytes_read: int = 0
        self._layer_bytes_read: List[int] = [0 for _ in range(self.num_layers)]
        self.per_tier_token_counts: Dict[str, int] = {
            "local_window_fp16": 0,
            "retrieval_fp16": 0,
            "peaked_h2o_fp16": 0,
            "diffuse_lexico_sparse": 0,
            "evicted_h2o": 0,
        }

        # Optional AtlasKV-inspired HiKVP hierarchical offloaded retrieval & persistent Atlas bank
        self.hikvp_config: Optional[HiKVPConfig] = hikvp_config
        self.hikvp_store: Optional[HiKVPStore] = (
            HiKVPStore(
                num_layers=self.num_layers,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                head_roles=self.head_roles,
                config=hikvp_config,
                U_Q=self.U_Q,
            )
            if hikvp_config is not None
            else None
        )
        self.atlas_bank: Optional[Any] = atlas_bank
        self._query_hook_handles: List[Any] = []
        self._latest_pre_rope_q: List[Optional[torch.Tensor]] = [None for _ in range(self.num_layers)]
        self._atlas_retrievals: int = 0
        self._atlas_tokens_retrieved: int = 0
        self._atlas_pcie_bytes: int = 0
        self._draft_pcie_bytes: int = 0

    def set_hikvp_draft_mode(self, enabled: bool) -> None:
        """
        Toggles zero-PCIe draft mode on any attached `HiKVPStore` (and `AtlasMemoryBank` store).
        During Matryoshka speculative drafting, leaf fetches from CPU/NVMe are suppressed so PCIe
        transfers are amortized across the batched parallel verification pass.
        """
        if self.hikvp_store is not None:
            self.hikvp_store.set_draft_mode(bool(enabled))
        if self.atlas_bank is not None and hasattr(self.atlas_bank, "store") and self.atlas_bank.store is not None:
            self.atlas_bank.store.set_draft_mode(bool(enabled))

    def set_atlas_bank(self, bank: Optional[Any]) -> None:
        """Attaches or detaches a persistent `AtlasMemoryBank` for Rectangular Attention."""
        self.atlas_bank = bank

    def attach_query_hooks(self, model: nn.Module) -> None:
        """
        Registers lightweight forward hooks on each layer's `self_attn.q_proj` to capture live
        position-free pre-RoPE queries (`[B, H_kv, Q, D]`, mean-pooled per GQA group) for HiKVP
        hierarchical centroid pruning and layer-pipelined async prefetching. Remains 100%
        FlashAttention/SDPA-compatible (`output_attentions=False`).
        """
        self.detach_query_hooks()
        inner = getattr(model, "model", model)
        layers = getattr(inner, "layers", None)
        if layers is None:
            layers = getattr(model, "layers", None)
        if layers is None:
            raise ValueError("Cannot attach query hooks: model exposes neither `.model.layers` nor `.layers`.")

        num_kv = self.num_kv_heads
        d_head = self.head_dim

        for l_idx, layer in enumerate(layers):
            if l_idx >= self.num_layers:
                break
            attn = getattr(layer, "self_attn", None)
            q_proj = getattr(attn, "q_proj", None) if attn is not None else None
            if q_proj is None:
                continue

            def _make_q_hook(layer_index: int):
                def _hook(_mod: nn.Module, _inp: Any, out: torch.Tensor) -> None:
                    q_t = out[0] if isinstance(out, tuple) else out
                    if not isinstance(q_t, torch.Tensor) or q_t.dim() != 3:
                        return
                    bsz_q, q_len_q, total_dim = q_t.shape
                    if total_dim % d_head != 0:
                        return
                    n_q_heads = total_dim // d_head
                    if n_q_heads == num_kv:
                        q_kv = q_t.detach().view(bsz_q, q_len_q, num_kv, d_head).transpose(1, 2)
                    elif n_q_heads > num_kv and n_q_heads % num_kv == 0:
                        # Preserve all GQA query heads in the query axis [B, H_kv, g * Q, D]
                        # so HiKVP max-pooling across queries lets a single sharp retrieval head
                        # vote directly without being diluted by mean-pooling other query heads.
                        g_size = n_q_heads // num_kv
                        q_kv = (
                            q_t.detach()
                            .view(bsz_q, q_len_q, num_kv, g_size, d_head)
                            .permute(0, 2, 1, 3, 4)
                            .reshape(bsz_q, num_kv, q_len_q * g_size, d_head)
                        )
                    else:
                        q_kv = (
                            q_t.detach()
                            .view(bsz_q, q_len_q, n_q_heads, d_head)
                            .mean(dim=2, keepdim=True)
                            .expand(bsz_q, q_len_q, num_kv, d_head)
                            .transpose(1, 2)
                        )
                    self._latest_pre_rope_q[layer_index] = q_kv

                    # Layer-pipelined async prefetch for the next layer when not in draft mode
                    if (
                        self.hikvp_store is not None
                        and self.hikvp_config is not None
                        and self.hikvp_config.async_prefetch
                        and not self.hikvp_store.draft_mode
                        and (layer_index + 1) < self.num_layers
                    ):
                        self.hikvp_store.prefetch_layer(
                            layer_idx=layer_index + 1,
                            query_pre_rope=q_kv,
                            device=q_kv.device,
                            dtype=q_kv.dtype,
                        )

                return _hook

            self._query_hook_handles.append(q_proj.register_forward_hook(_make_q_hook(l_idx)))

    def detach_query_hooks(self) -> None:
        """Removes all registered `q_proj` forward hooks and clears cached pre-RoPE queries."""
        for handle in self._query_hook_handles:
            try:
                handle.remove()
            except Exception:
                pass
        self._query_hook_handles.clear()
        self._latest_pre_rope_q = [None for _ in range(self.num_layers)]

    def hikvp_telemetry(self) -> Dict[str, Any]:
        """Returns HiKVP offload + AtlasMemoryBank Rectangular Attention telemetry."""
        if self.hikvp_store is None and self.atlas_bank is None and self._atlas_retrievals == 0:
            return {}
        tel: Dict[str, Any] = dict(self.hikvp_store.telemetry()) if self.hikvp_store is not None else {}
        tel["draft_pcie_bytes"] = int(self._draft_pcie_bytes)
        tel["atlas_attached"] = self.atlas_bank is not None
        tel["atlas_retrievals"] = int(self._atlas_retrievals)
        tel["atlas_tokens_retrieved"] = int(self._atlas_tokens_retrieved)
        tel["atlas_pcie_bytes"] = int(self._atlas_pcie_bytes)
        if self.atlas_bank is not None and hasattr(self.atlas_bank, "stats"):
            tel["atlas_bank_stats"] = self.atlas_bank.stats()
        return tel

    def _get_pre_rope_query_for_layer(
        self,
        layer_idx: int,
        local_k: torch.Tensor,
        local_pos: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Returns live pre-RoPE query `[B, H_kv, Q, D]` captured via `attach_query_hooks`, or
        falls back cleanly to un-rotated recent local keys when hooks are not installed.
        """
        bsz = int(local_k.shape[0])
        q_cached = self._latest_pre_rope_q[layer_idx]
        if q_cached is not None and int(q_cached.shape[0]) == bsz:
            return q_cached
        win_len = int(local_k.shape[-2])
        tail_n = min(4, win_len)
        k_tail = local_k[:, :, -tail_n:, :]
        pos_tail = local_pos[-tail_n:] if local_pos is not None else None
        return self._rope_rotate(k_tail, pos_tail, inverse=True, force=True)

    def set_matryoshka_mode(self, active_atoms: int) -> None:
        """
        Sets the number of active sorted dictionary atoms `m` (`1 <= m <= s_budget`)
        used when reconstructing `ROLE_DIFFUSE_LEXICO` KV states:
        - `active_atoms = m` (e.g. `m = 4`): fast draft mode reading `3*m + 2` bytes/vector.
        - `active_atoms = s` (e.g. `s = 16`): full verification mode reading `3*s + 2` bytes/vector.
        """
        m = max(1, min(int(active_atoms), self.s_budget))
        if m != self.active_atoms:
            self.active_atoms = m
            # Mark cached Lexico reconstructions dirty so next read uses top-m atoms
            for l_idx in range(self.num_layers):
                for h_idx in range(self.num_kv_heads):
                    st = self._tiers[l_idx][h_idx]
                    if st.role == ROLE_DIFFUSE_LEXICO:
                        st.lex_recon_atoms = -1

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        idx = 0 if layer_idx is None else int(layer_idx)
        if idx < 0 or idx >= self.num_layers:
            return 0
        return int(self._layer_seq_len[idx])

    def get_mask_sizes(self, query_length: int, layer_idx: int = 0) -> Tuple[int, int]:
        """
        Returns the KV cache length and offset for the given layer and query length.
        Uses our internal sequence length tracking instead of HF DynamicCache's internal layers.
        Hugging Face asks before this layer's ``update`` call, so the incoming
        query tokens are not yet reflected in ``_layer_seq_len``.
        """
        idx = 0 if layer_idx is None else int(layer_idx)
        idx = max(0, min(idx, self.num_layers - 1)) if self.num_layers > 0 else 0
        seq_len = int(self._layer_seq_len[idx]) if self.num_layers > 0 else 0
        return (seq_len + int(query_length), 0)

    def get_query_offset(self, layer_idx: int = 0) -> int:
        return 0

    def get_max_cache_shape(self, layer_idx: int = 0) -> int:
        return -1

    def get_max_length(self) -> Optional[int]:
        return None

    def get_usable_length(self, new_seq_length: int, layer_idx: Optional[int] = 0) -> int:
        return self.get_seq_length(layer_idx)

    @property
    def is_initialized(self) -> bool:
        return self.get_seq_length(0) > 0

    @property
    def is_compileable(self) -> bool:
        return False

    @property
    def is_sliding(self) -> List[bool]:
        return [False] * self.num_layers

    @property
    def is_linear(self) -> List[bool]:
        return [False] * self.num_layers

    @property
    def is_croppable(self) -> bool:
        return True

    @property
    def batch_size(self) -> Optional[int]:
        if self._local_k and self._local_k[0] is not None:
            return int(self._local_k[0].shape[0])
        return None

    def __len__(self) -> int:
        return self.num_layers if self.is_initialized else 0

    def __getitem__(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if layer_idx < 0 or layer_idx >= self.num_layers or self._local_k[layer_idx] is None:
            raise KeyError(f"Cache at layer {layer_idx} is empty or out of bounds.")
        lk = self._local_k[layer_idx]
        k_out, v_out, _ = self._materialize_layer_kv(
            layer_idx=layer_idx, dtype=lk.dtype, device=lk.device, q_len=1
        )
        return k_out, v_out

    def __iter__(self):
        for l_idx in range(len(self)):
            yield self[l_idx]

    def reset(self) -> None:
        self.crop(0)

    @torch.no_grad()
    def _apply_batch_index(self, idx: torch.Tensor) -> None:
        for l_idx in range(self.num_layers):
            self._materialize_pending_slab_scores(l_idx)
            self._local_buf_k[l_idx] = None
            self._local_buf_v[l_idx] = None
            self._local_buf_sc[l_idx] = None
            self._local_buf_pos[l_idx] = None
            self._slab_next_pos[l_idx] = -1
            self._slab_scored_len[l_idx] = 0
            self._layer_compact_k[l_idx] = None
            self._layer_compact_v[l_idx] = None
            self._unified_k[l_idx] = None
            self._unified_v[l_idx] = None
            self._layer_lex_pack[l_idx] = None
            if self._local_k[l_idx] is not None:
                dev = self._local_k[l_idx].device
                b_idx = idx.to(dev)
                self._local_k[l_idx] = self._local_k[l_idx].index_select(0, b_idx)
                self._local_v[l_idx] = self._local_v[l_idx].index_select(0, b_idx)
                self._local_scores[l_idx] = self._local_scores[l_idx].index_select(0, b_idx)
            if self._flushed_ring_k[l_idx] is not None:
                dev = self._flushed_ring_k[l_idx].device
                b_idx = idx.to(dev)
                self._flushed_ring_k[l_idx] = self._flushed_ring_k[l_idx].index_select(0, b_idx)
                self._flushed_ring_v[l_idx] = self._flushed_ring_v[l_idx].index_select(0, b_idx)
                self._flushed_ring_scores[l_idx] = self._flushed_ring_scores[l_idx].index_select(0, b_idx)
            for h_idx in range(self.num_kv_heads):
                st = self._tiers[l_idx][h_idx]
                if st.fp16_k is not None:
                    b_idx = idx.to(st.fp16_k.device)
                    st.fp16_k = st.fp16_k.index_select(0, b_idx)
                    st.fp16_v = st.fp16_v.index_select(0, b_idx)
                if st.h2o_k is not None:
                    b_idx = idx.to(st.h2o_k.device)
                    st.h2o_k = st.h2o_k.index_select(0, b_idx)
                    st.h2o_v = st.h2o_v.index_select(0, b_idx)
                    st.h2o_indices = st.h2o_indices.index_select(0, b_idx)
                    st.h2o_scores = st.h2o_scores.index_select(0, b_idx)
                if st.lex_k_idx is not None and st.lex_k_val is not None:
                    b_idx = idx.to(st.lex_k_idx.device)
                    st.lex_k_idx = st.lex_k_idx.index_select(0, b_idx)
                    st.lex_v_idx = st.lex_v_idx.index_select(0, b_idx)
                    raw_k = torch.Tensor._make_subclass(torch.Tensor, st.lex_k_val).index_select(0, b_idx)
                    raw_v = torch.Tensor._make_subclass(torch.Tensor, st.lex_v_val).index_select(0, b_idx)
                    st.lex_k_val = QuantizedInt8Values(raw_k, st.lex_k_val.scale.index_select(0, b_idx))
                    st.lex_v_val = QuantizedInt8Values(raw_v, st.lex_v_val.scale.index_select(0, b_idx))
                    st.lex_recon_k = None
                    st.lex_recon_v = None
                    st.lex_recon_atoms = -1
                if st.quant_k_packed is not None:
                    b_idx = idx.to(st.quant_k_packed.device)
                    st.quant_k_packed = st.quant_k_packed.index_select(0, b_idx)
                    st.quant_k_min = st.quant_k_min.index_select(0, b_idx)
                    st.quant_k_scale = st.quant_k_scale.index_select(0, b_idx)
                    st.quant_v_packed = st.quant_v_packed.index_select(0, b_idx)
                    st.quant_v_min = st.quant_v_min.index_select(0, b_idx)
                    st.quant_v_scale = st.quant_v_scale.index_select(0, b_idx)

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        self._apply_batch_index(beam_idx)

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        self._apply_batch_index(indices)

    def batch_repeat_interleave(self, repeats: int) -> None:
        bsz = self.batch_size
        if bsz is None:
            return
        idx = torch.arange(bsz).repeat_interleave(int(repeats))
        self._apply_batch_index(idx)

    @torch.no_grad()
    def _ensure_rope_table(self, max_pos: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        if (
            self._rope_cos_cache is None
            or self._rope_sin_cache is None
            or self._rope_cache_dev != device
            or self._rope_cache_len < max_pos
        ):
            cap = max(max_pos, 4096)
            inv_freq = self._rope_inv_freq.to(device=device)
            pos_vec = torch.arange(cap, device=device, dtype=torch.float32)
            freqs = torch.outer(pos_vec, inv_freq)
            self._rope_cos_half = freqs.cos().contiguous()
            self._rope_sin_half = freqs.sin().contiguous()
            self._rope_cos_cache = torch.cat((self._rope_cos_half, self._rope_cos_half), dim=-1)
            self._rope_sin_cache = torch.cat((self._rope_sin_half, self._rope_sin_half), dim=-1)
            self._rope_cache_len = cap
            self._rope_cache_dev = device
        return self._rope_cos_cache, self._rope_sin_cache

    @torch.no_grad()
    def _rope_rotate(
        self, x: torch.Tensor, positions: Optional[torch.Tensor], inverse: bool, force: bool = False
    ) -> torch.Tensor:
        """
        Applies (or inverts) standard half-split RoPE to `x` (`[..., T, D]`) at absolute
        `positions` (`[T]`, or `0..T-1` when `positions is None`). Matches `_compute_rope_cos_sin`
        / `_apply_rotary_pos_emb`: `x_rot = x * cos + rotate_half(x) * sin`; the inverse uses `-sin`.
        """
        t_len = int(x.shape[-2])
        if (not force and self.dict_key_frame != "pre_rope") or t_len == 0:
            return x
        if positions is None:
            cos_tab, sin_tab = self._ensure_rope_table(t_len, x.device)
            cos = cos_tab[:t_len]
            sin = sin_tab[:t_len]
        else:
            max_p = int(positions[-1].item()) + 1 if positions.numel() > 0 else t_len
            if max_p <= 65536 and int(positions[0].item()) >= 0:
                cos_tab, sin_tab = self._ensure_rope_table(max_p, x.device)
                cos = cos_tab.index_select(0, positions)
                sin = sin_tab.index_select(0, positions)
            else:
                inv_freq = self._rope_inv_freq.to(device=x.device)
                freqs = torch.outer(positions.to(device=x.device, dtype=torch.float32), inv_freq)
                emb = torch.cat((freqs, freqs), dim=-1)  # [T, D]
                cos = emb.cos()
                sin = emb.sin()
        if inverse:
            sin = -sin
        xf = x.float()
        half = xf.shape[-1] // 2
        x_rot_half = torch.cat((-xf[..., half:], xf[..., :half]), dim=-1)
        return (xf * cos + x_rot_half * sin).to(x.dtype)

    @torch.no_grad()
    def _compute_step_scores_and_controller(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        start_pos: int,
    ) -> Tuple[torch.Tensor, float, float, int]:
        """
        Computes FlashAttention-compatible write-time directional probe score `I_t`
        (softplus-gated on pretrained HuggingFace models; sigmoid-gated at unit scale,
        plus `+100.0` sink gate for global token indices `0..3` and pre-RoPE rarity spans
        during hybrid prefill) and queries the online regime-change decay controller.
        Never requires `output_attentions=True`.
        """
        bsz, n_kv, q_len, d_head = key_states.shape

        # Layers with zero H2O heads (pure FP16, pure Lexico, pure Retrieval+Lexico, Lowbit)
        # never consume importance scores unless the regime controller needs Layer 0.
        if (
            (not self._layer_has_h2o[layer_idx])
            and self.controller_fn is None
            and (layer_idx > 0 or not self.controller.enable_hmm_decay)
        ):
            if (
                self._zero_score_buf is None
                or self._zero_score_buf.device != key_states.device
                or self._zero_score_buf.shape[0] < bsz
                or self._zero_score_buf.shape[1] < n_kv
                or self._zero_score_buf.shape[2] < q_len
            ):
                with torch.inference_mode(False):
                    self._zero_score_buf = torch.zeros(
                        (max(1, bsz), max(1, n_kv), max(64, q_len)),
                        device=key_states.device,
                        dtype=torch.float32,
                    )
            zero_sc = self._zero_score_buf[:bsz, :n_kv, :q_len]
            eff_keep = (
                self.controller.last_keep_ratio
                if self.controller.enable_stoikov_control
                else self.keep_ratio
            )
            return zero_sc, float(self.controller.last_beta_t), float(eff_keep), self.s_budget

        if self._u_q_fp32.device != key_states.device:
            self._u_q_fp32 = self._u_q_fp32.to(device=key_states.device)
        u_q_l = self._u_q_fp32[layer_idx]  # [H, D, r]

        # Fast single-launch fused GPU kernel path when projection tensor `proj` is not needed
        # by SnapKV pooling, hybrid prefill rarity-span probing (q_len >= 64), or Layer-0 HMM.
        need_layer0_hmm = (layer_idx == 0) and (self.controller_fn is None) and self.controller.enable_hmm_decay
        need_rarity_span = (
            self._is_hf_model
            and self.policy_mode in ("hybrid", "stoikov", "stoikov_hybrid")
            and q_len >= 64
        )
        if (
            self.policy_mode != "snapkv"
            and (not need_layer0_hmm)
            and (not need_rarity_span)
            and can_use_triton(key_states)
        ):
            scores = triton_fused_write_probe(
                key_states,
                value_states,
                u_q_l,
                start_pos=start_pos,
                sink_tokens=self.sink_tokens,
                sink_bonus=self.sink_bonus,
                softplus_mode=self._is_hf_model,
            )
            seq_after = start_pos + q_len
            if self.controller_fn is not None:
                beta_t, keep_ratio, s_budget = self.controller_fn(layer_idx, seq_after)
            elif layer_idx == 0:
                beta_t = 1.0
                keep_ratio = self.controller.compute_dynamic_keep_ratio(seq_after)
                s_budget = self.s_budget
            else:
                beta_t = self.controller.last_beta_t
                keep_ratio = (
                    self.controller.last_keep_ratio
                    if self.controller.enable_stoikov_control
                    else self.keep_ratio
                )
                s_budget = self.s_budget
            return scores, float(beta_t), float(keep_ratio), int(s_budget)

        k_f = key_states.float()
        v_f = value_states.float()

        # Project K onto U_Q: [B, H, T, D] @ [H, D, r] -> [B, H, T, r]
        proj = torch.einsum("bhtd,hdr->bhtr", k_f, u_q_l)
        v_norm = torch.linalg.vector_norm(v_f, ord=2, dim=-1)  # [B, H, T]
        if self.policy_mode == "snapkv":
            # SnapKV baseline: observation-window 1D pooling (kernel_size=5) without directional
            # sign-gating, retrieval head protection, or HMM topic-shift adaptation.
            raw_obs = torch.linalg.vector_norm(proj[..., :4], ord=2, dim=-1) * v_norm  # [B, H, T]
            if q_len >= 5:
                pooled = F.avg_pool1d(
                    raw_obs.reshape(bsz * n_kv, 1, q_len),
                    kernel_size=5,
                    stride=1,
                    padding=2,
                ).reshape(bsz, n_kv, q_len)
                scores = pooled
            else:
                scores = raw_obs
        elif self._is_hf_model:
            dir_gate = F.softplus(proj[..., 0])
            proj_norm = torch.linalg.vector_norm(F.relu(proj), ord=2, dim=-1)  # [B, H, T]
            v_factor = torch.where(v_norm > 0, v_norm.clamp(min=1e-6).sqrt(), torch.zeros_like(v_norm))
            scores = dir_gate * (1.0 + 0.25 * proj_norm) * v_factor  # [B, H, T]
            if self.policy_mode in ("hybrid", "stoikov", "stoikov_hybrid") and q_len >= 64:
                # Pre-RoPE Key Rarity Span Probing: repeated background sentences collapse to
                # identical key directions in the pre-RoPE frame, whereas unique facts/passkeys
                # have low distant key similarity (|t - t'| > 40). Smoothing over a 15-token span
                # keeps multi-token BPE entities/numbers contiguous across eviction boundaries.
                pos_vec = torch.arange(start_pos, start_pos + q_len, device=key_states.device, dtype=torch.long)
                k_pre = self._rope_rotate(key_states, pos_vec, inverse=True).float()
                k_pre_norm = F.normalize(k_pre, p=2, dim=-1)
                sim = torch.matmul(k_pre_norm, k_pre_norm.transpose(-1, -2))
                dist_mask = (pos_vec.unsqueeze(1) - pos_vec.unsqueeze(0)).abs() > 40
                sim_distant = sim.masked_fill(~dist_mask.unsqueeze(0).unsqueeze(0), -1.0)
                k_rep = min(6, max(1, q_len // 24))
                top_rep = torch.topk(sim_distant, k=k_rep, dim=-1).values.clamp(min=0.0).mean(dim=-1)
                rarity = (1.0 - top_rep).clamp(min=0.0)
                rarity_span = F.avg_pool1d(
                    rarity.reshape(bsz * n_kv, 1, q_len), kernel_size=15, stride=1, padding=7
                ).reshape(bsz, n_kv, q_len)
                k_anchor = max(8, min(32, int(q_len * 0.045)))
                _, anchor_idx = torch.topk(rarity_span, k=min(k_anchor, q_len), dim=-1)
                anchor_mask = torch.zeros_like(scores)
                anchor_mask.scatter_(-1, anchor_idx, 1.0)
                scores = scores + 80.0 * anchor_mask + 8.0 * rarity_span
        else:
            dir_gate = torch.sigmoid(proj[..., 0])
            proj_norm = torch.linalg.vector_norm(proj, ord=2, dim=-1)  # [B, H, T]
            scores = dir_gate * proj_norm * v_norm  # [B, H, T]

        # Apply sink gate (+100.0) for global token indices 0..3
        if self.sink_tokens > 0 and start_pos < self.sink_tokens:
            sink_end = min(q_len, self.sink_tokens - start_pos)
            scores[:, :, :sink_end] = scores[:, :, :sink_end] + self.sink_bonus

        # Query controller (update HMM on layer 0 or via custom controller_fn)
        seq_after = start_pos + q_len
        if self.controller_fn is not None:
            beta_t, keep_ratio, s_budget = self.controller_fn(layer_idx, seq_after)
        elif layer_idx == 0:
            # Compute subspace projection energy profile across tokens
            # During prefill (q_len > 128), evaluate sliding windows so topic shifts inside prefill are detected
            if q_len > 128 and self.controller.enable_hmm_decay:
                chunk_sz = 64
                beta_t = self.controller.beta_stationary
                keep_ratio = self.keep_ratio
                s_budget = self.s_budget
                if self._is_hf_model:
                    self.controller.ema_dev = 0.015
                for c_start in range(0, q_len, chunk_sz):
                    c_end = min(q_len, c_start + chunk_sz)
                    if self._is_hf_model:
                        # Use position-free value states so RoPE rotation does not mask semantic topic shifts
                        chunk_energy = v_f[:, :, c_start:c_end, :32].mean(dim=(0, 2))
                        feat = F.normalize(chunk_energy.detach().float().flatten(), p=2.0, dim=0, eps=1e-8)
                        b_step = self.controller._update_drift_and_microstructure(feat, min_dev_floor=0.01)
                        keep_ratio = self.controller.compute_dynamic_keep_ratio(start_pos + c_end)
                    else:
                        chunk_energy = proj[:, :, c_start:c_end, :].abs().mean(dim=(0, 2))
                        b_step, keep_ratio, s_budget = self.controller.observe_and_query(
                            chunk_energy, start_pos + c_end
                        )
                    if b_step < beta_t:
                        beta_t = b_step
                self.controller.last_beta_t = float(beta_t)
            elif self.controller.enable_hmm_decay:
                step_energy = proj.abs().mean(dim=(0, 2))
                beta_t, keep_ratio, s_budget = self.controller.observe_and_query(
                    step_energy, seq_after
                )
            else:
                beta_t = 1.0
                keep_ratio = self.controller.compute_dynamic_keep_ratio(seq_after)
                s_budget = self.s_budget
        else:
            beta_t = self.controller.last_beta_t
            keep_ratio = self.controller.last_keep_ratio if self.controller.enable_stoikov_control else self.keep_ratio
            s_budget = self.s_budget

        return scores, float(beta_t), float(keep_ratio), int(s_budget)

    @torch.no_grad()
    def _route_overflow_to_tiers(
        self,
        layer_idx: int,
        overflow_k: torch.Tensor,
        overflow_v: torch.Tensor,
        overflow_scores: torch.Tensor,
        overflow_pos: torch.Tensor,
        beta_t: float,
        keep_ratio: float,
        s_budget: int,
        total_seq_len: int,
    ) -> None:
        """
        Routes tokens aging out of the `n_b = 64` local FP16 window into per-head tiers:
        - `ROLE_RETRIEVAL_FP16 (0)`: 100% uncompressed FP16
        - `ROLE_PEAKED_H2O (1)`: Top `k = int(seq_len * keep_ratio)` heavy hitters by `V_i(t)`
        - `ROLE_DIFFUSE_LEXICO (2)`: 1-step ElasticNet sparse dictionary atoms (`s_budget`)
        """
        bsz, n_kv, n_over, d_head = overflow_k.shape
        if n_over == 0:
            return
        self._layer_compact_k[layer_idx] = None
        self._layer_compact_v[layer_idx] = None
        self._unified_k[layer_idx] = None
        self._unified_v[layer_idx] = None
        self._layer_lex_pack[layer_idx] = None

        # A single non-finite activation would otherwise poison the persistent
        # compressed tier and make every later decode fail.  Preserve the valid
        # state while mapping exceptional values to neutral zero vectors/scores.
        overflow_k = torch.nan_to_num(overflow_k, nan=0.0, posinf=0.0, neginf=0.0)
        overflow_v = torch.nan_to_num(overflow_v, nan=0.0, posinf=0.0, neginf=0.0)
        overflow_scores = torch.nan_to_num(overflow_scores, nan=0.0, posinf=0.0, neginf=0.0)

        # Save last `max_rollback_window` flushed tokens in ring buffer for lossless `crop()`
        if self.max_rollback_window > 0:
            if self._flushed_ring_k[layer_idx] is None:
                self._flushed_ring_k[layer_idx] = overflow_k[:, :, -self.max_rollback_window :, :].clone()
                self._flushed_ring_v[layer_idx] = overflow_v[:, :, -self.max_rollback_window :, :].clone()
                self._flushed_ring_scores[layer_idx] = overflow_scores[:, :, -self.max_rollback_window :].clone()
                self._flushed_ring_pos[layer_idx] = overflow_pos[-self.max_rollback_window :].clone()
            else:
                prev_pos = self._flushed_ring_pos[layer_idx]
                keep_prev = prev_pos < overflow_pos[0]
                self._flushed_ring_k[layer_idx] = torch.cat(
                    [self._flushed_ring_k[layer_idx][:, :, keep_prev, :], overflow_k], dim=-2
                )[:, :, -self.max_rollback_window :, :]
                self._flushed_ring_v[layer_idx] = torch.cat(
                    [self._flushed_ring_v[layer_idx][:, :, keep_prev, :], overflow_v], dim=-2
                )[:, :, -self.max_rollback_window :, :]
                self._flushed_ring_scores[layer_idx] = torch.cat(
                    [self._flushed_ring_scores[layer_idx][:, :, keep_prev], overflow_scores], dim=-1
                )[:, :, -self.max_rollback_window :]
                self._flushed_ring_pos[layer_idx] = torch.cat(
                    [prev_pos[keep_prev], overflow_pos], dim=0
                )[-self.max_rollback_window :]

        # Compute positional decay weights inside overflow chunk only when a shift decay is active
        if n_over > 1 and beta_t < 1.0:
            exps = torch.arange(n_over - 1, -1, -1, device=overflow_scores.device, dtype=torch.float32)
            if self._is_hf_model:
                # Scale exponent per 16-token window so single-pass prefill does not underflow float32,
                # while shielding high-rarity/sink anchors (score >= 60.0) from topic-shift eviction.
                decay_vec = (beta_t ** (exps / 16.0)).view(1, 1, n_over)
                is_anchor = overflow_scores >= 60.0
                overflow_scores_decayed = torch.where(is_anchor, overflow_scores, overflow_scores * decay_vec)
            else:
                decay_vec = (beta_t ** exps).view(1, 1, n_over)
                overflow_scores_decayed = overflow_scores * decay_vec
        else:
            overflow_scores_decayed = overflow_scores

        # H2O heavy-hitter budget: k = int(seq_len * keep_ratio)
        k_budget = max(1, int(total_seq_len * keep_ratio))

        # Vectorized multi-head batch encode for non-offloaded ROLE_DIFFUSE_LEXICO heads:
        # Packs [K_pre_rope, V] across all Lexico heads in this layer into ONE [2, B, H_lex, n_over, D]
        # GEMM + soft-threshold + top-s pass instead of 2 * H_lex sequential Python calls.
        batched_lex_heads = [
            h_i
            for h_i in self._layer_lex_heads[layer_idx]
            if not (self.hikvp_store is not None and self.hikvp_store.is_offloaded(layer_idx, h_i))
        ]
        pre_encoded_lex: Dict[int, Tuple[torch.Tensor, QuantizedInt8Values, torch.Tensor, QuantizedInt8Values]] = {}
        if len(batched_lex_heads) > 1:
            k_lex_sub = overflow_k[:, batched_lex_heads, :, :]  # [B, H_lex, n_over, D]
            v_lex_sub = overflow_v[:, batched_lex_heads, :, :]  # [B, H_lex, n_over, D]
            k_lex_code = self._rope_rotate(k_lex_sub, overflow_pos, inverse=True)
            kv_stacked = torch.stack([k_lex_code, v_lex_sub], dim=0)  # [2, B, H_lex, n_over, D]
            idx_all, val_all = self.codec.encode(
                kv_stacked, layer_idx=layer_idx, s=s_budget, lambda1=self.lambda1, validate_finite=False
            )
            raw_all = torch.Tensor._make_subclass(torch.Tensor, val_all)
            sc_all = val_all.scale
            for pos_i, h_i in enumerate(batched_lex_heads):
                pre_encoded_lex[h_i] = (
                    idx_all[0, :, pos_i],
                    QuantizedInt8Values(raw_all[0, :, pos_i], sc_all[0, :, pos_i]),
                    idx_all[1, :, pos_i],
                    QuantizedInt8Values(raw_all[1, :, pos_i], sc_all[1, :, pos_i]),
                )

        for h_idx in range(n_kv):
            st = self._tiers[layer_idx][h_idx]
            k_h = overflow_k[:, h_idx, :, :]  # [B, n_over, D]
            v_h = overflow_v[:, h_idx, :, :]  # [B, n_over, D]
            sc_h = overflow_scores_decayed[:, h_idx, :]  # [B, n_over]

            # HiKVP hierarchical offloaded path (un-rotates into pre-RoPE frame and offloads to CPU/mmap)
            if self.hikvp_store is not None and self.hikvp_store.is_offloaded(layer_idx, h_idx):
                k_pre = self._rope_rotate(k_h, overflow_pos, inverse=True, force=True)
                self.hikvp_store.append_pre_rope(layer_idx, h_idx, k_pre, v_h, overflow_pos)
                continue

            if st.role == ROLE_RETRIEVAL_FP16:
                if st.fp16_k is None:
                    st.fp16_k = k_h
                    st.fp16_v = v_h
                else:
                    st.fp16_k = torch.cat([st.fp16_k, k_h], dim=-2)
                    st.fp16_v = torch.cat([st.fp16_v, v_h], dim=-2)

            elif st.role == ROLE_PEAKED_H2O:
                pos_b = overflow_pos.unsqueeze(0).expand(bsz, n_over)
                if st.h2o_k is None:
                    cand_k = k_h
                    cand_v = v_h
                    cand_sc = sc_h
                    cand_pos = pos_b
                else:
                    # Existing heavy hitters have already been decayed by beta_t^n_over in update()
                    cand_k = torch.cat([st.h2o_k, k_h], dim=-2)
                    cand_v = torch.cat([st.h2o_v, v_h], dim=-2)
                    cand_sc = torch.cat([st.h2o_scores, sc_h], dim=-1)
                    cand_pos = torch.cat([st.h2o_indices, pos_b], dim=-1)

                if cand_k.shape[-2] > k_budget:
                    # Seed neighborhoods from the highest running scores, then
                    # keep the best ``k_budget`` positions with neighborhood
                    # membership taking priority.  This retains context without
                    # increasing the configured memory budget.
                    if self.h2o_context_window > 0:
                        span = 2 * self.h2o_context_window + 1
                        seed_count = max(1, (k_budget + span - 1) // span)
                        seed_idx = torch.topk(
                            cand_sc, k=min(seed_count, cand_sc.shape[-1]), dim=-1, largest=True, sorted=False
                        ).indices
                        positions = torch.arange(cand_sc.shape[-1], device=cand_sc.device).view(1, 1, -1)
                        seed_positions = seed_idx.unsqueeze(-1)
                        in_context = (
                            (positions - seed_positions).abs() <= self.h2o_context_window
                        ).any(dim=1)
                        # Scores are finite at this boundary; a large priority
                        # term makes contextual tokens win before other tokens.
                        priority = in_context.to(cand_sc.dtype) * 1e12 + cand_sc
                        _, top_idx = torch.topk(priority, k=k_budget, dim=-1, largest=True, sorted=False)
                    else:
                        _, top_idx = torch.topk(cand_sc, k=k_budget, dim=-1, largest=True, sorted=False)
                    top_idx_sorted, _ = torch.sort(top_idx, dim=-1)
                    gather_idx = top_idx_sorted.unsqueeze(-1).expand(-1, -1, d_head)
                    st.h2o_k = torch.gather(cand_k, dim=-2, index=gather_idx)
                    st.h2o_v = torch.gather(cand_v, dim=-2, index=gather_idx)
                    st.h2o_scores = torch.gather(cand_sc, dim=-1, index=top_idx_sorted)
                    st.h2o_indices = torch.gather(cand_pos, dim=-1, index=top_idx_sorted)
                else:
                    st.h2o_k = cand_k
                    st.h2o_v = cand_v
                    st.h2o_scores = cand_sc
                    st.h2o_indices = cand_pos

            elif st.role == ROLE_DIFFUSE_LEXICO:
                if h_idx in pre_encoded_lex:
                    k_idx, k_val, v_idx, v_val = pre_encoded_lex[h_idx]
                else:
                    k_h_code = self._rope_rotate(k_h, overflow_pos, inverse=True)
                    k_idx, k_val = self.codec.encode(
                        k_h_code, layer_idx=layer_idx, s=s_budget, lambda1=self.lambda1, validate_finite=False
                    )
                    v_idx, v_val = self.codec.encode(
                        v_h, layer_idx=layer_idx, s=s_budget, lambda1=self.lambda1, validate_finite=False
                    )
                if st.lex_k_idx is None:
                    st.lex_k_idx = k_idx
                    st.lex_k_val = k_val
                    st.lex_v_idx = v_idx
                    st.lex_v_val = v_val
                else:
                    st.lex_k_idx = torch.cat([st.lex_k_idx, k_idx], dim=-2)
                    st.lex_v_idx = torch.cat([st.lex_v_idx, v_idx], dim=-2)
                    new_k_raw = torch.cat(
                        [
                            torch.Tensor._make_subclass(torch.Tensor, st.lex_k_val),
                            torch.Tensor._make_subclass(torch.Tensor, k_val),
                        ],
                        dim=-2,
                    )
                    new_k_sc = torch.cat([st.lex_k_val.scale, k_val.scale], dim=-2)
                    st.lex_k_val = QuantizedInt8Values(new_k_raw, new_k_sc)

                    new_v_raw = torch.cat(
                        [
                            torch.Tensor._make_subclass(torch.Tensor, st.lex_v_val),
                            torch.Tensor._make_subclass(torch.Tensor, v_val),
                        ],
                        dim=-2,
                    )
                    new_v_sc = torch.cat([st.lex_v_val.scale, v_val.scale], dim=-2)
                    st.lex_v_val = QuantizedInt8Values(new_v_raw, new_v_sc)

                if (
                    self.cache_lexico_reconstructions
                    and st.lex_recon_k is not None
                    and st.lex_recon_atoms == self.active_atoms
                ):
                    rk = self.codec.decode(
                        k_idx,
                        k_val,
                        layer_idx=layer_idx,
                        active_atoms=self.active_atoms,
                        check_bounds=False,
                    ).to(device=overflow_k.device, dtype=overflow_k.dtype)
                    rk = self._rope_rotate(rk, overflow_pos, inverse=False)
                    rv = self.codec.decode(
                        v_idx,
                        v_val,
                        layer_idx=layer_idx,
                        active_atoms=self.active_atoms,
                        check_bounds=False,
                    ).to(device=overflow_v.device, dtype=overflow_v.dtype)
                    st.lex_recon_k = torch.cat([st.lex_recon_k, rk], dim=-2)
                    st.lex_recon_v = torch.cat([st.lex_recon_v, rv], dim=-2)
                    st.lex_recon_len = int(st.lex_recon_k.shape[-2])
                else:
                    st.lex_recon_k = None
                    st.lex_recon_v = None
                    st.lex_recon_atoms = -1

            elif st.role == ROLE_QUANT_LOWBIT:
                kp, km, ks = _quantize_lowbit_2bit(k_h)
                vp, vm, vs = _quantize_lowbit_2bit(v_h)
                if st.quant_k_packed is None:
                    st.quant_k_packed, st.quant_k_min, st.quant_k_scale = kp, km, ks
                    st.quant_v_packed, st.quant_v_min, st.quant_v_scale = vp, vm, vs
                else:
                    st.quant_k_packed = torch.cat([st.quant_k_packed, kp], dim=-2)
                    st.quant_k_min = torch.cat([st.quant_k_min, km], dim=-2)
                    st.quant_k_scale = torch.cat([st.quant_k_scale, ks], dim=-2)
                    st.quant_v_packed = torch.cat([st.quant_v_packed, vp], dim=-2)
                    st.quant_v_min = torch.cat([st.quant_v_min, vm], dim=-2)
                    st.quant_v_scale = torch.cat([st.quant_v_scale, vs], dim=-2)

    @torch.no_grad()
    def _bind_unified_slab(
        self,
        layer_idx: int,
        ck_src: torch.Tensor,
        cv_src: torch.Tensor,
        local_k: torch.Tensor,
        local_v: torch.Tensor,
        is_h2o: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Packs historical `ck_src`/`cv_src` (`[B, H_kv, hist_kept, D]`) and the active local decode
        slab (`cap = n_b + _decode_flush_stride`) into a single contiguous buffer so 1-token decode
        steps return an O(1) slice `uk[:, :, :hist_kept + win_len, :]` with zero `torch.cat` and
        zero duplicate VRAM (`st.h2o_k`/`st.fp16_k` and `_local_buf_k` share storage).
        """
        bsz, n_kv, hist_kept, d_head = ck_src.shape
        win_len = int(local_k.shape[-2])
        cap = self.n_b + self._decode_flush_stride
        if win_len > cap or hist_kept <= 0:
            return ck_src, cv_src
        with torch.inference_mode(False):
            uk = torch.empty((bsz, n_kv, hist_kept + cap, d_head), dtype=ck_src.dtype, device=ck_src.device)
            uv = torch.empty((bsz, n_kv, hist_kept + cap, d_head), dtype=cv_src.dtype, device=cv_src.device)
        uk[:, :, :hist_kept, :].copy_(ck_src)
        uv[:, :, :hist_kept, :].copy_(cv_src)
        uk[:, :, hist_kept : hist_kept + win_len, :].copy_(local_k)
        uv[:, :, hist_kept : hist_kept + win_len, :].copy_(local_v)
        ck_view = uk[:, :, :hist_kept, :]
        cv_view = uv[:, :, :hist_kept, :]
        for h_i in range(n_kv):
            if is_h2o:
                self._tiers[layer_idx][h_i].h2o_k = ck_view[:, h_i]
                self._tiers[layer_idx][h_i].h2o_v = cv_view[:, h_i]
            else:
                self._tiers[layer_idx][h_i].fp16_k = ck_view[:, h_i]
                self._tiers[layer_idx][h_i].fp16_v = cv_view[:, h_i]
        self._layer_compact_k[layer_idx] = ck_view
        self._layer_compact_v[layer_idx] = cv_view
        self._local_buf_k[layer_idx] = uk[:, :, hist_kept : hist_kept + cap, :]
        self._local_buf_v[layer_idx] = uv[:, :, hist_kept : hist_kept + cap, :]
        self._local_k[layer_idx] = uk[:, :, hist_kept : hist_kept + win_len, :]
        self._local_v[layer_idx] = uv[:, :, hist_kept : hist_kept + win_len, :]
        self._unified_k[layer_idx] = uk
        self._unified_v[layer_idx] = uv
        self._unified_hist_len[layer_idx] = hist_kept
        return ck_view, cv_view

    @torch.no_grad()
    def _get_or_build_lexico_pack(
        self, layer_idx: int, lex_heads: List[int]
    ) -> Optional[Tuple[torch.Tensor, ...]]:
        """
        Builds shared-storage 4D `[B, H_lex, T_hist, s]` views over all Lexico heads in `layer_idx`
        for single-launch `triton_fused_kv_lexico_materialize`, rebinding per-head tier views so
        zero duplicate VRAM is retained.
        """
        pack = self._layer_lex_pack[layer_idx]
        if pack is not None:
            return pack
        if not lex_heads:
            return None
        first_st = self._tiers[layer_idx][lex_heads[0]]
        if (
            first_st.lex_k_idx is None
            or first_st.lex_k_val is None
            or first_st.lex_v_idx is None
            or first_st.lex_v_val is None
        ):
            return None
        t_lex = int(first_st.lex_k_idx.shape[-2])
        if t_lex <= 0:
            return None
        for h_i in lex_heads[1:]:
            st_i = self._tiers[layer_idx][h_i]
            if st_i.lex_k_idx is None or int(st_i.lex_k_idx.shape[-2]) != t_lex:
                return None
        k_idx_4d = torch.stack([self._tiers[layer_idx][h_i].lex_k_idx for h_i in lex_heads], dim=1).contiguous()
        v_idx_4d = torch.stack([self._tiers[layer_idx][h_i].lex_v_idx for h_i in lex_heads], dim=1).contiguous()
        k_raw_4d = torch.stack(
            [torch.Tensor._make_subclass(torch.Tensor, self._tiers[layer_idx][h_i].lex_k_val) for h_i in lex_heads],
            dim=1,
        ).contiguous()
        v_raw_4d = torch.stack(
            [torch.Tensor._make_subclass(torch.Tensor, self._tiers[layer_idx][h_i].lex_v_val) for h_i in lex_heads],
            dim=1,
        ).contiguous()
        k_sc_4d = torch.stack([self._tiers[layer_idx][h_i].lex_k_val.scale for h_i in lex_heads], dim=1).contiguous()
        v_sc_4d = torch.stack([self._tiers[layer_idx][h_i].lex_v_val.scale for h_i in lex_heads], dim=1).contiguous()
        for pos_i, h_i in enumerate(lex_heads):
            st_i = self._tiers[layer_idx][h_i]
            st_i.lex_k_idx = k_idx_4d[:, pos_i]
            st_i.lex_v_idx = v_idx_4d[:, pos_i]
            st_i.lex_k_val = QuantizedInt8Values(k_raw_4d[:, pos_i], k_sc_4d[:, pos_i])
            st_i.lex_v_val = QuantizedInt8Values(v_raw_4d[:, pos_i], v_sc_4d[:, pos_i])
        pack = (k_idx_4d, k_raw_4d, k_sc_4d, v_idx_4d, v_raw_4d, v_sc_4d)
        self._layer_lex_pack[layer_idx] = pack
        return pack

    @torch.no_grad()
    def _materialize_layer_kv(
        self,
        layer_idx: int,
        dtype: torch.dtype,
        device: torch.device,
        q_len: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Materializes the rectangular `[batch, num_kv_heads, kv_len, head_dim]` tensor for
        SDPA / FlashAttention and returns `(K_out, V_out, bytes_read_this_layer)`.
        """
        local_k = self._local_k[layer_idx]
        local_v = self._local_v[layer_idx]
        if local_k is None or local_v is None:
            empty = torch.empty((1, self.num_kv_heads, 0, self.head_dim), dtype=dtype, device=device)
            return empty, empty, 0

        bsz, n_kv, win_len, d_head = local_k.shape
        total_seq_len = self._layer_seq_len[layer_idx]
        hist_len = max(0, total_seq_len - win_len)
        local_pos = self._local_positions[layer_idx]
        has_atlas = self.atlas_bank is not None
        any_offloaded = self.hikvp_store is not None and any(
            self.hikvp_store.is_offloaded(layer_idx, h_i) for h_i in range(n_kv)
        )
        q_layer: Optional[torch.Tensor] = (
            self._get_pre_rope_query_for_layer(layer_idx, local_k, local_pos)
            if (any_offloaded or has_atlas)
            else None
        )

        # Fast path when all tokens are still inside the n_b = 64 local FP16 sliding window
        if hist_len == 0:
            bytes_read = int(bsz * n_kv * win_len * 2 * d_head * 2)
            k_out, v_out = local_k, local_v
            if has_atlas and q_layer is not None:
                k_atlas, v_atlas, _, atlas_stats = self.atlas_bank.retrieve_and_rope_for_layer(
                    layer_idx=layer_idx,
                    query_pre_rope=q_layer,
                    device=device,
                    dtype=dtype,
                    position_offset=0,
                    use_original_positions=False,
                )
                t_atlas = int(k_atlas.shape[-2])
                if t_atlas > 0:
                    k_out = torch.cat([k_atlas, k_out], dim=-2)
                    v_out = torch.cat([v_atlas, v_out], dim=-2)
                    self._atlas_retrievals += 1
                    self._atlas_tokens_retrieved += t_atlas
                    self._atlas_pcie_bytes += int(atlas_stats.get("pcie_bytes", 0))
                    bytes_read += int(bsz * n_kv * t_atlas * 2 * d_head * 2)
            return k_out, v_out, bytes_read

        all_fp16 = (not any_offloaded) and self._layer_all_fp16[layer_idx]
        if all_fp16:
            ck = self._layer_compact_k[layer_idx]
            cv = self._layer_compact_v[layer_idx]
            if ck is None or cv is None:
                k_heads = [
                    self._tiers[layer_idx][h_i].fp16_k
                    if self._tiers[layer_idx][h_i].fp16_k is not None
                    else torch.empty((bsz, 0, d_head), dtype=dtype, device=device)
                    for h_i in range(n_kv)
                ]
                v_heads = [
                    self._tiers[layer_idx][h_i].fp16_v
                    if self._tiers[layer_idx][h_i].fp16_v is not None
                    else torch.empty((bsz, 0, d_head), dtype=dtype, device=device)
                    for h_i in range(n_kv)
                ]
                ck = torch.stack(k_heads, dim=1)
                cv = torch.stack(v_heads, dim=1)
                self._layer_compact_bytes[layer_idx] = int(bsz * n_kv * int(ck.shape[-2]) * 2 * d_head * 2)
                if not has_atlas and int(ck.shape[-2]) > 0 and win_len <= (self.n_b + self._decode_flush_stride):
                    ck, cv = self._bind_unified_slab(
                        layer_idx, ck, cv, local_k, local_v, is_h2o=False
                    )
                else:
                    # Share underlying storage with per-head tiers so zero duplicate VRAM is retained
                    for h_i in range(n_kv):
                        self._tiers[layer_idx][h_i].fp16_k = ck[:, h_i]
                        self._tiers[layer_idx][h_i].fp16_v = cv[:, h_i]
                    self._layer_compact_k[layer_idx] = ck
                    self._layer_compact_v[layer_idx] = cv
            bytes_read = int(bsz * n_kv * win_len * 2 * d_head * 2) + self._layer_compact_bytes[layer_idx]
            uk = self._unified_k[layer_idx]
            uv = self._unified_v[layer_idx]
            u_hist = self._unified_hist_len[layer_idx]
            cur_lk = self._local_k[layer_idx]
            if (
                not has_atlas
                and uk is not None
                and uv is not None
                and cur_lk is not None
                and cur_lk.untyped_storage().data_ptr() == uk.untyped_storage().data_ptr()
                and cur_lk.storage_offset() == uk.storage_offset() + u_hist * d_head
            ):
                return uk[:, :, : u_hist + win_len, :], uv[:, :, : u_hist + win_len, :], bytes_read
            k_out = torch.cat([ck, local_k], dim=-2)
            v_out = torch.cat([cv, local_v], dim=-2)
            if has_atlas and q_layer is not None:
                k_atlas, v_atlas, _, atlas_stats = self.atlas_bank.retrieve_and_rope_for_layer(
                    layer_idx=layer_idx, query_pre_rope=q_layer, device=device, dtype=dtype,
                    position_offset=0, use_original_positions=False,
                )
                t_atlas = int(k_atlas.shape[-2])
                if t_atlas > 0:
                    k_out = torch.cat([k_atlas, k_out], dim=-2)
                    v_out = torch.cat([v_atlas, v_out], dim=-2)
                    self._atlas_retrievals += 1
                    self._atlas_tokens_retrieved += t_atlas
                    self._atlas_pcie_bytes += int(atlas_stats.get("pcie_bytes", 0))
                    bytes_read += int(bsz * n_kv * t_atlas * 2 * d_head * 2)
            return k_out, v_out, bytes_read

        all_h2o = (not any_offloaded) and self._layer_all_h2o[layer_idx]

        # Fast compact path when all heads in this layer are ROLE_PEAKED_H2O and all
        # incoming query positions lie inside the recent window (q_len <= win_len).
        # Every kept heavy hitter precedes every query position, so SDPA / FlashAttention
        # can attend directly to the compact [B, n_kv, k_kept + win_len, d_head] tensor
        # without padding to hist_len, scattering, or injecting anti_k dummy slots.
        if all_h2o and not self._hf_eager_mode and q_len <= win_len:
            ck = self._layer_compact_k[layer_idx]
            cv = self._layer_compact_v[layer_idx]
            if ck is None or cv is None:
                k_heads = []
                v_heads = []
                hist_bytes = 0
                kept_lens = [
                    int(self._tiers[layer_idx][h_idx].h2o_k.shape[-2])
                    if self._tiers[layer_idx][h_idx].h2o_k is not None
                    else 0
                    for h_idx in range(n_kv)
                ]
                max_kept = max(kept_lens) if kept_lens else 0
                equal_kept = all(kl == max_kept for kl in kept_lens)
                has_sink = (not self._is_hf_model) and (hist_len > max_kept)
                for h_idx in range(n_kv):
                    st = self._tiers[layer_idx][h_idx]
                    k_kept = kept_lens[h_idx]
                    if k_kept > 0 and st.h2o_k is not None and st.h2o_v is not None:
                        hk = st.h2o_k
                        hv = st.h2o_v
                        hist_bytes += int(bsz * k_kept * 2 * d_head * 2)
                    else:
                        hk = torch.empty((bsz, 0, d_head), dtype=dtype, device=device)
                        hv = torch.empty((bsz, 0, d_head), dtype=dtype, device=device)
                    if hk.shape[-2] < max_kept:
                        pad_n = max_kept - int(hk.shape[-2])
                        anti_col = 0 if self._is_hf_model else -1
                        anti_k = (-30.0 * self.U_Q[layer_idx, h_idx, :, anti_col]).to(dtype=dtype, device=device)
                        hk = torch.cat([anti_k.view(1, 1, d_head).expand(bsz, pad_n, d_head), hk], dim=-2)
                        hv = torch.cat([torch.zeros((bsz, pad_n, d_head), dtype=dtype, device=device), hv], dim=-2)
                    if has_sink:
                        n_evicted = max(1, hist_len - k_kept)
                        u0 = self.U_Q[layer_idx, h_idx, :, 0].to(dtype=dtype, device=device)
                        k_sink = (4.05 * math.log(n_evicted) * u0).view(1, 1, d_head).expand(bsz, 1, d_head)
                        v_sink = torch.zeros((bsz, 1, d_head), dtype=dtype, device=device)
                        hk = torch.cat([k_sink, hk], dim=-2)
                        hv = torch.cat([v_sink, hv], dim=-2)
                    k_heads.append(hk)
                    v_heads.append(hv)
                ck = torch.stack(k_heads, dim=1)
                cv = torch.stack(v_heads, dim=1)
                # Share underlying storage when all heads have equal kept length and no synthetic sink
                if equal_kept and not has_sink and max_kept > 0:
                    self._layer_compact_bytes[layer_idx] = hist_bytes
                    if not has_atlas and win_len <= (self.n_b + self._decode_flush_stride):
                        ck, cv = self._bind_unified_slab(
                            layer_idx, ck, cv, local_k, local_v, is_h2o=True
                        )
                    else:
                        for h_idx in range(n_kv):
                            self._tiers[layer_idx][h_idx].h2o_k = ck[:, h_idx]
                            self._tiers[layer_idx][h_idx].h2o_v = cv[:, h_idx]
                        self._layer_compact_k[layer_idx] = ck
                        self._layer_compact_v[layer_idx] = cv
                else:
                    # Do not retain duplicate padded/sink buffers in VRAM if shapes differ
                    bytes_read = int(bsz * n_kv * win_len * 2 * d_head * 2) + hist_bytes
                    k_out = torch.cat([ck, local_k], dim=-2)
                    v_out = torch.cat([cv, local_v], dim=-2)
                    if has_atlas and q_layer is not None:
                        k_atlas, v_atlas, _, atlas_stats = self.atlas_bank.retrieve_and_rope_for_layer(
                            layer_idx=layer_idx, query_pre_rope=q_layer, device=device, dtype=dtype,
                            position_offset=0, use_original_positions=False,
                        )
                        t_atlas = int(k_atlas.shape[-2])
                        if t_atlas > 0:
                            k_out = torch.cat([k_atlas, k_out], dim=-2)
                            v_out = torch.cat([v_atlas, v_out], dim=-2)
                            self._atlas_retrievals += 1
                            self._atlas_tokens_retrieved += t_atlas
                            self._atlas_pcie_bytes += int(atlas_stats.get("pcie_bytes", 0))
                            bytes_read += int(bsz * n_kv * t_atlas * 2 * d_head * 2)
                    return k_out, v_out, bytes_read

            bytes_read = int(bsz * n_kv * win_len * 2 * d_head * 2) + self._layer_compact_bytes[layer_idx]
            uk = self._unified_k[layer_idx]
            uv = self._unified_v[layer_idx]
            u_hist = self._unified_hist_len[layer_idx]
            cur_lk = self._local_k[layer_idx]
            if (
                not has_atlas
                and uk is not None
                and uv is not None
                and cur_lk is not None
                and cur_lk.untyped_storage().data_ptr() == uk.untyped_storage().data_ptr()
                and cur_lk.storage_offset() == uk.storage_offset() + u_hist * d_head
            ):
                return uk[:, :, : u_hist + win_len, :], uv[:, :, : u_hist + win_len, :], bytes_read
            k_out = torch.cat([ck, local_k], dim=-2)
            v_out = torch.cat([cv, local_v], dim=-2)
            if has_atlas and q_layer is not None:
                k_atlas, v_atlas, _, atlas_stats = self.atlas_bank.retrieve_and_rope_for_layer(
                    layer_idx=layer_idx,
                    query_pre_rope=q_layer,
                    device=device,
                    dtype=dtype,
                    position_offset=0,
                    use_original_positions=False,
                )
                t_atlas = int(k_atlas.shape[-2])
                if t_atlas > 0:
                    k_out = torch.cat([k_atlas, k_out], dim=-2)
                    v_out = torch.cat([v_atlas, v_out], dim=-2)
                    self._atlas_retrievals += 1
                    self._atlas_tokens_retrieved += t_atlas
                    self._atlas_pcie_bytes += int(atlas_stats.get("pcie_bytes", 0))
                    bytes_read += int(bsz * n_kv * t_atlas * 2 * d_head * 2)
            return k_out, v_out, bytes_read

        m_atoms = max(1, min(int(self.active_atoms), self.s_budget))

        # Fast single-launch multi-head fused K+V+RoPE+LocalWindow path for pure-Lexico and
        # mixed (Retrieval FP16 + Diffuse Lexico) layers (eliminates per-head loop, cat, and stack).
        if (
            (not any_offloaded)
            and (not self.cache_lexico_reconstructions)
            and self._layer_pure_ret_lex[layer_idx]
        ):
            lex_heads = self._layer_lex_heads[layer_idx]
            ret_heads = self._layer_ret_heads[layer_idx]
            pack = self._get_or_build_lexico_pack(layer_idx, lex_heads)
            if pack is not None:
                k_idx_4d, k_raw_4d, k_sc_4d, v_idx_4d, v_raw_4d, v_sc_4d = pack
                t_lex = int(k_idx_4d.shape[-2])
                ret_ok = all(
                    self._tiers[layer_idx][h_r].fp16_k is not None
                    and int(self._tiers[layer_idx][h_r].fp16_k.shape[-2]) == t_lex
                    for h_r in ret_heads
                )
                if ret_ok:
                    c_half, s_half = None, None
                    if self.dict_key_frame == "pre_rope" and t_lex > 0:
                        self._ensure_rope_table(t_lex, device)
                        c_half, s_half = self._rope_cos_half, self._rope_sin_half
                    if (
                        hasattr(self.codec, "_D_T_fp32")
                        and self.codec._D_T_fp32.device == device
                        and tuple(self.codec._D_T_fp32.shape)
                        == (self.codec.num_layers, self.codec.n_atoms, self.codec.head_dim)
                    ):
                        d_layer_t = self.codec._D_T_fp32[layer_idx]
                    else:
                        d_layer_t = self.codec.D[layer_idx].transpose(0, 1).contiguous().to(
                            device=device, dtype=torch.float32
                        )
                    bytes_read = (
                        int(bsz * n_kv * win_len * 2 * d_head * 2)
                        + int(bsz * len(lex_heads) * t_lex * 2 * (3 * m_atoms + 2))
                        + int(bsz * len(ret_heads) * t_lex * 2 * d_head * 2)
                    )
                    if len(ret_heads) == 0:
                        k_out, v_out = triton_fused_kv_lexico_materialize(
                            k_idx_4d,
                            k_raw_4d,
                            k_sc_4d,
                            v_idx_4d,
                            v_raw_4d,
                            v_sc_4d,
                            d_layer_t,
                            local_k=local_k,
                            local_v=local_v,
                            active_atoms=m_atoms,
                            rope_cos=c_half,
                            rope_sin=s_half,
                            out_dtype=dtype,
                        )
                    else:
                        k_out = torch.empty((bsz, n_kv, t_lex + win_len, d_head), dtype=dtype, device=device)
                        v_out = torch.empty((bsz, n_kv, t_lex + win_len, d_head), dtype=dtype, device=device)
                        for h_r in ret_heads:
                            st_r = self._tiers[layer_idx][h_r]
                            k_out[:, h_r, :t_lex, :].copy_(st_r.fp16_k)
                            v_out[:, h_r, :t_lex, :].copy_(st_r.fp16_v)
                            k_out[:, h_r, t_lex:, :].copy_(local_k[:, h_r])
                            v_out[:, h_r, t_lex:, :].copy_(local_v[:, h_r])
                        is_contig_lex = lex_heads == list(range(lex_heads[0], lex_heads[0] + len(lex_heads)))
                        if is_contig_lex:
                            hs = slice(lex_heads[0], lex_heads[0] + len(lex_heads))
                            triton_fused_kv_lexico_materialize(
                                k_idx_4d,
                                k_raw_4d,
                                k_sc_4d,
                                v_idx_4d,
                                v_raw_4d,
                                v_sc_4d,
                                d_layer_t,
                                local_k=local_k[:, hs],
                                local_v=local_v[:, hs],
                                active_atoms=m_atoms,
                                rope_cos=c_half,
                                rope_sin=s_half,
                                out_k=k_out[:, hs],
                                out_v=v_out[:, hs],
                                out_dtype=dtype,
                            )
                        else:
                            lk_m, lv_m = triton_fused_kv_lexico_materialize(
                                k_idx_4d,
                                k_raw_4d,
                                k_sc_4d,
                                v_idx_4d,
                                v_raw_4d,
                                v_sc_4d,
                                d_layer_t,
                                local_k=local_k[:, lex_heads],
                                local_v=local_v[:, lex_heads],
                                active_atoms=m_atoms,
                                rope_cos=c_half,
                                rope_sin=s_half,
                                out_dtype=dtype,
                            )
                            k_out[:, lex_heads] = lk_m
                            v_out[:, lex_heads] = lv_m
                    if has_atlas and q_layer is not None:
                        k_atlas, v_atlas, _, atlas_stats = self.atlas_bank.retrieve_and_rope_for_layer(
                            layer_idx=layer_idx,
                            query_pre_rope=q_layer,
                            device=device,
                            dtype=dtype,
                            position_offset=0,
                            use_original_positions=False,
                        )
                        t_atlas = int(k_atlas.shape[-2])
                        if t_atlas > 0:
                            k_out = torch.cat([k_atlas, k_out], dim=-2)
                            v_out = torch.cat([v_atlas, v_out], dim=-2)
                            self._atlas_retrievals += 1
                            self._atlas_tokens_retrieved += t_atlas
                            self._atlas_pcie_bytes += int(atlas_stats.get("pcie_bytes", 0))
                            bytes_read += int(bsz * n_kv * t_atlas * 2 * d_head * 2)
                    return k_out, v_out, bytes_read

        k_heads = []
        v_heads = []
        bytes_read = int(bsz * n_kv * win_len * 2 * d_head * 2)

        for h_idx in range(n_kv):
            st = self._tiers[layer_idx][h_idx]
            if self.hikvp_store is not None and self.hikvp_store.is_offloaded(layer_idx, h_idx):
                assert q_layer is not None
                k_ret_pre, v_ret, pos_ret, r_stats = self.hikvp_store.retrieve(
                    layer_idx=layer_idx,
                    head_idx=h_idx,
                    query_pre_rope=q_layer[:, h_idx],
                    device=device,
                    dtype=dtype,
                )
                if self.hikvp_store.draft_mode:
                    self._draft_pcie_bytes += int(r_stats.get("pcie_bytes", 0))
                t_ret = int(k_ret_pre.shape[-2])
                if t_ret > 0:
                    hk = self._rope_rotate(k_ret_pre, pos_ret.to(device=device), inverse=False, force=True)
                    hv = v_ret
                    bytes_read += int(bsz * t_ret * 2 * d_head * 2)
                else:
                    hk = torch.empty((bsz, 0, d_head), dtype=dtype, device=device)
                    hv = torch.empty((bsz, 0, d_head), dtype=dtype, device=device)
                k_heads.append(torch.cat([hk, local_k[:, h_idx]], dim=-2))
                v_heads.append(torch.cat([hv, local_v[:, h_idx]], dim=-2))
                continue

            if st.role == ROLE_RETRIEVAL_FP16:
                hk = st.fp16_k if st.fp16_k is not None else torch.empty((bsz, 0, d_head), dtype=dtype, device=device)
                hv = st.fp16_v if st.fp16_v is not None else torch.empty((bsz, 0, d_head), dtype=dtype, device=device)
                bytes_read += int(bsz * hk.shape[-2] * 2 * d_head * 2)
                k_heads.append(torch.cat([hk, local_k[:, h_idx]], dim=-2))
                v_heads.append(torch.cat([hv, local_v[:, h_idx]], dim=-2))

            elif st.role == ROLE_DIFFUSE_LEXICO:
                if self.cache_lexico_reconstructions:
                    if (
                        st.lex_recon_k is None
                        or st.lex_recon_atoms != m_atoms
                        or st.lex_recon_len != hist_len
                    ):
                        if st.lex_k_idx is not None and st.lex_k_val is not None:
                            st.lex_recon_k = self.codec.decode(
                                st.lex_k_idx,
                                st.lex_k_val,
                                layer_idx=layer_idx,
                                active_atoms=m_atoms,
                                check_bounds=False,
                            ).to(device=device, dtype=dtype)
                            st.lex_recon_k = self._rope_rotate(
                                st.lex_recon_k,
                                None,
                                inverse=False,
                            )
                            st.lex_recon_v = self.codec.decode(
                                st.lex_v_idx,
                                st.lex_v_val,
                                layer_idx=layer_idx,
                                active_atoms=m_atoms,
                                check_bounds=False,
                            ).to(device=device, dtype=dtype)
                            st.lex_recon_atoms = m_atoms
                            st.lex_recon_len = int(st.lex_recon_k.shape[-2])
                        else:
                            st.lex_recon_k = torch.empty((bsz, 0, d_head), dtype=dtype, device=device)
                            st.lex_recon_v = torch.empty((bsz, 0, d_head), dtype=dtype, device=device)
                            st.lex_recon_atoms = m_atoms
                            st.lex_recon_len = 0
                    rk, rv = st.lex_recon_k, st.lex_recon_v
                    recon_len = st.lex_recon_len
                else:
                    if st.lex_k_idx is not None and st.lex_k_val is not None:
                        t_lex = int(st.lex_k_idx.shape[-2])
                        c_half, s_half = None, None
                        if self.dict_key_frame == "pre_rope" and t_lex > 0:
                            self._ensure_rope_table(t_lex, device)
                            c_half, s_half = self._rope_cos_half, self._rope_sin_half
                        rk = self.codec.decode(
                            st.lex_k_idx,
                            st.lex_k_val,
                            layer_idx=layer_idx,
                            active_atoms=m_atoms,
                            check_bounds=False,
                            rope_cos=c_half,
                            rope_sin=s_half,
                        ).to(device=device, dtype=dtype)
                        rv = self.codec.decode(
                            st.lex_v_idx,
                            st.lex_v_val,
                            layer_idx=layer_idx,
                            active_atoms=m_atoms,
                            check_bounds=False,
                        ).to(device=device, dtype=dtype)
                        recon_len = t_lex
                    else:
                        rk = torch.empty((bsz, 0, d_head), dtype=dtype, device=device)
                        rv = torch.empty((bsz, 0, d_head), dtype=dtype, device=device)
                        recon_len = 0

                # Each vector reads `3 * m_atoms + 2` bytes for K and `3 * m_atoms + 2` bytes for V
                bytes_read += int(bsz * recon_len * 2 * (3 * m_atoms + 2))
                k_heads.append(torch.cat([rk, local_k[:, h_idx]], dim=-2))
                v_heads.append(torch.cat([rv, local_v[:, h_idx]], dim=-2))

            elif st.role == ROLE_QUANT_LOWBIT:
                if st.quant_k_packed is not None and st.quant_v_packed is not None:
                    qk = _dequantize_lowbit_2bit(st.quant_k_packed, st.quant_k_min, st.quant_k_scale, dtype=dtype)
                    qv = _dequantize_lowbit_2bit(st.quant_v_packed, st.quant_v_min, st.quant_v_scale, dtype=dtype)
                    q_len_hist = int(qk.shape[-2])
                    bytes_read += int(bsz * q_len_hist * 2 * 44)
                else:
                    qk = torch.empty((bsz, 0, d_head), dtype=dtype, device=device)
                    qv = torch.empty((bsz, 0, d_head), dtype=dtype, device=device)
                k_heads.append(torch.cat([qk, local_k[:, h_idx]], dim=-2))
                v_heads.append(torch.cat([qv, local_v[:, h_idx]], dim=-2))

            elif st.role == ROLE_PEAKED_H2O:
                # Place kept heavy hitters at their valid positions; fill evicted slots with an
                # anti-aligned key vector (-30.0 * U_Q[:, anti_col]) and zero value vector
                anti_col = 0 if self._is_hf_model else -1
                anti_k = (-30.0 * self.U_Q[layer_idx, h_idx, :, anti_col]).to(dtype=dtype, device=device)
                hk_pad = anti_k.view(1, 1, d_head).expand(bsz, hist_len, d_head).clone()
                hv_pad = torch.zeros((bsz, hist_len, d_head), dtype=dtype, device=device)
                if st.h2o_k is not None and st.h2o_indices is not None and st.h2o_k.shape[-2] > 0:
                    k_kept = int(st.h2o_k.shape[-2])
                    idx_clamped = st.h2o_indices.clamp(0, max(0, hist_len - 1)).unsqueeze(-1).expand(-1, -1, d_head)
                    hk_pad.scatter_(dim=-2, index=idx_clamped, src=st.h2o_k)
                    hv_pad.scatter_(dim=-2, index=idx_clamped, src=st.h2o_v)
                    bytes_read += int(bsz * k_kept * 2 * d_head * 2)
                k_heads.append(torch.cat([hk_pad, local_k[:, h_idx]], dim=-2))
                v_heads.append(torch.cat([hv_pad, local_v[:, h_idx]], dim=-2))

        # Align head sequence lengths if offloaded HiKVP heads returned fewer tokens than full hist_len
        head_lens = [int(kh.shape[-2]) for kh in k_heads]
        max_head_len = max(head_lens) if head_lens else 0
        if any(hl != max_head_len for hl in head_lens):
            anti_col = 0 if self._is_hf_model else -1
            for h_idx in range(n_kv):
                cur_len = head_lens[h_idx]
                if cur_len < max_head_len:
                    pad_n = max_head_len - cur_len
                    anti_k = (-30.0 * self.U_Q[layer_idx, h_idx, :, anti_col]).to(dtype=dtype, device=device)
                    k_pad = anti_k.view(1, 1, d_head).expand(bsz, pad_n, d_head)
                    v_pad = torch.zeros((bsz, pad_n, d_head), dtype=dtype, device=device)
                    k_heads[h_idx] = torch.cat([k_pad, k_heads[h_idx]], dim=-2)
                    v_heads[h_idx] = torch.cat([v_pad, v_heads[h_idx]], dim=-2)

        k_out = torch.stack(k_heads, dim=1)
        v_out = torch.stack(v_heads, dim=1)
        if has_atlas and q_layer is not None:
            k_atlas, v_atlas, _, atlas_stats = self.atlas_bank.retrieve_and_rope_for_layer(
                layer_idx=layer_idx,
                query_pre_rope=q_layer,
                device=device,
                dtype=dtype,
                position_offset=0,
                use_original_positions=False,
            )
            t_atlas = int(k_atlas.shape[-2])
            if t_atlas > 0:
                k_out = torch.cat([k_atlas, k_out], dim=-2)
                v_out = torch.cat([v_atlas, v_out], dim=-2)
                self._atlas_retrievals += 1
                self._atlas_tokens_retrieved += t_atlas
                self._atlas_pcie_bytes += int(atlas_stats.get("pcie_bytes", 0))
                bytes_read += int(bsz * n_kv * t_atlas * 2 * d_head * 2)
        return k_out, v_out, bytes_read

    @torch.no_grad()
    def _refresh_telemetry(self) -> None:
        """Recomputes exact physical byte telemetry and per-tier token counts across all layers."""
        total_seq = self.get_seq_length(0)
        if total_seq == 0 or self._local_k[0] is None:
            self.fp16_bytes = 0
            self.compressed_bytes = 0
            self.physical_resident_bytes = 0
            self.raw_tensor_bytes = 0
            self.raw_fp16_tensor_bytes = 0
            self.bytes_read_last_step = 0
            for k in self.per_tier_token_counts:
                self.per_tier_token_counts[k] = 0
            return

        bsz = int(self._local_k[0].shape[0])
        elem_sz = int(self._local_k[0].element_size())
        vec_fp16_bytes = 2 * self.head_dim * 2  # K and V in FP16: 2 * d_head * 2 bytes
        self.fp16_bytes = int(bsz * self.num_layers * self.num_kv_heads * total_seq * vec_fp16_bytes)
        self.raw_fp16_tensor_bytes = int(
            bsz * self.num_layers * self.num_kv_heads * total_seq * 2 * self.head_dim * elem_sz
        )

        comp_bytes = 0
        phys_fp16_equiv_bytes = 0
        raw_bytes = 0
        cnt_local = 0
        cnt_ret = 0
        cnt_h2o = 0
        cnt_lex = 0
        cnt_evicted = 0

        for l_idx in range(self.num_layers):
            lk = self._local_k[l_idx]
            lv = self._local_v[l_idx]
            if lk is None or lv is None:
                continue
            win_len = int(lk.shape[-2])
            hist_len = max(0, self._layer_seq_len[l_idx] - win_len)

            # Local FP16 window bytes across all KV heads
            local_b = int(bsz * self.num_kv_heads * win_len * vec_fp16_bytes)
            comp_bytes += local_b
            phys_fp16_equiv_bytes += local_b
            raw_bytes += int(lk.numel() * lk.element_size() + lv.numel() * lv.element_size())
            cnt_local += int(bsz * self.num_kv_heads * win_len)

            for h_idx in range(self.num_kv_heads):
                st = self._tiers[l_idx][h_idx]
                if self.hikvp_store is not None and self.hikvp_store.is_offloaded(l_idx, h_idx):
                    h_index = self.hikvp_store.get_head(l_idx, h_idx)
                    if h_index is not None:
                        b_gpu = int(h_index.gpu_index_bytes() + h_index.gpu_lru_bytes())
                        comp_bytes += b_gpu
                        phys_fp16_equiv_bytes += b_gpu
                        raw_bytes += b_gpu
                        cnt_ret += int(bsz * h_index.num_tokens)
                    continue
                if st.role == ROLE_RETRIEVAL_FP16:
                    n_t = int(st.fp16_k.shape[-2]) if st.fp16_k is not None else 0
                    b_tier = int(bsz * n_t * vec_fp16_bytes)
                    comp_bytes += b_tier
                    phys_fp16_equiv_bytes += b_tier
                    if st.fp16_k is not None and st.fp16_v is not None:
                        raw_bytes += int(
                            st.fp16_k.numel() * st.fp16_k.element_size()
                            + st.fp16_v.numel() * st.fp16_v.element_size()
                        )
                    cnt_ret += int(bsz * n_t)
                elif st.role == ROLE_PEAKED_H2O:
                    n_t = int(st.h2o_k.shape[-2]) if st.h2o_k is not None else 0
                    b_tier = int(bsz * n_t * vec_fp16_bytes)
                    comp_bytes += b_tier
                    phys_fp16_equiv_bytes += b_tier
                    if st.h2o_k is not None and st.h2o_v is not None:
                        raw_bytes += int(
                            st.h2o_k.numel() * st.h2o_k.element_size()
                            + st.h2o_v.numel() * st.h2o_v.element_size()
                        )
                    cnt_h2o += int(bsz * n_t)
                    cnt_evicted += int(bsz * max(0, hist_len - n_t))
                elif st.role == ROLE_DIFFUSE_LEXICO:
                    n_t = int(st.lex_k_idx.shape[-2]) if st.lex_k_idx is not None else 0
                    # Each K and V vector in Lexico format uses (3 * s_budget + 2) bytes
                    b_tier = int(bsz * n_t * 2 * (3 * self.s_budget + 2))
                    comp_bytes += b_tier
                    phys_fp16_equiv_bytes += b_tier
                    if st.lex_k_idx is not None and st.lex_k_val is not None:
                        raw_k = torch.Tensor._make_subclass(torch.Tensor, st.lex_k_val)
                        raw_v = torch.Tensor._make_subclass(torch.Tensor, st.lex_v_val)
                        raw_bytes += int(
                            st.lex_k_idx.numel() * st.lex_k_idx.element_size()
                            + st.lex_v_idx.numel() * st.lex_v_idx.element_size()
                            + raw_k.numel() * raw_k.element_size()
                            + raw_v.numel() * raw_v.element_size()
                            + st.lex_k_val.scale.numel() * st.lex_k_val.scale.element_size()
                            + st.lex_v_val.scale.numel() * st.lex_v_val.scale.element_size()
                        )
                    if st.lex_recon_k is not None and st.lex_recon_v is not None:
                        phys_fp16_equiv_bytes += int(
                            st.lex_recon_k.numel() * 2 + st.lex_recon_v.numel() * 2
                        )
                        raw_bytes += int(
                            st.lex_recon_k.numel() * st.lex_recon_k.element_size()
                            + st.lex_recon_v.numel() * st.lex_recon_v.element_size()
                        )
                    cnt_lex += int(bsz * n_t)
                elif st.role == ROLE_QUANT_LOWBIT:
                    n_t = int(st.quant_k_packed.shape[-2]) if st.quant_k_packed is not None else 0
                    # Each K and V vector in 2-bit asymmetric group format uses 44 bytes (88 B total)
                    b_tier = int(bsz * n_t * 2 * 44)
                    comp_bytes += b_tier
                    phys_fp16_equiv_bytes += b_tier
                    if st.quant_k_packed is not None and st.quant_v_packed is not None:
                        raw_bytes += int(
                            st.quant_k_packed.numel() * st.quant_k_packed.element_size()
                            + st.quant_k_min.numel() * st.quant_k_min.element_size()
                            + st.quant_k_scale.numel() * st.quant_k_scale.element_size()
                            + st.quant_v_packed.numel() * st.quant_v_packed.element_size()
                            + st.quant_v_min.numel() * st.quant_v_min.element_size()
                            + st.quant_v_scale.numel() * st.quant_v_scale.element_size()
                        )
                    cnt_lex += int(bsz * n_t)

        # Include any duplicate buffers if present in key_cache / value_cache / layers
        for t in self.key_cache:
            if isinstance(t, torch.Tensor) and t.numel() > 0:
                phys_fp16_equiv_bytes += int(t.numel() * 2)
                raw_bytes += int(t.numel() * t.element_size())
        for t in self.value_cache:
            if isinstance(t, torch.Tensor) and t.numel() > 0:
                phys_fp16_equiv_bytes += int(t.numel() * 2)
                raw_bytes += int(t.numel() * t.element_size())
        for layer_obj in getattr(self, "layers", []):
            k_t = getattr(layer_obj, "keys", None)
            v_t = getattr(layer_obj, "values", None)
            if isinstance(k_t, torch.Tensor) and k_t.numel() > 0:
                phys_fp16_equiv_bytes += int(k_t.numel() * 2)
                raw_bytes += int(k_t.numel() * k_t.element_size())
            if isinstance(v_t, torch.Tensor) and v_t.numel() > 0:
                phys_fp16_equiv_bytes += int(v_t.numel() * 2)
                raw_bytes += int(v_t.numel() * v_t.element_size())

        # Shared Lexico model overhead (NOT included in `compressed_bytes`; reported separately).
        # D [L, d, N] is read at decode time and W_ridge [L, N, d] at encode time. Both are
        # model-level constants shared by every sequence/batch, so they amortize with
        # concurrent requests, but for a single short sequence they are a real resident cost.
        if cnt_lex > 0:
            lex_layers = sorted(
                {
                    l
                    for l in range(self.num_layers)
                    for h in range(self.num_kv_heads)
                    if self._tiers[l][h].role == ROLE_DIFFUSE_LEXICO
                }
            )
            per_layer = 0
            if self.codec is not None and hasattr(self.codec, "D"):
                d_t = self.codec.D
                per_layer += int(d_t[0].numel() * d_t.element_size())
            if self.codec is not None and hasattr(self.codec, "W_ridge"):
                w_t = self.codec.W_ridge
                per_layer += int(w_t[0].numel() * w_t.element_size())
            self.dictionary_bytes = int(per_layer * len(lex_layers))
        else:
            self.dictionary_bytes = 0

        self.compressed_bytes = comp_bytes
        self.physical_resident_bytes = phys_fp16_equiv_bytes
        self.raw_tensor_bytes = raw_bytes
        self.bytes_read_last_step = int(sum(self._layer_bytes_read))
        self.per_tier_token_counts = {
            "local_window_fp16": cnt_local,
            "retrieval_fp16": cnt_ret,
            "peaked_h2o_fp16": cnt_h2o,
            "diffuse_lexico_sparse": cnt_lex,
            "evicted_h2o": cnt_evicted,
        }

    @torch.no_grad()
    def _materialize_pending_slab_scores(self, layer_idx: int) -> None:
        """
        Evaluates write-time importance scores in one batched GPU kernel call for any staged
        1-token decode entries `[scored_w : cur_w]` before decay, overflow flush, or `crop()`.
        Reduces `64 * L` single-token kernel launches to `L` batched Tensor Core `tl.dot` launches.
        """
        lk = self._local_k[layer_idx]
        if lk is None:
            self._slab_scored_len[layer_idx] = 0
            return
        cur_w = int(lk.shape[-2])
        scored_w = self._slab_scored_len[layer_idx]
        if scored_w >= cur_w:
            return
        bsc = self._local_buf_sc[layer_idx]
        lv = self._local_v[layer_idx]
        if bsc is None or lv is None:
            self._slab_scored_len[layer_idx] = cur_w
            return
        if not self._layer_has_h2o[layer_idx]:
            bsc[:, :, scored_w:cur_w].zero_()
            self._local_scores[layer_idx] = bsc[:, :, :cur_w]
            self._slab_scored_len[layer_idx] = cur_w
            return
        if self._u_q_fp32.device != lk.device:
            self._u_q_fp32 = self._u_q_fp32.to(device=lk.device)
        s_pos = max(0, self._layer_seq_len[layer_idx] - (cur_w - scored_w))
        scores_chunk = triton_fused_write_probe(
            lk[:, :, scored_w:cur_w, :],
            lv[:, :, scored_w:cur_w, :],
            self._u_q_fp32[layer_idx],
            start_pos=s_pos,
            sink_tokens=self.sink_tokens,
            sink_bonus=self.sink_bonus,
            softplus_mode=self._is_hf_model,
        )
        bsc[:, :, scored_w:cur_w].copy_(scores_chunk)
        self._local_scores[layer_idx] = bsc[:, :, :cur_w]
        self._slab_scored_len[layer_idx] = cur_w

    @torch.no_grad()
    def _ensure_local_slab(
        self, layer_idx: int, bsz: int, n_kv: int, d_head: int, dtype: torch.dtype, device: torch.device
    ) -> None:
        cap = self.n_b + self._decode_flush_stride
        bk = self._local_buf_k[layer_idx]
        if bk is None or bk.shape[0] != bsz or bk.shape[2] < cap or bk.dtype != dtype or bk.device != device or bk.is_inference():
            self._materialize_pending_slab_scores(layer_idx)
            with torch.inference_mode(False):
                new_bk = torch.empty((bsz, n_kv, cap, d_head), dtype=dtype, device=device)
                new_bv = torch.empty((bsz, n_kv, cap, d_head), dtype=dtype, device=device)
                new_bsc = torch.empty((bsz, n_kv, cap), dtype=torch.float32, device=device)
                new_bpos = torch.empty((cap,), dtype=torch.long, device=device)
            lk = self._local_k[layer_idx]
            if lk is not None and int(lk.shape[-2]) > 0 and int(lk.shape[-2]) <= cap and lk.shape[0] == bsz:
                w = int(lk.shape[-2])
                new_bk[:, :, :w, :].copy_(lk)
                new_bv[:, :, :w, :].copy_(self._local_v[layer_idx])
                new_bsc[:, :, :w].copy_(self._local_scores[layer_idx])
                new_bpos[:w].copy_(self._local_positions[layer_idx])
                self._local_k[layer_idx] = new_bk[:, :, :w, :]
                self._local_v[layer_idx] = new_bv[:, :, :w, :]
                self._local_scores[layer_idx] = new_bsc[:, :, :w]
                self._local_positions[layer_idx] = new_bpos[:w]
                self._slab_scored_len[layer_idx] = w
            self._local_buf_k[layer_idx] = new_bk
            self._local_buf_v[layer_idx] = new_bv
            self._local_buf_sc[layer_idx] = new_bsc
            self._local_buf_pos[layer_idx] = new_bpos

    @torch.no_grad()
    def _flush_all_pending_overflow(self) -> None:
        """Flushes any staged decode tokens above `n_b` down to exact `n_b` across all layers."""
        for l_idx in range(self.num_layers):
            lk = self._local_k[l_idx]
            if lk is None or int(lk.shape[-2]) <= self.n_b:
                continue
            self._materialize_pending_slab_scores(l_idx)
            w = int(lk.shape[-2])
            n_over = w - self.n_b
            lv = self._local_v[l_idx]
            lsc = self._local_scores[l_idx]
            lpos = self._local_positions[l_idx]
            over_k = lk[:, :, :n_over, :].clone()
            over_v = lv[:, :, :n_over, :].clone()
            over_sc = lsc[:, :, :n_over].clone()
            over_pos = lpos[:n_over].clone()

            bk = self._local_buf_k[l_idx]
            if bk is not None and bk.shape[0] == lk.shape[0]:
                bv = self._local_buf_v[l_idx]
                bsc = self._local_buf_sc[l_idx]
                bpos = self._local_buf_pos[l_idx]
                rem_k = lk[:, :, n_over:, :].clone()
                rem_v = lv[:, :, n_over:, :].clone()
                rem_sc = lsc[:, :, n_over:].clone()
                rem_pos = lpos[n_over:].clone()
                bk[:, :, : self.n_b, :].copy_(rem_k)
                bv[:, :, : self.n_b, :].copy_(rem_v)
                bsc[:, :, : self.n_b].copy_(rem_sc)
                bpos[: self.n_b].copy_(rem_pos)
                self._local_k[l_idx] = bk[:, :, : self.n_b, :]
                self._local_v[l_idx] = bv[:, :, : self.n_b, :]
                self._local_scores[l_idx] = bsc[:, :, : self.n_b]
                self._local_positions[l_idx] = bpos[: self.n_b]
                self._slab_scored_len[l_idx] = self.n_b
            else:
                self._local_k[l_idx] = lk[:, :, n_over:, :]
                self._local_v[l_idx] = lv[:, :, n_over:, :]
                self._local_scores[l_idx] = lsc[:, :, n_over:]
                self._local_positions[l_idx] = lpos[n_over:]
                self._slab_scored_len[l_idx] = self.n_b

            self._route_overflow_to_tiers(
                layer_idx=l_idx,
                overflow_k=over_k,
                overflow_v=over_v,
                overflow_scores=over_sc,
                overflow_pos=over_pos,
                beta_t=float(self.controller.last_beta_t),
                keep_ratio=self.keep_ratio,
                s_budget=self.s_budget,
                total_seq_len=self._layer_seq_len[l_idx],
            )

    @torch.no_grad()
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the hybrid StoikovKV cache for `layer_idx` with incoming `key_states` and
        `value_states` (`[batch, num_kv_heads, q_len, head_dim]`) and returns materialized
        `(key_states_full, value_states_full)` for SDPA / FlashAttention.
        """
        bsz, n_kv, q_len, d_head = key_states.shape
        start_pos = self._layer_seq_len[layer_idx]
        new_seq_len = start_pos + q_len
        self._layer_seq_len[layer_idx] = new_seq_len
        if layer_idx == 0:
            self._seen_tokens = new_seq_len

        # 1. Fast deferred batch-scoring check for 1-token slab decode:
        # On layers that do not drive the Layer-0 HMM controller, staging K/V in the slab and
        # computing write-time scores in ONE 64-token Tensor Core `tl.dot` call upon flush/decay/crop
        # produces identical scores while cutting 64 * L kernel launches down to L.
        cap = self.n_b + self._decode_flush_stride
        cur_lk = self._local_k[layer_idx]
        cur_w = int(cur_lk.shape[-2]) if cur_lk is not None else 0
        layer0_stride_due = (
            layer_idx == 0
            and self.controller.enable_hmm_decay
            and ((new_seq_len & 15) == 0 or (cur_w + 1) >= cap)
        )
        defer_slab_score = (
            q_len == 1
            and cur_w + 1 <= cap
            and self.policy_mode != "snapkv"
            and self.controller_fn is None
            and (not layer0_stride_due)
            and can_use_triton(key_states)
        )
        if defer_slab_score:
            new_scores = None
            if layer_idx == 0:
                beta_t = self.controller.beta_stationary if self.controller.enable_hmm_decay else 1.0
                self.controller.last_beta_t = float(beta_t)
                keep_ratio = self.controller.compute_dynamic_keep_ratio(new_seq_len)
                s_budget = self.s_budget
            else:
                beta_t = float(self.controller.last_beta_t)
                keep_ratio = float(
                    self.controller.last_keep_ratio
                    if self.controller.enable_stoikov_control
                    else self.keep_ratio
                )
                s_budget = self.s_budget
        else:
            new_scores, beta_t, keep_ratio, s_budget = self._compute_step_scores_and_controller(
                key_states, value_states, layer_idx=layer_idx, start_pos=start_pos
            )

        # Decay existing H2O scores once per update step (1.0 when stationary, beta_shift on regime shift)
        decay_factor = float(beta_t)
        if decay_factor < 1.0:
            for h_idx in range(n_kv):
                st = self._tiers[layer_idx][h_idx]
                if st.role == ROLE_PEAKED_H2O and st.h2o_scores is not None:
                    st.h2o_scores = st.h2o_scores * decay_factor

        # 2. Fast in-place slab path for single-token decode (zero torch.cat, batched tier flush)
        if q_len == 1 and cur_w + 1 <= cap:
            self._ensure_local_slab(layer_idx, bsz, n_kv, d_head, key_states.dtype, key_states.device)
            bk = self._local_buf_k[layer_idx]
            bv = self._local_buf_v[layer_idx]
            bsc = self._local_buf_sc[layer_idx]
            bpos = self._local_buf_pos[layer_idx]
            if decay_factor < 1.0 and cur_w > 0:
                self._materialize_pending_slab_scores(layer_idx)
                bsc[:, :, :cur_w].mul_(decay_factor)
            bk[:, :, cur_w : cur_w + 1, :].copy_(key_states)
            bv[:, :, cur_w : cur_w + 1, :].copy_(value_states)
            if new_scores is not None:
                if self._slab_scored_len[layer_idx] < cur_w:
                    self._materialize_pending_slab_scores(layer_idx)
                bsc[:, :, cur_w : cur_w + 1].copy_(new_scores)
                self._slab_scored_len[layer_idx] = cur_w + 1
            if self._slab_next_pos[layer_idx] == start_pos:
                self._slab_next_pos[layer_idx] = new_seq_len
            else:
                rem_slots = cap - cur_w
                bpos[cur_w:cap] = torch.arange(
                    start_pos, start_pos + rem_slots, device=key_states.device, dtype=torch.long
                )
                self._slab_next_pos[layer_idx] = new_seq_len
            total_w = cur_w + 1

            if total_w >= cap:
                self._local_k[layer_idx] = bk[:, :, :total_w, :]
                self._local_v[layer_idx] = bv[:, :, :total_w, :]
                self._local_positions[layer_idx] = bpos[:total_w]
                self._materialize_pending_slab_scores(layer_idx)
                n_over = total_w - self.n_b
                over_k = bk[:, :, :n_over, :].clone()
                over_v = bv[:, :, :n_over, :].clone()
                over_sc = bsc[:, :, :n_over].clone()
                over_pos = bpos[:n_over].clone()
                rem_k = bk[:, :, n_over:total_w, :].clone()
                rem_v = bv[:, :, n_over:total_w, :].clone()
                rem_sc = bsc[:, :, n_over:total_w].clone()
                bk[:, :, : self.n_b, :].copy_(rem_k)
                bv[:, :, : self.n_b, :].copy_(rem_v)
                bsc[:, :, : self.n_b].copy_(rem_sc)
                bpos[:cap] = torch.arange(
                    new_seq_len - self.n_b,
                    new_seq_len - self.n_b + cap,
                    device=key_states.device,
                    dtype=torch.long,
                )
                self._slab_next_pos[layer_idx] = new_seq_len
                self._slab_scored_len[layer_idx] = self.n_b
                self._local_k[layer_idx] = bk[:, :, : self.n_b, :]
                self._local_v[layer_idx] = bv[:, :, : self.n_b, :]
                self._local_scores[layer_idx] = bsc[:, :, : self.n_b]
                self._local_positions[layer_idx] = bpos[: self.n_b]
                self._route_overflow_to_tiers(
                    layer_idx=layer_idx,
                    overflow_k=over_k,
                    overflow_v=over_v,
                    overflow_scores=over_sc,
                    overflow_pos=over_pos,
                    beta_t=beta_t,
                    keep_ratio=keep_ratio,
                    s_budget=s_budget,
                    total_seq_len=new_seq_len,
                )
            else:
                self._local_k[layer_idx] = bk[:, :, :total_w, :]
                self._local_v[layer_idx] = bv[:, :, :total_w, :]
                self._local_scores[layer_idx] = bsc[:, :, :total_w]
                self._local_positions[layer_idx] = bpos[:total_w]
        else:
            if cur_lk is not None:
                # Note: self._layer_seq_len[layer_idx] was already incremented by q_len above,
                # so temporarily pass prior sequence length for any deferred 1-token scores.
                self._layer_seq_len[layer_idx] = start_pos
                self._materialize_pending_slab_scores(layer_idx)
                self._layer_seq_len[layer_idx] = new_seq_len
            new_pos = torch.arange(start_pos, new_seq_len, device=key_states.device, dtype=torch.long)
            # Multi-token prefill / verification path: flush immediately to exact n_b
            if cur_lk is None:
                cat_k = key_states
                cat_v = value_states
                cat_sc = new_scores
                cat_pos = new_pos
            else:
                prev_sc = self._local_scores[layer_idx] * decay_factor if decay_factor < 1.0 else self._local_scores[layer_idx]
                cat_k = torch.cat([cur_lk, key_states], dim=-2)
                cat_v = torch.cat([self._local_v[layer_idx], value_states], dim=-2)
                cat_sc = torch.cat([prev_sc, new_scores], dim=-1)
                cat_pos = torch.cat([self._local_positions[layer_idx], new_pos], dim=0)

            total_in_buf = int(cat_k.shape[-2])
            if total_in_buf > self.n_b:
                n_over = total_in_buf - self.n_b
                overflow_k = cat_k[:, :, :n_over, :]
                overflow_v = cat_v[:, :, :n_over, :]
                overflow_sc = cat_sc[:, :, :n_over]
                overflow_pos = cat_pos[:n_over]
                keep_k = cat_k[:, :, n_over:, :]
                keep_v = cat_v[:, :, n_over:, :]
                keep_sc = cat_sc[:, :, n_over:]
                keep_pos = cat_pos[n_over:]
                self._route_overflow_to_tiers(
                    layer_idx=layer_idx,
                    overflow_k=overflow_k,
                    overflow_v=overflow_v,
                    overflow_scores=overflow_sc,
                    overflow_pos=overflow_pos,
                    beta_t=beta_t,
                    keep_ratio=keep_ratio,
                    s_budget=s_budget,
                    total_seq_len=new_seq_len,
                )
            else:
                keep_k, keep_v, keep_sc, keep_pos = cat_k, cat_v, cat_sc, cat_pos

            # Seed pre-allocated decode slab with the retained <= n_b local window
            self._local_k[layer_idx] = None
            self._ensure_local_slab(layer_idx, bsz, n_kv, d_head, key_states.dtype, key_states.device)
            w_keep = int(keep_k.shape[-2])
            self._local_buf_k[layer_idx][:, :, :w_keep, :].copy_(keep_k)
            self._local_buf_v[layer_idx][:, :, :w_keep, :].copy_(keep_v)
            self._local_buf_sc[layer_idx][:, :, :w_keep].copy_(keep_sc)
            self._local_buf_pos[layer_idx][:w_keep].copy_(keep_pos)
            if w_keep < cap:
                self._local_buf_pos[layer_idx][w_keep:cap] = torch.arange(
                    new_seq_len,
                    new_seq_len + (cap - w_keep),
                    device=key_states.device,
                    dtype=torch.long,
                )
            self._slab_next_pos[layer_idx] = new_seq_len
            self._slab_scored_len[layer_idx] = w_keep
            self._local_k[layer_idx] = self._local_buf_k[layer_idx][:, :, :w_keep, :]
            self._local_v[layer_idx] = self._local_buf_v[layer_idx][:, :, :w_keep, :]
            self._local_scores[layer_idx] = self._local_buf_sc[layer_idx][:, :, :w_keep]
            self._local_positions[layer_idx] = self._local_buf_pos[layer_idx][:w_keep]

        # 4. Materialize transient rectangular KV states for this layer's SDPA / FlashAttention call
        # without retaining full-sequence copies in self.key_cache, self.value_cache, or DynamicCache.layers.
        # During the initial bulk prefill (start_pos == 0 and q_len > 1), use the exact incoming
        # (key_states, value_states) for intra-prompt hidden-state propagation across layers while
        # storing only the compressed/evicted tiers in self._tiers for all subsequent decode/verify steps.
        if self._is_hf_model and self.atlas_bank is None and start_pos == 0 and q_len > 1:
            win_len = int(self._local_k[layer_idx].shape[-2])
            bytes_read = int(bsz * n_kv * win_len * 2 * d_head * 2)
            m_atoms = max(1, min(int(self.active_atoms), self.s_budget))
            for h_idx in range(n_kv):
                st = self._tiers[layer_idx][h_idx]
                if st.role == ROLE_RETRIEVAL_FP16 and st.fp16_k is not None:
                    bytes_read += int(bsz * int(st.fp16_k.shape[-2]) * 2 * d_head * 2)
                elif st.role == ROLE_PEAKED_H2O and st.h2o_k is not None:
                    bytes_read += int(bsz * int(st.h2o_k.shape[-2]) * 2 * d_head * 2)
                elif st.role == ROLE_DIFFUSE_LEXICO and st.lex_k_idx is not None:
                    bytes_read += int(bsz * int(st.lex_k_idx.shape[-2]) * 2 * (3 * m_atoms + 2))
                elif st.role == ROLE_QUANT_LOWBIT and st.quant_k_packed is not None:
                    bytes_read += int(bsz * int(st.quant_k_packed.shape[-2]) * 2 * 44)
            self._layer_bytes_read[layer_idx] = bytes_read
            self.cumulative_bytes_read += bytes_read
            if layer_idx == self.num_layers - 1:
                self._refresh_telemetry()
            return key_states, value_states

        k_out, v_out, bytes_read = self._materialize_layer_kv(
            layer_idx=layer_idx, dtype=key_states.dtype, device=key_states.device, q_len=q_len
        )

        # During chunked prefill (start_pos > 0 and q_len > n_b), preserve the full incoming chunk
        # uncompressed at the tail of (k_out, v_out) so intra-chunk causal attention is exact.
        if start_pos > 0 and q_len > self.n_b:
            win_len_cur = int(self._local_k[layer_idx].shape[-2]) if self._local_k[layer_idx] is not None else self.n_b
            trim_tail = q_len if int(k_out.shape[-2]) >= q_len else min(win_len_cur, int(k_out.shape[-2]))
            k_out = torch.cat([k_out[:, :, :-trim_tail, :], key_states], dim=-2)
            v_out = torch.cat([v_out[:, :, :-trim_tail, :], value_states], dim=-2)

        self._layer_bytes_read[layer_idx] = bytes_read
        self.cumulative_bytes_read += bytes_read
        if layer_idx == self.num_layers - 1:
            self.bytes_read_last_step = int(sum(self._layer_bytes_read))
            if q_len > 1:
                self._refresh_telemetry()

        return k_out, v_out

    @torch.no_grad()
    def crop(self, max_length: int) -> None:
        """
        Rolls back the KV cache to `max_length` tokens across all layers and tiers.
        Used by Matryoshka speculative decoding to discard rejected draft tokens cleanly.
        """
        cur_len = self.get_seq_length(0)
        if max_length < 0:
            max_length = cur_len - abs(max_length)
        max_length = max(0, int(max_length))
        if cur_len <= max_length:
            return

        self._seen_tokens = max_length
        if max_length == 0 and self.hikvp_store is not None:
            self.hikvp_store.crop(0)
        for l_idx in range(self.num_layers):
            l_cur = self._layer_seq_len[l_idx]
            if l_cur <= max_length:
                continue
            if max_length > 0:
                self._materialize_pending_slab_scores(l_idx)
            self._layer_seq_len[l_idx] = max_length
            self._local_buf_k[l_idx] = None
            self._local_buf_v[l_idx] = None
            self._local_buf_sc[l_idx] = None
            self._local_buf_pos[l_idx] = None
            self._slab_next_pos[l_idx] = -1
            self._slab_scored_len[l_idx] = 0
            self._layer_compact_k[l_idx] = None
            self._layer_compact_v[l_idx] = None
            self._unified_k[l_idx] = None
            self._unified_v[l_idx] = None
            self._layer_lex_pack[l_idx] = None

            if max_length == 0:
                self._local_k[l_idx] = None
                self._local_v[l_idx] = None
                self._local_scores[l_idx] = None
                self._local_positions[l_idx] = None
                self._flushed_ring_k[l_idx] = None
                self._flushed_ring_v[l_idx] = None
                self._flushed_ring_scores[l_idx] = None
                self._flushed_ring_pos[l_idx] = None
                for h_idx in range(self.num_kv_heads):
                    self._tiers[l_idx][h_idx] = _HeadTierState(
                        role=int(self.head_roles[l_idx, h_idx].item())
                    )
                continue

            target_win_len = min(self.n_b, max_length)
            target_hist_len = max(0, max_length - target_win_len)

            # Reconstruct the exact local FP16 window (preserving any unflushed decode staging tokens)
            lk = self._local_k[l_idx]
            lv = self._local_v[l_idx]
            lsc = self._local_scores[l_idx]
            lpos = self._local_positions[l_idx]

            if lk is not None and lpos is not None:
                cur_hist_floor = int(lpos[0].item()) if lpos.numel() > 0 else target_hist_len
                eff_hist_len = min(cur_hist_floor, target_hist_len)
                # Combine ring buffer of recently flushed tokens with current local window
                if self._flushed_ring_k[l_idx] is not None and self._flushed_ring_pos[l_idx] is not None:
                    comb_k = torch.cat([self._flushed_ring_k[l_idx], lk], dim=-2)
                    comb_v = torch.cat([self._flushed_ring_v[l_idx], lv], dim=-2)
                    comb_sc = torch.cat([self._flushed_ring_scores[l_idx], lsc], dim=-1)
                    comb_pos = torch.cat([self._flushed_ring_pos[l_idx], lpos], dim=0)
                else:
                    comb_k, comb_v, comb_sc, comb_pos = lk, lv, lsc, lpos

                valid_mask = (comb_pos >= eff_hist_len) & (comb_pos < max_length)
                self._local_k[l_idx] = comb_k[:, :, valid_mask, :]
                self._local_v[l_idx] = comb_v[:, :, valid_mask, :]
                self._local_scores[l_idx] = comb_sc[:, :, valid_mask]
                self._local_positions[l_idx] = comb_pos[valid_mask]

                # Trim flushed ring buffer to `< eff_hist_len`
                ring_mask = comb_pos < eff_hist_len
                if torch.any(ring_mask):
                    self._flushed_ring_k[l_idx] = comb_k[:, :, ring_mask, :][:, :, -self.max_rollback_window :, :]
                    self._flushed_ring_v[l_idx] = comb_v[:, :, ring_mask, :][:, :, -self.max_rollback_window :, :]
                    self._flushed_ring_scores[l_idx] = comb_sc[:, :, ring_mask][:, :, -self.max_rollback_window :]
                    self._flushed_ring_pos[l_idx] = comb_pos[ring_mask][-self.max_rollback_window :]
                else:
                    self._flushed_ring_k[l_idx] = None
                    self._flushed_ring_v[l_idx] = None
                    self._flushed_ring_scores[l_idx] = None
                    self._flushed_ring_pos[l_idx] = None
                target_hist_len = eff_hist_len

            # Crop historical tiers to `< target_hist_len`
            for h_idx in range(self.num_kv_heads):
                if self.hikvp_store is not None and self.hikvp_store.is_offloaded(l_idx, h_idx):
                    h_index = self.hikvp_store.get_head(l_idx, h_idx)
                    if h_index is not None:
                        h_index.crop(target_hist_len)
                    continue
                st = self._tiers[l_idx][h_idx]
                if st.role == ROLE_RETRIEVAL_FP16:
                    if st.fp16_k is not None:
                        st.fp16_k = st.fp16_k[:, :target_hist_len, :]
                        st.fp16_v = st.fp16_v[:, :target_hist_len, :]
                elif st.role == ROLE_PEAKED_H2O:
                    if st.h2o_indices is not None and st.h2o_k is not None:
                        keep_mask = st.h2o_indices[0] < target_hist_len
                        st.h2o_k = st.h2o_k[:, keep_mask, :]
                        st.h2o_v = st.h2o_v[:, keep_mask, :]
                        st.h2o_scores = st.h2o_scores[:, keep_mask]
                        st.h2o_indices = st.h2o_indices[:, keep_mask]
                elif st.role == ROLE_DIFFUSE_LEXICO:
                    if st.lex_k_idx is not None and st.lex_k_val is not None:
                        st.lex_k_idx = st.lex_k_idx[:, :target_hist_len, :]
                        st.lex_v_idx = st.lex_v_idx[:, :target_hist_len, :]
                        st.lex_k_val = st.lex_k_val[:, :target_hist_len, :]
                        st.lex_v_val = st.lex_v_val[:, :target_hist_len, :]
                        if st.lex_recon_k is not None and st.lex_recon_len >= target_hist_len:
                            st.lex_recon_k = st.lex_recon_k[:, :target_hist_len, :]
                            st.lex_recon_v = st.lex_recon_v[:, :target_hist_len, :]
                            st.lex_recon_len = target_hist_len
                        else:
                            st.lex_recon_atoms = -1
                elif st.role == ROLE_QUANT_LOWBIT:
                    if st.quant_k_packed is not None:
                        st.quant_k_packed = st.quant_k_packed[:, :target_hist_len, :]
                        st.quant_k_min = st.quant_k_min[:, :target_hist_len, :]
                        st.quant_k_scale = st.quant_k_scale[:, :target_hist_len, :]
                        st.quant_v_packed = st.quant_v_packed[:, :target_hist_len, :]
                        st.quant_v_min = st.quant_v_min[:, :target_hist_len, :]
                        st.quant_v_scale = st.quant_v_scale[:, :target_hist_len, :]

            if l_idx < len(self.key_cache) and self.key_cache[l_idx].numel() > 0 and self._local_k[l_idx] is not None:
                k_out, v_out, _ = self._materialize_layer_kv(
                    l_idx, dtype=self._local_k[l_idx].dtype, device=self._local_k[l_idx].device
                )
                self.key_cache[l_idx] = k_out
                self.value_cache[l_idx] = v_out

        self._refresh_telemetry()

    def get_telemetry_summary(self) -> Dict[str, Any]:
        """Returns a dictionary of memory and routing telemetry metrics."""
        self._flush_all_pending_overflow()
        self._refresh_telemetry()
        ratio = (
            float(self.compressed_bytes) / float(self.fp16_bytes)
            if self.fp16_bytes > 0
            else 1.0
        )
        phys_ratio = (
            float(self.physical_resident_bytes) / float(self.fp16_bytes)
            if self.fp16_bytes > 0
            else 1.0
        )
        return {
            "seq_length": self.get_seq_length(0),
            "policy_mode": self.policy_mode,
            "active_atoms": self.active_atoms,
            "s_budget": self.s_budget,
            "keep_ratio": self.keep_ratio,
            "n_b": self.n_b,
            "fp16_bytes": self.fp16_bytes,
            "compressed_bytes": self.compressed_bytes,
            "physical_resident_bytes": self.physical_resident_bytes,
            "raw_tensor_bytes": int(getattr(self, "raw_tensor_bytes", 0)),
            "raw_fp16_tensor_bytes": int(getattr(self, "raw_fp16_tensor_bytes", 0)),
            "memory_ratio_vs_fp16": ratio,
            "physical_memory_ratio_vs_fp16": phys_ratio,
            "memory_savings_pct": (1.0 - ratio) * 100.0,
            "dictionary_bytes": int(getattr(self, "dictionary_bytes", 0)),
            "memory_ratio_incl_dictionary": (
                float(self.compressed_bytes + int(getattr(self, "dictionary_bytes", 0)))
                / float(self.fp16_bytes)
                if self.fp16_bytes > 0
                else 1.0
            ),
            "bytes_read_last_step": self.bytes_read_last_step,
            "cumulative_bytes_read": self.cumulative_bytes_read,
            "per_tier_token_counts": dict(self.per_tier_token_counts),
            "hmm_last_beta_t": self.controller.last_beta_t,
            "hmm_last_z_score": self.controller.last_z_score,
            "hmm_last_vpin": self.controller.last_vpin,
            "hmm_shift_events": self.controller.shift_events,
            "stoikov_inventory_q": self.controller.last_inventory_q,
            "stoikov_reservation_skew": self.controller.last_reservation_skew,
            "stoikov_half_spread": self.controller.last_half_spread,
            "stoikov_kappa_t": self.controller.last_kappa_t,
            "stoikov_liquidity_rebate": self.controller.last_liquidity_rebate,
            "effective_keep_ratio": self.controller.last_keep_ratio,
            "head_roles": self.head_roles.detach().cpu().tolist(),
            "hikvp": self.hikvp_telemetry(),
        }
