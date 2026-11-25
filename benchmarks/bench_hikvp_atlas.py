#!/usr/bin/env python3
"""
Reproducible Benchmark Suite for StoikovKV HiKVP (Hierarchical Offloaded Fast Retrieval)
and AtlasMemoryBank (Doc2KV / KG2KV Rectangular Attention).

Modes
-----
1. `--mode synthetic` (fast CPU-runnable benchmark):
   - Sweeps sequence lengths (e.g. 2K -> 8K -> 32K -> 64K) on `HierarchicalKVHeadIndex`
     for both `U_Q` subspace centroids and full-dim centroids.
   - Measures `gpu_index_bytes`, `gpu_lru_bytes`, `cpu_leaf_bytes`, `gpu_to_cpu_ratio`,
     5-depth synthetic needle recall, prune latency, cold/warm retrieve latency,
     LRU warm hit rate, and `draft_zero_pcie` PCIe transfer count.
   - Runs an end-to-end synthetic `AtlasMemoryBank` (`Doc2KV` + `KG2KV`) build,
     Rectangular Attention query, and disk save/load micro-benchmark on
     `FallbackQwen2ForCausalLM`.

2. `--mode hf` (CUDA / real HuggingFace model benchmark):
   - Loads a real Qwen2.5 (or compatible) CausalLM + calibrated artifacts.
   - Evaluates live `StoikovKVCache` + `HiKVPConfig` across `--seq-lengths` on
     Paul Graham-style long-context essays with planted needles, plus Matryoshka
     speculative decoding with `draft_zero_pcie=True`.
   - Benchmarks `AtlasMemoryBank` (`Doc2KV` & `KG2KV`) Rectangular Attention
     against full prompt re-prefill.

Example usage
-------------
    python3 benchmarks/bench_hikvp_atlas.py \
        --mode synthetic \
        --seq-lengths 2048,8192,32768,65536 \
        --page-size 64 \
        --top-k-pages 4 \
        --device cpu \
        --output benchmarks/results/hikvp_atlas_benchmark.json
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/triton_stoikov")

ROOT_DIR = Path(__file__).resolve().parent.parent
for _p in (ROOT_DIR, ROOT_DIR / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import torch

from stoikov_kv.atlas_memory import (
    AtlasMemoryBank,
    AtlasTriple,
)
from stoikov_kv.cache import (
    StoikovKVCache,
    create_mock_artifacts,
    load_or_create_artifacts,
)
from stoikov_kv.calibration import (
    ROLE_DIFFUSE_LEXICO,
    ROLE_PEAKED_H2O,
    ROLE_RETRIEVAL_FP16,
    FallbackQwen2Config,
    FallbackQwen2ForCausalLM,
    FallbackQwen2Tokenizer,
    load_model_and_tokenizer,
)
from stoikov_kv.hikvp import (
    HierarchicalKVHeadIndex,
    HiKVPConfig,
)
from stoikov_kv.matryoshka_spec import matryoshka_speculative_generate


# Built-in Paul Graham-style essay passages for reproducible --mode hf runs when
# no external --corpus-dir is supplied.
_BUILTIN_PG_ESSAYS: List[Tuple[str, str]] = [
    (
        "pg_startups_and_systems.txt",
        (
            "One of the most surprising lessons in building ambitious software is that "
            "the hardest bottlenecks are rarely where you first expect them to be. "
            "Early teams spend weeks debating elegant abstractions while memory bandwidth "
            "quietly dominates wall-clock latency. When you measure what actually happens "
            "during long-context decoding, moving inactive key-value pages across memory "
            "tiers outweighs almost every floating-point operation in the attention kernel. "
            "The best systems look simple in hindsight because they separate cold storage "
            "from hot working sets and keep only tiny routing summaries close to compute. "
        ),
    ),
    (
        "pg_makers_and_focus.txt",
        (
            "Builders work on a different schedule from managers. A single interruption "
            "breaks an entire afternoon of deep architectural work because holding a "
            "complex invariant in working memory takes twenty minutes to warm up. "
            "In much the same way, a language model loses fidelity when a naive eviction "
            "policy discards a rare foundational fact planted thousands of tokens earlier. "
            "Preserving retrieval heads with exact offloaded pages lets you scale context "
            "length without losing the thread of the original design specification. "
        ),
    ),
    (
        "pg_compounding_knowledge.txt",
        (
            "Writing essays and curating structured knowledge graphs are two sides of "
            "the same habit: externalizing thought so you never have to re-derive it from "
            "scratch. If every query forced you to re-read an entire library word for word, "
            "research would grind to a halt. Pre-encoding documents and knowledge-graph "
            "triples into persistent KV memory banks allows instant rectangular attention "
            "over external knowledge with zero redundant prefill cost. "
        ),
    ),
]


def _sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _extract_page_ids(raw: Any) -> List[int]:
    if isinstance(raw, tuple):
        raw = raw[0]
    if isinstance(raw, torch.Tensor):
        return [int(x) for x in raw.reshape(-1).tolist()]
    return [int(x) for x in list(raw)]


def _unpack_retrieve(raw: Any) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    if isinstance(raw, dict):
        k = raw["k"] if "k" in raw else raw["keys"]
        v = raw["v"] if "v" in raw else raw["values"]
        pos = raw.get("positions", raw.get("pos", None))
        return k, v, pos
    if isinstance(raw, (tuple, list)):
        if len(raw) >= 3:
            return raw[0], raw[1], raw[2]
        if len(raw) == 2:
            return raw[0], raw[1], None
    raise TypeError(f"Unexpected retrieve() return type: {type(raw)}")


def _get_head_lru(head: HierarchicalKVHeadIndex) -> Any:
    for attr in ("lru_cache", "lru", "_lru_cache", "_lru", "page_cache", "hot_cache"):
        obj = getattr(head, attr, None)
        if obj is not None:
            return obj
    return None


def _orthonormal_basis(head_dim: int, rank: int, seed: int = 42) -> torch.Tensor:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    mat = torch.randn(head_dim, rank, generator=gen, dtype=torch.float32)
    q, _ = torch.linalg.qr(mat, mode="reduced")
    return q[:, :rank].contiguous()


def _parse_seq_lengths(raw: str) -> List[int]:
    out = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not out:
        raise ValueError("--seq-lengths must contain at least one positive integer")
    return out


# ============================================================================
# Mode 1: Synthetic CPU/GPU Scaling + Atlas Micro-Benchmark
# ============================================================================


def benchmark_hikvp_synthetic_scaling(
    seq_lengths: List[int],
    page_size: int,
    top_k_pages: int,
    inter_fanout: int,
    max_hot_pages: int,
    head_dim: int,
    subspace_rank: int,
    device: torch.device,
    seed: int,
) -> List[Dict[str, Any]]:
    """
    Sweeps sequence lengths across both U_Q subspace centroids and full-dim centroids,
    measuring GPU/CPU memory scaling, 5-depth needle recall, prune & retrieve latency,
    warm LRU hit rate, and zero-PCIe draft mode.
    """
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    rank = min(subspace_rank, head_dim)
    u_q = _orthonormal_basis(head_dim, rank, seed=seed)
    needle_depths = [0.05, 0.25, 0.50, 0.75, 0.95]

    rows: List[Dict[str, Any]] = []

    for use_subspace in (True, False):
        mode_label = f"subspace_r{rank}" if use_subspace else "full_dim"

        for seq_len in seq_lengths:
            gen = torch.Generator(device="cpu").manual_seed(seed + seq_len)
            k_all = (0.05 * torch.randn(seq_len, head_dim, generator=gen, dtype=torch.float32)).to(dtype)
            v_all = (0.05 * torch.randn(seq_len, head_dim, generator=gen, dtype=torch.float32)).to(dtype)
            pos_all = torch.arange(seq_len, dtype=torch.long)

            num_pages = max(1, (seq_len + page_size - 1) // page_size)
            needles: List[Tuple[int, int, torch.Tensor]] = []
            used_pages = set()
            for idx_n, depth in enumerate(needle_depths):
                tok_idx = min(seq_len - 1, max(0, int(depth * seq_len)))
                target_page = tok_idx // page_size
                if target_page in used_pages:
                    continue
                used_pages.add(target_page)
                basis_col = idx_n % rank
                direction = u_q[:, basis_col].to(dtype)
                k_all[tok_idx] = (14.0 * direction).to(dtype)
                v_all[tok_idx] = torch.full((head_dim,), float(target_page + 1), dtype=dtype)
                needles.append((target_page, tok_idx, direction.unsqueeze(0).to(device)))

            cfg = HiKVPConfig(
                page_size=page_size,
                inter_fanout=inter_fanout,
                top_k_roots=max(4, top_k_pages),
                top_k_inter=max(8, top_k_pages * 2),
                top_k_pages=top_k_pages,
                max_hot_pages_per_head=max_hot_pages,
                use_subspace_centroids=use_subspace,
                include_min_max_bounds=True,
                pin_memory=(device.type == "cuda"),
                async_prefetch=(device.type == "cuda"),
                draft_zero_pcie=True,
            )

            index = HierarchicalKVHeadIndex(
                layer_idx=0,
                head_idx=0,
                head_dim=head_dim,
                config=cfg,
                u_q=u_q.to(dtype) if use_subspace else None,
            )

            # Build index incrementally in 1024-token chunks
            t_build0 = time.perf_counter()
            chunk_sz = 1024
            for start in range(0, seq_len, chunk_sz):
                end = min(seq_len, start + chunk_sz)
                index.append_pre_rope(k_all[start:end], v_all[start:end], pos_all[start:end])
            _sync_device(device)
            build_ms = (time.perf_counter() - t_build0) * 1000.0

            lru = _get_head_lru(index)

            # 1. Verify draft_zero_pcie=True transfers strictly 0 PCIe bytes on cold pages
            pcie_before_draft = int(getattr(lru, "pcie_bytes_transferred", 0)) if lru is not None else 0
            for _, _, q_vec in needles:
                _ = index.retrieve(
                    q_vec,
                    device=device,
                    dtype=dtype,
                    top_k_pages=top_k_pages,
                    draft_zero_pcie=True,
                )
            _sync_device(device)
            pcie_after_draft = int(getattr(lru, "pcie_bytes_transferred", 0)) if lru is not None else 0
            draft_pcie_bytes = pcie_after_draft - pcie_before_draft

            # 2. Measure prune_page_ids latency + needle recall
            prune_times_ms: List[float] = []
            needle_hits = 0
            for target_page, _tok_idx, q_vec in needles:
                _sync_device(device)
                t0 = time.perf_counter()
                page_ids = _extract_page_ids(index.prune_page_ids(q_vec, top_k_pages=top_k_pages))
                _sync_device(device)
                prune_times_ms.append((time.perf_counter() - t0) * 1000.0)
                if target_page in page_ids:
                    needle_hits += 1
            needle_recall = float(needle_hits) / float(max(1, len(needles)))

            # 3. Measure cold retrieve latency (clear LRU first)
            if lru is not None:
                lru.clear()
            cold_times_ms: List[float] = []
            for _target_page, _tok_idx, q_vec in needles:
                if lru is not None:
                    lru.clear()
                _sync_device(device)
                t0 = time.perf_counter()
                _ = index.retrieve(
                    q_vec,
                    device=device,
                    dtype=dtype,
                    top_k_pages=top_k_pages,
                    draft_zero_pcie=False,
                )
                _sync_device(device)
                cold_times_ms.append((time.perf_counter() - t0) * 1000.0)

            # 4. Measure warm retrieve latency & hit rate (repeat on hot query)
            q_hot = needles[0][2]
            if lru is not None:
                lru.clear()
            _ = index.retrieve(
                q_hot,
                device=device,
                dtype=dtype,
                top_k_pages=top_k_pages,
                draft_zero_pcie=False,
            )
            hits_start = int(getattr(lru, "hits", 0)) if lru is not None else 0
            misses_start = int(getattr(lru, "misses", 0)) if lru is not None else 0

            warm_times_ms: List[float] = []
            for _ in range(10):
                _sync_device(device)
                t0 = time.perf_counter()
                _ = index.retrieve(
                    q_hot,
                    device=device,
                    dtype=dtype,
                    top_k_pages=top_k_pages,
                    draft_zero_pcie=False,
                )
                _sync_device(device)
                warm_times_ms.append((time.perf_counter() - t0) * 1000.0)

            hits_delta = (int(getattr(lru, "hits", 0)) - hits_start) if lru is not None else 0
            misses_delta = (int(getattr(lru, "misses", 0)) - misses_start) if lru is not None else 0
            warm_hit_rate = (
                float(hits_delta) / float(hits_delta + misses_delta)
                if (hits_delta + misses_delta) > 0
                else 0.0
            )

            gpu_index_bytes = int(index.gpu_index_bytes())
            gpu_lru_bytes = int(index.gpu_lru_bytes())
            gpu_total_bytes = gpu_index_bytes + gpu_lru_bytes
            cpu_leaf_bytes = int(index.cpu_leaf_bytes())
            gpu_to_cpu_ratio = (
                float(gpu_total_bytes) / float(cpu_leaf_bytes) if cpu_leaf_bytes > 0 else 0.0
            )

            rows.append(
                {
                    "centroid_mode": mode_label,
                    "use_subspace_centroids": use_subspace,
                    "seq_len": seq_len,
                    "num_pages": num_pages,
                    "page_size": page_size,
                    "top_k_pages": top_k_pages,
                    "head_dim": head_dim,
                    "dtype": str(dtype).replace("torch.", ""),
                    "build_ms": round(build_ms, 3),
                    "gpu_index_bytes": gpu_index_bytes,
                    "gpu_lru_bytes": gpu_lru_bytes,
                    "gpu_total_bytes": gpu_total_bytes,
                    "cpu_leaf_bytes": cpu_leaf_bytes,
                    "gpu_to_cpu_ratio": round(gpu_to_cpu_ratio, 6),
                    "needle_recall": round(needle_recall, 4),
                    "prune_latency_ms": round(statistics.median(prune_times_ms), 4),
                    "retrieve_cold_latency_ms": round(statistics.median(cold_times_ms), 4),
                    "retrieve_warm_latency_ms": round(statistics.median(warm_times_ms), 4),
                    "warm_lru_hit_rate": round(warm_hit_rate, 4),
                    "draft_zero_pcie_bytes": draft_pcie_bytes,
                }
            )

    return rows


def benchmark_atlas_synthetic(
    page_size: int,
    top_k_pages: int,
    device: torch.device,
    seed: int,
) -> Dict[str, Any]:
    """
    Micro-benchmark for AtlasMemoryBank (Doc2KV + KG2KV) on FallbackQwen2ForCausalLM:
    measures build time, bank stats, Rectangular Attention query latency (zero re-prefill),
    and save/load serialization latency & disk footprint.
    """
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    cfg = FallbackQwen2Config(
        hidden_size=128,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=512,
    )
    model = FallbackQwen2ForCausalLM(cfg, device=str(device), dtype=dtype, seed=seed)
    model.eval()
    tokenizer = FallbackQwen2Tokenizer(vocab_size=512)
    artifacts = create_mock_artifacts(model, tokenizer=tokenizer, subspace_rank=8, dict_atoms=64)
    roles = artifacts["head_roles"].clone()
    roles[0, :] = ROLE_RETRIEVAL_FP16
    roles[1, :] = ROLE_PEAKED_H2O
    roles[2:, :] = ROLE_DIFFUSE_LEXICO
    artifacts["head_roles"] = roles

    hikvp_cfg = HiKVPConfig(
        page_size=min(page_size, 32),
        inter_fanout=4,
        top_k_roots=4,
        top_k_inter=4,
        top_k_pages=top_k_pages,
        max_hot_pages_per_head=8,
        use_subspace_centroids=True,
        pin_memory=(device.type == "cuda"),
        async_prefetch=False,
    )

    # 1. Doc2KV build from multi-essay corpus
    docs = [body * 4 for _, body in _BUILTIN_PG_ESSAYS]
    sources = [name for name, _ in _BUILTIN_PG_ESSAYS]
    t0 = time.perf_counter()
    doc_bank = AtlasMemoryBank.build_from_texts(
        model=model,
        tokenizer=tokenizer,
        texts=docs,
        artifacts=artifacts,
        hikvp_config=hikvp_cfg,
        sources=sources,
        chunk_tokens=64,
        all_heads=True,
    )
    _sync_device(device)
    doc_build_ms = (time.perf_counter() - t0) * 1000.0

    # Save / Load benchmark
    with tempfile.TemporaryDirectory() as tmpdir:
        t0 = time.perf_counter()
        doc_bank.save(tmpdir)
        doc_save_ms = (time.perf_counter() - t0) * 1000.0
        disk_bytes = sum(
            os.path.getsize(os.path.join(r, f))
            for r, _, files in os.walk(tmpdir)
            for f in files
        )
        t0 = time.perf_counter()
        _reloaded = AtlasMemoryBank.load(tmpdir, U_Q=artifacts["U_Q"])
        doc_load_ms = (time.perf_counter() - t0) * 1000.0

    # 2. KG2KV build from 64 structured triples
    triples = [
        AtlasTriple(
            subject=f"ClusterNode_{i:02d}",
            relation="routes_to" if i % 2 == 0 else "stores_tier",
            obj=f"Gateway_{i % 5}" if i % 2 == 0 else f"HiKVP_Level_{(i % 3) + 1}",
        )
        for i in range(64)
    ]
    t0 = time.perf_counter()
    kg_bank = AtlasMemoryBank.build_from_kg_triples(
        model=model,
        tokenizer=tokenizer,
        triples=triples,
        artifacts=artifacts,
        hikvp_config=hikvp_cfg,
        triples_per_chunk=16,
        all_heads=True,
    )
    _sync_device(device)
    kg_build_ms = (time.perf_counter() - t0) * 1000.0

    # 3. Rectangular Attention query latency (zero re-prefill of KG)
    cache = StoikovKVCache(
        model=model,
        artifacts=artifacts,
        n_b=16,
        keep_ratio=0.25,
        s_budget=8,
        policy_mode="hybrid",
    )
    kg_bank.attach_to_cache(cache)
    cache.attach_query_hooks(model)
    try:
        enc = tokenizer("Which gateway does ClusterNode_00 route to?", return_tensors="pt")
        q_ids = (enc["input_ids"] if isinstance(enc, dict) else enc.input_ids).to(device)
        _sync_device(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model(
                input_ids=q_ids,
                past_key_values=cache,
                use_cache=True,
                output_attentions=False,
            )
            tok = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
            for _ in range(4):
                out = model(
                    input_ids=tok,
                    past_key_values=cache,
                    use_cache=True,
                    output_attentions=False,
                )
                tok = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
        _sync_device(device)
        rect_attn_ms = (time.perf_counter() - t0) * 1000.0
        local_seq_len = int(cache.get_seq_length(0))
        rect_telemetry = cache.hikvp_telemetry()
    finally:
        kg_bank.detach_from_cache(cache)
        cache.detach_query_hooks()

    return {
        "doc2kv": {
            "build_ms": round(doc_build_ms, 3),
            "save_ms": round(doc_save_ms, 3),
            "load_ms": round(doc_load_ms, 3),
            "disk_bytes": int(disk_bytes),
            "stats": doc_bank.stats(),
        },
        "kg2kv": {
            "num_triples": len(triples),
            "build_ms": round(kg_build_ms, 3),
            "rectangular_attention_5step_ms": round(rect_attn_ms, 3),
            "local_cache_seq_len": local_seq_len,
            "stats": kg_bank.stats(),
            "telemetry": rect_telemetry,
        },
    }


# ============================================================================
# Mode 2: Real HuggingFace Model Benchmark (CUDA GPU Ready)
# ============================================================================


def _load_essays(corpus_dir: Optional[str]) -> List[Tuple[str, str]]:
    if corpus_dir and os.path.isdir(corpus_dir):
        found: List[Tuple[str, str]] = []
        for path in sorted(Path(corpus_dir).glob("*.txt")):
            text = path.read_text(encoding="utf-8", errors="ignore").strip()
            if text:
                found.append((path.name, text))
        if found:
            return found
    return list(_BUILTIN_PG_ESSAYS)


def _build_long_essay_prompt(
    tokenizer: Any,
    essays: List[Tuple[str, str]],
    target_tokens: int,
    passkey: str = "73918",
    needle_depth: float = 0.20,
) -> Tuple[str, torch.Tensor]:
    needle_sentence = (
        f"\n[ARCHIVAL NOTE: The secret verification passkey for this essay collection is {passkey}.]\n"
    )
    query_suffix = "\nQuestion: What is the secret verification passkey mentioned in the archival note? Answer with only the 5-digit passkey:"
    needle_ids = tokenizer.encode(needle_sentence, add_special_tokens=False)
    q_ids_list = tokenizer.encode(query_suffix, add_special_tokens=False)
    haystack_budget = max(64, target_tokens - len(needle_ids) - len(q_ids_list))

    # Concatenate essays only until we have enough tokens (avoids tokenizing 8MB unnecessarily)
    haystack_ids: List[int] = []
    idx = 0
    while len(haystack_ids) < haystack_budget:
        _, text = essays[idx % len(essays)]
        haystack_ids.extend(tokenizer.encode(text + "\n\n", add_special_tokens=False))
        idx += 1
        if idx > 1000:
            break
    haystack_ids = haystack_ids[:haystack_budget]

    insert_tok = max(16, min(len(haystack_ids) - 16, int(len(haystack_ids) * needle_depth)))
    full_ids = haystack_ids[:insert_tok] + needle_ids + haystack_ids[insert_tok:] + q_ids_list
    final_ids = torch.tensor([full_ids], dtype=torch.long)
    return needle_sentence.strip(), final_ids


def benchmark_hf_mode(args: argparse.Namespace, device: torch.device) -> Dict[str, Any]:
    """Runs live HiKVP + AtlasMemoryBank benchmarks on a real HF CausalLM."""
    model, tokenizer = load_model_and_tokenizer(
        model_name=args.model,
        device=str(device),
        use_real_model=True,
    )
    is_fallback = model.__class__.__name__ == "FallbackQwen2ForCausalLM"
    model.eval()
    artifacts_path = getattr(args, "artifacts", None)
    if artifacts_path and os.path.isfile(artifacts_path):
        artifacts = torch.load(artifacts_path, map_location="cpu", weights_only=False)
    else:
        artifacts = load_or_create_artifacts(model=model, tokenizer=tokenizer)
    essays = _load_essays(args.corpus_dir)

    hikvp_cfg = HiKVPConfig(
        page_size=args.page_size,
        inter_fanout=args.inter_fanout,
        top_k_roots=max(4, args.top_k_pages),
        top_k_inter=max(8, args.top_k_pages * 2),
        top_k_pages=args.top_k_pages,
        max_hot_pages_per_head=args.max_hot_pages,
        use_subspace_centroids=not getattr(args, "no_subspace_centroids", False),
        include_min_max_bounds=True,
        pin_memory=(device.type == "cuda"),
        async_prefetch=(device.type == "cuda"),
        offload_roles=(ROLE_RETRIEVAL_FP16,),
        draft_zero_pcie=True,
    )

    seq_results: List[Dict[str, Any]] = []
    passkey = "73918"

    for seq_len in _parse_seq_lengths(args.seq_lengths):
        _, input_ids = _build_long_essay_prompt(
            tokenizer=tokenizer,
            essays=essays,
            target_tokens=seq_len,
            passkey=passkey,
            needle_depth=0.20,
        )
        input_ids = input_ids.to(device)

        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)

        cache = StoikovKVCache(
            model=model,
            artifacts=artifacts,
            n_b=64,
            keep_ratio=0.20,
            s_budget=16,
            policy_mode="hybrid",
            hikvp_config=hikvp_cfg,
        )
        cache.attach_query_hooks(model)
        try:
            _sync_device(device)
            t0 = time.perf_counter()
            spec_res = matryoshka_speculative_generate(
                model=model,
                tokenizer=tokenizer,
                prompt=input_ids,
                cache=cache,
                gamma_spec=4,
                draft_atoms=4,
                verify_atoms=16,
                max_new_tokens=args.max_new_tokens,
                strict_rejection_sampling=True,
                do_sample=False,
                seed=args.seed,
                prefill_chunk_size=int(getattr(args, "chunk_prefill", 0)),
            )
            _sync_device(device)
            elapsed_s = time.perf_counter() - t0

            tel = cache.hikvp_telemetry()
            gen_text = spec_res.get("generated_text", "")
            if not gen_text and hasattr(tokenizer, "decode"):
                gen_text = tokenizer.decode(spec_res["generated_ids"], skip_special_tokens=True)

            peak_vram_bytes = (
                int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
            )
            seq_results.append(
                {
                    "seq_len": int(input_ids.shape[1]),
                    "max_new_tokens": args.max_new_tokens,
                    "total_time_s": round(elapsed_s, 4),
                    "tokens_per_sec": round(args.max_new_tokens / max(elapsed_s, 1e-6), 3),
                    "acceptance_rate": round(float(spec_res.get("acceptance_rate", 0.0)), 4),
                    "passkey_hit": (passkey in str(gen_text)),
                    "generated_preview": str(gen_text)[:120],
                    "peak_cuda_vram_bytes": peak_vram_bytes,
                    "hikvp_telemetry": tel,
                }
            )
        finally:
            cache.detach_query_hooks()

    # AtlasMemoryBank Doc2KV Rectangular Attention benchmark on real model
    texts = [body for _, body in essays]
    sources = [name for name, _ in essays]
    max_atlas_chunks = getattr(args, "max_atlas_chunks", 32)
    t0 = time.perf_counter()
    bank = AtlasMemoryBank.build_from_texts(
        model=model,
        tokenizer=tokenizer,
        texts=texts,
        artifacts=artifacts,
        hikvp_config=hikvp_cfg,
        sources=sources,
        chunk_tokens=256,
        max_chunks=max_atlas_chunks,
        all_heads=True,
    )
    _sync_device(device)
    bank_build_ms = (time.perf_counter() - t0) * 1000.0

    atlas_cache = StoikovKVCache(
        model=model,
        artifacts=artifacts,
        n_b=32,
        keep_ratio=0.20,
        s_budget=16,
        policy_mode="hybrid",
    )
    bank.attach_to_cache(atlas_cache)
    atlas_cache.attach_query_hooks(model)
    try:
        q_enc = tokenizer(
            "Summarize the core systems lesson about memory bandwidth from the essays.",
            return_tensors="pt",
        )
        q_ids = (q_enc["input_ids"] if isinstance(q_enc, dict) else q_enc.input_ids).to(device)
        _sync_device(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(
                input_ids=q_ids,
                past_key_values=atlas_cache,
                use_cache=True,
                output_attentions=False,
            )
        _sync_device(device)
        atlas_ttft_ms = (time.perf_counter() - t0) * 1000.0
        atlas_tel = atlas_cache.hikvp_telemetry()
    finally:
        bank.detach_from_cache(atlas_cache)
        atlas_cache.detach_query_hooks()

    return {
        "model": args.model,
        "is_fallback_model": bool(is_fallback),
        "hikvp_seq_sweep": seq_results,
        "atlas_doc2kv": {
            "num_essays": len(essays),
            "build_ms": round(bank_build_ms, 3),
            "rectangular_attention_ttft_ms": round(atlas_ttft_ms, 3),
            "stats": bank.stats(),
            "telemetry": atlas_tel,
        },
    }


# ============================================================================
# CLI & Console Reporting
# ============================================================================


def _format_bytes(n: int) -> str:
    if n >= 1024 * 1024:
        return f"{n / (1024 * 1024):.2f} MiB"
    if n >= 1024:
        return f"{n / 1024:.1f} KiB"
    return f"{n} B"


def _print_synthetic_table(rows: List[Dict[str, Any]]) -> None:
    header = (
        f"{'Mode':<14} {'SeqLen':>7} {'GPU Idx':>10} {'GPU LRU':>10} "
        f"{'CPU Leaf':>11} {'GPU/CPU':>8} {'Recall':>7} {'Prune(ms)':>10} "
        f"{'Cold(ms)':>9} {'Warm(ms)':>9} {'HitRate':>8} {'DraftPCIe':>10}"
    )
    print("\n" + "=" * len(header))
    print("HiKVP 3-Level Tree Scaling Benchmark")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['centroid_mode']:<14} {r['seq_len']:>7d} "
            f"{_format_bytes(r['gpu_index_bytes']):>10} "
            f"{_format_bytes(r['gpu_lru_bytes']):>10} "
            f"{_format_bytes(r['cpu_leaf_bytes']):>11} "
            f"{r['gpu_to_cpu_ratio'] * 100:>7.2f}% "
            f"{r['needle_recall'] * 100:>6.1f}% "
            f"{r['prune_latency_ms']:>10.3f} "
            f"{r['retrieve_cold_latency_ms']:>9.3f} "
            f"{r['retrieve_warm_latency_ms']:>9.3f} "
            f"{r['warm_lru_hit_rate'] * 100:>7.1f}% "
            f"{r['draft_zero_pcie_bytes']:>10d}"
        )
    print("=" * len(header) + "\n")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark StoikovKV HiKVP (3-level offloaded KV retrieval) and AtlasMemoryBank."
    )
    parser.add_argument(
        "--mode",
        choices=("synthetic", "hf"),
        default="synthetic",
        help="Benchmark mode: fast CPU 'synthetic' scaling or real 'hf' CausalLM.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="HuggingFace model ID for --mode hf.",
    )
    parser.add_argument(
        "--seq-lengths",
        type=str,
        default="2048,8192,32768",
        help="Comma-separated sequence lengths to sweep (e.g. 2048,8192,32768,65536).",
    )
    parser.add_argument("--page-size", type=int, default=64, help="Tokens per HiKVP leaf page.")
    parser.add_argument("--top-k-pages", type=int, default=4, help="Pages retrieved per query.")
    parser.add_argument("--inter-fanout", type=int, default=8, help="Intermediate tree node fanout.")
    parser.add_argument("--max-hot-pages", type=int, default=8, help="GPU LRU capacity per head.")
    parser.add_argument("--subspace-rank", type=int, default=16, help="U_Q subspace centroid rank.")
    parser.add_argument("--head-dim", type=int, default=128, help="Head dimension for synthetic sweep.")
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="Execution device ('cpu' or 'cuda').",
    )
    parser.add_argument(
        "--corpus-dir",
        type=str,
        default=None,
        help="Optional directory of .txt essays (e.g. Paul Graham essays) for --mode hf.",
    )
    parser.add_argument(
        "--artifacts",
        type=str,
        default=None,
        help="Optional calibrated artifacts .pt path for --mode hf.",
    )
    parser.add_argument(
        "--max-atlas-chunks",
        type=int,
        default=32,
        help="Max chunks to index into AtlasMemoryBank in --mode hf.",
    )
    parser.add_argument(
        "--chunk-prefill",
        type=int,
        default=0,
        help="Optional chunked prefill size (e.g. 1024) to cap activation VRAM at 8K-32K context.",
    )
    parser.add_argument(
        "--no-subspace-centroids",
        action="store_true",
        help="Use full head_dim pre-RoPE Quest min/max bounds instead of U_Q subspace projection.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=16,
        help="Tokens to decode per sequence length in --mode hf.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--output",
        type=str,
        default="benchmarks/results/hikvp_atlas_benchmark.json",
        help="Path to write JSON benchmark report.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but unavailable; falling back to CPU.", file=sys.stderr)
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    report: Dict[str, Any] = {
        "mode": args.mode,
        "device": str(device),
        "config": {
            "seq_lengths": _parse_seq_lengths(args.seq_lengths),
            "page_size": args.page_size,
            "top_k_pages": args.top_k_pages,
            "inter_fanout": args.inter_fanout,
            "max_hot_pages": args.max_hot_pages,
            "subspace_rank": args.subspace_rank,
            "head_dim": args.head_dim,
            "seed": args.seed,
        },
    }

    if args.mode == "synthetic":
        scaling_rows = benchmark_hikvp_synthetic_scaling(
            seq_lengths=_parse_seq_lengths(args.seq_lengths),
            page_size=args.page_size,
            top_k_pages=args.top_k_pages,
            inter_fanout=args.inter_fanout,
            max_hot_pages=args.max_hot_pages,
            head_dim=args.head_dim,
            subspace_rank=args.subspace_rank,
            device=device,
            seed=args.seed,
        )
        _print_synthetic_table(scaling_rows)
        atlas_res = benchmark_atlas_synthetic(
            page_size=args.page_size,
            top_k_pages=args.top_k_pages,
            device=device,
            seed=args.seed,
        )
        report["hikvp_scaling"] = scaling_rows
        report["atlas_synthetic"] = atlas_res
        print(
            f"AtlasMemoryBank Doc2KV build={atlas_res['doc2kv']['build_ms']:.2f}ms "
            f"(disk={_format_bytes(atlas_res['doc2kv']['disk_bytes'])}) | "
            f"KG2KV ({atlas_res['kg2kv']['num_triples']} triples) build={atlas_res['kg2kv']['build_ms']:.2f}ms, "
            f"RectAttn 5-step={atlas_res['kg2kv']['rectangular_attention_5step_ms']:.2f}ms"
        )
    else:
        report["hf_benchmark"] = benchmark_hf_mode(args, device=device)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"Saved benchmark report to {out_path}")


if __name__ == "__main__":
    main()
