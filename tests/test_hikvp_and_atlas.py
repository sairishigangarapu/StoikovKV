"""
Unit tests for StoikovKV HiKVP (Hierarchical Offloaded Fast Retrieval)
and AtlasMemoryBank (Doc2KV / KG2KV Rectangular Attention).

Covers:
1.  GPULRUPageCache hit/miss/eviction/invalidate_ge/clear semantics
2.  HierarchicalKVHeadIndex 3-level tree needle recall (full-dim & U_Q subspace)
3.  Sub-linear GPU index + LRU scaling vs linear CPU leaf growth (512 -> 16,384)
4.  Exact mid-page, page-boundary, and zero crop rollback
5.  Draft zero-PCIe mode and LRU temporal locality (100% warm hit rate)
6.  NVMe mmap disk backing and HiKVPStore / HeadIndex save/load round-trip
7.  End-to-end StoikovKVCache + HiKVP forward pass, telemetry, and crop
8.  Matryoshka self-speculative decoding with HiKVP (strict greedy parity + zero draft PCIe)
9.  AtlasMemoryBank Doc2KV build, stats, retrieve_for_layer, and save/load
10. AtlasMemoryBank KG2KV triples formatting, build, and Rectangular Attention attach/detach
11. HiKVPStore.prefetch_layer pre-warming the GPU LRU cache for 100% hits
12. Backward compatibility when hikvp_config=None and atlas_bank=None
"""

from __future__ import annotations

import inspect
import os
import sys
import tempfile
import unittest
from typing import Any, Dict, List, Optional, Tuple

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch

from stoikov_kv.atlas_memory import (
    AtlasDocumentChunk,
    AtlasMemoryBank,
    AtlasTriple,
    format_kg_triples,
)
from stoikov_kv.cache import (
    StoikovKVCache,
    create_mock_artifacts,
)
from stoikov_kv.calibration import (
    ROLE_DIFFUSE_LEXICO,
    ROLE_PEAKED_H2O,
    ROLE_RETRIEVAL_FP16,
    FallbackQwen2Config,
    FallbackQwen2ForCausalLM,
    FallbackQwen2Tokenizer,
)
from stoikov_kv.hikvp import (
    GPULRUPageCache,
    HierarchicalKVHeadIndex,
    HiKVPConfig,
    HiKVPStore,
)
from stoikov_kv.matryoshka_spec import matryoshka_speculative_generate


# ============================================================================
# Helpers for signature-resilient API interaction
# ============================================================================


def _lru_put(
    lru: GPULRUPageCache,
    page_id: int,
    k: torch.Tensor,
    v: torch.Tensor,
    pos: torch.Tensor,
) -> Any:
    """Calls GPULRUPageCache.put whether it accepts (page_id, k, v, pos) or (page_id, entry)."""
    sig = inspect.signature(lru.put)
    positional = [
        p
        for p in sig.parameters.values()
        if p.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
    ]
    if len(positional) >= 4:
        return lru.put(page_id, k, v, pos)
    if len(positional) == 3:
        return lru.put(page_id, k, v)
    return lru.put(page_id, (k, v, pos))


def _extract_page_ids(raw: Any) -> List[int]:
    """Normalizes prune_page_ids output (list, tuple(ids, scores), or Tensor) to List[int]."""
    if isinstance(raw, tuple):
        raw = raw[0]
    if isinstance(raw, torch.Tensor):
        return [int(x) for x in raw.reshape(-1).tolist()]
    return [int(x) for x in list(raw)]


def _unpack_retrieve(raw: Any) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Normalizes retrieve() output to (k, v, positions)."""
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


def _get_head_lru(head: HierarchicalKVHeadIndex) -> Optional[GPULRUPageCache]:
    """Finds the GPULRUPageCache attribute attached to a HierarchicalKVHeadIndex."""
    for attr in ("lru_cache", "lru", "_lru_cache", "_lru", "page_cache", "hot_cache"):
        obj = getattr(head, attr, None)
        if obj is not None:
            return obj
    return None


def _get_pcie_bytes_from_telemetry(telemetry: Dict[str, Any]) -> int:
    """Extracts cumulative PCIe bytes transferred from telemetry dict."""
    for key in (
        "pcie_bytes_transferred",
        "total_pcie_bytes",
        "pcie_bytes",
        "h2d_bytes",
    ):
        if key in telemetry and telemetry[key] is not None:
            return int(telemetry[key])
    return 0


def _store_append_head(
    store: HiKVPStore,
    layer_idx: int,
    head_idx: int,
    k_pre: torch.Tensor,
    v: torch.Tensor,
    positions: torch.Tensor,
) -> None:
    """Appends pre-RoPE KV states to HiKVPStore (supports per-head or per-layer signatures)."""
    sig = inspect.signature(store.append_pre_rope)
    if "head_idx" in sig.parameters:
        store.append_pre_rope(layer_idx, head_idx, k_pre, v, positions)
    else:
        head = store.get_head(layer_idx, head_idx)
        assert head is not None
        head.append_pre_rope(k_pre, v, positions)


def _store_retrieve_head(
    store: HiKVPStore,
    layer_idx: int,
    head_idx: int,
    query_pre_rope: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    top_k_pages: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Retrieves KV states for a head via HiKVPStore (supports per-head or per-layer signatures)."""
    sig = inspect.signature(store.retrieve)
    if "head_idx" in sig.parameters:
        res = store.retrieve(
            layer_idx,
            head_idx,
            query_pre_rope,
            device=device,
            dtype=dtype,
            top_k_pages=top_k_pages,
        )
        return _unpack_retrieve(res)
    head = store.get_head(layer_idx, head_idx)
    assert head is not None
    res = head.retrieve(
        query_pre_rope,
        device=device,
        dtype=dtype,
        top_k_pages=top_k_pages,
    )
    return _unpack_retrieve(res)


def _make_hybrid_artifacts(
    model: FallbackQwen2ForCausalLM,
    tokenizer: FallbackQwen2Tokenizer,
    subspace_rank: int = 8,
    dict_atoms: int = 64,
) -> Dict[str, Any]:
    """
    Creates mock artifacts with deterministic hybrid roles across 4 layers:
      - Layer 0: ROLE_RETRIEVAL_FP16 (offloaded by HiKVP)
      - Layer 1: ROLE_PEAKED_H2O
      - Layer 2..3: ROLE_DIFFUSE_LEXICO
    """
    art = create_mock_artifacts(
        model,
        tokenizer=tokenizer,
        subspace_rank=subspace_rank,
        dict_atoms=dict_atoms,
    )
    roles = art["head_roles"].clone()
    num_layers = roles.shape[0]
    roles[0, :] = ROLE_RETRIEVAL_FP16
    if num_layers > 1:
        roles[1, :] = ROLE_PEAKED_H2O
    if num_layers > 2:
        roles[2:, :] = ROLE_DIFFUSE_LEXICO
    art["head_roles"] = roles
    return art


def _orthonormal_basis(head_dim: int, rank: int, seed: int = 123) -> torch.Tensor:
    """Returns a deterministic [head_dim, rank] orthonormal matrix U_Q."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    mat = torch.randn(head_dim, rank, generator=gen, dtype=torch.float32)
    q, _ = torch.linalg.qr(mat, mode="reduced")
    return q[:, :rank].contiguous()


# ============================================================================
# Unit Test Suite
# ============================================================================


class TestHiKVPAndAtlas(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.device = "cpu"
        cls.dtype = torch.float32
        cls.config = FallbackQwen2Config(
            hidden_size=96,
            num_hidden_layers=4,
            num_attention_heads=6,
            num_key_value_heads=2,
            head_dim=16,
            vocab_size=512,
        )
        cls.model = FallbackQwen2ForCausalLM(
            cls.config, device=cls.device, dtype=cls.dtype, seed=42
        )
        cls.model.eval()
        cls.tokenizer = FallbackQwen2Tokenizer(vocab_size=512)
        cls.artifacts = _make_hybrid_artifacts(
            cls.model, cls.tokenizer, subspace_rank=8, dict_atoms=64
        )

    def test_01_lru_page_cache_hit_miss_eviction_and_invalidate(self) -> None:
        """Verifies GPULRUPageCache LRU ordering, eviction count, hit rate, invalidate_ge, and clear."""
        lru = GPULRUPageCache(max_pages=3)
        self.assertEqual(lru.hits, 0)
        self.assertEqual(lru.misses, 0)
        self.assertEqual(lru.evictions, 0)
        self.assertEqual(lru.resident_bytes, 0)
        self.assertEqual(lru.hit_rate, 0.0)

        # Cold miss
        self.assertIsNone(lru.get(0))
        self.assertEqual(lru.misses, 1)
        self.assertEqual(lru.hits, 0)

        def _make_page(pid: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            k = torch.full((16, 16), float(pid), dtype=torch.float32)
            v = torch.full((16, 16), float(pid + 10), dtype=torch.float32)
            pos = torch.arange(pid * 16, (pid + 1) * 16, dtype=torch.long)
            return k, v, pos

        # Insert 3 pages (fills capacity=3, 0 evictions)
        for pid in (0, 1, 2):
            _lru_put(lru, pid, *_make_page(pid))
        self.assertEqual(lru.evictions, 0)
        self.assertGreater(lru.resident_bytes, 0)
        self.assertGreaterEqual(lru.pcie_bytes_transferred, 0)

        # Touch page 0 -> promotes 0 to MRU, so LRU order becomes [1, 2, 0]
        hit0 = lru.get(0)
        self.assertIsNotNone(hit0)
        self.assertEqual(lru.hits, 1)
        self.assertEqual(lru.misses, 1)
        self.assertAlmostEqual(lru.hit_rate, 0.5, places=6)

        # Insert page 3 -> must evict LRU page 1
        _lru_put(lru, 3, *_make_page(3))
        self.assertEqual(lru.evictions, 1)
        self.assertIsNone(lru.get(1))
        self.assertIsNotNone(lru.get(0))
        self.assertIsNotNone(lru.get(2))
        self.assertIsNotNone(lru.get(3))

        # Invalidate pages >= 2 -> removes 2 and 3, preserves page 0
        lru.invalidate_ge(2)
        self.assertIsNone(lru.get(2))
        self.assertIsNone(lru.get(3))
        self.assertIsNotNone(lru.get(0))

        # Clear empties cache and resets resident bytes
        lru.clear()
        self.assertIsNone(lru.get(0))
        self.assertEqual(lru.resident_bytes, 0)

    def test_02_hikvp_3level_tree_needle_retrieval_recall(self) -> None:
        """
        Inserts 2,048 tokens (32 pages x 64) with planted synthetic needles at page 5 (early),
        page 17 (mid), and page 29 (late); verifies 100% needle recall for both full-dim
        and U_Q rank-16 subspace centroids.
        """
        head_dim = 64
        rank = 16
        page_size = 64
        total_tokens = 2048  # 32 pages
        u_q = _orthonormal_basis(head_dim, rank, seed=7)

        gen = torch.Generator(device="cpu").manual_seed(101)
        k_all = 0.05 * torch.randn(total_tokens, head_dim, generator=gen, dtype=torch.float32)
        v_all = 0.05 * torch.randn(total_tokens, head_dim, generator=gen, dtype=torch.float32)
        pos_all = torch.arange(total_tokens, dtype=torch.long)

        # Plant 3 orthogonal needles strictly inside the column space of U_Q
        needle_specs = [
            (5, 5 * page_size + 17, 0),
            (17, 17 * page_size + 31, 1),
            (29, 29 * page_size + 42, 2),
        ]
        needle_queries: List[Tuple[int, int, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for target_page, tok_idx, basis_col in needle_specs:
            direction = u_q[:, basis_col]  # unit vector in span(U_Q)
            k_needle = 12.0 * direction
            v_needle = torch.full((head_dim,), float(target_page) + 0.5, dtype=torch.float32)
            k_all[tok_idx] = k_needle
            v_all[tok_idx] = v_needle
            needle_queries.append(
                (target_page, tok_idx, direction.unsqueeze(0), k_needle, v_needle)
            )

        for use_subspace in (False, True):
            cfg = HiKVPConfig(
                page_size=page_size,
                inter_fanout=4,  # 32 leaves -> 8 inter -> 2 roots (exercises all 3 levels)
                top_k_roots=2,
                top_k_inter=4,
                top_k_pages=4,
                max_hot_pages_per_head=8,
                use_subspace_centroids=use_subspace,
                include_min_max_bounds=True,
                pin_memory=False,
                async_prefetch=False,
            )
            index = HierarchicalKVHeadIndex(
                layer_idx=0,
                head_idx=0,
                head_dim=head_dim,
                config=cfg,
                u_q=u_q if use_subspace else None,
            )

            # Append in 256-token chunks to verify incremental tree growth
            for start in range(0, total_tokens, 256):
                end = start + 256
                index.append_pre_rope(k_all[start:end], v_all[start:end], pos_all[start:end])

            self.assertEqual(index.num_tokens, total_tokens)
            self.assertEqual(index.num_pages, total_tokens // page_size)

            hits = 0
            for target_page, tok_idx, q_vec, k_needle, v_needle in needle_queries:
                pruned_ids = _extract_page_ids(index.prune_page_ids(q_vec, top_k_pages=4))
                self.assertLessEqual(len(pruned_ids), 4)
                self.assertIn(
                    target_page,
                    pruned_ids,
                    f"Missing needle page {target_page} (use_subspace={use_subspace}), got {pruned_ids}",
                )

                k_ret, v_ret, pos_ret = _unpack_retrieve(
                    index.retrieve(
                        q_vec,
                        device=torch.device("cpu"),
                        dtype=torch.float32,
                        top_k_pages=4,
                    )
                )
                self.assertIsNotNone(pos_ret)
                pos_flat = pos_ret.reshape(-1)
                matches = torch.nonzero(pos_flat == tok_idx, as_tuple=False)
                self.assertGreater(matches.numel(), 0)
                m_idx = int(matches[0].item())
                k_flat = k_ret.reshape(-1, head_dim)
                v_flat = v_ret.reshape(-1, head_dim)
                self.assertTrue(torch.allclose(k_flat[m_idx], k_needle, atol=1e-4))
                self.assertTrue(torch.allclose(v_flat[m_idx], v_needle, atol=1e-4))
                hits += 1

            self.assertEqual(hits, len(needle_queries))

    def test_03_hikvp_sublinear_gpu_index_scaling(self) -> None:
        """
        Compares gpu_index_bytes + gpu_lru_bytes vs cpu_leaf_bytes across 512 -> 4,096 -> 16,384
        tokens; asserts active GPU footprint stays strictly sub-linear (< 20% at 16K).
        """
        head_dim = 128
        rank = 16
        u_q = _orthonormal_basis(head_dim, rank, seed=19)
        cfg = HiKVPConfig(
            page_size=64,
            inter_fanout=8,
            top_k_roots=4,
            top_k_inter=8,
            top_k_pages=4,
            max_hot_pages_per_head=8,
            use_subspace_centroids=True,
            include_min_max_bounds=True,
            pin_memory=False,
            async_prefetch=False,
        )
        index = HierarchicalKVHeadIndex(
            layer_idx=0,
            head_idx=0,
            head_dim=head_dim,
            config=cfg,
            u_q=u_q,
        )

        gen = torch.Generator(device="cpu").manual_seed(202)
        checkpoints = (512, 4096, 16384)
        ratios: List[float] = []
        cpu_bytes_list: List[int] = []
        current = 0

        for target_len in checkpoints:
            delta = target_len - current
            k_chunk = torch.randn(delta, head_dim, generator=gen, dtype=torch.float16)
            v_chunk = torch.randn(delta, head_dim, generator=gen, dtype=torch.float16)
            pos_chunk = torch.arange(current, target_len, dtype=torch.long)
            index.append_pre_rope(k_chunk, v_chunk, pos_chunk)
            current = target_len

            # Trigger retrieval so LRU cache is actively populated
            q = torch.randn(1, head_dim, generator=gen, dtype=torch.float16)
            _ = index.retrieve(q, device=torch.device("cpu"), dtype=torch.float16, top_k_pages=4)

            gpu_idx = int(index.gpu_index_bytes())
            gpu_lru = int(index.gpu_lru_bytes())
            cpu_leaf = int(index.cpu_leaf_bytes())

            self.assertGreater(gpu_idx, 0)
            self.assertGreater(gpu_lru, 0)
            self.assertGreater(cpu_leaf, 0)

            ratio = float(gpu_idx + gpu_lru) / float(cpu_leaf)
            ratios.append(ratio)
            cpu_bytes_list.append(cpu_leaf)

        # CPU leaf storage scales linearly (~32x from 512 to 16,384)
        self.assertGreater(cpu_bytes_list[-1] / float(cpu_bytes_list[0]), 25.0)
        # GPU/CPU ratio drops sharply as sequence grows and is strictly < 20% at 16K
        self.assertLess(ratios[1], ratios[0])
        self.assertLess(ratios[2], ratios[1])
        self.assertLess(
            ratios[2],
            0.20,
            f"Expected GPU bytes < 20% of CPU leaves at 16K, got {ratios[2]:.3%}",
        )

    def test_04_hikvp_exact_crop_rollback(self) -> None:
        """
        Appends 350 tokens (5 full pages + 30 tail), crops back to 270 (mid-page),
        128 (exact page boundary), and 0; verifies counts and exact tensor reconstruction.
        """
        head_dim = 32
        page_size = 64
        total = 350
        cfg = HiKVPConfig(
            page_size=page_size,
            inter_fanout=4,
            top_k_roots=4,
            top_k_inter=8,
            top_k_pages=16,
            max_hot_pages_per_head=16,
            use_subspace_centroids=False,
            pin_memory=False,
            async_prefetch=False,
        )
        index = HierarchicalKVHeadIndex(
            layer_idx=0,
            head_idx=0,
            head_dim=head_dim,
            config=cfg,
        )

        gen = torch.Generator(device="cpu").manual_seed(303)
        k_all = torch.randn(total, head_dim, generator=gen, dtype=torch.float32)
        v_all = torch.randn(total, head_dim, generator=gen, dtype=torch.float32)
        pos_all = torch.arange(total, dtype=torch.long)

        index.append_pre_rope(k_all, v_all, pos_all)
        self.assertEqual(index.num_tokens, 350)
        self.assertIn(index.num_pages, (5, 6))

        q = torch.ones(1, head_dim, dtype=torch.float32)

        def _assert_exact_prefix(expected_len: int) -> None:
            self.assertEqual(index.num_tokens, expected_len)
            k_ret, v_ret, pos_ret = _unpack_retrieve(
                index.retrieve(
                    q,
                    device=torch.device("cpu"),
                    dtype=torch.float32,
                    top_k_pages=16,
                )
            )
            k_flat = k_ret.reshape(-1, head_dim)
            v_flat = v_ret.reshape(-1, head_dim)
            self.assertEqual(k_flat.shape[0], expected_len)
            if expected_len > 0:
                self.assertIsNotNone(pos_ret)
                pos_flat = pos_ret.reshape(-1)
                order = torch.argsort(pos_flat)
                self.assertTrue(torch.equal(pos_flat[order], pos_all[:expected_len]))
                self.assertTrue(torch.allclose(k_flat[order], k_all[:expected_len], atol=1e-5))
                self.assertTrue(torch.allclose(v_flat[order], v_all[:expected_len], atol=1e-5))

        _assert_exact_prefix(350)

        # Mid-page rollback: 270 = 4 * 64 + 14
        index.crop(270)
        self.assertIn(index.num_pages, (4, 5))
        _assert_exact_prefix(270)

        # Exact page-boundary rollback: 128 = 2 * 64
        index.crop(128)
        self.assertEqual(index.num_pages, 2)
        _assert_exact_prefix(128)

        # Full reset rollback: 0
        index.crop(0)
        self.assertEqual(index.num_pages, 0)
        _assert_exact_prefix(0)

    def test_05_hikvp_draft_zero_pcie_and_lru_temporal_locality(self) -> None:
        """
        Verifies draft_zero_pcie=True records strictly 0 PCIe bytes even on cold pages,
        and consecutive identical normal queries achieve 100% LRU hit rate (0 extra PCIe bytes).
        """
        head_dim = 32
        page_size = 64
        cfg = HiKVPConfig(
            page_size=page_size,
            inter_fanout=4,
            top_k_roots=4,
            top_k_inter=4,
            top_k_pages=2,
            max_hot_pages_per_head=4,
            use_subspace_centroids=False,
            pin_memory=False,
            async_prefetch=False,
            draft_zero_pcie=True,
        )
        index = HierarchicalKVHeadIndex(
            layer_idx=0,
            head_idx=0,
            head_dim=head_dim,
            config=cfg,
        )
        gen = torch.Generator(device="cpu").manual_seed(404)
        k_all = torch.randn(512, head_dim, generator=gen, dtype=torch.float32)
        v_all = torch.randn(512, head_dim, generator=gen, dtype=torch.float32)
        index.append_pre_rope(k_all, v_all, torch.arange(512, dtype=torch.long))

        lru = _get_head_lru(index)
        self.assertIsNotNone(lru, "Expected HierarchicalKVHeadIndex to expose its GPULRUPageCache")
        assert lru is not None

        q = torch.randn(1, head_dim, generator=gen, dtype=torch.float32)

        # 1. Cold query in draft_zero_pcie=True mode must transfer strictly 0 PCIe bytes
        _ = index.retrieve(
            q,
            device=torch.device("cpu"),
            dtype=torch.float32,
            top_k_pages=2,
            draft_zero_pcie=True,
        )
        self.assertEqual(lru.pcie_bytes_transferred, 0)

        # 2. Normal query loads the top-2 cold pages into LRU
        k1, v1, _ = _unpack_retrieve(
            index.retrieve(
                q,
                device=torch.device("cpu"),
                dtype=torch.float32,
                top_k_pages=2,
                draft_zero_pcie=False,
            )
        )
        self.assertGreater(k1.reshape(-1, head_dim).shape[0], 0)
        pcie_after_cold = int(lru.pcie_bytes_transferred)
        self.assertGreater(pcie_after_cold, 0)
        hits_before = int(lru.hits)
        misses_before = int(lru.misses)

        # 3. Three repeated identical queries must achieve 100% LRU hits and 0 extra PCIe bytes
        for _ in range(3):
            k_warm, v_warm, _ = _unpack_retrieve(
                index.retrieve(
                    q,
                    device=torch.device("cpu"),
                    dtype=torch.float32,
                    top_k_pages=2,
                    draft_zero_pcie=False,
                )
            )
            self.assertTrue(torch.allclose(k_warm, k1, atol=1e-6))
            self.assertTrue(torch.allclose(v_warm, v1, atol=1e-6))

        self.assertEqual(int(lru.pcie_bytes_transferred), pcie_after_cold)
        self.assertEqual(int(lru.misses), misses_before)
        self.assertGreaterEqual(int(lru.hits) - hits_before, 3 * 2)

    def test_06_hikvp_mmap_nvme_backing_and_save_load(self) -> None:
        """
        Tests mmap_dir disk-backed leaf pages AND HiKVPStore.save()/load() +
        HierarchicalKVHeadIndex.state_dict()/from_state_dict() round-trip fidelity.
        """
        head_dim = 16
        rank = 8
        u_q = torch.stack(
            [
                torch.stack([_orthonormal_basis(head_dim, rank, seed=500 + l * 10 + h) for h in range(2)])
                for l in range(2)
            ]
        )  # [2, 2, 16, 8]
        head_roles = torch.tensor(
            [
                [ROLE_RETRIEVAL_FP16, ROLE_RETRIEVAL_FP16],
                [ROLE_DIFFUSE_LEXICO, ROLE_DIFFUSE_LEXICO],
            ],
            dtype=torch.long,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            mmap_dir = os.path.join(tmpdir, "nvme_pages")
            save_dir = os.path.join(tmpdir, "store_ckpt")

            cfg = HiKVPConfig(
                page_size=32,
                inter_fanout=4,
                top_k_roots=2,
                top_k_inter=4,
                top_k_pages=2,
                max_hot_pages_per_head=4,
                use_subspace_centroids=True,
                pin_memory=False,
                async_prefetch=False,
                mmap_dir=mmap_dir,
                offload_roles=(ROLE_RETRIEVAL_FP16,),
            )
            store = HiKVPStore(
                num_layers=2,
                num_kv_heads=2,
                head_dim=head_dim,
                head_roles=head_roles,
                config=cfg,
                U_Q=u_q,
            )
            self.assertTrue(store.is_offloaded(0, 0))
            self.assertTrue(store.is_offloaded(0, 1))
            self.assertFalse(store.is_offloaded(1, 0))
            self.assertFalse(store.is_offloaded(1, 1))

            gen = torch.Generator(device="cpu").manual_seed(606)
            for h in range(2):
                k_h = torch.randn(128, head_dim, generator=gen, dtype=torch.float32)
                v_h = torch.randn(128, head_dim, generator=gen, dtype=torch.float32)
                pos_h = torch.arange(128, dtype=torch.long)
                _store_append_head(store, 0, h, k_h, v_h, pos_h)

            # Verify mmap backing files were created on disk
            self.assertTrue(os.path.isdir(mmap_dir))
            mmap_files = []
            for _root, _dirs, files in os.walk(mmap_dir):
                mmap_files.extend(files)
            self.assertGreater(len(mmap_files), 0)

            q = torch.randn(1, head_dim, generator=gen, dtype=torch.float32)
            k_orig, v_orig, pos_orig = _store_retrieve_head(
                store, 0, 0, q, device=torch.device("cpu"), dtype=torch.float32, top_k_pages=2
            )

            # Test HeadIndex state_dict / from_state_dict round-trip directly
            head0 = store.get_head(0, 0)
            self.assertIsNotNone(head0)
            assert head0 is not None
            sd = head0.state_dict()
            rebuilt_head = HierarchicalKVHeadIndex.from_state_dict(sd, u_q=u_q[0, 0])
            self.assertEqual(rebuilt_head.num_tokens, head0.num_tokens)
            self.assertEqual(rebuilt_head.num_pages, head0.num_pages)
            k_sd, v_sd, _ = _unpack_retrieve(
                rebuilt_head.retrieve(q, device=torch.device("cpu"), dtype=torch.float32, top_k_pages=2)
            )
            self.assertTrue(torch.allclose(k_sd, k_orig, atol=1e-5))
            self.assertTrue(torch.allclose(v_sd, v_orig, atol=1e-5))

            # Test full HiKVPStore save / load round-trip
            store.save(save_dir)
            load_sig = inspect.signature(HiKVPStore.load)
            if "U_Q" in load_sig.parameters:
                loaded = HiKVPStore.load(save_dir, U_Q=u_q)
            else:
                loaded = HiKVPStore.load(save_dir)

            self.assertTrue(loaded.is_offloaded(0, 0))
            self.assertFalse(loaded.is_offloaded(1, 0))
            loaded_head0 = loaded.get_head(0, 0)
            self.assertIsNotNone(loaded_head0)
            assert loaded_head0 is not None
            self.assertEqual(loaded_head0.num_tokens, 128)

            k_ld, v_ld, pos_ld = _store_retrieve_head(
                loaded, 0, 0, q, device=torch.device("cpu"), dtype=torch.float32, top_k_pages=2
            )
            self.assertTrue(torch.allclose(k_ld, k_orig, atol=1e-5))
            self.assertTrue(torch.allclose(v_ld, v_orig, atol=1e-5))
            if pos_orig is not None and pos_ld is not None:
                self.assertTrue(torch.equal(pos_ld, pos_orig))

    def test_07_stoikov_cache_with_hikvp_end_to_end_forward_and_crop(self) -> None:
        """
        Runs FallbackQwen2ForCausalLM with StoikovKVCache + HiKVPConfig + query hooks over
        prefill + 8 decode steps + crop(), verifying finite logits, output_attentions=False,
        sub-linear GPU telemetry, and clean rollback.
        """
        hikvp_cfg = HiKVPConfig(
            page_size=16,
            inter_fanout=4,
            top_k_roots=2,
            top_k_inter=4,
            top_k_pages=2,
            max_hot_pages_per_head=4,
            use_subspace_centroids=True,
            pin_memory=False,
            async_prefetch=False,
            offload_roles=(ROLE_RETRIEVAL_FP16,),
        )
        cache = StoikovKVCache(
            model=self.model,
            artifacts=self.artifacts,
            n_b=16,
            keep_ratio=0.25,
            s_budget=8,
            policy_mode="hybrid",
            hikvp_config=hikvp_cfg,
        )
        self.assertIsNotNone(cache.hikvp_store)
        cache.attach_query_hooks(self.model)
        try:
            gen = torch.Generator(device="cpu").manual_seed(707)
            prompt = torch.randint(4, 250, (1, 128), generator=gen, device=self.device)
            out = self.model(
                input_ids=prompt,
                past_key_values=cache,
                use_cache=True,
                output_attentions=False,
            )
            self.assertIsNone(out.attentions)
            self.assertTrue(torch.isfinite(out.logits).all())
            self.assertEqual(cache.get_seq_length(0), 128)

            next_tok = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
            for step in range(8):
                step_out = self.model(
                    input_ids=next_tok,
                    past_key_values=cache,
                    use_cache=True,
                    output_attentions=False,
                )
                self.assertIsNone(step_out.attentions)
                self.assertTrue(torch.isfinite(step_out.logits).all())
                self.assertEqual(cache.get_seq_length(0), 128 + step + 1)
                next_tok = torch.argmax(step_out.logits[:, -1, :], dim=-1, keepdim=True)

            tel = cache.hikvp_telemetry()
            self.assertIsInstance(tel, dict)
            self.assertGreater(len(tel), 0)
            self.assertGreater(int(tel.get("cpu_leaf_bytes", 0)), 0)
            self.assertGreater(int(tel.get("gpu_index_bytes", 0)), 0)

            # Roll back across page boundaries and continue decoding cleanly
            cache.crop(100)
            self.assertEqual(cache.get_seq_length(0), 100)
            post_crop_out = self.model(
                input_ids=next_tok,
                past_key_values=cache,
                use_cache=True,
                output_attentions=False,
            )
            self.assertTrue(torch.isfinite(post_crop_out.logits).all())
            self.assertEqual(cache.get_seq_length(0), 101)
        finally:
            cache.detach_query_hooks()

    def test_08_matryoshka_speculative_with_hikvp_strict_greedy_parity(self) -> None:
        """
        Runs matryoshka_speculative_generate with HiKVP enabled (do_sample=False,
        strict_rejection_sampling=True) and verifies 100% lossless greedy token parity
        against sequential decode AND strictly 0 PCIe bytes transferred during draft mode.
        """
        gen = torch.Generator(device="cpu").manual_seed(808)
        prompt_ids = torch.randint(4, 250, (1, 48), generator=gen, device=self.device)
        max_new_tokens = 10

        hikvp_cfg = HiKVPConfig(
            page_size=16,
            inter_fanout=4,
            top_k_roots=4,
            top_k_inter=4,
            top_k_pages=4,
            max_hot_pages_per_head=2,  # Smaller than offloaded pages to ensure cold evictions
            use_subspace_centroids=True,
            pin_memory=False,
            async_prefetch=False,
            offload_roles=(ROLE_RETRIEVAL_FP16,),
            draft_zero_pcie=True,
        )

        # 1. Sequential greedy baseline with identical verify_atoms=8 and HiKVP config
        seq_cache = StoikovKVCache(
            model=self.model,
            artifacts=self.artifacts,
            n_b=8,
            keep_ratio=0.25,
            s_budget=8,
            policy_mode="hybrid",
            hikvp_config=hikvp_cfg,
        )
        seq_cache.attach_query_hooks(self.model)
        seq_generated: List[int] = []
        try:
            seq_cache.set_matryoshka_mode(active_atoms=8)
            seq_cache.set_hikvp_draft_mode(False)
            with torch.no_grad():
                out = self.model(
                    input_ids=prompt_ids,
                    past_key_values=seq_cache,
                    use_cache=True,
                    output_attentions=False,
                )
                tok = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
                seq_generated.append(int(tok.item()))
                for _ in range(max_new_tokens - 1):
                    out = self.model(
                        input_ids=tok,
                        past_key_values=seq_cache,
                        use_cache=True,
                        output_attentions=False,
                    )
                    tok = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
                    seq_generated.append(int(tok.item()))
        finally:
            seq_cache.detach_query_hooks()

        # 2. Matryoshka speculative decoding with HiKVP + draft PCIe tracking
        spec_cache = StoikovKVCache(
            model=self.model,
            artifacts=self.artifacts,
            n_b=8,
            keep_ratio=0.25,
            s_budget=8,
            policy_mode="hybrid",
            hikvp_config=hikvp_cfg,
        )
        draft_entries = 0
        draft_pcie_delta = 0
        in_draft = False
        pcie_at_enter = 0
        orig_set_draft = spec_cache.set_hikvp_draft_mode

        def _tracked_set_draft(enabled: bool) -> Any:
            nonlocal draft_entries, draft_pcie_delta, in_draft, pcie_at_enter
            cur_pcie = _get_pcie_bytes_from_telemetry(spec_cache.hikvp_telemetry())
            if enabled and not in_draft:
                draft_entries += 1
                pcie_at_enter = cur_pcie
                in_draft = True
            elif not enabled and in_draft:
                draft_pcie_delta += cur_pcie - pcie_at_enter
                in_draft = False
            return orig_set_draft(enabled)

        spec_cache.set_hikvp_draft_mode = _tracked_set_draft  # type: ignore[method-assign]
        spec_cache.attach_query_hooks(self.model)
        try:
            res = matryoshka_speculative_generate(
                model=self.model,
                tokenizer=self.tokenizer,
                prompt=prompt_ids,
                cache=spec_cache,
                gamma_spec=3,
                draft_atoms=2,
                verify_atoms=8,
                max_new_tokens=max_new_tokens,
                strict_rejection_sampling=True,
                do_sample=False,
                seed=42,
            )
        finally:
            spec_cache.detach_query_hooks()

        self.assertEqual(len(res["generated_ids"]), max_new_tokens)
        self.assertEqual(
            list(res["generated_ids"]),
            seq_generated,
            "Speculative decode with HiKVP must match sequential greedy tokens 100%",
        )
        self.assertGreater(draft_entries, 0, "Expected draft mode to be enabled during speculation")
        self.assertEqual(
            draft_pcie_delta,
            0,
            f"Expected strictly 0 PCIe bytes transferred during draft mode, got {draft_pcie_delta}",
        )
        tel = spec_cache.hikvp_telemetry()
        for key in ("draft_pcie_bytes", "draft_pcie_bytes_transferred"):
            if key in tel:
                self.assertEqual(int(tel[key]), 0)

    def test_09_atlas_memory_bank_doc2kv_build_save_load(self) -> None:
        """
        Builds AtlasMemoryBank from multi-document texts (Doc2KV), checks stats() and
        retrieve_for_layer(), saves to disk, reloads, and verifies fidelity.
        """
        docs = [
            (
                "Architecture specification for AtlasKV offloaded storage. "
                "The primary coordinator listens on API_PORT = 8443 and manages "
                "three-level hierarchical centroids for fast page pruning across CPU RAM. "
            )
            * 3,
            (
                "Telemetry and distributed consensus protocol notes. "
                "Worker cluster alpha checkpoints every 64 tokens into immutable pages "
                "with min-max bounding boxes and rank-16 subspace centroids. "
            )
            * 3,
            (
                "Security and passkey verification guide. "
                "The vault recovery passkey is 94812 and rotates only after full "
                "quorum acknowledgement from all retrieval guard heads. "
            )
            * 3,
        ]
        sources = ["architecture.md", "consensus.md", "security.md"]
        hikvp_cfg = HiKVPConfig(
            page_size=16,
            inter_fanout=4,
            top_k_roots=2,
            top_k_inter=4,
            top_k_pages=2,
            max_hot_pages_per_head=4,
            use_subspace_centroids=True,
            pin_memory=False,
            async_prefetch=False,
        )

        bank = AtlasMemoryBank.build_from_texts(
            model=self.model,
            tokenizer=self.tokenizer,
            texts=docs,
            artifacts=self.artifacts,
            hikvp_config=hikvp_cfg,
            sources=sources,
            chunk_tokens=48,
            all_heads=True,
        )

        stats = bank.stats()
        self.assertIsInstance(stats, dict)
        self.assertGreater(len(stats), 0)
        num_chunks = stats.get("num_chunks", stats.get("chunks", 0))
        num_tokens = stats.get("num_tokens", stats.get("total_tokens", 0))
        self.assertGreaterEqual(int(num_chunks), 3)
        self.assertGreater(int(num_tokens), 0)

        # Verify retrieve_for_layer works with GQA query [B, H_q, Q, D]
        gen = torch.Generator(device="cpu").manual_seed(909)
        q_layer = torch.randn(
            1,
            self.config.num_attention_heads,
            1,
            self.config.head_dim,
            generator=gen,
            dtype=self.dtype,
            device=self.device,
        )
        k_ret, v_ret, _ = _unpack_retrieve(
            bank.retrieve_for_layer(
                0,
                q_layer,
                device=torch.device(self.device),
                dtype=self.dtype,
                top_k_pages=2,
            )
        )
        self.assertTrue(torch.isfinite(k_ret).all())
        self.assertTrue(torch.isfinite(v_ret).all())
        self.assertGreater(k_ret.numel(), 0)
        self.assertEqual(k_ret.shape, v_ret.shape)

        # Verify save() and load() round-trip
        with tempfile.TemporaryDirectory() as tmpdir:
            bank.save(tmpdir)
            reloaded = AtlasMemoryBank.load(tmpdir, U_Q=self.artifacts["U_Q"])
            reloaded_stats = reloaded.stats()
            self.assertEqual(
                int(reloaded_stats.get("num_chunks", reloaded_stats.get("chunks", -1))),
                int(num_chunks),
            )
            self.assertEqual(
                int(reloaded_stats.get("num_tokens", reloaded_stats.get("total_tokens", -1))),
                int(num_tokens),
            )
            k_ld, v_ld, _ = _unpack_retrieve(
                reloaded.retrieve_for_layer(
                    0,
                    q_layer,
                    device=torch.device(self.device),
                    dtype=self.dtype,
                    top_k_pages=2,
                )
            )
            self.assertTrue(torch.allclose(k_ld, k_ret, atol=1e-4))
            self.assertTrue(torch.allclose(v_ld, v_ret, atol=1e-4))

    def test_10_atlas_memory_bank_kg2kv_triples_and_rectangular_attention(self) -> None:
        """
        Builds AtlasMemoryBank from 40 structured AtlasTriple facts (KG2KV), attaches via
        Rectangular Attention with zero re-prefill, runs a short query, and verifies finite
        logits and non-zero atlas retrieval telemetry.
        """
        triples: List[AtlasTriple] = [
            AtlasTriple(
                subject=f"ServiceNode_{i:02d}",
                relation="uses_protocol" if i % 2 == 0 else "deployed_in_region",
                obj=f"gRPC_v{(i % 4) + 1}" if i % 2 == 0 else f"us_west_{(i % 3) + 1}",
            )
            for i in range(40)
        ]

        formatted = format_kg_triples(triples[:4])
        formatted_str = "\n".join(formatted) if isinstance(formatted, list) else str(formatted)
        self.assertIn("ServiceNode_00", formatted_str)
        self.assertIn("uses_protocol", formatted_str)
        self.assertIn("gRPC_v1", formatted_str)

        hikvp_cfg = HiKVPConfig(
            page_size=16,
            inter_fanout=4,
            top_k_roots=2,
            top_k_inter=4,
            top_k_pages=2,
            max_hot_pages_per_head=4,
            use_subspace_centroids=True,
            pin_memory=False,
            async_prefetch=False,
        )
        bank = AtlasMemoryBank.build_from_kg_triples(
            model=self.model,
            tokenizer=self.tokenizer,
            triples=triples,
            artifacts=self.artifacts,
            hikvp_config=hikvp_cfg,
            triples_per_chunk=10,
            all_heads=True,
        )
        kg_stats = bank.stats()
        self.assertGreaterEqual(int(kg_stats.get("num_chunks", kg_stats.get("chunks", 0))), 4)

        # Fresh cache with NO re-prefill of the 40 KG triples
        cache = StoikovKVCache(
            model=self.model,
            artifacts=self.artifacts,
            n_b=16,
            keep_ratio=0.25,
            s_budget=8,
            policy_mode="hybrid",
        )
        bank.attach_to_cache(cache)
        cache.attach_query_hooks(self.model)
        try:
            enc = self.tokenizer(
                "Which protocol does ServiceNode_00 use in production?",
                return_tensors="pt",
            )
            q_ids = (enc["input_ids"] if isinstance(enc, dict) else enc.input_ids).to(self.device)
            q_len = int(q_ids.shape[1])

            out = self.model(
                input_ids=q_ids,
                past_key_values=cache,
                use_cache=True,
                output_attentions=False,
            )
            self.assertIsNone(out.attentions)
            self.assertTrue(torch.isfinite(out.logits).all())
            # Local cache sequence length contains ONLY the short query (zero KG re-prefill!)
            self.assertEqual(cache.get_seq_length(0), q_len)

            next_tok = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
            for step in range(3):
                step_out = self.model(
                    input_ids=next_tok,
                    past_key_values=cache,
                    use_cache=True,
                    output_attentions=False,
                )
                self.assertTrue(torch.isfinite(step_out.logits).all())
                self.assertEqual(cache.get_seq_length(0), q_len + step + 1)
                next_tok = torch.argmax(step_out.logits[:, -1, :], dim=-1, keepdim=True)

            tel = cache.hikvp_telemetry()
            bank_stats_after = bank.stats()
            atlas_activity = (
                int(tel.get("atlas_retrievals", 0))
                + int(tel.get("atlas_tokens_retrieved", 0))
                + int(tel.get("atlas_queries", 0))
                + int(tel.get("atlas_pcie_bytes", 0))
                + int(bank_stats_after.get("retrievals", 0))
                + int(bank_stats_after.get("pcie_bytes_transferred", 0))
                + _get_pcie_bytes_from_telemetry(tel)
            )
            self.assertGreater(
                atlas_activity,
                0,
                f"Expected non-zero atlas retrieval telemetry; tel={tel}, stats={bank_stats_after}",
            )
        finally:
            bank.detach_from_cache(cache)
            cache.detach_query_hooks()

        # Verify clean operation after detachment
        post_out = self.model(
            input_ids=next_tok,
            past_key_values=cache,
            use_cache=True,
            output_attentions=False,
        )
        self.assertTrue(torch.isfinite(post_out.logits).all())

    def test_11_hikvp_prefetch_layer_populates_lru(self) -> None:
        """
        Tests HiKVPStore.prefetch_layer(...) pre-warms the GPU LRU cache so the immediately
        following retrieve(...) achieves 100% LRU hits with zero additional PCIe fetches.
        """
        head_dim = 32
        num_kv_heads = 2
        head_roles = torch.tensor(
            [
                [ROLE_RETRIEVAL_FP16, ROLE_RETRIEVAL_FP16],
                [ROLE_DIFFUSE_LEXICO, ROLE_DIFFUSE_LEXICO],
            ],
            dtype=torch.long,
        )
        cfg = HiKVPConfig(
            page_size=32,
            inter_fanout=4,
            top_k_roots=2,
            top_k_inter=4,
            top_k_pages=2,
            max_hot_pages_per_head=4,
            use_subspace_centroids=False,
            pin_memory=False,
            async_prefetch=True,
            offload_roles=(ROLE_RETRIEVAL_FP16,),
        )
        store = HiKVPStore(
            num_layers=2,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            head_roles=head_roles,
            config=cfg,
        )

        gen = torch.Generator(device="cpu").manual_seed(1111)
        for h in range(num_kv_heads):
            k_h = torch.randn(256, head_dim, generator=gen, dtype=torch.float32)
            v_h = torch.randn(256, head_dim, generator=gen, dtype=torch.float32)
            _store_append_head(store, 0, h, k_h, v_h, torch.arange(256, dtype=torch.long))

        # Shape [1, num_kv_heads, 1, head_dim] works for both GQA (group=1) and per-kv-head queries
        q_layer = torch.randn(1, num_kv_heads, 1, head_dim, generator=gen, dtype=torch.float32)

        # Prefetch layer 0 ahead of time
        store.prefetch_layer(
            0,
            q_layer,
            device=torch.device("cpu"),
            dtype=torch.float32,
            top_k_pages=2,
        )

        heads = [store.get_head(0, h) for h in range(num_kv_heads)]
        lrus = [_get_head_lru(hd) for hd in heads if hd is not None]
        self.assertEqual(len(lrus), num_kv_heads)

        hits_before = sum(int(l.hits) for l in lrus if l is not None)
        misses_before = sum(int(l.misses) for l in lrus if l is not None)
        pcie_before = sum(int(l.pcie_bytes_transferred) for l in lrus if l is not None)

        # Now retrieve for each head with the exact same query -> must be 100% LRU hits
        for h in range(num_kv_heads):
            q_h = q_layer[0, h]  # [1, head_dim]
            k_ret, _, _ = _store_retrieve_head(
                store,
                0,
                h,
                q_h,
                device=torch.device("cpu"),
                dtype=torch.float32,
                top_k_pages=2,
            )
            self.assertGreater(k_ret.numel(), 0)

        hits_after = sum(int(l.hits) for l in lrus if l is not None)
        misses_after = sum(int(l.misses) for l in lrus if l is not None)
        pcie_after = sum(int(l.pcie_bytes_transferred) for l in lrus if l is not None)

        self.assertEqual(misses_after, misses_before)
        self.assertEqual(pcie_after, pcie_before)
        self.assertGreaterEqual(hits_after - hits_before, num_kv_heads * 2)

    def test_12_backward_compatibility_without_hikvp(self) -> None:
        """
        Verifies StoikovKVCache with default hikvp_config=None and atlas_bank=None
        leaves cache.hikvp_store as None and behaves identically to standard StoikovKV.
        """
        cache = StoikovKVCache(
            model=self.model,
            artifacts=self.artifacts,
            n_b=16,
            keep_ratio=0.25,
            s_budget=8,
            policy_mode="hybrid",
        )
        self.assertIsNone(cache.hikvp_store)

        # Draft mode toggle and telemetry must be safe no-ops when HiKVP is disabled
        cache.set_hikvp_draft_mode(True)
        cache.set_hikvp_draft_mode(False)
        tel = cache.hikvp_telemetry()
        self.assertIsInstance(tel, dict)

        gen = torch.Generator(device="cpu").manual_seed(1212)
        input_ids = torch.randint(4, 250, (1, 48), generator=gen, device=self.device)
        out = self.model(
            input_ids=input_ids,
            past_key_values=cache,
            use_cache=True,
            output_attentions=False,
        )
        self.assertIsNone(out.attentions)
        self.assertTrue(torch.isfinite(out.logits).all())
        self.assertEqual(cache.get_seq_length(0), 48)
        self.assertLess(cache.compressed_bytes, cache.fp16_bytes)

        # Decode 4 tokens + crop
        tok = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
        for _ in range(4):
            out = self.model(
                input_ids=tok,
                past_key_values=cache,
                use_cache=True,
                output_attentions=False,
            )
            self.assertTrue(torch.isfinite(out.logits).all())
            tok = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)

        self.assertEqual(cache.get_seq_length(0), 52)
        cache.crop(32)
        self.assertEqual(cache.get_seq_length(0), 32)
        self.assertIsNone(cache.hikvp_store)


if __name__ == "__main__":
    unittest.main()
