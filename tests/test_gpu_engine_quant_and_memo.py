"""
Unit tests for StoikovKV General GPU Kernel Engine, Quant Microstructure Upgrades
(Hawkes Arrival + VPIN Toxicity + GLFT/Almgren-Chriss), MeMo Bidirectional KG2KV
Consolidation, and Multimodal/Looped `text_config` Architecture Discovery.
"""

from __future__ import annotations

import os
import sys
import types
import unittest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
import torch.nn as nn
import torch.nn.functional as F

from stoikov_kv import (
    AtlasMemoryBank,
    AtlasTriple,
    ElasticLexicoCodec,
    GPUHardwareProfile,
    OnlineRegimeChangeController,
    StoikovKVCache,
    can_use_triton,
    consolidate_kg_triples,
    create_mock_artifacts,
    detect_gpu_profile,
    format_kg_triples,
    select_kernel_config,
    simulate_hardware_profile,
    triton_fused_kv_lexico_materialize,
    triton_fused_write_probe,
)
from stoikov_kv.calibration import (
    FallbackQwen2Attention,
    FallbackQwen2Config,
    FallbackQwen2ForCausalLM,
    FallbackQwen2Tokenizer,
    _inspect_model_architecture,
)
from stoikov_kv.triton_kernels import _pytorch_write_probe


class TestGPUEngineQuantAndMeMo(unittest.TestCase):
    def test_01_hardware_profile_detection_and_all_8_presets(self) -> None:
        live = detect_gpu_profile()
        self.assertIsInstance(live, GPUHardwareProfile)
        self.assertIn(live.backend, ("cuda", "hip", "xpu", "mps", "cpu"))

        presets = [
            "rtx4060_ada",
            "a100_ampere",
            "h100_hopper",
            "b200_blackwell",
            "rtx5090_blackwell",
            "mi300x_rocm",
            "intel_xpu",
            "cpu_fallback",
        ]
        for p in presets:
            prof = simulate_hardware_profile(p)
            cfg = select_kernel_config(prof, op="fused_write_probe", n_vecs=4096, head_dim=128)
            if p == "cpu_fallback":
                self.assertFalse(cfg["use_triton"])
                continue
            self.assertTrue(cfg["use_triton"])
            self.assertLessEqual(cfg["est_sram_bytes"], cfg["sram_budget_bytes"])
            # Backend kwargs must be strictly isolated
            if prof.backend != "hip":
                self.assertNotIn("waves_per_eu", cfg["launch_kwargs"])
            else:
                self.assertIn("waves_per_eu", cfg["launch_kwargs"])
            if prof.backend != "xpu":
                self.assertNotIn("grf_mode", cfg["launch_kwargs"])
            else:
                self.assertIn("grf_mode", cfg["launch_kwargs"])

        # Datacenter Blackwell sm_100 has TMEM/tcgen05; Consumer Blackwell sm_120 does NOT
        b200 = simulate_hardware_profile("b200_blackwell")
        rtx5090 = simulate_hardware_profile("rtx5090_blackwell")
        self.assertTrue(b200.has_tmem_tcgen05)
        self.assertFalse(rtx5090.has_tmem_tcgen05)
        self.assertTrue(b200.has_tma and rtx5090.has_tma)
        # Unaligned pointers must disable tma_aligned
        cfg_unaligned = select_kernel_config(b200, op="lexico_reconstruct", ptr_aligned_16b=False)
        self.assertFalse(cfg_unaligned["tma_aligned"])

    def test_02_triton_fused_kv_lexico_materialize_parity_and_strided_views(self) -> None:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if dev.type == "cuda" else torch.float32
        torch.manual_seed(17)

        d = torch.randn(1, 128, 128, device=dev, dtype=dtype)
        codec = ElasticLexicoCodec(D=d)
        k_in = torch.randn(1, 3, 96, 128, device=dev, dtype=dtype)
        v_in = torch.randn(1, 3, 96, 128, device=dev, dtype=dtype)
        loc_k = torch.randn(1, 3, 32, 128, device=dev, dtype=dtype)
        loc_v = torch.randn(1, 3, 32, 128, device=dev, dtype=dtype)

        k_idx, k_val = codec.encode(k_in, layer_idx=0, s=16)
        v_idx, v_val = codec.encode(v_in, layer_idx=0, s=16)
        freqs = torch.outer(
            torch.arange(96, device=dev, dtype=torch.float32),
            torch.linspace(0.01, 1.0, 64, device=dev),
        )
        c_h, s_h = freqs.cos().contiguous(), freqs.sin().contiguous()

        # Test writing directly into strided slice views [:, 1:4, :, :] of a 4-head mixed layer
        full_k = torch.zeros((1, 4, 128, 128), device=dev, dtype=dtype)
        full_v = torch.zeros((1, 4, 128, 128), device=dev, dtype=dtype)
        triton_fused_kv_lexico_materialize(
            k_idx,
            k_val,
            None,
            v_idx,
            v_val,
            None,
            codec._D_T_fp32[0],
            local_k=loc_k,
            local_v=loc_v,
            active_atoms=12,
            rope_cos=c_h,
            rope_sin=s_h,
            out_k=full_k[:, 1:],
            out_v=full_v[:, 1:],
            out_dtype=dtype,
        )

        for h in range(3):
            rk_ref = codec.decode(
                k_idx[:, h], k_val[:, h], layer_idx=0, active_atoms=12, check_bounds=True, rope_cos=c_h, rope_sin=s_h
            ).to(dtype)
            rv_ref = codec.decode(
                v_idx[:, h], v_val[:, h], layer_idx=0, active_atoms=12, check_bounds=True
            ).to(dtype)
            self.assertLessEqual((full_k[:, h + 1, :96] - rk_ref).float().abs().max().item(), 2e-3)
            self.assertLessEqual((full_v[:, h + 1, :96] - rv_ref).float().abs().max().item(), 2e-3)
            self.assertTrue(torch.equal(full_k[:, h + 1, 96:], loc_k[:, h]))
            self.assertTrue(torch.equal(full_v[:, h + 1, 96:], loc_v[:, h]))

    def test_03_triton_fused_write_probe_decode_and_prefill_parity(self) -> None:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if dev.type == "cuda" else torch.float32
        torch.manual_seed(23)
        u_q = torch.linalg.qr(torch.randn(4, 128, 16, device=dev, dtype=torch.float32))[0].contiguous()

        for t_len, start_pos in [(1, 0), (1, 10), (64, 0), (193, 2)]:
            k = torch.randn(1, 4, t_len, 128, device=dev, dtype=dtype)
            v = torch.randn(1, 4, t_len, 128, device=dev, dtype=dtype)
            v[:, 0, 0, :] = 0.0  # Verify exact zero-norm V guard
            for sp_mode in (True, False):
                ref = _pytorch_write_probe(k, v, u_q, start_pos, 4, 100.0, sp_mode)
                got = triton_fused_write_probe(
                    k, v, u_q, start_pos=start_pos, sink_tokens=4, sink_bonus=100.0, softplus_mode=sp_mode
                )
                self.assertLessEqual((ref - got).abs().max().item(), 2e-3)

    def test_04_deferred_slab_scoring_flush_and_crop_invariants(self) -> None:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if dev == "cuda" else torch.float32
        cfg = FallbackQwen2Config(
            hidden_size=128,
            num_hidden_layers=4,
            num_attention_heads=8,
            num_key_value_heads=2,
            head_dim=16,
            vocab_size=512,
        )
        model = FallbackQwen2ForCausalLM(cfg, device=dev, dtype=dtype, seed=42).eval()
        tok = FallbackQwen2Tokenizer(vocab_size=512)
        art = create_mock_artifacts(model, tok, subspace_rank=8, dict_atoms=32)

        cache = StoikovKVCache(model=model, artifacts=art, n_b=8, keep_ratio=0.25, s_budget=4, policy_mode="hybrid")
        inp = torch.randint(4, 250, (1, 20), device=dev)
        _ = model(input_ids=inp, past_key_values=cache, use_cache=True, output_attentions=False)
        for _ in range(12):
            step = torch.randint(4, 250, (1, 1), device=dev)
            _ = model(input_ids=step, past_key_values=cache, use_cache=True, output_attentions=False)
        cache.crop(26)
        self.assertEqual(cache.get_seq_length(0), 26)
        tel = cache.get_telemetry_summary()
        self.assertEqual(tel["physical_resident_bytes"], tel["compressed_bytes"])

    def test_05_quant_microstructure_hawkes_vpin_and_glft_almgren_chriss(self) -> None:
        ctrl = OnlineRegimeChangeController(
            keep_ratio=0.20,
            inventory_target_tokens=1000,
            max_h2o_tokens=300,
        )
        e0 = torch.zeros(16)
        e0[0] = 1.0
        ctrl.observe_and_query(e0, seq_len=200)

        # Symmetric oscillating chatter (+d, -d, +d, -d) yields low directional VPIN
        e_pos = e0.clone()
        e_pos[1] = 0.08
        e_neg = e0.clone()
        e_neg[1] = -0.08
        for i in range(6):
            ctrl.observe_and_query(e_pos if i % 2 == 0 else e_neg, seq_len=250 + i)
        self.assertLess(ctrl.last_vpin, 0.50)

        # Aligned orthogonal regime shift triggers high VPIN, beta_shift, and Hawkes kappa_t > kappa_0
        e_shift = torch.zeros(16)
        e_shift[7] = 1.0
        beta_t, _, _ = ctrl.observe_and_query(e_shift, seq_len=400)
        self.assertEqual(beta_t, ctrl.beta_shift)
        self.assertGreater(ctrl.last_kappa_t, ctrl.kappa_arrival)
        self.assertGreater(ctrl.last_liquidity_rebate, 0.0)

        # Above inventory target with max_h2o_tokens=300 at seq_len=3000 -> GLFT cap enforces <= 0.10
        k_capped = ctrl.compute_dynamic_keep_ratio(seq_len=3000)
        self.assertLessEqual(k_capped, 300.0 / 3000.0 + 1e-6)

    def test_06_memo_bidirectional_kg_consolidation(self) -> None:
        triples = [
            AtlasTriple("NodeAlpha", "uses_port", "8443"),
            AtlasTriple("NodeAlpha", "region", "ap-south-1"),
            AtlasTriple("NodeBeta", "uses_port", "8443"),
        ]
        fwd_only = format_kg_triples(triples)
        self.assertEqual(len(fwd_only), 3)

        cons = consolidate_kg_triples(triples, max_inverse_subjects=8)
        joined = "\n".join(cons)
        # Forward subject deduplication + inverse entity surfacing
        self.assertIn("NodeAlpha | uses_port: 8443; region: ap-south-1.", joined)
        self.assertIn("8443 | inverse_uses_port: NodeAlpha, NodeBeta.", joined)

        cfg = FallbackQwen2Config(
            hidden_size=64, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=16, vocab_size=512
        )
        model = FallbackQwen2ForCausalLM(cfg, device="cpu", dtype=torch.float32, seed=7).eval()
        tok = FallbackQwen2Tokenizer(vocab_size=512)
        art = create_mock_artifacts(model, tok, subspace_rank=4, dict_atoms=16)
        bank = AtlasMemoryBank.build_from_kg_triples(
            model, tok, triples, art, consolidate_bidirectional=True
        )
        self.assertGreater(bank.stats()["total_indexed_tokens"], 0)

    def test_07_nested_text_config_and_multimodal_layer_discovery(self) -> None:
        inner_cfg = FallbackQwen2Config(
            hidden_size=64, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=16, vocab_size=256
        )
        outer_cfg = types.SimpleNamespace(model_type="inkling2_loop", text_config=inner_cfg)

        class _MockLoopedWrapper(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.config = outer_cfg
                self.emb = nn.Embedding(256, 64)
                layers = nn.ModuleList(
                    [nn.ModuleDict({"self_attn": FallbackQwen2Attention(inner_cfg, i)}) for i in range(2)]
                )
                self.model = nn.ModuleDict({"llm": nn.ModuleDict({"layers": layers})})

        m = _MockLoopedWrapper()
        info = _inspect_model_architecture(m, r_subspace=4, dict_atoms=16)
        self.assertEqual(info["num_layers"], 2)
        self.assertEqual(info["num_q_heads"], 4)
        self.assertEqual(info["num_kv_heads"], 2)


if __name__ == "__main__":
    unittest.main()
