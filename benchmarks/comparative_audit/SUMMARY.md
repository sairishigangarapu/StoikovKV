# StoikovKV Comparative Audit - Summary

**Date:** 2026-11-25 16:38:41
**Model Track:** real_7b
**Git Commit:** e54013363e3345971ccb9d1f80934a309426cce9

## Environment
- GPU: NVIDIA GeForce RTX 4060 Laptop GPU
- VRAM: 7.6 GB
- PyTorch: 2.12.0+cu130
- CUDA: 13.0

## Results Summary

### needle_retrieval

| Method | Success | Total |
|--------|---------|-------|
| B0_full_fp16 | 12 | 12 |
| B3_standalone_h2o | 12 | 12 |
| B4_stoikovkv_hybrid | 12 | 12 |
| B3_standalone_lexico | 12 | 12 |
| routing_no_eviction | 12 | 12 |
| full_stoikovkv_no_adaptation | 12 | 12 |
| full_stoikovkv_with_adaptation | 12 | 12 |
| B3_h2o_b25 | 12 | 12 |
| B4_stoikovkv_b25 | 12 | 12 |
| B3_h2o_b40 | 12 | 12 |
| B4_stoikovkv_b40 | 12 | 12 |
| B3_h2o_b60 | 12 | 12 |
| B4_stoikovkv_b60 | 12 | 12 |
| B3_h2o_b80 | 12 | 12 |
| B4_stoikovkv_b80 | 12 | 12 |

### long_context_quality

| Method | Success | Total |
|--------|---------|-------|
| B0_full_fp16 | 8 | 8 |
| B3_standalone_h2o | 8 | 8 |
| B4_stoikovkv_hybrid | 8 | 8 |
| B3_standalone_lexico | 8 | 8 |

### context_drift

| Method | Success | Total |
|--------|---------|-------|
| B0_full_fp16 | 1 | 1 |
| B3_standalone_h2o | 1 | 1 |
| B4_stoikovkv_hybrid | 1 | 1 |
| B3_standalone_lexico | 1 | 1 |
| routing_no_eviction | 1 | 1 |
| full_stoikovkv_no_adaptation | 1 | 1 |
| full_stoikovkv_with_adaptation | 1 | 1 |

### memory_accounting

| Method | Success | Total |
|--------|---------|-------|
| B0_full_fp16 | 2 | 2 |
| B3_standalone_h2o | 2 | 2 |
| B4_stoikovkv_hybrid | 2 | 2 |
| B3_standalone_lexico | 2 | 2 |
| routing_no_eviction | 2 | 2 |
| full_stoikovkv_no_adaptation | 2 | 2 |
| full_stoikovkv_with_adaptation | 2 | 2 |

### latency_throughput

| Method | Success | Total |
|--------|---------|-------|
| B0_full_fp16 | 4 | 4 |
| B3_standalone_h2o | 4 | 4 |
| B4_stoikovkv_hybrid | 4 | 4 |
| B3_standalone_lexico | 4 | 4 |
| routing_no_eviction | 4 | 4 |
| full_stoikovkv_no_adaptation | 4 | 4 |
| full_stoikovkv_with_adaptation | 4 | 4 |

### encoder_quality

| Method | Success | Total |
|--------|---------|-------|
| B4_stoikovkv_hybrid | 5 | 5 |
| B3_standalone_lexico | 5 | 5 |

