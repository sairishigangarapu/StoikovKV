# StoikovKV

**StoikovKV** is a hybrid KV-cache compression, closed-loop inventory control, and hierarchical offloading engine for Grouped-Query Attention (GQA) language models. It combines FlashAttention/SDPA-compatible write-time query-subspace probing (`output_attentions=False`), per-head role routing, fused Triton sparse dictionary decoding (+ RoPE), Avellaneda–Stoikov (2008) inventory/volatility control, 3-level CPU-offloaded `HiKVP`/`AtlasKV` retrieval, and Matryoshka self-speculative decoding.

> **🏛️ System Architecture, Evolution & Benchmark Documentation:**
> - **[`docs/SYSTEM_ARCHITECTURE_AND_EVOLUTION.md`](./docs/SYSTEM_ARCHITECTURE_AND_EVOLUTION.md)** — Complete end-to-end topological architecture, subsystem derivations, defense-in-depth security model, 7-phase project evolution, and master benchmark tables.
> - **[`docs/benchmarks/01_BENCHMARK_NUMBERS.md`](./docs/benchmarks/01_BENCHMARK_NUMBERS.md)** — Complete measured numbers across calibration, multi-arch fused Triton kernels (`7.57x` 3D KV materialize, `288.85x` batch-deferred probe, `26.2x` 16K decode), 1-step ElasticNet vs. OMP/FISTA, 6-way equal-budget ablations, Matryoshka speculative decoding, `HiKVP`/`AtlasKV` (`MeMo` bidirectional KG2KV), 100% GPU `Qwen/Qwen2.5-7B-Instruct` (4-bit NF4: **`36.85 tok/s`**, `79.0%` KV saved, `100%` NIAH), and 48-pass looped `Inkling-2-Mini` (`77.6%` KV saved).
> - **[`docs/benchmarks/02_BENCHMARK_THEORY_AND_INTERPRETATION.md`](./docs/benchmarks/02_BENCHMARK_THEORY_AND_INTERPRETATION.md)** — Detailed theory guide explaining **what each test means**, **how each result was produced**, and **how to interpret the metrics**.

---

## Repository Layout

```text
.
├── pyproject.toml                  # Package metadata, dependencies & stoikov-kv CLI entrypoint
├── src/
│   └── stoikov_kv/                 # Core Python package
│       ├── __init__.py
│       ├── calibration.py          # Streaming O(B*T) subspace (U_Q), head-role & dictionary calibration
│       ├── elastic_lexico.py       # 1-step ElasticNet sparse dictionary codec + OMP/FISTA baselines
│       ├── triton_kernels.py       # Fused Triton sparse dictionary decode + half-split RoPE kernel
│       ├── cache.py                # StoikovKVCache(DynamicCache) + Avellaneda-Stoikov controller
│       ├── hikvp.py                # 3-level pre-RoPE CPU-offloaded KV index + GPU LRU page cache
│       ├── atlas_memory.py         # AtlasKV Doc2KV / KG2KV persistent memory & rectangular attention
│       ├── matryoshka_spec.py      # Single-pass (n_draft+1) Matryoshka self-speculative decoding
│       └── cli.py                  # Command-line interface (calibrate, chat, bench, serve)
├── docs/
│   ├── benchmarks/                 # Ordered benchmark numbers & theory documentation
│   │   ├── README.md               # Ordered reading guide for benchmarks & theory
│   │   ├── 01_BENCHMARK_NUMBERS.md # Consolidated empirical benchmark tables (Sections 1-8)
│   │   └── 02_BENCHMARK_THEORY_AND_INTERPRETATION.md # Test meanings, derivations & interpretation
│   └── reports/historical/         # Archived hackathon reports & raw snapshots
├── server/                         # FastAPI OpenAI-compatible (/v1/chat/completions) & telemetry backend
├── frontend/                       # Next.js real-time cache telemetry dashboard
├── config/                         # YAML configuration files (model, cache, server)
├── scripts/                        # Launch & E2E verification scripts
├── benchmarks/                     # Reproducible benchmark suites
│   ├── bench_calibration_codec.py  # Probe correlation, ElasticNet vs OMP/FISTA, domain L* break-even
│   ├── bench_cache_ablation.py     # Equal-budget 4-way cache ablation & Matryoshka sweep
│   ├── bench_context_drift.py      # Multi-turn topic-shift & H2O freshness evaluation
│   ├── bench_context_switch.py     # Multi-turn context switching evaluation
│   ├── bench_longbench.py          # Long-context task evaluation suite
│   ├── bench_hikvp_atlas.py        # 3-level HiKVP/AtlasKV CPU-offloading & LRU cache sweep
│   ├── comparative_audit/          # Configurable multi-seed comparative audit runner
│   └── results/                    # Generated plots & JSON metrics (gitignored)
├── experiments/
│   └── 7b_gpu_validation/          # WikiText-103 & 7B GPU calibration/offloading scripts
├── tests/                          # Unit test suites (33 tests)
│   ├── test_calibration_and_codec.py
│   ├── test_cache_and_speculative.py
│   └── test_hikvp_and_atlas.py
├── artifacts/                      # Local calibrated .pt artifacts (gitignored)
├── models/                         # Local model weight shards (gitignored)
└── data/                           # Local evaluation datasets (gitignored)
```

---

## Architecture Overview

1. **Offline Streaming Calibration (`stoikov_kv.calibration`):**
   - Extracts a rank-$r$ ($r = 16$) orthonormal query subspace basis $U_Q \in \mathbb{R}^{L \times H_{kv} \times d \times r}$ per KV head group via regularized uncentered eigendecomposition ($Q^\top Q + 0.1 I_d$) sign-aligned against $\bar{Q}$ ($\bar{Q} U_Q \ge 0$), running in $O(B \cdot T)$ streaming hook memory with `output_attentions=False`.
   - Scores incoming tokens at write time without materializing the $O(T^2)$ attention matrix:
     $$I_t = \phi\!\left((K_t U_Q)_0\right) \cdot \|K_t U_Q\|_2 \cdot \psi\!\left(\|V_t\|_2\right)$$
   - Classifies each KV head into one of three tiers:
     - `ROLE_RETRIEVAL_FP16 (0)`: High needle-copy heads (`~5%`) retained in exact FP16.
     - `ROLE_PEAKED_H2O (1)`: Low-entropy heads (`~84%`) managed via decayed heavy-hitter eviction (`keep_ratio`).
     - `ROLE_DIFFUSE_LEXICO (2)`: High-entropy diffuse heads (`~11%`) compressed into $s$ sorted dictionary atoms (`3s + 2` bytes/vector).
   - Trains per-layer normalized dictionaries $D \in \mathbb{R}^{L \times d \times N}$ on pre-RoPE keys and values, and precomputes the ridge projection matrix $W_{\text{ridge}} = (D^\top D + \lambda_2 I)^{-1} D^\top$ in FP64 (`residual < 1.5e-15`).

2. **Fused Triton Sparse Dictionary Codec (`stoikov_kv.elastic_lexico`, `stoikov_kv.triton_kernels`):**
   - Encodes vectors in a single matrix-multiply + soft-threshold + top-$s$ selection pass (`0.53 ms` on GPU, `3.17x–37.58x` faster than OMP and `12.11x–16.87x` faster than FISTA), sorting atoms by descending magnitude (`|y_(1)| >= ... >= |y_(s)|`) and quantizing values to `int8` with a per-vector FP16 scale.
   - Reconstructs pre-RoPE keys and applies half-split RoPE rotation in a **single fused Triton kernel** (`_lexico_reconstruct_kernel`), achieving **`8.68x` speedup at 4K** and **`26.20x` speedup at 16K** (`2.500 ms -> 0.095 ms`, cosine similarity `1.0000000`) while cutting temporary VRAM overhead by **`33.5x`** (zero intermediate `[N, m, d]` HBM allocation).

3. **In-Place Compacted Hybrid Cache & Avellaneda–Stoikov Controller (`stoikov_kv.cache`):**
   - Stages incoming decode tokens in a pre-allocated in-place FP16 slab (`n_b = 64` + batch-flush stride) with shared-storage pure-layer views (`physical_resident_bytes == compressed_bytes`).
   - Implements closed-loop **Avellaneda–Stoikov (2008)** inventory control ($\text{skew}_t = q_t \gamma \sigma_t^2$, $\delta_t^* = \frac{1}{2}\gamma\sigma_t^2 + \frac{1}{\gamma}\ln(1 + \gamma/\kappa)$) via `OnlineRegimeChangeController`: relaxes `keep_ratio` when VRAM headroom $q_t < 0$ is ample (`3.5x` gain on 7B multi-fact QA: `12.5% -> 43.8%`), tightens back to `~78.8%` KV memory savings as $q_t \to +1$, and triggers single-step decay ($\beta_{\text{shift}} = 0.65$) on semantic regime shifts.

4. **3-Level Offloaded `HiKVP` & `AtlasMemoryBank` (`stoikov_kv.hikvp`, `stoikov_kv.atlas_memory`):**
   - Offloads pre-RoPE leaf pages to pinned CPU RAM / `mmap` under an $O(N^{1/3})$ GPU root index and fixed-capacity GPU LRU cache (`1.14 MB` flat GPU footprint across `2K -> 16K`), gated by the Avellaneda–Stoikov optimal half-spread to prune redundant PCIe fetches.

5. **Matryoshka Self-Speculative Decoding (`stoikov_kv.matryoshka_spec`):**
   - Drafts candidate tokens using top $m < s$ dictionary atoms (`3m + 2` bytes/vector) with Avellaneda–Stoikov confidence-margin early stopping, and verifies all drafted positions in a single batched forward pass using all $s$ atoms with lossless `crop()` rollback.

---

## Engineering Status & Validation Summary

- **Complete Unit, Multi-Arch GPU & Red-Team Security Test Coverage:** All **48 automated tests** (`pytest tests/`) pass, covering calibration, 3D fused Triton KV materialization, hardware-adaptive kernel selection (`Ada`, `Ampere`, `Hopper`, `Blackwell sm_100/sm_120`, `ROCm MI300X`, `Intel XPU`), quantitative microstructure control (`Avellaneda–Stoikov`, `Hawkes`, `GLFT/Almgren–Chriss`, `VPIN`), lossless speculative rollback, `HiKVP`/`AtlasKV` (`MeMo` bidirectional KG2KV), and defense-in-depth RCE/GPU-OOB/ASGI security hardening (`_security.py`).
- **Physical == Logical KV Memory & Zero-Cat Shared-Storage Slabs:** `StoikovKVCache` binds historical `ck/cv` and the 64-token decode slab into unified storage so 63/64 decode steps execute as `O(1)` slices with zero `torch.cat` and zero duplicate `DynamicCache` buffers.
- **100% GPU `Qwen/Qwen2.5-7B-Instruct` (4-bit NF4) & 48-Pass Looped `Inkling-2-Mini` Validation (8 GB RTX 4060):**
  - `Qwen2.5-7B-Instruct` (NF4): **`36.85 tok/s`** (**`95.4%` of uncompressed FP16 `38.63 tok/s`**, up `1.91x` from `19.33 tok/s`), **`79.0%` physical KV savings** (`35.33 MB` vs `168.16 MB`), **`100.0%` NIAH** (vs `33.3%` H2O), and **`43.8%` multi-fact QA** (`3.5x` over static).
  - `Inkling-2-Mini` (`48`-pass Looped GQA, `384` KV heads/token): **`395.81 MB -> 88.82 MB` (`77.6%` physical KV savings)** at **`36.04 tok/s`** with **`0.9837–0.9960` next-token logit cosine similarity**.

---

## Installation & Quickstart

```bash
# Install in editable mode
pip install -e .

# Run all 33 unit tests
PYTHONPATH=src python3 -m pytest tests/ -v

# Run comparative audit on real Qwen2.5-7B-Instruct (4-bit NF4 on CUDA)
PYTHONPATH=src python3 benchmarks/comparative_audit/scripts/run_benchmarks.py --track real_7b --quick

# Run calibration + sparse codec benchmark (writes to benchmarks/results/)
python3 benchmarks/bench_calibration_codec.py

# Run 4-way equal-budget cache ablation & Matryoshka sweep (writes to benchmarks/results/)
python3 benchmarks/bench_cache_ablation.py

# Use the CLI or launch the OpenAI-compatible FastAPI server
stoikov-kv --help
python3 scripts/serve.py
```
