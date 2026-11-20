# 01 — StoikovKV Benchmark Numbers Reference

> **Navigation:** [Folder Index (`docs/benchmarks/`)](./README.md) | **Next:** [02 — Benchmark Theory, Causal Mechanics & Interpretation Guide](./02_BENCHMARK_THEORY_AND_INTERPRETATION.md)

This document consolidates the exact measured numbers across all StoikovKV benchmark suites (`benchmarks/bench_calibration_codec.py`, `benchmarks/bench_cache_ablation.py`, `benchmarks/comparative_audit/`, `experiments/7b_gpu_validation/`, and `benchmarks/bench_real_qwen_publication.py`). For a complete explanation of **what each test means**, **how the algorithm produced each number**, and **how to interpret the results**, see [02 — Benchmark Theory, Causal Mechanics & Interpretation Guide](./02_BENCHMARK_THEORY_AND_INTERPRETATION.md).

> **Scope & Accounting Note:**
> - **Hardware:** NVIDIA GeForce RTX 4060 Laptop GPU (8 GB VRAM), CUDA 13.0, PyTorch `2.12.0+cu130` (`torch.float16`).
> - **Model Tracks:**
>   1. **Pretrained `Qwen/Qwen2.5-1.5B-Instruct` End-to-End GPU Publication Track** (`Qwen2ForCausalLM` real HuggingFace weights on `cuda:0`) in **Section 1** (Probe Correlation, 1-Step ElasticNet vs. OMP/FISTA on Real Activations, 6-Way Equal-Budget Multi-Turn Cache Ablation incl. `SnapKV` & `KIVI-2bit Quant`, Live `model.generate` / Matryoshka Speculative Decoding, and Multi-Batch Memory-Ceiling Throughput Scaling).
>   2. **Controlled Standalone GQA `Qwen/Qwen2.5-1.5B-Instruct` Architecture Track** (`28` layers, `12` Q heads, `2` KV heads, `q_per_kv = 6`, `head_dim = 128`) in **Sections 2–6**.
>   3. **WikiText-103 & Pretrained `Qwen/Qwen2.5-7B-Instruct` Offloaded Hardware Track** in **Section 7**.
> - **Memory Metrics:** `KV Mem %` reports logical and physical resident tier payload bytes (`compressed_bytes / fp16_bytes == physical_resident_bytes / fp16_bytes`); `+Dict %` (`memory_ratio_incl_dictionary`) adds the shared resident dictionary overhead (`D` + `W_ridge`) across active Lexico layers.

---

## 1. Pretrained `Qwen/Qwen2.5-1.5B-Instruct` End-to-End GPU Publication Suite (`bench_real_qwen_publication.py`)

Evaluated directly on real HuggingFace `Qwen2ForCausalLM` pretrained weights (`Qwen/Qwen2.5-1.5B-Instruct`, `28` layers, `12` Q heads, `2` KV heads, `head_dim = 128`, `torch.float16` on `cuda:0`) with streaming $O(B \cdot T)$ calibration (`output_attentions=False`), pre-RoPE key rarity span probing, and SDPA compact mask slicing:

### 1.1 Real-Weights Write-Time Subspace Probe Correlation (`336` Held-Out `(layer, prompt)` Pairs)
- **Calibration Wall Time on RTX 4060 Laptop GPU:** **`8.92 s`** (`D` shape `(28, 128, 128)`, mean reconstruction MSE `0.001545`, FP64 solve relative residual `7.31e-16`)
- **Held-Out Needle Selectivity Score:** **`0.9923`** (`ROLE_RETRIEVAL_FP16 = 3 (5.4%)`, `ROLE_PEAKED_H2O = 50 (89.3%)`, `ROLE_DIFFUSE_LEXICO = 3 (5.4%)` co-located onto the 3 retrieval layers so `25/28` layers are 100% pure H2O)

| Probe Variant (Real `Qwen/Qwen2.5-1.5B-Instruct`) | Spearman $\rho$ (Mean $\pm$ SD) | Recall@20% (Mean $\pm$ SD) | Gain over $\|V_t\|_2$ Baseline |
|---|---:|---:|---:|
| **Sink-Augmented Directional Subspace Probe** ($+100.0$ on `0..3`) | **0.5733 $\pm$ 0.2716** | **52.19% $\pm$ 18.09%** | **+0.6087 $\rho$** (+29.49% Recall@20%) |
| **Raw Directional Subspace Probe** ($t \ge 4$, no sink bonus) | **0.4020 $\pm$ 0.2887** | **36.73% $\pm$ 18.67%** | **+0.4374 $\rho$** (+14.03% Recall@20%) |
| **Value-Norm Baseline** ($\|V_t\|_2$ only) | -0.0354 | 22.70% | — |

### 1.2 Real-Weights 1-Step ElasticNet vs. OMP vs. FISTA (`1,024` Real `Qwen2.5-1.5B` KV Vectors)
Evaluated on `1,024` real pre-RoPE KV activation vectors extracted from `Qwen/Qwen2.5-1.5B-Instruct` layer 10 on CUDA:

| Sparsity $s$ | Bytes / Vec (`3s+2`) | Ratio vs. FP16 (`256 B`) | Fast 1-Step Cosine | OMP Cosine | FISTA Cosine | Fast Encode (ms) |OMP Encode (ms) | FISTA Encode (ms) | Speedup vs. OMP | Speedup vs. FISTA |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **4** | 14 B | 18.29$\times$ | **0.5419** | 0.5461 | 0.5449 | **0.940** | 9.597 | 11.167 | **10.21$\times$** | **11.88$\times$** |
| **8** | 26 B | 9.85$\times$ | **0.6436** | 0.6499 | 0.6485 | **0.912** | 4.998 | 10.999 | **5.48$\times$** | **12.06$\times$** |
| **12** | 38 B | 6.74$\times$ | **0.7090** | 0.7163 | 0.7149 | **0.938** | 6.866 | 10.749 | **7.32$\times$** | **11.46$\times$** |
| **16** | 50 B | 5.12$\times$ | **0.7566** | 0.7648 | 0.7635 | **0.883** | 8.671 | 11.594 | **9.82$\times$** | **13.13$\times$** |
| **32** | 98 B | 2.61$\times$ | **0.8678** | 0.8798 | 0.8785 | **0.906** | 27.497 | 11.542 | **30.35$\times$** | **12.74$\times$** |

### 1.3 Real-Weights 6-Way Equal-Budget (~20% KV Memory) Multi-Turn Cache Ablation (`Qwen/Qwen2.5-1.5B-Instruct`)
Evaluated on a `1,024`-token two-turn prompt on real `Qwen/Qwen2.5-1.5B-Instruct` (Turn 1: Distributed systems notes with planted passkey `"94812"` across 4 early depths `{5%, 12%, 22%, 35%}`; Turn 2: Abrupt topic shift to CSS glassmorphism design tokens):

| Configuration (Real `Qwen2.5-1.5B-Instruct`) | Policy Mode | `keep_ratio` / `s` | KV Mem % (Physical = Logical) | +Dict % | Needle `94812` Acc | Needle Logit Margin | Turn-2 CSS Logit Cosine | Turn-2 H2O Freshness | HMM Shifts | Composite Quality | Eval Time (s) |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Full FP16** | `full_fp16` | `1.000` / `16` | 100.00% | 100.00% | **100.0%** (4/4) | +7.09 | **1.0000** | 40.04% | 0 | 100.00% | **2.42 s** |
| **Standalone H2O** | `standalone_h2o` | `0.193` / `16` | 21.93% | 21.93% | 75.0% (3/4) | +4.68 | 0.9967 | 23.03% | 0 | 87.34% | 2.66 s |
| **SnapKV** | `snapkv` | `0.193` / `16` | 21.93% | 21.93% | **0.0%** (0/4) | -4.19 | 0.7703 | 37.92% | 0 | 38.52% | 2.58 s |
| **KIVI-2bit Quant** | `quant_int4` | `1.000` / `16` | 19.87% | 19.87% | **0.0%** (0/4) | -8.97 | 0.0160 | 0.00% | 0 | 0.80% | 4.77 s |
| **Standalone Lexico** | `standalone_lexico` | `1.000` / `16` | 21.00% | 27.47% | **100.0%** (4/4) | +3.99 | 0.9621 | 0.00% | 0 | 98.11% | 6.68 s |
| **StoikovKV Hybrid** | `hybrid` | `0.145` / `12` | **20.56%** | **21.25%** | **100.0%** (4/4) | **+7.63** | **0.9850** | **67.70%** | **11** | **99.25%** | **3.09 s** |

### 1.4 Real-Weights Multi-Batch Memory-Ceiling & Throughput Scaling (`128 MB` KV Pool at `L = 1,024`)
Evaluated on real `Qwen/Qwen2.5-1.5B-Instruct` (`cuda:0`) under a fixed `128 MB` KV-cache memory pool as batch size scales from `B = 1` to `B = 12`:

| Batch Size $B$ | Full FP16 Physical KV (MB) | Full FP16 Status | Full FP16 System Throughput (`tok/s`) | StoikovKV Hybrid Physical KV (MB) | StoikovKV Hybrid Status | StoikovKV Hybrid System Throughput (`tok/s`) | Throughput Gain at Memory Ceiling |
|---:|---:|---|---:|---:|---|---:|---:|
| **B = 1** | 27.48 MB | `OK` | 18.7 tok/s | **6.03 MB** (`4.56x` smaller) | `OK` | 14.1 tok/s | — |
| **B = 2** | 54.96 MB | `OK` | 36.5 tok/s | **12.05 MB** (`4.56x` smaller) | `OK` | 27.2 tok/s | — |
| **B = 4** | 109.92 MB | `OK` (Max Batch) | **70.8 tok/s** (Ceiling) | **24.11 MB** (`4.56x` smaller) | `OK` | 54.6 tok/s | — |
| **B = 8** | 227.50 MB | **`KV_POOL_OOM`** | `0.0 tok/s` (OOM) | **48.22 MB** | **`OK`** | **105.9 tok/s** | **+49.6% over FP16 Max** |
| **B = 12** | 341.25 MB | **`KV_POOL_OOM`** | `0.0 tok/s` (OOM) | **72.33 MB** | **`OK`** | **149.1 tok/s** | **2.11$\times$ over FP16 Max (`70.8 tok/s`)** |

### 1.5 Real-Weights Compressed-Cache Passkey Retrieval, `model.generate` & Matryoshka Speculative Decoding
| Policy (`Qwen/Qwen2.5-1.5B-Instruct`) | Effective Bytes / Vec | Compression vs. FP16 | Calibration Passkey Acc | Held-Out Passkey Acc | Overall Top-1 Acc | Passkey Prob Retention | Logit Cosine | Next-Token KL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Full FP16** (`full_fp16`) | 256.00 B | 1.00$\times$ | **100.0%** | **100.0%** | **100.0%** | 100.0% | 1.0000 | 0.0000 |
| **StoikovKV Hybrid Verify** (`stoikov_hybrid_s16`) | 171.36 B | **1.49$\times$** | **100.0%** | **100.0%** | **100.0%** | 82.80% | **0.9096** | 0.2394 |
| **StoikovKV Hybrid Draft** (`stoikov_hybrid_draft_m4`) | 132.79 B | **1.93$\times$** | **100.0%** | **100.0%** | **100.0%** | 77.19% | **0.9076** | 0.3432 |
| **Unprotected Ablation** (`unprotected_ablation`) | 5.00 B | 51.20$\times$ | **0.0%** | **0.0%** | **0.0%** | 2.68% | 0.5777 | 3.6446 |

- **Drop-In `model.generate(..., past_key_values=cache)` (`412`-Token Vault Prompt):** Outputs exact `'94812.'` at **`39.50%` of FP16 (`60.50%` physical KV memory reduction)**.
- **Matryoshka Self-Speculative Decoding (`gamma_spec = 4, draft_atoms = 4, verify_atoms = 16`):** Outputs exact `'94812. Remember this code...'` with **`100.0%` empirical acceptance rate ($\alpha = 1.0000$)**.

---

## 2. Controlled Standalone GQA Calibration & Subspace Probe (`bench_calibration_codec.py`)

Evaluated on **336 held-out `(layer, prompt)` pairs** (`6` held-out prompts $\times$ `28` layers $\times$ `2` KV heads, strictly disjoint from the `16` calibration training prompts):

| Probe Variant | Spearman $\rho$ (Mean $\pm$ SD) | Recall@20% (Mean $\pm$ SD) | Sink Tokens | Sink Bonus |
|---|---:|---:|---:|---:|
| **Raw Directional Subspace Probe** ($\sigma((K_t U_Q)_0) \cdot \|K_t U_Q\|_2 \cdot \|V_t\|_2$, $t \ge 4$) | **0.5776 $\pm$ 0.2240** | **52.27% $\pm$ 17.50%** | 0 | 0.0 |
| **Sink-Augmented Probe** ($+100.0$ on indices `0..3`) | **0.4892 $\pm$ 0.2233** | **56.51% $\pm$ 18.02%** | 4 | +100.0 |
| **Value-Norm Baseline** ($\|V_t\|_2$ only) | 0.5612 | 58.70% | 0 | 0.0 |

- **Head Role Classification:** `ROLE_RETRIEVAL_FP16 = 5.0%–5.4%`, `ROLE_PEAKED_H2O = 83.9%–85.0%`, `ROLE_DIFFUSE_LEXICO = 10.0%–10.7%`; Held-Out Needle Selectivity Score = **`1.0000`**.
- **Universal Dictionary & FP64 Ridge Preconditioner:** $D \in \mathbb{R}^{128 \times 128}$, $W_{\text{ridge}} \in \mathbb{R}^{128 \times 128}$ ($\lambda_2 = 0.1$), Mean Reconstruction MSE = `0.000220`, Max FP64 Linear Solve Relative Residual = **`7.20e-16`**.

---

## 3. Controlled Sparse Dictionary Codec Benchmark (`bench_calibration_codec.py`)

Evaluated on `1,024` held-out KV vectors (`head_dim = 128`, `torch.float16` on CUDA, `15` timed repetitions after `5` warmups):

| Sparsity $s$ | Bytes / Vec (`3s+2`) | Ratio vs. FP16 (`256 B`) | Fast 1-Step Cosine | OMP Cosine | FISTA Cosine | Fast Encode (ms) | Fast Decode (ms) | OMP Encode (ms) | FISTA Encode (ms) | Speedup vs. OMP | Speedup vs. FISTA |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **4** | 14 B | 18.29$\times$ | 0.6129 | 0.6142 | 0.6140 | **0.529** | **0.108** | 1.676 | 8.917 | **3.17$\times$** | **16.87$\times$** |
| **8** | 26 B | 9.85$\times$ | 0.7133 | 0.7159 | 0.7156 | **0.547** | **0.137** | 2.702 | 8.766 | **4.94$\times$** | **16.03$\times$** |
| **12** | 38 B | 6.74$\times$ | 0.7713 | 0.7750 | 0.7748 | **0.531** | **0.139** | 3.816 | 8.365 | **7.19$\times$** | **15.76$\times$** |
| **16** | 50 B | 5.12$\times$ | 0.8103 | 0.8153 | 0.8151 | **0.527** | **0.167** | 4.957 | 8.336 | **9.40$\times$** | **15.81$\times$** |
| **32** | 98 B | 2.61$\times$ | 0.8945 | 0.9067 | 0.9064 | **0.594** | **0.243** | 18.776 | 8.979 | **31.60$\times$** | **15.11$\times$** |

---

## 4. 6-Way Equal-Budget (~20% KV Bytes) Cache Ablation (`bench_cache_ablation.py`)

Evaluated on a **2,037-token multi-turn prompt** (Turn 1: Python backend service logs containing early needle `"API_PORT = 8443"` across depths `{5%, 12%, 22%, 35%}`; Turn 2: abrupt topic shift to CSS glassmorphism stylesheet rules at token `1,170`). Compression parameters `(keep_ratio, s_budget)` are solved automatically by `solve_equal_budget_params` so every compressed configuration uses $\le 20\%$ of FP16 KV payload bytes:

| Configuration | Policy Mode | `keep_ratio` / `s` | KV Mem % (Logical = Physical) | +Dict % | KV Payload Bytes | Dict Bytes | Needle `8443` Acc | Needle Logit Margin | Turn-2 CSS Logit Cosine | Turn-2 H2O Freshness | HMM Shifts | Composite Quality |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Full FP16** | `full_fp16` | `1.000` / `16` | 100.00% | 100.00% | 58,404,864 | 0 | **100.0%** (4/4) | +2457.81 | **1.0000** | 42.56% | 0 | 100.00% |
| **Standalone H2O** | `standalone_h2o` | `0.168` / `16` | 19.93% | 19.93% | 11,640,832 | 0 | **100.0%** (4/4) | +2346.73 | **0.9815** | 44.62% | 0 | 99.08% |
| **SnapKV** | `snapkv` | `0.168` / `16` | 19.93% | 19.93% | 11,640,832 | 0 | **50.0%** (2/4) | +1542.10 | 0.9858 | 40.18% | 0 | 74.29% |
| **KIVI-2bit Quant** | `quant_int4` | `1.000` / `16` | 19.79% | 19.79% | 11,557,952 | 0 | **0.0%** (0/4) | -412.50 | 0.4102 | 0.00% | 0 | 20.51% |
| **Standalone Lexico** | `standalone_lexico` | `1.000` / `14` | 19.79% | 22.93% | 11,557,952 | 1,835,008 | **50.0%** (2/4) | +1367.86 | 0.8569 | 0.00% | 0 | 67.84% |
| **StoikovKV Hybrid** | `hybrid` | `0.1211` / `10` | **19.76%** | **20.44%** | **11,542,912** | **393,216** | **100.0%** (4/4) | **+2540.95** | **0.9093** | **100.00%** | **8** | **95.47%** |

---

## 5. Matryoshka Self-Speculative Decoding Sweep (`bench_cache_ablation.py`)

Evaluated on a `1,068`-token prompt generating `64` new tokens with verify sparsity $s = 16$ (`50 B/vec`), speculative draft length $\gamma_{\text{spec}} = 4$, single-pass $\gamma_{\text{spec}} + 1$ verification with bonus token emission, and draft atoms $m \in \{2, 4, 8, 12\}$:

| Draft Atoms $m$ | Draft Bytes / Vec (`3m+2`) | Empirical Acceptance $\alpha$ | Expected $\mathbb{E}[\alpha]$ | Avg Accepted / Verify Pass | Avg Tokens / Cycle | KV Bytes Read / Token | Bandwidth Savings vs. Standard Decode | Bandwidth Savings vs. Full FP16 | Passkey `8443` Retrieved |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **m = 2** | 8 B | **0.9615** | 0.9445 | 3.33 / 4 | 4.33 | 7,422.9 KB | **4.90%** | **75.88%** | Yes |
| **m = 4** | 14 B | **0.9615** | 0.9527 | 3.33 / 4 | 4.33 | 7,477.5 KB | **4.20%** | **75.70%** | Yes |
| **m = 8** | 26 B | **0.9615** | 0.9584 | 3.33 / 4 | 4.33 | 7,586.6 KB | **2.80%** | **75.35%** | Yes |
| **m = 12** | 38 B | **0.9615** | 0.9601 | 3.33 / 4 | 4.33 | 7,695.8 KB | **1.40%** | **74.99%** | Yes |

*(Standard decode on Hybrid reads `7,804.9 KB/token`; standard Full FP16 decode reads `30,772.0 KB/token`.)*

---

## 6. Multi-Length Needle Retrieval & Latency Audit (`benchmarks/comparative_audit`)

`15` runs (`5` trials $\times$ `3` seeds) per `(context_length, needle_position)` across `512`, `2,048`, and `4,096` tokens:

| Context Length | Needle Position | Full FP16 (B0) | Standalone H2O (B3) | StoikovKV Hybrid (B4) | Unprotected Ablation |
|---:|---|---:|---:|---:|---:|
| **512** | Early / Middle / Late | 100% / 100% / 100% | **0%** / 100% / 100% | **100% / 100% / 100%** | **0%** |
| **2,048** | Early / Middle / Late | 100% / 100% / 100% | **0%** / 100% / 100% | **100% / 100% / 100%** | **0%** |
| **4,096** | Early / Middle / Late | 100% / 100% / 100% | **0%** / 100% / 100% | **100% / 100% / 100%** | **0%** |

---

## 7. WikiText-103 & 7B Validation Summary (`experiments/7b_gpu_validation`)

### 7.1 WikiText-103 Evaluation (`50` Samples, Compact 12L/4KV GQA Config)
| Policy | Throughput (tok/s) | Peak GPU VRAM | Physical = Logical KV Mem % | Memory Savings % | HMM Shifts Detected |
|---|---:|---:|---:|---:|---:|
| **Full FP16** | **3,285.6 tok/s** | 0.32 GB | 100.0% | 0.0% | 0 |
| **Standalone H2O** | **3,109.3 tok/s** | **0.26 GB** | 13.5% | **86.5%** | 0 |
| **StoikovKV Hybrid** | **1,910.8 tok/s** | **0.29 GB** | **20.0%** | **80.0%** | **15** |
| **Standalone Lexico** | **687.6 tok/s** | **0.30 GB** | 41.2% | 58.8% | 0 |

### 7.2 Pretrained `Qwen/Qwen2.5-7B-Instruct` Offloaded Hardware Check (8 GB RTX 4060 Laptop)
- **Placement:** Layers `0..9` on GPU (`5.75 GB`), Layers `10..27` offloaded to CPU (`8.25 GB`), peak VRAM `6.94 GB`.
- **KV Memory Reduction on Real 7B:** `60.1%` (Early, 28 MB FP16 KV), `63.1%` (Middle, 32 MB FP16 KV), `67.3%` (Late, 40 MB FP16 KV) at `0.77 - 0.82 tok/s` (CPU offload bound).
