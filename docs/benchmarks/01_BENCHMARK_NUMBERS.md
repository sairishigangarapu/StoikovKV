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

## 0. Multi-Scale External Paul Graham Essays Audit (`bench_real_models_audit.py`)

Evaluated directly on real pretrained HuggingFace weights (**`Qwen/Qwen2.5-1.5B-Instruct`**, **`Qwen/Qwen2.5-3B-Instruct`**, and **`Qwen/Qwen2.5-7B-Instruct`**) using strictly unseen 5-digit passkeys (disjoint from calibration code `"94812"`) planted inside independent essays from the external **Kamradt `LLMTest_NeedleInAHaystack` Paul Graham Essays** corpus (`50` essays, `8.3 MB`). All accuracies report exact numerators/denominators and **95% Wilson score confidence intervals**.

### 0.1 Multi-Scale Held-Out Write-Time Subspace Probe (`1,440` Combined `(layer, prompt)` Pairs)

| Real Pretrained Checkpoint | Placement / Calib Time | Held-Out Pairs | Sink-Aug. Probe $\rho$ (Mean $\pm$ SD) | Sink-Aug. Recall@20% | Raw Probe ($t \ge 4$) $\rho$ | Value-Norm ($\|V_t\|_2$) $\rho$ | Dict MSE / FP64 Residual |
|---|---|---:|---:|---:|---:|---:|---|
| **`Qwen/Qwen2.5-1.5B-Instruct`** | 100% GPU FP16 (`11.57 s`) | 336 | **0.5733 $\pm$ 0.2716** | **52.19% $\pm$ 18.09%** | **0.4020 $\pm$ 0.2887** | -0.0354 | `0.001545` / `7.31e-16` |
| **`Qwen/Qwen2.5-3B-Instruct`** | 100% GPU FP16 (`15.59 s`) | 432 | **0.5655 $\pm$ 0.2456** | **52.64% $\pm$ 15.99%** | **0.3903 $\pm$ 0.2758** | -0.0031 | `0.001585` / `6.89e-16` |
| **`Qwen/Qwen2.5-7B-Instruct`** | GPU=7/CPU=5/NVMe=16 (`85.88 s`) | 672 | **0.4838 $\pm$ 0.2908** | **49.72% $\pm$ 16.55%** | **0.3491 $\pm$ 0.3413** | +0.0043 | `0.001797` / `6.97e-16` |

### 0.2 External Paul Graham Essays NIAH Across 1.5B, 3B, and 7B (Equalized $\sim 21\%$ KV Budget)

| Model / Policy | `keep_ratio` / `s` | Hits / $N$ | Acc. (%) | 95% Wilson CI | KV Mem % | +Dict % | Next-Token Logit Cos | Decode (`tok/s`) |
|---|---:|---:|---:|---|---:|---:|---:|---:|
| *Real `Qwen/Qwen2.5-1.5B-Instruct` (100% GPU FP16, $N=20$ essays, $L \approx 1{,}024$, depths $\{10, 25, 50, 75\}\%$)* | | | | | | | | |
| **Full FP16** | `1.000` / `16` | 20 / 20 | 100.0% | `[83.9%, 100.0%]` | 100.00% | 100.00% | 1.0000 | 17.76 |
| **Standalone H2O** | `0.193` / `16` | 0 / 20 | 0.0% | `[0.0%, 16.1%]` | 21.93% | 21.93% | 0.8912 | 14.89 |
| **SnapKV** | `0.193` / `16` | 12 / 20 | 60.0% | `[38.7%, 78.1%]` | 21.93% | 21.93% | 0.7973 | 16.28 |
| **KIVI-2bit Quant** | `1.000` / `16` | 0 / 20 | 0.0% | `[0.0%, 16.1%]` | 19.89% | 19.89% | 0.5882 | 6.40 |
| **Standalone Lexico** | `1.000` / `15` | 5 / 20 | 25.0% | `[11.2%, 46.9%]` | 21.03% | 27.56% | 0.8903 | 4.70 |
| **StoikovKV Hybrid** | `0.144` / `12` | **17 / 20** | **85.0%** | **`[64.0%, 94.8%]`** | **21.56%** | **22.26%** | **0.9351** | **12.91** |
| *Real `Qwen/Qwen2.5-3B-Instruct` (100% GPU FP16, $N=16$ essays, $L \approx 1{,}024$, depths $\{10, 25, 50, 75\}\%$)* | | | | | | | | |
| **Full FP16** | `1.000` / `16` | 16 / 16 | 100.0% | `[80.6%, 100.0%]` | 100.00% | 100.00% | 1.0000 | 13.98 |
| **Standalone H2O** | `0.188` / `16` | 0 / 16 | 0.0% | `[0.0%, 19.4%]` | 21.84% | 21.84% | 0.7749 | 11.81 |
| **SnapKV** | `0.188` / `16` | 16 / 16 | 100.0% | `[80.6%, 100.0%]` | 21.84% | 21.84% | 0.7161 | 12.39 |
| **KIVI-2bit Quant** | `1.000` / `16` | 1 / 16 | 6.2% | `[1.1%, 28.3%]` | 19.83% | 19.83% | 0.3159 | 4.93 |
| **Standalone Lexico** | `1.000` / `15` | 5 / 16 | 31.2% | `[14.2%, 55.6%]` | 20.96% | 27.33% | 0.8441 | 3.24 |
| **StoikovKV Hybrid** | `0.132` / `11` | **16 / 16** | **100.0%** | **`[80.6%, 100.0%]`** | **20.65%** | **21.35%** | **0.9371** | **9.92** |
| *Real `Qwen/Qwen2.5-7B-Instruct` (GPU=7/CPU=5/NVMe=16 Offloaded FP16, $N=16$ essays, $L \approx 512$, depths $\{10, 25, 50, 75\}\%$)* | | | | | | | | |
| **Full FP16** | `1.000` / `16` | 16 / 16 | 100.0% | `[80.6%, 100.0%]` | 100.00% | 100.00% | 1.0000 | 0.23 |
| **Standalone H2O** | `0.158` / `16` | 0 / 16 | 0.0% | `[0.0%, 19.4%]` | 21.87% | 21.87% | 0.7488 | 0.24 |
| **SnapKV** | `0.158` / `16` | 16 / 16 | 100.0% | `[80.6%, 100.0%]` | 21.78% | 21.78% | 0.7729 | 0.23 |
| **StoikovKV Hybrid** | `0.106` / `9` | **16 / 16** | **100.0%** | **`[80.6%, 100.0%]`** | **20.75%** | **21.19%** | **0.8926** | **0.25** |

### 0.3 `1,024` Strictly Distinct, Non-Tiled Real Paul Graham Prose KV Vectors (`Qwen2.5-1.5B` Layer 10)

| Sparsity $s$ | Bytes / Vec (`3s+2`) | 1-Step Cosine | OMP Cosine | FISTA Cosine | 1-Step Enc (ms) | 1-Step Dec (ms) | OMP Enc (ms) | FISTA Enc (ms) | Speedup vs. OMP | Speedup vs. FISTA |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **4** | 14 B | 0.4677 | 0.4736 | 0.4718 | **1.896** | 0.198 | 16.812 | 10.419 | **8.87$\times$** | **5.50$\times$** |
| **8** | 26 B | 0.5852 | 0.5921 | 0.5903 | **0.869** | 0.106 | 5.021 | 10.367 | **5.78$\times$** | **11.93$\times$** |
| **12** | 38 B | 0.6609 | 0.6686 | 0.6667 | **0.847** | 0.105 | 6.492 | 10.257 | **7.67$\times$** | **12.11$\times$** |
| **16** | 50 B | 0.7167 | 0.7249 | 0.7232 | **0.849** | 0.108 | 8.208 | 10.287 | **9.66$\times$** | **12.11$\times$** |
| **32** | 98 B | 0.8488 | 0.8586 | 0.8572 | **0.877** | 0.160 | 32.959 | 11.643 | **37.58$\times$** | **13.27$\times$** |

### 0.4 Strict Lossless Matryoshka Speculative Parity Audit (`strict_rejection_sampling=True`, Real `Qwen2.5-1.5B`)

| Draft Atoms $m$ | Greedy Exact Parity | Greedy $\alpha$ / `tok/cyc` | Stochastic ($T=0.7$) Emp. $\alpha$ | Stochastic $\mathbb{E}[\alpha]$ | Stochastic `tok/cyc` | BW Saved vs. Full FP16 |
|---:|---|---:|---:|---:|---:|---:|
| **$m = 2$ (8 B)** | **32/32 (100.0%)** | `1.0000` / `4.71` | `0.9259` | `0.9574` | `4.12` | **66.34%** |
| **$m = 4$ (14 B)** | **32/32 (100.0%)** | `1.0000` / `4.71` | `0.9630` | `0.9735` | `4.71` | **66.25%** |
| **$m = 8$ (26 B)** | **32/32 (100.0%)** | `1.0000` / `4.71` | `0.9630` | `0.9856` | `4.71` | **66.07%** |
| **$m = 12$ (38 B)** | **32/32 (100.0%)** | `1.0000` / `4.71` | `0.9630` | `0.9908` | `4.71` | **65.90%** |

### 0.5 $10{,}000$-Draw Monte Carlo Speculative Distributional Equivalence Audit (`audit_speculative_distribution.py`, Real `Qwen2.5-1.5B`, $T=1.0$, `top_k=32`, $H(p)=1.881\text{ nats}$, i.i.d. Ref. $\text{TVD}=0.0128$)

| Draft Atoms $m$ | Raw Logit Diff $\|\ell_p - \ell_q\|_2$ | Draft Dist. $\text{TVD}(q, p)$ | Analytical Marginal Err | Empirical $\alpha$ ($\mathbb{E}[\alpha]$) | 10k-Draw $\text{TVD}(\hat{p}_{\text{spec}}, p)$ | Pearson $\chi^2$ (`df=16`) | Goodness-of-Fit $p$-value |
|---:|---:|---:|---:|---:|---:|---:|---:|
| **$m = 2$** | `26.001` | `0.0443` | `2.9e-17` | `0.9539` (`0.9557`) | **0.0129** | `19.83` | **0.2280** |
| **$m = 4$** | `16.984` | `0.0268` | `3.0e-17` | `0.9721` (`0.9732`) | **0.0096** | `15.58` | **0.4823** |
| **$m = 8$** | `8.934` | `0.0085` | `6.2e-17` | `0.9914` (`0.9915`) | **0.0080** | `4.40` | **0.9980** |
| **$m = 12$** | `4.534` | `0.0039` | `2.8e-17` | `0.9957` (`0.9961`) | **0.0110** | `22.50` | **0.1279** |

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

### 7.3 100% GPU `Qwen/Qwen2.5-7B-Instruct` (4-bit NF4) Comparative Audit & Avellaneda–Stoikov Control (`benchmarks/comparative_audit`)
Evaluated 100% on `cuda:0` (`NVIDIA GeForce RTX 4060 Laptop GPU`, `6.5 GiB` budget, zero CPU offload, `~5.35 GB` allocated VRAM) using `bitsandbytes` 4-bit NF4 weights, `artifacts/real_qwen_7b_audit_artifacts.pt` (`28` layers $\times$ `4` KV heads: `6` `ROLE_RETRIEVAL_FP16`, `100` `ROLE_PEAKED_H2O`, `6` `ROLE_DIFFUSE_LEXICO`), fused Triton decode+RoPE, in-place batch-flush slab, and closed-loop **Avellaneda–Stoikov (2008)** inventory control ($\text{skew}_t = q_t \gamma \sigma_t^2$, $\delta_t^* = \frac{1}{2}\gamma\sigma_t^2 + \frac{1}{\gamma}\ln(1 + \gamma/\kappa)$):

| Method (`Qwen/Qwen2.5-7B-Instruct` 4-bit NF4) | Needle Retrieval Acc | Context-Drift Top-1 (`P(' ditch')`) | Multi-Fact QA (`long_context_quality`) | KV Savings @ `4K` | Decode Throughput @ `4K, gen=128` |
|---|---:|---:|---:|---:|---:|
| **Full FP16** (`B0_full_fp16`) | **100.0%** | **100.0%** (`1.0000`) | **100.0%** | 0.0% | **24.43 tok/s** *(up from 19.66)* |
| **Standalone H2O** (`B3_standalone_h2o`) | 50.0%–55.6% | 0.0% (`0.1654`) | 15.6%–21.9% | 78.5% | **24.03 tok/s** *(up from 13.74, `1.75x`)* |
| **Standalone Lexico** (`B3_standalone_lexico`) | **100.0%** | 0.0% (`0.0122`) | 9.4%–12.5% | **81.6%** | **13.50 tok/s** *(up from 4.67, `2.89x`)* |
| **StoikovKV Hybrid** (`B4_stoikovkv_hybrid`) | **100.0%** | **100.0%** (`1.0000`) | **43.8%** *(`3.5x` over static)* | **78.8%** | **19.33 tok/s** *(up from 13.27, `1.46x`)* |

---

## 8. Fused Triton Sparse Decode + Half-Split RoPE Kernel Benchmark (`stoikov_kv.triton_kernels`)

Evaluated on `NVIDIA GeForce RTX 4060 Laptop GPU` (`s = 16` atoms, `head_dim = 128`, `200` timed repetitions after `10` warmups), comparing eager PyTorch (`F.embedding` + `float32` broadcast reduction) against single-launch `_lexico_reconstruct_kernel` (direct `int16`/`int8`/`fp16` register accumulation with zero intermediate `[N, s, d]` HBM allocation):

| Sequence Length ($N$) | PyTorch Eager Latency (Temp VRAM) | Fused Triton Latency (Temp VRAM) | Kernel Speedup | Temp VRAM Reduction | Reconstruction Cosine vs. PyTorch |
|---:|---:|---:|---:|---:|---:|
| **1,024** | `0.097 ms` (`+16.75 MB`) | **`0.070 ms`** (`+0.50 MB`) | **1.38$\times$** | **33.5$\times$ less** | `1.0000000` (1 ULP FP16) |
| **4,096** | `0.621 ms` (`+67.00 MB`) | **`0.072 ms`** (`+2.00 MB`) | **8.68$\times$** | **33.5$\times$ less** | `1.0000000` (1 ULP FP16) |
| **16,384** | `2.500 ms` (`+268.12 MB`) | **`0.095 ms`** (`+8.00 MB`) | **26.20$\times$** | **33.5$\times$ less** | `1.0000000` (1 ULP FP16) |

- **End-to-End 28-Layer Standalone GQA (`1.5B` FP16 @ `ctx = 4,096, gen = 128`):**
  - `standalone_lexico`: **`5.49 -> 19.47 tok/s` (`3.55x` faster)** at `81.6%` KV savings
  - `hybrid` (`StoikovKV`): **`15.36 -> 22.16 tok/s` (`1.44x` faster)** at `78.6%` KV savings
  - `full_fp16`: **`24.64 -> 42.96 tok/s` (`1.74x` faster)**

---

## 9. Multi-Architecture GPU Engine, Microstructure Extensions, Real `Qwen2.5-7B` & `Inkling-2-Mini` (`bench_poc_7b_and_inkling.py`)

Evaluated after adding unified shared-storage decode slabs (`_bind_unified_slab`), single-launch 3D multi-head `triton_fused_kv_lexico_materialize`, 64-token batch-deferred `triton_fused_write_probe`, multi-architecture hardware adaptation (`CUDA sm_80..sm_120`, `ROCm gfx942`, `Intel XPU`, `CPU`), Hawkes/GLFT/Almgren–Chriss/VPIN microstructure controls, zero-SFT `MeMo` bidirectional KG2KV consolidation (`arXiv:2605.15156`), and full defense-in-depth security hardening (`48/48` tests passing):

### 9.1 Kernel & Controlled 28-Layer GQA Decode Speedups (`RTX 4060 Laptop GPU, sm_89`)
- **`triton_fused_kv_lexico_materialize` (`4K` tokens, `H = 8`):** `1.508 ms -> 0.199 ms` (**`7.57x` faster**, max abs error `0.0e+00`)
- **64-Token Batch-Deferred `triton_fused_write_probe`:** `6.943 ms -> 0.024 ms` (**`288.85x` faster**)
- **Controlled 28-Layer GQA Decode (`ctx = 4,096, gen = 128`):**
  - `StoikovKV Hybrid`: **`50.67 -> 89.58 tok/s`** (`19.73 -> 11.16 ms/tok`, **`+76.8%` throughput**)
  - `Standalone Lexico`: **`29.37 -> 80.73 tok/s`** (`34.05 -> 12.39 ms/tok`, **`2.75x` faster**)
  - `Standalone H2O`: **`63.59 -> 101.22 tok/s`** (**`+59.2%`**)
  - `Full FP16`: **`75.54 -> 117.30 tok/s`** (**`+55.3%`**)

### 9.2 Real `Qwen/Qwen2.5-7B-Instruct` (4-bit NF4, 100% GPU, `ctx = 2,063, gen = 96`)
| Policy | Prior Speed (`tok/s`) | **Current Speed (`tok/s`)** | Latency (`ms/tok`) | Speedup vs. Prior | NIAH (`Early/Mid/Late`) | Physical = Logical KV (`MB`) | KV Savings |
|---|---:|---:|---:|---:|---|---:|---:|
| **Full FP16** | `24.43` | **`38.63`** | `25.89` | `1.58x` | **3/3 (100.0%)** | `168.16 MB` | `0.0%` |
| **Standalone H2O** | `24.03` | **`38.97`** | `25.66` | `1.62x` | 1/3 (33.3%) | `28.71 MB` | `82.9%` |
| **Standalone Lexico** | `13.50` | **`35.10`** | `28.49` | **`2.60x`** | 2/3 (66.7%) | `35.66 MB` | `78.8%` |
| **StoikovKV Hybrid** | `19.33` | **`36.85`** | `27.14` | **`1.91x` (`95.4%` FP16)** | **3/3 (100.0%)** | **`35.33 MB`** | **`79.0%`** |

### 9.3 Real-Weights `Inkling-2-Mini` (`48`-Pass Looped GQA, `384` KV Heads/Token, `ctx = 2,048, gen = 64`)
| Policy (`Inkling-2-Mini` 48L) | Decode (`tok/s`) | Latency (`ms/tok`) | Physical = Logical KV (`MB`) | KV Savings | Next-Token Logit Cosine vs. FP16 |
|---|---:|---:|---:|---:|---:|
| **Full FP16** | `53.66` | `18.63` | `395.81 MB` | `0.0%` | `1.0000` |
| **Standalone H2O** | `53.05` | `18.85` | `71.25 MB` | `82.0%` | `0.9784` |
| **Standalone Lexico** | `38.99` | `25.65` | `86.96 MB` | `78.0%` | **`0.9960`** |
| **StoikovKV Hybrid** | `36.04` | `27.75` | **`88.82 MB`** | **`77.6%`** | **`0.9837`** |
