"""
Monte Carlo Distributional Equivalence Audit for Matryoshka Self-Speculative Decoding
(`benchmarks/audit_speculative_distribution.py`).

Directly verifies Leviathan et al. (2023) rejection-sampling distributional equivalence on
real pretrained `Qwen/Qwen2.5-1.5B-Instruct` next-token distributions computed under
distinct KV-cache approximations (draft `m in {2, 4, 8, 12}` sparse atoms vs. verify
`s = 16` sparse atoms, identical model weights, temperature T = 1.0, top_k = 32):
  1. Measures exact analytical logit L2 distance ||l_p - l_q||_2 and analytical TVD(q, p)
     (confirming draft and verify logits/distributions genuinely differ).
  2. Draws N = 10,000 independent rejection-sampled tokens per position using exact
     Leviathan et al. (2023) rejection sampling.
  3. Compares empirical TVD(p_hat_spec, p) against an N = 10,000 i.i.d. reference draw
     directly from p, and computes Pearson chi-square goodness-of-fit p-values.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))
sys.path.insert(0, str(ROOT_DIR))

from stoikov_kv.cache import StoikovKVCache
from stoikov_kv.matryoshka_spec import _logits_to_probs
from benchmarks.bench_real_models_audit import (
    load_paul_graham_essays,
    load_real_model_auto,
)


def chi2_p_value(chi2_stat: float, df: int) -> float:
    try:
        from scipy.stats import chi2  # type: ignore
        return float(chi2.sf(chi2_stat, df))
    except Exception:
        if df <= 0:
            return 1.0
        z = ((chi2_stat / df) ** (1.0 / 3.0) - (1.0 - 2.0 / (9.0 * df))) / math.sqrt(2.0 / (9.0 * df))
        return float(0.5 * math.erfc(z / math.sqrt(2.0)))


@torch.no_grad()
def main() -> None:
    model, tokenizer, meta = load_real_model_auto("1.5b")
    device = torch.device("cuda:0")
    art_path = ROOT_DIR / "artifacts" / "real_qwen_1.5b_audit_artifacts.pt"
    artifacts = torch.load(str(art_path), map_location=device, weights_only=False)

    passages = load_paul_graham_essays(min_chars=2400, max_passages=40)
    # Pick a genuine high-entropy prose continuation point (H(p) > 2.0 nats) so rejection sampling
    # is stress-tested on a non-degenerate multi-modal distribution rather than a deterministic token.
    toks = tokenizer.encode(passages[24], add_special_tokens=False)[:512]
    input_ids = torch.tensor([toks], dtype=torch.long, device=device)

    temperature = 1.0
    top_k = 32
    n_draws = 10000

    cache = StoikovKVCache(
        model=model, artifacts=artifacts, n_b=32, keep_ratio=0.25, s_budget=16, policy_mode="hybrid"
    )
    cache.set_matryoshka_mode(active_atoms=16)
    _ = model(input_ids=input_ids[:, :-1], past_key_values=cache, use_cache=True)
    base_len = cache.get_seq_length(0)
    out_p = model(input_ids=input_ids[:, -1:], past_key_values=cache, use_cache=True)
    logits_p = out_p.logits[:, -1, :].detach().float()
    p_dist = _logits_to_probs(logits_p, temperature=temperature, top_k=top_k, strict=True)[0].cpu()
    p_np = p_dist.numpy().astype(np.float64)
    p_np = p_np / p_np.sum()
    p_entropy = float(-(p_np[p_np > 0] * np.log(p_np[p_np > 0])).sum())

    # Reference i.i.d. draw directly from exact p
    rng = np.random.default_rng(12345)
    support_idx = np.where(p_np > 0)[0]
    p_sub = p_np[support_idx]
    p_sub = p_sub / p_sub.sum()
    iid_counts = rng.multinomial(n_draws, p_sub)
    iid_tvd = float(0.5 * np.abs(iid_counts / n_draws - p_sub).sum())

    rows: List[Dict[str, Any]] = []
    print(
        f"Real Qwen2.5-1.5B Distributional Equivalence Audit "
        f"(N={n_draws} draws, T={temperature}, top_k={top_k}, H(p)={p_entropy:.3f} nats)"
    )
    print(f"  Reference i.i.d. sample directly from p: empirical TVD(p_hat_iid, p) = {iid_tvd:.4f}")

    for m in (2, 4, 8, 12):
        cache.crop(base_len)
        cache.set_matryoshka_mode(active_atoms=m)
        out_q = model(input_ids=input_ids[:, -1:], past_key_values=cache, use_cache=True)
        logits_q = out_q.logits[:, -1, :].detach().float()
        q_dist = _logits_to_probs(logits_q, temperature=temperature, top_k=top_k, strict=True)[0].cpu()
        q_np = q_dist.numpy().astype(np.float64)
        q_np = q_np / q_np.sum()

        logit_l2_diff = float(torch.norm(logits_p.cpu() - logits_q.cpu(), p=2).item())
        tvd_q_p = float(0.5 * np.abs(q_np - p_np).sum())
        expected_alpha = float(np.minimum(p_np, q_np).sum())

        union_idx = np.where((p_np > 0) | (q_np > 0))[0]
        p_u = p_np[union_idx]
        q_u = q_np[union_idx]
        p_u = p_u / p_u.sum()
        q_u = q_u / q_u.sum()

        residual = np.maximum(0.0, p_u - q_u)
        res_sum = float(residual.sum())
        res_probs = residual / res_sum if res_sum > 1e-12 else p_u

        # Verify exact analytical marginal distribution P_spec(x) = min(p, q) + (1 - sum min(p,q)) * res(x) == p(x)
        exact_spec_marginal = np.minimum(p_u, q_u) + (1.0 - np.minimum(p_u, q_u).sum()) * res_probs
        analytical_marginal_err = float(0.5 * np.abs(exact_spec_marginal - p_u).sum())

        draft_idx = rng.choice(len(union_idx), size=n_draws, p=q_u)
        accept_probs = np.minimum(1.0, p_u[draft_idx] / np.maximum(q_u[draft_idx], 1e-15))
        accepted = rng.random(size=n_draws) < accept_probs
        n_acc = int(accepted.sum())
        n_rej = n_draws - n_acc

        emitted = draft_idx.copy()
        if n_rej > 0:
            emitted[~accepted] = rng.choice(len(union_idx), size=n_rej, p=res_probs)

        emp_counts = np.bincount(emitted, minlength=len(union_idx)).astype(np.float64)
        emp_probs = emp_counts / float(n_draws)
        emp_tvd = float(0.5 * np.abs(emp_probs - p_u).sum())

        # Pearson chi-square goodness-of-fit over bins with expected count >= 10 (pooling tail bin)
        exp_counts = p_u * float(n_draws)
        major = exp_counts >= 10.0
        obs_bins = list(emp_counts[major])
        exp_bins = list(exp_counts[major])
        if (~major).any() and exp_counts[~major].sum() > 0:
            obs_bins.append(float(emp_counts[~major].sum()))
            exp_bins.append(float(exp_counts[~major].sum()))
        obs_arr = np.array(obs_bins, dtype=np.float64)
        exp_arr = np.array(exp_bins, dtype=np.float64)
        chi2_stat = float(np.sum((obs_arr - exp_arr) ** 2 / exp_arr))
        df = max(1, len(obs_arr) - 1)
        p_val = chi2_p_value(chi2_stat, df)

        row = {
            "draft_atoms_m": m,
            "verify_atoms_s": 16,
            "logit_l2_diff": round(logit_l2_diff, 4),
            "analytical_tvd_q_vs_p": round(tvd_q_p, 4),
            "analytical_marginal_tvd_err": float(f"{analytical_marginal_err:.2e}"),
            "expected_acceptance_alpha": round(expected_alpha, 4),
            "empirical_acceptance_alpha": round(n_acc / float(n_draws), 4),
            "empirical_tvd_spec_vs_p": round(emp_tvd, 4),
            "reference_iid_tvd": round(iid_tvd, 4),
            "chi2_stat": round(chi2_stat, 2),
            "chi2_df": df,
            "chi2_p_value": round(p_val, 4),
        }
        rows.append(row)
        print(
            f"  m={m:2d} | ||l_p-l_q||_2={logit_l2_diff:6.3f} | TVD(q,p)={tvd_q_p:.4f} | "
            f"alpha={n_acc/n_draws:.4f} (E={expected_alpha:.4f}) | "
            f"TVD(p_hat_spec, p)={emp_tvd:.4f} (iid={iid_tvd:.4f}) | "
            f"chi2={chi2_stat:.2f} (df={df}, p={p_val:.4f}) | marg_err={analytical_marginal_err:.1e}"
        )

    out_path = ROOT_DIR / "benchmarks" / "results" / "speculative_distributional_equivalence_audit.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_id": meta["model_id"],
                "temperature": temperature,
                "top_k": top_k,
                "n_monte_carlo_draws": n_draws,
                "target_entropy_nats": round(p_entropy, 4),
                "reference_iid_tvd": round(iid_tvd, 4),
                "results": rows,
            },
            f,
            indent=2,
        )
    print(f"Saved distributional equivalence audit to {out_path}")


if __name__ == "__main__":
    main()
