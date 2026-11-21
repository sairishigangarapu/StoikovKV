"""
Extends the real `Qwen/Qwen2.5-7B-Instruct` Paul Graham Essays NIAH evaluation from
N = 8 independent essays (trials 0..7) to N = 16 strictly independent unseen essays
(appending trials 8..15 with 8 new unseen essays and 8 new unseen 5-digit passkeys),
reusing the exact calibrated `artifacts/real_qwen_7b_audit_artifacts.pt` and OOM-safe
GPU/CPU/NVMe offloading (`max_memory={0: "5.5GiB", "cpu": "3.5GiB"}`).
"""

from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))
sys.path.insert(0, str(ROOT_DIR))

from stoikov_kv.cache import StoikovKVCache
from benchmarks.bench_cache_ablation import solve_equal_budget_params
from benchmarks.bench_real_models_audit import (
    build_wikitext_niah_prompt,
    load_paul_graham_essays,
    load_real_model_auto,
    wilson_ci_95,
)


@torch.no_grad()
def main() -> None:
    res_path = ROOT_DIR / "benchmarks" / "results" / "real_qwen_7b_paul_graham_audit_results.json"
    prev = json.loads(res_path.read_text(encoding="utf-8"))
    art_path = ROOT_DIR / "artifacts" / "real_qwen_7b_audit_artifacts.pt"
    assert art_path.exists(), f"Missing calibrated 7B artifacts at {art_path}"

    passages = load_paul_graham_essays(min_chars=2400, max_passages=40)
    model, tokenizer, meta = load_real_model_auto("7b")
    artifacts = torch.load(str(art_path), map_location="cuda:0", weights_only=False)
    emb_dev = next(model.model.embed_tokens.parameters()).device

    depths = [0.10, 0.25, 0.50, 0.75]
    unseen_passkeys = [
        "37492", "81563", "24907", "65814", "49271",
        "73618", "18459", "92046", "56382", "30795",
        "68124", "45903", "87261", "19538", "74029",
        "52864", "36197", "90415", "28736", "61384",
    ]

    # Trials 8..15: 8 brand-new independent Paul Graham essays & 8 brand-new unseen passkeys
    new_trials: List[Tuple[torch.Tensor, Dict[str, Any]]] = []
    for idx in range(8, 16):
        p_txt = passages[idx % len(passages)]
        depth = depths[idx % len(depths)]
        code = unseen_passkeys[idx % len(unseen_passkeys)]
        p_ids, t_meta = build_wikitext_niah_prompt(
            tokenizer, p_txt, target_tokens=512, depth_ratio=depth, passkey=code
        )
        new_trials.append((p_ids, t_meta))

    seq_len_ref = int(np.mean([m["total_tokens"] for _, m in new_trials]))
    budget_params = solve_equal_budget_params(
        artifacts["head_roles"].detach().cpu(),
        seq_len=seq_len_ref,
        n_b=32,
        target_ratio=0.22,
        head_dim=128,
    )

    policies = (
        ("Full FP16", "full_fp16", False),
        ("Standalone H2O", "standalone_h2o", False),
        ("SnapKV", "snapkv", False),
        ("StoikovKV Hybrid", "hybrid", True),
    )

    print("=" * 90)
    print("Extending Real Qwen2.5-7B-Instruct Paul Graham Essays NIAH: Appending Trials 8..15 (Total N=16)")
    print("=" * 90)

    ref_logits_list: List[torch.Tensor] = []
    merged_niah: Dict[str, Any] = {}

    for short_name, mode, hmm_on in policies:
        bp = budget_params[mode]
        kr = float(bp["keep_ratio"])
        sb = int(bp["s_budget"])

        new_hits = 0
        new_hits_by_depth: Dict[str, List[int]] = {f"{int(d*100)}%": [] for d in depths}
        mem_pcts: List[float] = []
        dict_pcts: List[float] = []
        cos_list: List[float] = []
        decode_toks = 0
        decode_sec = 0.0

        for t_idx, (p_ids, t_meta) in enumerate(new_trials):
            p_ids_d = p_ids.to(emb_dev)
            passkey = t_meta["passkey"]
            cache = StoikovKVCache(
                model=model,
                artifacts=artifacts,
                n_b=32,
                keep_ratio=kr,
                s_budget=sb,
                policy_mode=mode,
                enable_hmm_decay=hmm_on,
            )
            _ = model(input_ids=p_ids_d[:, :-1], past_key_values=cache, use_cache=True)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = model(input_ids=p_ids_d[:, -1:], past_key_values=cache, use_cache=True)
            logits0 = out.logits[0, -1].detach().float().cpu()
            if mode == "full_fp16":
                ref_logits_list.append(logits0)
                cos_list.append(1.0)
            else:
                cos_list.append(
                    float(F.cosine_similarity(ref_logits_list[t_idx].unsqueeze(0), logits0.unsqueeze(0), dim=-1).item())
                )

            gen_ids = [int(logits0.argmax().item())]
            cur = torch.tensor([[gen_ids[0]]], device=emb_dev)
            for _ in range(4):
                o_n = model(input_ids=cur, past_key_values=cache, use_cache=True)
                nxt = int(o_n.logits[0, -1].argmax().item())
                gen_ids.append(nxt)
                cur = torch.tensor([[nxt]], device=emb_dev)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            decode_sec += time.perf_counter() - t0
            decode_toks += 5

            gen_str = tokenizer.decode(gen_ids)
            hit = int(passkey in gen_str)
            new_hits += hit
            new_hits_by_depth[f"{int(t_meta['depth_ratio']*100)}%"].append(hit)

            cache.crop(int(p_ids_d.shape[1]))
            telem = cache.get_telemetry_summary()
            mem_pcts.append(float(telem["memory_ratio_vs_fp16"]) * 100.0)
            dict_pcts.append(float(telem["memory_ratio_incl_dictionary"]) * 100.0)
            del cache
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Merge previous 8 trials (trials 0..7) with new 8 trials (trials 8..15) -> N = 16
        old_r = prev["niah_results"][short_name]
        tot_hits = int(old_r["hits"]) + new_hits
        tot_n = int(old_r["n_trials"]) + 8
        ci_lo, ci_hi = wilson_ci_95(tot_hits, tot_n)

        merged_depth: Dict[str, str] = {}
        for dk in ("10%", "25%", "50%", "75%"):
            old_h = int(str(old_r["accuracy_by_depth"][dk]).split("/")[0])
            old_d = int(str(old_r["accuracy_by_depth"][dk]).split("/")[1].split()[0])
            nh = sum(new_hits_by_depth[dk])
            nd = len(new_hits_by_depth[dk])
            ch, cd = old_h + nh, old_d + nd
            merged_depth[dk] = f"{ch}/{cd} ({100.0 * ch / max(1, cd):.0f}%)"

        avg_kv = round(0.5 * float(old_r["kv_mem_pct"]) + 0.5 * float(np.mean(mem_pcts)), 2)
        avg_dict = round(0.5 * float(old_r["total_mem_incl_dict_pct"]) + 0.5 * float(np.mean(dict_pcts)), 2)
        avg_cos = round(0.5 * float(old_r["next_token_logit_cosine"]) + 0.5 * float(np.mean(cos_list)), 4)
        new_tps = decode_toks / max(decode_sec, 1e-6)
        avg_tps = round(0.5 * float(old_r["decode_tok_per_s"]) + 0.5 * new_tps, 2)

        merged_niah[short_name] = {
            "policy_mode": mode,
            "keep_ratio": kr,
            "s_budget": sb,
            "hits": tot_hits,
            "n_trials": tot_n,
            "accuracy_pct": round(100.0 * tot_hits / tot_n, 1),
            "wilson_95_ci_pct": [ci_lo, ci_hi],
            "accuracy_by_depth": merged_depth,
            "kv_mem_pct": avg_kv,
            "total_mem_incl_dict_pct": avg_dict,
            "next_token_logit_cosine": avg_cos,
            "decode_tok_per_s": avg_tps,
        }
        print(
            f"  {short_name:<18s} | New 8: {new_hits}/8 | Combined {tot_hits:2d}/{tot_n:2d} "
            f"({merged_niah[short_name]['accuracy_pct']:5.1f}%, 95% CI [{ci_lo:4.1f}%, {ci_hi:5.1f}%]) | "
            f"KV={avg_kv:6.2f}% (+Dict={avg_dict:6.2f}%) | Cos={avg_cos:.4f} | {avg_tps:5.2f} tok/s"
        )

    prev["niah_results"] = merged_niah
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump(prev, f, indent=2)
    print(f"\nUpdated {res_path} with full N=16 independent Paul Graham essays!")


if __name__ == "__main__":
    main()
