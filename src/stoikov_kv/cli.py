"""
Command-Line Interface for StoikovKV (`stoikov_kv/cli.py`).

Subcommands:
- `stoikov calibrate`: Calibrates query subspaces `U_Q`, head roles, and dictionaries `D`/`W_ridge`
  for a local or HuggingFace model without requiring `output_attentions=True`.
- `stoikov chat`: Runs single-prompt or interactive generation with `StoikovKVCache` and
  optional Matryoshka self-speculative decoding, printing live memory and bandwidth telemetry.
- `stoikov serve`: Starts the OpenAI-compatible HTTP server (`/v1/models`, `/v1/chat/completions`,
  `/v1/completions`, `/v1/telemetry`, `/health`).
- `stoikov bench`: Runs the benchmark suite (`all`, `calibration`, `cache`, `e2e`, `hikvp`).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from ._security import require_bounded_int, validate_safe_path
from .calibration import calibrate_model, load_model_and_tokenizer
from .server import StoikovInferenceEngine, create_http_server

_REPO_ROOT = Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stoikov",
        description="StoikovKV: Hybrid Evict-or-Encode KV Cache & Matryoshka Speculative Decoding CLI",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # 1. stoikov calibrate
    p_cal = sub.add_parser("calibrate", help="Calibrate U_Q, head roles, and dictionaries for a model")
    p_cal.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct", help="Model ID or local path")
    p_cal.add_argument("--out", default="artifacts/calibrated_artifacts.pt", help="Output .pt artifact path")
    p_cal.add_argument("--real-model", action="store_true", help="Load real HuggingFace/local weights")
    p_cal.add_argument("--load-in-4bit", action="store_true", help="Use 4-bit NF4 quantization")
    p_cal.add_argument("--trust-remote-code", action="store_true", help="Explicitly allow HuggingFace remote code")
    p_cal.add_argument("--subspace-rank", type=int, default=16, help="Query subspace rank r")
    p_cal.add_argument("--dict-atoms", type=int, default=1024, help="Dictionary atoms N")
    p_cal.add_argument("--dict-steps", type=int, default=60, help="Dictionary Adam steps")
    p_cal.add_argument("--max-seq-len", type=int, default=512, help="Calibration sequence length")
    p_cal.add_argument("--device", default=None, help="Device override (e.g., cuda, cpu)")

    # 2. stoikov chat
    p_chat = sub.add_parser("chat", help="Generate text with StoikovKVCache and print telemetry")
    p_chat.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct", help="Model ID or local path")
    p_chat.add_argument("--artifacts", default="artifacts/calibrated_artifacts.pt", help="Calibrated artifacts path")
    p_chat.add_argument("--prompt", default="Explain how hybrid KV-cache routing reduces memory bandwidth:", help="Prompt string")
    p_chat.add_argument("--max-new-tokens", type=int, default=32, help="Max new tokens to generate")
    p_chat.add_argument("--policy", default="hybrid", choices=["hybrid", "full_fp16", "standalone_h2o", "standalone_lexico"])
    p_chat.add_argument("--no-speculative", action="store_true", help="Disable Matryoshka speculative decoding")
    p_chat.add_argument("--real-model", action="store_true", help="Load real HuggingFace/local weights")
    p_chat.add_argument("--load-in-4bit", action="store_true", help="Use 4-bit NF4 quantization")
    p_chat.add_argument("--trust-remote-code", action="store_true", help="Explicitly allow HuggingFace remote code")
    p_chat.add_argument("--device", default=None, help="Device override (e.g., cuda, cpu)")

    # 3. stoikov serve
    p_srv = sub.add_parser("serve", help="Run OpenAI-compatible HTTP server (/v1/chat/completions)")
    p_srv.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct", help="Model ID or local path")
    p_srv.add_argument("--artifacts", default="artifacts/calibrated_artifacts.pt", help="Calibrated artifacts path")
    p_srv.add_argument("--host", default="127.0.0.1", help="Bind host")
    p_srv.add_argument("--port", type=int, default=8000, help="Bind port")
    p_srv.add_argument("--api-key", default=None, help="Optional Bearer API key (or set STOIKOV_API_KEY)")
    p_srv.add_argument("--policy", default="hybrid", choices=["hybrid", "full_fp16", "standalone_h2o", "standalone_lexico"])
    p_srv.add_argument("--no-speculative", action="store_true", help="Disable Matryoshka speculative decoding by default")
    p_srv.add_argument("--real-model", action="store_true", help="Load real HuggingFace/local weights")
    p_srv.add_argument("--load-in-4bit", action="store_true", help="Use 4-bit NF4 quantization")
    p_srv.add_argument("--trust-remote-code", action="store_true", help="Explicitly allow HuggingFace remote code")
    p_srv.add_argument("--device", default=None, help="Device override (e.g., cuda, cpu)")

    # 4. stoikov bench
    p_bench = sub.add_parser("bench", help="Run StoikovKV benchmark suites")
    p_bench.add_argument(
        "--suite",
        default="all",
        choices=["all", "calibration", "cache", "e2e", "hikvp"],
        help="Benchmark suite to run",
    )

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "calibrate":
        subspace_rank = require_bounded_int(args.subspace_rank, "subspace_rank", 1, 512)
        dict_atoms = require_bounded_int(args.dict_atoms, "dict_atoms", 1, 32767)
        dict_steps = require_bounded_int(args.dict_steps, "dict_steps", 1, 50000)
        max_seq_len = require_bounded_int(args.max_seq_len, "max_seq_len", 16, 131072)
        validate_safe_path(args.out)

        model, tokenizer = load_model_and_tokenizer(
            model_name=args.model,
            device=args.device,
            use_real_model=args.real_model,
            load_in_4bit=args.load_in_4bit,
            trust_remote_code=args.trust_remote_code,
        )
        artifacts = calibrate_model(
            model=model,
            tokenizer=tokenizer,
            save_path=args.out,
            subspace_rank=subspace_rank,
            dict_atoms=dict_atoms,
            dict_steps=dict_steps,
            max_seq_len=max_seq_len,
        )
        print(
            json.dumps(
                {
                    "saved_to": args.out,
                    "raw_spearman_mean": artifacts["probe_metrics"]["raw_spearman_mean"],
                    "sink_aug_recall_at_20_mean": artifacts["probe_metrics"]["sink_aug_recall_at_20_mean"],
                },
                indent=2,
            )
        )
        return 0

    if args.command == "chat":
        max_new_tokens = require_bounded_int(args.max_new_tokens, "max_new_tokens", 1, 4096)
        validate_safe_path(args.artifacts)
        engine = StoikovInferenceEngine.from_pretrained(
            model_name=args.model,
            artifacts_path=args.artifacts,
            device=args.device,
            use_real_model=args.real_model,
            load_in_4bit=args.load_in_4bit,
            policy_mode=args.policy,
            speculative=not args.no_speculative,
            trust_remote_code=args.trust_remote_code,
        )
        res = engine.generate_text(
            prompt=args.prompt,
            max_new_tokens=max_new_tokens,
            speculative=not args.no_speculative,
            policy_mode=args.policy,
        )
        print(
            json.dumps(
                {
                    "prompt": args.prompt,
                    "generated_text": res["text"],
                    "completion_tokens": res["completion_tokens"],
                    "telemetry": {
                        "seq_length": res["telemetry"]["seq_length"],
                        "fp16_bytes": res["telemetry"]["fp16_bytes"],
                        "compressed_bytes": res["telemetry"]["compressed_bytes"],
                        "physical_resident_bytes": res["telemetry"]["physical_resident_bytes"],
                        "memory_savings_pct": round(res["telemetry"]["memory_savings_pct"], 2),
                        "acceptance_rate": round(res["telemetry"]["acceptance_rate"], 4),
                        "tokens_per_sec": round(res["telemetry"]["tokens_per_sec"], 2),
                    },
                },
                indent=2,
            )
        )
        return 0

    if args.command == "serve":
        port = require_bounded_int(args.port, "port", 1, 65535)
        validate_safe_path(args.artifacts)
        engine = StoikovInferenceEngine.from_pretrained(
            model_name=args.model,
            artifacts_path=args.artifacts,
            device=args.device,
            use_real_model=args.real_model,
            load_in_4bit=args.load_in_4bit,
            policy_mode=args.policy,
            speculative=not args.no_speculative,
            trust_remote_code=args.trust_remote_code,
            api_key=args.api_key,
        )
        srv = create_http_server(engine, host=args.host, port=port)
        print(f"[StoikovKV] OpenAI-compatible server listening on http://{args.host}:{port}")
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            srv.server_close()
        return 0

    if args.command == "bench":
        bench_dir = _REPO_ROOT / "benchmarks"
        scripts = {
            "calibration": ["bench_calibration_codec.py"],
            "cache": ["bench_cache_ablation.py"],
            "e2e": ["bench_poc_7b_and_inkling.py"],
            "hikvp": ["bench_hikvp_atlas.py"],
            "all": [
                "bench_calibration_codec.py",
                "bench_cache_ablation.py",
                "bench_hikvp_atlas.py",
            ],
        }[args.suite]
        for sc_name in scripts:
            sc_path = validate_safe_path(bench_dir / sc_name, allowed_roots=[bench_dir])
            if not os.path.isfile(sc_path):
                raise FileNotFoundError(f"Trusted benchmark script not found: {sc_path}")
            rc = subprocess.call([sys.executable, str(sc_path)])
            if rc != 0:
                return rc
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
