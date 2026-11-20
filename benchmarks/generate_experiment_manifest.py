"""
Generates a reproducible provenance manifest (`benchmarks/results/experiment_manifest.json`)
pinning the exact Git commit, SHA-256 source file hashes, pretrained HuggingFace model
revision hashes, external dataset metadata, and verified benchmark result files referenced
in the StoikovKV IEEE paper.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

import torch


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(root)).decode().strip()
    git_branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=str(root)).decode().strip()

    source_files = [
        "src/stoikov_kv/calibration.py",
        "src/stoikov_kv/cache.py",
        "src/stoikov_kv/elastic_lexico.py",
        "src/stoikov_kv/matryoshka_spec.py",
        "benchmarks/bench_real_models_audit.py",
        "benchmarks/bench_real_qwen_publication.py",
        "benchmarks/bench_cache_ablation.py",
        "benchmarks/bench_calibration_codec.py",
    ]
    source_hashes: Dict[str, str] = {}
    for rel in source_files:
        p = root / rel
        if p.exists():
            source_hashes[rel] = sha256_file(p)

    result_files = sorted((root / "benchmarks" / "results").glob("*.json"))
    result_hashes: Dict[str, str] = {
        str(p.relative_to(root)): sha256_file(p)
        for p in result_files
        if p.name != "experiment_manifest.json"
    }

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "git_branch": git_branch,
        "git_commit": git_commit,
        "hardware": {
            "os": platform.platform(),
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "gpu_vram_gb": round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 2)
            if torch.cuda.is_available()
            else 0.0,
        },
        "pretrained_models": {
            "Qwen/Qwen2.5-1.5B-Instruct": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
            "Qwen/Qwen2.5-3B-Instruct": "aa8e72537993ba99e69dfaafa59ed015b17504d1",
            "Qwen/Qwen2.5-7B-Instruct": "a09a35458c702b33eeacc393d103063234e8bc28",
        },
        "external_datasets": {
            "paul_graham_essays_niah": {
                "source": "gkamradt/LLMTest_NeedleInAHaystack (PaulGrahamEssays)",
                "num_essays": 50,
                "local_dir": "data/niah_external/paul_graham/needle-in-a-haystack-main/needlehaystack/PaulGrahamEssays",
            },
            "wikitext103": {
                "source": "Salesforce/wikitext (wikitext-103-raw-v1)",
                "local_dir": "data/wikitext/wikitext-103-raw-v1",
            },
        },
        "source_sha256": source_hashes,
        "result_json_sha256": result_hashes,
    }

    out_path = root / "benchmarks" / "results" / "experiment_manifest.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote experiment manifest to {out_path} ({len(source_hashes)} sources, {len(result_hashes)} result JSONs)")


if __name__ == "__main__":
    main()
