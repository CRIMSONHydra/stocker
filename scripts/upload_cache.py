#!/usr/bin/env python3
"""Push the local .cache/council/ directory to an HF dataset repo so the
env Space can download it at startup (instead of bundling in git).

Usage:
    export HF_TOKEN=hf_xxx
    python scripts/upload_cache.py                          # default repo
    python scripts/upload_cache.py --repo Hydr473/stocker-cache
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / ".cache"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", default=os.getenv("STOCKER_CACHE_REPO", "Hydr473/stocker-cache"),
                   help="HF dataset repo (default: Hydr473/stocker-cache)")
    p.add_argument("--token", default=os.getenv("HF_TOKEN"),
                   help="HF write token (default: $HF_TOKEN env)")
    args = p.parse_args()

    if not args.token:
        sys.exit("HF_TOKEN not set (and no --token provided). Aborting.")

    if not (CACHE_DIR / "council").is_dir():
        sys.exit(f"No cache to upload — {CACHE_DIR/'council'} doesn't exist. "
                 "Run scripts/precache_endpoint.py first.")

    n_files = sum(1 for _ in (CACHE_DIR / "council").rglob("*.json"))
    if n_files == 0:
        sys.exit("Cache directory is empty. Run scripts/precache_endpoint.py first.")

    from huggingface_hub import HfApi

    api = HfApi(token=args.token)
    api.create_repo(repo_id=args.repo, repo_type="dataset", exist_ok=True)
    print(f"Uploading {n_files} cache entries → https://huggingface.co/datasets/{args.repo}")

    api.upload_folder(
        folder_path=str(CACHE_DIR),
        repo_id=args.repo,
        repo_type="dataset",
        allow_patterns=["council/**/*.json"],
        commit_message=f"Update council cache ({n_files} entries)",
    )
    print(f"Done. The env Space will pull this on next startup if STOCKER_CACHE_REPO={args.repo}.")


if __name__ == "__main__":
    main()
