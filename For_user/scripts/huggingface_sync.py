#!/usr/bin/env python3
"""Small Hugging Face Hub downloader used by the public evaluation bundle."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Download TongBench files from Hugging Face Hub.")
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--local-dir", required=True, type=Path)
    parser.add_argument("--allow-pattern", action="append", default=[])
    parser.add_argument("--max-workers", type=int, default=4)
    args = parser.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:
        raise SystemExit(
            "Missing Hugging Face Hub package. Install it with: "
            "python -m pip install -U huggingface_hub"
        ) from exc

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN") or None
    args.local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        local_dir=str(args.local_dir),
        allow_patterns=args.allow_pattern or None,
        token=token,
        max_workers=args.max_workers,
    )
    print(f"Downloaded {args.repo_id} into {args.local_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
