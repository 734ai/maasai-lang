#!/usr/bin/env python3
"""
Push a local Maasai model directory to Hugging Face through the canonical publisher.

This wrapper keeps older entrypoints usable while ensuring model cards, scaffold
behavior, and Hub metadata anchors stay aligned with `scripts/publish_to_hf.py`.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Publish a local Maasai model directory to Hugging Face",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="outputs/maasai-en-mt-qlora",
        help="Path to the local model directory",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default="NorthernTribe-Research/maasai-en-mt",
        help="Target Hugging Face repo id",
    )
    parser.add_argument(
        "--private",
        type=lambda x: x.lower() in ("true", "1", "yes"),
        default=False,
        help="Create or update the target model repo as private",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Optional Hugging Face token override",
    )
    parser.add_argument(
        "--skip_validation",
        action="store_true",
        help="Accepted for backward compatibility; validation is handled by the publisher",
    )
    return parser.parse_args()


def split_repo_id(repo_id: str) -> tuple[str | None, str]:
    if "/" not in repo_id:
        return None, repo_id
    username, repo_name = repo_id.split("/", 1)
    return username, repo_name


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent
    model_dir = (project_root / args.model_path).resolve()

    if not model_dir.exists():
        print(f"Model directory not found: {model_dir}", file=sys.stderr)
        return 1

    username, repo_name = split_repo_id(args.repo_id)
    publisher = project_root / "scripts" / "publish_to_hf.py"
    cmd = [
        sys.executable,
        str(publisher),
        "--skip-space",
        "--skip-dataset",
        "--execute",
        "--create-model-repo",
        "--model-dir",
        str(model_dir),
        "--model-repo",
        repo_name,
    ]

    if username:
        cmd.extend(["--username", username])
    if args.private:
        cmd.append("--private-model")
    if args.token:
        cmd.extend(["--token", args.token])

    return subprocess.run(cmd, cwd=project_root, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
