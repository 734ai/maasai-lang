#!/usr/bin/env python3
"""
Merge a LoRA adapter into its base model, then publish the merged output through
the canonical Hugging Face publisher.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Merge and publish a trained Maasai model")
    parser.add_argument(
        "--adapter_dir",
        type=str,
        default="outputs/maasai-en-mt-qlora",
        help="Path to the LoRA adapter directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/maasai-en-mt-merged",
        help="Path for the merged model output",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default="NorthernTribe-Research/maasai-en-mt",
        help="Target Hugging Face repo id",
    )
    parser.add_argument(
        "--private",
        action="store_true",
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
        "--skip_merge",
        action="store_true",
        help="Reuse an existing merged model directory",
    )
    return parser.parse_args()


def merge_adapter_to_base(adapter_dir: Path, output_dir: Path) -> None:
    """Merge LoRA weights into the base model and persist the merged checkpoint."""
    print(f"Merging adapter from {adapter_dir} into {output_dir}")
    model = AutoPeftModelForCausalLM.from_pretrained(
        str(adapter_dir),
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model = model.merge_and_unload()

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir))

    tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"Merged model saved to {output_dir}")


def split_repo_id(repo_id: str) -> tuple[str | None, str]:
    if "/" not in repo_id:
        return None, repo_id
    username, repo_name = repo_id.split("/", 1)
    return username, repo_name


def publish_model(project_root: Path, model_dir: Path, repo_id: str, private: bool, token: str | None) -> int:
    username, repo_name = split_repo_id(repo_id)
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
    if private:
        cmd.append("--private-model")
    if token:
        cmd.extend(["--token", token])
    return subprocess.run(cmd, cwd=project_root, check=False).returncode


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent
    adapter_dir = (project_root / args.adapter_dir).resolve()
    output_dir = (project_root / args.output_dir).resolve()

    if not args.skip_merge:
        if not adapter_dir.exists():
            print(f"Adapter directory not found: {adapter_dir}", file=sys.stderr)
            return 1
        merge_adapter_to_base(adapter_dir, output_dir)
    elif not output_dir.exists():
        print(f"Merged model directory not found: {output_dir}", file=sys.stderr)
        return 1

    return publish_model(project_root, output_dir, args.repo_id, args.private, args.token)


if __name__ == "__main__":
    raise SystemExit(main())
