#!/usr/bin/env python3
"""
Build and optionally push a Kaggle kernel that trains from HF and pushes
checkpoints back to the HF model repo.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

HF_TOKEN_ENV_VARS = ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HF_API_KEY")
WANDB_TOKEN_ENV_VARS = ("WANDB_API_KEY", "WANDB_TOKEN")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and push the Maasai Kaggle training kernel")
    parser.add_argument("--kernel-slug", default="maasai-daily-hf-training")
    parser.add_argument("--title", default=None)
    parser.add_argument("--output-dir", default="dist/kaggle_kernel")
    parser.add_argument("--project-git-url", default="https://github.com/734ai/maasai-lang.git")
    parser.add_argument("--project-branch", default="main")
    parser.add_argument("--dataset-repo", default="NorthernTribe-Research/maasai-translation-corpus")
    parser.add_argument("--model-repo", default="NorthernTribe-Research/maasai-en-mt-staging")
    parser.add_argument("--bucket-uri", default="hf://buckets/NorthernTribe-Research/maasai-project-storage")
    parser.add_argument("--bucket-prefix", default="training_runs")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--work-dir", default="/kaggle/working/maasai-daily-hf")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=800)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--logging-steps", type=int, default=20)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=32)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--report-to", default="none")
    parser.add_argument("--story-seed-file", default="data/raw/maasai_story_generation_seed.jsonl")
    parser.add_argument("--max-bible-passages", type=int, default=48)
    parser.add_argument("--bible-passage-window", type=int, default=3)
    generation_group = parser.add_mutually_exclusive_group()
    generation_group.add_argument(
        "--augment-with-generation-tasks",
        dest="augment_with_generation_tasks",
        action="store_true",
        help="Include Maa generation/story tasks in the Kaggle training mixture",
    )
    generation_group.add_argument(
        "--no-augment-with-generation-tasks",
        dest="augment_with_generation_tasks",
        action="store_false",
        help="Keep the first Kaggle run translation-only",
    )
    parser.set_defaults(augment_with_generation_tasks=False)
    parser.add_argument("--private-model-repo", action="store_true")
    parser.add_argument("--torch-version", default="2.5.1")
    parser.add_argument("--torchvision-version", default="0.20.1")
    parser.add_argument("--torchaudio-version", default="2.5.1")
    parser.add_argument("--torch-index-url", default="https://download.pytorch.org/whl/cu118")
    parser.add_argument(
        "--require-4bit",
        action="store_true",
        help="Require 4-bit quantization in the Kaggle runtime instead of allowing fallback loads",
    )
    parser.add_argument(
        "--embed-local-hf-token",
        action="store_true",
        help="Embed the local Hugging Face token from huggingface-api-key.json into the private Kaggle kernel package",
    )
    parser.add_argument(
        "--embed-env-hf-token",
        action="store_true",
        help="Embed the Hugging Face token from CI/local environment variables into the private Kaggle kernel package",
    )
    parser.add_argument(
        "--embed-env-wandb-api-key",
        action="store_true",
        help="Embed the W&B API key from CI/local environment variables into the private Kaggle kernel package",
    )
    parser.add_argument("--public", action="store_true", help="Make the Kaggle kernel public (default is private)")
    parser.add_argument("--execute", action="store_true", help="Actually push the kernel to Kaggle")
    parser.add_argument("--status", action="store_true", help="Query Kaggle status after push")
    return parser.parse_args()


def default_title_from_slug(slug: str) -> str:
    words = [part for part in slug.replace("_", "-").split("-") if part]
    return " ".join(word.upper() if word.lower() == "hf" else word.capitalize() for word in words)


def load_kaggle_config(project_root: Path) -> dict:
    config_path = project_root / "kaggle.json"
    if not config_path.exists():
        raise FileNotFoundError("Missing kaggle.json in project root.")
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_hf_keyfile(project_root: Path) -> dict:
    config_path = project_root / "huggingface-api-key.json"
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def extract_wandb_key(raw: str) -> str | None:
    raw = raw.strip()
    if not raw:
        return None

    try:
        payload = json.loads(raw)
        if isinstance(payload, dict):
            for field in ("WANDB_API_KEY", "wandb_api_key", "key", "api_key", "token"):
                value = payload.get(field)
                if value:
                    return str(value).strip()
    except json.JSONDecodeError:
        pass

    match = re.search(r"(wandb_[A-Za-z0-9_]+)", raw)
    if match:
        return match.group(1)
    return None


def load_local_wandb_key(project_root: Path) -> str | None:
    path = project_root / "wandb-keys.json"
    if not path.exists():
        return None
    return extract_wandb_key(path.read_text(encoding="utf-8"))


def first_env_value(*names: str) -> str | None:
    for name in names:
        value = str(os.getenv(name, "")).strip()
        if value:
            return value
    return None


def resolve_kaggle_username(project_root: Path) -> str:
    username = first_env_value("KAGGLE_USERNAME")
    if username:
        return username
    config = load_kaggle_config(project_root)
    username = str(config.get("username", "")).strip()
    if username:
        return username
    raise RuntimeError("Kaggle username is missing. Set KAGGLE_USERNAME or provide kaggle.json.")


def resolve_hf_token(args: argparse.Namespace, project_root: Path) -> str | None:
    if args.embed_env_hf_token:
        token = first_env_value(*HF_TOKEN_ENV_VARS)
        if not token:
            raise RuntimeError(
                "--embed-env-hf-token was requested, but no HF token was found in "
                f"{', '.join(HF_TOKEN_ENV_VARS)}."
            )
        return token
    if args.embed_local_hf_token:
        token = str(load_hf_keyfile(project_root).get("key", "")).strip()
        if not token:
            raise RuntimeError(
                "--embed-local-hf-token was requested, but huggingface-api-key.json did not contain a key."
            )
        return token
    return None


def resolve_wandb_api_key(args: argparse.Namespace, project_root: Path) -> str | None:
    if args.report_to != "wandb":
        return None
    if args.embed_env_wandb_api_key:
        wandb_api_key = first_env_value(*WANDB_TOKEN_ENV_VARS)
        if not wandb_api_key:
            raise RuntimeError(
                "--embed-env-wandb-api-key was requested, but no W&B key was found in "
                f"{', '.join(WANDB_TOKEN_ENV_VARS)}."
            )
        return wandb_api_key
    return load_local_wandb_key(project_root)


def build_runtime_config(args: argparse.Namespace, project_root: Path) -> dict:
    wandb_api_key = resolve_wandb_api_key(args, project_root)
    hf_token = resolve_hf_token(args, project_root)

    return {
        "project_git_url": args.project_git_url,
        "project_branch": args.project_branch,
        "dataset_repo": args.dataset_repo,
        "model_repo": args.model_repo,
        "bucket_uri": args.bucket_uri,
        "bucket_prefix": args.bucket_prefix,
        "base_model": args.base_model,
        "work_dir": args.work_dir,
        "max_length": args.max_length,
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.num_train_epochs,
        "max_steps": args.max_steps,
        "save_steps": args.save_steps,
        "eval_steps": args.eval_steps,
        "logging_steps": args.logging_steps,
        "save_total_limit": args.save_total_limit,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "torch_version": args.torch_version,
        "torchvision_version": args.torchvision_version,
        "torchaudio_version": args.torchaudio_version,
        "torch_index_url": args.torch_index_url,
        "report_to": args.report_to,
        "augment_with_generation_tasks": args.augment_with_generation_tasks,
        "story_seed_file": args.story_seed_file,
        "max_bible_passages": args.max_bible_passages,
        "bible_passage_window": args.bible_passage_window,
        "private_model_repo": args.private_model_repo,
        "wandb_api_key": wandb_api_key,
        "hf_token": hf_token,
        "require_4bit": args.require_4bit,
    }


def build_kernel_metadata(username: str, args: argparse.Namespace) -> dict:
    return {
        "id": f"{username}/{args.kernel_slug}",
        "title": args.title or default_title_from_slug(args.kernel_slug),
        "code_file": "train_and_push.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "false" if args.public else "true",
        "enable_gpu": "true",
        "enable_internet": "true",
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }


def resolve_kaggle_cli(project_root: Path) -> str:
    venv_cli = project_root / ".venv" / "bin" / "kaggle"
    if venv_cli.exists():
        return str(venv_cli)
    return "kaggle"


def build_embedded_script(project_root: Path, runtime_config: dict) -> str:
    template = (project_root / "kaggle" / "train_and_push.py").read_text(encoding="utf-8")
    return template.replace("__RUNTIME_CONFIG_JSON__", json.dumps(runtime_config))


def run(cmd: list[str], *, env: dict[str, str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, env=env, check=True)


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent
    username = resolve_kaggle_username(project_root)
    if args.public and (args.embed_local_hf_token or args.embed_env_hf_token or args.embed_env_wandb_api_key):
        raise RuntimeError("Refusing to embed CI/local secrets into a public Kaggle kernel.")

    output_dir = Path(args.output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "train_and_push.py").write_text(
        build_embedded_script(project_root, build_runtime_config(args, project_root)),
        encoding="utf-8",
    )
    shutil.copy2(project_root / "kaggle" / "requirements-kaggle.txt", output_dir / "requirements-kaggle.txt")
    write_json(output_dir / "kernel-metadata.json", build_kernel_metadata(username, args))

    print(f"Prepared Kaggle kernel package at: {output_dir}")
    print(f"Kernel ref: {username}/{args.kernel_slug}")

    if not args.execute:
        print("Preview only. Re-run with --execute to push to Kaggle.")
        return 0

    env = os.environ.copy()
    env["KAGGLE_CONFIG_DIR"] = str(project_root)
    kaggle_cli = resolve_kaggle_cli(project_root)

    run([kaggle_cli, "kernels", "push", "-p", str(output_dir)], env=env)
    if args.status:
        run([kaggle_cli, "kernels", "status", f"{username}/{args.kernel_slug}"], env=env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
