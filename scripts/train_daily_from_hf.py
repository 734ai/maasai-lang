#!/usr/bin/env python3
"""
Daily resumable training entrypoint for Colab or scheduled GPU runners.

This script:
1. Authenticates to Hugging Face
2. Downloads dataset splits from the dataset repo
3. Restores the latest checkpoint from the model repo when present
4. Runs QLoRA training with frequent Hub checkpoint pushes
5. Optionally disconnects the Colab runtime when the session goal is reached
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download, login, snapshot_download


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run daily resumable Hugging Face training")
    parser.add_argument(
        "--dataset-repo",
        type=str,
        default=os.getenv("HF_DATASET_REPO", "NorthernTribe-Research/maasai-translation-corpus"),
    )
    parser.add_argument(
        "--model-repo",
        type=str,
        default=os.getenv("HF_MODEL_REPO", "NorthernTribe-Research/maasai-en-mt"),
    )
    parser.add_argument("--model-name", type=str, default=os.getenv("HF_BASE_MODEL", "Qwen/Qwen2.5-3B-Instruct"))
    parser.add_argument("--work-dir", type=str, default=os.getenv("HF_DAILY_WORKDIR", "/tmp/maasai-daily-hf"))
    parser.add_argument("--token", type=str, default=None)
    parser.add_argument("--private-model-repo", action="store_true")
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=800)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--logging-steps", type=int, default=20)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--require-4bit", action="store_true")
    parser.add_argument("--report-to", type=str, default="none")
    parser.add_argument("--hub-strategy", type=str, default="checkpoint")
    parser.add_argument("--story-seed-file", type=str, default="data/raw/maasai_story_generation_seed.jsonl")
    parser.add_argument("--max-bible-passages", type=int, default=48)
    parser.add_argument("--bible-passage-window", type=int, default=3)
    generation_group = parser.add_mutually_exclusive_group()
    generation_group.add_argument("--augment-with-generation-tasks", dest="augment_with_generation_tasks", action="store_true")
    generation_group.add_argument("--no-augment-with-generation-tasks", dest="augment_with_generation_tasks", action="store_false")
    parser.set_defaults(augment_with_generation_tasks=True)
    parser.add_argument("--disconnect-colab", action="store_true")
    return parser.parse_args()


def maybe_get_colab_secret(name: str) -> str | None:
    try:
        from google.colab import userdata  # type: ignore

        value = userdata.get(name)
        return value or None
    except Exception:
        return None


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


def maybe_get_local_wandb_key(project_root: Path) -> str | None:
    path = project_root / "wandb-keys.json"
    if not path.exists():
        return None
    return extract_wandb_key(path.read_text(encoding="utf-8"))


def resolve_token(args: argparse.Namespace) -> str:
    if args.token:
        return args.token
    for env_var in ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HF_API_KEY"):
        value = os.getenv(env_var)
        if value:
            return value
    for secret_name in ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HF_API_KEY"):
        secret_value = maybe_get_colab_secret(secret_name)
        if secret_value:
            return secret_value
    raise RuntimeError("No HF token found. Set HF_TOKEN or pass --token.")


def maybe_configure_wandb(args: argparse.Namespace, project_root: Path) -> None:
    if args.report_to != "wandb":
        return
    for env_var in ("WANDB_API_KEY",):
        if os.getenv(env_var):
            return
    secret_value = maybe_get_colab_secret("WANDB_API_KEY")
    if secret_value:
        os.environ["WANDB_API_KEY"] = secret_value
        return

    local_value = maybe_get_local_wandb_key(project_root)
    if local_value:
        os.environ["WANDB_API_KEY"] = local_value


def write_run_manifest(
    manifest_path: Path,
    *,
    args: argparse.Namespace,
    train_file: Path,
    valid_file: Path,
    output_dir: Path,
    resume_checkpoint: str | None,
) -> None:
    manifest = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "dataset_repo": args.dataset_repo,
        "model_repo": args.model_repo,
        "model_name": args.model_name,
        "train_file": str(train_file),
        "valid_file": str(valid_file),
        "output_dir": str(output_dir),
        "resume_checkpoint": resume_checkpoint,
        "max_steps": args.max_steps,
        "save_steps": args.save_steps,
        "eval_steps": args.eval_steps,
        "report_to": args.report_to,
        "require_4bit": args.require_4bit,
        "hub_strategy": args.hub_strategy,
        "augment_with_generation_tasks": args.augment_with_generation_tasks,
        "story_seed_file": args.story_seed_file,
        "max_bible_passages": args.max_bible_passages,
        "bible_passage_window": args.bible_passage_window,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def ensure_model_repo(api: HfApi, token: str, repo_id: str, private: bool) -> None:
    api.create_repo(
        repo_id=repo_id,
        repo_type="model",
        private=private,
        exist_ok=True,
        token=token,
    )


def download_dataset_splits(repo_id: str, token: str, dataset_root: Path) -> tuple[Path, Path]:
    dataset_root.mkdir(parents=True, exist_ok=True)

    def download_split(split: str) -> Path:
        for candidate in (f"data/{split}.jsonl", f"{split}.jsonl"):
            try:
                downloaded = hf_hub_download(
                    repo_id=repo_id,
                    repo_type="dataset",
                    filename=candidate,
                    token=token,
                    local_dir=str(dataset_root),
                    local_dir_use_symlinks=False,
                )
                return Path(downloaded)
            except Exception:
                continue
        raise FileNotFoundError(f"Could not find {split}.jsonl in dataset repo {repo_id}")

    train_file = download_split("train")
    valid_file = download_split("valid")
    return train_file, valid_file


def checkpoint_step(path: Path) -> int:
    match = re.search(r"checkpoint-(\d+)", path.name)
    return int(match.group(1)) if match else -1


def restore_latest_checkpoint(repo_id: str, token: str, output_dir: Path) -> str | None:
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            local_dir=str(output_dir),
            local_dir_use_symlinks=False,
            token=token,
            allow_patterns=[
                "checkpoint-*",
                "checkpoint-*/*",
                "README.md",
                "adapter_config.json",
                "tokenizer*",
                "special_tokens_map.json",
            ],
            ignore_patterns=["*.gguf"],
        )
    except Exception as exc:
        print(f"Checkpoint restore skipped: {exc}", file=sys.stderr)
        return None

    checkpoints = sorted(
        [path for path in output_dir.glob("checkpoint-*") if path.is_dir()],
        key=checkpoint_step,
    )
    if not checkpoints:
        return None
    return str(checkpoints[-1])


def build_train_command(
    args: argparse.Namespace,
    project_root: Path,
    train_file: Path,
    valid_file: Path,
    output_dir: Path,
    resume_checkpoint: str | None,
) -> list[str]:
    story_seed_file = Path(args.story_seed_file)
    if not story_seed_file.is_absolute():
        story_seed_file = project_root / story_seed_file

    cmd = [
        sys.executable,
        str(project_root / "scripts" / "train_qlora.py"),
        "--model_name",
        args.model_name,
        "--train_file",
        str(train_file),
        "--valid_file",
        str(valid_file),
        "--output_dir",
        str(output_dir),
        "--max_length",
        str(args.max_length),
        "--learning_rate",
        str(args.learning_rate),
        "--num_train_epochs",
        str(args.num_train_epochs),
        "--per_device_train_batch_size",
        str(args.per_device_train_batch_size),
        "--per_device_eval_batch_size",
        str(args.per_device_eval_batch_size),
        "--gradient_accumulation_steps",
        str(args.gradient_accumulation_steps),
        "--warmup_ratio",
        str(args.warmup_ratio),
        "--weight_decay",
        str(args.weight_decay),
        "--logging_steps",
        str(args.logging_steps),
        "--eval_steps",
        str(args.eval_steps),
        "--save_steps",
        str(args.save_steps),
        "--save_total_limit",
        str(args.save_total_limit),
        "--max_steps",
        str(args.max_steps),
        "--lora_r",
        str(args.lora_r),
        "--lora_alpha",
        str(args.lora_alpha),
        "--lora_dropout",
        str(args.lora_dropout),
        "--require_4bit" if args.require_4bit else "",
        "--push_to_hub",
        "--hub_model_id",
        args.model_repo,
        "--hub_strategy",
        args.hub_strategy,
        "--report_to",
        args.report_to,
        "--story_seed_file",
        str(story_seed_file),
        "--max_bible_passages",
        str(args.max_bible_passages),
        "--bible_passage_window",
        str(args.bible_passage_window),
    ]
    if args.augment_with_generation_tasks:
        cmd.append("--augment_with_generation_tasks")
    if args.private_model_repo:
        cmd.append("--hub_private_repo")
    if resume_checkpoint:
        cmd.extend(["--resume_from_checkpoint", resume_checkpoint])
    return [part for part in cmd if part]


def maybe_disconnect_colab() -> None:
    try:
        from google.colab import runtime  # type: ignore

        runtime.unassign()
    except Exception:
        pass


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent
    work_dir = Path(args.work_dir).resolve()
    dataset_root = work_dir / "dataset"
    output_dir = work_dir / "model_output"
    manifest_path = work_dir / "run_manifest.json"

    token = resolve_token(args)
    maybe_configure_wandb(args, project_root)
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    login(token=token, add_to_git_credential=False)

    api = HfApi()
    ensure_model_repo(api, token, args.model_repo, args.private_model_repo)

    print(f"Downloading dataset from {args.dataset_repo}")
    train_file, valid_file = download_dataset_splits(args.dataset_repo, token, dataset_root)

    print(f"Restoring latest checkpoint from {args.model_repo}")
    resume_checkpoint = restore_latest_checkpoint(args.model_repo, token, output_dir)
    if resume_checkpoint:
        print(f"Resuming from checkpoint: {resume_checkpoint}")
    else:
        print("No remote checkpoint found. Starting a fresh run.")

    write_run_manifest(
        manifest_path,
        args=args,
        train_file=train_file,
        valid_file=valid_file,
        output_dir=output_dir,
        resume_checkpoint=resume_checkpoint,
    )

    cmd = build_train_command(
        args=args,
        project_root=project_root,
        train_file=train_file,
        valid_file=valid_file,
        output_dir=output_dir,
        resume_checkpoint=resume_checkpoint,
    )

    print("Launching training command:")
    print(" ".join(cmd))
    print(f"Run manifest: {manifest_path}")

    result = subprocess.run(cmd, cwd=project_root, check=False)

    if args.disconnect_colab:
        maybe_disconnect_colab()

    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
