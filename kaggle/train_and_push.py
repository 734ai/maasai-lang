#!/usr/bin/env python3
"""
Kaggle training entrypoint for the Maasai project.

This script is pushed as a Kaggle kernel. It clones the project repo,
installs the training dependencies, fetches secrets from Kaggle, and
reuses scripts/train_daily_from_hf.py so Hugging Face remains the system
of record for datasets, checkpoints, and the model repo.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


WORK_ROOT = Path("/kaggle/working")
PROJECT_ROOT = WORK_ROOT / "maasai-lang"
RUNTIME_CONFIG_JSON = """__RUNTIME_CONFIG_JSON__"""
CORE_TRAINING_REQUIREMENTS = [
    "transformers>=4.51.0",
    "datasets>=3.0.0",
    "accelerate>=1.0.0",
    "peft>=0.14.0",
    "trl>=0.16.0",
    "bitsandbytes>=0.45.0",
    "sentencepiece>=0.2.0",
    "sacrebleu>=2.4.0",
    "evaluate>=0.4.3",
    "pyyaml>=6.0.2",
    "huggingface_hub>=0.30.0",
]
DEFAULT_TORCH_VERSION = "2.5.1"
DEFAULT_TORCH_INDEX_URL = "https://download.pytorch.org/whl/cu118"


def run(cmd: list[str], *, cwd: Path | None = None) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd) if cwd is not None else None, check=True)


def load_config() -> dict:
    return json.loads(RUNTIME_CONFIG_JSON)


def get_secret(name: str) -> str | None:
    try:
        from kaggle_secrets import UserSecretsClient  # type: ignore

        client = UserSecretsClient()
        value = client.get_secret(name)
        return value or None
    except Exception:
        return None


def get_first_available_secret(*names: str) -> str | None:
    for name in names:
        value = get_secret(name)
        if value:
            return value
    return None


def install_dependencies(config: dict) -> None:
    torch_version = str(config.get("torch_version") or DEFAULT_TORCH_VERSION).strip()
    torch_index_url = str(config.get("torch_index_url") or DEFAULT_TORCH_INDEX_URL).strip()
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "--upgrade",
            "--force-reinstall",
            "--index-url",
            torch_index_url,
            f"torch=={torch_version}",
        ]
    )

    requirements = list(CORE_TRAINING_REQUIREMENTS)
    if config.get("report_to") == "wandb":
        requirements.append("wandb>=0.19.0")

    run([sys.executable, "-m", "pip", "install", "-q", "--upgrade", *requirements])


def log_runtime_gpu() -> None:
    try:
        import torch
    except Exception:
        return

    if not torch.cuda.is_available():
        print("No CUDA GPU detected by torch.")
        return

    capability = torch.cuda.get_device_capability(0)
    arch = f"sm_{capability[0]}{capability[1]}"
    supported_arches = ", ".join(torch.cuda.get_arch_list())
    print(f"Detected GPU: {torch.cuda.get_device_name(0)} ({arch})")
    print(f"PyTorch CUDA architectures: {supported_arches}")


def validate_runtime_gpu() -> None:
    try:
        import torch
    except Exception:
        return

    if not torch.cuda.is_available():
        return

    capability = torch.cuda.get_device_capability(0)
    current_arch = f"sm_{capability[0]}{capability[1]}"
    supported_arches = {arch for arch in torch.cuda.get_arch_list() if arch.startswith("sm_")}
    if supported_arches and current_arch not in supported_arches:
        gpu_name = torch.cuda.get_device_name(0)
        supported = ", ".join(sorted(supported_arches))
        raise RuntimeError(
            "The current PyTorch CUDA build does not support the assigned GPU "
            f"{gpu_name} ({current_arch}). Supported CUDA architectures: {supported}. "
            "On Kaggle this usually means a Tesla P100 / Pascal runtime was assigned; "
            "rerun until Kaggle assigns a T4/L4-class GPU, or use Colab/self-hosted training."
        )


def clone_repo(config: dict) -> None:
    if PROJECT_ROOT.exists():
        run(["rm", "-rf", str(PROJECT_ROOT)])
    run(["git", "clone", "--depth", "1", "--branch", config["project_branch"], config["project_git_url"], str(PROJECT_ROOT)])


def main() -> int:
    config = load_config()
    hf_token = get_first_available_secret("HF_TOKEN", "HUGGINGFACE_TOKEN", "HF_API_KEY")
    if not hf_token:
        hf_token = config.get("hf_token")
    if not hf_token:
        raise RuntimeError(
            "Missing Kaggle secret for Hugging Face access. Add one of: "
            "HF_TOKEN, HUGGINGFACE_TOKEN, or HF_API_KEY."
        )

    wandb_api_key = get_first_available_secret("WANDB_API_KEY", "WANDB_TOKEN")
    if not wandb_api_key:
        wandb_api_key = config.get("wandb_api_key")
    if wandb_api_key:
        os.environ["WANDB_API_KEY"] = wandb_api_key

    os.environ["HF_TOKEN"] = hf_token
    os.environ["HUGGINGFACE_TOKEN"] = hf_token
    os.environ.setdefault("HF_HOME", "/kaggle/working/.cache/huggingface")
    os.environ.setdefault("TRANSFORMERS_CACHE", "/kaggle/working/.cache/huggingface/transformers")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

    install_dependencies(config)
    log_runtime_gpu()
    validate_runtime_gpu()
    clone_repo(config)

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "train_daily_from_hf.py"),
        "--dataset-repo",
        config["dataset_repo"],
        "--model-repo",
        config["model_repo"],
        "--bucket-uri",
        config["bucket_uri"],
        "--bucket-prefix",
        config["bucket_prefix"],
        "--model-name",
        config["base_model"],
        "--work-dir",
        config["work_dir"],
        "--max-length",
        str(config["max_length"]),
        "--learning-rate",
        str(config["learning_rate"]),
        "--num-train-epochs",
        str(config["num_train_epochs"]),
        "--max-steps",
        str(config["max_steps"]),
        "--save-steps",
        str(config["save_steps"]),
        "--eval-steps",
        str(config["eval_steps"]),
        "--logging-steps",
        str(config["logging_steps"]),
        "--save-total-limit",
        str(config["save_total_limit"]),
        "--per-device-train-batch-size",
        str(config["per_device_train_batch_size"]),
        "--per-device-eval-batch-size",
        str(config["per_device_eval_batch_size"]),
        "--gradient-accumulation-steps",
        str(config["gradient_accumulation_steps"]),
        "--warmup-ratio",
        str(config["warmup_ratio"]),
        "--weight-decay",
        str(config["weight_decay"]),
        "--lora-r",
        str(config["lora_r"]),
        "--lora-alpha",
        str(config["lora_alpha"]),
        "--lora-dropout",
        str(config["lora_dropout"]),
        "--report-to",
        config["report_to"],
    ]
    if config.get("require_4bit"):
        cmd.append("--require-4bit")

    if config.get("augment_with_generation_tasks", True):
        cmd.append("--augment-with-generation-tasks")
    else:
        cmd.append("--no-augment-with-generation-tasks")

    if config.get("story_seed_file"):
        cmd.extend(["--story-seed-file", config["story_seed_file"]])
    if config.get("max_bible_passages") is not None:
        cmd.extend(["--max-bible-passages", str(config["max_bible_passages"])])
    if config.get("bible_passage_window") is not None:
        cmd.extend(["--bible-passage-window", str(config["bible_passage_window"])])
    if config.get("private_model_repo"):
        cmd.append("--private-model-repo")

    run(cmd, cwd=PROJECT_ROOT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
