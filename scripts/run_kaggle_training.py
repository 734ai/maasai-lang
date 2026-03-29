#!/usr/bin/env python3
"""
Push the Kaggle training kernel and automatically retry when Kaggle assigns
an unsupported GPU architecture such as Tesla P100 / sm_60.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


UNSUPPORTED_GPU_MARKERS = (
    "does not support the assigned GPU",
    "Tesla P100-PCIE-16GB",
    "sm_60",
)

TRAINING_PROGRESS_MARKERS = (
    "Starting training",
    "Tokenizing dataset",
    "trainable params",
)


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Push and monitor the Maasai Kaggle training kernel")
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument("--poll-seconds", type=int, default=25)
    parser.add_argument("--startup-timeout-seconds", type=int, default=420)
    parser.add_argument("--log-dir", default="/tmp/maasai_kaggle_monitor")
    parser.add_argument("--kernel-slug", default="maasai-daily-hf-training")
    parser.add_argument("--no-retry-unsupported-gpu", action="store_true")
    return parser.parse_known_args()


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def resolve_python(root: Path) -> str:
    venv_python = root / ".venv" / "bin" / "python"
    if venv_python.exists():
        return str(venv_python)
    return sys.executable


def resolve_kaggle_cli(root: Path) -> str:
    venv_cli = root / ".venv" / "bin" / "kaggle"
    if venv_cli.exists():
        return str(venv_cli)
    return "kaggle"


def kaggle_env(root: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["KAGGLE_CONFIG_DIR"] = str(root)
    return env


def load_kaggle_username(root: Path) -> str:
    username = str(os.getenv("KAGGLE_USERNAME", "")).strip()
    if username:
        return username

    with (root / "kaggle.json").open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    username = str(payload.get("username", "")).strip()
    if not username:
        raise RuntimeError("Kaggle username is missing. Set KAGGLE_USERNAME or provide kaggle.json.")
    return username


def run_capture(cmd: list[str], *, cwd: Path, env: dict[str, str], check: bool = True) -> subprocess.CompletedProcess[str]:
    print("+", " ".join(cmd))
    result = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        text=True,
        capture_output=True,
    )
    if check and result.returncode != 0:
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        raise subprocess.CalledProcessError(result.returncode, cmd, result.stdout, result.stderr)
    return result


def fetch_status(kaggle_cli: str, root: Path, env: dict[str, str], kernel_ref: str) -> str:
    result = run_capture([kaggle_cli, "kernels", "status", kernel_ref], cwd=root, env=env)
    text = (result.stdout or result.stderr).strip()
    print(text)
    if '"KernelWorkerStatus.' in text:
        return text.rsplit("KernelWorkerStatus.", 1)[1].rstrip('"')
    return text


def fetch_log_text(kaggle_cli: str, root: Path, env: dict[str, str], kernel_ref: str, log_dir: Path, kernel_slug: str) -> str:
    log_dir.mkdir(parents=True, exist_ok=True)
    run_capture(
        [
            kaggle_cli,
            "kernels",
            "output",
            kernel_ref,
            "-p",
            str(log_dir),
            "-o",
            "--file-pattern",
            rf"{kernel_slug}\.log",
        ],
        cwd=root,
        env=env,
        check=False,
    )
    log_path = log_dir / f"{kernel_slug}.log"
    if not log_path.exists():
        return ""
    return log_path.read_text(encoding="utf-8", errors="replace")


def has_any_marker(text: str, markers: tuple[str, ...]) -> bool:
    return any(marker in text for marker in markers)


def wait_for_kernel(
    *,
    kaggle_cli: str,
    root: Path,
    env: dict[str, str],
    kernel_ref: str,
    kernel_slug: str,
    log_dir: Path,
    poll_seconds: int,
    startup_timeout_seconds: int,
) -> tuple[str, str]:
    deadline = time.time() + startup_timeout_seconds
    last_log = ""

    while time.time() < deadline:
        status = fetch_status(kaggle_cli, root, env, kernel_ref)
        log_text = fetch_log_text(kaggle_cli, root, env, kernel_ref, log_dir, kernel_slug)
        if log_text:
            last_log = log_text

        if has_any_marker(last_log, TRAINING_PROGRESS_MARKERS):
            return "training_started", last_log

        if status == "ERROR":
            if has_any_marker(last_log, UNSUPPORTED_GPU_MARKERS):
                return "unsupported_gpu", last_log
            return "error", last_log

        if status == "COMPLETE":
            return "complete", last_log

        time.sleep(poll_seconds)

    return "timeout", last_log


def ensure_push_flags(extra_args: list[str]) -> list[str]:
    final_args = list(extra_args)
    if "--execute" not in final_args:
        final_args.append("--execute")
    if "--status" not in final_args:
        final_args.append("--status")
    return final_args


def main() -> int:
    args, extra_args = parse_args()
    root = project_root()
    env = kaggle_env(root)
    kaggle_cli = resolve_kaggle_cli(root)
    python_bin = resolve_python(root)
    username = load_kaggle_username(root)
    kernel_ref = f"{username}/{args.kernel_slug}"
    push_args = ensure_push_flags(extra_args)
    if "--kernel-slug" not in push_args:
        push_args.extend(["--kernel-slug", args.kernel_slug])
    monitor_log_dir = Path(args.log_dir)

    for attempt in range(1, args.max_attempts + 1):
        print(f"=== Kaggle attempt {attempt}/{args.max_attempts} ===")
        run_capture(
            [python_bin, str(root / "scripts" / "push_kaggle_kernel.py"), *push_args],
            cwd=root,
            env=env,
        )

        outcome, log_text = wait_for_kernel(
            kaggle_cli=kaggle_cli,
            root=root,
            env=env,
            kernel_ref=kernel_ref,
            kernel_slug=args.kernel_slug,
            log_dir=monitor_log_dir / f"attempt_{attempt}",
            poll_seconds=args.poll_seconds,
            startup_timeout_seconds=args.startup_timeout_seconds,
        )

        if outcome in {"training_started", "complete"}:
            print(f"Kaggle kernel reached state: {outcome}")
            return 0

        if outcome == "unsupported_gpu" and not args.no_retry_unsupported_gpu and attempt < args.max_attempts:
            print("Unsupported Kaggle GPU detected. Retrying with a fresh kernel run.")
            continue

        if log_text:
            print("\n=== Final Kaggle Log Tail ===")
            tail_lines = log_text.splitlines()[-60:]
            print("\n".join(tail_lines))

        if outcome == "unsupported_gpu":
            print("Kaggle kept assigning unsupported GPUs across the allowed attempts.", file=sys.stderr)
        elif outcome == "timeout":
            print("Kaggle kernel did not reach training or terminal failure within the startup timeout.", file=sys.stderr)
        else:
            print(f"Kaggle kernel ended with outcome: {outcome}", file=sys.stderr)
        return 1

    print("Exhausted all Kaggle attempts without reaching a usable training run.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
