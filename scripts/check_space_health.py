#!/usr/bin/env python3
"""Check the live health of the Maasai Hugging Face Space."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_SPACE_OWNER = "NorthernTribe-Research"
DEFAULT_SPACE_NAME = "maasai-language-showcase"
DEFAULT_TIMEOUT = 20.0
HEALTHY_STAGES = {"RUNNING", "READY"}


def load_default_repo_id(project_root: Path) -> str:
    creds_path = project_root / "huggingface-api-key.json"
    if creds_path.exists():
        try:
            creds = json.loads(creds_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            creds = {}
        username = str(creds.get("username", "")).strip()
        if username:
            return f"{username}/{DEFAULT_SPACE_NAME}"
    return f"{DEFAULT_SPACE_OWNER}/{DEFAULT_SPACE_NAME}"


def parse_repo_id(repo_id: str) -> tuple[str, str]:
    owner, sep, name = repo_id.strip().partition("/")
    if not sep or not owner or not name:
        raise ValueError("repo id must be in the form owner/name")
    return owner, name


def fetch_json(url: str, timeout: float) -> dict[str, Any]:
    result = subprocess.run(
        [
            "curl",
            "-sS",
            "-L",
            "--connect-timeout",
            str(timeout),
            "--max-time",
            str(timeout),
            "-H",
            "Accept: application/json",
            "-H",
            "User-Agent: maasai-space-healthcheck/1.0",
            url,
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout + 5,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"curl failed for {url}")
    return json.loads(result.stdout)


def probe_url(url: str, timeout: float, accept: str, head_only: bool = False) -> dict[str, Any]:
    cmd = [
        "curl",
        "-sS",
        "-L",
        "--connect-timeout",
        str(timeout),
        "--max-time",
        str(timeout),
        "-H",
        f"Accept: {accept}",
        "-H",
        "User-Agent: maasai-space-healthcheck/1.0",
        "-o",
        "/dev/null",
        "-w",
        "__META__%{http_code}|%{url_effective}|%{content_type}",
    ]
    if head_only:
        cmd.append("-I")
    cmd.append(url)

    try:
        result = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout + 5,
        )
    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "status": None,
            "url": url,
            "content_type": None,
            "error": "curl probe timed out",
        }

    marker = "__META__"
    if marker not in result.stdout:
        return {
            "ok": False,
            "status": None,
            "url": url,
            "content_type": None,
            "error": result.stderr.strip() or result.stdout.strip() or "curl probe failed",
        }

    meta = result.stdout.split(marker, 1)[1].strip()
    status_str, _, remainder = meta.partition("|")
    effective_url, _, content_type = remainder.partition("|")
    try:
        status = int(status_str)
    except ValueError:
        status = None
    if status == 0:
        status = None
    return {
        "ok": result.returncode == 0 and status is not None and 200 <= status < 400,
        "status": status,
        "url": effective_url or url,
        "content_type": content_type or None,
        "error": None if result.returncode == 0 else result.stderr.strip() or None,
    }


def classify_status(
    runtime_stage: str,
    domain_stage: str,
    page_probe: dict[str, Any],
    api_probe: dict[str, Any],
) -> tuple[str, str]:
    page_ok = page_probe.get("ok", False)
    api_ok = api_probe.get("ok", False)

    if not page_ok and not api_ok:
        return "down", "Space page and Gradio API probes both failed."
    if not page_ok or not api_ok:
        return "degraded", "One of the live probes failed even though the Space metadata resolved."
    if runtime_stage in HEALTHY_STAGES and domain_stage in {"READY", "RUNNING"}:
        return "healthy", "Live page and API probes succeeded, and Hugging Face reports a ready runtime."
    return "degraded", "Live probes succeeded, but the Hugging Face control plane still reports a transitional runtime."


def build_report(repo_id: str, timeout: float, project_root: Path) -> dict[str, Any]:
    owner, name = parse_repo_id(repo_id)
    api_url = f"https://huggingface.co/api/spaces/{owner}/{name}"
    metadata = fetch_json(api_url, timeout=timeout)

    runtime = metadata.get("runtime") or {}
    domains = runtime.get("domains") or []
    domain_stage = str(domains[0].get("stage", "unknown")).upper() if domains else "UNKNOWN"
    runtime_stage = str(runtime.get("stage", "unknown")).upper()
    host = str(metadata.get("host") or f"https://{metadata.get('subdomain', f'{owner}-{name}')}.hf.space").rstrip("/")

    page_probe = probe_url(host, timeout=timeout, accept="text/html,*/*;q=0.8", head_only=True)
    api_probe = probe_url(
        f"{host}/gradio_api/openapi.json",
        timeout=timeout,
        accept="application/json",
        head_only=False,
    )
    overall_status, summary = classify_status(runtime_stage, domain_stage, page_probe, api_probe)

    return {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "repo_id": repo_id,
        "project_root": str(project_root),
        "api_url": api_url,
        "host": host,
        "overall_status": overall_status,
        "summary": summary,
        "space_card": {
            "sdk": metadata.get("sdk"),
            "sha": metadata.get("sha"),
            "last_modified": metadata.get("lastModified"),
            "models": metadata.get("models") or [],
            "datasets": metadata.get("datasets") or [],
        },
        "runtime": {
            "stage": runtime_stage,
            "hardware_current": (runtime.get("hardware") or {}).get("current"),
            "hardware_requested": (runtime.get("hardware") or {}).get("requested"),
            "replicas_current": (runtime.get("replicas") or {}).get("current"),
            "replicas_requested": (runtime.get("replicas") or {}).get("requested"),
            "domain_stage": domain_stage,
            "runtime_sha": runtime.get("sha"),
        },
        "probes": {
            "page": {
                "ok": page_probe.get("ok"),
                "status": page_probe.get("status"),
                "content_type": page_probe.get("content_type"),
                "url": page_probe.get("url"),
            },
            "gradio_api": {
                "ok": api_probe.get("ok"),
                "status": api_probe.get("status"),
                "content_type": api_probe.get("content_type"),
                "url": api_probe.get("url"),
            },
        },
    }


def print_text_report(report: dict[str, Any]) -> None:
    print(f"Checked at:        {report['checked_at']}")
    print(f"Space:             {report['repo_id']}")
    print(f"Host:              {report['host']}")
    print(f"Overall status:    {report['overall_status'].upper()}")
    print(f"Summary:           {report['summary']}")
    print()
    print(f"Runtime stage:     {report['runtime']['stage']}")
    print(f"Domain stage:      {report['runtime']['domain_stage']}")
    print(
        "Replicas:          "
        f"{report['runtime']['replicas_current']}/{report['runtime']['replicas_requested']}"
    )
    print(
        "Hardware:          "
        f"{report['runtime']['hardware_current']} -> {report['runtime']['hardware_requested']}"
    )
    print()
    print(
        "Page probe:        "
        f"{report['probes']['page']['status']} {report['probes']['page']['content_type']}"
    )
    if report["probes"]["page"].get("error"):
        print(f"Page error:        {report['probes']['page']['error']}")
    print(
        "Gradio API probe:  "
        f"{report['probes']['gradio_api']['status']} {report['probes']['gradio_api']['content_type']}"
    )
    if report["probes"]["gradio_api"].get("error"):
        print(f"Gradio API error:  {report['probes']['gradio_api']['error']}")


def main() -> int:
    project_root = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-id",
        default=load_default_repo_id(project_root),
        help="Hugging Face Space repo id in owner/name form.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help="HTTP timeout in seconds.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON.",
    )
    args = parser.parse_args()

    try:
        report = build_report(args.repo_id, timeout=args.timeout, project_root=project_root)
    except (ValueError, RuntimeError, json.JSONDecodeError, subprocess.TimeoutExpired) as exc:
        error_report = {
            "checked_at": datetime.now(timezone.utc).isoformat(),
            "repo_id": args.repo_id,
            "overall_status": "down",
            "summary": str(exc),
        }
        if args.json:
            print(json.dumps(error_report, indent=2))
        else:
            print(f"Overall status:    DOWN")
            print(f"Summary:           {exc}")
        return 1

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print_text_report(report)

    if report["overall_status"] == "healthy":
        return 0
    if report["overall_status"] == "degraded":
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
