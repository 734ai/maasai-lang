#!/usr/bin/env python3
"""
Summarize and plan lawful harvests for vetted Maasai web sources.

This script reads the machine-readable source registry and supports two
workflows:

1. `summary`: review vetted sources grouped by approval status.
2. `plan`: prepare a download plan for sources explicitly marked as
   plannable under the registry policy.

The script does not download anything. It exists to keep source expansion
traceable and intentionally constrained.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any


DEFAULT_REGISTRY = Path("data/registry/maasai_vetted_web_sources.json")
ALLOWED_STATUSES = {"approved", "gated_access", "reference_only"}
REQUIRED_SOURCE_KEYS = {
    "id",
    "title",
    "status",
    "rights_status",
    "license",
    "access",
    "provenance",
    "usage_policy",
    "harvest",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize or plan lawful harvests for vetted Maasai web sources."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    summary_parser = subparsers.add_parser("summary", help="Summarize registry contents")
    add_common_filters(summary_parser)
    summary_parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format for the summary.",
    )

    plan_parser = subparsers.add_parser(
        "plan",
        help="Prepare a download plan for approved sources only",
    )
    add_common_filters(plan_parser)
    plan_parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON path for the generated plan. Prints to stdout when omitted.",
    )

    return parser.parse_args()


def add_common_filters(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--registry",
        type=Path,
        default=DEFAULT_REGISTRY,
        help="Path to the vetted Maasai source registry.",
    )
    parser.add_argument(
        "--status",
        action="append",
        choices=sorted(ALLOWED_STATUSES),
        help="Optional status filter. Can be provided multiple times.",
    )
    parser.add_argument(
        "--source-id",
        action="append",
        help="Optional source id filter. Can be provided multiple times.",
    )


def load_registry(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "sources" not in payload or not isinstance(payload["sources"], list):
        raise ValueError(f"{path} is missing a top-level 'sources' list")

    for source in payload["sources"]:
        missing = REQUIRED_SOURCE_KEYS - set(source)
        if missing:
            raise ValueError(
                f"Registry entry '{source.get('id', 'unknown')}' is missing keys: {sorted(missing)}"
            )
        status = source["status"]
        if status not in ALLOWED_STATUSES:
            raise ValueError(
                f"Registry entry '{source['id']}' has unsupported status '{status}'"
            )

    return payload


def select_sources(
    sources: list[dict[str, Any]],
    *,
    statuses: list[str] | None,
    source_ids: list[str] | None,
) -> list[dict[str, Any]]:
    selected = sources

    if statuses:
        allowed = set(statuses)
        selected = [source for source in selected if source["status"] in allowed]

    if source_ids:
        allowed_ids = set(source_ids)
        selected = [source for source in selected if source["id"] in allowed_ids]

    return selected


def build_summary(
    registry: dict[str, Any],
    sources: list[dict[str, Any]],
) -> dict[str, Any]:
    by_status = Counter(source["status"] for source in sources)
    plannable = sum(
        1 for source in sources if bool(source["usage_policy"].get("download_plannable"))
    )

    return {
        "registry_id": registry.get("registry_id"),
        "registry_version": registry.get("registry_version"),
        "updated_on": registry.get("updated_on"),
        "source_count": len(sources),
        "status_counts": dict(sorted(by_status.items())),
        "plannable_count": plannable,
        "sources": [
            {
                "id": source["id"],
                "title": source["title"],
                "status": source["status"],
                "rights_status": source["rights_status"],
                "license": source["license"],
                "access_model": source["access"].get("model"),
                "download_plannable": bool(source["usage_policy"].get("download_plannable")),
            }
            for source in sources
        ],
    }


def format_text_summary(summary: dict[str, Any]) -> str:
    lines = [
        f"Registry: {summary['registry_id']} v{summary['registry_version']}",
        f"Updated: {summary['updated_on']}",
        f"Sources: {summary['source_count']}",
        f"Plannable: {summary['plannable_count']}",
        "Status counts:",
    ]

    for status, count in summary["status_counts"].items():
        lines.append(f"- {status}: {count}")

    if summary["sources"]:
        lines.append("Entries:")
    for source in summary["sources"]:
        lines.append(
            "- {id} | {status} | {rights_status} | {access_model} | plannable={download_plannable}".format(
                **source
            )
        )

    return "\n".join(lines)


def build_plan(
    registry: dict[str, Any],
    sources: list[dict[str, Any]],
) -> dict[str, Any]:
    approved_statuses = set(
        registry.get("policy", {}).get("download_plan_statuses", ["approved"])
    )
    planned_sources = [
        source
        for source in sources
        if source["status"] in approved_statuses
        and bool(source["usage_policy"].get("download_plannable"))
    ]

    return {
        "registry_id": registry.get("registry_id"),
        "registry_version": registry.get("registry_version"),
        "generated_on": date.today().isoformat(),
        "selection_policy": {
            "statuses": sorted(approved_statuses),
            "download_plannable_only": True,
        },
        "downloads": [
            {
                "id": source["id"],
                "title": source["title"],
                "license": source["license"],
                "rights_status": source["rights_status"],
                "source_type": source.get("source_type"),
                "download_url": source["access"].get("download_url"),
                "target_path": source["harvest"].get("target_path"),
                "expected_format": source["harvest"].get("expected_format"),
                "method": source["harvest"].get("method"),
                "notes": source["usage_policy"].get("notes"),
                "provenance_note": source["provenance"].get("rights_basis"),
                "fetch_command": [
                    "curl",
                    "-L",
                    "--fail",
                    "--output",
                    source["harvest"].get("target_path"),
                    source["access"].get("download_url"),
                ],
            }
            for source in planned_sources
        ],
    }


def write_output(payload: dict[str, Any], output_path: Path | None) -> None:
    rendered = json.dumps(payload, indent=2, ensure_ascii=False)
    if output_path is None:
        print(rendered)
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered + "\n", encoding="utf-8")
    print(f"Wrote plan to {output_path}")


def main() -> None:
    args = parse_args()
    registry = load_registry(args.registry)
    selected_sources = select_sources(
        registry["sources"],
        statuses=args.status,
        source_ids=args.source_id,
    )

    if args.command == "summary":
        summary = build_summary(registry, selected_sources)
        if args.format == "json":
            print(json.dumps(summary, indent=2, ensure_ascii=False))
        else:
            print(format_text_summary(summary))
        return

    if args.command == "plan":
        plan = build_plan(registry, selected_sources)
        write_output(plan, args.output)
        return

    raise RuntimeError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
