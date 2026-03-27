#!/usr/bin/env python3
"""
Publish project artifacts to Hugging Face Hub.

This script can publish:
- A Gradio Space bundle (from ./space + glossary data)
- A dataset repo bundle (from the public parallel-pair corpus in ./data/final_v3)
- A model repo bundle (from ./outputs/maasai-en-mt-qlora by default)

By default it runs in preview mode. Use --execute to perform uploads.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish Space, dataset, and model artifacts to Hugging Face")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face token (fallback: HF_TOKEN env or huggingface-api-key.json)")
    parser.add_argument("--username", type=str, default=None, help="Hugging Face username/org (fallback: key file username, then whoami)")

    parser.add_argument("--space-repo", type=str, default="maasai-language-showcase", help="Space repo name (without username)")
    parser.add_argument("--dataset-repo", type=str, default="maasai-translation-corpus", help="Dataset repo name (without username)")
    parser.add_argument("--model-repo", type=str, default="maasai-en-mt", help="Model repo name (without username)")
    parser.add_argument("--dataset-dir", type=str, default="data/final_v3", help="Local parallel-pair dataset directory to publish")
    parser.add_argument("--model-dir", type=str, default="outputs/maasai-en-mt-qlora", help="Local model folder to upload")

    parser.add_argument("--skip-space", action="store_true", help="Skip Space upload")
    parser.add_argument("--skip-dataset", action="store_true", help="Skip dataset upload")
    parser.add_argument("--skip-model", action="store_true", help="Skip model upload")

    parser.add_argument("--private-space", action="store_true", help="Create Space repo as private")
    parser.add_argument("--private-dataset", action="store_true", help="Create dataset repo as private")
    parser.add_argument("--private-model", action="store_true", help="Create model repo as private")
    parser.add_argument(
        "--create-model-repo",
        action="store_true",
        help="Create the model repo scaffold even when trained weights are not available yet",
    )

    parser.add_argument(
        "--export-dir",
        type=str,
        default=None,
        help="Optional local directory to write publishable bundles into before upload",
    )
    parser.add_argument("--execute", action="store_true", help="Actually create/upload repos (default is preview-only)")
    return parser.parse_args()


def load_keyfile(project_root: Path) -> dict:
    key_path = project_root / "huggingface-api-key.json"
    if not key_path.exists():
        return {}

    try:
        return json.loads(key_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def load_token(args: argparse.Namespace, key_info: dict) -> str:
    if args.token:
        return args.token

    for env_var in ("HF_TOKEN", "HUGGINGFACE_TOKEN"):
        value = os.getenv(env_var)
        if value:
            return value

    key_value = key_info.get("key")
    if key_value:
        return str(key_value)

    raise RuntimeError(
        "No Hugging Face token found. Set HF_TOKEN or pass --token, "
        "or provide huggingface-api-key.json with a 'key' field."
    )


def load_username(args: argparse.Namespace, key_info: dict, api: HfApi, token: str) -> str:
    if args.username:
        return args.username

    key_username = key_info.get("username")
    if key_username:
        return str(key_username)

    who = api.whoami(token=token)
    return who.get("name") or who.get("fullname") or who.get("id")


def copy_file(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Missing required file: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def copy_tree(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Missing required directory: {src}")
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def load_jsonl_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_dataset_summary(dataset_dir: Path) -> dict:
    split_counts: dict[str, int] = {}
    lang_pairs: collections.Counter[str] = collections.Counter()
    domains: collections.Counter[str] = collections.Counter()
    tiers: collections.Counter[str] = collections.Counter()
    source_names: collections.Counter[str] = collections.Counter()

    for split in ("train", "valid", "test"):
        split_path = dataset_dir / f"{split}.jsonl"
        rows = load_jsonl_rows(split_path)
        split_counts[split] = len(rows)
        for row in rows:
            lang_pairs[f"{row.get('source_lang', 'unknown')}->{row.get('target_lang', 'unknown')}"] += 1
            domains[str(row.get("domain", "unknown"))] += 1
            tiers[str(row.get("tier", "unknown"))] += 1
            source_names[str(row.get("source_name", "unknown"))] += 1

    total_rows = sum(split_counts.values())
    return {
        "dataset_format": "parallel_translation_pairs",
        "total_rows": total_rows,
        "splits": split_counts,
        "lang_pairs": dict(sorted(lang_pairs.items())),
        "top_domains": dict(domains.most_common(10)),
        "tiers": dict(tiers.most_common(10)),
        "top_source_names": dict(source_names.most_common(10)),
    }


def build_space_bundle(project_root: Path, temp_root: Path) -> Path:
    bundle = temp_root / "space_bundle"
    space_dir = project_root / "space"

    copy_file(space_dir / "app.py", bundle / "app.py")
    copy_file(space_dir / "style.css", bundle / "style.css")
    copy_file(space_dir / "requirements.txt", bundle / "requirements.txt")
    copy_file(space_dir / "README.md", bundle / "README.md")
    copy_tree(space_dir / "examples", bundle / "examples")

    copy_file(
        project_root / "data" / "glossary" / "maasai_glossary.json",
        bundle / "data" / "glossary" / "maasai_glossary.json",
    )
    return bundle


def build_dataset_bundle(project_root: Path, temp_root: Path, dataset_dir: Path) -> Path:
    bundle = temp_root / "dataset_bundle"
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Missing dataset directory: {dataset_dir}")

    copy_file(project_root / "docs" / "dataset_card.md", bundle / "README.md")
    for split in ("train", "valid", "test"):
        copy_file(dataset_dir / f"{split}.jsonl", bundle / "data" / f"{split}.jsonl")
    summary = build_dataset_summary(dataset_dir)
    (bundle / "data").mkdir(parents=True, exist_ok=True)
    (bundle / "data" / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (bundle / "dataset_info.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    copy_file(
        project_root / "data" / "glossary" / "maasai_glossary.json",
        bundle / "glossary" / "maasai_glossary.json",
    )
    return bundle


def looks_like_placeholder_artifact(path: Path) -> bool:
    """Detect mock or obviously invalid weight placeholders."""
    try:
        if path.stat().st_size < 1024:
            text = path.read_text(encoding="utf-8", errors="ignore").upper()
            return any(marker in text for marker in ("MOCK_", "PLACEHOLDER", "DUMMY"))
    except OSError:
        return False
    return False


def inspect_model_artifacts(model_dir: Path) -> dict:
    """Inspect model files and distinguish real artifacts from placeholders."""
    weight_markers = {
        "adapter_model.safetensors",
        "adapter_model.bin",
        "pytorch_model.bin",
        "model.safetensors",
    }
    info = {
        "state": "missing",
        "real_files": [],
        "placeholder_files": [],
        "model_dir": str(model_dir),
    }

    if not model_dir.exists() or not model_dir.is_dir():
        return info

    for path in model_dir.rglob("*"):
        if not path.is_file():
            continue
        is_weight_file = (
            path.name in weight_markers
            or path.suffix == ".gguf"
            or (path.name.startswith("model-") and path.suffix == ".safetensors")
        )
        if not is_weight_file:
            continue

        if looks_like_placeholder_artifact(path):
            info["placeholder_files"].append(path.name)
        else:
            info["real_files"].append(path.name)

    if info["real_files"]:
        info["state"] = "real"
    elif info["placeholder_files"]:
        info["state"] = "placeholder"

    return info


def build_model_download_meta(model_artifacts: dict) -> str:
    """Build lightweight Hub metadata for scaffold and trained model repos."""
    publication_status = "weights_available" if model_artifacts["state"] == "real" else "scaffold"
    real_files = model_artifacts.get("real_files", [])
    placeholder_files = model_artifacts.get("placeholder_files", [])

    lines = [
        "project: maasai-en-mt",
        "task: translation",
        "base_model: Qwen/Qwen2.5-3B-Instruct",
        "training_recipe: qlora",
        f"publication_status: {publication_status}",
        "download_count_anchor: true",
        "languages:",
        "  - en",
        "  - mas",
        "related_assets:",
        "  dataset: NorthernTribe-Research/maasai-translation-corpus",
        "  space: NorthernTribe-Research/maasai-language-showcase",
        "notes: >-",
        "  Lightweight repository metadata kept in the model repo so Hugging Face download",
        "  statistics can resolve against meta.yaml before or alongside full weight uploads.",
    ]

    if real_files:
        lines.extend(["real_artifacts:"] + [f"  - {name}" for name in real_files])
    if placeholder_files:
        lines.extend(["placeholder_artifacts:"] + [f"  - {name}" for name in placeholder_files])

    return "\n".join(lines) + "\n"


def build_model_bundle(project_root: Path, temp_root: Path, model_dir: Path) -> Path:
    bundle = temp_root / "model_bundle"
    model_artifacts = inspect_model_artifacts(model_dir)

    if model_artifacts["state"] == "real":
        copy_tree(model_dir, bundle)
        copy_file(project_root / "docs" / "model_card.md", bundle / "README.md")
        (bundle / "meta.yaml").write_text(
            build_model_download_meta(model_artifacts),
            encoding="utf-8",
        )
        return bundle

    bundle.mkdir(parents=True, exist_ok=True)
    readme = (project_root / "docs" / "model_card.md").read_text(encoding="utf-8")
    readme += (
        "\n\n## Publication Status\n\n"
        "This repository has been created ahead of the first full model upload. "
        "The current local `outputs/maasai-en-mt-qlora` directory only contains mock placeholder files used "
        "to test the deployment pipeline, so this publish intentionally excludes model weights.\n\n"
        "Run GPU-backed training, replace the placeholder artifacts with a real adapter or merged checkpoint, "
        "and publish again to upload the actual model."
    )
    (bundle / "README.md").write_text(readme, encoding="utf-8")
    (bundle / "meta.yaml").write_text(
        build_model_download_meta(model_artifacts),
        encoding="utf-8",
    )
    (bundle / "MODEL_STATUS.json").write_text(
        json.dumps(model_artifacts, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return bundle


def export_bundle(bundle: Path, export_root: Path, bundle_name: str) -> Path:
    export_path = export_root / bundle_name
    copy_tree(bundle, export_path)
    return export_path


def print_plan(
    username: str,
    args: argparse.Namespace,
    do_space: bool,
    do_dataset: bool,
    do_model: bool,
    dataset_dir: Path,
    model_dir: Path,
    model_artifacts: dict,
) -> None:
    print("\n=== Hugging Face Publish Plan ===")
    print(f"Account: {username}")

    if do_space:
        print(f"- Space:   {username}/{args.space_repo}  (private={args.private_space})")
        print("  source:  ./space + ./data/glossary/maasai_glossary.json")

    if do_dataset:
        print(f"- Dataset: {username}/{args.dataset_repo}  (private={args.private_dataset})")
        print(f"  source:  {dataset_dir} + ./docs/dataset_card.md + glossary")

    if do_model:
        print(f"- Model:   {username}/{args.model_repo}  (private={args.private_model})")
        if model_artifacts["state"] == "real":
            print(f"  source:  {model_dir} + ./docs/model_card.md")
        else:
            print("  source:  README scaffold only (local model folder has placeholder or missing weights)")
    elif not args.skip_model:
        if model_artifacts["state"] == "placeholder":
            print(f"- Model:   skipped (placeholder/mock artifacts found in: {model_dir})")
        else:
            print(f"- Model:   skipped (no model artifacts found in: {model_dir})")


def create_and_upload(
    api: HfApi,
    *,
    token: str,
    repo_id: str,
    repo_type: str,
    folder_path: Path,
    private: bool,
    commit_message: str,
) -> None:
    if repo_type == "space":
        api.create_repo(
            repo_id=repo_id,
            repo_type="space",
            space_sdk="gradio",
            private=private,
            exist_ok=True,
            token=token,
        )
    else:
        api.create_repo(
            repo_id=repo_id,
            repo_type=repo_type,
            private=private,
            exist_ok=True,
            token=token,
        )

    api.upload_folder(
        repo_id=repo_id,
        repo_type=repo_type,
        folder_path=str(folder_path),
        commit_message=commit_message,
        token=token,
    )


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent
    dataset_dir = (project_root / args.dataset_dir).resolve()
    model_dir = (project_root / args.model_dir).resolve()
    export_root = None
    if args.export_dir:
        export_root = Path(args.export_dir)
        if not export_root.is_absolute():
            export_root = (project_root / export_root).resolve()
        export_root.mkdir(parents=True, exist_ok=True)

    key_info = load_keyfile(project_root)
    token: str | None = None
    username = args.username or key_info.get("username")
    api = HfApi()

    if args.execute:
        token = load_token(args, key_info)
        username = load_username(args, key_info, api, token)
    elif username is None:
        try:
            token = load_token(args, key_info)
            username = load_username(args, key_info, api, token)
        except RuntimeError:
            username = "<set-username>"

    model_artifacts = inspect_model_artifacts(model_dir)

    do_space = not args.skip_space
    do_dataset = not args.skip_dataset
    do_model = (
        (not args.skip_model)
        and (
            model_artifacts["state"] == "real"
            or args.create_model_repo
        )
    )

    print_plan(username, args, do_space, do_dataset, do_model, dataset_dir, model_dir, model_artifacts)

    if not args.execute and export_root is None:
        print("\nPreview mode only. Re-run with --execute to publish.")
        return

    with tempfile.TemporaryDirectory(prefix="hf_publish_") as temp_dir:
        temp_root = Path(temp_dir)

        if do_space:
            space_bundle = build_space_bundle(project_root, temp_root)
            if export_root is not None:
                export_path = export_bundle(space_bundle, export_root, "space_bundle")
                print(f"Exported Space bundle: {export_path}")
            space_repo_id = f"{username}/{args.space_repo}"
            if args.execute:
                print(f"\nPublishing Space: {space_repo_id}")
                create_and_upload(
                    api,
                    token=token or "",
                    repo_id=space_repo_id,
                    repo_type="space",
                    folder_path=space_bundle,
                    private=args.private_space,
                    commit_message="Publish Maasai Space app",
                )
                print(f"Published Space: {space_repo_id}")

        if do_dataset:
            dataset_bundle = build_dataset_bundle(project_root, temp_root, dataset_dir)
            if export_root is not None:
                export_path = export_bundle(dataset_bundle, export_root, "dataset_bundle")
                print(f"Exported Dataset bundle: {export_path}")
            dataset_repo_id = f"{username}/{args.dataset_repo}"
            if args.execute:
                print(f"\nPublishing Dataset: {dataset_repo_id}")
                create_and_upload(
                    api,
                    token=token or "",
                    repo_id=dataset_repo_id,
                    repo_type="dataset",
                    folder_path=dataset_bundle,
                    private=args.private_dataset,
                    commit_message="Publish Maasai dataset bundle",
                )
                print(f"Published Dataset: {dataset_repo_id}")

        if do_model:
            model_bundle = build_model_bundle(project_root, temp_root, model_dir)
            if export_root is not None:
                export_path = export_bundle(model_bundle, export_root, "model_bundle")
                print(f"Exported Model bundle: {export_path}")
            model_repo_id = f"{username}/{args.model_repo}"
            if args.execute:
                print(f"\nPublishing Model: {model_repo_id}")
                create_and_upload(
                    api,
                    token=token or "",
                    repo_id=model_repo_id,
                    repo_type="model",
                    folder_path=model_bundle,
                    private=args.private_model,
                    commit_message=(
                        "Publish Maasai model artifacts"
                        if model_artifacts["state"] == "real"
                        else "Create Maasai model repo scaffold"
                    ),
                )
                print(f"Published Model: {model_repo_id}")

    if args.execute:
        print("\nAll requested publish tasks completed.")
    else:
        print("\nPreview/export completed. Re-run with --execute to publish.")


if __name__ == "__main__":
    main()
