#!/usr/bin/env python3
"""
Simple direct push to HuggingFace using huggingface_hub library.
No CLI tools required - uses API key directly.
"""

import json
from pathlib import Path
from huggingface_hub import HfApi, CommitOperationAdd, CommitOperationDelete
import sys

def push_dataset_direct():
    """Push dataset using huggingface_hub API."""
    project_root = Path(__file__).parent.parent
    
    # Load credentials
    with open(project_root / "huggingface-api-key.json") as f:
        creds = json.load(f)
    
    token = creds.get("key")
    username = creds.get("username")
    
    if not token or not username:
        print("❌ Missing HF credentials")
        sys.exit(1)
    
    print(f"🔐 User: {username}")
    print(f"🔑 Token: {token[:20]}...")
    
    # Initialize API
    api = HfApi(token=token)
    repo_id = f"{username}/maasai-translation-corpus"
    
    print(f"\n📊 Pushing to: datasets/{repo_id}")
    
    # Prepare files to upload
    data_dir = project_root / "data" / "final_v3"
    docs_dir = project_root / "docs"
    
    files_to_upload = []
    
    # Add training data
    for split in ["train", "valid", "test"]:
        src = data_dir / f"{split}.jsonl"
        if src.exists():
            files_to_upload.append(
                CommitOperationAdd(
                    path_in_repo=f"{split}.jsonl",
                    path_or_fileobj=str(src)
                )
            )
            print(f"  ✓ {split}.jsonl")
    
    # Add dataset card
    readme_src = docs_dir / "DATASET_README_V2.md"
    if readme_src.exists():
        files_to_upload.append(
            CommitOperationAdd(
                path_in_repo="README.md",
                path_or_fileobj=str(readme_src)
            )
        )
        print(f"  ✓ README.md")
    
    # Add dataset info
    datasetinfo = {
        "dataset_name": "maasai-translation-corpus",
        "dataset_summary": "Comprehensive English-Maasai translation dataset (9,910 pairs)",
        "language": ["en", "mas"],
        "license": "CC-BY-4.0",
        "task_ids": ["translation"],
        "tags": ["translation", "maasai", "low-resource", "cultural-preservation"],
        "size_categories": ["1K<n<10K"],
        "domains": ["religious", "cultural", "proverbs", "lexicon"],
        "version": "2.0"
    }
    
    datasetinfo_json = json.dumps(datasetinfo, indent=2)
    files_to_upload.append(
        CommitOperationAdd(
            path_in_repo="datasetinfo.json",
            path_or_fileobj=datasetinfo_json.encode()
        )
    )
    print(f"  ✓ datasetinfo.json")
    
    # Commit
    try:
        print("\n[Uploading...]")
        commit_info = api.create_commit(
            repo_id=repo_id,
            operations=files_to_upload,
            commit_message="Update v2.0: translation corpus refresh (9,910 pairs, Hollis + Hinde + ASJP supplement)",
            repo_type="dataset"
        )
        print(f"✅ Dataset pushed successfully!")
        print(f"   Commit: {commit_info.commit_url}")
        print(f"   URL: https://huggingface.co/datasets/{repo_id}")
        return True
    except Exception as e:
        print(f"❌ Push failed: {e}")
        return False

if __name__ == "__main__":
    push_dataset_direct()
