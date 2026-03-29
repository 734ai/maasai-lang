#!/usr/bin/env python3
"""
Complete Maasai Project Deployment Orchestrator

This script coordinates:
1. HuggingFace credential setup
2. Dataset push to HF Hub
3. Space update to HF Spaces
4. Model push (after training)
5. Verification and status checks

Usage:
    python scripts/deploy_to_hf.py --all
    python scripts/deploy_to_hf.py --dataset --space
    python scripts/deploy_to_hf.py --status
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime

def run_cmd(cmd: str, cwd: str = None) -> tuple[int, str]:
    """Run shell command and return exit code + output."""
    try:
        result = subprocess.run(
            cmd, shell=True, cwd=cwd, capture_output=True, text=True, timeout=600
        )
        return result.returncode, result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return -1, "Command timed out"
    except Exception as e:
        return -1, str(e)

def print_section(title: str):
    """Print formatted section header."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")

def check_credentials(project_root: Path) -> bool:
    """Verify HF credentials exist."""
    creds_file = project_root / "huggingface-api-key.json"
    if not creds_file.exists():
        print("❌ HuggingFace credentials not found at", creds_file)
        return False
    
    try:
        with open(creds_file) as f:
            creds = json.load(f)
        username = creds.get("username")
        token = creds.get("key")
        
        if not username or not token:
            print("❌ Credentials incomplete (missing username or token)")
            return False
        
        print(f"✓ Credentials found for: {username}")
        return True
    except Exception as e:
        print(f"❌ Error reading credentials: {e}")
        return False

def check_data_ready(project_root: Path) -> bool:
    """Verify training data is prepared."""
    data_dir = project_root / "data" / "final_v3"
    required_files = ["train.jsonl", "valid.jsonl", "test.jsonl"]
    
    missing = []
    for f in required_files:
        if not (data_dir / f).exists():
            missing.append(f)
    
    if missing:
        print(f"❌ Missing data files: {missing}")
        return False
    
    # Check file sizes
    for f in required_files:
        fpath = data_dir / f
        size_mb = fpath.stat().st_size / 1024**2
        print(f"  ✓ {f}: {size_mb:.1f} MB")
    
    return True

def describe_model_state(model_dir: Path) -> tuple[str, bool]:
    """Describe whether local model outputs look real or placeholder-only."""
    if not model_dir.exists():
        return "❌ Not yet trained", False

    readme = model_dir / "README.md"
    if readme.exists():
        try:
            readme_text = readme.read_text(encoding="utf-8").lower()
        except OSError:
            readme_text = ""
        if "placeholder artifacts" in readme_text:
            return "⚠️ Placeholder artifacts only", False

    adapter_file = model_dir / "adapter_model.bin"
    if adapter_file.exists() and adapter_file.stat().st_size < 1024:
        return "⚠️ Placeholder artifacts only", False

    return "✓ Trained checkpoint", True

def push_dataset(project_root: Path) -> bool:
    """Execute dataset push."""
    print_section("PUSHING DATASET TO HF")
    
    script = project_root / "scripts" / "push_dataset_to_hf.py"
    if not script.exists():
        print(f"❌ Script not found: {script}")
        return False
    
    code, output = run_cmd(
        f"{project_root}/.venv/bin/python {script}",
        cwd=project_root
    )
    
    print(output)
    return code == 0

def push_space(project_root: Path) -> bool:
    """Execute space update."""
    print_section("UPDATING SPACE ON HF")
    
    script = project_root / "scripts" / "push_space_to_hf.py"
    if not script.exists():
        print(f"❌ Script not found: {script}")
        return False
    
    code, output = run_cmd(
        f"{project_root}/.venv/bin/python {script}",
        cwd=project_root
    )
    
    print(output)
    return code == 0

def push_model(project_root: Path) -> bool:
    """Execute model push (if trained)."""
    print_section("PUSHING MODEL TO HF")
    
    model_dir = project_root / "outputs" / "maasai-en-mt-qlora"
    
    if not model_dir.exists():
        print(f"⚠️ Model checkpoint not found at {model_dir}")
        print("   Run training first: bash training/run_train.sh")
        return False

    model_state, has_real_checkpoint = describe_model_state(model_dir)
    print(f"Local model state: {model_state}")
    if not has_real_checkpoint:
        print("   Continuing will publish only the model repo scaffold/metadata via the unified publisher.")
    
    script = project_root / "scripts" / "push_model_to_hf.py"
    if not script.exists():
        print(f"❌ Script not found: {script}")
        return False
    
    code, output = run_cmd(
        f"{project_root}/.venv/bin/python {script}",
        cwd=project_root
    )
    
    print(output)
    return code == 0

def show_status(project_root: Path) -> None:
    """Display deployment status."""
    print_section("DEPLOYMENT STATUS")
    
    with open(project_root / "huggingface-api-key.json") as f:
        creds = json.load(f)
    
    username = creds["username"]
    
    print(f"\\n🔐 HuggingFace User: {username}")
    print(f"\\n📊 Dataset:")
    print(f"   https://huggingface.co/datasets/{username}/maasai-translation-corpus")
    print(f"   Status: {'✓' if (project_root / 'data' / 'final_v3' / 'train.jsonl').exists() else '❌'} Ready")
    
    print(f"\\n🚀 Space:")
    print(f"   https://huggingface.co/spaces/{username}/maasai-language-showcase")
    print(f"   Status: {'✓' if (project_root / 'space' / 'app.py').exists() else '❌'} Ready")
    
    print(f"\\n🤖 Model:")
    model_path = project_root / "outputs" / "maasai-en-mt-qlora"
    print(f"   https://huggingface.co/{username}/maasai-en-mt")
    model_state, _ = describe_model_state(model_path)
    print(f"   Status: {model_state}")
    
    if model_path.exists():
        model_size_mb = sum(f.stat().st_size for f in model_path.rglob("*")) / 1024**2
        print(f"   Checkpoint size: {model_size_mb:.1f} MB")
    
    print(f"\\n📊 Dataset Stats:")
    train_file = project_root / "data" / "final_v3" / "train.jsonl"
    if train_file.exists():
        with open(train_file) as f:
            train_count = sum(1 for _ in f)
        print(f"   Training pairs: {train_count}")
        print(f"   Validation pairs: {sum(1 for _ in open(project_root / 'data' / 'final_v3' / 'valid.jsonl'))}")
        print(f"   Test pairs: {sum(1 for _ in open(project_root / 'data' / 'final_v3' / 'test.jsonl'))}")
    
    print(f"\\nℹ️ Deployment Timestamp: {datetime.now().isoformat()}")

def main():
    parser = argparse.ArgumentParser(
        description="Maasai Project Deployment Orchestrator"
    )
    parser.add_argument("--all", action="store_true", help="Push everything (dataset + space + model)")
    parser.add_argument("--dataset", action="store_true", help="Push dataset only")
    parser.add_argument("--space", action="store_true", help="Update space only")
    parser.add_argument("--model", action="store_true", help="Push model only")
    parser.add_argument("--status", action="store_true", help="Show deployment status")
    parser.add_argument("--verify", action="store_true", help="Verify all prerequisites")
    
    args = parser.parse_args()
    
    project_root = Path(__file__).parent.parent
    
    # Show welcome
    print(f"""
╔════════════════════════════════════════════════════════════╗
║  🦁 Maasai Language Project - HuggingFace Deployment      ║
║      Version: 2.0 | Dataset: 9,406 pairs                  ║
╚════════════════════════════════════════════════════════════╝
""")
    
    # Default: status
    if not any([args.all, args.dataset, args.space, args.model, args.status, args.verify]):
        args.status = True
    
    # Verify credentials first
    if not check_credentials(project_root):
        sys.exit(1)
    
    if args.verify:
        print_section("VERIFICATION")
        print("Checking prerequisites...")
        
        print("\\n1. HF Credentials: ✓")
        print("2. Data files:", end=" ")
        if check_data_ready(project_root):
            print("✓")
        else:
            sys.exit(1)
        
        print("3. Scripts: ", end="")
        scripts = [
            "scripts/push_dataset_to_hf.py",
            "scripts/push_space_to_hf.py",
            "scripts/push_model_to_hf.py",
        ]
        all_exist = all((project_root / s).exists() for s in scripts)
        print("✓" if all_exist else "❌")
        
        if all_exist:
            print("\\n✅ All prerequisites met! Ready for deployment.")
        else:
            sys.exit(1)
    
    elif args.status:
        show_status(project_root)
    
    else:
        # Verify data before any push
        if not check_data_ready(project_root):
            sys.exit(1)
        
        print()
        results = {}
        
        if args.all or args.dataset:
            results["dataset"] = push_dataset(project_root)
        
        if args.all or args.space:
            results["space"] = push_space(project_root)
        
        if args.all or args.model:
            results["model"] = push_model(project_root)
        
        # Summary
        print_section("DEPLOYMENT SUMMARY")
        for component, success in results.items():
            status = "✅ SUCCESS" if success else "❌ FAILED"
            print(f"{component.upper():12} {status}")
        
        if all(results.values()):
            print("\\n✅ All components deployed successfully!")
            show_status(project_root)
        else:
            print("\\n⚠️ Some components failed. Please review errors above.")
            sys.exit(1)

if __name__ == "__main__":
    main()
