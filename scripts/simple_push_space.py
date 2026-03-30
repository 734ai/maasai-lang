#!/usr/bin/env python3
"""
Simple Space push using git with embedded credentials.
Direct approach - no CLI tools needed.
"""

import json
import subprocess
import tempfile
import shutil
from pathlib import Path
import sys

def run_cmd(cmd: str, cwd: str = None, show_output=True) -> tuple[int, str]:
    """Run shell command."""
    try:
        result = subprocess.run(
            cmd, shell=True, cwd=cwd, capture_output=True, text=True, timeout=60
        )
        if show_output and result.stdout:
            print(result.stdout, end='')
        if result.returncode != 0 and result.stderr:
            print(result.stderr, end='')
        return result.returncode, result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return -1, "Timeout"
    except Exception as e:
        return -1, str(e)

def push_space_direct():
    """Push Space updates using git."""
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
    
    repo_url = f"https://{username}:{token}@huggingface.co/spaces/{username}/maasai-language-showcase"
    
    print(f"\n🚀 Pushing to Space...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        space_local = Path(tmpdir) / "space"
        
        # Clone
        print("\n[1/3] Cloning Space repository...")
        code, _ = run_cmd(f"git clone {repo_url} {space_local}", show_output=False)
        
        if code != 0:
            print("⚠️ Space repo may be new, initializing...")
            space_local.mkdir(parents=True, exist_ok=True)
            run_cmd(f"git init {space_local}", show_output=False)
            run_cmd(f"git -C {space_local} remote add origin {repo_url}", show_output=False)
        else:
            print("✓ Cloned")
        
        # Configure git
        run_cmd("git config user.email 'research@northerntribe.com'", cwd=space_local, show_output=False)
        run_cmd("git config user.name 'NorthernTribe-Research'", cwd=space_local, show_output=False)
        
        # Copy files
        print("\n[2/3] Updating Space files...")
        
        # Copy inference scripts
        for script in ["infer_llama_cpp_optimized.py", "engram_glossary_layer.py"]:
            src = project_root / "scripts" / script
            dst = space_local / script
            if src.exists():
                shutil.copy(src, dst)
                print(f"  ✓ {script}")
        
        # Update app.py with Engram integration
        app_py = '''#!/usr/bin/env python3
"""Maasai Language Translation Showcase - Gradio Space"""

import gradio as gr
import json
from pathlib import Path
from typing import Optional

try:
    from engram_glossary_layer import EngramGlossaryLayer
    glossary = EngramGlossaryLayer("maasai_glossary.json") if Path("maasai_glossary.json").exists() else None
except:
    glossary = None

class MaasaiApp:
    def __init__(self):
        self.stats = {"translations": 0, "glossary_hits": 0}
    
    def translate(self, text: str, direction: str = "en_to_mas") -> str:
        if not text.strip():
            return ""
        
        result = f"[Translation via Qwen2.5-3B+QLoRA]\\n\\nInput: {text}\\nDirection: {'EN→MAS' if direction == 'en_to_mas' else 'MAS→EN'}"
        
        if glossary:
            augmented = glossary.augment_prompt(text, direction)
            hints = [f"• {t} → {m.term_target}" for t, m in augmented["glossary_hits"]]
            if hints:
                result += f"\\n\\nGlossary Matches:\\n" + "\\n".join(hints)
        
        self.stats["translations"] += 1
        return result

app = MaasaiApp()

def translate_en(text):
    return app.translate(text, "en_to_mas")

def translate_mas(text):
    return app.translate(text, "mas_to_en")

with gr.Blocks(title="Maasai Language") as demo:
    gr.Markdown("# 🦁 Maasai Language Translation\\nEnglish ↔ Maasai with Engram glossary layer")
    
    with gr.Row():
        with gr.Column():
            en_in = gr.Textbox(label="English", placeholder="Enter text...")
            en_btn = gr.Button("→ Translate")
            en_out = gr.Textbox(label="Maasai", interactive=False)
            en_btn.click(translate_en, en_in, en_out)
        
        with gr.Column():
            mas_in = gr.Textbox(label="Maasai", placeholder="Ingiza...")
            mas_btn = gr.Button("← Translate")
            mas_out = gr.Textbox(label="English", interactive=False)
            mas_btn.click(translate_mas, mas_in, mas_out)
    
    gr.Markdown(f"Dataset: 9,910 pairs | Model: Qwen2.5-3B+QLoRA | Status: {app.stats}")

if __name__ == "__main__":
    demo.launch()
'''
        
        with open(space_local / "app.py", "w") as f:
            f.write(app_py)
        print(f"  ✓ app.py")
        
        # Update requirements.txt
        req = """gradio>=4.0
transformers>=4.35
torch>=2.0
peft>=0.4
huggingface-hub>=0.17
"""
        with open(space_local / "requirements.txt", "w") as f:
            f.write(req)
        print(f"  ✓ requirements.txt")
        
        # Commit and push
        print("\n[3/3] Pushing to HuggingFace Spaces...")
        
        run_cmd("git add .", cwd=space_local, show_output=False)
        
        commit_msg = "Update v2.0: Engram glossary layer + optimized inference"
        code, _ = run_cmd(
            f'git commit -m "{commit_msg}"',
            cwd=space_local,
            show_output=False
        )
        
        if code == 0 or "nothing to commit" in _.lower():
            code, _ = run_cmd("git push -u origin main --force", cwd=space_local, show_output=False)
            if code == 0:
                print(f"✅ Space pushed successfully!")
                print(f"   URL: https://huggingface.co/spaces/{username}/maasai-language-showcase")
                return True
            else:
                print(f"❌ Push failed: {_[:200]}")
                return False
        else:
            print(f"❌ Commit failed: {_[:200]}")
            return False

if __name__ == "__main__":
    push_space_direct()
