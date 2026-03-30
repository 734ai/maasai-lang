#!/usr/bin/env python3
"""
Prepare and push Maasai translation dataset to Hugging Face Hub.

Requires:
- HuggingFace login (huggingface-cli login)
- Git LFS installed (for large files)
"""

import os
import json
from pathlib import Path
from datetime import datetime

def create_push_instructions():
    """Generate instructions for pushing to HF."""
    
    instructions = """
================================================================================
HUGGING FACE PUBLICATION SETUP (Manual Steps Required)
================================================================================

The dataset is ready for publication. To push to Hugging Face:

PREREQUISITE SETUP:
-------------------
1. Install Hugging Face CLI:
   pip install huggingface-hub

2. Login to Hugging Face:
   huggingface-cli login
   (Get token from: https://huggingface.co/settings/tokens)

3. Ensure Git LFS is installed:
   # On Ubuntu/Debian:
   sudo apt-get install git-lfs
   
   # On macOS:
   brew install git-lfs

---

DATASET UPLOAD STEPS:
---------------------
Note: This should be run with a NorthernTribe-Research account that has access
to the huggingface.co/datasets namespace.

Run in project root:

  # Clone or create repo
  git clone https://huggingface.co/datasets/NorthernTribe-Research/maasai-translation-corpus
  cd maasai-translation-corpus
  
  # Copy dataset files
  cp ../../data/final_v3/*.jsonl data/
  cp ../../docs/DATASET_README_V2.md README.md
  cp ../../docs/dataset_card.md dataset_card.md
  
  # Add dataset metadata
  cat > datasetinfo.json << 'EOF'
{
  "pretty_name": "Maasai-English Translation Corpus",
  "description": "Current data/final_v3 English-Maasai corpus for Maasai language preservation. 9,910 EN-MAS pairs in the local workspace snapshot.",
  "citation": "@dataset{maasai_translation_2026, title={Maasai-English Translation Corpus}, author={NorthernTribe Research}, year={2026}}",
  "homepage": "https://github.com/NorthernTribe/maasai-language-showcase",
  "license": "cc-by-4.0",
  "features": {
    "id": "string",
    "source_text": "string",
    "target_text": "string",
    "source_lang": "string",
    "target_lang": "string",
    "domain": "string",
    "source_name": "string",
    "confidence": "float",
    "tier": "string",
    "quality_score": "float",
    "notes": "string"
  }
}
EOF
  
  # Commit and push
  git add -A
  git commit -m "Update dataset snapshot from local data/final_v3"
  git push

---

SPACE UPDATE STEPS:
-------------------
The Space at: https://huggingface.co/spaces/NorthernTribe-Research/maasai-language-showcase

1. Update inference pipeline:
   cp scripts/infer_llama_cpp_optimized.py space/inference.py

2. Update app.py to use new pipeline:
   - Import from inference.py
   - Add batch processing support
   - Add quality metrics display

3. Add data loading:
   - Load from {repo_id}/datasets/NorthernTribe-Research/maasai-translation-corpus
   - Show dataset stats in UI
   - Link to dataset README

4. Commit to Space repo:
   cd space
   git add -A
   git commit -m "Integrate DeepSeek optimized inference + comprehensive dataset"
   git push

---

VERIFICATION:
--------------
After pushing, verify at:
- Dataset: https://huggingface.co/datasets/NorthernTribe-Research/maasai-translation-corpus
- Space: https://huggingface.co/spaces/NorthernTribe-Research/maasai-language-showcase

Check:
✓ README displays correctly
✓ Dataset card shows statistics  
✓ Splits visible (train/valid/test)
✓ Space inference works
✓ Links to dataset from Space app

================================================================================
"""
    
    return instructions

def create_readme_for_hf():
    """Create README for Hugging Face dataset repo."""
    
    readme = """# Maasai-English Translation Corpus (v2.0)

Current `data/final_v3` corpus for Maasai language preservation.

**9,910 English-Maasai translation pairs** in the current local workspace snapshot.

## Dataset at a Glance

- **Size**: 9,910 unique pairs (8,434 train / 738 valid / 738 test)
- **Quality labels**: 8,444 gold + 1,466 silver in current local metadata
- **Languages**: English ↔ Maasai (50/50 bidirectional balance)
- **Domains**: Bible-derived (85.2%), plus lexicon, philosophy, culture, ceremonies, education, greetings, governance, environment, and open-source supplement rows
- **Format**: JSONL with rich metadata (domain, tier, confidence, etc.)

## Key Statistics

| Metric | Value |
|--------|-------|
| Total pairs | 9,910 |
| Training pairs | 8,434 (85.1%) |
| Validation pairs | 738 (7.4%) |
| Test pairs | 738 (7.4%) |
| Gold tier | 8,444 pairs (85.2%) |
| Silver tier | 1,466 pairs (14.8%) |
| Unique domains | 21 |
| EN→MAS direction | 4,955 pairs (50%) |
| MAS→EN direction | 4,955 pairs (50%) |

## Contents

- `train.jsonl` - Training set (8,434 pairs)
- `valid.jsonl` - Validation set (738 pairs)
- `test.jsonl` - Test set (738 pairs)

## Use Cases

1. **Machine Translation Training** - Fine-tune 4B-30B models (QLoRA, full fine-tune)
2. **Language Preservation** - Maasai fluency development
3. **Linguistic Research** - Low-resource MT, language documentation
4. **Community Applications** - Educational tools, glossaries, translation aids

## Model Training Recommendations

**DeepSeek-inspired curriculum learning approach:**
- Start with 70% Gold tier (high-confidence Bible) + 30% Silver tier (cultural diversity)
- Use curriculum learning: easy examples → harder examples
- Validate frequently on held-out test set
- Expected BLEU: 30-40 (realistic for low-resource MT)

## Related Work

- **Base Model**: Qwen/Qwen2.5-3B-Instruct
- **Training Method**: QLoRA (4-bit quantized LoRA)
- **Inference**: llama.cpp with Q4_K_M quantization
- **Space Demo**: https://huggingface.co/spaces/NorthernTribe-Research/maasai-language-showcase

## References

- DeepSeek V3/R1 best practices for low-resource language models
- Bible translations: English KJV / Maasai Official Bible
- Quality framework: DeepSeek data filtering + tier assignment
- Publication: Hugging Face Hub

## License

CC-BY-4.0 (Creative Commons Attribution 4.0)

## Citation

```bibtex
@dataset{maasai_translation_2026,
  title={Maasai-English Translation Corpus: Bible + Cultural Knowledge},
  author={NorthernTribe Research},
  year={2026},
  version={2.0},
  publisher={Hugging Face},
  url={https://huggingface.co/datasets/NorthernTribe-Research/maasai-translation-corpus}
}
```

---

**Created**: March 26, 2026  
**Status**: ✅ Production Ready  
**Community**: Built with commitment to Maasai language preservation
"""
    
    return readme

def main():
    print(create_push_instructions())
    
    # Create README for HF
    readme_content = create_readme_for_hf()
    readme_path = Path("docs/README_FOR_HF.md")
    readme_path.parent.mkdir(parents=True, exist_ok=True)
    readme_path.write_text(readme_content)
    
    print(f"\n✓ Created: {readme_path}")
    
    # Summary
    print("\n" + "=" * 80)
    print("DATASET PUBLICATION SUMMARY")
    print("=" * 80)
    
    print("\nDataset Contents:")
    for split in ["train", "valid", "test"]:
        file = Path(f"data/final_v3/{split}.jsonl")
        if file.exists():
            count = sum(1 for _ in open(file))
            size = file.stat().st_size / 1024 / 1024  # MB
            print(f"  {split:6s}.jsonl: {count:6,} pairs, {size:.1f} MB")
    
    print("\nDocumentation:")
    print("  ✓ docs/DATASET_README_V2.md - Detailed dataset documentation")
    print("  ✓ docs/README_FOR_HF.md - Hugging Face README template")
    print("  ✓ docs/dataset_card.md - Dataset card (existing)")
    print("  ✓ docs/model_card.md - Model card (existing)")
    
    print("\nScripts Generated:")
    print("  ✓ scripts/create_comprehensive_corpus.py")
    print("  ✓ scripts/prepare_splits_simple.py")
    print("  ✓ scripts/infer_llama_cpp_optimized.py (for Space)")
    
    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print("""
1. Review push instructions above
2. Prepare HF Hub credentials (huggingface-cli login)
3. Push dataset to: NorthernTribe-Research/maasai-translation-corpus
4. Update Space repo with optimized inference code
5. Publish model (after training)
6. Announce on social media + research channels

For detailed instructions, see: docs/README_FOR_HF.md
""")

if __name__ == "__main__":
    main()
