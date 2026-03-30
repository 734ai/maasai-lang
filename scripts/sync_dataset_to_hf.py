#!/usr/bin/env python3
"""
Synchronize and update Maasai dataset on HuggingFace Hub.

This script:
1. Validates local dataset
2. Creates/updates dataset card with statistics
3. Uploads or updates splits (train/valid/test.jsonl)
4. Pushes glossary and metadata
5. Verifies upload completeness

Usage:
    python scripts/sync_dataset_to_hf.py \
        --repo_id "NorthernTribe-Research/maasai-translation-corpus" \
        --data_dir "data/final_v3" \
        --push_glossary true
"""

import argparse
import json
import logging
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

from huggingface_hub import HfApi, create_repo, upload_file, upload_folder

logger = logging.getLogger("sync_dataset_to_hf")


def setup_logging():
    """Configure logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Synchronize Maasai dataset to HuggingFace Hub",
    )
    
    parser.add_argument(
        "--repo_id",
        type=str,
        default="NorthernTribe-Research/maasai-translation-corpus",
        help="HF dataset repo ID",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/final_v3",
        help="Path to dataset directory (with train/valid/test JSONL)",
    )
    parser.add_argument(
        "--push_glossary",
        type=lambda x: x.lower() in ("true", "1", "yes"),
        default=True,
        help="Whether to include glossary",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HF API token (or set HF_TOKEN env var)",
    )
    
    return parser.parse_args()


def count_jsonl_entries(filepath: Path) -> int:
    """Count lines in JSONL file."""
    try:
        with open(filepath) as f:
            return sum(1 for _ in f)
    except Exception as e:
        logger.error(f"Error counting {filepath}: {e}")
        return 0


def load_jsonl_rows(filepath: Path) -> list[dict[str, Any]]:
    """Load JSONL rows from disk."""
    rows: list[dict[str, Any]] = []
    try:
        with open(filepath, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
    except Exception as e:
        logger.error(f"Error loading {filepath}: {e}")
    return rows


def validate_dataset(data_dir: str) -> Dict[str, Any]:
    """Validate dataset structure and return statistics."""
    
    data_path = Path(data_dir)
    logger.info(f"🔍 Validating dataset in {data_path}")
    
    stats = {
        "valid": True,
        "files": {},
        "total_samples": 0,
        "splits": {},
        "tier_counts": Counter(),
        "domain_counts": Counter(),
        "source_counts": Counter(),
        "lang_pair_counts": Counter(),
        "missing_id_count": 0,
        "missing_quality_assessment_count": 0,
    }
    
    # Check for required splits
    required_splits = {
        "train.jsonl": "Training set",
        "valid.jsonl": "Validation set",
        "test.jsonl": "Test set",
    }
    
    for filename, description in required_splits.items():
        filepath = data_path / filename
        if not filepath.exists():
            logger.error(f"   ❌ Missing: {description} ({filename})")
            stats["valid"] = False
        else:
            rows = load_jsonl_rows(filepath)
            count = len(rows)
            size_mb = filepath.stat().st_size / 1024 / 1024
            stats["files"][filename] = {
                "size_mb": round(size_mb, 2),
                "count": count,
                "description": description,
            }
            stats["splits"][filename.replace(".jsonl", "")] = count
            stats["total_samples"] += count
            for row in rows:
                stats["tier_counts"][str(row.get("tier", "unknown"))] += 1
                stats["domain_counts"][str(row.get("domain", "unknown"))] += 1
                stats["source_counts"][str(row.get("source_name", "unknown"))] += 1
                stats["lang_pair_counts"][
                    f"{row.get('source_lang', 'unknown')}->{row.get('target_lang', 'unknown')}"
                ] += 1
                if "id" not in row:
                    stats["missing_id_count"] += 1
                if "quality_assessment" not in row:
                    stats["missing_quality_assessment_count"] += 1
            logger.info(f"   ✅ {description}: {count} samples ({size_mb:.1f} MB)")
    
    if not stats["valid"]:
        logger.error("❌ Dataset validation failed")
        return stats
    
    # Check for glossary
    glossary_path = data_path.parent / "glossary" / "maasai_glossary.json"
    if glossary_path.exists():
        with open(glossary_path) as f:
            glossary = json.load(f)
        logger.info(f"   ✅ Glossary: {len(glossary)} terms")
        stats["glossary_terms"] = len(glossary)
    
    logger.info(f"✅ Dataset validation passed ({stats['total_samples']} total samples)")
    return stats


def create_dataset_card(stats: Dict, repo_id: str) -> str:
    """Create comprehensive dataset card."""

    total_pairs = int(stats.get("total_samples", 0))
    train_pairs = int(stats.get("splits", {}).get("train", 0))
    valid_pairs = int(stats.get("splits", {}).get("valid", 0))
    test_pairs = int(stats.get("splits", {}).get("test", 0))
    gold_pairs = int(stats.get("tier_counts", {}).get("gold", 0))
    silver_pairs = int(stats.get("tier_counts", {}).get("silver", 0))
    glossary_terms = stats.get("glossary_terms", "N/A")
    en_to_mas = int(stats.get("lang_pair_counts", {}).get("en->mas", 0))
    mas_to_en = int(stats.get("lang_pair_counts", {}).get("mas->en", 0))
    missing_id_count = int(stats.get("missing_id_count", 0))
    missing_quality_assessment_count = int(stats.get("missing_quality_assessment_count", 0))

    def pct(count: int) -> float:
        return round((count / total_pairs) * 100, 1) if total_pairs else 0.0

    top_domain_lines = "\n".join(
        f"- `{domain}`: {count}"
        for domain, count in stats.get("domain_counts", Counter()).most_common(10)
    ) or "- No domain breakdown available"

    top_source_lines = "\n".join(
        f"- `{source}`: {count}"
        for source, count in stats.get("source_counts", Counter()).most_common(5)
    ) or "- No source breakdown available"

    dataset_card = f"""---
license: cc-by-4.0
language:
  - en
  - mas
task_categories:
  - translation
tags:
  - machine-translation
  - low-resource-language
  - maasai
  - language-preservation
  - parallel-corpus
pretty_name: Maasai-English Translation Corpus
---

# Maasai-English Translation Corpus

Parallel English↔Maasai translation pairs for low-resource MT, language preservation, and culturally grounded tooling.

![Dataset Size](https://img.shields.io/badge/Samples-{stats.get('total_samples', 'N/A')}-blue)
![Quality](https://img.shields.io/badge/Quality-{pct(gold_pairs)}%25%20Gold%2C%20{pct(silver_pairs)}%25%20Silver-brightgreen)
![License](https://img.shields.io/badge/License-CC--BY--4.0-orange)

---

## Dataset Summary

- Current local corpus: {total_pairs} pairs from `data/final_v3`
- Splits: {train_pairs} train / {valid_pairs} valid / {test_pairs} test
- Directions: {en_to_mas} en→mas and {mas_to_en} mas→en
- Quality labels in current local metadata: {gold_pairs} gold / {silver_pairs} silver
- Glossary terms: {glossary_terms}
- Schema note: {missing_id_count} rows omit `id`; {missing_quality_assessment_count} rows omit `quality_assessment`

### Key Statistics

| Metric | Value |
|--------|-------|
| **Total Pairs** | {total_pairs} |
| **Training Split** | {train_pairs} pairs |
| **Validation Split** | {valid_pairs} pairs |
| **Test Split** | {test_pairs} pairs |
| **Quality (Gold)** | {gold_pairs} pairs ({pct(gold_pairs)}%) |
| **Quality (Silver)** | {silver_pairs} pairs ({pct(silver_pairs)}%) |
| **Glossary Terms** | {glossary_terms} protected cultural terms |

---

## Splits

The dataset is split into three parts for standard machine learning practice:

1. **`train.jsonl`** ({train_pairs} samples)
   - Used for model training
   - Balanced English→Maasai and Maasai→English pairs

2. **`valid.jsonl`** ({valid_pairs} samples)
   - Used for hyperparameter tuning
   - Representative of test distribution

3. **`test.jsonl`** ({test_pairs} samples)
   - Held-out test set for final evaluation
   - Never used in training

---

## Data Fields

Each record in `.jsonl` format contains the common fields below. The current local corpus is not fully schema-normalized, so `id` and `quality_assessment` are not present on every row.

```json
{{
  "id": "bible-mas2en-00938",
  "source_lang": "mas",
  "target_lang": "en",
  "source_text": "Metaa Olaitoriani Enkai ino te nebo iyie!",
  "target_text": "Then all the people shall say, Amen!",
  "domain": "bible",
  "source_name": "bible_english_maasai",
  "quality_score": 0.98,
  "tier": "gold",
  "confidence": 0.98,
  "notes": "Bible-derived sentence chunk pair",
  "quality_assessment": {{
    "overall_score": 0.98,
    "tier": "gold",
    "issues": []
  }}
}}
```

### Field Descriptions

- **`id`**: Optional identifier for the pair; older `cultural_manual` rows currently omit it
- **`source_lang`** / **`target_lang`**: Language codes (`en` = English, `mas` = Maasai)
- **`source_text`** / **`target_text`**: Parallel text in both languages
- **`domain`**: Document domain (bible, culture, proverbs, lexicon, etc.)
- **`source_name`**: Source dataset (`bible_english_maasai`, `cultural_manual`, etc.)
- **`quality_score`**: Confidence score (0–1, higher is better)
- **`tier`**: Dataset-internal quality label from the local curation pipeline
- **`confidence`**: Confidence score carried in the local metadata
- **`notes`**: Metadata and annotations
- **`quality_assessment`**: Optional structured quality metadata

---

## Top Domains

{top_domain_lines}

---

## Data Sources

{top_source_lines}

---

## Sub-Tribes & Dialects Covered

This corpus represents multiple Maasai iloshon (sections):

- 🔴 **Ilkisongo** (Tanzania) — Largest section
- 🔴 **Ilpurko** (Kenya) — Standard dialect reference
- 🔴 **Ildamat** (Narok, Kenya) — Major Kenyan group
- 🔴 **Ilkeekonyokie** (Ngong Hills) — Near Nairobi
- 🔴 **Ilkaputiei** (Kitengela) — Southeast Kenya
- 🔴 **Ilmatapato** (Kajiado) — Southern Kenya
- 🔴 **Laikipiak** (Laikipia Plateau) — Historically significant
- 🔴 **Isiria** (Mara) — Near Maasai Mara
- 🔴 **Ilarusa** (Arusha, Tanzania) — More settled
- 🔴 **Ilparakuyo** (Eastern Tanzania) — Pastoral group
- 🔴 **Ldikiri** (Various) — Unique traditions
- 🔴 **Samburu** (Northern Kenya) — Related Maa-speakers
- 🔴 **Lmomonyot** (Samburu region) — Samburu variant

---

## Quality Assessment

### Quality Tiers

**Gold Tier** ({pct(gold_pairs)}%, {gold_pairs} pairs)
- Current local metadata label carried forward from the curation pipeline
- Dominated by Bible-derived rows

**Silver Tier** ({pct(silver_pairs)}%, {silver_pairs} pairs)
- Current local metadata label for curated, knowledge-driven, and open-source supplement rows
- Useful for maintaining non-Bible coverage in training and evaluation

### Validation Methodology

The local corpus carries forward metadata from earlier validation and curation passes, but the current files are not fully normalized. Before a long production run, spot-check alignment quality and review Bible-derived rows carefully.

---

## Limitations & Caveats

⚠️ **Please note:**

- **Orthography Variation:** Maasai orthography is not standardized; this corpus reflects variation found in sources
- **Domain Bias:** Heavy emphasis on Bible-derived text ({pct(stats.get('domain_counts', {}).get('bible', 0))}%); limited modern technical vocabulary
- **Vocabulary Coverage:** Limited slang, colloquialisms, and contemporary terms
- **Dialect Diversity:** While multiple sections represented, not equally distributed
- **Schema Variation:** Some older rows omit `id` and `quality_assessment`
- **Manual Review:** Do not read the local `tier` labels as proof of native-speaker review for every row

---

## Usage & Loading

### Load with Datasets Library

```python
from datasets import load_dataset

dataset = load_dataset("NorthernTribe-Research/maasai-translation-corpus")

# Access splits
train = dataset["train"]
valid = dataset["validation"]
test = dataset["test"]

# Inspect a sample
print(train[0])
```

### Load with Pandas

```python
import pandas as pd

train_df = pd.read_json("train.jsonl", lines=True)
print(train_df.head())
```

### Manual Loading

```python
import json

def load_jsonl(filepath):
    data = []
    with open(filepath) as f:
        for line in f:
            data.append(json.loads(line))
    return data

train_data = load_jsonl("train.jsonl")
```

---

## Citation

Please cite this corpus when used in research:

```bibtex
@dataset{{maasai_translation_corpus_2026,
  title={{Maasai-English Translation Corpus}},
  author={{NorthernTribe-Research}},
  year={{2026}},
  publisher={{Hugging Face}},
  url={{https://huggingface.co/datasets/NorthernTribe-Research/maasai-translation-corpus}}
}}
```

---

## License

This dataset is licensed under **CC-BY-SA-4.0**:

- ✅ You can use it for research, commercial, and non-commercial purposes
- ✅ You can modify and redistribute it
- ✅ You **must** attribute NorthernTribe-Research
- ✅ Any derivatives **must** use the same license

See [Creative Commons BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) for details.

---

## Community & Support

- **Project:** [Maasai Language Showcase](https://huggingface.co/spaces/NorthernTribe-Research/maasai-language-showcase)
- **Models:** [Maasai Translation Models](https://huggingface.co/NorthernTribe-Research)
- **Discussions:** [Dataset Discussions](https://huggingface.co/datasets/NorthernTribe-Research/maasai-translation-corpus/discussions)

---

## Version History

- **v1.0** (March 27, 2026): Initial public release
  - 9,910 parallel pairs in the current local snapshot
  - 103-entry glossary
  - 14+ Maasai sections represented

---

Built with ❤️ for the preservation and accessibility of the Maasai (Maa) language.

*Enkutuk oo lMaa* — The Maasai Language

---

**Data Controller:** NorthernTribe-Research  
**Last Updated:** {datetime.now().strftime('%Y-%m-%d')}  
**Repository:** {repo_id}
"""
    
    return dataset_card


def sync_dataset_to_hub(
    repo_id: str,
    data_dir: str,
    push_glossary: bool = True,
    token: Optional[str] = None,
) -> bool:
    """
    Synchronize dataset to HuggingFace Hub.
    
    Args:
        repo_id: HF dataset repo ID
        data_dir: Local dataset directory
        push_glossary: Whether to include glossary
        token: HF API token
        
    Returns:
        True if successful
    """
    
    data_path = Path(data_dir)
    
    if not data_path.exists():
        logger.error(f"❌ Data directory not found: {data_path}")
        return False
    
    # Validate dataset
    stats = validate_dataset(str(data_path))
    if not stats["valid"]:
        return False
    
    api = HfApi(token=token)
    
    try:
        # Create or access repo
        logger.info(f"📦 Creating or accessing dataset repository: {repo_id}")
        try:
            repo_url = create_repo(
                repo_id=repo_id,
                repo_type="dataset",
                private=False,
                exist_ok=True,
                token=token,
            )
            logger.info(f"✅ Repository: {repo_url}")
        except Exception as e:
            logger.warning(f"⚠️  {e}")
        
        # Create dataset card
        logger.info("📝 Creating dataset card...")
        card_content = create_dataset_card(stats, repo_id)
        card_path = data_path / "README.md"
        with open(card_path, "w") as f:
            f.write(card_content)
        logger.info("✅ Dataset card created")
        
        # Upload data splits
        logger.info(f"📤 Uploading data splits to {repo_id}...")
        for split_file in ["train.jsonl", "valid.jsonl", "test.jsonl"]:
            filepath = data_path / split_file
            if filepath.exists():
                logger.info(f"   Uploading {split_file}...")
                upload_file(
                    path_or_fileobj=str(filepath),
                    path_in_repo=split_file,
                    repo_id=repo_id,
                    repo_type="dataset",
                    token=token,
                    commit_message=f"Update {split_file}",
                )
                logger.info(f"   ✅ {split_file} uploaded")
        
        # Upload glossary
        if push_glossary:
            glossary_path = data_path.parent / "glossary" / "maasai_glossary.json"
            if glossary_path.exists():
                logger.info("📝 Uploading glossary...")
                upload_file(
                    path_or_fileobj=str(glossary_path),
                    path_in_repo="glossary.json",
                    repo_id=repo_id,
                    repo_type="dataset",
                    token=token,
                    commit_message="Update glossary",
                )
                logger.info("✅ Glossary uploaded")
        
        # Upload dataset card
        logger.info("📝 Uploading dataset card...")
        upload_file(
            path_or_fileobj=str(card_path),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
            commit_message="Update dataset card",
        )
        logger.info("✅ Dataset card uploaded")
        
        logger.info(f"✅ Dataset synchronization complete!")
        logger.info(f"   View at: https://huggingface.co/datasets/{repo_id}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Sync failed: {e}", exc_info=True)
        return False


def main():
    """Main entry point."""
    setup_logging()
    args = parse_args()
    
    logger.info("=" * 70)
    logger.info("HuggingFace Dataset Sync — Maasai-English Translation")
    logger.info("=" * 70)
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    logger.info("")
    
    # Sync dataset
    success = sync_dataset_to_hub(
        repo_id=args.repo_id,
        data_dir=args.data_dir,
        push_glossary=args.push_glossary,
        token=args.token,
    )
    
    if success:
        logger.info("")
        logger.info("=" * 70)
        logger.info("✅ Dataset sync completed successfully!")
        logger.info("=" * 70)
        logger.info("")
        sys.exit(0)
    else:
        logger.error("")
        logger.error("=" * 70)
        logger.error("❌ Dataset sync failed. Check logs above.")
        logger.error("=" * 70)
        sys.exit(1)


if __name__ == "__main__":
    main()
