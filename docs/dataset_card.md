---
language:
  - en
  - mas
pretty_name: Maasai-English Translation Corpus
license: cc-by-4.0
task_categories:
  - translation
size_categories:
  - 1K<n<10K
tags:
  - translation
  - maasai
  - low-resource
  - cultural-preservation
  - bilingual
---
# Maasai-English Translation Corpus

Parallel English↔Maasai translation pairs for low-resource MT, language preservation, and culturally grounded tooling.

## Overview

- Total pairs: 9,406
- Splits: 7,991 train / 707 valid / 708 test
- Directions: 4,703 en→mas and 4,703 mas→en
- Quality tiers: 9,124 gold and 282 silver
- Main sources: 8,444 Bible-aligned pairs, 680 cultural manual pairs, 70 knowledge-driven cultural pairs, 132 public-domain Hollis proverb pairs, and 80 ASJP lexical pairs

The public dataset repo publishes the parallel-pair corpus from `data/final_v3/*.jsonl`. Training code may derive instruction prompts at load time, but `prompt` and `completion` are not stored in this public dataset.

## Record Schema

Each JSONL row contains:

| Field | Type | Description |
|---|---|---|
| `id` | string | Unique example identifier |
| `source_text` | string | Source sentence |
| `target_text` | string | Target translation |
| `source_lang` | string | `en` or `mas` |
| `target_lang` | string | `mas` or `en` |
| `domain` | string | Domain label such as `bible`, `culture`, or `philosophy` |
| `source_name` | string | Provenance label |
| `quality_score` | float | Normalized quality score |
| `notes` | string | Free-form provenance notes |
| `tier` | string | Quality tier (`gold` or `silver`) |
| `confidence` | float | Confidence score for the pair |
| `quality_assessment` | string | Quality assessment label |

Example:

```json
{
  "id": "bible-en2mas-00000",
  "source_text": "And there was evening, and there was morning- the first day.",
  "target_text": "Nerishu Enkai ewang'an aitung'uaa enaimin.",
  "source_lang": "en",
  "target_lang": "mas",
  "domain": "bible",
  "source_name": "bible_english_maasai",
  "quality_score": 0.98,
  "notes": "Aligned Bible verse",
  "tier": "gold",
  "confidence": 0.98,
  "quality_assessment": "authenticated_parallel_text"
}
```

## Domain Coverage

Top domains in the current release:

- `bible`: 8,444
- `proverbs`: 158
- `philosophy`: 100
- `culture`: 84
- `lexicon`: 80
- `environment`: 64
- `education`: 60
- `ceremony`: 58
- `greetings`: 52
- `governance`: 48
- `livestock`: 46
- `daily_life`: 46

## Intended Use

- Train English↔Maasai translation models
- Evaluate low-resource MT systems on a held-out split
- Support glossary-aware and culturally grounded translation tools
- Aid documentation and preservation workflows

## Limitations

- Maasai orthography is not fully standardized.
- Coverage is strongest in Bible-aligned and culturally grounded domains.
- Outputs trained on this corpus should still be reviewed by native Maa speakers for formal use.
- Dialect and regional variation are only partially represented.

## Related Assets

- Glossary: `glossary/maasai_glossary.json`
- Space: `NorthernTribe-Research/maasai-language-showcase`
- Model repo: `NorthernTribe-Research/maasai-en-mt`

## Citation

If you use this dataset, cite NorthernTribe-Research and note that the corpus serves Maasai language preservation and accessibility work.
