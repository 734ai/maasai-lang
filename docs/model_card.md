---
language:
  - en
  - mas
library_name: transformers
pipeline_tag: translation
base_model: Qwen/Qwen2.5-3B-Instruct
license: apache-2.0
tags:
  - translation
  - maasai
  - maa
  - low-resource
  - qlora
  - cultural-preservation
datasets:
  - NorthernTribe-Research/maasai-translation-corpus
---
# Maasai-English Translation Model

English↔Maasai translation model for low-resource machine translation and language-preservation workflows.

## Model Details

- Repository name: `maasai-en-mt`
- Base model: inspect the published checkpoint config or adapter metadata for the exact source model used in that release
- Fine-tuning recipe: QLoRA / LoRA adapters
- Target directions: English→Maasai and Maasai→English
- Paired public dataset: `NorthernTribe-Research/maasai-translation-corpus`

## Training Data

The training recipe in this repo uses the public parallel corpus from `data/final_v3`:

- 9,910 total pairs
- 8,434 train / 738 valid / 738 test
- 4,955 en→mas and 4,955 mas→en
- 8,444 gold-tier and 1,466 silver-tier examples

This release now includes a larger open-source supplement layer from public-domain Hollis proverb pairs, conservative public-domain Hinde vocabulary pairs, and the CC BY 4.0 ASJP Maasai wordlist, in addition to the Bible-derived and curated cultural data already present in the local corpus.

The raw published dataset stores parallel pairs and metadata. The trainer constructs instruction prompts at runtime when needed, so the model can be trained from either prompt/completion records or plain translation pairs. The trainer does not require `id` or `quality_assessment`, which matters because some older `cultural_manual` rows still omit those fields.

## Intended Use

- Research and benchmarking for English↔Maasai translation
- Language preservation and educational tooling
- Culturally grounded translation assistance with human review

## Not Intended For

- Legal, medical, or safety-critical translation
- Unreviewed authoritative translation in public-facing settings
- Claims of dialect-complete or culturally exhaustive coverage

## Limitations

- Maasai remains a low-resource language, so quality will vary by domain.
- The corpus is strongest in Bible-derived and cultural content.
- Orthographic and dialectal variation are not fully normalized.
- The current dataset schema is not fully normalized across every row.
- Native Maa speaker review remains necessary for formal or sensitive use.

## Hub Download Metrics

This repository publishes a lightweight `meta.yaml` file alongside the model card. The file is metadata only and is not a loadable checkpoint. It exists so scaffold releases and early repo states still expose a stable Hub metadata artifact that can act as a download-count anchor before full model weights are uploaded.

When adapter or merged model files are published, the real model artifacts remain the primary release assets. The metadata file is retained to keep the repo machine-readable and to avoid treating placeholder states as if they were runnable weights.

## Evaluation

This template intentionally avoids fixed metric claims. When a new checkpoint is published, add the measured BLEU, chrF++, and glossary-sensitive evaluation results for that exact run.

## Related Assets

- Dataset: `NorthernTribe-Research/maasai-translation-corpus`
- Space: `NorthernTribe-Research/maasai-language-showcase`
- Glossary file used by the app: `data/glossary/maasai_glossary.json`

## Citation

If you publish results based on this model, cite the model repo, the paired dataset, and NorthernTribe-Research.
