# Maasai Language Showcase
## English ↔ Maasai Translation + Maasai Speech Transcription

A modern English ↔ Maasai translation and Maasai speech transcription showcase built for language preservation, accessibility, and low-resource AI research.

---

## Features

- **English → Maasai translation** — powered by a QLoRA fine-tuned model
- **Maasai → English translation** — bidirectional support
- **Maasai speech transcription** — via Microsoft Paza ASR
- **Speech → Translation pipeline** — transcribe then translate
- **Cultural terminology preservation** — 90+ term glossary with protected terms
- **Sub-tribe awareness** — covers Ldikiri, Laikipiak, Samburu (Lmomonyot), Ilkisongo, Ilpurko, and more

## Architecture

| Component | Model | Purpose |
|-----------|-------|---------|
| Translation | `Qwen/Qwen2.5-3B-Instruct` (QLoRA) | English ↔ Maasai text translation and Maa composition |
| Speech | `microsoft/paza-whisper-large-v3-turbo` | Maasai speech transcription |
| UI | Gradio | Interactive Hugging Face Space |

## Quick Start

```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env

# Generate training data
python scripts/generate_cultural_pairs.py
python scripts/prepare_data.py

# Train
bash training/run_train.sh

# Evaluate
python scripts/evaluate_mt.py

# Run Space locally
python space/app.py

# Check hosted Space health
.venv/bin/python scripts/check_space_health.py
```

## Publish to Hugging Face

```bash
# Optional: set token explicitly (recommended)
export HF_TOKEN=hf_xxx

# Preview publish plan (no uploads)
python scripts/publish_to_hf.py

# Export refreshed publish bundles locally
python scripts/publish_to_hf.py --export-dir dist/hf_publish --create-model-repo

# Execute publish for Space + dataset + model repo scaffold
python scripts/publish_to_hf.py --execute --create-model-repo
```

The publisher uploads:
- Space bundle from `space/` + glossary data
- Dataset bundle from `data/final_v3/` + dataset card
- Model artifacts from `outputs/maasai-en-mt-qlora` when real weights exist
- A model repo scaffold with `README.md`, `meta.yaml`, and status metadata when local outputs are still mock placeholders

Legacy helpers such as `scripts/push_hf_model.py` and `scripts/push_trained_model.py` now route through the same publisher so model cards and Hub metadata stay consistent.

## Daily Training Automation

For daily resumable training from Hugging Face:

```bash
# Colab or any GPU runner
python scripts/train_daily_from_hf.py \
  --dataset-repo NorthernTribe-Research/maasai-translation-corpus \
  --model-repo NorthernTribe-Research/maasai-en-mt \
  --max-steps 800 \
  --save-steps 100
```

For Kaggle, use the retrying wrapper so unsupported P100 assignments are rerun automatically until a supported GPU is allocated:

```bash
KAGGLE_CONFIG_DIR="$PWD" .venv/bin/python scripts/run_kaggle_training.py \
  --max-attempts 5 \
  --report-to wandb \
  --embed-local-hf-token
```

- `scripts/train_daily_from_hf.py` downloads the dataset from HF, restores the latest checkpoint from the model repo, and pushes new checkpoints back with `hub_strategy="checkpoint"`.
- The default open base model is `Qwen/Qwen2.5-3B-Instruct` so Kaggle and other GPU runners are not blocked on gated-model access.
- `.github/workflows/daily-train.yml` schedules the same flow for a `self-hosted` GPU runner, reads `HF_DATASET_REPO` / `HF_MODEL_REPO` from GitHub repo variables when set, and uploads a per-run manifest artifact from the runner temp directory.
- For the future `734ai` GitHub repo, GitHub becomes the control plane while Hugging Face stays the durable checkpoint backend.

## Project Structure

```
├── data/
│   ├── raw/              # Raw parallel corpora
│   ├── processed/        # Train/valid/test JSONL
│   ├── glossary/         # Maasai terminology glossary
│   └── eval/             # Evaluation results
├── scripts/
│   ├── prepare_data.py   # Data cleaning & formatting
│   ├── train_qlora.py    # QLoRA fine-tuning
│   ├── evaluate_mt.py    # BLEU/chrF++ evaluation
│   ├── infer_translate.py    # Translation inference
│   ├── infer_asr.py      # Speech transcription
│   ├── generate_cultural_pairs.py # Cultural data generator
│   └── build_glossary.py # Glossary validation
├── src/
│   ├── config.py         # Project configuration
│   ├── prompts.py        # Prompt templates
│   ├── glossary.py       # Glossary loader
│   ├── preprocessing.py  # Text cleaning
│   ├── postprocessing.py # Output post-processing
│   ├── metrics.py        # Evaluation metrics
│   └── utils.py          # Utilities
├── space/
│   └── app.py            # Gradio demo application
├── training/
│   ├── lora_config.yaml  # LoRA configuration
│   └── run_train.sh      # Training script
└── docs/
    ├── model_card.md
    ├── dataset_card.md
    ├── evaluation_plan.md
    └── deployment.md
```

## Maasai Sections Covered

Ldikiri · Laikipiak · Samburu (Lmomonyot) · Ilkisongo · Ilpurko · Iloitai · Ildamat · Ilkeekonyokie · Iloodokilani · Ilkaputiei · Ilmatapato · Ilwuasinkishu · Isiria · Ilmoitanik · Ildalalekutuk · Ilaitayiok · Ilarusa · Ilparakuyo

## Limitations

- Low-resource language quality may vary
- Maasai orthography is not fully standardized
- Outputs should be reviewed by native Maa speakers for formal use
- Not all dialectal variations are equally represented

## License

TBD

## Acknowledgments

Built by NorthernTribe-Research for the preservation and accessibility of the Maasai (Maa) language.
