# Deployment Guide

## Local Development

```bash
# Clone repo
git clone <repo-url>
cd maasai-language-showcase

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy environment config
cp .env.example .env
# Edit .env with your HF token
```

## Data Pipeline

```bash
# 1. Generate cultural training pairs
python scripts/generate_cultural_pairs.py

# 2. Validate glossary
python scripts/build_glossary.py

# 3. Prepare the current MT dataset snapshot
python scripts/prepare_data.py --input_dir data/raw --output_dir data/final_v3
```

## Training

```bash
# Run QLoRA fine-tuning
bash training/run_train.sh

# Or with custom params
python scripts/train_qlora.py --model_name Qwen/Qwen2.5-3B-Instruct --num_train_epochs 5

# Resumable daily training from Hugging Face
python scripts/train_daily_from_hf.py \
    --dataset-repo NorthernTribe-Research/maasai-translation-corpus \
    --model-repo NorthernTribe-Research/maasai-en-mt \
    --max-steps 800 \
    --save-steps 100

# Kaggle retry wrapper
KAGGLE_CONFIG_DIR="$PWD" .venv/bin/python scripts/run_kaggle_training.py \
    --max-attempts 5 \
    --embed-local-hf-token
```

## Evaluation

```bash
python scripts/evaluate_mt.py \
    --model_dir outputs/maasai-en-mt-qlora \
    --test_file data/final_v3/test.jsonl \
    --glossary_file data/glossary/maasai_glossary.json
```

## Inference

```bash
# Single translation
python scripts/infer_translate.py --text "Hello, how are you?" --direction en_to_mas

# Interactive mode
python scripts/infer_translate.py --model_dir outputs/maasai-en-mt-qlora

# Speech transcription
python scripts/infer_asr.py --audio_file path/to/maasai_audio.wav
```

## Hugging Face Space Deployment

Recommended path:

```bash
# Preview the Space publish plan
python scripts/publish_to_hf.py --skip-dataset --skip-model

# Publish only the Space bundle
python scripts/push_space_to_hf.py
```

The maintained Space bundle includes:

- `space/app.py`
- `space/style.css`
- `space/README.md`
- `space/examples/*`
- bundled glossary data from `data/glossary/maasai_glossary.json`

Relevant runtime environment variables:

- `TRANSLATION_MODEL_ID`
- `ASR_MODEL_ID`

Live health check:

```bash
# Probe the hosted Space and Gradio API directly
.venv/bin/python scripts/check_space_health.py

# JSON output for automation / monitoring
.venv/bin/python scripts/check_space_health.py --json
```

Exit codes:

- `0` = healthy
- `2` = reachable but degraded / transitional
- `1` = down or probe failure

Do not use `scripts/simple_push_space.py` for the current Space path; it is a legacy helper and does not match the maintained publisher bundle.

## Hugging Face Model/Dataset Publishing

```bash
# Preview the full publish plan
python scripts/publish_to_hf.py

# Publish dataset + Space + model/scaffold
python scripts/publish_to_hf.py --execute --create-model-repo

# Publish only the dataset
python scripts/push_dataset_to_hf.py

# Publish only the model/scaffold
python scripts/push_model_to_hf.py
```

If `outputs/maasai-en-mt-qlora/` still contains placeholder artifacts, the publisher creates the model repo scaffold and metadata without pretending mock weights are a finished checkpoint.
