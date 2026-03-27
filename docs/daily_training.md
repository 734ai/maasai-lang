# Daily Training Control Plane

This project now supports a resumable Hugging Face training loop driven from either:

- Google Colab with secrets stored in the Colab key vault
- A GitHub Actions schedule running on a self-hosted GPU runner

## Colab Flow

1. Open Colab and store `HF_TOKEN` in the left sidebar secrets panel.
2. Optionally store `WANDB_API_KEY` there as well.
3. Clone this repo and install dependencies.
4. Run:

```bash
python scripts/train_daily_from_hf.py \
  --dataset-repo NorthernTribe-Research/maasai-translation-corpus \
  --model-repo NorthernTribe-Research/maasai-en-mt \
  --max-steps 800 \
  --save-steps 100 \
  --disconnect-colab
```

What happens:

- The script authenticates with `huggingface_hub.login(...)`
- Downloads `train.jsonl` and `valid.jsonl` from the dataset repo
- Pulls the latest `checkpoint-*` folders from the model repo when they exist
- Resumes training from the newest checkpoint
- Pushes checkpoints back to the same model repo
- Optionally disconnects the Colab runtime at the end

## GitHub Flow

The repo now includes [`.github/workflows/daily-train.yml`](/home/ntr/Documents/dev/maasai-lang/.github/workflows/daily-train.yml).

It is designed for a `self-hosted` GPU runner because GitHub-hosted runners do not provide the GPU resources needed for daily QLoRA training on Qwen 2.5 3B-class models.

Required GitHub secrets:

- `HF_TOKEN`
- `WANDB_API_KEY` if you want remote W&B logging

Recommended GitHub repo variables:

- `HF_DATASET_REPO`
- `HF_MODEL_REPO`
- `HF_BASE_MODEL`

The workflow falls back to the current NorthernTribe Hugging Face repos when those vars are not set, and defaults the base model to `Qwen/Qwen2.5-3B-Instruct`.

## Checkpoint Policy

- Frequent checkpoint upload is handled through `push_to_hub=True`
- The trainer uses `hub_strategy="checkpoint"`
- The model repo therefore acts as the durable resume point between daily sessions
- Each run also writes a `run_manifest.json` file so GitHub can retain a lightweight execution artifact even when the runner workspace is ephemeral

## Recommended Repo Target

When the project is moved to GitHub, use the `734ai` namespace as the main control-plane repository and keep Hugging Face as the storage and checkpoint backend.
