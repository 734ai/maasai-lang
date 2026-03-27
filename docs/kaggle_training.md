# Kaggle Training

This project can now train on Kaggle while still using Hugging Face as the system of record for:

- the dataset repo
- the resumable checkpoint history
- the target model repo

## What Kaggle Does

The Kaggle kernel does not ship a separate training stack.
It clones this GitHub repo at runtime and executes:

`scripts/train_daily_from_hf.py`

That means the Kaggle run:

- downloads `NorthernTribe-Research/maasai-translation-corpus` from HF
- restores the latest checkpoint from `NorthernTribe-Research/maasai-en-mt`
- trains with the current generation-aware mixture
- pushes checkpoints and final adapter artifacts back to HF

## Files Added

- `kaggle/train_and_push.py`
- `kaggle/requirements-kaggle.txt`
- `scripts/push_kaggle_kernel.py`

## Required Secrets

There are two different secret systems involved.

### Local machine

Use root-level `kaggle.json` for the Kaggle API on your machine.
This is used only to push/update the Kaggle kernel from local development.

### Kaggle runtime

Inside Kaggle itself, add this secret in the notebook UI:

- `HF_TOKEN`

Optional:

- `WANDB_API_KEY`

If you push the kernel with `--report-to wandb` and a local root-level `wandb-keys.json`
exists, the helper will embed that W&B key into the Kaggle kernel package.
That is local-only behavior and the file remains gitignored.

`HF_TOKEN` must have access to the base model if it is gated and must have permission to push to:

- `NorthernTribe-Research/maasai-en-mt`

## Local Push Flow

Install the Kaggle CLI into the project venv if needed:

```bash
.venv/bin/python -m pip install kaggle
```

Build the kernel package locally:

```bash
.venv/bin/python scripts/push_kaggle_kernel.py
```

Push the kernel to Kaggle:

```bash
KAGGLE_CONFIG_DIR="$PWD" .venv/bin/python scripts/push_kaggle_kernel.py --execute --status
```

## Default Kernel Target

By default the helper creates:

- slug: `maasai-daily-hf-training`
- title: `Maasai Daily HF Training`

The kernel is private by default and GPU + internet are enabled.

## Useful Overrides

Example:

```bash
.venv/bin/python scripts/push_kaggle_kernel.py \
  --kernel-slug maasai-hf-nightly \
  --title "Maasai HF Nightly" \
  --max-steps 1200 \
  --report-to wandb
```

## Runtime Notes

- Kaggle GPU sessions are time-limited, so the HF checkpoint resume path matters.
- The kernel uses the existing `--augment-with-generation-tasks` default.
- If Kaggle shows the run as failed immediately, the first thing to check is whether `HF_TOKEN` was added in Kaggle secrets.
