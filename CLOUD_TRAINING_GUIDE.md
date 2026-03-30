# Cloud Training Guide: Maasai Model on Free GPU Resources

**Status:** Local system is CPU-only (Intel i5). Model training (3 epochs on 8.4K pairs) requires GPU acceleration.

## Option 1: Google Colab (Recommended - Free)

### Step 1: Upload Dataset to HuggingFace Hub
```bash
cd /home/ntr/Documents/huggingface/maasai-lang
huggingface-cli login  # Enter token from huggingface-api-key.json
git config --global user.email "research@northerntribe.com"
git config --global user.name "NorthernTribe-Research"

# Clone the dataset repo
git clone https://huggingface.co/datasets/NorthernTribe-Research/maasai-translation-corpus
cd maasai-translation-corpus

# Copy training data
cp ../data/final_v3/* .
cp ../docs/DATASET_README_V2.md README.md

# Push to HF
git add .
git commit -m "Add current translation corpus snapshot (9,910 pairs, 8434/738/738 splits)"
git push
```

### Step 2: Create Colab Training Notebook

Paste this into a new Colab cell:
```python
# Colab: Mount HF data + run training
!git clone https://huggingface.co/datasets/NorthernTribe-Research/maasai-translation-corpus dataset_repo
!cd dataset_repo && git pull

!git clone https://github.com/NorthernTribe-Research/maasai-lang training_repo
%cd training_repo

# Install dependencies
!pip install -q torch transformers peft accelerate bitsandbytes datasets

# Login to HF
from huggingface_hub import notebook_login
notebook_login()

# Run training
!bash training/run_train.sh --output_dir outputs/maasai-en-mt-qlora-colab

# Export GGUF (if llama.cpp available)
# !python scripts/export_gguf.py outputs/maasai-en-mt-qlora-colab outputs/gguf/

# Push model to HF
!python scripts/push_model_to_hf.py outputs/maasai-en-mt-qlora-colab
```

## Option 2: Lambda Labs (Paid, ~$0.33/hr GPU)

```bash
# 1. Create Lambda Account: https://lambdalabs.com
# 2. Rent GPU instance (RTX A40, A100, etc.)
# 3. SSH into instance & clone repo:

git clone https://huggingface.co/NorthernTribe-Research/maasai-lang
cd maasai-lang

# 4. Set up environment & install deps
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 5. Clone dataset
git clone https://huggingface.co/datasets/NorthernTribe-Research/maasai-translation-corpus

# 6. Run training
huggingface-cli login
bash training/run_train.sh

# 7. Export & push
python scripts/export_gguf.py outputs/maasai-en-mt-qlora outputs/gguf/
python scripts/push_model_to_hf.py outputs/maasai-en-mt-qlora
```

## Option 3: Cloud GPU (Kaggle, Paperspace, Modal)

### Kaggle (Free, limited GPU hours)
```bash
# Go to: https://www.kaggle.com/code/
# Create new notebook
# Enable GPU in settings (Settings > Accelerator > GPU)
# Upload this as a cell and run:

!git clone https://huggingface.co/NorthernTribe-Research/maasai-lang
%cd maasai-lang
!pip install -q -r requirements.txt
!bash training/run_train.sh
```

## Training Expected Duration

| Resource | Type | Duration | Cost |
|----------|------|----------|------|
| Local (Intel i5) | CPU | 7-10 days | $0 |
| Google Colab | TPU/GPU | 4-6 hrs | Free (limited usage) |
| Lambda RTX A40 | GPU | 2-3 hrs | ~$1.00 |
| Lambda A100 | GPU | 45-60 min | ~$2.00 |

## Post-Training Steps

1. **Export GGUF**: Convert model to quantized format for llama.cpp
   ```bash
   python scripts/export_gguf.py outputs/maasai-en-mt-qlora outputs/gguf/
   ```

2. **Push Model to HF**:
   ```bash
   python scripts/push_model_to_hf.py outputs/maasai-en-mt-qlora
   ```

3. **Update Space**:
   ```bash
   git clone https://huggingface.co/spaces/NorthernTribe-Research/maasai-language-showcase space_repo
   cp outputs/gguf/model.Q4_K_M.gguf space_repo/model.gguf
   cd space_repo && git add . && git commit -m "Update model to v2.0" && git push
   ```

## Dataset Already Prepared

✅ Training data ready at `data/final_v3/`:
- 8,434 training pairs (85.1%)
- 738 validation pairs (7.4%)
- 738 test pairs (7.4%)
- 9,910 total unique pairs (8,444 gold / 1,466 silver in current metadata)

Ready for immediate GPU training!
