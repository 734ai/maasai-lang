# Maasai Project v2.0 - Deployment Status

**Deployment Date:** March 26, 2026  
**Status:** ✅ Dataset & Space Live | ⏳ Model Training Pending

---

> Note (March 28, 2026): `data/final_v3` later grew after the Hollis + ASJP supplement merge. The current local model artifacts are still placeholders, not a completed training run.

## ✅ Completed

### 1. Dataset Published to HuggingFace
- **Repository:** https://huggingface.co/datasets/NorthernTribe-Research/maasai-translation-corpus
- **Current local size:** 9,910 unique pairs
- **Splits:**
  - Train: 8,434 pairs (85.1%)
  - Valid: 738 pairs (7.4%)
  - Test: 738 pairs (7.4%)
- **Quality labels in current metadata:** 8,444 gold / 1,466 silver
- **Domains:** Bible-derived (85.2%) + curated/open-source supplement (14.8%)
- **Files Uploaded:** train.jsonl, valid.jsonl, test.jsonl, README.md, datasetinfo.json
- **Commit:** https://huggingface.co/datasets/NorthernTribe-Research/maasai-translation-corpus/commit/50a5933a6ada0c41aa744d2baf90ba751c89c817

### 2. Space Updated on HuggingFace
- **Repository:** https://huggingface.co/spaces/NorthernTribe-Research/maasai-language-showcase
- **Features:**
  - ✅ English → Maasai translation interface
  - ✅ Maasai → English translation interface
  - ✅ Engram glossary layer integration (term protection)
  - ✅ Glossary statistics display
  - ✅ Project information & about tab
  - ✅ Auto-deployment on push
- **Updated Files:**
  - app.py (Gradio interface with Engram)
  - requirements.txt (dependencies)
  - engram_glossary_layer.py (glossary retrieval)
  - infer_llama_cpp_optimized.py (optimized inference)

### 3. Engram Glossary Layer Created
- **Purpose:** Static memory-based retrieval for O(1) term protection
- **Features:**
  - Exact matching (hash-based lookup)
  - Fuzzy matching (fallback for variations)
  - Batch processing support
  - Prompt augmentation with glossary hints
  - Statistics tracking (hits, fallthrough)
- **Status:** ✅ Integrated into Space and Inference Pipeline
- **File:** scripts/engram_glossary_layer.py

### 4. Optimized Inference Scripts Ready
- **Script:** scripts/infer_llama_cpp_optimized.py
- **Features:**
  - Batch processing (4-32 sequences)
  - Temperature tuning (0.3-0.5 for translation)
  - Quantization support (Q4_K_M, Q5_K_M, Q6_K)
  - GPU/CPU layer offloading
  - Interactive mode with metrics
- **Expected Performance:**
  - Latency: 150-300ms (CPU), 30-50ms (GPU)
  - Throughput: 4-8 tx/s (CPU), 20+ (GPU)

### 5. Documentation Enhanced
- **SKILLS.md:** Updated with Engram layer (Section 2.5)
- **CLOUD_TRAINING_GUIDE.md:** Complete training setup instructions
- **docs/DATASET_README_V2.md:** Data-neutral framing (de-emphasized Bible source)
- **Deployment scripts:** push_dataset_to_hf.py, push_space_to_hf.py, deploy_to_hf.py

---

## ⏳ Next Steps: Model Training

### Training Requirements
- **Dataset:** ✅ Ready (8,434 pairs at `data/final_v3/train.jsonl`)
- **GPU:** ❌ Not available locally (Intel i5 CPU-only system)
- **Solution:** Use cloud GPU provider

### Option 1: Google Colab (Recommended - Free)

```bash
# In Colab cell:
!git clone https://huggingface.co/NorthernTribe-Research/maasai-lang repo
cd repo
!pip install -q torch transformers peft accelerate

# Login to HF
from huggingface_hub import notebook_login
notebook_login()

# Train
!bash training/run_train.sh
```

**Expected Duration:** 4-6 hours on Colab GPU

### Option 2: Lambda Labs (Paid, ~$0.33/hr)

```bash
# On Lambda GPU instance:
git clone https://huggingface.co/NorthernTribe-Research/maasai-lang
cd maasai-lang
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

huggingface-cli login  # Use token from huggingface-api-key.json
bash training/run_train.sh
```

**Expected Duration:** 2-3 hours on RTX A40

### Option 3: Kaggle (Free, limited GPU hours)

- Go to https://www.kaggle.com/code/
- Create notebook, enable GPU backend
- Clone repo and run training script

### Training Configuration
- **Model:** Qwen/Qwen2.5-3B-Instruct (3B parameters, instruction-tuned)
- **Method:** QLoRA (LoRA r=16, alpha=32, lr=2e-4)
- **Epochs:** 3
- **Batch Size:** 4 per device + 8 gradient accumulation
- **Learning Rate:** 2e-4
- **Max Tokens:** 512
- **Warmup:** 3%

**Training Script:** `bash training/run_train.sh`  
**Output Directory:** `outputs/maasai-en-mt-qlora/`

---

## Post-Training Pipeline

### 1. GGUF Export
```bash
python scripts/export_gguf.py outputs/maasai-en-mt-qlora outputs/gguf/
```
Creates quantized models (Q4_K_M, Q5_K_M, Q6_K) for llama.cpp inference

### 2. Push Model to HuggingFace
```bash
python scripts/push_model_to_hf.py
```
- Uploads checkpoint to https://huggingface.co/NorthernTribe-Research/maasai-translation-model
- Creates comprehensive model card
- Links to dataset & Space

### 3. Update Space with Model
```bash
# Clone space, copy model files, push
git clone https://huggingface.co/spaces/NorthernTribe-Research/maasai-language-showcase
cp outputs/gguf/model.Q4_K_M.gguf space_repo/
cd space_repo && git add . && git commit -m "Add trained model" && git push
```

### 4. Publish to HF Hub
- ✅ Dataset: Published
- ⏳ Model: Ready after training
- ✅ Space: Live (awaiting model integration)

---

## Project Statistics

### Dataset Composition
- **Total Pairs:** 9,910 (deduplicated)
- **Bidirectional Balance:** 50/50 (en→mas / mas→en)
- **Source Mix:**
  - Bible-derived: 8,444 pairs (85.2%)
  - Curated/manual/open-source supplement: 1,466 pairs (14.8%)
- **Quality Distribution:**
  - Gold: 8,444 pairs
  - Silver: 1,466 pairs

### Model Specifications
- **Base:** Qwen/Qwen2.5-3B-Instruct (3B parameters)
- **Architecture:** Transformer (instruction-tuned variant)
- **Adapter:** LoRA (Low-Rank Adaptation)
  - Rank: 16
  - Alpha: 32
  - Dropout: 0.05
- **Quantization:** Q4_K_M (2.2GB RAM for inference)

### Infrastructure
- **Local Development:** Intel i5-8365U (CPU-only)
- **Training:** GPU required (Colab/Lambda/Kaggle)
- **Inference:** CPU/GPU flexible (llama.cpp optimized)
- **Deployment:** HuggingFace Hub (models + datasets + spaces)

---

## Verification Checklist

- ✅ Dataset published to HF Hub (current local snapshot: 9,910 pairs, train/valid/test splits)
- ✅ Space updated with Engram glossary layer
- ✅ Engram layer integrated into inference pipeline
- ✅ Optimized inference scripts ready (llama.cpp)
- ✅ Training configuration prepared (QLoRA)
- ✅ Documentation complete (SKILLS.md, CLOUD_TRAINING_GUIDE.md)
- ✅ HuggingFace credentials configured (no more prompts)
- ⏳ Model training (pending GPU access)
- ⏳ GGUF export (post-training)
- ⏳ Model publishing to HF Hub (post-training)

---

## Quick Links

| Component | Link | Status |
|-----------|------|--------|
| Dataset | https://huggingface.co/datasets/NorthernTribe-Research/maasai-translation-corpus | ✅ Live |
| Space | https://huggingface.co/spaces/NorthernTribe-Research/maasai-language-showcase | ✅ Live |
| Model (expected) | https://huggingface.co/NorthernTribe-Research/maasai-translation-model | ⏳ Training |
| GitHub | https://github.com/NorthernTribe-Research/maasai-lang | ✅ Source |

---

## Next Immediate Action

⏬ **Train the model on GPU** using one of the cloud options above (Colab recommended)

This will:
1. Create `outputs/maasai-en-mt-qlora/` checkpoint
2. Enable GGUF export for optimized inference
3. Allow model publication to HF Hub
4. Integrate trained model into Space for live translation

**Estimated time to full production:** 6-8 hours (training + post-processing)
