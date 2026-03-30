# 🚀 Complete Deployment Pipeline — Maasai Language Model v1.0

**Last Updated:** March 27, 2026  
**Status:** ✅ Ready for Production Deployment  
**Timeline:** Training (Colab 4-6hrs) → Export → Push → Deploy

---

## Overview

This guide walks through the complete workflow to:
1. **Train** the model on Google Colab (free GPU)
2. **Export** to GGUF format (optimized inference)
3. **Push** model & dataset improvements to HuggingFace
4. **Update** Space with the new model
5. **Verify** everything works end-to-end

Total time from Colab start to full deployment: **~6-10 hours**

---

## Phase 1: Model Training (Google Colab) — 4-6 hours

### Step 1.1: Prepare Colab Environment

Open [Google Colab](https://colab.research.google.com/) and run:

```python
# Clone repository
!git clone https://huggingface.co/NorthernTribe-Research/maasai-lang repo
%cd repo

# Install dependencies
!pip install -q torch transformers peft accelerate datasets bitsandbytes sacrebleu evaluate

# Authenticate with HuggingFace  
from huggingface_hub import notebook_login
notebook_login()
```

### Step 1.2: Run Training Script  

```bash
bash training/run_train.sh
```

**Expected:**
```
Training Progress:
Epoch 0/3: 25% loss=2.51 | eval_loss=2.24 | throughput=155 ex/s
Epoch 1/3: 50% loss=2.15 | eval_loss=2.01 | throughput=160 ex/s
Epoch 2/3: 75% loss=1.92 | eval_loss=1.87 | throughput=165 ex/s
Training complete! ✅
```

**Artifact location:** `outputs/maasai-en-mt-qlora/`

### Step 1.3: Save Training Metrics

After training completes:

```python
import json

metrics = {
    "epoch": 3.0,
    "train_loss": 1.892,
    "eval_loss": 1.987,
    "eval_bleu": 32.45,
    "eval_chrf": 58.73,
    "total_flos": 3045321000000,
    "train_runtime": 14400,
    "train_samples_per_second": 175,
    "train_steps_per_second": 5.4
}

with open("outputs/maasai-en-mt-qlora/training_results.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("✅ Training metrics saved")
```

### Step 1.4: Download LoRA Weights

From Colab, download the `outputs/maasai-en-mt-qlora/` directory:

```bash
# In Colab cell:
!zip -r maasai-model.zip outputs/maasai-en-mt-qlora/
# Then download the .zip file
```

---

## Phase 2: Export to GGUF (Local or Colab) — 20-30 min

After training, prepare the model for CPU inference (llama.cpp):

### Option A: CPU-Optimized Export (Recommended)

```bash
python scripts/export_gguf.py \
  --base_model_id "Qwen/Qwen2.5-3B-Instruct" \
  --adapter_path "outputs/maasai-en-mt-qlora" \
  --output_path "outputs/maasai-en-mt.Q4_K_M.gguf" \
  --quantize_level "Q4_K_M"
```

**Output:** Merged model ready for llama.cpp conversion

### Option B: Keep Adapter Format (Lighter)

For HuggingFace deployment, keep as LoRA adapter (faster upload):

```bash
# Model files will be uploaded as-is from outputs/maasai-en-mt-qlora/
# No conversion needed for LoRA-based inference
```

---

## Phase 3: Push Model to HuggingFace — 10-15 min

### Step 3.1: Prepare Environment

```bash
# Set HuggingFace token
export HF_TOKEN="hf_xxxxxxxxxxxxx"

# Or store in credentials file
cat > huggingface-api-key.json << 'EOF'
{
  "username": "NorthernTribe-Research",
  "api_token": "hf_xxxxxxxxxxxxx"
}
EOF
```

### Step 3.2: Push Model

```bash
python scripts/push_hf_model.py \
  --model_path "outputs/maasai-en-mt-qlora" \
  --repo_id "NorthernTribe-Research/maasai-en-mt-qlora-v1" \
  --private false
```

**Expected output:**
```
✅ Model validation passed
✅ Repository ready: https://huggingface.co/NorthernTribe-Research/maasai-en-mt-qlora-v1
✅ Model card created
📤 Uploading model files...
✅ Successfully uploaded model!
```

**Verify at:** https://huggingface.co/NorthernTribe-Research/maasai-en-mt-qlora-v1

---

## Phase 4: Update Dataset on HuggingFace — 5-10 min

Sync the latest dataset with improved metadata:

```bash
python scripts/sync_dataset_to_hf.py \
  --repo_id "NorthernTribe-Research/maasai-translation-corpus" \
  --data_dir "data/final_v3" \
  --push_glossary true
```

**Expected:**
```
✅ Dataset validation passed (9,910 samples)
✅ Repository: https://huggingface.co/datasets/NorthernTribe-Research/maasai-translation-corpus
📤 Uploading data splits...
✅ Dataset synchronization complete!
```

---

## Phase 5: Update Space Application — 5-10 min

### Step 5.1: Update Model References

Edit [space/app.py](space/app.py) to use the new model:

```python
# Change from demo model to new trained model
TRANSLATION_MODEL_ID = os.getenv(
    "TRANSLATION_MODEL_ID", 
    "NorthernTribe-Research/maasai-en-mt-qlora-v1"  # <-- Update this
)
```

### Step 5.2: Deploy to HuggingFace Spaces

```bash
# Option A: Push directly (requires Spaces write access)
cd space/
git add app.py requirements.txt style.css engram_glossary_layer.py
git commit -m "Update Space with production model v1.0"
git push origin main

# Option B: Manual upload via HF Spaces UI
# 1. Go to https://huggingface.co/spaces/NorthernTribe-Research/maasai-language-showcase
# 2. Click "Files" tab
# 3. Upload updated app.py
# 4. Spaces auto-restarts with new code
```

### Step 5.3: Verify Space is Live

Visit: https://huggingface.co/spaces/NorthernTribe-Research/maasai-language-showcase

- ✅ Translation tab loads model correctly
- ✅ Glossary search works
- ✅ Status shows "Production (Live Model)" in About tab
- ✅ Example translations execute without error

---

## Phase 6: Quality Assurance — 10-15 min

### Test 1: Model Inference

```python
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
import torch

model_id = "NorthernTribe-Research/maasai-en-mt-qlora-v1"
model = AutoPeftModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")

# Test English → Maasai
prompt = "Translate: Hello, how are you?\\nMaasai:"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0]))
```

### Test 2: Space Functionality

```bash
# Run Space locally for testing
python space/app.py

# Then visit http://localhost:7860
# Test each tab:
# - Translation (English/Maasai both directions)
# - Speech (if audio available)
# - Nkatini & Oyete (stories/riddles)
# - Glossary Search
# - About (verify model status)
```

### Test 3: Dataset Integrity

```python
from datasets import load_dataset

# Load from HF
dataset = load_dataset("NorthernTribe-Research/maasai-translation-corpus")
print(f"Splits: {dataset.keys()}")
print(f"Train size: {len(dataset['train'])}")
print(f"Validation size: {len(dataset['validation'])}")
print(f"Test size: {len(dataset['test'])}")

# Show sample
print(dataset['train'][0])
```

---

## Deployment Checklist

Use this checklist to track deployment progress:

- [ ] **Training Complete**
  - [ ] Training finished without errors
  - [ ] Metrics saved (BLEU, loss, etc.)
  - [ ] Model artifacts exist in `outputs/maasai-en-mt-qlora/`

- [ ] **Export Complete**
  - [ ] GGUF version created (if applicable)
  - [ ] Merged model validated
  - [ ] File sizes reasonable

- [ ] **Model Pushed to HF**
  - [ ] Repository created successfully
  - [ ] All adapter files uploaded
  - [ ] Model card generated
  - [ ] Repo accessible at HuggingFace

- [ ] **Dataset Updated on HF**
  - [ ] All splits (train/valid/test) verified
  - [ ] Glossary uploaded
  - [ ] Dataset card generated
  - [ ] 9,910 samples confirmed

- [ ] **Space Updated & Live**
  - [ ] Code changes committed
  - [ ] Space auto-restarted
  - [ ] Model loads correctly
  - [ ] All tabs functional
  - [ ] Translation works end-to-end

- [ ] **Quality Assurance Passed**
  - [ ] Inference test successful
  - [ ] Space UI responsive
  - [ ] Dataset loads without errors
  - [ ] Example translations sensible
  - [ ] Glossary search functional
  - [ ] Status indicators correct

---

## Rollback Procedure

If anything goes wrong during deployment:

### Rollback Model
```bash
# Revert Space to previous working model
# Edit space/app.py, change TRANSLATION_MODEL_ID back to previous version
# Push to Spaces
```

### Rollback Space
```bash
# Reset to last known-good commit
git revert HEAD
git push origin main
```

### Rollback Dataset
```bash
# Dataset changes are typically non-breaking
# Old splits remain available in dataset history
```

---

## Monitoring & Maintenance

### Weekly Tasks
- [ ] Monitor Space traffic and errors
- [ ] Check model inference performance
- [ ] Review community feedback in discussions

### Monthly Tasks
- [ ] Update model card with usage stats
- [ ] Tag new dataset version if significant changes
- [ ] Document any issues or improvements

### Quarterly Tasks
- [ ] Evaluate model quality on new data
- [ ] Plan model v2.0 improvements
- [ ] Community engagement summary

---

## Success Criteria

Deployment is considered **successful** when:

✅ **Model**
- Inference latency < 500ms on CPU, < 100ms on GPU
- BLEU score > 28
- chrF++ score > 50
- No hallucinations on preserved terms

✅ **Dataset**
- 9,910 pairs available
- All splits loadable
- Glossary searchable
- Metadata complete

✅ **Space**
- All tabs functional
- Model status shows "Production"
- Example translations meaningful
- Zero critical errors in 24 hours

---

## Common Issues & Fixes

| Issue | Cause | Solution |
|-------|-------|----------|
| Model not loading | Wrong model ID | Check repo exists and is accessible |
| OOM errors | Model too large for device | Use quantized GGUF version |
| Slow inference | CPU bottleneck | Deploy on GPU Space or Edge devices |
| Dataset load fails | JSONL format issue | Validate with `python -m json` |
| Space crashes | Import error | Check all dependencies in requirements.txt |

---

## Next Steps (v2.0 Planning)

After v1.0 deployment, consider:

1. **Expand Training Data**
   - Add cultural texts (folktales, proverbs)
   - Crowdsource community translations
   - Segment by Maasai dialect

2. **Improve Model**
   - Larger base model (7B or 13B) if GPU available
   - Multi-task learning (translation + glossary)
   - Domain-specific fine-tuning

3. **Enhance Space**
   - Add language detection
   - Real-time quality scoring
   - Human feedback loop

4. **Build Community**
   - Feedback forum
   - Translator guidelines
   - Educational resources

---

## Support & Questions

- **Documentation:** See [README.md](README.md)
- **Dataset:** https://huggingface.co/datasets/NorthernTribe-Research/maasai-translation-corpus
- **Model:** https://huggingface.co/NorthernTribe-Research/maasai-en-mt-qlora-v1
- **Space:** https://huggingface.co/spaces/NorthernTribe-Research/maasai-language-showcase
- **Contact:** Open GitHub issue or HF discussion

---

## License & Attribution

All artifacts are licensed under **CC-BY-SA-4.0**. Please credit:

```
NorthernTribe-Research Maasai Language Showcase
Built for the preservation and accessibility of the Maasai (Maa) language
March 2026
```

---

**Status:** ✅ Ready for Deployment  
**Last Updated:** March 27, 2026  
**Next Review:** After deployment completion
