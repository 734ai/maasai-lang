# EXECUTION CHECKLISTS - Maasai Language Showcase v1.0

**Created:** March 26, 2026  
**Project Phase:** Steps 1-5 (Dataset → Publication)

---

## PHASE 1: DATASET EXPANSION (Days 1-3)

### Phase 1.1: Expand Cultural Pairs (+500 pairs)

**Prerequisites:**
- [ ] Backup original `scripts/generate_cultural_pairs.py`
- [ ] Review maasai_glossary.json for term consistency
- [ ] Verify current pair count: `wc -l data/raw/cultural_pairs.jsonl`

**Execution:**

#### 1.1a: Philosophy & Spirituality Enhancement (+150 pairs)
Domain focus: Proverbs, age-set philosophy, elder wisdom, spiritual concepts
```bash
# Add to CULTURAL_PAIRS array in generate_cultural_pairs.py
# Lines: ~120 new tuples covering:
# - Proverbs about patience, wisdom, courage (30 pairs)
# - Age-set transitions & manyatta philosophy (40 pairs)
# - Enkai belief system deeper exploration (40 pairs)
# - Sacred concepts (osotua, enkanyit advanced) (40 pairs)
```

**Checklist:**
- [ ] Add 30 proverb pairs (e.g., "Wisdom...", "One stick...")
- [ ] Add 40 age-set/warrior philosophy pairs
- [ ] Add 40 theological/Enkai pairs
- [ ] Add 40 advanced cultural concepts
- [ ] Validate: No duplicates within section
- [ ] Validate: All Maasai spellings consistent with glossary

#### 1.1b: Ceremonies & Rituals Enhancement (+150 pairs)
Domain focus: Detailed e-unoto, emuratta, enkipaata, blessings, ritual terms
- [ ] Add 50 e-unoto (warrior→elder transition) pairs
- [ ] Add 50 emuratta (circumcision) pairs
- [ ] Add 30 enkipaata (pre-circumcision) pairs
- [ ] Add 20 blessing & ritual vocabulary pairs
- [ ] Validate: Preserve ceremonial term accuracy

#### 1.1c: Governance & Justice Enhancement (+100 pairs)
Domain focus: Council, laws, tradition, elder roles, conflict resolution
- [ ] Add 40 council (enkiama, olaiguenani) pairs
- [ ] Add 30 law & tradition (oloshon, ilmororrok) pairs
- [ ] Add 20 age-set roles & governance pairs
- [ ] Add 10 justice & conflict resolution pairs

#### 1.1d: Education & Knowledge Enhancement (+100 pairs)
Domain focus: School subjects, learning, intellectual traditions, knowledge transfer
- [ ] Add 40 school subjects in Maasai
- [ ] Add 30 learning & instruction phrases
- [ ] Add 20 intellectual & knowledge terms
- [ ] Add 10 traditional knowledge transmission pairs

#### 1.1e: Validation & Output
```bash
# Run the expanded script
python scripts/generate_cultural_pairs.py

# Verify output
wc -l data/raw/cultural_pairs.jsonl  # Should be ~1000 (500 pairs × 2)
head -5 data/raw/cultural_pairs.jsonl | jq '.'
tail -5 data/raw/cultural_pairs.jsonl | jq '.'
```

**Success Criteria:**
- [ ] Output file created: `data/raw/cultural_pairs.jsonl`
- [ ] Line count: ~1,000 records (500 bidirectional pairs)
- [ ] All records have `source_name: "cultural_manual"` and `quality_score: 1.0`
- [ ] Domains well-distributed across philosophy/ceremony/governance/education
- [ ] No errors in JSON format

---

### Phase 1.2: Generate Synthetic Extensions (+600 pairs)

**Prerequisites:**
- [ ] Create new file: `scripts/generate_synthetic_extensions.py`
- [ ] Review existing synthetic methodology
- [ ] Have glossary terms loaded for vocabulary

**Execution:**

#### 1.2a: Create Synthetic Extensions Script
Template structure:
```python
# scripts/generate_synthetic_extensions.py
HEALTH_VOCAB = {
    "diseases": ["fever", "cough", "wound", ...],
    "treatments": ["medicine", "herb", "healing", ...],
    "anatomy": ["head", "heart", "blood", ...],
}

NUMBERS_VOCAB = {
    "cardinal": ["one", "two", "three", ...],
    "ordinal": ["first", "second", ...],
    "operations": ["add", "share", ...],
}

KINSHIP_VOCAB = {
    "relations": ["cousin", "nephew", "aunt", ...],
    "age_groups": ["baby", "child", "young", "elder"],
}

# Generate permutations: [subject] [action] [object]
# Output: data/raw/synthetic_extended.jsonl
```

#### 1.2b: Health & Medicine Vocabulary (+200 pairs)
- [ ] 50 disease/condition pairs
- [ ] 50 treatment/remedy pairs
- [ ] 50 anatomy/body part pairs
- [ ] 50 health action/instruction pairs

#### 1.2c: Numbers & Time (+200 pairs)
- [ ] 60 cardinal numbers + counting phrases
- [ ] 50 time references (days, seasons, hours)
- [ ] 50 date/calendar references
- [ ] 40 mathematical/quantity operations

#### 1.2d: Kinship & Family (+200 pairs)
- [ ] 70 family relation pairs (mother, uncle, nephew, etc.)
- [ ] 60 age-group & generational terms
- [ ] 50 family action pairs (marry, adopt, inherit)
- [ ] 20 kinship obligation & tradition pairs

#### 1.2e: Validation & Output
```bash
# Run synthetic extensions
python scripts/generate_synthetic_extensions.py

# Verify
wc -l data/raw/synthetic_extended.jsonl  # Should be ~600
# Check domain distribution
jq '.domain' data/raw/synthetic_extended.jsonl | sort | uniq -c
```

**Success Criteria:**
- [ ] File created: `data/raw/synthetic_extended.jsonl`
- [ ] Line count: ≥ 600 records
- [ ] Domains: health, numbers, kinship well-represented
- [ ] All records have `quality_score: 0.9` (synthetic)
- [ ] JSON format valid

---

### Phase 1.3: Merge & Deduplicate Data

**Execution:**

#### 1.3a: Create Merge Script
```bash
# Script: merge_and_deduplicate.py (inline execution)

python << 'EOF'
import json
from pathlib import Path
from collections import defaultdict

print("Step 1: Loading all raw data files...")
raw_dir = Path('data/raw')
all_pairs = []
source_tracking = defaultdict(list)

for f in sorted(raw_dir.glob('*.jsonl')):
    if f.name.startswith('.'):
        continue
    print(f"  Loading {f.name}...")
    count = 0
    with open(f) as fh:
        for line in fh:
            pair = json.loads(line)
            all_pairs.append(pair)
            key = (pair.get('source_text'), pair.get('target_text'))
            source_tracking[key].append(pair.get('source_name', 'unknown'))
            count += 1
    print(f"    → {count} records")

print(f"\nTotal loaded: {len(all_pairs)} records")

# Step 2: Deduplicate (keep highest quality per unique pair)
print("\nStep 2: Deduplicating...")
unique_pairs = {}
duplicates_removed = 0

for pair in all_pairs:
    key = (pair.get('source_text'), pair.get('target_text'))
    if key not in unique_pairs:
        unique_pairs[key] = pair
    else:
        # Keep higher quality version
        existing_quality = unique_pairs[key].get('quality_score', 0)
        new_quality = pair.get('quality_score', 0)
        if new_quality > existing_quality:
            unique_pairs[key] = pair
        duplicates_removed += 1

print(f"  Duplicates removed: {duplicates_removed}")
print(f"  Unique pairs: {len(unique_pairs)}")

# Step 3: Validation checks
print("\nStep 3: Validation...")
null_rows = 0
invalid_langs = 0

for pair in unique_pairs.values():
    if not pair.get('source_text') or not pair.get('target_text'):
        null_rows += 1
    langs = {pair.get('source_lang'), pair.get('target_lang')}
    if not langs.issubset({'en', 'mas'}):
        invalid_langs += 1

print(f"  Null text rows: {null_rows}")
print(f"  Invalid language pairs: {invalid_langs}")

# Step 4: Quality distribution
from collections import Counter
quality_scores = Counter(
    round(pair.get('quality_score', 0), 1) 
    for pair in unique_pairs.values()
)
print(f"\n  Quality score distribution:")
for score in sorted(quality_scores.keys(), reverse=True):
    print(f"    {score}: {quality_scores[score]} pairs")

# Step 5: Domain distribution
domain_dist = Counter(pair.get('domain', 'unknown') for pair in unique_pairs.values())
print(f"\n  Domain distribution (top 10):")
for domain, count in domain_dist.most_common(10):
    pct = 100 * count / len(unique_pairs)
    print(f"    {domain}: {count} ({pct:.1f}%)")

# Step 6: Source distribution
source_dist = Counter(pair.get('source_name', 'unknown') for pair in unique_pairs.values())
print(f"\n  Source distribution:")
for source, count in source_dist.most_common():
    pct = 100 * count / len(unique_pairs)
    print(f"    {source}: {count} ({pct:.1f}%)")

# Step 7: Write merged file
output_path = Path('data/raw/merged_all.jsonl')
print(f"\nStep 4: Writing merged file to {output_path}...")
with open(output_path, 'w', encoding='utf-8') as fh:
    for pair in unique_pairs.values():
        fh.write(json.dumps(pair, ensure_ascii=False) + '\n')

print(f"  ✓ Wrote {len(unique_pairs)} unique pairs to merged_all.jsonl")
EOF
```

#### 1.3b: Execute Merge
```bash
# Run the inline script (see above)
# Observe: Line counts, duplicates, quality distribution, domain breakdown
```

**Validation Checks:**
- [ ] `merged_all.jsonl` created
- [ ] Total unique pairs: ≥ 4,500
- [ ] Quality scores ≥ 0.7 (majority)
- [ ] No NULL source_text or target_text
- [ ] Domain distribution printed
- [ ] Source distribution shows: synthetic (50%), cultural_manual (30%), extended_cultural (15%), synthetic_extended (5%)

---

### Phase 1.4: Re-run Data Preparation Pipeline

**Execution:**

#### 1.4a: Run prepare_data.py
```bash
python scripts/prepare_data.py \
  --input_dir data/raw \
  --output_dir data/final_v3 \
  --test_size 0.10 \
  --valid_size 0.10 \
  --seed 42 \
  --min_quality_score 0.7

# Expected output:
# - data/final_v3/train.jsonl (7,991 pairs)
# - data/final_v3/valid.jsonl (707 pairs)
# - data/final_v3/test.jsonl (708 pairs)
```

#### 1.4b: Generate Statistics
```bash
python << 'EOF'
import json
from pathlib import Path
from collections import Counter

summary = {"total_rows": 0, "splits": {}, "lang_pairs": {}, "top_domains": {}, "top_sources": {}}

for split in ['train', 'valid', 'test']:
    f = Path(f'data/final_v3/{split}.jsonl')
    if not f.exists():
        continue
    
    count = 0
    domains = Counter()
    sources = Counter()
    langs = Counter()
    
    with open(f) as fh:
        for line in fh:
            pair = json.loads(line)
            count += 1
            domains[pair.get('domain', 'unknown')] += 1
            sources[pair.get('source_name', 'unknown')] += 1
            lang_key = f"{pair['source_lang']}->{pair['target_lang']}"
            langs[lang_key] += 1
    
    summary['splits'][split] = count
    summary['total_rows'] += count
    for k, v in langs.items():
        summary['lang_pairs'][k] = summary['lang_pairs'].get(k, 0) + v

# Aggregate top domains and sources
all_domains = Counter()
all_sources = Counter()
for split in ['train', 'valid', 'test']:
    f = Path(f'data/final_v3/{split}.jsonl')
    if f.exists():
        with open(f) as fh:
            for line in fh:
                pair = json.loads(line)
                all_domains[pair.get('domain', 'unknown')] += 1
                all_sources[pair.get('source_name', 'unknown')] += 1

summary['top_domains'] = dict(all_domains.most_common(10))
summary['top_source_names'] = dict(all_sources.most_common(10))

# Write summary
with open('data/final_v3/summary.json', 'w') as fh:
    json.dump(summary, fh, indent=2)

# Print
print("=" * 60)
print("DATA PREPARATION COMPLETE")
print("=" * 60)
print(f"\nTotal rows: {summary['total_rows']}")
print(f"\nSplits:")
for split, count in summary['splits'].items():
    print(f"  {split}: {count}")
print(f"\nLanguage pairs:")
for pair, count in summary['lang_pairs'].items():
    print(f"  {pair}: {count}")
print(f"\nTop domains:")
for domain, count in summary['top_domains'].items():
    pct = 100 * count / summary['total_rows']
    print(f"  {domain}: {count} ({pct:.1f}%)")
print(f"\nTop sources:")
for source, count in summary['top_source_names'].items():
    pct = 100 * count / summary['total_rows']
    print(f"  {source}: {count} ({pct:.1f}%)")
print("\n✓ Summary written to: data/final_v3/summary.json")
EOF
```

**Validation Checks:**
- [ ] Train split: 7,991 pairs
- [ ] Valid split: 707 pairs
- [ ] Test split: 708 pairs
- [ ] Total: 9,910 pairs
- [ ] Domain distribution includes: philosophy, ceremony, governance (20%+)
- [ ] Source distribution improved (cultural > 30%)
- [ ] `summary.json` generated and readable

---

### Phase 1.5: Success Validation

**Final Checklist:**
- [ ] ✅ Total pairs ≥ 5,000
- [ ] ✅ Authentic cultural ≥ 30% (up from 12%)
- [ ] ✅ Philosophy/culture/ceremony/governance ≥ 20% (up from 8%)
- [ ] ✅ Zero exact duplicates
- [ ] ✅ All pairs quality_score ≥ 0.7
- [ ] ✅ `summary.json` complete with metrics
- [ ] ✅ No errors during execution
- [ ] ✅ Session memory updated with results

**Metrics to Record:**
```
Phase 1 Complete Summary:
- Original pairs: 3,010
- New pairs added: ~2,000
- Final merged: ~4,500+ unique
- After split: ~4,100 train / 260 valid / 260 test
- Cultural ratio: 12% → 35%
- Source distribution: Synthetic 50%, Cultural 35%, Extended 15%
```

---

## PHASE 2: MODEL TRAINING & EVALUATION (Days 4-6)

**Depends On:** Phase 1 complete

### Phase 2.1: Pre-Training Setup
- [ ] Verify `data/final_v3/*.jsonl` exist and are valid
- [ ] Check GPU availability: `nvidia-smi`
- [ ] Verify `transformers`, `peft`, `torch` installed
- [ ] Review `training/run_train.sh` configuration

### Phase 2.2: Launch Training
```bash
bash training/run_train.sh
# Monitor: outputs/maasai-en-mt-qlora/
```
- [ ] No OOM errors
- [ ] Loss curves decreasing
- [ ] Checkpoints saved every 200 steps
- [ ] Training completes without interruption

### Phase 2.3: Post-Training Merge
```bash
python -c "
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
# Merge LoRA weights
# Save to outputs/maasai-en-mt-merged
"
```
- [ ] Model merged and saved
- [ ] Tokenizer copied to output

### Phase 2.4: Evaluation
```bash
python scripts/evaluate_mt.py \
  --model_dir outputs/maasai-en-mt-merged \
  --test_file data/final_v3/test.jsonl \
  --glossary_file data/glossary/maasai_glossary.json \
  --output_file data/eval/metrics_v1.json
```
- [ ] Metrics computed (BLEU, chrF++, glossary accuracy)
- [ ] Results written to `metrics_v1.json`
- [ ] BLEU ≥ 15, chrF++ ≥ 62

### Phase 2.5: Error Analysis
- [ ] Sample 50 test predictions
- [ ] Manually review for hallucinations, orthography, term flattening
- [ ] Document findings

---

## PHASE 3: SPACE APPLICATION POLISH (Days 7-8)

**Depends On:** Phase 2 complete

### Phase 3.1: Glossary Integration
- [ ] Implement glossary search function
- [ ] Add domain filtering
- [ ] Test end-to-end search

### Phase 3.2: Speech Pipeline
- [ ] Implement Maasai→text (ASR)
- [ ] Implement text→English (translation)
- [ ] Chain pipeline: Maasai audio → English

### Phase 3.3: Error Handling & UI
- [ ] Add user warnings
- [ ] Implement length validation
- [ ] Add loading indicators
- [ ] Test space locally

### Phase 3.4: CSS & Styling
- [ ] Apply beadwork color theme
- [ ] Verify responsive design
- [ ] Test on mobile + desktop

---

## PHASE 4: DOCUMENTATION (Days 9-10)

**Checklist:**
- [ ] Model card complete (v1.0)
- [ ] Dataset card updated (v1.0)
- [ ] Evaluation report written
- [ ] Ethics statement created
- [ ] README updated with links

---

## PHASE 5: PUBLICATION (Days 11-12)

**Checklist:**
- [ ] Model uploaded to HF
- [ ] Dataset updated to HF
- [ ] Space deployed & tested
- [ ] README links verified
- [ ] All repos public

---

**End of Execution Checklists**
