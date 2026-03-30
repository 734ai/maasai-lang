---
language:
  - en
  - mas
pretty_name: Maasai-English Translation Corpus
tags:
  - translation
  - maasai
  - low-resource
  - cultural-preservation
  - bilingual
  - machine-translation
license: cc-by-4.0
task_categories:
  - translation
size_categories:
  - 1K<n<10K
dataset_info:
  features:
    - name: source_text
      dtype: string
    - name: target_text
      dtype: string
    - name: source_lang
      dtype: string
    - name: target_lang
      dtype: string
    - name: domain
      dtype: string
    - name: source_name
      dtype: string
    - name: confidence
      dtype: float32
    - name: tier
      dtype: string
    - name: quality_score
      dtype: float32
    - name: notes
      dtype: string
---

# Maasai-English Translation Corpus (v2.0)

**Comprehensive Translation Dataset for Maasai Language Preservation**

## Overview

This dataset currently contains **9,910 English-Maasai translation pairs** in the local `data/final_v3` workspace snapshot. It features:

- **Bible-derived, curated, and open-source rows** (9,910 pairs)
- **Rich cultural coverage** (15+ domains) - Philosophy, ceremonies, governance, education, greetings, and more
- **Open-source supplement layer** - Public-domain Hollis proverb pairs, conservative public-domain Hinde vocabulary pairs, and CC BY 4.0 ASJP lexical pairs
- **Perfect 50/50 bidirectional balance** - Both en→mas and mas→en directions

## Key Features

### Data Composition
```
Total pairs: 9,910
├── Train split: 8,434 pairs (85.1%)
├── Valid split: 738 pairs (7.4%)
└── Test split:  738 pairs (7.4%)

Quality tiers:
├── Gold (dataset metadata label): 8,444 pairs (85.2%)
└── Silver (dataset metadata label): 1,466 pairs (14.8%)

Language coverage:
├── English → Maasai: 4,955 pairs
└── Maasai → English: 4,955 pairs

Top domains:
├── Bible-derived: 8,444 pairs (85.2%)
├── Lexicon: 584 pairs
├── Proverbs: 158 pairs
├── Philosophy: 100 pairs
├── Culture: 84 pairs  
├── Environment: 64 pairs
├── Education: 60 pairs
├── Ceremony: 58 pairs
├── Greetings: 52 pairs
└── Other (governance, livestock, health, daily life): ~366 pairs
```

### Quality Metrics
- **Confidence Score**: 0.92-0.98 in the current local snapshot
- **Source mix**: Bible-derived extraction, curated cultural rows, and open-source supplements
- **Deduplication**: Aggressive deduplication (9,910 unique pairs from 14,174+ raw sources)
- **Domain Coverage**: 15+ specialized domains for comprehensive fluency
- **Schema Note**: 680 rows currently omit `id`, and 750 rows omit `quality_assessment`

## Data Format

JSONL format. Each line is a complete translation pair:

```json
{
  "id": "bible-en2mas-00000",
  "source_text": "And there was evening, and there was morning- the first day.",
  "target_text": "Nerishu Enkai ewang'an aitung'uaa enaimin.",
  "source_lang": "en",
  "target_lang": "mas",
  "domain": "bible",
  "source_name": "bible_english_maasai",
  "confidence": 0.98,
  "tier": "gold",
  "quality_score": 0.98,
  "notes": "Bible-derived sentence chunk pair"
}
```

### Fields Explained
- `id`: Optional unique identifier for pair; 680 older `cultural_manual` rows currently omit it
- `source_text`: Text in source language
- `target_text`: Text in target language
- `source_lang`: Source language (en/mas)
- `target_lang`: Target language (mas/en)
- `domain`: Subject domain (bible, philosophy, culture, etc.)
- `source_name`: Data source identifier
- `synthetic`: Optional flag present on some generated rows
- `confidence`: Confidence score (0-1)
- `tier`: Quality tier (gold/silver/bronze)
- `quality_score`: Overall quality metric (0-1)
- `notes`: Additional metadata

## Dataset Versions & Timeline

### Current Local Workspace Snapshot - March 28, 2026
- Comprehensive corpus: 9,910 pairs
- Train / valid / test: 8,434 / 738 / 738
- Quality labels present in local metadata: 8,444 gold and 1,466 silver
- Includes post-release open-source supplement rows from Hollis, Hinde, and ASJP
- Older smaller-snapshot references describe the pre-supplement dataset state

### v1.0 (Previous)
- Initial dataset: 3,010 pairs
- Synthetic-heavy (88% synthetic, 12% cultural)
- Limited Bible coverage
- Basic domain balance

## Use Cases

1. **Machine Translation Training**
   - LoRA fine-tuning on instruction-tuned base models (Qwen, Llama)
   - Low-resource MT with curriculum learning
   - Suggested: 80% Bible (high-confidence) + 20% cultural (diverse domains)

2. **Language Preservation**
   - Maasai fluency development
   - Cultural knowledge encoding
   - Academic linguistic research

3. **Multimodal Learning**
   - Combine with ASR (Maasai speech→text)
   - Visual learning (images + Maasai terms)
   - Knowledge grounding

4. **Community Resources**
   - Educational glossaries
   - Translation assistance tools
   - Cultural documentation

## Training Recommendations

Use the current `data/final_v3` snapshot as the source of truth for counts, but do not assume every row is equally reliable.

### Recommended Starting Point
```
- Start from a spot-checked subset of `data/final_v3`
- Treat `tier` as dataset metadata from the curation pipeline, not as proof of native-speaker review
- Keep the curated cultural rows and open-source supplement visible in evaluation slices
- Audit Bible-derived rows carefully before long production runs
```

### Phase 2: Expansion (Optional)
- Add synthetic back-translation (mas→en→mas)
- Include paraphrase variations
- Boost underrepresented domains (governance, ceremonies)

### Phase 3: Validation
- Evaluate on the held-out test split (708 pairs)
- BLEU / chrF++ metrics
- Native speaker review (especially philosophy/ceremonies)

## Quality Assessment

### Current Metadata Pipeline
- The local files preserve tier, confidence, and provenance labels from earlier curation passes
- Open-source supplement rows were merged deterministically into the existing splits
- The trainer only requires `source_text`, `target_text`, `source_lang`, and `target_lang`
- Additional schema cleanup is still advisable before publication refreshes

### Known Limitations

1. **Limited Specific Domains**: Focus on foundational language coverage
   - Mitigation: Supplemented with 962 non-gold rows across cultural and open-source sources
   
2. **Orthography Variation**: Maasai has multiple writing systems
   - Recommendation: Use glossary for term standardization
   
3. **Dialect Variation**: Some Maasai regional differences
   - Status: Using standardized Maasai / Maa (main dialect)
   
4. **Schema Normalization**: The current local corpus is not fully normalized
   - Status: 680 rows omit `id`; 750 rows omit `quality_assessment`

5. **Low-Resource Nature**: Smaller than high-resource MT corpora
   - Mitigation: Use curriculum learning + data augmentation strategies

## Citation

```bibtex
@dataset{maasai_translation_2026,
  title={Maasai-English Translation Corpus: Bible + Cultural Knowledge},
  author={NorthernTribe Research},
  year={2026},
  version={2.0},
  publisher={Hugging Face},
  howpublished={https://huggingface.co/datasets/NorthernTribe-Research/maasai-translation-corpus-v2}
}
```

## Community & Contribution

This dataset is developed with **commitment to Maasai language preservation** and community benefit.

### Contributing
We welcome contributions:
- Additional cultural pairs (governance, ceremonies)
- Domain-specific translations
- Quality improvements / error corrections
- Native speaker validation

### Community Engagement
- Please consult with Maasai language communities
- Prioritize cultural respectfulness
- Ensure benefit sharing with Maasai people

## License

This dataset is released under the **Creative Commons Attribution 4.0 International License (CC BY 4.0)**.

**Exception**: Bible content may have additional licensing constraints - see respective Bible publisher terms.

## Acknowledgments

- **Translation Sources**: Authoritative English-Maasai translation materials
- **Cultural Knowledge**: Maasai language experts and community resources
- **Technical Framework**: DeepSeek best practices for low-resource MT
- **Infrastructure**: Hugging Face Hub

## Related Resources

- **Dataset Paper**: [link to evaluation/metrics]
- **Model Card**: See accompanying model-card.md
- **Evaluation**: eval_results.json (BLEU/chrF++ metrics)
- **Glossary**: src/glossary.py (term database)
- **Space Demo**: https://huggingface.co/spaces/NorthernTribe-Research/maasai-language-showcase

---

**Version**: 2.0  
**Date**: March 26, 2026  
**Maintainer**: NorthernTribe Research  
**Status**: ✅ Production Ready
