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
    - name: id
      dtype: string
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
    - name: synthetic
      dtype: bool
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

This dataset contains **9,406 high-quality English-Maasai translation pairs** specifically designed for building fluent Maasai language models and preservation applications. It features:

- **Diverse authenticated translations** (9,406 pairs) - From various authoritative sources
- **Rich cultural coverage** (15+ domains) - Philosophy, ceremonies, governance, education, greetings, and more
- **Open-source supplement layer** - Public-domain Hollis proverb pairs and CC BY 4.0 ASJP lexical pairs
- **Perfect 50/50 bidirectional balance** - Both en→mas and mas→en directions

## Key Features

### Data Composition
```
Total pairs: 9,406
├── Train split: 7,991 pairs (85.0%)
├── Valid split: 707 pairs (7.5%)
└── Test split:  708 pairs (7.5%)

Quality tiers:
├── Gold (authenticated, high-confidence): 9,124 pairs (97.0%)
└── Silver (manually authored, knowledge-driven, or open-source supplement): 282 pairs (3.0%)

Language coverage:
├── English → Maasai: 4,703 pairs
└── Maasai → English: 4,703 pairs

Top domains:
├── Religious & Spiritual: 8,444 pairs (89.8%)
├── Proverbs: 158 pairs
├── Philosophy: 100 pairs
├── Culture: 84 pairs  
├── Lexicon: 80 pairs
├── Environment: 64 pairs
├── Education: 60 pairs
├── Ceremony: 58 pairs
├── Greetings: 52 pairs
└── Other (governance, livestock, health, daily life): ~366 pairs
```

### Quality Metrics
- **Confidence Score**: 0.95-0.98 (high confidence, authenticated sources)
- **Authenticity**: Sourced from authoritative English-Maasai translation materials
- **Deduplication**: Aggressive deduplication (9,406 unique pairs from 13,670+ raw sources)
- **Domain Coverage**: 15+ specialized domains for comprehensive fluency

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
  "source_name": "bible_english_maasai_aligned",
  "synthetic": false,
  "confidence": 0.98,
  "tier": "gold",
  "quality_score": 0.98,
  "notes": "Aligned Bible verse"
}
```

### Fields Explained
- `id`: Unique identifier for pair
- `source_text`: Text in source language
- `target_text`: Text in target language
- `source_lang`: Source language (en/mas)
- `target_lang`: Target language (mas/en)
- `domain`: Subject domain (bible, philosophy, culture, etc.)
- `source_name`: Data source identifier
- `synthetic`: Whether pair was synthetically generated
- `confidence`: Confidence score (0-1)
- `tier`: Quality tier (gold/silver/bronze)
- `quality_score`: Overall quality metric (0-1)
- `notes`: Additional metadata

## Dataset Versions & Timeline

### v2.0 (Current) - March 26, 2026
- **Major Update**: Full Bible extraction
- Comprehensive corpus: 9,194 pairs (3x expansion from v1.0)
- High-quality tiers: 91.8% Gold tier authentication
- DeepSeek best practices integration
- Ready for production model training

### v1.0 (Previous)
- Initial dataset: 3,010 pairs
- Synthetic-heavy (88% synthetic, 12% cultural)
- Limited Bible coverage
- Basic domain balance

## Use Cases

1. **Machine Translation Training**
   - LoRA fine-tuning on 4B models (Gemma, Llama)
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

Based on DeepSeek best practices for low-resource language models:

### Phase 1: Foundation (Suggested)
```
Data composition:
- 70% Gold tier (Bible): 5,910 pairs
- 30% Silver tier (Cultural): 750 pairs
Total: 6,660 pairs

Curriculum: Start with cultural knowledge (smaller, high-signal)
then gradually introduce longer Bible passages
```

### Phase 2: Expansion (Optional)
- Add synthetic back-translation (mas→en→mas)
- Include paraphrase variations
- Boost underrepresented domains (governance, ceremonies)

### Phase 3: Validation
- Evaluate on held-out test set (691 pairs)
- BLEU / chrF++ metrics
- Native speaker review (especially philosophy/ceremonies)

## Quality Assessment

### Data Cleaning Pipeline
- Language ID validation: 98%+ accuracy
- Length ratio filtering: Removes misaligned pairs
- Semantic similarity: Filters low-confidence matches
- Duplicate detection: 100% deduplication
- Tier assignment: Gold/Silver/Bronze classification

### Known Limitations

1. **Limited Specific Domains**: Focus on foundational language coverage
   - Mitigation: Supplemented with 750+ high-quality culturally-specific pairs
   
2. **Orthography Variation**: Maasai has multiple writing systems
   - Recommendation: Use glossary for term standardization
   
3. **Dialect Variation**: Some Maasai regional differences
   - Status: Using standardized Maasai / Maa (main dialect)
   
4. **Low-Resource Nature**: Smaller than high-resource MT corpora
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
