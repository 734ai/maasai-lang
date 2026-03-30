# Open Maasai Source Inventory

This project should not blindly “scrape the whole internet.”
It should expand through a lawful, source-traceable acquisition pipeline:

- ingest only sources that are clearly reusable
- keep gated, copyrighted, or unclear-rights sources out of automatic training
- separate “download now”, “parse next”, “request access”, and “reference only” lanes

The machine-readable registry for this workflow lives at:

- `data/registry/maasai_vetted_web_sources.json`

The planning helper for this workflow lives at:

- `scripts/discover_vetted_maasai_sources.py`

## Approved For Training Now

### 1. A.C. Hollis, *The Masai: their language and folklore* (1905)

- Status: approved for training
- License status: public domain
- URL: `https://archive.org/download/masaitheirlangua00holluoft/masaitheirlangua00holluoft_djvu.txt`
- Current use:
  - proverb translation pairs extracted into `data/raw/public_domain_hollis_proverbs.jsonl`
- Best next use:
  - conservative extraction of additional folklore/customs parallel passages

### 2. ASJP Maasai Wordlist

- Status: approved for training
- License status: CC BY 4.0
- URL: `https://asjp.clld.org/languages/MAASAI.json`
- Current use:
  - lexical translation pairs extracted into `data/raw/open_asjp_maasai_wordlist.jsonl`
- Best next use:
  - continue lexical grounding and glossary expansion

## Expand Next

### 3. Hildegarde Hinde, *The Masai language: grammatical notes together with a vocabulary* (1901)

- Status: approved for conservative training ingest
- License status: public domain
- URL: `https://archive.org/download/masailanguagegra00hindrich/masailanguagegra00hindrich_djvu.txt`
- Current use:
  - conservative vocabulary pairs extracted into `data/raw/public_domain_hinde_vocabulary.jsonl`
- Best next use:
  - extend coverage only after reviewing earlier alphabet blocks and mixed-layout OCR pages
- Caution:
  - do not chunk OCR prose blindly or promote noisy grammar pages into MT training

## Request Access Next

### 4. African Next Voices: Pilot Data Collection in Kenya

- Status: gated-access
- License status: CC BY 4.0
- URL: `https://huggingface.co/datasets/MCAA1-MSU/anv_data_ke`
- Value:
  - 505 Maasai hours listed on the dataset card
  - scripted Maasai with English translation fields
  - fully transcribed unscripted Maasai speech
- Best next use:
  - request access for ASR training and speech evaluation

### 5. Maa Maasai Language Project (University of Oregon)

- Status: permission / access request
- License status: copyrighted
- URL: `https://pages.uoregon.edu/maasai/`
- Value:
  - dictionary and text catalogue
  - multiple dialects and strong linguistic quality
- Caution:
  - the site asks users not to copy without acknowledgment
- Best next use:
  - contact maintainers for explicit permission before ingestion

## Policy Review / Permission Required

### 6. Universal Declaration of Human Rights: translation into Maa (2023)

- Status: policy review
- License status: review before bulk training use
- URL: `https://searchlibrary.ohchr.org/record/30740`
- Value:
  - high-value modern parallel text
- Best next use:
  - evaluation/reference first, then confirm reuse policy before ingestion

### 7. Wikitongues Maasai sample on Wikimedia Commons

- Status: policy review
- License status: CC BY-SA 4.0
- URL: `https://commons.wikimedia.org/wiki/File:Massai.ogg`
- Value:
  - small public Maa speech sample for ASR/demo checks
- Best next use:
  - evaluation/demo use unless share-alike policy is cleared for training use

## Reference Only For Now

### 8. Frans Mol, *Maasai Language & Culture: Dictionary* (Google Books preview)

- Status: reference / clearance lead
- License status: no open reuse grant found
- URL: `https://books.google.com/books/about/Maasai_Language_Culture.html?id=22kOAAAAYAAJ`
- Best next use:
  - bibliographic lead only unless rights are cleared with the rightsholder

### 9. MasaiMara.ke Learn Maasai Language Guide (January 2025 PDF)

- Status: reference only
- License status: not confirmed for redistribution or training reuse
- URL: `https://masaimara.ke/wp-content/uploads/2025/01/Learn-Maasai-Language-Guide-1.pdf`
- Best next use:
  - manual review, prompt ideas, and evaluation examples only

## Recommended Run

```bash
python scripts/discover_vetted_maasai_sources.py summary
python scripts/discover_vetted_maasai_sources.py plan --output outputs/maasai_source_download_plan.json
```

## Near-Term Priorities

1. Extend the Hinde parser beyond the currently ingested high-confidence blocks.
2. Keep expanding Hollis conservatively beyond proverb-only coverage.
3. Request access to ANV for speech training and evaluation.
4. Contact UOregon before any dictionary/text ingestion.
5. Treat OHCHR, Wikimedia CC BY-SA, Frans Mol preview, and modern travel-guide materials as review/permission lanes until rights are explicit.
