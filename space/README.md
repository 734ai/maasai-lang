---
title: Maasai Language Showcase
colorFrom: gray
colorTo: red
sdk: gradio
sdk_version: 5.23.0
app_file: app.py
pinned: false
---
# Maasai Language Showcase

Gradio Space for English↔Maasai translation, Maasai speech transcription, glossary-aware cultural exploration, and curated bilingual content.

## Connected Assets

- Translation model: `NorthernTribe-Research/maasai-en-mt`
- Dataset: `NorthernTribe-Research/maasai-translation-corpus`
- ASR model: `microsoft/paza-whisper-large-v3-turbo`
- Glossary: bundled `data/glossary/maasai_glossary.json`

## App Scope

- English→Maasai and Maasai→English translation
- Browser-side voice playback for translated output, including Maa/Maasai output through the closest available regional browser voice
- Maasai speech transcription and transcribe-then-translate flow
- Glossary-backed cultural terminology support
- Additional bilingual content tabs for stories, riddles, history, and cultural reference material

## Runtime Notes

- The app lazy-loads models and falls back to demo behavior if a model is unavailable in the deployment environment.
- Voice playback uses the browser Web Speech API. Dedicated Maa voices are uncommon, so Maasai playback falls back to the closest available Kenyan/regional browser voice.
- Outputs should be reviewed by native Maa speakers before formal or public-critical use.
