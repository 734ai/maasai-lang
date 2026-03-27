"""
Central configuration for the Maasai Language Showcase project.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
GLOSSARY_DIR = DATA_DIR / "glossary"
EVAL_DIR = DATA_DIR / "eval"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# ---------------------------------------------------------------------------
# Model IDs
# ---------------------------------------------------------------------------
TRANSLATION_BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
TRANSLATION_MODEL_ID = os.getenv("TRANSLATION_MODEL_ID", "NorthernTribe-Research/maasai-en-mt")
ASR_MODEL_ID = os.getenv("ASR_MODEL_ID", "microsoft/paza-whisper-large-v3-turbo")

# ---------------------------------------------------------------------------
# Language codes
# ---------------------------------------------------------------------------
LANG_EN = "en"
LANG_MAS = "mas"
SUPPORTED_DIRECTIONS = [
    (LANG_EN, LANG_MAS),
    (LANG_MAS, LANG_EN),
]

# ---------------------------------------------------------------------------
# Maasai sub-tribes / sections reference
# ---------------------------------------------------------------------------
MAASAI_SECTIONS = [
    "Ilkisongo",
    "Ilpurko",
    "Iloitai",
    "Ildamat",
    "Ilkeekonyokie",
    "Iloodokilani",
    "Ilkaputiei",
    "Ilmatapato",
    "Laikipiak",
    "Ilwuasinkishu",
    "Isiria",
    "Ilmoitanik",
    "Ildalalekutuk",
    "Ilaitayiok",
    "Ilarusa",
    "Ilparakuyo",
    "Samburu",           # Lmomonyot / Sambur / Loikop
    "Ldikiri",
    "Lmomonyot",
]

# ---------------------------------------------------------------------------
# Domains for data tagging
# ---------------------------------------------------------------------------
DOMAINS = [
    "general",
    "greetings",
    "education",
    "health",
    "livestock",
    "environment",
    "daily_life",
    "culture",
    "ceremony",
    "governance",
    "numbers",
    "kinship",
    "philosophy",
    "proverbs",
]


@dataclass
class TrainingConfig:
    """Default QLoRA training hyperparameters."""
    model_name: str = TRANSLATION_BASE_MODEL
    max_length: int = 512
    learning_rate: float = 2e-4
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 8
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
