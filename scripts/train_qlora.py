#!/usr/bin/env python3
"""
QLoRA fine-tuning script for English <-> Maasai translation.

Expected input:
- data/final_v3/train.jsonl
- data/final_v3/valid.jsonl
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import sys
from typing import Any

import torch
from datasets import Dataset, DatasetDict, load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.generation_data import build_instruction_mixture

try:
    from transformers import BitsAndBytesConfig
except ImportError:  # pragma: no cover - depends on transformers version
    BitsAndBytesConfig = None


LOGGER = logging.getLogger("train_qlora")


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--train_file", type=str, default="data/final_v3/train.jsonl")
    parser.add_argument("--valid_file", type=str, default="data/final_v3/valid.jsonl")
    parser.add_argument("--output_dir", type=str, default="outputs/maasai-en-mt-qlora")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--logging_steps", type=int, default=20)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_model_id", type=str, default=None)
    parser.add_argument("--hub_private_repo", action="store_true")
    parser.add_argument(
        "--hub_strategy",
        type=str,
        default="every_save",
        choices=["end", "every_save", "checkpoint", "all_checkpoints"],
    )
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--report_to", type=str, default="none")
    parser.add_argument("--require_4bit", action="store_true")
    parser.add_argument("--augment_with_generation_tasks", action="store_true")
    parser.add_argument("--story_seed_file", type=str, default="data/raw/maasai_story_generation_seed.jsonl")
    parser.add_argument("--max_bible_passages", type=int, default=48)
    parser.add_argument("--bible_passage_window", type=int, default=3)
    return parser.parse_args()


def load_data(train_file: str, valid_file: str):
    data_files = {
        "train": train_file,
        "validation": valid_file,
    }
    return load_dataset("json", data_files=data_files)


def maybe_augment_instruction_mixture(
    dataset: DatasetDict,
    *,
    enabled: bool,
    story_seed_file: str,
    max_bible_passages: int,
    bible_passage_window: int,
    seed: int,
) -> DatasetDict:
    if not enabled:
        return dataset

    train_rows = build_instruction_mixture(
        dataset["train"].to_list(),
        "train",
        story_seed_file=Path(story_seed_file),
        max_bible_passages=max_bible_passages,
        bible_passage_window=bible_passage_window,
        seed=seed,
    )
    valid_rows = build_instruction_mixture(
        dataset["validation"].to_list(),
        "valid",
        story_seed_file=Path(story_seed_file),
        max_bible_passages=max(1, max_bible_passages // 6),
        bible_passage_window=bible_passage_window,
        seed=seed,
    )

    LOGGER.info(
        "Augmented instruction mixture: train %d -> %d, validation %d -> %d",
        len(dataset["train"]),
        len(train_rows),
        len(dataset["validation"]),
        len(valid_rows),
    )
    return DatasetDict(
        {
            "train": Dataset.from_list(train_rows),
            "validation": Dataset.from_list(valid_rows),
        }
    )


def build_prompt(prompt: str, completion: str) -> str:
    return f"{prompt}\n\n### Response:\n{completion}"


def build_translation_prompt(source_text: str, source_lang: Any, target_lang: Any) -> str:
    source_lang = str(source_lang or "").strip().lower()
    target_lang = str(target_lang or "").strip().lower()
    source_text = str(source_text or "").strip()

    if source_lang == "en" and target_lang == "mas":
        return f'Translate the following English sentence to Maasai:\n"{source_text}"'
    if source_lang == "mas" and target_lang == "en":
        return f'Translate the following Maasai sentence to English:\n"{source_text}"'
    if source_lang and target_lang:
        return f'Translate from {source_lang} to {target_lang}:\n"{source_text}"'
    return f'Translate the following sentence:\n"{source_text}"'


def get_column_value(examples: dict[str, list[Any]], column: str, index: int) -> Any:
    values = examples.get(column)
    if values is None:
        return None
    return values[index]


def normalize_examples(examples: dict[str, list[Any]]) -> tuple[list[str], list[str]]:
    if not examples:
        raise ValueError("Empty example batch received")

    batch_size = len(next(iter(examples.values())))
    prompts: list[str] = []
    completions: list[str] = []

    for idx in range(batch_size):
        prompt = get_column_value(examples, "prompt", idx)
        completion = get_column_value(examples, "completion", idx)

        if prompt is None or completion is None:
            source_text = get_column_value(examples, "source_text", idx)
            target_text = get_column_value(examples, "target_text", idx)
            if source_text is None or target_text is None:
                raise KeyError(
                    "Training data must include either prompt/completion or "
                    "source_text/target_text fields."
                )
            prompt = build_translation_prompt(
                source_text=source_text,
                source_lang=get_column_value(examples, "source_lang", idx),
                target_lang=get_column_value(examples, "target_lang", idx),
            )
            completion = target_text

        prompts.append(str(prompt).strip())
        completions.append(str(completion).strip())

    return prompts, completions


def tokenize_examples(examples, tokenizer, max_length: int):
    prompts, completions = normalize_examples(examples)
    texts = [build_prompt(prompt, completion) for prompt, completion in zip(prompts, completions)]
    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding=False,
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


def maybe_limit_split(dataset, split_name: str, max_samples: int | None):
    if max_samples is None:
        return dataset

    limit = min(max_samples, len(dataset[split_name]))
    LOGGER.info("Limiting %s split to %d samples", split_name, limit)
    dataset[split_name] = dataset[split_name].select(range(limit))
    return dataset


def preferred_torch_dtype() -> torch.dtype:
    if not torch.cuda.is_available():
        return torch.float32

    is_bf16_supported = getattr(torch.cuda, "is_bf16_supported", None)
    if callable(is_bf16_supported) and is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def load_model(args: argparse.Namespace):
    dtype = preferred_torch_dtype()
    model_kwargs = {
        "torch_dtype": dtype,
        "trust_remote_code": True,
        "local_files_only": args.local_files_only,
    }
    use_4bit = torch.cuda.is_available() and BitsAndBytesConfig is not None

    if args.require_4bit and not torch.cuda.is_available():
        raise RuntimeError("--require_4bit requires a CUDA runtime.")
    if args.require_4bit and BitsAndBytesConfig is None:
        raise RuntimeError("4-bit loading requested, but BitsAndBytesConfig is unavailable in this transformers build.")

    if use_4bit:
        LOGGER.info("Loading 4-bit base model")
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        try:
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name,
                quantization_config=quant_config,
                device_map="auto",
                **model_kwargs,
            )
            model.config.use_cache = False
            model = prepare_model_for_kbit_training(model)
            return model
        except Exception as exc:  # pragma: no cover - depends on local runtime
            if args.require_4bit:
                raise RuntimeError(f"4-bit model load failed: {exc}") from exc
            LOGGER.warning("4-bit load failed, falling back to standard precision: %s", exc)
    else:
        if args.require_4bit:
            raise RuntimeError("4-bit loading requested, but the current runtime cannot satisfy it.")
        LOGGER.info("4-bit quantization unavailable; loading standard-precision base model")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map="auto" if torch.cuda.is_available() else None,
        **model_kwargs,
    )
    model.config.use_cache = False
    if not torch.cuda.is_available():
        model.to("cpu")
    return model


def main() -> None:
    setup_logging()
    args = parse_args()

    if args.push_to_hub and not args.hub_model_id:
        raise ValueError("--hub_model_id is required when --push_to_hub is enabled")

    os.makedirs(args.output_dir, exist_ok=True)
    LOGGER.info("Loading dataset")
    dataset = load_data(args.train_file, args.valid_file)
    dataset = maybe_augment_instruction_mixture(
        dataset,
        enabled=args.augment_with_generation_tasks,
        story_seed_file=args.story_seed_file,
        max_bible_passages=args.max_bible_passages,
        bible_passage_window=args.bible_passage_window,
        seed=args.seed,
    )
    dataset = maybe_limit_split(dataset, "train", args.max_train_samples)
    dataset = maybe_limit_split(dataset, "validation", args.max_eval_samples)

    LOGGER.info("Loading tokenizer: %s", args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        use_fast=True,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = load_model(args)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    LOGGER.info("Tokenizing dataset")
    tokenized = dataset.map(
        lambda x: tokenize_examples(x, tokenizer, args.max_length),
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing",
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        max_steps=args.max_steps,
        bf16=torch.cuda.is_available(),
        fp16=False,
        gradient_checkpointing=torch.cuda.is_available(),
        report_to=args.report_to,
        seed=args.seed,
        dataloader_num_workers=2 if torch.cuda.is_available() else 0,
        load_best_model_at_end=False,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
        hub_private_repo=args.hub_private_repo,
        hub_strategy=args.hub_strategy,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=data_collator,
        processing_class=tokenizer,
    )

    LOGGER.info("Starting training")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint or None)

    LOGGER.info("Saving adapter and tokenizer to %s", args.output_dir)
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    if args.push_to_hub:
        LOGGER.info("Pushing final trainer state to the Hugging Face Hub")
        trainer.push_to_hub()

    LOGGER.info("Done")


if __name__ == "__main__":
    main()
