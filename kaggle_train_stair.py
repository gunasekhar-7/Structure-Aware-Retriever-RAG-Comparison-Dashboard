"""
Kaggle T4 GPU 1-Click Fine-Tuning Script for STAIR.

Run this script directly in a Kaggle Notebook terminal with GPU Accelerator enabled:
    !python kaggle_train_stair.py

Requirements on Kaggle:
    !pip install --no-deps unsloth
    !pip install trl datasets transformers peft bitsandbytes accelerate
"""

import os
import sys
import json
import shutil
import zipfile
from pathlib import Path

# Verify CUDA GPU on Kaggle
import torch
print("=" * 60)
print(f"🔥 Kaggle GPU Detected: {torch.cuda.get_device_name(0)}")
print(f"🔥 Total VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
print("=" * 60)

# Import Unsloth & SFTTrainer
from unsloth import FastLanguageModel
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers import TrainingArguments
from datasets import load_dataset

# 1. Configuration parameters
MAX_SEQ_LENGTH = 4096
MODEL_NAME = "unsloth/llama-3.2-3b-instruct-bnb-4bit"
OUTPUT_DIR = "stair_lora_adapter"
DATA_FILE = "data/synthetic_qa_train.jsonl"
VAL_FILE = "data/synthetic_qa_val.jsonl"

# Check if data exists; if not, create sample data
if not os.path.exists(DATA_FILE):
    print("[INFO] Data file not found locally. Initializing sample data generator...")
    from sample_pdf_processor import main as run_sample
    run_sample()
    from src.generate_qa import main as run_qa
    run_qa()

# 2. Load Model & Tokenizer with 4-bit Quantization
print(f"\n[INFO] Loading {MODEL_NAME} with 4-bit QLoRA...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=True,
)

# 3. Add LoRA Adapters (r=16, alpha=32 per STAIR paper)
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.0,
    bias="none",
    use_gradient_checkpointing=True,
    random_state=42,
)

# 4. Load Datasets
train_dataset = load_dataset("json", data_files=DATA_FILE, split="train")
val_dataset = load_dataset("json", data_files=VAL_FILE, split="train") if os.path.exists(VAL_FILE) else None
print(f"✓ Train dataset loaded: {len(train_dataset)} examples")
if val_dataset:
    print(f"✓ Val dataset loaded:   {len(val_dataset)} examples")

# 5. Format Prompts & Completion Collator
def formatting_prompts_func(examples):
    return [p + c + tokenizer.eos_token for p, c in zip(examples["prompt"], examples["completion"])]

collator = DataCollatorForCompletionOnlyLM(
    response_template="### Section: leaf ",
    tokenizer=tokenizer
)

# 6. Training Arguments optimized for Kaggle T4
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    num_train_epochs=3,
    fp16=True,
    logging_steps=5,
    eval_strategy="epoch" if val_dataset else "no",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True if val_dataset else False,
    report_to="none",
    remove_unused_columns=False,
    dataloader_pin_memory=False,
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    formatting_func=formatting_prompts_func,
    data_collator=collator,
    max_seq_length=MAX_SEQ_LENGTH,
    args=training_args,
)

print("\n🚀 Starting Fine-Tuning...")
trainer.train()

# 7. Save Adapter & Export as Zip for 1-Click Download
print(f"\n[INFO] Saving LoRA adapter to {OUTPUT_DIR}...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

zip_filename = "stair_lora_adapter.zip"
print(f"[INFO] Packaging adapter into {zip_filename} for download...")
with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zipf:
    for root, _, files in os.walk(OUTPUT_DIR):
        for file in files:
            full_path = os.path.join(root, file)
            arcname = os.path.relpath(full_path, start=os.path.dirname(OUTPUT_DIR))
            zipf.write(full_path, arcname)

print("\n" + "=" * 60)
print("✅ STAIR Training Complete!")
print(f"📦 Download your trained adapter from Kaggle Output: {zip_filename}")
print("   Unzip this file into your local project root as 'stair_lora_adapter/'")
print("=" * 60)
