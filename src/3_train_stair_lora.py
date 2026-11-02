"""
Phase 3: STAIR Model Fine-Tuning Pipeline (Optimized for Kaggle T4 GPU).

Fine-tunes Llama-3.2-3B or Mistral-7B using 4-bit quantization (QLoRA)
and Unsloth on Kaggle Tesla T4 GPU (16GB VRAM constraint).
Saves the resulting adapter ready for constrained inference.
"""

import argparse
import json
import os
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import torch


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def check_gpu_environment():
    """Verify GPU hardware capabilities and memory availability."""
    print("=" * 60)
    print("⚡ STAIR Kaggle GPU Environment Verification")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("[WARNING] CUDA GPU not detected! Training on CPU will be extremely slow.")
        print("          For training, please run in a Kaggle T4 GPU notebook.")
        return {"device": "cpu", "vram_gb": 0.0, "is_gpu": False}

    device_name = torch.cuda.get_device_name(0)
    vram_bytes = torch.cuda.get_device_properties(0).total_memory
    vram_gb = vram_bytes / (1024 ** 3)
    
    print(f"✓ GPU Device:     {device_name}")
    print(f"✓ Total VRAM:     {vram_gb:.2f} GB")
    print(f"✓ PyTorch CUDA:   {torch.version.cuda}")
    print(f"✓ Device Count:   {torch.cuda.device_count()}")
    
    return {"device": "cuda", "vram_gb": vram_gb, "is_gpu": True}


def load_datasets(data_dir: Path):
    """Load train and validation splits from JSONL files."""
    train_path = data_dir / "synthetic_qa_train.jsonl"
    val_path = data_dir / "synthetic_qa_val.jsonl"
    
    if not train_path.exists():
        raise FileNotFoundError(f"Training dataset not found at {train_path}. Run 2_generate_qa.py first.")
        
    from datasets import load_dataset
    print(f"\n[INFO] Loading training dataset from {train_path.name}...")
    train_dataset = load_dataset("json", data_files=str(train_path), split="train")
    
    val_dataset = None
    if val_path.exists():
        print(f"[INFO] Loading validation dataset from {val_path.name}...")
        val_dataset = load_dataset("json", data_files=str(val_path), split="train")
        
    print(f"✓ Train examples:      {len(train_dataset)}")
    if val_dataset:
        print(f"✓ Validation examples: {len(val_dataset)}")
        
    return train_dataset, val_dataset


def train_stair_lora(
    model_name: str = "unsloth/llama-3.2-3b-instruct-bnb-4bit",
    output_dir: str = "stair_lora_adapter",
    num_train_epochs: int = 3,
    per_device_batch_size: int = 2,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 2e-4,
    max_seq_length: int = 4096,
    dry_run: bool = False
):
    """
    Main training execution function.
    Supports Unsloth acceleration with HuggingFace PEFT fallback.
    """
    project_root = get_project_root()
    data_dir = project_root / "data"
    output_path = project_root / output_dir
    
    env_info = check_gpu_environment()
    
    if dry_run:
        print("\n[DRY RUN] Dry run mode requested. Checking datasets and configuration only...")
        train_ds, val_ds = load_datasets(data_dir)
        print("✓ Dry run configuration valid. Ready for Kaggle execution.")
        return

    train_ds, val_ds = load_datasets(data_dir)
    
    # Check if Unsloth is available
    use_unsloth = False
    try:
        from unsloth import FastLanguageModel
        use_unsloth = True
        print("\n[INFO] Unsloth FastLanguageModel engine detected. Enabling fast fused kernels.")
    except ImportError:
        print("\n[INFO] Unsloth not detected. Falling back to standard HuggingFace PEFT + BitsAndBytes.")

    model = None
    tokenizer = None

    if use_unsloth:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            load_in_4bit=True,
        )
        
        # Apply LoRA with STAIR paper hyperparameters (r=16, alpha=32)
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
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
            
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        ) if env_info["is_gpu"] else None
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto" if env_info["is_gpu"] else "cpu",
            torch_dtype=torch.float16 if env_info["is_gpu"] else torch.float32,
        )
        
        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, peft_config)

    # Formatting function
    def formatting_prompts_func(examples):
        prompts = examples["prompt"]
        completions = examples["completion"]
        texts = []
        for p, c in zip(prompts, completions):
            texts.append(f"{p}{c}{tokenizer.eos_token}")
        return texts

    from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
    from transformers import TrainingArguments

    response_template = "### Section: leaf "
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer
    )

    training_args = TrainingArguments(
        output_dir=str(output_path),
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        num_train_epochs=num_train_epochs,
        fp16=env_info["is_gpu"],
        logging_steps=5,
        eval_strategy="epoch" if val_ds is not None else "no",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True if val_ds is not None else False,
        report_to="none",
        remove_unused_columns=False,
        dataloader_pin_memory=False,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        formatting_func=formatting_prompts_func,
        data_collator=collator,
        max_seq_length=max_seq_length,
        args=training_args,
    )

    print("\n" + "=" * 60)
    print(f"🚀 Launching STAIR LoRA Fine-Tuning ({num_train_epochs} Epochs)...")
    print(f"   Effective Batch Size: {per_device_batch_size * gradient_accumulation_steps}")
    print(f"   Target VRAM Footprint: ~4-6 GB (Fits easily on Kaggle 16GB T4)")
    print("=" * 60 + "\n")

    trainer.train()

    # Save fine-tuned adapter & tokenizer
    print(f"\n[INFO] Saving fine-tuned adapter to {output_path}...")
    model.save_pretrained(str(output_path))
    tokenizer.save_pretrained(str(output_path))
    print(f"✓ Model successfully trained and saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="STAIR Model Fine-Tuning Pipeline for Kaggle T4.")
    parser.add_argument("--model", type=str, default="unsloth/llama-3.2-3b-instruct-bnb-4bit", help="Base model to tune.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=2, help="Per-device batch size.")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps.")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate.")
    parser.add_argument("--output-dir", type=str, default="stair_lora_adapter", help="Output directory.")
    parser.add_argument("--dry-run", action="store_true", help="Validate environment and data without full training.")
    args = parser.parse_args()

    train_stair_lora(
        model_name=args.model,
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        dry_run=args.dry_run
    )


if __name__ == "__main__":
    main()
