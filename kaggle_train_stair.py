"""
Kaggle T4 GPU 1-Click Fine-Tuning Script for STAIR.

Run this script directly in a Kaggle Notebook terminal with GPU Accelerator enabled:
    !python kaggle_train_stair.py

Requirements on Kaggle:
    !pip install unsloth unsloth_zoo trl datasets transformers peft bitsandbytes accelerate rank-bm25
"""

import os
import sys
import json
import random
import zipfile
from pathlib import Path

# Verify CUDA GPU on Kaggle
import torch
print("=" * 60)
if torch.cuda.is_available():
    print(f"🔥 Kaggle GPU Detected: {torch.cuda.get_device_name(0)}")
    print(f"🔥 Total VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
else:
    print("⚠️ WARNING: No CUDA GPU detected! Running on CPU.")
print("=" * 60)

from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers import TrainingArguments
from datasets import load_dataset

# 1. Configuration parameters
MAX_SEQ_LENGTH = 4096
MODEL_NAME = "unsloth/llama-3.2-3b-instruct-bnb-4bit"
OUTPUT_DIR = "stair_lora_adapter"
DATA_FILE = "data/synthetic_qa_train.jsonl"
VAL_FILE = "data/synthetic_qa_val.jsonl"
TOC_FILE = "data/toc_and_sections.json"


def build_self_contained_stair_dataset():
    """Generates STAIR documentation hierarchy, leaf sections, and synthetic QA training pairs."""
    os.makedirs("data", exist_ok=True)
    book_title = "Python 3.11 & FastAPI Production Guide"
    
    toc_prompt = """# Python 3.11 & FastAPI Production Guide
Chapter 1: Python Core Architecture
    leaf Section 1.1: Syntax, Operators & Control Flow
    leaf Section 1.2: Advanced Iterators, Generators & Context Managers
Chapter 2: Data Structures & Typing
    leaf Section 2.1: Built-in Collections & Time Complexities
    leaf Section 2.2: Structural Pattern Matching & Type Annotations
Chapter 3: Asynchronous Programming
    leaf Section 3.1: Asyncio Event Loop & Coroutine Mechanics
    leaf Section 3.2: Task Groups, Exception Handling & Deadlock Avoidance
Chapter 4: FastAPI Web Framework
    leaf Section 4.1: Routing, Path Parameters & Request Validation
    leaf Section 4.2: Dependency Injection & Database Session Lifecycles
    leaf Section 4.3: Background Tasks, Middleware & Security OAuth2
"""

    leaves = {
        "Section 1.1: Syntax, Operators & Control Flow": [
            "How does the walrus operator assignment work in Python 3.11 expressions?",
            "How are error locations highlighted in Python 3.11 tracebacks?",
            "What are the performance benefits of adaptive bytecode specialization in Python 3.11?",
            "How to use control flow statements if, elif, and while loops effectively?"
        ],
        "Section 1.2: Advanced Iterators, Generators & Context Managers": [
            "How do generator functions and yield conserve memory when streaming datasets?",
            "How to build custom context managers using contextlib and @contextmanager?",
            "What are the differences between iterables, iterators, and generators?",
            "How do __enter__ and __exit__ methods manage deterministic resource cleanup?"
        ],
        "Section 2.1: Built-in Collections & Time Complexities": [
            "What are the algorithmic time complexities of Python dicts, sets, and lists?",
            "Why is collections.deque preferred over list for FIFO queue operations?",
            "How does dictionary key hashing and collision handling work in Python?",
            "What is the memory footprint and slice assignment cost of dynamic lists?"
        ],
        "Section 2.2: Structural Pattern Matching & Type Annotations": [
            "How do match and case statements implement structural pattern matching in Python?",
            "How to validate dictionary schemas and guards using pattern matching?",
            "How do Union and pipe operator type hints integrate with Mypy and Pydantic?",
            "How to enforce strict compile-time and runtime type annotations in Python?"
        ],
        "Section 3.1: Asyncio Event Loop & Coroutine Mechanics": [
            "How does the single-threaded asyncio event loop handle cooperative multitasking?",
            "How to execute blocking CPU or synchronous I/O using asyncio.to_thread?",
            "What happens if time.sleep is called inside an asynchronous coroutine?",
            "How do async def functions schedule tasks with create_task?"
        ],
        "Section 3.2: Task Groups, Exception Handling & Deadlock Avoidance": [
            "How do asyncio.TaskGroup context managers provide structured concurrency?",
            "What happens to child tasks when an exception occurs inside a TaskGroup?",
            "How does Python 3.11 bundle concurrent errors into an ExceptionGroup?",
            "How to prevent task leaks and deadlocks during concurrent coroutine execution?"
        ],
        "Section 4.1: Routing, Path Parameters & Request Validation": [
            "How to define path parameters, query parameters, and Pydantic models in FastAPI?",
            "How does FastAPI automatically generate OpenAPI Swagger documentation?",
            "How to validate numerical ranges with Path(..., ge=1) and string lengths?",
            "What is the best way to return structured JSON responses with status codes?"
        ],
        "Section 4.2: Dependency Injection & Database Session Lifecycles": [
            "How does FastAPI Depends() manage asynchronous database session lifecycles?",
            "How do yield dependencies guarantee database connection teardown?",
            "How to share authentication and authorization contexts across multiple routes?",
            "How to mock database dependencies during automated pytest testing?"
        ],
        "Section 4.3: Background Tasks, Middleware & Security OAuth2": [
            "How to dispatch background email notifications after returning an HTTP response?",
            "How to configure CORSMiddleware to allow specific origins and headers?",
            "How does OAuth2PasswordBearer validate JWT tokens in FastAPI?",
            "How to implement role-based access control (RBAC) in FastAPI applications?"
        ]
    }

    toc_data = {
        "toc_prompt": toc_prompt,
        "leaves": {name: f"{name} content description and implementation details." for name in leaves},
        "source": "self_contained_fastapi_guide"
    }
    with open(TOC_FILE, "w", encoding="utf-8") as f:
        json.dump(toc_data, f, indent=2)

    all_examples = []
    for target_leaf, queries in leaves.items():
        for q in queries:
            prompt = (
                f"### Book: {book_title}\n"
                f"Pick the best section from the table of contents below which can answer the user query below. "
                f"Only generate the section name, do not generate any explanation!\n"
                f"### Table of Contents\n{toc_prompt}\n"
                f"### Query: {q}\n"
                f"### Section: leaf "
            )
            all_examples.append({"prompt": prompt, "completion": target_leaf})

    random.seed(42)
    random.shuffle(all_examples)
    val_count = max(4, int(len(all_examples) * 0.20))
    train_data = all_examples[val_count:]
    val_data = all_examples[:val_count]

    with open(DATA_FILE, "w", encoding="utf-8") as f:
        for ex in train_data:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    with open(VAL_FILE, "w", encoding="utf-8") as f:
        for ex in val_data:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"✓ Self-contained STAIR dataset created: {len(train_data)} train, {len(val_data)} val examples.")


# Ensure dataset exists
if not os.path.exists(DATA_FILE):
    print("[INFO] Dataset not found. Generating self-contained STAIR dataset...")
    build_self_contained_stair_dataset()

# 2. Load Model & Tokenizer with 4-bit Quantization (Unsloth or PEFT fallback)
model = None
tokenizer = None

try:
    from unsloth import FastLanguageModel
    print(f"\n[INFO] Loading {MODEL_NAME} with Unsloth FastLanguageModel...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
    )
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
    print("✓ Unsloth FastLanguageModel loaded successfully.")
except Exception as e:
    print(f"\n⚠️ Unsloth not active ({e}). Falling back to native Hugging Face PEFT 4-bit QLoRA...")
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
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
    print("✓ Native Hugging Face PEFT loaded successfully.")

# 3. Load Datasets
train_dataset = load_dataset("json", data_files=DATA_FILE, split="train")
val_dataset = load_dataset("json", data_files=VAL_FILE, split="train")
print(f"✓ Train dataset loaded: {len(train_dataset)} examples")
print(f"✓ Val dataset loaded:   {len(val_dataset)} examples")

# 4. Format Prompts & Completion Collator
def formatting_prompts_func(examples):
    return [p + c + tokenizer.eos_token for p, c in zip(examples["prompt"], examples["completion"])]

collator = DataCollatorForCompletionOnlyLM(
    response_template="### Section: leaf ",
    tokenizer=tokenizer
)

# 5. Training Arguments optimized for Kaggle T4
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    num_train_epochs=3,
    fp16=torch.cuda.is_available(),
    logging_steps=5,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
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

# 6. Save Adapter & Export as Zip for 1-Click Download
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
