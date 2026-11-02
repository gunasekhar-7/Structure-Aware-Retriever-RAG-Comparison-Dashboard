# Project Title: Structure-Aware Retriever — A Vectorless, Structure-Aware RAG Assistant for Technical Manuals

Building a **single-book interactive Q&A comparison dashboard in Streamlit** is the cleanest way to showcase your mastery of the STAIR (Structure Aware Information Retriever) architecture without needing massive compute. By training STAIR on **one structured open textbook** (such as *Principles of Mechanics*—the smallest book in the SearchTome benchmark at 179 pages and 125 leaf sections—or an official Python/FastAPI documentation PDF) and placing it side-by-side against traditional chunked retrieval, you visually prove *why* STAIR is superior.

---

## 1. What Your Project Will Do (The Portfolio Hook)

When an interviewer or user opens your Streamlit app and asks a question about the book:
1. **Left Column (Standard Chunked RAG / BM25):** Shows traditional 500-token chunk retrieval failing or returning a cut-off paragraph lacking context.
2. **Right Column (Your STAIR Engine):** 
   * Displays the **hierarchical Table of Contents (ToC)** with the exact predicted `leaf` node highlighted (e.g., `leaf 4.2.3 Electoral Systems & Political Parties`).
   * Fetches the **complete, semantically coherent subsection** from a lightweight JSON store (zero vector database used).
   * Generates a grounded final answer with **0% section-ID hallucination** via constrained decoding.

---

## 2. Minimal Project Structure

```text
docustair-rag/
├── data/
│   ├── book.pdf                  # 1 structured PDF (100-200 pages with bookmarks)
│   ├── toc_and_sections.json     # Extracted ToC string + {leaf_title: full_text}
│   └── synthetic_qa_train.jsonl  # ~1,000 generated (ToC + Query -> Leaf) pairs
├── src/
│   ├── 1_extract_toc.py          # PyMuPDF script to build ToC & leaf nodes
│   ├── 2_generate_qa.py          # LLM API script to generate synthetic questions
│   ├── 3_train_stair_lora.py     # Unsloth / PEFT LoRA fine-tuning script
│   └── 4_inference_engine.py     # Constrained generation (validates leaf output)
├── app.py                        # Streamlit UI comparing Chunked RAG vs. STAIR
└── README.md                     # Architecture diagram + Recall@1 evaluation table
```

---

## 3. Step-by-Step Implementation Code

### Step 1: Parse the PDF & Identify `leaf` Nodes (`1_extract_toc.py`)
The STAIR paper defines a leaf node as a ToC entry that has no subsections beneath it. Using `PyMuPDF` (`fitz`), you can extract both the formatted ToC string (with `leaf` prefixes) and the text mapping in under 40 lines of code:

```python
import fitz  # PyMuPDF
import json

def extract_stair_toc(pdf_path: str):
    doc = fitz.open(pdf_path)
    toc = doc.get_toc()  # Returns list of [level, title, page_num]
    
    formatted_toc_lines = []
    leaf_sections = {}
    
    for i, (level, title, start_page) in enumerate(toc):
        # A node is a leaf if it's the last entry OR the next entry is not a deeper sub-level
        is_leaf = (i == len(toc) - 1) or (toc[i + 1][0] <= level)
        clean_title = title.strip()
        
        if is_leaf:
            leaf_label = f"leaf {clean_title}"
            formatted_toc_lines.append(leaf_label)
            
            # Extract text from start_page to the next section's start_page
            end_page = toc[i + 1][2] if (i + 1 < len(toc)) else len(doc)
            section_text = ""
            for p in range(start_page - 1, min(end_page, len(doc))):
                section_text += doc[p].get_text("text") + "\n"
            
            leaf_sections[clean_title] = section_text.strip()
        else:
            formatted_toc_lines.append(clean_title)
            
    return "\n".join(formatted_toc_lines), leaf_sections

if __name__ == "__main__":
    toc_string, leaves = extract_stair_toc("data/book.pdf")
    with open("data/toc_and_sections.json", "w") as f:
        json.dump({"toc_prompt": toc_string, "leaves": leaves}, f, indent=2)
    print(f"Extracted {len(leaves)} leaf nodes successfully!")
```

---

### Step 2: Generate Synthetic Training Data (`2_generate_qa.py`)
Instead of running a heavy 8x7B model locally, you can use a fast free-tier API (like `Gemini 2.5 Flash` or `Groq Llama-3.3-70B`) to generate 3–5 questions per paragraph for your single book (takes ~5 minutes for a 150-page book) and format each sample exactly like **Table 2** of the paper:

```python
def format_stair_training_example(book_title: str, toc_string: str, query: str, target_leaf: str):
    prompt = (
        f"### Book: {book_title}\n"
        f"Pick the best section from the table of contents below which can answer the user query below. "
        f"Only generate the section name, do not generate any explanation!\n"
        f"### Table of Contents\n{toc_string}\n"
        f"### Query: {query}\n"
        f"### Section: leaf "
    )
    return {"prompt": prompt, "completion": target_leaf}
```

---

### Step 3: Fine-Tune with LoRA on Free Google Colab (`3_train_stair_lora.py`)
Using **Unsloth** on a free Google Colab T4 GPU (or Kaggle 2x T4), you can fine-tune `Mistral-7B-Instruct-v0.2` (or `Llama-3.2-3B-Instruct` for faster iteration) using the paper's exact LoRA hyperparameters ($r=16, \alpha=32$):

```python
from unsloth import FastLanguageModel
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers import TrainingArguments
from datasets import load_dataset

max_seq_length = 8192 # Fits full ToC + Query easily
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/mistral-7b-instruct-v0.2-bnb-4bit",
    max_seq_length=max_seq_length,
    load_in_4bit=True,
)

# Paper hyperparameters: r=16, alpha=32
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)

dataset = load_dataset("json", data_files="data/synthetic_qa_train.jsonl", split="train")

def formatting_prompts_func(examples):
    return [p + c + tokenizer.eos_token for p, c in zip(examples["prompt"], examples["completion"])]

# Mask loss on the ToC prompt so the LLM only learns to predict the target leaf section
collator = DataCollatorForCompletionOnlyLM(response_template="### Section: leaf ", tokenizer=tokenizer)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    formatting_func=formatting_prompts_func,
    data_collator=collator,
    max_seq_length=max_seq_length,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=5,
        learning_rate=2e-4,
        fp16=True,
        output_dir="stair_lora_adapter",
    ),
)
trainer.train()
model.save_pretrained("stair_lora_adapter")
```

---

### Step 4: Constrained Inference + Streamlit Demo (`app.py`)
To implement **Stage 3 (Constrained Generation)** from Figure 2 of the paper, use a prefix tree or `Outlines` (`outlines.generate.choice(model, list_of_valid_leaves)`) so your model is mathematically restricted to outputting *only* valid leaf titles from `toc_and_sections.json`.

In your `app.py`:
1. User enters a query (e.g., *"How do you calculate centripetal acceleration?"*).
2. Your fine-tuned STAIR model predicts the exact leaf node (`1.4 Uniform Circular Motion`).
3. Your app looks up `leaves["1.4 Uniform Circular Motion"]` in `toc_and_sections.json` and displays both the **ToC Tree Path** and the **Retrieved Section Text** alongside a small metrics badge (`Hallucination Rate: 0.0% | Vector DB Used: None`).

---

## 4. How to Present This on Your Resume / GitHub

Include a small benchmark table in your `README.md` on 100 held-out test questions from your chosen book comparing:
* **BM25 (`rank_bm25`)** vs. **DSI (LoRA fine-tuned *without* ToC in the prompt)** vs. **Your STAIR Model (LoRA fine-tuned *with* ToC + Constrained Decoding)**.
* Highlight that you implemented **parametric knowledge ingestion**, **ToC-augmented prompt engineering**, and **constrained leaf-node decoding** based on the IBM Research STAIR architecture.