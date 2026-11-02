# ⚡ Kaggle T4 GPU Fine-Tuning Guide (STAIR Architecture)

This guide walks you through fine-tuning the **STAIR (Structure-Aware Information Retriever)** model on **Kaggle's free Tesla T4 GPU** (16GB VRAM) in ~15 minutes.

---

## 🎯 Architecture & Objective

The goal is to fine-tune `unsloth/llama-3.2-3b-instruct-bnb-4bit` using **QLoRA (4-bit)** to predict the exact leaf section from a technical document's Table of Contents given an arbitrary user query:

- **Input Prompt:** `### Book: ... \n ### Table of Contents ... \n ### Query: ... \n ### Section: leaf `
- **Target Completion:** `Section Name`
- **Inference Guarantee:** Token generation is constrained via a **Prefix Trie** (`LeafTrie`), guaranteeing **0.00% section ID hallucination**.

---

## 🚀 Quick Start (Option A: 1-Click Kaggle Notebook)

1. Open [kaggle.com/code](https://www.kaggle.com/code) and click **"New Notebook"**.
2. Go to **File ➔ Import Notebook** and upload:
   `notebooks/kaggle_stair_finetuning.ipynb`
3. In the right-hand **Notebook Settings** panel:
   - **Accelerator:** Select **GPU T4 x 2** (or **GPU P100**).
   - **Internet:** Toggle **On**.
4. Click **Run All** (or execute cells sequentially).
5. Once training completes, download `stair_lora_adapter.zip` from the **Output** tab.
6. Extract the zip into your local project root as:
   `./stair_lora_adapter/`

---

## 🛠️ Quick Start (Option B: Standalone Script on Kaggle)

If you prefer uploading and running via the Kaggle terminal:

### Step 1: Install Dependencies
In a Kaggle code cell:
```bash
!pip install --no-deps unsloth
!pip install trl datasets transformers peft bitsandbytes accelerate rank-bm25
```

### Step 2: Upload Files
Upload the following files to Kaggle (via "Add Data" or drag-and-drop into `/kaggle/working`):
- `kaggle_train_stair.py`
- `data/synthetic_qa_train.jsonl`
- `data/synthetic_qa_val.jsonl`

### Step 3: Run Training
```bash
!python kaggle_train_stair.py
```

### Step 4: Download Model
Download `stair_lora_adapter.zip` from the Kaggle Output pane and unzip into your local repository root.

---

## ⚙️ Hyperparameters (Matching IBM STAIR Paper)

| Parameter | Value | Rationale |
| :--- | :--- | :--- |
| **Base Model** | `unsloth/llama-3.2-3b-instruct-bnb-4bit` | Fits in ~4-6GB VRAM on T4, leaving 10GB+ headroom |
| **LoRA Rank ($r$)** | `16` | Standard STAIR paper rank |
| **LoRA Alpha ($\alpha$)** | `32` | Standard 2x scaling ratio |
| **Target Modules** | `q, k, v, o, gate, up, down` | Full attention + MLP projections |
| **Batch Size** | `2` | Minimal peak memory per step |
| **Gradient Accumulation** | `4` | Effective batch size of 8 |
| **Learning Rate** | `2e-4` | Cosine learning rate decay with 5% warmup |
| **Precision** | `FP16` | Optimized for NVIDIA Turing (T4) Tensor Cores |
| **Training Loss Objective** | Completion-Only | Supervised solely on `### Section: leaf <name>` |

---

## 📊 Local Verification After Download

Once `stair_lora_adapter` is placed in the project root:

```bash
# Verify the adapter is loaded and run the evaluation benchmark
python src/5_evaluate_benchmark.py

# Launch the comparison dashboard
streamlit run app.py
```

The dashboard will detect the neural adapter, initialize the **Trie-Constrained Logits Processor**, and execute real neural inference with 0.00% section hallucination!
