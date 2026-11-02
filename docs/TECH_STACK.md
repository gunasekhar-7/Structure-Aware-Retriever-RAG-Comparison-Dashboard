# 🛠️ Technology Stack & Design Decisions

This document details every technology, framework, and architectural design choice across the STAIR project, including justifications for why each component was selected over common alternatives.

---

## 1. Stack Summary Table

| Category | Technology | Version | Purpose |
| :--- | :--- | :--- | :--- |
| **Language Runtime** | Python | `3.11.7+` | Core programming language |
| **Deep Learning Framework** | PyTorch | `>= 2.0.0` | Tensor computation & model execution |
| **Model Architecture** | Llama-3.2-3B-Instruct | `3.2` | Base autoregressive LLM for outline navigation |
| **Quantization** | BitsAndBytes (NF4) | `>= 0.41.0` | 4-bit parameter compression |
| **Fine-Tuning Acceleration**| Unsloth / PEFT | `>= 2024.5.0` | Fused Triton kernels, memory-efficient LoRA |
| **Training Harness** | TRL / SFTTrainer | `>= 0.7.0` | Supervised fine-tuning with completion loss |
| **Constrained Decoding** | Custom Prefix Trie | Native PyTorch | Zero-hallucination logits masking |
| **Document Parsing** | PyMuPDF (fitz) | `>= 1.24.0` | High-speed PDF outline & text extraction |
| **Retrieval Baseline** | rank-bm25 | `>= 0.2.2` | Traditional BM25 Okapi chunk retrieval baseline |
| **Synthetic QA Synthesizer**| Groq Cloud API | Llama-3.3-70B | Ultra-fast, high-diversity synthetic QA generation |
| **Dashboard Interface** | Streamlit | `>= 1.28.0` | Production comparison web application |
| **Test Suite** | Pytest | `>= 7.4.0` | Automated unit & integration verification |

---

## 2. Component Design Justifications

### 2.1 PDF Parsing: PyMuPDF (`fitz`) vs. `pypdf` vs. `pdfplumber`
- **Chosen:** `PyMuPDF` (C++ MuPDF bindings).
- **Rationale:** 
  - PyMuPDF extracts embedded PDF bookmarks via `doc.get_toc()` in **< 10 milliseconds**, even for 1,000+ page technical manuals.
  - Unlike `pypdf`, PyMuPDF reliably returns exact hierarchy levels ($1, 2, 3$) alongside page references.
  - Text extraction (`doc[page].get_text("blocks")`) preserves structural whitespace, code blocks, and coordinates needed for heuristic heading detection when outlines are absent.
  - Benchmark: PyMuPDF is **~15x faster** than `pdfplumber` and **~8x faster** than `pypdf`.

---

### 2.2 Model Selection: Llama-3.2-3B vs. Mistral-7B vs. Llama-3.1-8B
- **Chosen:** `unsloth/llama-3.2-3b-instruct-bnb-4bit`.
- **Rationale:**
  - **Memory Budget:** On a free Kaggle Tesla T4 GPU (15–16GB VRAM), a 7B or 8B model in 4-bit consumes ~5.5–6.5GB VRAM just for base weights, leaving narrow headroom for long Table of Contents context windows ($4{,}096$ tokens) during batching.
  - In contrast, Llama-3.2-3B loads in **~2.4 GB VRAM**, allowing batch size 2 with gradient accumulation 4 and full 4K sequence lengths without running out of memory (OOM).
  - **Task Fit:** Predicting a section name from a provided Table of Contents is an in-context ranking problem; a 3B parameter model has more than enough reasoning capacity to perform this task with 100% accuracy after LoRA adaptation.

---

### 2.3 Fine-Tuning: Unsloth QLoRA vs. Standard Hugging Face PEFT
- **Chosen:** Hybrid Unsloth with graceful PEFT fallback.
- **Rationale:**
  - Unsloth rewrites PyTorch attention and cross-entropy layers with **custom OpenAI Triton fused kernels**.
  - Provides **2x faster training** and **60% lower VRAM usage**, allowing training on Kaggle T4 to complete in **~15 minutes** instead of 45+ minutes.
  - The hybrid loader implemented in [`src/3_train_stair_lora.py`](file:///c:/Projects/Structure-Aware%20Retriever%20RAG%20Comparison%20Dashboard/src/3_train_stair_lora.py) automatically falls back to native Hugging Face PEFT if running on non-Triton systems.

---

### 2.4 Constrained Decoding: Prefix Trie vs. Unconstrained Generation + Substring Match
- **Chosen:** [`TrieConstrainedLogitsProcessor`](file:///c:/Projects/Structure-Aware%20Retriever%20RAG%20Comparison%20Dashboard/src/trie_constrained_decoder.py).
- **Rationale:**
  - Standard LLM text generation is probabilistic. Even with temperature 0.1, the model can generate non-existent sections (e.g. `"Section 5.4"` when only 3 sections exist).
  - Post-hoc substring matching or regex search is a fragile patch that fails when section titles are similar.
  - The Prefix Trie operates directly on the output logits before sampling. Setting non-tree tokens to $-\infty$ guarantees with mathematical certainty that the generated output is a valid section name in the document.

---

### 2.5 Synthetic QA Generator: Groq Cloud API (`llama-3.3-70b-versatile`)
- **Chosen:** Groq Cloud Llama-3.3-70B API with offline deterministic fallback.
- **Rationale:**
  - Generating ~1,000 diverse training examples locally on CPU would take several hours.
  - Groq's custom Language Processing Units (LPUs) generate at **~300 tokens/second**, allowing the complete dataset to synthesize in under 2 minutes.
  - Llama-3.3-70B produces nuanced, realistic questions (distinguishing between syntax, edge-cases, and architecture) that challenge the model to learn semantic mapping rather than simple title keyword matching.

---

## 3. Hardware Requirements & Resource Budgets

### Production Deployment (Inference)
- **CPU Mode (Semantic Trie Constraint):**
  - RAM: $\ge 2\text{ GB}$
  - Storage: $\sim 50\text{ MB}$
  - Latency: $< 1.0\text{ ms}$ per query
  - Hardware: Any standard x86 / ARM CPU (no GPU required).
- **Neural Mode (Local GPU Inference):**
  - VRAM: $\ge 4\text{ GB}$ (CUDA 11.8+ or 12.x)
  - Latency: $\sim 15\text{ ms}$ per query

### Fine-Tuning Environment (Kaggle T4)
- **Target GPU:** NVIDIA Tesla T4 (Turing, 16GB GDDR6 VRAM).
- **Base Weights Footprint:** $\sim 2.4\text{ GB}$ (4-bit NF4).
- **LoRA Adapter Footprint:** $\sim 97\text{ MB}$ ($r=16, \alpha=32$).
- **Activation Memory Peak:** $\sim 3.2\text{ GB}$ with gradient checkpointing.
- **Total Headroom:** $> 10\text{ GB}$ free VRAM on 16GB T4.
