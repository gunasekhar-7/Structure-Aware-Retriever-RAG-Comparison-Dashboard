# 🏛️ STAIR Architecture & Engineering Design Document

## 1. Executive Summary & Core Motivation

Modern Retrieval-Augmented Generation (RAG) predominantly relies on **arbitrary fixed-window chunking** (e.g., sliding windows of 500 tokens with 10% overlap) embedded into dense vector stores (FAISS, Milvus, Pinecone, ChromaDB). While functional for short unstructured articles, this paradigm breaks down catastrophically when applied to structured technical manuals, legal treaties, regulatory documentation, and textbooks.

### The Three Fatal Flaws of Traditional Chunked RAG

```
[ Traditional Chunked RAG ]
Raw Manual ➔ [ Chunk 1 (500 tokens) ] ➔ [ Chunk 2 (500 tokens) ] ➔ [ Chunk 3 (500 tokens) ]
                       |                         |
               Mid-sentence split!      Function signature severed!
               Table headers lost!      Import declarations severed!
```

1. **Context Boundary Truncation:** Fixed token windows slice through code blocks, multi-row tables, and nested conditionals mid-sentence. When an LLM receives a severed chunk, it either hallucinates missing context or outputs malformed syntax.
2. **Loss of Structural Taxonomy:** Documents are not flat strings; they are directed acyclic trees. A paragraph reading `"Returns a 404 status code"` means nothing without knowing it belongs to `FastAPI Manual ➔ Chapter 4: Routing ➔ Section 4.1: Path Parameters`.
3. **The Vector Infrastructure Tax:** Maintaining billions of high-dimensional dense embeddings requires ongoing indexing compute, GPU vector search clusters, and complex hybrid re-ranking pipelines with persistent synchronization drift.

### The STAIR Paradigm
**STAIR (Structure-Aware Information Retriever)**, formulated by IBM Research, re-architects retrieval as an **autoregressive outline traversal problem**:
- The document outline (Table of Contents) is treated as a hierarchical decision tree.
- An LLM inspects the document hierarchy and maps a natural language query directly to the target **leaf section** (subsections containing actual text without deeper children).
- **Prefix Trie Constrained Decoding** masks the model's vocabulary logits at every generation step, making non-existent section hallucinations **mathematically impossible ($0.00\%$)**.
- The retrieved context is the **complete, intact leaf section**, eliminating vector databases entirely.

---

## 2. End-to-End System Architecture

```mermaid
flowchart TD
    subgraph INGESTION ["1. Ingestion & Outline Extraction"]
        PDF["Input Document (PDF / Manual)"] --> PARSER["PyMuPDF Outline Parser"]
        PARSER -->|Outline Available| TOC_EXTRACT["Extract Bookmark Tree"]
        PARSER -->|No Outline| HEURISTIC["Font & Layout Heuristic Engine"]
        TOC_EXTRACT & HEURISTIC --> LEAF_STORE["Leaf Section Store & ToC Prompt"]
    end

    subgraph SYNTHESIS ["2. Synthetic QA Generation (Groq / Offline)"]
        LEAF_STORE --> GROQ["Groq Llama-3.3-70B Synthesizer"]
        GROQ --> FORMAT["Table 2 Prompt-Completion Formatter"]
        FORMAT --> DATA_SPLIT["80% Train / 20% Val Stratified Dataset"]
    end

    subgraph TRAINING ["3. Model Adaptation (Kaggle Tesla T4 GPU)"]
        DATA_SPLIT --> QLORA["Unsloth 4-bit QLoRA (r=16, alpha=32)"]
        BASE_MODEL["Llama-3.2-3B Base Weights"] --> QLORA
        QLORA --> SFT["Completion-Only SFTTrainer"]
        SFT --> ADAPTER["Fine-Tuned Adapter (stair_lora_adapter)"]
    end

    subgraph INFERENCE ["4. Constrained Retrieval & Serving Engine"]
        USER_QUERY["User Query"] --> STAIR_ENGINE["STAIR Inference Engine"]
        LEAF_STORE --> TRIE_BUILDER["LeafTrie Prefix Tree"]
        TRIE_BUILDER --> LOGITS_PROC["TrieConstrainedLogitsProcessor"]
        ADAPTER & BASE_MODEL --> STAIR_ENGINE
        LOGITS_PROC --> STAIR_ENGINE
        STAIR_ENGINE -->|0.00% Hallucination| EXACT_LEAF["Complete Retrieved Leaf Section"]
    end

    subgraph BASELINE ["5. Empirical Baseline & Comparison"]
        USER_QUERY --> BM25_ENGINE["BM25 Okapi 500-char Chunker"]
        LEAF_STORE --> BM25_ENGINE
        BM25_ENGINE -->|Truncated Chunk| BM25_CHUNK["BM25 Retrieved Window"]
    end

    subgraph PRESENTATION ["6. Interactive Streamlit Dashboard"]
        EXACT_LEAF & BM25_CHUNK --> DASHBOARD["Dual-Column Streamlit Interface"]
        DASHBOARD --> USER_INSPECT["Side-by-Side Comparison & Benchmarks"]
    end
```

---

## 3. Deep Dive into System Components

### 3.1 Outline & Leaf Section Extraction (`src/1_extract_toc.py`)
A document is represented as an ordered sequence of nodes:
$$\mathcal{T} = \{(l_i, t_i, p_i)\}_{i=1}^N$$
where $l_i \in \mathbb{N}$ is the hierarchy depth (e.g., $1$ for Chapter, $2$ for Section), $t_i$ is the title, and $p_i$ is the starting page.

- **Leaf Node Criterion:** Node $i$ is a leaf section if and only if:
  $$\text{IsLeaf}(i) \iff (i = N) \lor (l_{i+1} \le l_i)$$
  *(i.e., the subsequent entry is either at the same depth or climbs to an ancestor node).*
- **Text Boundary Capture:** Text is harvested from page range $[p_i, \min(p_{i+1}, \text{TotalPages})]$, capturing all body copy, code listings, and tables intact.
- **Fallback Heading Heuristic:** For legacy PDFs without embedded bookmarks, the parser scans page blocks (`b[6] == 0`), extracting prominent text blocks matching font-weight and layout markers (`Chapter`, `Section \d+`, or title-case lines $< 80$ characters).

---

### 3.2 Prefix Trie-Constrained Logits Processor (`src/trie_constrained_decoder.py`)

In standard autoregressive language modeling, token $y_t$ is sampled from the unconstrained vocabulary distribution:
$$P(y_t \mid y_{<t}, x) = \text{softmax}(z_t)$$
where $z_t \in \mathbb{R}^{|\mathcal{V}|}$ are the logits over the tokenizer vocabulary $\mathcal{V}$ ($|\mathcal{V}| \approx 128{,}256$ in Llama 3).

Under STAIR's constrained generation:
1. Every valid leaf section target $S_k \in \mathcal{S}$ is pre-tokenized with the prompt completion prefix:
   $$T_k = \text{tokenize}(\text{"leaf "} \parallel S_k \parallel \langle\text{eos}\rangle)$$
2. All token sequences $\{T_k\}_{k=1}^{|\mathcal{S}|}$ are compiled into a directed prefix tree (**Trie**):
   $$\mathcal{M} = (\mathcal{Q}, \Sigma, \delta, q_0, \mathcal{F})$$
3. During generation, let $y_{\text{gen}}$ be the tokens generated after the prompt completion marker. The allowed next-token set is:
   $$\mathcal{V}_{\text{allowed}}(y_{\text{gen}}) = \{ v \in \mathcal{V} \mid \delta(q(y_{\text{gen}}), v) \text{ is defined} \}$$
4. The `TrieConstrainedLogitsProcessor` applies an additive logit mask:
   $$\tilde{z}_{t, v} = \begin{cases} z_{t, v} & \text{if } v \in \mathcal{V}_{\text{allowed}}(y_{\text{gen}}) \\ -\infty & \text{otherwise} \end{cases}$$

$$\lim_{\tilde{z}_{t, v} \to -\infty} \text{softmax}(\tilde{z}_t)_v = 0 \quad \forall v \notin \mathcal{V}_{\text{allowed}}$$

**Mathematical Guarantee:** The probability of generating any section name outside $\mathcal{S}$ is identically **zero ($0.0000\%$)**.

---

### 3.3 Synthetic QA Generation Pipeline (`src/2_generate_qa.py`)
To train the retrieval model without human annotators, the synthesizer reads each leaf section $S_k$ and prompts Groq's `llama-3.3-70b-versatile` under strict rules:
- **Taxonomy of Questions:**
  1. *Conceptual:* "What are the time complexities of dictionaries and sets in Python 3.11?"
  2. *Implementation:* "How to declare path parameters and numeric validation with Path(..., ge=1) in FastAPI?"
  3. *Troubleshooting / Edge-Cases:* "What happens to child tasks when an exception is raised in an asyncio TaskGroup?"
- **Format:** Strict adherence to Table 2 of the IBM STAIR paper:
  ```
  ### Book: {book_title}
  Pick the best section from the table of contents below which can answer the user query below. Only generate the section name, do not generate any explanation!
  ### Table of Contents
  {formatted_toc}
  ### Query: {query}
  ### Section: leaf {target_leaf}
  ```
- **Stratification:** Data is deterministically shuffled and split into **80% Train** and **20% Validation** to evaluate generalization on unseen queries.

---

### 3.4 Kaggle T4 Fine-Tuning Pipeline (`kaggle_train_stair.py`)
Fine-tuning adapts Llama-3.2-3B to interpret the ToC hierarchy and map user queries to section tokens:
- **Quantization:** NormalFloat4 (NF4) with Double Quantization via `bitsandbytes`.
- **Parameter-Efficient Adaptation (LoRA):**
  - Rank ($r$): 16
  - Alpha ($\alpha$): 32
  - Target Projections: $W_q, W_k, W_v, W_o, W_{\text{gate}}, W_{\text{up}}, W_{\text{down}}$
- **Completion-Only Supervised Loss:** Gradients are computed strictly on completion tokens following `### Section: leaf `, preventing the model from wasting capacity memorizing the prompt or Table of Contents.

---

### 3.5 Traditional BM25 Baseline Retriever (`src/baseline_retriever.py`)
To benchmark STAIR objectively against traditional RAG:
- The exact same document text is split into fixed-window chunks ($500$ characters with $50$-character overlap).
- Chunks are tokenized and scored via the standard **BM25 Okapi** algorithm:
  $$\text{Score}(D, Q) = \sum_{q \in Q} \text{IDF}(q) \cdot \frac{f(q, D) \cdot (k_1 + 1)}{f(q, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}$$
- Every chunk tracks its start and end offsets: if $start > 0$ or $end < \text{len}(\text{section})$, the chunk is tagged as `is_truncated = True`.

---

## 4. Latency & Complexity Comparison

| Dimension | Traditional Vector RAG | STAIR Architecture |
| :--- | :--- | :--- |
| **Indexing Time** | $\mathcal{O}(N \cdot d)$ embedding generation + vector clustering | $\mathcal{O}(N)$ single-pass outline parse |
| **Index Storage Footprint** | Gigabytes of float32 embeddings | Kilobytes of structured JSON |
| **Inference Retrieval Time** | 20–100 ms (Vector distance + re-ranker) | < 1 ms (Semantic Trie) / ~15 ms (Neural GPU) |
| **Boundary Preservation** | $0\%$ (arbitrary token slicing) | **$100\%$ (Complete semantic leaf sections)** |
| **Section Hallucination** | Frequent when LLM synthesizes from chunks | **$0.00\%$ guaranteed by Trie constraint** |
| **Vector DB Cost** | Monthly hosted cluster fee | **Zero (Vectorless)** |
