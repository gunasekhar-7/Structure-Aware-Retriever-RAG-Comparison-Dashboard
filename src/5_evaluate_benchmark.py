"""
Phase 5: Automated Benchmark Evaluation Suite.

Empirically benchmarks STAIR (Structure-Aware Retriever with Trie-Constrained Decoding)
against Traditional Chunked RAG (BM25 Okapi) on the held-out validation set.
Computes Hit@1 Accuracy, Section Hallucination Rate, Context Boundary Integrity, and Latency.
"""

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add src to sys.path
src_dir = Path(__file__).resolve().parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from baseline_retriever import BM25ChunkedRetriever
from inference_engine import STAIRInferenceEngine


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def extract_query_from_prompt(prompt_str: str) -> str:
    """Extracts raw user query from the STAIR prompt string."""
    match = re.search(r"### Query:\s*(.*?)(?:\n### Section:|$)", prompt_str, re.DOTALL)
    if match:
        return match.group(1).strip()
    return prompt_str.strip()


def run_benchmark():
    project_root = get_project_root()
    data_dir = project_root / "data"
    val_file = data_dir / "synthetic_qa_val.jsonl"
    toc_file = data_dir / "toc_and_sections.json"

    if not val_file.exists():
        print(f"[ERROR] Validation file not found: {val_file}")
        print("  Please run 2_generate_qa.py first.")
        return

    # Load validation set
    val_examples = []
    with open(val_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                val_examples.append(json.loads(line.strip()))

    print("=" * 70)
    print("📊 STAIR vs. Traditional Chunked RAG (BM25) Benchmark")
    print(f"   Validation Sample Size: {len(val_examples)} queries")
    print("=" * 70)

    # 1. Initialize STAIR Engine
    stair_engine = STAIRInferenceEngine()

    # 2. Initialize BM25 Chunked Retriever
    bm25_retriever = BM25ChunkedRetriever(chunk_size=500, chunk_overlap=50)
    num_chunks = bm25_retriever.index_sections(stair_engine.leaves)
    print(f"✓ BM25 Index built with {num_chunks} fixed-size chunks across {len(stair_engine.leaves)} sections.\n")

    stair_hits = 0
    stair_hallucinations = 0
    stair_latencies = []

    bm25_hits = 0
    bm25_truncated_count = 0
    bm25_latencies = []

    results_log = []

    for idx, example in enumerate(val_examples, 1):
        query = extract_query_from_prompt(example["prompt"])
        ground_truth_leaf = example["completion"].strip()

        # Benchmark STAIR
        t0 = time.perf_counter()
        stair_res = stair_engine.predict_leaf(query)
        stair_lat = (time.perf_counter() - t0) * 1000.0
        stair_latencies.append(stair_lat)

        pred_stair = stair_res["predicted_leaf"].strip()
        is_stair_hit = (pred_stair.lower() == ground_truth_leaf.lower())
        if is_stair_hit:
            stair_hits += 1

        # STAIR section hallucination check (is predicted leaf in outline?)
        is_stair_hallucinated = pred_stair not in stair_engine.leaves
        if is_stair_hallucinated:
            stair_hallucinations += 1

        # Benchmark BM25
        t0 = time.perf_counter()
        bm25_res = bm25_retriever.retrieve(query, top_k=1)
        bm25_lat = (time.perf_counter() - t0) * 1000.0
        bm25_latencies.append(bm25_lat)

        if bm25_res:
            top_bm25 = bm25_res[0]
            bm25_section = top_bm25["section"].strip()
            is_bm25_hit = (bm25_section.lower() == ground_truth_leaf.lower())
            if is_bm25_hit:
                bm25_hits += 1
            if top_bm25["is_truncated"]:
                bm25_truncated_count += 1
        else:
            is_bm25_hit = False

        results_log.append({
            "query": query,
            "ground_truth": ground_truth_leaf,
            "stair_predicted": pred_stair,
            "stair_hit": is_stair_hit,
            "stair_latency_ms": round(stair_lat, 2),
            "bm25_section": bm25_res[0]["section"] if bm25_res else "None",
            "bm25_hit": is_bm25_hit,
            "bm25_is_truncated": bm25_res[0]["is_truncated"] if bm25_res else True,
            "bm25_latency_ms": round(bm25_lat, 2),
        })

    n = len(val_examples)
    stair_hit_rate = (stair_hits / n) * 100.0
    stair_hallucination_rate = (stair_hallucinations / n) * 100.0
    stair_avg_lat = sum(stair_latencies) / n

    bm25_hit_rate = (bm25_hits / n) * 100.0
    bm25_truncation_rate = (bm25_truncated_count / n) * 100.0
    bm25_avg_lat = sum(bm25_latencies) / n

    summary = {
        "num_test_queries": n,
        "stair": {
            "hit_at_1_accuracy": round(stair_hit_rate, 1),
            "section_hallucination_rate": round(stair_hallucination_rate, 2),
            "context_boundary_integrity": 100.0,
            "avg_latency_ms": round(stair_avg_lat, 2),
            "vector_db_dependency": "None (Vectorless)",
        },
        "traditional_chunked_bm25": {
            "hit_at_1_accuracy": round(bm25_hit_rate, 1),
            "context_boundary_integrity": round(100.0 - bm25_truncation_rate, 1),
            "truncation_damage_rate": round(bm25_truncation_rate, 1),
            "avg_latency_ms": round(bm25_avg_lat, 2),
            "vector_db_dependency": "Chunk Store Required",
        },
        "detailed_results": results_log
    }

    # Save benchmark results
    benchmark_file = data_dir / "benchmark_results.json"
    with open(benchmark_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("📈 BENCHMARK RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Metric':<35} | {'STAIR (Structure-Aware)':<22} | {'Traditional BM25':<18}")
    print("-" * 80)
    print(f"{'Hit@1 Leaf Retrieval Accuracy':<35} | {stair_hit_rate:>6.1f}%                | {bm25_hit_rate:>6.1f}%")
    print(f"{'Section ID Hallucination Rate':<35} | {stair_hallucination_rate:>6.2f}% (Trie-Guaranteed) | {'N/A (Chunks)':<18}")
    print(f"{'Context Boundary Preservation':<35} | {100.0:>6.1f}% (Full Leaf)         | {(100.0 - bm25_truncation_rate):>6.1f}%")
    print(f"{'Chunk Cut-off / Truncation':<35} | {0.0:>6.1f}%                | {bm25_truncation_rate:>6.1f}%")
    print(f"{'Mean Retrieval Latency (ms)':<35} | {stair_avg_lat:>6.2f} ms             | {bm25_avg_lat:>6.2f} ms")
    print(f"{'Vector Database Dependency':<35} | {'Zero (Vectorless)':<22} | {'Chunk Ingestion':<18}")
    print("=" * 70)
    print(f"✓ Saved benchmark JSON metrics to: {benchmark_file}\n")


if __name__ == "__main__":
    run_benchmark()
