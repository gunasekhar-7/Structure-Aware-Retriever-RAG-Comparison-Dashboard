"""
Unit & Integration Test Suite for Structure-Aware Retriever (STAIR).
"""

import sys
from pathlib import Path
import pytest
import json

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.trie_constrained_decoder import LeafTrie, TrieNode
from src.baseline_retriever import BM25ChunkedRetriever
from src.inference_engine import STAIRInferenceEngine


class MockTokenizer:
    """Deterministic offline tokenizer for Trie testing."""
    def __init__(self):
        self.eos_token_id = 999
        self.pad_token_id = 999
        self.vocab = {}
        self.rev_vocab = {}

    def encode(self, text: str, add_special_tokens: bool = False):
        tokens = text.lower().split()
        ids = []
        for t in tokens:
            if t not in self.vocab:
                new_id = len(self.vocab) + 1
                self.vocab[t] = new_id
                self.rev_vocab[new_id] = t
            ids.append(self.vocab[t])
        return ids


def test_trie_construction_and_allowed_tokens():
    """Verify prefix tree strictly constrains candidate token IDs."""
    tokenizer = MockTokenizer()
    leaf_names = [
        "Section 1.1: Syntax",
        "Section 1.2: Iterators",
        "Section 2.1: Collections"
    ]
    trie = LeafTrie(tokenizer=tokenizer, leaf_names=leaf_names, prefix="leaf ")

    # Root allowed tokens must correspond to 'leaf'
    leaf_token_id = tokenizer.vocab["leaf"]
    root_allowed = trie.get_allowed_next_tokens([])
    assert leaf_token_id in root_allowed
    assert len(root_allowed) == 1

    # Following 'leaf', next allowed must be 'section'
    sec_allowed = trie.get_allowed_next_tokens([leaf_token_id])
    assert tokenizer.vocab["section"] in sec_allowed

    # Following 'leaf section', allowed must be 1.1, 1.2, 2.1
    sub_allowed = trie.get_allowed_next_tokens([leaf_token_id, tokenizer.vocab["section"]])
    expected_ids = {tokenizer.vocab["1.1:"], tokenizer.vocab["1.2:"], tokenizer.vocab["2.1:"]}
    assert expected_ids.issubset(set(sub_allowed))

    # An invalid token sequence must diverge to EOS
    diverged = trie.get_allowed_next_tokens([99999])
    assert diverged == [trie.eos_token_id]


def test_bm25_chunking_and_retrieval():
    """Verify BM25 chunking preserves boundaries and scores realistically."""
    sample_leaves = {
        "SecA": "Python async programming using coroutines and task groups for high concurrency.",
        "SecB": "Database connection pooling and SQL queries for relational data access."
    }
    retriever = BM25ChunkedRetriever(chunk_size=50, chunk_overlap=10)
    num_chunks = retriever.index_sections(sample_leaves)
    assert num_chunks >= 2

    # Query for async
    results = retriever.retrieve("coroutine task groups", top_k=1)
    assert len(results) == 1
    assert results[0]["section"] == "SecA"
    assert results[0]["score"] > 0.0

    # Query for database
    results_db = retriever.retrieve("SQL connection database", top_k=1)
    assert len(results_db) == 1
    assert results_db[0]["section"] == "SecB"


def test_stair_inference_zero_hallucination():
    """Verify STAIR inference strictly returns valid leaf sections."""
    engine = STAIRInferenceEngine()
    test_queries = [
        "How do task groups handle exceptions?",
        "What are the time complexities of dictionaries and sets?",
        "How does dependency injection work with FastAPI?"
    ]
    for q in test_queries:
        res = engine.predict_leaf(q)
        assert res["is_valid"] is True
        assert res["predicted_leaf"] in engine.leaves
        assert res["hallucination_rate"] == 0.0
        assert res["latency_ms"] >= 0.0


def test_stair_toc_breadcrumb_path():
    """Verify ToC breadcrumb generator parses section hierarchy correctly."""
    engine = STAIRInferenceEngine()
    breadcrumb = engine.get_toc_tree("Section 3.2: Task Groups")
    assert "Chapter 3" in breadcrumb
    assert "Section 3.2" in breadcrumb
