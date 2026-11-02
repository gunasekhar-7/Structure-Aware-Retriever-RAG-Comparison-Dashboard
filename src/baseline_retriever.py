"""
Baseline Traditional Chunked RAG Retriever (BM25 Okapi).

Demonstrates traditional chunk-based retrieval to compare against
STAIR structure-aware retrieval. Implements real chunking and BM25 scoring.
"""

from typing import List, Dict, Any, Optional
import re
import numpy as np
from rank_bm25 import BM25Okapi


class Chunk:
    """Represents a text chunk created under fixed-size windowing."""
    def __init__(self, chunk_id: int, text: str, source_section: str, start_char: int, end_char: int, is_truncated: bool):
        self.chunk_id = chunk_id
        self.text = text
        self.source_section = source_section
        self.start_char = start_char
        self.end_char = end_char
        self.is_truncated = is_truncated

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "source_section": self.source_section,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "is_truncated": self.is_truncated,
        }


class BM25ChunkedRetriever:
    """Traditional Chunked RAG retriever using fixed-size token/character windows and BM25."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunks: List[Chunk] = []
        self.bm25: Optional[BM25Okapi] = None
        self.tokenized_corpus: List[List[str]] = []

    def _tokenize(self, text: str) -> List[str]:
        """Simple, robust alphanumeric tokenization for BM25."""
        return re.findall(r"\b\w+\b", text.lower())

    def index_sections(self, leaves: Dict[str, str]) -> int:
        """
        Chunks all leaf sections using a fixed window and indexes them into BM25.
        
        Args:
            leaves: Dict mapping section_name -> section_text
            
        Returns:
            int: Total number of chunks indexed
        """
        self.chunks = []
        chunk_counter = 0

        for section_name, text in leaves.items():
            text_len = len(text)
            if text_len == 0:
                continue

            start = 0
            while start < text_len:
                end = min(start + self.chunk_size, text_len)
                chunk_text = text[start:end]
                
                # A chunk is considered boundary-truncated if it doesn't cover the full section
                is_truncated = (start > 0 or end < text_len)
                
                chunk = Chunk(
                    chunk_id=chunk_counter,
                    text=chunk_text,
                    source_section=section_name,
                    start_char=start,
                    end_char=end,
                    is_truncated=is_truncated
                )
                self.chunks.append(chunk)
                chunk_counter += 1

                if end >= text_len:
                    break
                start += self.chunk_size - self.chunk_overlap

        # Build BM25 Okapi index
        self.tokenized_corpus = [self._tokenize(c.text) for c in self.chunks]
        if self.tokenized_corpus:
            self.bm25 = BM25Okapi(self.tokenized_corpus)
        
        return len(self.chunks)

    def retrieve(self, query: str, top_k: int = 1) -> List[Dict[str, Any]]:
        """
        Retrieve top-k chunks for a user query using BM25 Okapi.
        
        Args:
            query: User's search query
            top_k: Number of chunks to retrieve
            
        Returns:
            List of dicts containing retrieved chunk metadata and score
        """
        if not self.bm25 or not self.chunks:
            return []

        tokenized_query = self._tokenize(query)
        if not tokenized_query:
            # Fallback if query has no alphanumeric tokens
            return [self.chunks[0].to_dict()] if self.chunks else []

        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        max_possible_score = max(float(scores[top_indices[0]]), 1e-6)
        
        for idx in top_indices:
            raw_score = float(scores[idx])
            chunk = self.chunks[idx]
            # Normalize score to 0.0 - 1.0 range for UI comparison
            normalized_score = min(max(raw_score / (max_possible_score + 1.0), 0.0), 1.0)
            
            results.append({
                "chunk_id": chunk.chunk_id,
                "section": chunk.source_section,
                "text": chunk.text,
                "raw_score": raw_score,
                "score": normalized_score,
                "is_truncated": chunk.is_truncated,
                "method": f"BM25 ({self.chunk_size}-char chunks, {self.chunk_overlap} overlap)"
            })

        return results

