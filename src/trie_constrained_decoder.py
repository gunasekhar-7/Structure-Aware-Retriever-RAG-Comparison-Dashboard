"""
Trie-Based Constrained Logits Processor for STAIR Architecture.

Guarantees 0.00% section hallucination during LLM decoding by restricting
next-token candidates strictly to valid paths in the document Table of Contents Trie.
"""

from typing import Dict, List, Optional, Set
import torch
from transformers import LogitsProcessor, PreTrainedTokenizer


class TrieNode:
    """A node in the prefix Trie representing allowed token sequences."""
    def __init__(self):
        self.children: Dict[int, "TrieNode"] = {}
        self.is_terminal: bool = False
        self.leaf_name: Optional[str] = None


class LeafTrie:
    """Prefix tree (Trie) built over the vocabulary tokens of valid leaf sections."""
    
    def __init__(self, tokenizer: PreTrainedTokenizer, leaf_names: List[str], prefix: str = "leaf "):
        self.tokenizer = tokenizer
        self.prefix = prefix
        self.leaf_names = leaf_names
        self.root = TrieNode()
        self.eos_token_id = tokenizer.eos_token_id or tokenizer.pad_token_id or 0
        self._build_trie()

    def _build_trie(self) -> None:
        """Encodes each valid leaf section name into token IDs and inserts into Trie."""
        for name in self.leaf_names:
            clean_name = name.strip()
            # Handle whether the leaf title already has the prefix or not
            if clean_name.startswith(self.prefix):
                target_str = clean_name
            else:
                target_str = f"{self.prefix}{clean_name}"
            
            # Encode target string into token IDs
            # Strip BOS token if tokenizer prepends it, because this is a completion
            token_ids = self.tokenizer.encode(target_str, add_special_tokens=False)
            
            # Append EOS token to mark valid completion boundary
            token_ids.append(self.eos_token_id)
            
            # Insert into Trie
            node = self.root
            for token_id in token_ids:
                if token_id not in node.children:
                    node.children[token_id] = TrieNode()
                node = node.children[token_id]
            node.is_terminal = True
            node.leaf_name = clean_name

    def get_allowed_next_tokens(self, generated_tokens: List[int]) -> List[int]:
        """Traverse the Trie given already generated tokens and return allowed next tokens."""
        node = self.root
        for token_id in generated_tokens:
            if token_id not in node.children:
                # Sequence diverged (should not happen if constrained throughout)
                return [self.eos_token_id]
            node = node.children[token_id]
        
        # If node has children, return all allowed next token IDs
        if node.children:
            return list(node.children.keys())
        
        # If terminal reached, only EOS is allowed
        return [self.eos_token_id]


class TrieConstrainedLogitsProcessor(LogitsProcessor):
    """
    HuggingFace-compatible LogitsProcessor that enforces strict Trie-constrained generation.
    Masks out all vocabulary tokens that do not lead to a valid leaf section name.
    """

    def __init__(self, trie: LeafTrie, prompt_token_lens: List[int]):
        super().__init__()
        self.trie = trie
        self.prompt_token_lens = prompt_token_lens
        self.eos_token_id = trie.eos_token_id

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        batch_size = input_ids.shape[0]
        
        # Create -inf mask
        mask = torch.full_like(scores, fill_value=-float("inf"))
        
        for batch_idx in range(batch_size):
            prompt_len = self.prompt_token_lens[batch_idx] if batch_idx < len(self.prompt_token_lens) else self.prompt_token_lens[0]
            # Tokens generated AFTER the prompt
            generated_tokens = input_ids[batch_idx, prompt_len:].tolist()
            
            allowed_tokens = self.trie.get_allowed_next_tokens(generated_tokens)
            if not allowed_tokens:
                allowed_tokens = [self.eos_token_id]
                
            mask[batch_idx, allowed_tokens] = 0.0
            
        return scores + mask
