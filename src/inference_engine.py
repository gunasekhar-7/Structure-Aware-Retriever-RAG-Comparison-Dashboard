"""
STAIR Inference Engine Module Entrypoint.
Enables standard Python module imports (from src.inference_engine import STAIRInferenceEngine).
"""

import sys
from pathlib import Path
import importlib.util

_current_dir = Path(__file__).resolve().parent
_engine_file = _current_dir / "4_inference_engine.py"

_spec = importlib.util.spec_from_file_location("inference_engine_impl", str(_engine_file))
_module = importlib.util.module_from_spec(_spec)
sys.modules["inference_engine_impl"] = _module
_spec.loader.exec_module(_module)

STAIRInferenceEngine = _module.STAIRInferenceEngine
LeafTrie = _module.LeafTrie
TrieConstrainedLogitsProcessor = _module.TrieConstrainedLogitsProcessor

