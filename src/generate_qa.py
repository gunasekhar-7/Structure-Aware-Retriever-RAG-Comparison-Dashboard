"""
Generate QA Module Entrypoint.
Enables standard Python module imports (from src.generate_qa import main).
"""

import sys
from pathlib import Path
import importlib.util

_current_dir = Path(__file__).resolve().parent
_target_file = _current_dir / "2_generate_qa.py"

_spec = importlib.util.spec_from_file_location("generate_qa_impl", str(_target_file))
_module = importlib.util.module_from_spec(_spec)
sys.modules["generate_qa_impl"] = _module
_spec.loader.exec_module(_module)

main = _module.main
query_groq_api = _module.query_groq_api
load_toc_and_leaves = _module.load_toc_and_leaves
format_stair_training_example = _module.format_stair_training_example
