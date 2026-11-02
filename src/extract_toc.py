"""
Extract ToC Module Entrypoint.
Enables standard Python module imports (from src.extract_toc import main).
"""

import sys
from pathlib import Path
import importlib.util

_current_dir = Path(__file__).resolve().parent
_target_file = _current_dir / "1_extract_toc.py"

_spec = importlib.util.spec_from_file_location("extract_toc_impl", str(_target_file))
_module = importlib.util.module_from_spec(_spec)
sys.modules["extract_toc_impl"] = _module
_spec.loader.exec_module(_module)

main = _module.main
extract_stair_toc = _module.extract_stair_toc
