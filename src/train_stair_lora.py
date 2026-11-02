"""
Train STAIR LoRA Module Entrypoint.
Enables standard Python module imports (from src.train_stair_lora import main).
"""

import sys
from pathlib import Path
import importlib.util

_current_dir = Path(__file__).resolve().parent
_target_file = _current_dir / "3_train_stair_lora.py"

_spec = importlib.util.spec_from_file_location("train_stair_lora_impl", str(_target_file))
_module = importlib.util.module_from_spec(_spec)
sys.modules["train_stair_lora_impl"] = _module
_spec.loader.exec_module(_module)

main = _module.main
train_stair_lora = _module.train_stair_lora

