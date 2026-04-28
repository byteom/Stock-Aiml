"""Test configuration — add stock-aiml to Python path."""
from __future__ import annotations

import sys
from pathlib import Path

# stock-aiml/tests/conftest.py → stock-aiml/ = project root where 'backend' package lives
_stock_aiml_root = Path(__file__).parents[2]  # stock-aiml/tests/ → stock-aiml
if str(_stock_aiml_root) not in sys.path:
    sys.path.insert(0, str(_stock_aiml_root))
