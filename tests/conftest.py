"""Ensure ``src`` is importable during tests.

Created unconditionally (harmless even if a pytest.ini with pythonpath=src
already exists).
"""

import os
import sys

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
