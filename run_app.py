"""Launch the warpage-analysis Dash app.

    python run_app.py

Serves on http://127.0.0.1:8050 by default.
"""

import os
import sys

_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from matrix2d.ui.app import create_app  # noqa: E402


def main():
    app = create_app()
    app.run(debug=True)


if __name__ == "__main__":
    main()
