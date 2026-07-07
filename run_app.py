"""Launch the warpage-analysis Dash app.

    python run_app.py

Serves on http://127.0.0.1:8050 by default.
"""

import os
import sys

_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from matrix2d.logging_setup import setup_logging  # noqa: E402
from matrix2d.ui.app import create_app  # noqa: E402


def main():
    log_path = setup_logging()
    import logging
    logger = logging.getLogger("run_app")

    # Debug mode is opt-in (MATRIX2D_DEBUG=1). It runs the werkzeug reloader
    # (a second process that can restart mid-compute, killing the background
    # scan/compute worker threads) and hot reload (which refreshes the page
    # and wipes dcc.Store state) — both break long-running background work.
    debug = os.environ.get("MATRIX2D_DEBUG", "0") == "1"

    app = create_app()
    port = int(os.environ.get("PORT", "8050"))
    logger.info("Starting Dash app on port %s (debug=%s, log: %s)",
                port, debug, log_path)
    app.run(debug=debug, port=port)


if __name__ == "__main__":
    main()
