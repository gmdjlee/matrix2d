"""Executable entry point (for PyInstaller builds).

Same as run_app.py but tuned for a packaged single-file exe:
  - debug/reloader always OFF (reloader spawns a 2nd process → breaks a
    frozen exe and kills background scan/compute threads)
  - opens the default browser automatically once the server is up

Build with packaging/warpage.spec (see docs/build-exe.md).
"""

import os
import sys
import threading
import webbrowser


def _resource_root():
    """Folder that holds bundled data at runtime.

    PyInstaller onefile extracts everything under sys._MEIPASS; in a normal
    (non-frozen) run this is the repo root.
    """
    if getattr(sys, "frozen", False):
        return sys._MEIPASS  # type: ignore[attr-defined]
    return os.path.dirname(os.path.abspath(__file__))


# Make `import matrix2d` work both frozen and unfrozen.
if not getattr(sys, "frozen", False):
    _SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
    if _SRC not in sys.path:
        sys.path.insert(0, _SRC)


def main():
    from matrix2d.logging_setup import setup_logging
    from matrix2d.ui.app import create_app

    log_path = setup_logging()
    import logging
    logger = logging.getLogger("app_main")

    port = int(os.environ.get("PORT", "8050"))
    url = "http://127.0.0.1:{0}".format(port)

    app = create_app()
    logger.info("Starting packaged app on %s (log: %s)", url, log_path)

    # Open the browser shortly after the server starts. Guard so it fires once.
    if os.environ.get("MATRIX2D_NO_BROWSER", "0") != "1":
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()

    # debug=False → no werkzeug reloader, no hot reload.
    app.run(debug=False, port=port)


if __name__ == "__main__":
    main()
