"""App-wide logging configuration: rotating log file + console.

Called once from the app entry point (run_app.py). Routes every logger in the
package (services.pipeline job errors, repository scan warnings, ui.dialogs
warnings, ui.callbacks worker/thread events) plus Python warnings and uncaught
thread exceptions into a single rotating log file, so background-thread
failures are never silently lost.

Environment overrides:
    MATRIX2D_LOG_DIR    log directory (default: <repo>/logs)
    MATRIX2D_LOG_LEVEL  root level name, e.g. DEBUG/INFO (default: INFO)
"""

import logging
import logging.handlers
import os
import sys
import threading
from typing import Optional

_DEFAULT_LOG_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "logs"))

_FORMAT = "%(asctime)s %(levelname)-8s [%(threadName)s] %(name)s: %(message)s"

_configured = False


def setup_logging(log_dir: Optional[str] = None,
                  level: Optional[str] = None) -> str:
    """Configure root logging to a rotating file and the console.

    Safe to call more than once (subsequent calls are no-ops). Returns the
    log file path.
    """
    global _configured

    log_dir = log_dir or os.environ.get("MATRIX2D_LOG_DIR", _DEFAULT_LOG_DIR)
    level_name = (level or os.environ.get("MATRIX2D_LOG_LEVEL", "INFO")).upper()
    log_path = os.path.join(log_dir, "matrix2d.log")

    if _configured:
        return log_path

    os.makedirs(log_dir, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(getattr(logging, level_name, logging.INFO))

    formatter = logging.Formatter(_FORMAT)

    file_handler = logging.handlers.RotatingFileHandler(
        log_path, maxBytes=2 * 1024 * 1024, backupCount=5, encoding="utf-8")
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    root.addHandler(console)

    # Python warnings (DeprecationWarning etc.) -> 'py.warnings' logger.
    logging.captureWarnings(True)

    # Werkzeug logs every HTTP request at INFO; the 300-400ms progress polls
    # would flood the file, so keep only its warnings/errors.
    logging.getLogger("werkzeug").setLevel(logging.WARNING)

    # Uncaught exceptions in background threads (scan/compute workers) must
    # land in the log file instead of dying silently on stderr.
    def _thread_excepthook(args):
        logging.getLogger("threading").error(
            "Uncaught exception in thread %r",
            args.thread.name if args.thread else "?",
            exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
        )

    threading.excepthook = _thread_excepthook

    # Uncaught exceptions in the main thread.
    def _sys_excepthook(exc_type, exc_value, exc_tb):
        if not issubclass(exc_type, KeyboardInterrupt):
            logging.getLogger("main").error(
                "Uncaught exception",
                exc_info=(exc_type, exc_value, exc_tb))
        sys.__excepthook__(exc_type, exc_value, exc_tb)

    sys.excepthook = _sys_excepthook

    _configured = True
    logging.getLogger(__name__).info(
        "Logging configured -> %s (level %s)", log_path, level_name)
    return log_path
