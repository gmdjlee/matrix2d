"""Native folder-picker dialog for the local single-user Dash app.

tkinter is used because Dash runs in a browser, which cannot expose a real
filesystem path from a folder picker. This is acceptable here: the app is a
local desktop tool, so the dialog opens on the same machine as the browser.
A fresh Tk root is created and destroyed per call, which is safe from a Dash
callback thread on Windows.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def pick_folder(initial_dir: str = "") -> Optional[str]:
    """Open a native directory-selection dialog and return the chosen path.

    Args:
        initial_dir: Directory the dialog starts in ("" -> OS default).

    Returns:
        The selected absolute path, or None when the dialog is cancelled or
        tkinter is unavailable (e.g. headless environment).
    """
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as exc:  # noqa: BLE001 - ImportError/TclError variants
        logger.warning("tkinter unavailable, folder dialog disabled: %s", exc)
        return None

    try:
        root = tk.Tk()
        try:
            root.withdraw()
            root.attributes("-topmost", True)  # dialog in front of browser
            path = filedialog.askdirectory(
                initialdir=initial_dir or None, parent=root)
        finally:
            root.destroy()
    except Exception as exc:  # noqa: BLE001 - Tcl errors must not kill Dash
        logger.warning("Folder dialog failed: %s", exc)
        return None
    return path or None
