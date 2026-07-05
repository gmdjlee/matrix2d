"""Dash application factory."""

import os

import dash

from matrix2d.ui.layout import build_layout
from matrix2d.ui.callbacks import register_callbacks

_ASSETS = os.path.join(os.path.dirname(__file__), "assets")


def create_app() -> dash.Dash:
    """Create and configure the Dash app (single page)."""
    app = dash.Dash(
        __name__,
        assets_folder=_ASSETS,
        title="Warpage Analysis",
        suppress_callback_exceptions=True,  # pattern-matching / tab children
    )
    app.layout = build_layout()
    register_callbacks(app)
    return app
