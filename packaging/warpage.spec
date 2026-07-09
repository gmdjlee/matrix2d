# PyInstaller spec for the Warpage Analysis Dash app.
#
# Build (run from the REPO ROOT, not this folder):
#     pip install pyinstaller
#     pyinstaller packaging/warpage.spec
#
# Output: dist/WarpageAnalysis.exe  (onefile)
#
# Why a spec (not a bare `pyinstaller app_main.py`): Dash, plotly and kaleido
# ship data files (JS bundles, plotly validators, the kaleido renderer binary)
# that PyInstaller's default analysis does NOT pick up. collect_all() grabs
# them. The app's own assets/style.css is added by hand.

import os
from PyInstaller.utils.hooks import collect_all

block_cipher = None

# Paths are resolved relative to this spec file (SPECPATH = packaging/), so the
# build works no matter what cwd `pyinstaller` is launched from.
ROOT = os.path.dirname(SPECPATH)

datas = []
binaries = []
hiddenimports = []

# Bundle package data + hidden imports for the libraries that need it.
for pkg in ("dash", "plotly", "kaleido", "scipy", "pandas"):
    try:
        d, b, h = collect_all(pkg)
        datas += d
        binaries += b
        hiddenimports += h
    except Exception:
        pass

# The app reads assets from <matrix2d/ui/app.py dir>/assets at runtime, so the
# CSS must land at matrix2d/ui/assets inside the bundle.
datas += [(os.path.join(ROOT, "src", "matrix2d", "ui", "assets"),
           "matrix2d/ui/assets")]

# Optional: ship demo data so a fresh exe has something to Scan. Comment out
# to keep the exe smaller.
if os.path.isdir(os.path.join(ROOT, "demo_data")):
    datas += [(os.path.join(ROOT, "demo_data"), "demo_data")]

a = Analysis(
    [os.path.join(SPECPATH, "app_main.py")],
    pathex=[os.path.join(ROOT, "src")],   # so `import matrix2d` resolves
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=["pytest", "tkinter.test"],
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    name="WarpageAnalysis",
    console=True,          # keep a console window so users can read errors / Ctrl+C
    onefile=True,
    upx=False,
    # icon="packaging/app.ico",   # <- add a .ico here for a custom icon
)
