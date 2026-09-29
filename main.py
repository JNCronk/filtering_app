"""
Launcher for the Qt / PyQtGraph event filtering app.

Run with:
    python3 main.py [event_file.h5]
"""

from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from filtering_app.ui import main
else:
    from .ui import main


if __name__ == "__main__":
    main()
