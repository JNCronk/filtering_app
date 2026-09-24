"""
Compatibility launcher for the dwell_t preview app.

Run with:
    python3 app/main.py
"""

from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from filtering_app import main
else:
    from filtering_app import main


if __name__ == "__main__":
    main()
