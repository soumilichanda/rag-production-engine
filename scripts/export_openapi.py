"""
scripts/export_openapi.py
Generates an offline, versioned openapi.json specification directly from the FastAPI instance.
"""

import sys
import json
from pathlib import Path

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.main import app

OUTPUT_PATH = ROOT_DIR / "docs" / "specs" / "openapi.json"


def export_schema():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    schema = app.openapi()
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)
    print(f"OpenAPI schema exported cleanly to {OUTPUT_PATH}")


if __name__ == "__main__":
    export_schema()
