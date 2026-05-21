#!/usr/bin/env python3
"""One-time import of legacy data/manifest.json into MongoDB (metadata only).

PDFs on disk are not uploaded to GridFS; re-ingest books if PDFs are missing in MongoDB.
Run from repo root: python scripts/migrate_manifest_to_mongo.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

LEGACY_MANIFEST = ROOT / "data" / "manifest.json"


def main() -> None:
    if not LEGACY_MANIFEST.is_file():
        print(f"No legacy manifest at {LEGACY_MANIFEST}")
        return

    from app.services.book_service import upsert_book

    data = json.loads(LEGACY_MANIFEST.read_text(encoding="utf-8"))
    count = 0
    for book_id, payload in data.items():
        upsert_book(book_id, payload)
        count += 1
    print(f"Imported {count} book(s) into MongoDB.")


if __name__ == "__main__":
    main()
