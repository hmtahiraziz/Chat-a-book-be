import json
from pathlib import Path
from typing import Dict

from app.config import MANIFEST_FILE


def _read_manifest() -> Dict[str, dict]:
    if not MANIFEST_FILE.exists():
        return {}
    try:
        return json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_manifest(data: Dict[str, dict]) -> None:
    MANIFEST_FILE.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")


def upsert_book(book_id: str, payload: dict) -> None:
    data = _read_manifest()
    data[book_id] = payload
    _write_manifest(data)


def list_books() -> Dict[str, dict]:
    return _read_manifest()


def get_book(book_id: str) -> dict | None:
    return _read_manifest().get(book_id)
