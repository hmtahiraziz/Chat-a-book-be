#!/usr/bin/env python3
"""Create Pinecone serverless indexes for BookChat (reads names from .env by default).

Examples (from ai-book-chatbot-v2):
  python scripts/create_pinecone_indexes.py
  python scripts/create_pinecone_indexes.py --preset openai
  python scripts/create_pinecone_indexes.py --name bookchat-openai-3072 --dimension 3072
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OPENAI_DIMENSION = 3072


def _index_names_from_env() -> list[tuple[str, int]]:
    from app.config import PINECONE_INDEX, PINECONE_INDEX_OPENAI

    targets: list[tuple[str, int]] = []
    seen: set[str] = set()

    def add(name: str | None, dimension: int) -> None:
        if not name or name in seen:
            return
        seen.add(name)
        targets.append((name, dimension))

    add(PINECONE_INDEX_OPENAI, OPENAI_DIMENSION)
    if PINECONE_INDEX and PINECONE_INDEX not in seen:
        add(PINECONE_INDEX, OPENAI_DIMENSION)
    return targets


def _existing_index_names() -> set[str]:
    from app.services.pinecone_store import _pc

    pc = _pc()
    return {idx.name for idx in pc.list_indexes()}


def _create_one(
    name: str,
    dimension: int,
    *,
    cloud: str,
    region: str,
    metric: str,
    dry_run: bool,
) -> str:
    if dry_run:
        return f"[dry-run] would create {name!r} (dim={dimension}, metric={metric}, {cloud}/{region})"

    from pinecone.exceptions import PineconeApiException

    from app.services.pinecone_store import create_serverless_pinecone_index

    try:
        result = create_serverless_pinecone_index(
            name,
            dimension,
            metric=metric,
            cloud=cloud,
            region=region,
        )
        status = result.get("status") or "created"
        return f"Created {name!r} (dim={dimension}, status={status})"
    except PineconeApiException as exc:
        status = getattr(exc, "status", None)
        msg = str(exc)
        if status == 409 or "already exists" in msg.lower():
            return f"Skipped {name!r} (already exists)"
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preset",
        choices=["openai", "all"],
        default="all",
        help="Which index preset to create (default: all configured in .env).",
    )
    parser.add_argument("--name", help="Override Pinecone index name.")
    parser.add_argument("--dimension", type=int, help="Vector dimension (required with --name).")
    parser.add_argument("--metric", default="cosine", choices=["cosine", "dotproduct", "euclidean"])
    parser.add_argument("--cloud", default=None, help="Serverless cloud (default: PINECONE_SERVERLESS_CLOUD).")
    parser.add_argument("--region", default=None, help="Serverless region (default: PINECONE_SERVERLESS_REGION).")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without calling Pinecone.")
    args = parser.parse_args()

    from app.config import (
        PINECONE_API_KEY,
        PINECONE_INDEX_OPENAI,
        PINECONE_SERVERLESS_CLOUD,
        PINECONE_SERVERLESS_REGION,
        require_pinecone_config,
    )

    try:
        require_pinecone_config()
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    if not PINECONE_API_KEY:
        print("Error: PINECONE_API_KEY is not set in .env", file=sys.stderr)
        sys.exit(1)

    cloud = (args.cloud or PINECONE_SERVERLESS_CLOUD).strip()
    region = (args.region or PINECONE_SERVERLESS_REGION).strip()

    if args.name:
        if args.dimension is None:
            print("Error: --dimension is required when using --name", file=sys.stderr)
            sys.exit(1)
        targets = [(args.name.strip().lower(), args.dimension)]
    elif args.preset == "all":
        targets = _index_names_from_env()
        if not targets:
            print(
                "Error: no index names in .env. Set PINECONE_INDEX_OPENAI (see .env.example).",
                file=sys.stderr,
            )
            sys.exit(1)
    else:
        name = (PINECONE_INDEX_OPENAI or "bookchat-openai-3072").strip().lower()
        targets = [(name, OPENAI_DIMENSION)]

    existing = set() if args.dry_run else _existing_index_names()
    exit_code = 0
    for name, dimension in targets:
        if name in existing:
            print(f"Skipped {name!r} (already exists)")
            continue
        try:
            print(
                _create_one(
                    name,
                    dimension,
                    cloud=cloud,
                    region=region,
                    metric=args.metric,
                    dry_run=args.dry_run,
                )
            )
        except Exception as exc:
            print(f"Failed {name!r}: {exc}", file=sys.stderr)
            exit_code = 1

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
