"""Unit tests for chunk ID ordering used in Pinecone pagination."""

from app.services.pinecone_store import chunk_id_sort_key, sort_chunk_ids


def test_sort_chunk_ids_by_ordinal() -> None:
    ids = ["c00000010", "c00000002", "c00000001", "legacy_xyz"]
    assert sort_chunk_ids(ids) == ["c00000001", "c00000002", "c00000010", "legacy_xyz"]


def test_chunk_id_sort_key_numeric() -> None:
    assert chunk_id_sort_key("c00000042") == (42, "c00000042")
    assert chunk_id_sort_key("unknown")[0] == 10**9
