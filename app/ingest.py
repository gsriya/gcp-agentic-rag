"""Data pipeline: load unstructured docs (.md/.txt/.pdf) and structured tables (.csv),
split them into chunks, embed them and store them in a Chroma vector database.

Run:  python -m app.ingest data/
"""
import csv
import sys
from pathlib import Path

import chromadb

from .config import settings
from .llm import get_llm

CHUNK_CHARS = 800
OVERLAP = 120


def chunk_text(text: str, size: int = CHUNK_CHARS, overlap: int = OVERLAP) -> list[str]:
    """Split on paragraphs, then pack paragraphs into chunks of about `size` characters."""
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks, cur = [], ""
    for p in paras:
        if cur and len(cur) + len(p) > size:
            chunks.append(cur)
            cur = cur[-overlap:] + "\n" + p  # carry a little context over
        else:
            cur = f"{cur}\n\n{p}" if cur else p
    if cur:
        chunks.append(cur)
    return chunks


def load_unstructured(path: Path) -> str:
    if path.suffix == ".pdf":
        from pypdf import PdfReader
        return "\n\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
    return path.read_text(encoding="utf-8")


def load_structured(path: Path) -> list[str]:
    """Each CSV row becomes one self-describing record: 'col: value; col: value'."""
    with path.open(newline="", encoding="utf-8") as f:
        return ["; ".join(f"{k}: {v}" for k, v in row.items()) for row in csv.DictReader(f)]


def build_records(folder: Path) -> list[dict]:
    records = []
    for path in sorted(folder.rglob("*")):
        if path.suffix in {".md", ".txt", ".pdf"}:
            for i, c in enumerate(chunk_text(load_unstructured(path))):
                records.append({"id": f"{path.name}#{i}", "text": c,
                                "meta": {"source": path.name, "kind": "unstructured"}})
        elif path.suffix == ".csv":
            for i, row in enumerate(load_structured(path)):
                records.append({"id": f"{path.name}#row{i}", "text": row,
                                "meta": {"source": path.name, "kind": "structured"}})
    return records


def get_collection():
    client = chromadb.PersistentClient(path=settings.db_path)
    return client.get_or_create_collection(settings.collection,
                                           metadata={"hnsw:space": "cosine"})


def ingest(folder: str) -> int:
    records = build_records(Path(folder))
    if not records:
        raise SystemExit(f"No .md/.txt/.pdf/.csv files found in {folder}")
    vectors = get_llm().embed([r["text"] for r in records])
    get_collection().upsert(ids=[r["id"] for r in records],
                            documents=[r["text"] for r in records],
                            metadatas=[r["meta"] for r in records],
                            embeddings=vectors)
    return len(records)


if __name__ == "__main__":
    n = ingest(sys.argv[1] if len(sys.argv) > 1 else "data")
    print(f"Ingested {n} chunks into '{settings.collection}' at {settings.db_path}")
