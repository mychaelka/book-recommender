"""Offline: embed every book's text once and cache it.
Run:  uv run python -m recommender.embeddings
"""

from __future__ import annotations

import time
import numpy as np
import pandas as pd
from pathlib import Path

from recommender.data_preprocessing import ROOT, build_text, load_books, BOOKS_PATH

EMBEDDINGS_PATH = ROOT / "data" / "embeddings_minilm.npy"
EMBEDDING_IDS_PATH = ROOT / "data" / "embeddings_minilm_ids.npy"
MAX_CHARS = 1500          # MiniLM reads ~256 tokens (~1,000 chars)


def embedding_paths(books_path: Path) -> tuple[Path, Path]:
    """data/books_goodreads.parquet -> data/books_goodreads_embeddings.npy (+ _embedding_ids.npy)."""
    return (books_path.with_name(f"{books_path.stem}_embeddings.npy"),
            books_path.with_name(f"{books_path.stem}_embedding_ids.npy"))


def get_embedder():
    """all-MiniLM-L6-v2 from Chroma's ONNX build (runs locally)"""
    from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
    return DefaultEmbeddingFunction()


def embed_texts(texts: list[str], embed, batch_size: int = 512) -> np.ndarray:
    """(len(texts), 384) float32 matrix, rows L2-normalized."""
    chunks, start = [], time.perf_counter()
    for i in range(0, len(texts), batch_size):
        chunks.append(np.asarray(embed(texts[i:i + batch_size]), dtype=np.float32))
        if (i // batch_size) % 20 == 0:
            print(f"{min(i + batch_size, len(texts)):,}/{len(texts):,}  {time.perf_counter() - start:.0f}s", flush=True)
    vectors = np.vstack(chunks)
    return vectors


def build_embeddings(books_path: Path) -> np.ndarray:
    books = load_books(books_path)
    vectors = embed_texts(build_text(books).str[:MAX_CHARS].tolist(), get_embedder())
    vectors_path, ids_path = embedding_paths(books_path)
    np.save(vectors_path, vectors)
    np.save(ids_path, books.index.to_numpy().astype(str))
    return vectors


def load_embeddings(books_path: Path, ids: np.ndarray) -> np.ndarray:
    vectors_path, ids_path = embedding_paths(books_path)
    vectors = np.load(vectors_path)
    cached_ids = np.load(ids_path)
    if len(cached_ids) != len(ids) or (cached_ids != ids.astype(str)).any():
        raise ValueError(f"{vectors_path} belongs to a different book list: rebuild it")
    return vectors


if __name__ == "__main__":
    import sys
    build_embeddings(Path(sys.argv[1]) if len(sys.argv) > 1 else BOOKS_PATH)