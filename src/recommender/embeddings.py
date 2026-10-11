"""Offline: embed every book's text once and cache it.
Run:  uv run python -m recommender.embeddings
"""

from __future__ import annotations

import time
import numpy as np
import pandas as pd

from recommender.data_preprocessing import ROOT, build_text, load_books

EMBEDDINGS_PATH = ROOT / "data" / "embeddings_minilm.npy"
EMBEDDING_IDS_PATH = ROOT / "data" / "embeddings_minilm_ids.npy"
MAX_CHARS = 1500          # MiniLM reads ~256 tokens (~1,000 chars)


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


def build_embeddings(books: pd.DataFrame) -> np.ndarray:
    texts = build_text(books).str[:MAX_CHARS].tolist()
    vectors = embed_texts(texts, get_embedder())
    np.save(EMBEDDINGS_PATH, vectors)
    np.save(EMBEDDING_IDS_PATH, books.index.to_numpy().astype(str))
    return vectors


def load_embeddings(ids: np.ndarray) -> np.ndarray:
    """Load cached embeddings and check they belong to exactly these books, in this order."""
    if not EMBEDDINGS_PATH.exists():
        raise FileNotFoundError(f"{EMBEDDINGS_PATH} missing: run `uv run python -m recommender.embeddings`")
    vectors = np.load(EMBEDDINGS_PATH)
    cached_ids = np.load(EMBEDDING_IDS_PATH)
    if len(cached_ids) != len(ids) or (cached_ids != ids.astype(str)).any():
        raise ValueError("Embedding cache belongs to a different book list: rebuild it")
    return vectors


if __name__ == "__main__":
    build_embeddings(load_books())