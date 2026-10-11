"""
Offline evaluation on Book-Crossing readers.

Protocol (leave-one-out, one pair per reader):
    a reader liked books A and B (rating >= 7)  ->  give a model only A  ->  is B in its top k?

Two test sets:
    all pairs      any two books the reader liked
    related pairs  the two books also share an informative tag (filters "liked both, but unrelated")

Run:  uv run python -m recommender.eval
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import kagglehub
import numpy as np
import pandas as pd
from scipy.stats import chi2

from catalog.normalize import author_key
from recommender.data_preprocessing import ROOT, load_books
from recommender.embeddings import load_embeddings
from recommender.models import HybridRecommender, Recommender, seeds_from_positions
from recommender.service import RecommenderService

BOOKCROSSING_HANDLE = "simonbouchardk/book-recommendation-platform-data/versions/1"
LIKED_RATING = 7
K = 10
REFERENCE_MODEL = "tfidf"          # every model is compared against this one (McNemar p-value)
RESULTS_DIR = ROOT / "data" / "eval"

# Tags that say nothing about what a book is about (format, catch-all, nationality).
NON_INFORMATIVE_TAGS = frozenset({
    "fiction", "general", "adult", "novels", "literature", "nonfiction", "audiobook", "large type books",
    "juvenile fiction", "contemporary", "classics", "short stories", "adult fiction", "literary fiction",
    "american", "english", "british literature", "american literature", "juvenile literature", "reference",
})


# ---------- test sets ----------

def load_liked(books: pd.DataFrame) -> pd.DataFrame:
    """(user_id, book_id) pairs: explicit rating >= 7, on books the content models can recommend."""
    interactions = pd.read_csv(Path(kagglehub.dataset_download(BOOKCROSSING_HANDLE)) / "interactions.csv",
                               dtype={"user_id": str, "work_id": str})
    aliases = pd.read_parquet(ROOT / "data" / "bx_aliases.parquet")["kept_id"]
    interactions["book_id"] = interactions["work_id"].map(aliases).fillna(interactions["work_id"])
    candidates = books.index[books["has_description"]]
    liked = interactions[(interactions["rating"] >= LIKED_RATING) & interactions["book_id"].isin(candidates)]
    return liked.drop_duplicates(["user_id", "book_id"])[["user_id", "book_id"]]


def liked_per_reader(liked: pd.DataFrame) -> pd.Series:
    """user_id -> list of liked books, for readers with at least 2."""
    per_user = liked.groupby("user_id")["book_id"].agg(list)
    return per_user[per_user.map(len) >= 2]


def make_queries(liked: pd.DataFrame, n_queries: int, seed: int) -> pd.DataFrame:
    """One random (query, target) pair per sampled reader."""
    rng = np.random.default_rng(seed)
    per_user = liked_per_reader(liked)
    users = rng.choice(per_user.index.to_numpy(), size=min(n_queries, len(per_user)), replace=False)
    pairs = [rng.choice(per_user[u], size=2, replace=False) for u in users]
    return pd.DataFrame(pairs, columns=["query", "target"])


def make_related_queries(liked: pd.DataFrame, books: pd.DataFrame, n_queries: int, seed: int,
                         max_tries: int = 50) -> pd.DataFrame:
    """Like make_queries, but query and target must share at least one informative tag.
    Readers with no such pair (after `max_tries` random pairs) are skipped."""
    rng = np.random.default_rng(seed)
    tags = books["tags"].map(lambda t: frozenset(str(x).strip().casefold() for x in t) - NON_INFORMATIVE_TAGS)
    per_user = liked_per_reader(liked)
    pairs = []
    for user in rng.permutation(per_user.index.to_numpy()):
        if len(pairs) == n_queries:
            break
        for _ in range(max_tries):
            query, target = rng.choice(per_user[user], size=2, replace=False)
            if tags[query] & tags[target]:
                pairs.append((query, target))
                break
    return pd.DataFrame(pairs, columns=["query", "target"])


# ---------- running and scoring a model ----------

@dataclass
class EvalContext:
    """Everything a model is scored against, computed once per test set."""
    service: RecommenderService
    query_pos: np.ndarray          # (n_queries,) row position of each query book
    target_pos: np.ndarray         # (n_queries,) row position of each hidden target
    authors: np.ndarray            # (n_books,) author key, "" if unknown
    same_author_pair: np.ndarray   # (n_queries,) query and target have the same (known) author
    popularity: np.ndarray         # (n_books,) 0-1, computed WITHOUT the hidden targets


def build_context(books: pd.DataFrame, queries: pd.DataFrame, embeddings: np.ndarray) -> EvalContext:
    # Leakage: each hidden target's own reader must not count towards its popularity.
    hidden = queries["target"].value_counts()
    train = books.assign(bx_readers=books["bx_readers"].sub(hidden, fill_value=0).clip(lower=0).astype(int))
    service = RecommenderService(train, embeddings=embeddings)

    query_pos = queries["query"].map(service.positions).to_numpy()
    target_pos = queries["target"].map(service.positions).to_numpy()
    authors = books["author"].map(author_key).fillna("").to_numpy()
    same_author_pair = (authors[query_pos] == authors[target_pos]) & (authors[query_pos] != "")
    return EvalContext(service, query_pos, target_pos, authors, same_author_pair,
                       service.models["popularity"].popularity)


def recommend_all(model: Recommender, ctx: EvalContext, batch_size: int = 128) -> np.ndarray:
    """(n_queries, K) recommended row positions, best first. Batched to bound memory."""
    n_items = len(ctx.service.ids)
    batches = []
    for start in range(0, len(ctx.query_pos), batch_size):
        seeds = seeds_from_positions([[p] for p in ctx.query_pos[start:start + batch_size]], n_items)
        batches.append(model.recommend(seeds, K)[0])
    return np.vstack(batches)


def target_ranks(top: np.ndarray, target_pos: np.ndarray) -> np.ndarray:
    """1-based rank of each query's target in its top K; 0 = not recommended."""
    found = top == target_pos[:, None]
    return np.where(found.any(axis=1), found.argmax(axis=1) + 1, 0)


def score_model(model: Recommender, ctx: EvalContext) -> tuple[dict, np.ndarray]:
    """Metrics for one model, plus its per-query hits (for paired significance tests)."""
    start = time.perf_counter()
    top = recommend_all(model, ctx)
    seconds = time.perf_counter() - start

    ranks = target_ranks(top, ctx.target_pos)
    hits = ranks > 0
    hit_rate = hits.mean()
    query_authors = ctx.authors[ctx.query_pos][:, None]
    metrics = {
        "hit_rate": hit_rate,
        "std_error": np.sqrt(hit_rate * (1 - hit_rate) / len(hits)),
        "mrr": np.where(hits, 1 / np.maximum(ranks, 1), 0).mean(),
        "hit_same_author": hits[ctx.same_author_pair].mean(),
        "hit_other_author": hits[~ctx.same_author_pair].mean(),
        "coverage": len(np.unique(top)),
        "avg_popularity": ctx.popularity[top].mean(),
        "recs_by_query_author": ((ctx.authors[top] == query_authors) & (query_authors != "")).mean(),
        "ms_per_query": 1000 * seconds / len(hits),
    }
    return metrics, hits


def mcnemar_p(hits_a: np.ndarray, hits_b: np.ndarray) -> float:
    """p-value for 'A and B are equally good', from their hit/miss on the SAME queries."""
    only_a = np.sum(hits_a & ~hits_b)
    only_b = np.sum(~hits_a & hits_b)
    if only_a + only_b == 0:
        return 1.0
    statistic = (abs(only_a - only_b) - 1) ** 2 / (only_a + only_b)
    return float(chi2.sf(statistic, df=1))


def evaluate_models(models: dict[str, Recommender], ctx: EvalContext) -> pd.DataFrame:
    """One row per model; p_vs_reference compares each model with REFERENCE_MODEL (if present)."""
    rows, hits = {}, {}
    for name, model in models.items():
        rows[name], hits[name] = score_model(model, ctx)
    table = pd.DataFrame(rows).T
    if REFERENCE_MODEL in hits:
        table[f"p_vs_{REFERENCE_MODEL}"] = [mcnemar_p(hits[n], hits[REFERENCE_MODEL]) for n in table.index]
    return table


def alpha_sweep(ctx: EvalContext, alphas: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)) -> pd.DataFrame:
    """Hybrid (TF-IDF weight = alpha) for several alphas, on the same queries."""
    tfidf, embeddings = ctx.service.models["tfidf"], ctx.service.models["embeddings"]
    models = {f"alpha={a}": HybridRecommender(tfidf, embeddings, alpha=a) for a in alphas}
    return evaluate_models(models, ctx)


# ---------- report ----------

RATE_COLUMNS = ["hit_rate", "std_error", "hit_same_author", "hit_other_author", "recs_by_query_author"]


def show(table: pd.DataFrame) -> str:
    """Rates as percentages, everything else rounded."""
    shown = table.copy()
    shown[RATE_COLUMNS] = shown[RATE_COLUMNS] * 100
    return shown.round({**{c: 2 for c in RATE_COLUMNS}, "mrr": 4, "coverage": 0, "avg_popularity": 2,
                        "ms_per_query": 1, f"p_vs_{REFERENCE_MODEL}": 3}).to_string()


def main(n_queries: int = 2000, seed: int = 0) -> None:
    books = load_books()
    embeddings = load_embeddings(books.index.to_numpy())
    liked = load_liked(books)
    test_sets = {
        "all_pairs": make_queries(liked, n_queries, seed),
        "related_pairs": make_related_queries(liked, books, n_queries, seed),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for name, queries in test_sets.items():
        ctx = build_context(books, queries, embeddings)
        print(f"\n===== {name}: {len(queries)} queries (seed={seed}), "
              f"{ctx.same_author_pair.mean():.1%} have the same author as their target =====")
        table = evaluate_models(ctx.service.models, ctx)
        print(show(table))
        table.to_csv(RESULTS_DIR / f"{name}_seed{seed}.csv")

        if name == "all_pairs":
            print(f"\n----- alpha sweep (hybrid without popularity), {name} -----")
            print(show(alpha_sweep(ctx)))

    print("\nRates are in %. hit_* = share of queries whose hidden target is in the top "
          f"{K}; recs_by_query_author = share of recommendations written by the query's author.")


if __name__ == "__main__":
    main()
