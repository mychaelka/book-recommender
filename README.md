# Book recommender

*"I liked The Lord of the Rings — what else should I read?"*

Type a book you liked (typos are fine) and get book recommendations. A FastAPI service serves several
recommenders (popularity, TF-IDF, sentence embeddings, hybrid) over a catalogue built from Goodreads and
Book-Crossing, with a small Gradio web UI on top.  
Collaborative filtering in progress.

## Quickstart

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12.

```bash
uv sync                                                   # install dependencies
uv run python -m catalog.pipeline                         # build data/books.parquet
uv run python -m recommender.embeddings                   # embed all books (~1-2 h on CPU, once)
uv run fastapi dev                                        # API on http://127.0.0.1:8000  (docs: /docs)
uv run python -m ui.app                                   # UI on http://127.0.0.1:7860 (second terminal)
```

Data: download `books.csv` from the [Goodreads Best Books Ever dataset](https://github.com/scostap/goodreads_bbe_dataset)
into `data/`. The Book-Crossing data is downloaded automatically through `kagglehub`.

## How it works

```
Goodreads + Book-Crossing ──► catalogue pipeline ──► books.parquet ──► models ──► FastAPI ──► Gradio UI
 (descriptions, genres)        match by title +        one row per      (+ embeddings, built
 (readers, subjects)           author, merge            book              offline)
```

1. **Catalogue** (`src/catalog/`): loads both sources, cleans them, matches the same book across sources by
   normalised title + author, and merges each match into one row (Goodreads description first).
   255k books, ~99k with a description. A Goodreads-only catalogue can be built with `--goodreads-only`.
2. **Models** (`src/recommender/`): every model scores all books for one or more query books.

   | Model | Idea |
   |---|---|
   | `popularity` | Most read/rated books (baseline) |
   | `tfidf` | Word overlap of title + tags + description |
   | `embeddings` | Meaning of the text (all-MiniLM-L6-v2, runs locally) |
   | `hybrid` | Reciprocal Rank Fusion of TF-IDF and embeddings |
   | `hybrid_popular` | Hybrid, re-ranked with popularity |

3. **API** (`src/api/`): users never need IDs — a fuzzy title search resolves what they typed.

   | Endpoint | Purpose |
   |---|---|
   | `GET /recommendations/by-title?title=...&model=...` | Recommend from a typed title (or suggest matches) |
   | `GET /recommendations?book_id=...` | Recommend from one or more known books |
   | `GET /books/search?q=...` | Fuzzy title search: match / did you mean / not found |
   | `GET /books/popular` | Most popular books |
   | `GET /books/{book_id}` | Full book details |
   | `GET /health` | Status and loaded models |

4. **UI** (`src/ui/app.py`): a Gradio page that calls the API.

## Evaluation

```bash
uv run python -m recommender.eval
```

Leave-one-out on Book-Crossing readers: a reader liked books A and B → the model gets A → is B in its top 10?
Highlights (2,000 readers): TF-IDF hits 3.05% vs 1.65% for popularity — but almost all of its hits are other
books *by the same author*; for books by other authors, popularity is ~10× better. That points to collaborative
filtering as the next model. Full results and reasoning: [`docs/design_notes.md`](docs/design_notes.md).

## Project structure

```
src/catalog/        data pipeline: load, clean, match and merge the sources
src/recommender/    models, embeddings, title search, service, evaluation
src/api/            FastAPI app, routers, response schemas
src/ui/             Gradio frontend
tests/              unit and API tests (uv run pytest)
notebooks/          exploration
docs/               design notes, decisions and open issues
```

## Data sources

- **Goodreads Best Books Ever** — Lorena Casanova Lozano & Sergio Costa Planells (2020). *Best Books Ever Dataset*
  (Version 1.0.0) [Data set]. Zenodo. http://doi.org/10.5281/zenodo.4265096
- **Book-Crossing** — Cai-Nicolas Ziegler, Sean M. McNee, Joseph A. Konstan, Georg Lausen (2005). *Improving
  Recommendation Lists Through Topic Diversification*. WWW '05. Used in the Open Library–enriched version
  [book-recommendation-platform-data](https://www.kaggle.com/datasets/simonbouchardk/book-recommendation-platform-data)
  by Simon Bouchard.
