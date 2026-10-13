"""Minimal web UI for the book recommender. Talks to the FastAPI service over HTTP.

Run (with the API already running):  uv run python -m ui.app   ->  http://127.0.0.1:7860
"""

from __future__ import annotations

import os

import gradio as gr
import httpx

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")
client = httpx.Client(base_url=API_URL, timeout=30)

MODEL_LABELS = {
    "hybrid_cf": "Content + readers (TF-IDF + CF)",
    "cf": "Readers also liked (collaborative filtering)",
    "tfidf": "Similar description (TF-IDF)",
    "embeddings": "Similar meaning (embeddings)",
    "hybrid": "Content hybrid (TF-IDF + embeddings)",
    "hybrid_popular": "Content hybrid + popularity",
    "popularity": "Most popular (baseline)",
}


def model_choices(models: list[str]) -> list[tuple[str, str]]:
    """(label shown, model name sent to the API), best model first."""
    order = list(MODEL_LABELS)
    ranked = sorted(models, key=lambda m: order.index(m) if m in order else len(order))
    return [(MODEL_LABELS.get(m, m), m) for m in ranked]


def available_models() -> list[str]:
    """Models the running API actually has loaded (from /health)."""
    try:
        return client.get("/health").json()["models"]
    except httpx.HTTPError:
        return ["popularity", "tfidf"]


def render(books: list[dict], header: str) -> str:
    """Books as Markdown: numbered title, author, score, description snippet."""
    lines = [f"### {header}"]
    for i, book in enumerate(books, 1):
        score = f" · score {book['score']:.3f}" if "score" in book else ""
        lines.append(f"**{i}. {book['title']}** — {book.get('author') or 'unknown author'}{score}")
        if book.get("description"):
            lines.append(f"> {book['description']}")
    return "\n\n".join(lines)


def error_text(response: httpx.Response) -> str:
    return f"**{response.json().get('detail', response.text)}**"


HIDE_SUGGESTIONS = (gr.Radio(visible=False), gr.Button(visible=False), {})


def recommend_by_title(title: str, model: str, k: int):
    """Returns (results markdown, suggestions radio, 'use this book' button, {label: book_id})."""
    if not title.strip():
        return ("Type a book title.", *HIDE_SUGGESTIONS)
    try:
        response = client.get("/recommendations/by-title", params={"title": title, "model": model, "k": k})
    except httpx.HTTPError:
        return (f"The API is not reachable at {API_URL}. Is it running?", *HIDE_SUGGESTIONS)
    if response.status_code != 200:
        return (error_text(response), *HIDE_SUGGESTIONS)

    body = response.json()
    if body["status"] == "match":
        query = body["query"]
        header = f"Because you liked *{query['title']}* ({query.get('author') or 'unknown author'})"
        return (render(body["recommendations"], header), *HIDE_SUGGESTIONS)

    options = {f"{b['title']} — {b.get('author') or 'unknown author'}": b["book_id"] for b in body["suggestions"]}
    return ("Did you mean one of these? Pick one and press **Use this book**.",
            gr.Radio(choices=list(options), value=None, visible=True), gr.Button(visible=True), options)


def recommend_by_choice(choice: str | None, options: dict, model: str, k: int) -> str:
    if not choice:
        return "Pick a book first."
    response = client.get("/recommendations", params={"book_id": options[choice], "model": model, "k": k})
    if response.status_code != 200:
        return error_text(response)
    body = response.json()
    return render(body["recommendations"], f"Because you liked *{body['query'][0]['title']}*")


def popular(k: int) -> str:
    return render(client.get("/books/popular", params={"k": k}).json(), "Most popular books")


with gr.Blocks(title="Book recommender") as demo:
    gr.Markdown("# 📚 Book recommender\nTell me a book you liked, and I'll suggest what to read next.")

    with gr.Tab("I liked a book"):
        models = available_models()
        with gr.Row():
            title = gr.Textbox(label="Book title", placeholder="The Fellowship of the Ring", scale=3)
            choices = model_choices(available_models())
            model = gr.Dropdown(choices, value=choices[0][1], label="Model")
            k = gr.Slider(1, 20, value=10, step=1, label="How many")
        recommend_button = gr.Button("Recommend", variant="primary")
        suggestions = gr.Radio(label="Suggestions", visible=False)
        use_button = gr.Button("Use this book", visible=False)
        options = gr.State({})
        results = gr.Markdown()

        outputs = [results, suggestions, use_button, options]
        recommend_button.click(recommend_by_title, [title, model, k], outputs)
        title.submit(recommend_by_title, [title, model, k], outputs)          # Enter key
        use_button.click(recommend_by_choice, [suggestions, options, model, k], results)

    with gr.Tab("Popular"):
        k_popular = gr.Slider(1, 50, value=10, step=1, label="How many")
        popular_button = gr.Button("Show popular books")
        popular_results = gr.Markdown()
        popular_button.click(popular, k_popular, popular_results)


if __name__ == "__main__":
    demo.launch()