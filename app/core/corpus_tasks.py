"""Corpus task definitions — built from data/corpus/episodes.parquet.

These extend the 3 hand-curated tasks in `app.core.tasks` without
disturbing them: existing task IDs (task_easy/medium/hard) keep their
old `data/prices.parquet` path; corpus task IDs (prefixed `corpus_`) are
served from `app.data.corpus.episode_rows`.

The merge happens in `app.core.tasks.get_task_definition`.
"""
from __future__ import annotations

from functools import lru_cache

from app.data import corpus

# Light per-ticker fundamentals for the observation. Values are static
# stand-ins — the env doesn't grade them, they just give the LLM a hint.
TICKER_META: dict[str, dict] = {
    "AAPL":  {"sector": "tech",       "company": "Apple Inc."},
    "MSFT":  {"sector": "tech",       "company": "Microsoft Corp."},
    "GOOGL": {"sector": "tech",       "company": "Alphabet Inc."},
    "NVDA":  {"sector": "semis",      "company": "NVIDIA Corp."},
    "AMZN":  {"sector": "internet",   "company": "Amazon.com Inc."},
    "META":  {"sector": "internet",   "company": "Meta Platforms"},
    "INTC":  {"sector": "semis",      "company": "Intel Corp."},
    "JPM":   {"sector": "finance",    "company": "JPMorgan Chase"},
    "GS":    {"sector": "finance",    "company": "Goldman Sachs"},
    "BAC":   {"sector": "finance",    "company": "Bank of America"},
    "XOM":   {"sector": "energy",     "company": "Exxon Mobil"},
    "CVX":   {"sector": "energy",     "company": "Chevron Corp."},
    "KO":    {"sector": "consumer",   "company": "Coca-Cola"},
    "WMT":   {"sector": "consumer",   "company": "Walmart"},
    "BA":    {"sector": "industrial", "company": "Boeing Co."},
}


def _describe(ticker: str, start: str, end: str) -> str:
    meta = TICKER_META.get(ticker, {})
    co = meta.get("company", ticker)
    return f"{co} ({ticker}) — episode window {start} to {end}."


@lru_cache(maxsize=1)
def list_corpus_task_ids() -> list[str]:
    if not corpus.available():
        return []
    eps = corpus.episodes()
    if eps.empty:
        return []
    return eps["task_id"].astype(str).tolist()


def is_corpus_task(task_id: str) -> bool:
    return task_id.startswith("corpus_")


def get_corpus_task_definition(task_id: str) -> dict:
    """Build the same dict shape as `tasks.get_task_definition` for a corpus task."""
    if not corpus.available():
        raise RuntimeError("Corpus not built. Run scripts/build_corpus.py first.")
    eps = corpus.episodes()
    row = eps[eps["task_id"] == task_id]
    if row.empty:
        raise KeyError(f"Unknown corpus task_id: {task_id}")
    r = row.iloc[0]
    ticker = str(r["ticker"])
    start = str(r["episode_start"])
    end = str(r["episode_end"])

    rows = corpus.episode_rows(task_id)
    if rows.empty:
        raise RuntimeError(f"No price rows for {task_id} ({ticker} {start}..{end}).")

    return {
        "task_id": task_id,
        "description": _describe(ticker, start, end),
        "ticker": ticker,
        "starting_cash": 10000.0,
        "fundamentals": TICKER_META.get(ticker, {"sector": "unknown", "company": ticker}),
        "dates": rows["date"].tolist(),
        "prices": rows["close"].astype(float).tolist(),
    }
