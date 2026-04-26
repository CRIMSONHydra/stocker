"""Corpus loader — accessors for the larger dataset under data/corpus/.

Built by `scripts/build_corpus.py`. This module is independent of
`app.data.loader` (which still serves the original 3-task bundle). The
two are merged at lookup-time by `app.data.loader.chart_path` and by
`app.core.corpus_tasks`.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import pandas as pd

CORPUS_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "corpus"
CHART_CACHE = Path(__file__).resolve().parent.parent.parent / "data" / "charts" / "cache"


def available() -> bool:
    return (CORPUS_DIR / "prices.parquet").exists()


def _load(name: str) -> pd.DataFrame:
    path = CORPUS_DIR / f"{name}.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@lru_cache(maxsize=1)
def tickers() -> list[dict]:
    path = CORPUS_DIR / "tickers.json"
    if not path.exists():
        return []
    return json.loads(path.read_text())


@lru_cache(maxsize=1)
def prices() -> pd.DataFrame:
    return _load("prices")


@lru_cache(maxsize=1)
def indicators() -> pd.DataFrame:
    return _load("indicators")


@lru_cache(maxsize=1)
def filings() -> pd.DataFrame:
    return _load("filings")


@lru_cache(maxsize=1)
def news() -> pd.DataFrame:
    return _load("news")


@lru_cache(maxsize=1)
def episodes() -> pd.DataFrame:
    return _load("episodes")


def has_ticker(ticker: str) -> bool:
    df = prices()
    if df.empty:
        return False
    return bool((df["ticker"] == ticker).any())


def ticker_prices(ticker: str) -> pd.DataFrame:
    df = prices()
    if df.empty:
        return df
    return df[df["ticker"] == ticker].sort_values("date").reset_index(drop=True)


def episode_rows(task_id: str) -> pd.DataFrame:
    """Return per-day in-episode price rows for a corpus task, in date order."""
    eps = episodes()
    if eps.empty:
        return pd.DataFrame()
    row = eps[eps["task_id"] == task_id]
    if row.empty:
        return pd.DataFrame()
    r = row.iloc[0]
    px = prices()
    sl = px[
        (px["ticker"] == r["ticker"])
        & (px["date"] >= r["episode_start"])
        & (px["date"] <= r["episode_end"])
    ]
    return sl.sort_values("date").reset_index(drop=True)


def lookup_indicators(ticker: str, date: str) -> dict:
    df = indicators()
    if df.empty:
        return {}
    row = df[(df["ticker"] == ticker) & (df["date"] == date)]
    if row.empty:
        return {}
    rec = row.iloc[0].to_dict()
    rec.pop("ticker", None)
    rec.pop("date", None)
    return {k: (None if pd.isna(v) else float(v)) for k, v in rec.items()}


def lookup_headlines(ticker: str, date: str, lookback_days: int = 14) -> list[dict]:
    """Combine yfinance news + EDGAR filings into a single headline stream."""
    items: list[dict] = []
    cutoff_lo = pd.to_datetime(date) - pd.Timedelta(days=lookback_days)
    cutoff_hi = pd.to_datetime(date)

    nws = news()
    if not nws.empty:
        sub = nws[nws["ticker"] == ticker].copy()
        if not sub.empty:
            sub["d"] = pd.to_datetime(sub["date"])
            sub = sub[(sub["d"] >= cutoff_lo) & (sub["d"] <= cutoff_hi)]
            for _, r in sub.iterrows():
                items.append({
                    "date": r["d"].strftime("%Y-%m-%d"),
                    "headline": r["headline"],
                    "source": r["publisher"] or "Yahoo Finance",
                    "sentiment_label": "neutral",
                })

    fls = filings()
    if not fls.empty:
        sub = fls[fls["ticker"] == ticker].copy()
        if not sub.empty:
            sub["d"] = pd.to_datetime(sub["date"])
            sub = sub[(sub["d"] >= cutoff_lo) & (sub["d"] <= cutoff_hi)]
            for _, r in sub.iterrows():
                items.append({
                    "date": r["d"].strftime("%Y-%m-%d"),
                    "headline": f"[{r['form']}] {r['title']}",
                    "source": "SEC EDGAR",
                    "sentiment_label": "neutral",
                })

    items.sort(key=lambda x: x["date"])
    return items


def render_chart_cached(ticker: str, date: str, window_days: int = 60) -> str:
    """Render a candlestick PNG for (ticker, date) on demand, cache to disk.

    Returns the cache path as a string, or "" if we don't have prices for it.
    """
    if not has_ticker(ticker):
        return ""
    CHART_CACHE.mkdir(parents=True, exist_ok=True)
    out = CHART_CACHE / f"{ticker}_{date}.png"
    if out.exists():
        return str(out)

    px = ticker_prices(ticker)
    if px.empty:
        return ""
    idx = px.index[px["date"] == date]
    if len(idx) == 0:
        return ""
    end_i = int(idx[0])
    start_i = max(0, end_i - window_days)
    win = px.iloc[start_i : end_i + 1].copy()
    win.index = pd.to_datetime(win["date"])
    win = win[["open", "high", "low", "close", "volume"]]

    try:
        from scripts.render_charts import render_chart
        render_chart(win, out, title=f"{ticker} as of {date}")
    except Exception:
        return ""
    return str(out) if out.exists() else ""
