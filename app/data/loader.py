"""Bundled-dataset loader.

The dataset is materialized at build time by scripts/build_dataset.py and
shipped as parquet files in data/. This loader is a thin in-memory wrapper —
all parquet files are loaded once at import time (they're tiny: < 30 MB total).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
CHARTS_DIR = DATA_DIR / "charts"


def _load(name: str) -> pd.DataFrame:
    path = DATA_DIR / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing. Run `python scripts/build_dataset.py` first."
        )
    return pd.read_parquet(path)


@lru_cache(maxsize=1)
def prices() -> pd.DataFrame:
    return _load("prices")


@lru_cache(maxsize=1)
def indicators() -> pd.DataFrame:
    return _load("indicators")


@lru_cache(maxsize=1)
def news() -> pd.DataFrame:
    return _load("news")


@lru_cache(maxsize=1)
def forums() -> pd.DataFrame:
    return _load("forums")


@lru_cache(maxsize=1)
def peers() -> pd.DataFrame:
    return _load("peers")


@lru_cache(maxsize=1)
def macro() -> pd.DataFrame:
    return _load("macro")


def episode_rows(task_id: str) -> pd.DataFrame:
    """Episode prices for a task, sorted by date."""
    df = prices()
    mask = (df["task_id"] == task_id) & df["in_episode"]
    sub = df.loc[mask]
    return sub.sort_values(by="date").reset_index(drop=True)


def lookup_indicators(ticker: str, date: str) -> dict:
    df = indicators()
    row = df[(df["ticker"] == ticker) & (df["date"] == date)]
    if row.empty:
        try:
            from app.data import corpus
            if corpus.available():
                return corpus.lookup_indicators(ticker, date)
        except Exception:
            pass
        return {}
    rec = row.iloc[0].to_dict()
    rec.pop("ticker", None)
    rec.pop("date", None)
    return {k: (None if pd.isna(v) else float(v)) for k, v in rec.items()}


def lookup_headlines(ticker: str, date: str, lookback_days: int = 7) -> list[dict]:
    """Return headlines from the past `lookback_days` for this ticker.

    Pulls from the curated `news.parquet` first; if nothing is found there,
    falls back to the EDGAR + RSS stream in the corpus (decades-deep).
    """
    df = news()
    df = df[df["ticker"] == ticker].copy()
    bundled: list[dict] = []
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
        cutoff_lo = pd.to_datetime(date) - pd.Timedelta(days=lookback_days)
        cutoff_hi = pd.to_datetime(date)
        sl = df[(df["date"] >= cutoff_lo) & (df["date"] <= cutoff_hi)]
        bundled = [
            {"date": d.strftime("%Y-%m-%d"), "headline": h, "source": s, "sentiment_label": sl_}
            for d, h, s, sl_ in zip(sl["date"], sl["headline"], sl["source"], sl["sentiment_label"])
        ]
    if bundled:
        return bundled
    try:
        from app.data import corpus
        if corpus.available():
            return corpus.lookup_headlines(ticker, date, lookback_days=lookback_days)
    except Exception:
        pass
    return []


def lookup_forum_excerpts(ticker: str, date: str, lookback_days: int = 7) -> list[dict]:
    df = forums()
    df = df[df["ticker"] == ticker].copy()
    if df.empty:
        return []
    df["date"] = pd.to_datetime(df["date"])
    cutoff_lo = pd.to_datetime(date) - pd.Timedelta(days=lookback_days)
    cutoff_hi = pd.to_datetime(date)
    sl = df[(df["date"] >= cutoff_lo) & (df["date"] <= cutoff_hi)]
    return [
        {"date": d.strftime("%Y-%m-%d"), "subreddit": sr, "score": int(sc), "post_text": t}
        for d, sr, sc, t in zip(sl["date"], sl["subreddit"], sl["score"], sl["post_text"])
    ]


def lookup_peers(ticker: str, date: str) -> dict:
    df = peers()
    row = df[(df["ticker"] == ticker) & (df["date"] == date)]
    if row.empty:
        return {"peers": [], "commodity": None, "commodity_price": None}
    peer_list = [
        {"peer_ticker": pt, "peer_close": (None if pd.isna(pc) else float(pc))}
        for pt, pc in zip(row["peer_ticker"], row["peer_close"])
    ]
    com = row.iloc[0]
    return {
        "peers": peer_list,
        "commodity": com["commodity"],
        "commodity_price": (None if pd.isna(com["commodity_price"]) else float(com["commodity_price"])),
    }


def lookup_macro(date: str, lookback_days: int = 14) -> list[dict]:
    df = macro().copy()
    df["date"] = pd.to_datetime(df["date"])
    cutoff_lo = pd.to_datetime(date) - pd.Timedelta(days=lookback_days)
    cutoff_hi = pd.to_datetime(date)
    sl = df[(df["date"] >= cutoff_lo) & (df["date"] <= cutoff_hi)]
    return [
        {"date": d.strftime("%Y-%m-%d"), "country": c, "headline": h, "policy_signal": ps}
        for d, c, h, ps in zip(sl["date"], sl["country"], sl["headline"], sl["policy_signal"])
    ]


def chart_path(ticker: str, date: str) -> str:
    """Return chart PNG path for (ticker, date), rendering on-demand if needed.

    Resolution order:
      1. Pre-rendered bundle: data/charts/{ticker}_{date}.png
      2. On-demand cache:     data/charts/cache/{ticker}_{date}.png (rendered
                              from corpus prices the first time it's asked)
      3. ""                   if we don't have prices for that ticker/date
    """
    p = CHARTS_DIR / f"{ticker}_{date}.png"
    if p.exists():
        return str(p)
    try:
        from app.data import corpus
        if corpus.available():
            return corpus.render_chart_cached(ticker, date)
    except Exception:
        pass
    return ""
