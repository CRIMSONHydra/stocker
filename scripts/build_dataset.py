#!/usr/bin/env python3
"""Build the bundled Stocker dataset from public sources.

Reads:
  data/sources/news.jsonl
  data/sources/forums.jsonl
  data/sources/macro.jsonl

Pulls real OHLCV via yfinance for the 3 tasks (AAPL/INTC/META) plus peers
and a commodity proxy. Computes indicators with app/data/indicators.py
(hand-rolled, no pandas-ta dependency). Renders one
candlestick PNG per (ticker, date). Writes parquet outputs in data/.

Idempotent — re-running overwrites in-place.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yfinance as yf

from app.data import indicators as ta

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
SOURCES = DATA / "sources"
CHARTS = DATA / "charts"

# 3 tasks: ticker, episode-window start/end (used by the env), peer tickers,
# and a commodity proxy (e.g. gold for tech, oil for energy).
TASKS = [
    {
        "task_id": "task_easy",
        "ticker": "AAPL",
        "episode_start": "2023-08-01",
        "episode_end":   "2023-09-30",
        "peers": ["MSFT", "GOOGL"],
        "commodity": "GC=F",   # gold futures
        "commodity_label": "gold",
    },
    {
        "task_id": "task_medium",
        "ticker": "INTC",
        "episode_start": "2024-01-02",
        "episode_end":   "2024-02-29",
        "peers": ["AMD", "NVDA"],
        "commodity": "GC=F",
        "commodity_label": "gold",
    },
    {
        "task_id": "task_hard",
        "ticker": "META",
        "episode_start": "2022-09-01",
        "episode_end":   "2022-10-31",
        "peers": ["GOOGL", "SNAP"],
        "commodity": "CL=F",   # crude oil
        "commodity_label": "oil",
    },
]

# How much pre-history we need before episode_start for indicators + charts.
LOOKBACK_DAYS = 200


# ---------------------------------------------------------------------------
@dataclass
class TickerHistory:
    ticker: str
    df: pd.DataFrame  # indexed by date


def fetch_ohlcv(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(
        ticker, start=start, end=end, auto_adjust=False, progress=False
    )
    if df.empty:
        raise RuntimeError(f"yfinance returned empty for {ticker} {start}..{end}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df.columns = ["open", "high", "low", "close", "volume"]
    return df


def lookback_start(episode_start: str, days: int) -> str:
    return (pd.to_datetime(episode_start) - pd.Timedelta(days=days)).strftime("%Y-%m-%d")


def build_prices() -> pd.DataFrame:
    rows = []
    for task in TASKS:
        start = lookback_start(task["episode_start"], LOOKBACK_DAYS)
        end = (pd.to_datetime(task["episode_end"]) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        df = fetch_ohlcv(task["ticker"], start, end)
        for date, row in df.iterrows():
            rows.append({
                "task_id": task["task_id"],
                "ticker": task["ticker"],
                "date": date.strftime("%Y-%m-%d"),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]),
                "in_episode": (
                    pd.to_datetime(task["episode_start"])
                    <= date
                    <= pd.to_datetime(task["episode_end"])
                ),
            })
    return pd.DataFrame(rows)


def build_indicators(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute indicators per ticker on the full lookback+episode window."""
    out_rows = []
    for ticker, group in prices.groupby("ticker"):
        df = group.sort_values("date").set_index(pd.to_datetime(group["date"]))
        close = df["close"]
        high = df["high"]
        low = df["low"]

        ind = pd.DataFrame(index=df.index)
        ind["rsi14"] = ta.rsi(close, length=14)
        macd_df = ta.macd(close, fast=12, slow=26, signal=9)
        ind["macd"] = macd_df["macd"]
        ind["macd_signal"] = macd_df["macd_signal"]
        ind["sma20"] = ta.sma(close, length=20)
        ind["sma50"] = ta.sma(close, length=50)
        ind["sma200"] = ta.sma(close, length=200)
        bb = ta.bbands(close, length=20, std=2)
        ind["bb_lower"] = bb["bb_lower"]
        ind["bb_upper"] = bb["bb_upper"]
        ind["atr14"] = ta.atr(high, low, close, length=14)

        ind = ind.reset_index().rename(columns={"index": "date"})
        ind["ticker"] = ticker
        ind["date"] = ind["date"].dt.strftime("%Y-%m-%d")
        out_rows.append(ind)
    out = pd.concat(out_rows, ignore_index=True)
    cols = ["ticker", "date", "rsi14", "macd", "macd_signal",
            "sma20", "sma50", "sma200", "bb_lower", "bb_upper", "atr14"]
    return out[cols]


def build_peers() -> pd.DataFrame:
    rows = []
    for task in TASKS:
        start = lookback_start(task["episode_start"], 30)
        end = (pd.to_datetime(task["episode_end"]) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        # Commodity series
        try:
            com_df = fetch_ohlcv(task["commodity"], start, end)["close"]
        except Exception as e:
            print(f"[warn] commodity {task['commodity']} fetch failed: {e}", file=sys.stderr)
            com_df = pd.Series(dtype=float)
        # Per-peer series
        peer_dfs = {}
        for peer in task["peers"]:
            try:
                peer_dfs[peer] = fetch_ohlcv(peer, start, end)["close"]
            except Exception as e:
                print(f"[warn] peer {peer} fetch failed: {e}", file=sys.stderr)

        # Walk per date
        all_dates = sorted(
            set().union(*[s.index for s in [com_df, *peer_dfs.values()] if not s.empty])
        )
        for d in all_dates:
            for peer, s in peer_dfs.items():
                rows.append({
                    "ticker": task["ticker"],
                    "date": pd.Timestamp(d).strftime("%Y-%m-%d"),
                    "peer_ticker": peer,
                    "peer_close": float(s.loc[d]) if d in s.index else None,
                    "commodity": task["commodity_label"],
                    "commodity_price": float(com_df.loc[d]) if d in com_df.index else None,
                })
    return pd.DataFrame(rows)


def build_news() -> pd.DataFrame:
    rows = []
    for entry in json.loads((SOURCES / "news.json").read_text()):
        ticker = entry["ticker"]
        for it in entry["items"]:
            rows.append({
                "ticker": ticker,
                "date": it["date"],
                "headline": it["headline"],
                "source": it["source"],
                "sentiment_label": it["sentiment_label"],
            })
    return pd.DataFrame(rows)


def build_forums() -> pd.DataFrame:
    rows = []
    for entry in json.loads((SOURCES / "forums.json").read_text()):
        ticker = entry["ticker"]
        for it in entry["items"]:
            rows.append({
                "ticker": ticker,
                "date": it["date"],
                "subreddit": it["subreddit"],
                "score": int(it["score"]),
                "post_text": it["post_text"],
            })
    return pd.DataFrame(rows)


def build_macro() -> pd.DataFrame:
    return pd.DataFrame(json.loads((SOURCES / "macro.json").read_text()))


def render_charts(prices: pd.DataFrame) -> int:
    """Render a 60-day candlestick PNG for every in-episode (ticker, date)."""
    from scripts.render_charts import render_chart  # noqa: E402

    CHARTS.mkdir(parents=True, exist_ok=True)
    n = 0
    for ticker, group in prices.groupby("ticker"):
        group = group.sort_values("date").reset_index(drop=True)
        group["date_dt"] = pd.to_datetime(group["date"])
        for i, row in group.iterrows():
            if not row["in_episode"]:
                continue
            window_start_idx = max(0, i - 60)
            window = group.iloc[window_start_idx : i + 1].copy()
            window = window.set_index("date_dt")[["open", "high", "low", "close", "volume"]]
            png_path = CHARTS / f"{ticker}_{row['date']}.png"
            render_chart(window, png_path, title=f"{ticker} as of {row['date']}")
            n += 1
    return n


# ---------------------------------------------------------------------------
def main():
    DATA.mkdir(parents=True, exist_ok=True)
    CHARTS.mkdir(parents=True, exist_ok=True)

    print("[1/6] Fetching OHLCV ...")
    prices = build_prices()
    prices.to_parquet(DATA / "prices.parquet", index=False)
    print(f"  prices: {len(prices)} rows")

    print("[2/6] Computing indicators ...")
    indicators = build_indicators(prices)
    indicators.to_parquet(DATA / "indicators.parquet", index=False)
    print(f"  indicators: {len(indicators)} rows")

    print("[3/6] Fetching peers + commodity ...")
    peers = build_peers()
    peers.to_parquet(DATA / "peers.parquet", index=False)
    print(f"  peers: {len(peers)} rows")

    print("[4/6] Loading curated news/forums/macro ...")
    news = build_news()
    news.to_parquet(DATA / "news.parquet", index=False)
    forums = build_forums()
    forums.to_parquet(DATA / "forums.parquet", index=False)
    macro = build_macro()
    macro.to_parquet(DATA / "macro.parquet", index=False)
    print(f"  news: {len(news)} headlines, forums: {len(forums)} posts, macro: {len(macro)} events")

    print("[5/6] Rendering candlestick charts ...")
    n_charts = render_charts(prices)
    print(f"  charts: {n_charts} PNGs in {CHARTS.relative_to(ROOT)}")

    print("[6/6] Done.")


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    main()
