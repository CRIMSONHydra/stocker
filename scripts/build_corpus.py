#!/usr/bin/env python3
"""Build the *expanded* Stocker corpus: ~15 tickers x ~20y of daily data.

This is *additive* to the existing 3-task bundle in `data/`. The corpus
lives under `data/corpus/` and is consumed via `app.data.corpus`. The
3 hand-curated tasks keep working for tests and the existing GRPO loop.

Sources (all free, no API keys):
  - Prices/OHLCV : yfinance
  - Filings      : SEC EDGAR submissions API (8-K, 10-Q, 10-K, S-1)
  - News         : yfinance ticker .news (recent only)

Outputs (parquet unless noted):
  data/corpus/
    tickers.json          — ticker -> {name, sector, cik}
    prices.parquet        — (ticker, date, open, high, low, close, volume)
    indicators.parquet    — (ticker, date, rsi14, macd, macd_signal, sma20,
                             sma50, sma200, bb_lower, bb_upper, atr14)
    filings.parquet       — (ticker, date, form, title, accession, url)
    news.parquet          — (ticker, date, headline, publisher, url)
    episodes.parquet      — pre-sampled GRPO windows (task_id, ticker,
                             episode_start, episode_end)

Usage:
  python scripts/build_corpus.py                # full build
  python scripts/build_corpus.py --quick        # 3 tickers x 5y, fast smoke
  python scripts/build_corpus.py --skip-filings # skip EDGAR
  python scripts/build_corpus.py --skip-news    # skip yfinance news

Idempotent — overwrites parquets in place.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import yfinance as yf

from app.data import indicators as ta

DATA = ROOT / "data"
CORPUS = DATA / "corpus"

# ---------------------------------------------------------------------------
# Universe — 15 tickers across sectors. CIK is SEC's stable id (10 digits).
# ---------------------------------------------------------------------------
UNIVERSE: list[dict[str, str]] = [
    {"ticker": "AAPL",  "cik": "0000320193", "sector": "tech",       "name": "Apple Inc."},
    {"ticker": "MSFT",  "cik": "0000789019", "sector": "tech",       "name": "Microsoft Corp."},
    {"ticker": "GOOGL", "cik": "0001652044", "sector": "tech",       "name": "Alphabet Inc."},
    {"ticker": "NVDA",  "cik": "0001045810", "sector": "semis",      "name": "NVIDIA Corp."},
    {"ticker": "AMZN",  "cik": "0001018724", "sector": "internet",   "name": "Amazon.com Inc."},
    {"ticker": "META",  "cik": "0001326801", "sector": "internet",   "name": "Meta Platforms"},
    {"ticker": "INTC",  "cik": "0000050863", "sector": "semis",      "name": "Intel Corp."},
    {"ticker": "JPM",   "cik": "0000019617", "sector": "finance",    "name": "JPMorgan Chase"},
    {"ticker": "GS",    "cik": "0000886982", "sector": "finance",    "name": "Goldman Sachs"},
    {"ticker": "BAC",   "cik": "0000070858", "sector": "finance",    "name": "Bank of America"},
    {"ticker": "XOM",   "cik": "0000034088", "sector": "energy",     "name": "Exxon Mobil"},
    {"ticker": "CVX",   "cik": "0000093410", "sector": "energy",     "name": "Chevron Corp."},
    {"ticker": "KO",    "cik": "0000021344", "sector": "consumer",   "name": "Coca-Cola"},
    {"ticker": "WMT",   "cik": "0000104169", "sector": "consumer",   "name": "Walmart"},
    {"ticker": "BA",    "cik": "0000012927", "sector": "industrial", "name": "Boeing Co."},
]

DEFAULT_START = "2005-01-01"
QUICK_TICKERS = {"AAPL", "MSFT", "JPM"}
QUICK_START = "2020-01-01"
EDGAR_FORMS = {"8-K", "10-Q", "10-K", "S-1"}
EDGAR_UA = "stocker-research stocker@example.com"

# Pre-sampled GRPO episodes: a small catalog of 2-month windows scattered
# across years and sectors. Used by app.core.corpus_tasks to enrich training.
EPISODE_WINDOWS: list[dict[str, str]] = [
    # --- bull regimes -----------------------------------------------------
    {"task_id": "corpus_aapl_2014h2",  "ticker": "AAPL",  "start": "2014-08-01", "end": "2014-09-30"},
    {"task_id": "corpus_msft_2018h1",  "ticker": "MSFT",  "start": "2018-04-01", "end": "2018-05-31"},
    {"task_id": "corpus_nvda_2016h2",  "ticker": "NVDA",  "start": "2016-09-01", "end": "2016-10-31"},
    {"task_id": "corpus_amzn_2017h2",  "ticker": "AMZN",  "start": "2017-07-01", "end": "2017-08-31"},
    # --- drawdowns / shocks ----------------------------------------------
    {"task_id": "corpus_bac_2008",     "ticker": "BAC",   "start": "2008-09-01", "end": "2008-10-31"},
    {"task_id": "corpus_jpm_2008",     "ticker": "JPM",   "start": "2008-09-01", "end": "2008-10-31"},
    {"task_id": "corpus_gs_2011h2",    "ticker": "GS",    "start": "2011-08-01", "end": "2011-09-30"},
    {"task_id": "corpus_aapl_2020q1",  "ticker": "AAPL",  "start": "2020-02-15", "end": "2020-04-15"},
    {"task_id": "corpus_ba_2019h1",    "ticker": "BA",    "start": "2019-03-01", "end": "2019-04-30"},  # Max-8 grounding
    {"task_id": "corpus_xom_2014h2",   "ticker": "XOM",   "start": "2014-10-01", "end": "2014-11-30"},  # oil crash
    # --- choppy / sideways -----------------------------------------------
    {"task_id": "corpus_ko_2013h2",    "ticker": "KO",    "start": "2013-08-01", "end": "2013-09-30"},
    {"task_id": "corpus_wmt_2015h2",   "ticker": "WMT",   "start": "2015-09-01", "end": "2015-10-31"},
    {"task_id": "corpus_intc_2019",    "ticker": "INTC",  "start": "2019-04-01", "end": "2019-05-31"},
    {"task_id": "corpus_cvx_2016h1",   "ticker": "CVX",   "start": "2016-02-01", "end": "2016-03-31"},
    {"task_id": "corpus_meta_2021h2",  "ticker": "META",  "start": "2021-09-01", "end": "2021-10-31"},
    {"task_id": "corpus_googl_2022h1", "ticker": "GOOGL", "start": "2022-04-01", "end": "2022-05-31"},
]


# ---------------------------------------------------------------------------
def build_prices(universe: list[dict], start: str, end: str) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for entry in universe:
        t = entry["ticker"]
        try:
            df = yf.download(t, start=start, end=end, auto_adjust=False, progress=False)
        except Exception as e:
            print(f"  [warn] {t} fetch failed: {e}", file=sys.stderr)
            continue
        if df is None or df.empty:
            print(f"  [warn] {t} returned empty", file=sys.stderr)
            continue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        idx = pd.DatetimeIndex(pd.to_datetime(df.index)).tz_localize(None).normalize()  # type: ignore[attr-defined]
        df.index = idx
        df = df.loc[:, ["Open", "High", "Low", "Close", "Volume"]].copy()
        df.columns = ["open", "high", "low", "close", "volume"]
        df["ticker"] = t
        df["date"] = df.index.strftime("%Y-%m-%d")
        rows.append(df.reset_index(drop=True))
        print(f"  {t}: {len(df)} rows ({df['date'].iloc[0]}..{df['date'].iloc[-1]})")
    if not rows:
        return pd.DataFrame(columns=["ticker", "date", "open", "high", "low", "close", "volume"])
    out = pd.concat(rows, ignore_index=True)
    return out.loc[:, ["ticker", "date", "open", "high", "low", "close", "volume"]]


def build_indicators(prices: pd.DataFrame) -> pd.DataFrame:
    out_rows: list[pd.DataFrame] = []
    for ticker, group in prices.groupby("ticker"):
        df = group.sort_values("date").reset_index(drop=True)
        df.index = pd.to_datetime(df["date"])
        close = df["close"].astype(float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)

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
    out = pd.concat(out_rows, ignore_index=True) if out_rows else pd.DataFrame()
    cols = ["ticker", "date", "rsi14", "macd", "macd_signal",
            "sma20", "sma50", "sma200", "bb_lower", "bb_upper", "atr14"]
    return out.loc[:, cols] if not out.empty else pd.DataFrame(columns=cols)


def build_filings(universe: list[dict], start: str, end: str) -> pd.DataFrame:
    """Pull EDGAR submissions for each ticker. Filters to interesting forms.

    SEC requires a descriptive User-Agent. Submissions JSON is a single call
    per ticker that returns up to ~1000 recent filings.
    """
    import urllib.request

    start_ts = pd.to_datetime(start)
    end_ts = pd.to_datetime(end)

    rows: list[dict] = []
    for entry in universe:
        cik = entry["cik"].lstrip("0").zfill(10)
        url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        req = urllib.request.Request(url, headers={"User-Agent": EDGAR_UA, "Accept": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                payload = json.loads(resp.read())
        except Exception as e:
            print(f"  [warn] EDGAR {entry['ticker']}: {e}", file=sys.stderr)
            continue

        recent = payload.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        dates = recent.get("filingDate", [])
        accessions = recent.get("accessionNumber", [])
        descriptions = recent.get("primaryDocDescription", [])
        primary_docs = recent.get("primaryDocument", [])

        kept = 0
        for form, date, acc, desc, doc in zip(forms, dates, accessions, descriptions, primary_docs):
            if form not in EDGAR_FORMS:
                continue
            ts = pd.to_datetime(date)
            if ts < start_ts or ts > end_ts:
                continue
            acc_nodash = acc.replace("-", "")
            filing_url = (
                f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_nodash}/{doc}"
            )
            rows.append({
                "ticker": entry["ticker"],
                "date": ts.strftime("%Y-%m-%d"),
                "form": form,
                "title": desc or form,
                "accession": acc,
                "url": filing_url,
            })
            kept += 1
        print(f"  {entry['ticker']}: {kept} filings")
        # SEC asks for <= 10 req/sec; we're well under that with one call/ticker.
        time.sleep(0.15)
    return pd.DataFrame(rows, columns=["ticker", "date", "form", "title", "accession", "url"])


def build_news(universe: list[dict]) -> pd.DataFrame:
    """yfinance ticker.news — best-effort, recent-only."""
    rows: list[dict] = []
    for entry in universe:
        t = entry["ticker"]
        try:
            items = yf.Ticker(t).news or []
        except Exception as e:
            print(f"  [warn] news {t}: {e}", file=sys.stderr)
            continue
        kept = 0
        for it in items:
            content = it.get("content") or it
            title = content.get("title") or it.get("title")
            if not title:
                continue
            pub_ts = (
                content.get("pubDate")
                or content.get("displayTime")
                or content.get("providerPublishTime")
                or it.get("providerPublishTime")
            )
            if pub_ts is None:
                continue
            try:
                if isinstance(pub_ts, (int, float)):
                    date = pd.to_datetime(int(pub_ts), unit="s").strftime("%Y-%m-%d")
                else:
                    date = pd.to_datetime(pub_ts).strftime("%Y-%m-%d")
            except Exception:
                continue
            provider = (content.get("provider") or {}).get("displayName") or it.get("publisher") or ""
            link = (
                ((content.get("canonicalUrl") or {}).get("url"))
                or ((content.get("clickThroughUrl") or {}).get("url"))
                or it.get("link")
                or ""
            )
            rows.append({
                "ticker": t,
                "date": date,
                "headline": title,
                "publisher": provider,
                "url": link,
            })
            kept += 1
        print(f"  {t}: {kept} news items")
    return pd.DataFrame(rows, columns=["ticker", "date", "headline", "publisher", "url"])


def build_episodes(prices: pd.DataFrame) -> pd.DataFrame:
    """Filter EPISODE_WINDOWS to those with prices in the corpus."""
    available_tickers = set(prices["ticker"].unique())
    rows: list[dict] = []
    for w in EPISODE_WINDOWS:
        if w["ticker"] not in available_tickers:
            continue
        sub = prices[
            (prices["ticker"] == w["ticker"])
            & (prices["date"] >= w["start"])
            & (prices["date"] <= w["end"])
        ]
        if sub.empty:
            continue
        rows.append({
            "task_id": w["task_id"],
            "ticker": w["ticker"],
            "episode_start": w["start"],
            "episode_end": w["end"],
            "n_steps": len(sub),
        })
    return pd.DataFrame(rows, columns=["task_id", "ticker", "episode_start", "episode_end", "n_steps"])


# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true", help="3 tickers, 5y — smoke test")
    p.add_argument("--skip-filings", action="store_true")
    p.add_argument("--skip-news", action="store_true")
    p.add_argument("--start", default=None, help="ISO start date (default 2005-01-01 or 2020 for --quick)")
    p.add_argument("--end", default=None, help="ISO end date (default today)")
    args = p.parse_args()

    universe = (
        [u for u in UNIVERSE if u["ticker"] in QUICK_TICKERS] if args.quick else UNIVERSE
    )
    start = args.start or (QUICK_START if args.quick else DEFAULT_START)
    end = args.end or pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d")

    CORPUS.mkdir(parents=True, exist_ok=True)

    print(f"[corpus] universe={len(universe)} tickers, range={start}..{end}")
    (CORPUS / "tickers.json").write_text(json.dumps(universe, indent=2))

    print("[1/5] Fetching OHLCV ...")
    prices = build_prices(universe, start, end)
    prices.to_parquet(CORPUS / "prices.parquet", index=False)
    print(f"  prices: {len(prices)} rows across {prices['ticker'].nunique()} tickers")

    print("[2/5] Computing indicators ...")
    indicators = build_indicators(prices)
    indicators.to_parquet(CORPUS / "indicators.parquet", index=False)
    print(f"  indicators: {len(indicators)} rows")

    print("[3/5] Fetching EDGAR filings ..." + (" SKIPPED" if args.skip_filings else ""))
    if args.skip_filings:
        filings = pd.DataFrame(columns=["ticker", "date", "form", "title", "accession", "url"])
    else:
        filings = build_filings(universe, start, end)
    filings.to_parquet(CORPUS / "filings.parquet", index=False)
    print(f"  filings: {len(filings)} rows")

    print("[4/5] Fetching news ..." + (" SKIPPED" if args.skip_news else ""))
    if args.skip_news:
        news = pd.DataFrame(columns=["ticker", "date", "headline", "publisher", "url"])
    else:
        news = build_news(universe)
    news.to_parquet(CORPUS / "news.parquet", index=False)
    print(f"  news: {len(news)} rows")

    print("[5/5] Sampling episodes ...")
    episodes = build_episodes(prices)
    episodes.to_parquet(CORPUS / "episodes.parquet", index=False)
    print(f"  episodes: {len(episodes)} task windows")

    print(f"[corpus] done. Outputs in {CORPUS.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
