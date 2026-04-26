"""Corpus browse endpoints — surfaces the data/corpus/ tables over HTTP.

These are read-only summaries intended for the demo / judges. They show
that Stocker is no longer married to 3 hand-picked tasks.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from app.data import corpus

router = APIRouter(prefix="/corpus", tags=["corpus"])


@router.get("/summary")
async def summary() -> dict:
    if not corpus.available():
        return {"available": False, "hint": "Run scripts/build_corpus.py"}
    px = corpus.prices()
    eps = corpus.episodes()
    fls = corpus.filings()
    nws = corpus.news()
    by_ticker: dict[str, dict] = {}
    if not px.empty:
        for t, grp in px.groupby("ticker"):
            by_ticker[str(t)] = {
                "rows": int(len(grp)),
                "start": str(grp["date"].min()),
                "end": str(grp["date"].max()),
            }
    return {
        "available": True,
        "tickers": corpus.tickers(),
        "n_price_rows": int(len(px)),
        "n_filings": int(len(fls)),
        "n_news": int(len(nws)),
        "n_episodes": int(len(eps)),
        "by_ticker": by_ticker,
    }


@router.get("/episodes")
async def episodes() -> list[dict]:
    if not corpus.available():
        return []
    eps = corpus.episodes()
    if eps.empty:
        return []
    return eps.to_dict(orient="records")


@router.get("/filings")
async def filings(
    ticker: str = Query(..., description="Ticker symbol, e.g. AAPL"),
    limit: int = Query(50, ge=1, le=500),
) -> list[dict]:
    fls = corpus.filings()
    if fls.empty:
        return []
    sub = fls[fls["ticker"] == ticker.upper()].sort_values("date", ascending=False).head(limit)
    return sub.to_dict(orient="records")


@router.get("/news")
async def news(
    ticker: str = Query(..., description="Ticker symbol, e.g. AAPL"),
    limit: int = Query(50, ge=1, le=500),
) -> list[dict]:
    nws = corpus.news()
    if nws.empty:
        return []
    sub = nws[nws["ticker"] == ticker.upper()].sort_values("date", ascending=False).head(limit)
    return sub.to_dict(orient="records")


@router.get("/prices")
async def prices(
    ticker: str = Query(..., description="Ticker symbol, e.g. AAPL"),
    start: str | None = Query(None, description="ISO date inclusive"),
    end: str | None = Query(None, description="ISO date inclusive"),
    limit: int = Query(500, ge=1, le=10000),
) -> list[dict]:
    px = corpus.ticker_prices(ticker.upper())
    if px.empty:
        raise HTTPException(404, f"No corpus prices for ticker {ticker!r}")
    if start:
        px = px[px["date"] >= start]
    if end:
        px = px[px["date"] <= end]
    return px.head(limit).to_dict(orient="records")
