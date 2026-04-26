"""OHLCV bars for chart rendering.

Returns the full per-task daily series (warmup + episode rows) so the chart
panel has historical context. Sourced from data/prices.parquet via
:mod:`app.data.loader`. The agent's MarketObservation is unaffected — this
endpoint is purely a UI concern.
"""

from fastapi import APIRouter, HTTPException, Query

from app.data.loader import prices

router = APIRouter(tags=["ohlcv"])


@router.get("/ohlcv")
async def get_ohlcv(task_id: str = Query(...)) -> dict:
    df = prices()
    sub = df[df["task_id"] == task_id].sort_values("date")
    if sub.empty:
        raise HTTPException(status_code=404, detail=f"no price rows for task {task_id}")

    ticker = str(sub["ticker"].iloc[0])
    cols = ["date", "open", "high", "low", "close", "volume", "in_episode"]
    bars = [
        {
            "time": str(r["date"]),
            "open": float(r["open"]),
            "high": float(r["high"]),
            "low": float(r["low"]),
            "close": float(r["close"]),
            "volume": float(r["volume"]),
            "in_episode": bool(r["in_episode"]),
        }
        for r in sub[cols].to_dict("records")
    ]
    return {"task_id": task_id, "ticker": ticker, "bars": bars}
