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
    sub = df[df["task_id"] == task_id].sort_values("date").reset_index(drop=True)
    if sub.empty:
        raise HTTPException(status_code=404, detail=f"no price rows for task {task_id}")

    ticker = str(sub.iloc[0]["ticker"])
    bars = [
        {
            "time": str(row["date"]),
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row["volume"]),
            "in_episode": bool(row["in_episode"]),
        }
        for _, row in sub.iterrows()
    ]
    return {"task_id": task_id, "ticker": ticker, "bars": bars}
