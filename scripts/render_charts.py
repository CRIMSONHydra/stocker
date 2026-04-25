#!/usr/bin/env python3
"""Render a single candlestick PNG. Used by build_dataset.py."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import mplfinance as mpf
import pandas as pd


def render_chart(window: pd.DataFrame, path: Path, title: str = "") -> None:
    """Render a 768x768 candlestick PNG for the given OHLCV window."""
    path.parent.mkdir(parents=True, exist_ok=True)
    style = mpf.make_mpf_style(base_mpf_style="charles", rc={"font.size": 10})
    mpf.plot(
        window,
        type="candle",
        volume=True,
        style=style,
        title=title,
        savefig=dict(fname=str(path), dpi=96, bbox_inches="tight"),
        figsize=(8, 8),
        warn_too_much_data=10000,
    )
