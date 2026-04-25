"""The 7 specialist analyst agents."""
from __future__ import annotations

from typing import Any

from app.council.llm import LLMClient, encode_image_url, parse_json_object
from app.council.types import MarketObservation, SpecialistVote


SPECIALIST_RESPONSE_SCHEMA = (
    'Respond with ONLY a JSON object: '
    '{"signal": <float in [-1,1]>, "confidence": <float in [0,1]>, '
    '"rationale": "<one to three sentences>"}. '
    'signal: +1 = strong buy bias, -1 = strong sell bias, 0 = neutral.'
)


class Specialist:
    """Base class. Subclasses set ``name`` and override ``prepare_messages``."""

    name: str = "base"
    role_keyword: str = "base"  # used by MockLLMClient + cache key

    SYSTEM_PROMPT: str = "You are an analyst on a trading council."

    def __init__(self, client: LLMClient):
        self.client = client

    # ------------------------------------------------------------------ public
    def vote(self, obs: MarketObservation) -> SpecialistVote:
        messages = self.prepare_messages(obs)
        text = self.client.complete(messages, max_tokens=256, temperature=0.2)
        data = parse_json_object(text)
        return SpecialistVote(
            name=self.name,
            signal=_clip(_to_float(data.get("signal"), 0.0), -1.0, 1.0),
            confidence=_clip(_to_float(data.get("confidence"), 0.5), 0.0, 1.0),
            rationale=str(data.get("rationale", text[:300])).strip()[:400],
        )

    # --------------------------------------------------------------- override
    def prepare_messages(self, obs: MarketObservation) -> list[dict]:
        raise NotImplementedError


# ---------------------------------------------------------------------------
class ChartPatternAgent(Specialist):
    name = "chart_pattern"
    role_keyword = "chart-pattern"
    SYSTEM_PROMPT = (
        "You are the CHART-PATTERN specialist on a trading council. "
        "You read candlestick charts and identify visual patterns: "
        "support/resistance, breakouts, double tops/bottoms, head-and-shoulders, "
        "trend channels, gaps. Long-term lens (weeks-to-months), not HFT. "
        + SPECIALIST_RESPONSE_SCHEMA
    )

    def prepare_messages(self, obs: MarketObservation) -> list[dict]:
        user_text = (
            f"Ticker: {obs.ticker}    As of: {obs.date}\n"
            f"Current price: {obs.price:.2f}\n"
            "Identify the dominant chart pattern and your directional bias."
        )
        content: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
        if obs.chart_path:
            data_url = encode_image_url(obs.chart_path)
            if data_url:
                content.append({"type": "image_url", "image_url": {"url": data_url}})
        return [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]


# ---------------------------------------------------------------------------
class SeasonalTrendAgent(Specialist):
    name = "seasonal_trend"
    role_keyword = "seasonal"
    SYSTEM_PROMPT = (
        "You are the SEASONAL/LONG-TERM-TREND specialist on a trading council. "
        "You evaluate multi-month and seasonal patterns: secular trends, "
        "calendar effects, sector cycles. Long-term lens. "
        + SPECIALIST_RESPONSE_SCHEMA
    )

    def prepare_messages(self, obs: MarketObservation) -> list[dict]:
        history = obs.price_history
        if len(history) >= 2:
            ret_pct = (history[-1] / history[0] - 1.0) * 100
            ret_str = f"{ret_pct:+.2f}%"
        else:
            ret_str = "n/a"
        head = ", ".join(f"{p:.2f}" for p in history[: min(5, len(history))])
        tail = ", ".join(f"{p:.2f}" for p in history[-min(5, len(history)) :])
        user = (
            f"Ticker: {obs.ticker}    As of: {obs.date}\n"
            f"Episode prices so far ({len(history)} days): "
            f"first={[head]} ... last={[tail]}\n"
            f"Cumulative return so far: {ret_str}\n"
            f"Sector: {obs.fundamentals.get('sector', '?')}\n"
            "Give your long-term/seasonal bias."
        )
        return [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ]


# ---------------------------------------------------------------------------
class IndicatorAgent(Specialist):
    name = "indicator"
    role_keyword = "indicator"
    SYSTEM_PROMPT = (
        "You are the TECHNICAL-INDICATOR specialist. You read RSI, MACD, "
        "moving averages (SMA20/50/200), Bollinger Bands and ATR. "
        + SPECIALIST_RESPONSE_SCHEMA
    )

    def prepare_messages(self, obs: MarketObservation) -> list[dict]:
        ind = obs.indicators or {}

        def fmt(v: Any) -> str:
            return "n/a" if v is None else f"{float(v):.3f}"

        body = (
            f"Ticker: {obs.ticker}    As of: {obs.date}\n"
            f"Price: {obs.price:.2f}\n"
            f"RSI(14): {fmt(ind.get('rsi14'))}\n"
            f"MACD: {fmt(ind.get('macd'))} / signal: {fmt(ind.get('macd_signal'))}\n"
            f"SMA20: {fmt(ind.get('sma20'))} | SMA50: {fmt(ind.get('sma50'))} | "
            f"SMA200: {fmt(ind.get('sma200'))}\n"
            f"BB: [{fmt(ind.get('bb_lower'))}, {fmt(ind.get('bb_upper'))}]\n"
            f"ATR(14): {fmt(ind.get('atr14'))}\n"
            "Read these indicators and give your bias."
        )
        return [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": body},
        ]


# ---------------------------------------------------------------------------
class NewsAgent(Specialist):
    name = "news"
    role_keyword = "news"
    SYSTEM_PROMPT = (
        "You are the NEWS specialist on a trading council. You read recent "
        "headlines and decide whether the news flow is bullish, bearish, or "
        "neutral for this ticker on a multi-week horizon. "
        + SPECIALIST_RESPONSE_SCHEMA
    )

    def prepare_messages(self, obs: MarketObservation) -> list[dict]:
        if obs.headlines:
            lines = [
                f"  [{h['date']}] ({h.get('sentiment_label','?')}/{h.get('source','?')}) {h['headline']}"
                for h in obs.headlines[-10:]
            ]
            news_block = "\n".join(lines)
        else:
            news_block = "  (no headlines in lookback window)"

        body = (
            f"Ticker: {obs.ticker}    As of: {obs.date}\n"
            "Recent news (past ~7 days):\n"
            f"{news_block}\n"
            "Give your news-flow bias."
        )
        return [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": body},
        ]


# ---------------------------------------------------------------------------
class ForumSentimentAgent(Specialist):
    name = "forum_sentiment"
    role_keyword = "forum"
    SYSTEM_PROMPT = (
        "You are the FORUM-SENTIMENT specialist on a trading council. You "
        "read Reddit excerpts (wallstreetbets, stocks, investing) and detect "
        "retail sentiment and crowding. Be skeptical of euphoria. "
        + SPECIALIST_RESPONSE_SCHEMA
    )

    def prepare_messages(self, obs: MarketObservation) -> list[dict]:
        if obs.forum_excerpts:
            lines = [
                f"  [{p['date']}] r/{p['subreddit']} ({p.get('score','?')} upvotes): {p['post_text']}"
                for p in obs.forum_excerpts[-10:]
            ]
            forum_block = "\n".join(lines)
        else:
            forum_block = "  (no forum chatter in lookback window)"

        body = (
            f"Ticker: {obs.ticker}    As of: {obs.date}\n"
            "Recent forum chatter (past ~7 days):\n"
            f"{forum_block}\n"
            "Give your retail-sentiment bias."
        )
        return [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": body},
        ]


# ---------------------------------------------------------------------------
class PeerCommodityAgent(Specialist):
    name = "peer_commodity"
    role_keyword = "peer-commodity"
    SYSTEM_PROMPT = (
        "You are the PEER & COMMODITY specialist on a trading council. You "
        "compare the ticker's path to its peers and to a relevant commodity "
        "(gold, oil) to detect relative strength or sector rotation. "
        + SPECIALIST_RESPONSE_SCHEMA
    )

    def prepare_messages(self, obs: MarketObservation) -> list[dict]:
        peers = (obs.peers or {}).get("peers", []) or []
        com = obs.peers.get("commodity") if obs.peers else None
        com_price = obs.peers.get("commodity_price") if obs.peers else None

        peer_lines = ", ".join(
            f"{p['peer_ticker']}={p.get('peer_close','n/a')}" for p in peers
        ) or "n/a"
        com_str = f"{com}={com_price}" if com else "n/a"

        body = (
            f"Ticker: {obs.ticker} @ {obs.price:.2f}    As of: {obs.date}\n"
            f"Peers today: {peer_lines}\n"
            f"Commodity proxy: {com_str}\n"
            "Give your relative-positioning bias."
        )
        return [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": body},
        ]


# ---------------------------------------------------------------------------
class GeopoliticsAgent(Specialist):
    name = "geopolitics"
    role_keyword = "geopolitics"
    SYSTEM_PROMPT = (
        "You are the GEOPOLITICS / MACRO-POLICY specialist on a trading "
        "council. You read government and central-bank actions (CPI, FOMC, "
        "fiscal, sanctions) and decide whether the macro environment is "
        "supportive or hostile for risk assets. "
        + SPECIALIST_RESPONSE_SCHEMA
    )

    def prepare_messages(self, obs: MarketObservation) -> list[dict]:
        if obs.macro:
            lines = [
                f"  [{m['date']}] ({m.get('country','?')}/{m.get('policy_signal','?')}) {m['headline']}"
                for m in obs.macro[-10:]
            ]
            block = "\n".join(lines)
        else:
            block = "  (no macro events in lookback window)"

        body = (
            f"As of: {obs.date}\n"
            "Recent macro events (past ~14 days):\n"
            f"{block}\n"
            "Give your macro/policy bias for risk assets."
        )
        return [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": body},
        ]


# ---------------------------------------------------------------------------
SPECIALISTS: list[type[Specialist]] = [
    ChartPatternAgent,
    SeasonalTrendAgent,
    IndicatorAgent,
    NewsAgent,
    ForumSentimentAgent,
    PeerCommodityAgent,
    GeopoliticsAgent,
]


# ---------------------------------------------------------------------------
def _to_float(v: Any, default: float) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))
