"""Multi-agent council layer.

Public surface:
- Specialist (base class), 7 concrete subclasses
- Moderator (merges votes -> TradeAction)
- run_council (asyncio orchestrator, parallel calls + cache)
- LLMClient protocol + OpenAILLMClient + MockLLMClient
"""
from app.council.specialists import (
    ChartPatternAgent,
    ForumSentimentAgent,
    GeopoliticsAgent,
    IndicatorAgent,
    NewsAgent,
    PeerCommodityAgent,
    SeasonalTrendAgent,
    Specialist,
    SPECIALISTS,
)

__all__ = [
    "Specialist",
    "ChartPatternAgent",
    "SeasonalTrendAgent",
    "IndicatorAgent",
    "NewsAgent",
    "ForumSentimentAgent",
    "PeerCommodityAgent",
    "GeopoliticsAgent",
    "SPECIALISTS",
]
