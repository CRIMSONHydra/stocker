"""Council orchestrator: runs 7 specialists in parallel, then the moderator.

Public surface:
- `Council`: holds the LLM client, specialist instances, moderator, cache config
- `Council.run(obs) -> CouncilDecision`
- `Council.run_async(obs) -> CouncilDecision`

Cache layout:
  .cache/council/<role>/<lora>/<ticker>__<date>.json
  .cache/council/moderator/<lora>/<ticker>__<date>__<votes_hash>.json
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from app.council.llm import LLMClient
from app.council.moderator import Moderator
from app.council.specialists import SPECIALISTS, Specialist
from app.council.types import CouncilDecision, MarketObservation, SpecialistVote

logger = logging.getLogger(__name__)


CACHE_ROOT = Path(__file__).resolve().parent.parent.parent / ".cache" / "council"


@dataclass
class Council:
    client: LLMClient
    moderator_lora: Optional[str] = None
    use_cache: bool = True
    specialists: list[Specialist] = field(init=False)
    moderator: Moderator = field(init=False)

    def __post_init__(self):
        self.specialists = [cls(self.client) for cls in SPECIALISTS]
        self.moderator = Moderator(self.client, lora_name=self.moderator_lora)

    # ------------------------------------------------------------------ sync
    def run(self, obs: MarketObservation) -> CouncilDecision:
        return asyncio.run(self.run_async(obs))

    # ----------------------------------------------------------------- async
    async def run_async(self, obs: MarketObservation) -> CouncilDecision:
        loop = asyncio.get_event_loop()

        with ThreadPoolExecutor(max_workers=len(self.specialists)) as pool:
            tasks = [
                loop.run_in_executor(pool, self._cached_vote, sp, obs)
                for sp in self.specialists
            ]
            votes: list[SpecialistVote] = await asyncio.gather(*tasks)

        return self._cached_decide(obs, votes)

    # ------------------------------------------------------------- internals
    def _cached_vote(self, sp: Specialist, obs: MarketObservation) -> SpecialistVote:
        if not self.use_cache:
            return sp.vote(obs)

        key = self._vote_cache_path(sp, obs)
        if key.exists():
            try:
                return SpecialistVote.model_validate_json(key.read_text())
            except Exception:
                pass

        vote = sp.vote(obs)
        key.parent.mkdir(parents=True, exist_ok=True)
        key.write_text(vote.model_dump_json())
        return vote

    def _cached_decide(
        self, obs: MarketObservation, votes: list[SpecialistVote]
    ) -> CouncilDecision:
        if not self.use_cache:
            return self.moderator.decide(obs, votes)

        key = self._mod_cache_path(obs, votes)
        if key.exists():
            try:
                return CouncilDecision.model_validate_json(key.read_text())
            except Exception:
                pass

        dec = self.moderator.decide(obs, votes)
        key.parent.mkdir(parents=True, exist_ok=True)
        key.write_text(dec.model_dump_json())
        return dec

    def _vote_cache_path(self, sp: Specialist, obs: MarketObservation) -> Path:
        return (
            CACHE_ROOT
            / sp.role_keyword
            / "base"
            / f"{obs.ticker}__{obs.date}.json"
        )

    def _mod_cache_path(
        self, obs: MarketObservation, votes: list[SpecialistVote]
    ) -> Path:
        votes_blob = json.dumps(
            [v.model_dump() for v in votes], sort_keys=True
        ).encode()
        votes_hash = hashlib.md5(votes_blob).hexdigest()[:10]
        lora = self.moderator_lora or "base"
        return (
            CACHE_ROOT
            / "moderator"
            / lora
            / f"{obs.ticker}__{obs.date}__{votes_hash}.json"
        )
