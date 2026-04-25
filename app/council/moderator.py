"""ModeratorAgent — merges 7 specialist votes into a single TradeAction."""
from __future__ import annotations

from app.council.llm import LLMClient, parse_json_object
from app.council.types import (
    CouncilDecision,
    MarketObservation,
    SpecialistVote,
    TradeAction,
)

MODERATOR_SYSTEM_PROMPT = (
    "You are the MODERATOR of a trading council. Seven specialists each emit a "
    "(signal, confidence, rationale) read of the same situation. Your job is "
    "to weigh those reads against the agent's current portfolio (cash, "
    "position, recent prices) and emit ONE concrete trade action.\n\n"
    "Rules:\n"
    "- side must be one of: buy, sell, hold\n"
    "- quantity is an integer >= 0 (ignored when side=hold)\n"
    "- buy: cost = quantity * price must be <= cash\n"
    "- sell: quantity must be <= current position\n"
    "- prefer scaled moves over all-in; favor hold when specialists strongly disagree\n"
    "- LONG-TERM lens, not HFT — do not churn\n\n"
    'Respond with ONLY a JSON object: '
    '{"side": "buy|sell|hold", "quantity": <int>, "rationale": "<one to three sentences>"}'
)


class Moderator:
    """Runs the moderator LLM with the moderator LoRA adapter (when configured)."""

    name = "moderator"
    role_keyword = "moderator"

    def __init__(self, client: LLMClient, lora_name: str | None = None):
        self.client = client
        self.lora_name = lora_name

    def decide(
        self, obs: MarketObservation, votes: list[SpecialistVote]
    ) -> CouncilDecision:
        messages = self._build_messages(obs, votes)
        extra_body = None
        if self.lora_name:
            extra_body = {"lora_request": {"name": self.lora_name}}
        text = self.client.complete(
            messages, max_tokens=256, temperature=0.2, extra_body=extra_body
        )
        data = parse_json_object(text)

        side = str(data.get("side", "hold")).lower()
        if side not in ("buy", "sell", "hold"):
            side = "hold"
        try:
            qty = int(data.get("quantity", 0))
        except (ValueError, TypeError):
            qty = 0
        qty = max(0, qty)

        # Constrain qty to what's actually feasible.
        if side == "buy":
            max_buy = int(obs.cash // max(obs.price, 1e-9))
            qty = min(qty, max_buy)
        elif side == "sell":
            qty = min(qty, obs.position)

        action = TradeAction(side=side, quantity=qty)
        rationale = str(data.get("rationale", text[:300])).strip()[:600]
        return CouncilDecision(votes=votes, action=action, rationale=rationale)

    # -----------------------------------------------------------------------
    def _build_messages(
        self, obs: MarketObservation, votes: list[SpecialistVote]
    ) -> list[dict]:
        vote_block = "\n".join(
            f"- {v.name:<16s} signal={v.signal:+.2f}  conf={v.confidence:.2f}  | {v.rationale[:200]}"
            for v in votes
        )
        body = (
            f"Ticker: {obs.ticker}    Date: {obs.date}\n"
            f"Price: {obs.price:.2f}    Cash: {obs.cash:.2f}    Position: {obs.position}    "
            f"Portfolio: {obs.portfolio_value:.2f}\n"
            f"Step {obs.step_number} of {obs.total_steps}\n\n"
            "Specialist votes:\n"
            f"{vote_block}\n\n"
            "Decide the trade."
        )
        return [
            {"role": "system", "content": MODERATOR_SYSTEM_PROMPT},
            {"role": "user", "content": body},
        ]
