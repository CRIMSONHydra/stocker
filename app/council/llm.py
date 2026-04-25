"""LLM client abstraction for the council.

Specialists and the moderator only depend on `LLMClient.complete(messages,
extra_body, max_tokens, temperature) -> str`. Two concrete clients:

- `OpenAILLMClient` — wraps the OpenAI SDK (which speaks the vLLM /
  HF-Endpoints OpenAI-compatible API).
- `MockLLMClient` — deterministic, no network, used by tests + `--mock`.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Protocol


class LLMClient(Protocol):
    def complete(
        self,
        messages: list[dict],
        *,
        max_tokens: int = 256,
        temperature: float = 0.2,
        extra_body: dict | None = None,
    ) -> str: ...


@dataclass
class OpenAILLMClient:
    """OpenAI-compatible client. Lazily creates the underlying SDK client."""
    base_url: str
    api_key: str
    model: str
    _sdk: Any = None

    def __post_init__(self):
        from openai import OpenAI  # imported here so test envs without `openai` still work
        self._sdk = OpenAI(base_url=self.base_url, api_key=self.api_key)

    def complete(
        self,
        messages: list[dict],
        *,
        max_tokens: int = 256,
        temperature: float = 0.2,
        extra_body: dict | None = None,
    ) -> str:
        kwargs: dict = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if extra_body:
            kwargs["extra_body"] = extra_body
        resp = self._sdk.chat.completions.create(**kwargs)
        return (resp.choices[0].message.content or "").strip()


class MockLLMClient:
    """Deterministic stand-in for offline runs. Routes by system prompt
    keyword so each specialist gets a plausibly-shaped response."""

    SIGNAL_BIAS = {
        "chart-pattern": +0.15,
        "seasonal":      +0.05,
        "indicator":     -0.05,
        "news":           0.00,
        "forum":          0.00,
        "peer-commodity": +0.05,
        "geopolitics":   -0.10,
        "moderator":      0.00,
    }

    def complete(
        self,
        messages: list[dict],
        *,
        max_tokens: int = 256,
        temperature: float = 0.2,
        extra_body: dict | None = None,
    ) -> str:
        # Find the role keyword from the system prompt
        sys = next((m["content"] for m in messages if m.get("role") == "system"), "")
        role = "moderator"
        for k in self.SIGNAL_BIAS:
            if k in sys.lower():
                role = k
                break

        # Hash user content for deterministic small jitter
        user_blob = json.dumps(
            [m for m in messages if m.get("role") == "user"], default=str, sort_keys=True
        )
        h = int(hashlib.md5(user_blob.encode()).hexdigest(), 16)
        jitter = (((h % 1000) / 1000.0) - 0.5) * 0.4   # in [-0.2, +0.2]
        bias = self.SIGNAL_BIAS[role]
        signal = max(-1.0, min(1.0, bias + jitter))
        confidence = 0.4 + ((h // 1000) % 600) / 1000.0  # in [0.4, 1.0]

        if role == "moderator":
            # Moderator emits a TradeAction
            avg_signal = signal  # downstream code re-computes from votes anyway
            if avg_signal > 0.2:
                side, qty = "buy", 5
            elif avg_signal < -0.2:
                side, qty = "sell", 5
            else:
                side, qty = "hold", 0
            return json.dumps({
                "side": side,
                "quantity": qty,
                "rationale": f"mock moderator: avg_signal={avg_signal:.2f}",
            })

        return json.dumps({
            "signal": round(signal, 3),
            "confidence": round(confidence, 3),
            "rationale": f"mock {role} read of the situation",
        })


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def parse_json_object(text: str) -> dict:
    """Parse the first balanced JSON object from a model response."""
    cleaned = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`")
    start = cleaned.find("{")
    if start < 0:
        return {}
    depth = 0
    for i in range(start, len(cleaned)):
        if cleaned[i] == "{":
            depth += 1
        elif cleaned[i] == "}":
            depth -= 1
        if depth == 0:
            try:
                return json.loads(cleaned[start : i + 1])
            except json.JSONDecodeError:
                return {}
    return {}


def build_openai_client_from_env() -> OpenAILLMClient:
    api_key = (
        os.getenv("HF_TOKEN")
        or os.getenv("API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or "sk-local"
    )
    base_url = os.getenv("API_BASE_URL") or "http://localhost:8000/v1"
    model = os.getenv("MODEL_NAME") or "google/gemma-4-E4B-it"
    return OpenAILLMClient(base_url=base_url, api_key=api_key, model=model)


# ---------------------------------------------------------------------------
# Transformers-direct client (for Colab / single-process inference)
# ---------------------------------------------------------------------------
@dataclass
class TransformersLLMClient:
    """Calls a loaded HF transformers model in-process. Used by Colab/local
    when there's no separate vLLM server. Multimodal-aware: image_url parts
    are decoded back to PIL.Image and passed via the AutoProcessor.

    Construct with `from_pretrained(model_id, ...)` for a one-liner setup,
    or pass an already-loaded `model` + `processor` to share weights with
    a TRL trainer (so the GPU only holds one copy of Gemma).
    """
    model: Any
    processor: Any
    moderator_lora: Optional[str] = None  # PEFT adapter name when active

    @classmethod
    def from_pretrained(
        cls,
        model_id: str = "google/gemma-4-E4B-it",
        load_in_4bit: bool = True,
        device_map: str = "auto",
    ) -> "TransformersLLMClient":
        from transformers import AutoModelForCausalLM, AutoProcessor

        kwargs: dict = {"dtype": "auto", "device_map": device_map}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype="bfloat16",
                bnb_4bit_quant_type="nf4",
            )
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        model.eval()
        return cls(model=model, processor=processor)

    def complete(
        self,
        messages: list[dict],
        *,
        max_tokens: int = 256,
        temperature: float = 0.2,
        extra_body: dict | None = None,
    ) -> str:
        # Decode any embedded data-URL images back into PIL.Image objects
        # the processor expects.
        prepped = self._prep_messages(messages)
        inputs = self.processor.apply_chat_template(
            prepped,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        gen_kwargs: dict = {
            "max_new_tokens": max_tokens,
            "do_sample": temperature > 0.0,
        }
        if temperature > 0.0:
            gen_kwargs["temperature"] = temperature

        # Activate / deactivate the moderator LoRA if requested
        adapter_name = (
            (extra_body or {}).get("lora_request", {}).get("name")
            if extra_body
            else None
        )
        active_adapter = self._set_adapter(adapter_name)

        try:
            import torch
            with torch.no_grad():
                out = self.model.generate(**inputs, **gen_kwargs)
        finally:
            self._restore_adapter(active_adapter)

        gen_only = out[:, inputs["input_ids"].shape[1] :]
        return self.processor.batch_decode(gen_only, skip_special_tokens=True)[0].strip()

    # ----------------------------------------------------------------- helpers
    @staticmethod
    def _prep_messages(messages: list[dict]) -> list[dict]:
        import base64
        from io import BytesIO

        from PIL import Image

        out = []
        for m in messages:
            content = m.get("content")
            if isinstance(content, list):
                new_parts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "image_url":
                        url = part["image_url"]["url"]
                        if url.startswith("data:image"):
                            b64 = url.split(",", 1)[1]
                            img = Image.open(BytesIO(base64.b64decode(b64))).convert("RGB")
                            new_parts.append({"type": "image", "image": img})
                            continue
                    new_parts.append(part)
                out.append({**m, "content": new_parts})
            else:
                out.append(m)
        return out

    def _set_adapter(self, name: str | None):
        if name is None:
            return None
        peft_model = getattr(self.model, "peft_config", None)
        if not peft_model:
            return None
        try:
            current = self.model.active_adapter
            self.model.set_adapter(name)
            return current
        except Exception:
            return None

    def _restore_adapter(self, prev):
        if prev is None:
            return
        try:
            self.model.set_adapter(prev)
        except Exception:
            pass


def encode_image_url(path: str) -> str:
    """Encode a local PNG as a data URL for OpenAI-style image messages."""
    import base64
    from pathlib import Path

    p = Path(path)
    if not p.exists():
        return ""
    return "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode()
