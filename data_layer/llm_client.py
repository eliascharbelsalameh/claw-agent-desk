"""NVIDIA Build (integrate.api.nvidia.com) chat-completions client.

OpenAI-compatible endpoint (spec section 4). Retries on 429/5xx come for
free from http_utils.request_with_retry, same as every other data-layer
call. On top of that, calls are paced client-side per model against
DEFAULT_RPM_LIMIT: the Build free/trial tier's ~40 requests/minute is only
confirmed for a subset of models (checked directly on build.nvidia.com),
and unpublished limits are assumed to be no more generous than that, so
every model gets the same conservative budget unless overridden via
rpm_limits.

AGENT_MODELS assigns a different model family to each agent role (spec
section 4's "Design choice": independent failure modes, and rate limits
that don't stack if they're per-model). Every id below was live-verified
with a real chat_completion call against build.nvidia.com (Sept 2026, see
claw_agent_spec.md section 4) - not just present in GET /v1/models, since
the catalog lists plenty of models ("Not found for account") that 404 on
an actual call, and a couple that were flat-out 410 Gone (end of life).
Re-verify against your own Build account before a real run regardless,
since this catalog changes often.

Several of these are reasoning models that emit hidden chain-of-thought
before any visible content (OpenAI's gpt-oss, NVIDIA's Nemotron 3 family,
Z.ai's GLM, Moonshot's Kimi) - two consequences for callers: give them a
generous max_tokens or they'll hit finish_reason "length" with empty
content, and DEFAULT_TIMEOUT below is much higher than other data-layer
calls because gpt-oss-20b alone took up to ~90s to respond in testing.
"""
from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import Any

import requests

from .config import Settings, get_settings
from .http_utils import request_with_retry

DEFAULT_RPM_LIMIT = 40
DEFAULT_TIMEOUT = 120.0

AGENT_MODELS = {
    "macro": "nvidia/nemotron-3.5-lightning-30b-a3b",
    "analyst_1": "openai/gpt-oss-20b",
    "analyst_2": "nvidia/nemotron-3-super-120b-a12b",
    "critic": "z-ai/glm-5.3",
    "bias_1": "google/gemma-4-31b-it",
    "bias_2": "mistralai/mistral-nemotron",
    "technical": "moonshotai/kimi-k3",
}


class LlmClient:
    def __init__(
        self,
        settings: Settings | None = None,
        session: requests.Session | None = None,
        rpm_limits: dict[str, int] | None = None,
    ):
        self._settings = settings or get_settings()
        self._session = session or requests.Session()
        self._rpm_limits = rpm_limits or {}
        self._call_times: dict[str, deque[float]] = defaultdict(deque)

    def _wait_for_rate_limit(self, model: str) -> None:
        limit = self._rpm_limits.get(model, DEFAULT_RPM_LIMIT)
        window = self._call_times[model]
        now = time.monotonic()
        while window and now - window[0] > 60.0:
            window.popleft()
        if len(window) >= limit:
            sleep_for = 60.0 - (now - window[0])
            if sleep_for > 0:
                time.sleep(sleep_for)
            now = time.monotonic()
            while window and now - window[0] > 60.0:
                window.popleft()
        window.append(time.monotonic())

    def chat_completion(
        self,
        model: str,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        **kwargs: Any,
    ) -> dict[str, Any]:
        self._wait_for_rate_limit(model)
        url = f"{self._settings.nvidia_base_url}/chat/completions"
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        payload.update(kwargs)

        response = request_with_retry(
            self._session,
            "POST",
            url,
            headers={
                "Authorization": f"Bearer {self._settings.require('nvidia_api_key')}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=timeout,
        )
        response.raise_for_status()
        return response.json()

    def complete_text(
        self,
        model: str,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> str:
        data = self.chat_completion(model, messages, **kwargs)
        return data["choices"][0]["message"]["content"]
