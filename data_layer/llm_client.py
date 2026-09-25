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
Nemotron models accept chat_template_kwargs={"enable_thinking": False} to
skip the reasoning entirely - worth it for roles that only summarize.

Calls stream by default. Build's gateway closes any connection that sends
no bytes for 60s ("Remote end closed connection without response",
reproduced live Sept 2026), so a non-streaming call to a reasoning model
dies before DEFAULT_TIMEOUT ever applies; streaming keeps bytes flowing
and has run past 5 minutes. The chunks are reassembled into the same
shape as a non-streaming response, so callers don't care which was used.
Build also drops streams mid-response under load; those are retried
STREAM_RETRIES times on top of request_with_retry's own retries.
"""
from __future__ import annotations

import json
import logging
import random
import time
from collections import defaultdict, deque
from typing import Any

import requests

from .config import Settings, get_settings
from .http_utils import request_with_retry

logger = logging.getLogger(__name__)

DEFAULT_RPM_LIMIT = 40
DEFAULT_TIMEOUT = 120.0
STREAM_RETRIES = 2

# analyst_1 was openai/gpt-oss-20b until Sept 25, 2026: it failed 5 of 15
# calls (connection retries exhausted) and took 65-400s when it answered.
# gemma-4-31b-it was the only non-Nemotron candidate that answered reliably
# in a side-by-side screen (glm-5.3/-flash and kimi-k3 mostly dropped;
# kimi-k2.6, mistral-large-2, palmyra-fin 404 on this account), and at
# temperature 0 it returned identical verdicts across repeats.
# bias_1 still points at the same model and must be reassigned before the
# bias agents are built, or one model would be checking its own analysis.
# critic was z-ai/glm-5.3 until Sept 25, 2026: screened on the real task
# (reviewing a live buy/hold split), glm-5.3, kimi-k3 and deepseek-v4.1-flash
# all exhausted connection retries, glm-5.3-flash spent its whole token
# budget reasoning and returned nothing, and five others 404 on this account.
# muse-glimmer-30b (Meta, independent of both analysts' labs) answered in 19s
# with symmetric, fact-grounded challenges.
AGENT_MODELS = {
    "macro": "nvidia/nemotron-3.5-lightning-30b-a3b",
    "analyst_1": "google/gemma-4-31b-it",
    "analyst_2": "nvidia/nemotron-3-super-120b-a12b",
    "critic": "meta/muse-glimmer-30b",
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
        stream: bool = True,
        **kwargs: Any,
    ) -> dict[str, Any]:
        url = f"{self._settings.nvidia_base_url}/chat/completions"
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if stream:
            payload["stream"] = True
            payload["stream_options"] = {"include_usage": True}
        payload.update(kwargs)
        headers = {
            "Authorization": f"Bearer {self._settings.require('nvidia_api_key')}",
            "Content-Type": "application/json",
        }

        attempt = 0
        while True:
            self._wait_for_rate_limit(model)
            response = request_with_retry(
                self._session,
                "POST",
                url,
                headers=headers,
                json=payload,
                timeout=timeout,
                stream=stream,
            )
            response.raise_for_status()
            if not stream:
                return response.json()
            try:
                return _read_stream(response)
            except (requests.ConnectionError, requests.exceptions.ChunkedEncodingError) as exc:
                attempt += 1
                if attempt > STREAM_RETRIES:
                    raise
                delay = 2.0 * attempt + random.uniform(0, 0.25)
                logger.warning(
                    "stream from %s dropped (%s), retrying in %.1fs (attempt %d/%d)",
                    model, type(exc).__name__, delay, attempt, STREAM_RETRIES,
                )
                time.sleep(delay)

    def complete_text(
        self,
        model: str,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> str:
        data = self.chat_completion(model, messages, **kwargs)
        return data["choices"][0]["message"]["content"]


def _read_stream(response: requests.Response) -> dict[str, Any]:
    """Reassemble an OpenAI-style SSE stream into a non-streaming response.

    Reasoning arrives in delta.reasoning_content (or delta.reasoning,
    depending on the model) and is kept separate from the visible content.
    """
    content: list[str] = []
    reasoning: list[str] = []
    finish_reason: str | None = None
    usage: dict[str, Any] | None = None
    response_id: str | None = None
    model: str | None = None
    for raw in response.iter_lines():
        line = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        if not line.startswith("data:"):
            continue
        data = line[len("data:"):].strip()
        if data == "[DONE]":
            break
        chunk = json.loads(data)
        response_id = response_id or chunk.get("id")
        model = model or chunk.get("model")
        if chunk.get("usage"):
            usage = chunk["usage"]
        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}
            if delta.get("content"):
                content.append(delta["content"])
            thought = delta.get("reasoning_content") or delta.get("reasoning")
            if thought:
                reasoning.append(thought)
            finish_reason = choice.get("finish_reason") or finish_reason

    if finish_reason is None:
        # Every completed stream from Build carries a finish_reason. Without
        # one the stream was cut short - live, Build closed streams after
        # ~1s with no content at all, which callers then saw as an empty
        # reply instead of a retryable drop.
        raise requests.exceptions.ChunkedEncodingError(
            f"stream ended without a finish_reason after {len(''.join(content))} content chars"
        )
    message: dict[str, Any] = {"role": "assistant", "content": "".join(content)}
    if reasoning:
        message["reasoning_content"] = "".join(reasoning)
    return {
        "id": response_id,
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": usage,
    }
