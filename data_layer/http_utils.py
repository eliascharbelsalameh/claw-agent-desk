"""Shared HTTP retry logic.

Every outbound call in the data layer goes through request_with_retry so
429s and transient 5xxs are handled the same way everywhere (spec section 4:
"Must have: retry with backoff on HTTP 429 in every LLM call" - applied here
to every data-source call too).
"""
from __future__ import annotations

import logging
import random
import time
from typing import Any

import requests

logger = logging.getLogger(__name__)

RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})


class DataLayerError(Exception):
    """Base error for the data layer."""


class RateLimitExceeded(DataLayerError):
    """Raised when a request keeps getting rate-limited past the retry budget."""


def request_with_retry(
    session: requests.Session,
    method: str,
    url: str,
    *,
    max_retries: int = 5,
    backoff_base: float = 1.0,
    backoff_max: float = 60.0,
    timeout: float = 15.0,
    **kwargs: Any,
) -> requests.Response:
    """Issue an HTTP request, retrying on 429/5xx with exponential backoff.

    Honors a numeric Retry-After header on 429 responses instead of guessing.
    """
    attempt = 0
    while True:
        response = session.request(method, url, timeout=timeout, **kwargs)
        if response.status_code not in RETRYABLE_STATUS_CODES:
            return response

        attempt += 1
        if attempt > max_retries:
            if response.status_code == 429:
                raise RateLimitExceeded(
                    f"{method} {url} still rate-limited after {max_retries} retries"
                )
            response.raise_for_status()
            return response  # pragma: no cover - raise_for_status always raises here

        retry_after = response.headers.get("Retry-After")
        if retry_after is not None:
            try:
                delay = float(retry_after)
            except ValueError:
                delay = backoff_base * (2 ** (attempt - 1))
        else:
            delay = backoff_base * (2 ** (attempt - 1))
        delay = min(delay, backoff_max) + random.uniform(0, 0.25)

        logger.warning(
            "%s %s -> %s, retrying in %.1fs (attempt %d/%d)",
            method,
            url,
            response.status_code,
            delay,
            attempt,
            max_retries,
        )
        time.sleep(delay)
