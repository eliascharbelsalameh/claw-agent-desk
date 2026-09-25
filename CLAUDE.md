# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Data layer + (eventually) multi-agent pipeline for a NVIDIA Build a Claw Paris challenge entry: an "analyst desk" of independent open-weight LLM agents that debates and cross-checks before recommending a US-stock portfolio. `claw_agent_spec.md` is the full working spec — architecture, model/rate-limit choices, data sources, the IEX-only volume caveat, and the build schedule. Read it before making design decisions; it's a living doc with some open questions still unresolved (see its section 9).

This repo lives locally (not on the Google Drive folder where the spec originated) because Drive's sync layer makes venvs, pip installs, and the on-disk cache this project uses painfully slow. The Drive folder (`NVIDIA Build a Claw - Paris/`) keeps a reference copy of the spec only — this repo is where the code lives.

## Commands

```
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt   # Windows
# .venv/bin/python -m pip install -r requirements.txt     # if ever run on Linux (e.g. Oracle A1)

cp .env.example .env   # then fill in API keys — see .env.example for what's needed and why

.venv/Scripts/python -m pytest              # full suite
.venv/Scripts/python -m pytest -q tests/test_alpaca_client.py   # single file
.venv/Scripts/python -m pytest -k relative_volume                # by name
```

Tests never hit real APIs — every client is exercised with a fake `session.request` object, so the suite runs without any `.env` / API keys present.

## Architecture

`data_layer/` is a set of thin, independent clients for each data source in the spec (section 5), sharing plumbing so retry/caching behavior is identical everywhere rather than reimplemented per client:

- `http_utils.request_with_retry` — every outbound HTTP call goes through this. Retries on 429/500/502/503/504 with exponential backoff + jitter, honoring a numeric `Retry-After` header on 429s instead of guessing. This is the "must have: retry with backoff on 429" requirement from the spec, applied to all data calls, not just LLM calls.
- `cache.DiskCache` — dependency-free JSON-file cache with a per-entry TTL, keyed by URL+params. Callers pick the TTL per call site (e.g. EDGAR's ticker→CIK map caches for a week, Finnhub news for 15 minutes) — there's no single global policy because "cache everything that changes slowly" (spec section 5) means different things per source.
- `base_client.BaseClient._get_json` — wires the above two together (check cache → retry-aware GET → cache the result) so each client only implements URL-building and response shaping, not request mechanics.

Per-source clients, each a thin subclass of `BaseClient`:

- `alpaca_client.AlpacaClient` — bars (`get_bars`), a `get_bars_4h` that tries Alpaca's native `4Hour` timeframe and falls back to aggregating `1Hour` bars via `aggregate_bars` if the API rejects it (spec flagged native 4H support as unverified), a `get_relative_volume` helper that implements the spec's IEX-volume rule directly (section 6: never trust absolute IEX volume, compare against a rolling baseline instead), and paper-trading account/order methods (`get_account`, `list_positions`, `submit_market_order`).
- `fred_client.FredClient` — macro series observations.
- `edgar_client.EdgarClient` — ticker→CIK resolution (cached a week), filings (`get_company_submissions`), fundamentals (`get_company_facts`). Enforces SEC's ~10 req/s fair-access limit via an internal rate limiter, and refuses to run without `SEC_EDGAR_USER_AGENT` set — there's deliberately no hardcoded default contact, since the challenge entry may end up public.
- `finnhub_client.FinnhubClient` — company news only (the free plan doesn't reliably serve candles, so don't add a prices method here — use Alpaca). Tags every item with `age_hours` on the way out since the staleness check downstream needs that.
- `llm_client.LlmClient` — NVIDIA Build chat-completions client (`chat_completion`, `complete_text`). Same retry/backoff as every other client via `request_with_retry`, plus client-side per-model rate pacing (`DEFAULT_RPM_LIMIT = 40`, since Build's dashboard only confirms that figure for a subset of models). `AGENT_MODELS` maps each pipeline role (macro, analyst_1, analyst_2, critic, bias_1, bias_2, technical) to a specific model id from a different lab, per the spec's "independent failure modes" design choice (spec section 4) — re-verify those ids against your own Build account before a live run, since the catalog changes often and doesn't always match its own marketing copy.

`config.Settings` / `get_settings()` load everything from environment variables (via `.env`) and provide a `.require(field)` that raises `ConfigError` with a clear message instead of the client failing deep inside a request. Every client accepts an optional `Settings`, `requests.Session`, and `DiskCache` in its constructor for testability — tests always inject a fake session and skip real network calls.

## Not built yet

The LLM client (`llm_client.py`) exists but no agent logic sits on top of it yet — no macro/context agent, no analyst agents, no NVIDIA Build integration end-to-end, no Oracle A1 deployment. `NVIDIA_API_KEY` still needs to be generated from a real build.nvidia.com account and dropped into `.env` before any real call can be made. Next per the spec's schedule (section 8): macro/context agent wired to this data layer plus the LLM client, then the first analyst agent.
