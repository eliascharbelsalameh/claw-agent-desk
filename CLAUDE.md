# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Data layer + (eventually) multi-agent pipeline for a NVIDIA Build a Claw Paris challenge entry: an "analyst desk" of independent open-weight LLM agents that debates and cross-checks before recommending a US-stock portfolio. `claw_agent_spec.md` is the full working spec — architecture, the verified per-role model table, data sources, the IEX-only volume caveat, and the build schedule. Read it before making design decisions; it's a living doc, and its section 9 still has open questions (deadline is Oct 2, 2026, so the schedule in section 8 is tight).

This repo lives locally (not on the Google Drive folder where the spec originated) because Drive's sync layer makes venvs, pip installs, and the on-disk cache this project uses painfully slow. The Drive folder (`NVIDIA Build a Claw - Paris/`) keeps a reference copy of the spec only — this repo is where the code lives.

## Commands

```
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt   # Windows
# .venv/bin/python -m pip install -r requirements.txt     # if ever run on Linux (e.g. Oracle A1)

.venv/Scripts/python -m pytest              # full suite (28 tests)
.venv/Scripts/python -m pytest -q tests/test_alpaca_client.py   # single file
.venv/Scripts/python -m pytest -k relative_volume                # by name
```

Tests never hit real APIs — every client is exercised with a fake `session.request` object, so the suite runs without any credentials present.

## Credentials

All six credentials are **already set as Windows user-level environment variables** (not in `.env`): `ALPACA_API_KEY_ID`, `ALPACA_API_SECRET_KEY`, `FRED_API_KEY`, `FINNHUB_API_KEY`, `SEC_EDGAR_USER_AGENT`, `NVIDIA_API_KEY`. `.env.example` documents them; `.env` exists but its secret fields are deliberately blank (only non-secret base URLs / cache dir remain). `load_dotenv()` doesn't override already-set OS env vars, so the env vars win — nothing needs changing in code.

**Don't read, `cat`, or write credential values through a Claude Code session.** Claude Code's file-watcher diffs any tool-touched file back into the transcript when it changes on disk, which already leaked (and forced rotation of) two keys during setup. Verify presence without exposing values, e.g.:

```
powershell -c "[Environment]::GetEnvironmentVariable('NVIDIA_API_KEY','User').Length"
```

Note that a shell started *before* a `setx` won't see the new value — an existing session's Bash tool may report a var as unset when it's actually set. Read `HKCU\Environment` (via PowerShell or Python's `winreg`) to check reliably, or have one-off live scripts load from there directly.

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
- `llm_client.LlmClient` — NVIDIA Build chat-completions client (`chat_completion`, `complete_text`). Same retry/backoff as every other client via `request_with_retry`, plus client-side per-model rate pacing (`DEFAULT_RPM_LIMIT = 40`, the only figure Build confirms, applied to every model as a conservative default). `AGENT_MODELS` maps each pipeline role (macro, analyst_1, analyst_2, critic, bias_1, bias_2, technical) to a model id from a different lab, per the spec's "independent failure modes" design choice (spec section 4). Two gotchas worth knowing before touching this file:
  - **Verifying a model id needs a real call.** `GET /v1/models` lists plenty of ids that then 404 with "Not found for account", and two of the original picks were 410 Gone (end of life). Neither build.nvidia.com's catalog page nor the per-model `docs.api.nvidia.com` reference pages were reliable. All seven current ids were confirmed with an actual `chat_completion` (Sept 2026) — see spec section 4 for the table and what each replaced.
  - **Most of these are reasoning models** (hidden chain-of-thought before visible content), so a small `max_tokens` yields `finish_reason: "length"` with empty `content`, and they're slow — `DEFAULT_TIMEOUT` is 120s here versus 15s elsewhere in the data layer, because `openai/gpt-oss-20b` alone needed ~90s.

`config.Settings` / `get_settings()` load everything from environment variables (OS env vars, falling back to `.env`) and provide a `.require(field)` that raises `ConfigError` with a clear message instead of the client failing deep inside a request. Every client accepts an optional `Settings`, `requests.Session`, and `DiskCache` in its constructor for testability — tests always inject a fake session and skip real network calls.

## Status

**Data layer is done and live-verified** (as of Sept 24, 2026, commit `07b744b`, pushed to `origin/master` at https://github.com/eliascharbelsalameh/claw-agent-desk). All 28 tests pass, and every client has been smoke-tested against its real API with real credentials:

- Alpaca — paper account `ACTIVE`, real AAPL daily bars. Note `get_bars` returns 0 rows without an explicit `start`/`end`; always pass a range.
- FRED — real `FEDFUNDS` observations.
- Finnhub — real company news, `age_hours` populated.
- SEC EDGAR — AAPL → CIK `0000320193`.
- NVIDIA Build — all seven `AGENT_MODELS` ids answered a real call.

**Not built yet:** no agent logic at all sits on top of `llm_client.py` — no macro/context agent, no analyst agents, no critic loop, no bias/technical agents, no end-to-end pipeline, no logging layer, no Oracle A1 deployment.

**Next step** per the spec's schedule (section 8): the macro/context agent, wired to this data layer plus `LlmClient` — it only gathers and passes context (macro, filings, news, price/volume) to the analysts and does no analysis itself (spec section 3, step 1). Then the first analyst agent.

Live-run costs are still unmeasured: Build trial accounts carry a credit balance (~1,000, up to ~5,000) that drains per call independently of the 40 RPM ceiling, and a full multi-agent cycle will take minutes given the reasoning-model latencies above. Spec section 9 has the remaining open questions (paper-execute vs. log-only, final stock list, demo output shape).
