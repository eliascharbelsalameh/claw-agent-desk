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

.venv/Scripts/python -m pytest              # full suite (101 tests)
.venv/Scripts/python -m pytest -q tests/test_alpaca_client.py   # single file
.venv/Scripts/python -m pytest -k relative_volume                # by name

.venv/Scripts/python -m agents AAPL MSFT NVDA            # live macro/context agent run
.venv/Scripts/python -m agents AAPL --no-llm             # data only, no Build credits
.venv/Scripts/python -m agents AAPL --analysts analyst_1 analyst_2   # context + both analysts + cross-check
```

A live run needs the credentials in the *process* environment — see Credentials below for why a shell may not have them.

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

- `http_utils.request_with_retry` — every outbound HTTP call goes through this. Retries on 429/500/502/503/504 and on dropped connections (`requests.ConnectionError`; Build closes connections under load) with exponential backoff + jitter, honoring a numeric `Retry-After` header on 429s instead of guessing. Read timeouts are deliberately not retried. This is the "must have: retry with backoff on 429" requirement from the spec, applied to all data calls, not just LLM calls.
- `cache.DiskCache` — dependency-free JSON-file cache with a per-entry TTL, keyed by URL+params. Callers pick the TTL per call site (e.g. EDGAR's ticker→CIK map caches for a week, Finnhub news for 15 minutes) — there's no single global policy because "cache everything that changes slowly" (spec section 5) means different things per source.
- `base_client.BaseClient._get_json` — wires the above two together (check cache → retry-aware GET → cache the result) so each client only implements URL-building and response shaping, not request mechanics.

Per-source clients, each a thin subclass of `BaseClient`:

- `alpaca_client.AlpacaClient` — bars (`get_bars`), a `get_bars_4h` that tries Alpaca's native `4Hour` timeframe and falls back to aggregating `1Hour` bars via `aggregate_bars` if the API rejects it (spec flagged native 4H support as unverified), a `get_relative_volume` helper that implements the spec's IEX-volume rule directly (section 6: never trust absolute IEX volume, compare against a rolling baseline instead), and paper-trading account/order methods (`get_account`, `list_positions`, `submit_market_order`).
- `fred_client.FredClient` — macro series observations.
- `edgar_client.EdgarClient` — ticker→CIK resolution (cached a week), filings (`get_company_submissions`), fundamentals (`get_company_facts`). Enforces SEC's ~10 req/s fair-access limit via an internal rate limiter, and refuses to run without `SEC_EDGAR_USER_AGENT` set — there's deliberately no hardcoded default contact, since the challenge entry may end up public.
- `finnhub_client.FinnhubClient` — company news only (the free plan doesn't reliably serve candles, so don't add a prices method here — use Alpaca). Tags every item with `age_hours` on the way out since the staleness check downstream needs that.
- `llm_client.LlmClient` — NVIDIA Build chat-completions client (`chat_completion`, `complete_text`). Same retry/backoff as every other client via `request_with_retry`, plus client-side per-model rate pacing (`DEFAULT_RPM_LIMIT = 40`, the only figure Build confirms, applied to every model as a conservative default). `AGENT_MODELS` maps each pipeline role (macro, analyst_1, analyst_2, critic, bias_1, bias_2, technical) to a model id from a different lab, per the spec's "independent failure modes" design choice (spec section 4). Two gotchas worth knowing before touching this file:
  - **Verifying a model id needs a real call.** `GET /v1/models` lists plenty of ids that then 404 with "Not found for account", and two of the original picks were 410 Gone (end of life). Neither build.nvidia.com's catalog page nor the per-model `docs.api.nvidia.com` reference pages were reliable. All seven current ids were confirmed with an actual `chat_completion` (Sept 2026) — see spec section 4 for the table and what each replaced.
  - **Most of these are reasoning models** (hidden chain-of-thought before visible content), so a small `max_tokens` yields `finish_reason: "length"` with empty `content`, and they're slow — `DEFAULT_TIMEOUT` is 120s here versus 15s elsewhere in the data layer, because `openai/gpt-oss-20b` alone needed ~90s. Nemotron models take `chat_template_kwargs={"enable_thinking": False}` to skip reasoning entirely (one-line prompt: 59s → 2s); use it for roles that only summarize.
  - **Calls stream by default, and must.** Build's gateway kills any connection that sends no bytes for 60s, so a non-streaming call to a reasoning model dies at 60s regardless of `DEFAULT_TIMEOUT` (reproduced repeatedly, Sept 2026). `chat_completion` streams and reassembles the chunks into the normal non-streaming response shape (reasoning goes to `message.reasoning_content`), and retries dropped/truncated streams `STREAM_RETRIES` times. Pass `stream=False` only for calls known to finish fast.
  - **Build output quality is not guaranteed.** Live, 2 of 3 briefings from the macro model degenerated mid-text into "The The The…" / ",,,,,,," — both on calls whose first stream had dropped. Any agent that forwards LLM text should validate its shape before passing it on (see `agents.macro_agent.briefing_problems`).

`agents/` holds the pipeline agents (spec section 3), built on the data layer:

- `trace.TraceLogger` — append-only JSONL log of every agent's input and output with a UTC timestamp (spec section 3: the demo needs the back-and-forth, not just the final call). One line per event; every agent should log through this.
- `macro_agent.MacroContextAgent` — step 1 of the pipeline. `run(symbols)` gathers FRED macro (once, shared), Alpaca daily/4H bars + relative volume, EDGAR filings + fundamentals, and Finnhub news per symbol into a `StockContext`, then asks `AGENT_MODELS["macro"]` (thinking off) for a neutral factual briefing that is explicitly forbidden from opinions or recommendations. `StockContext.to_prompt()` is what downstream analysts consume: the briefing plus the full source facts, labeled as authoritative over the briefing. Design points:
  - Every source failure becomes a line in `StockContext.data_gaps` instead of an exception, so a 2-day run degrades rather than crashes; analysts are told what's missing.
  - The briefing is validated (`briefing_problems`: degenerate repetition, missing sections), retried once with a fresh call, then dropped to a facts-only packet with a gap recorded. Rejections are logged as `briefing_rejected`.
  - Finnhub's company-news feed is keyword-matched and mostly noise (live: 115–202 of ~130–220 weekly items for AAPL/MSFT/NVDA didn't mention the company), so `filter_news` keeps only items mentioning the ticker or the EDGAR-registered short name. Done deterministically because the no-thinking model miscounted when asked to filter.
  - Fundamentals take the most recent period across all candidate XBRL tags (companies switch revenue tags in both directions — NVDA's current revenue is under `Revenues`, AAPL/MSFT's under the ASC 606 tag), and anything older than `FUNDAMENTALS_STALE_DAYS` is flagged as a gap. A 10-K period-end yields the fiscal-year figure (no separate Q4 is tagged), so `period_start`/`period_end` always travel with each value.
  - Each fundamental also carries the same-period year-ago value and `yoy_pct_change` (quarter vs quarter, fiscal year vs fiscal year, searched across all candidate tags). Without it, analysts asserted "earnings growth" from a single quarter.
  - `facts()` always includes `limitations` (`DESK_LIMITATIONS`): what the desk never provides — no valuation data, no forward guidance, only latest-vs-year-ago fundamentals, unverified news, IEX-only volume. Identical for every stock and run (a disclosure, not a judgment), and deliberately separate from `data_gaps`, which is only what failed to load this run.
  - Rates and spreads get an absolute year-over-year change; only CPI gets a percentage change (`PERCENT_CHANGE_SERIES`).
- `analyst_agent.AnalystAgent` — step 2. One instance per role (`analyst_1`, `analyst_2`); `analyze(ctx)` sends `StockContext.to_prompt()` to that role's model and returns an `AnalystVerdict`: `recommendation` (buy/hold/avoid), `confidence`, `thesis`, `drivers`, `risks`, `evidence`, `data_concerns`. Design points:
  - Structured JSON so the cross-check compares `recommendation` fields, not prose. Invalid replies get one correction turn (the model sees its reply and the validation error); after that the verdict has `error` set. **A failed verdict must be treated as "no pick", never as agreement.**
  - **JSON quote repair.** `nemotron-3-super` intermittently drops the quotes on string values — every `"why"` in a reply at once (`"why": Indicates growth.`), or a list item's closing quote — in 5 of 38 replies on Sept 25, and the correction turn reproduced the identical error every time. When a reply doesn't parse, `repair_json_quotes` adds the missing quotes (whole lines only, no word changed), and the result must still parse and validate. Repairs are logged as `verdict_repaired` and kept in `AnalystVerdict.json_repairs`. All 5 real failures recovered, every string verbatim from the original. Build's structured-output options were tested: `nvext.guided_json` is rejected (400 unknown field); `response_format: {"type": "json_object"}` is accepted but not adopted — too few samples to show it prevents the bug, and it roughly halved nemotron's reasoning length.
  - Every evidence item cites a path into the facts (`price.change_20d_pct`, `news[2].headline`), and `check_evidence` resolves it deterministically: `verified`, `wrong_index` (real value at another list position — seen live), `mismatch` (with the `actual` value), or `unknown_path`. This catches quoted-number hallucinations; it cannot catch uncited prose claims, which is why the context has to carry the numbers (year-ago values) that make claims checkable.
  - **Decided:** long-only, real shares — no shorting, derivatives, leverage or negotiated deals; "avoid" means not holding, never shorting. **Decided:** `HORIZON` is the next 2–5 trading days — a forward projection judged only from data available now, chosen so the ~2-day paper run can test the calls. The prompt states only the window, not what to weigh at that range.
  - **Models (Sept 25, 2026):** `analyst_1` is `google/gemma-4-31b-it`, replacing `openai/gpt-oss-20b` (failed 5 of 15 calls, 65–400s per answer, left `data_concerns` empty). Candidates screened side by side on frozen contexts: only gemma answered reliably; `glm-5.3`/`glm-5.3-flash`/`kimi-k3` mostly exhausted connection retries; `kimi-k2.6`, `mistral-large-2-instruct`, `palmyra-fin-70b-32k` return 404 on this account. Nemotron models were excluded to keep analyst_1 independent of analyst_2.
  - **Temperature 0** (`ANALYST_TEMPERATURE`). Over 3 repeats on identical input, gemma returned identical verdicts and confidences; `nemotron-3-super` still split 2–1 on MSFT and NVDA, so disagreements remain for the critic loop to resolve.
  - To watch: gemma said buy on all three test stocks at 0.75–0.85 and lists fewer `data_concerns` (2–3 vs 4–5 for nemotron). Three momentum large caps is too few to call it a bias, but an always-buy analyst would reduce the cross-check to analyst_2 alone — re-check on a wider stock set.
- `cross_check.cross_check(a, b)` — the gate after the analysts. A deterministic rule, not an LLM call, returning a `CrossCheckResult` (`outcome`, agreed `recommendation`, human-readable `reason`, both verdict summaries) and logging it as a `cross_check`/`decision` trace event. Outcomes: identical recommendations → `agree`; buy vs hold → `critic`; buy vs avoid, hold vs avoid, any failed verdict, or both analysts on the same model → `abort`. Confidence is carried for the trace but never affects the outcome. Replayed on the 9 real temperature-0 verdict pairs: 5 agree (buy), 3 critic, 1 abort (failed call).

`config.Settings` / `get_settings()` load everything from environment variables (OS env vars, falling back to `.env`) and provide a `.require(field)` that raises `ConfigError` with a clear message instead of the client failing deep inside a request. Every client accepts an optional `Settings`, `requests.Session`, and `DiskCache` in its constructor for testability — tests always inject a fake session and skip real network calls.

## Status

**Data layer is done and live-verified** (Sept 24, 2026, commit `07b744b`, pushed to `origin/master` at https://github.com/eliascharbelsalameh/claw-agent-desk). Every client has been smoke-tested against its real API with real credentials:

- Alpaca — paper account `ACTIVE`, real daily bars. Note `get_bars` returns 0 rows without an explicit `start`/`end`; always pass a range.
- FRED, Finnhub, SEC EDGAR — real data.
- NVIDIA Build — all seven `AGENT_MODELS` ids answered a real call.

**Macro/context agent is done and live-verified** (Sept 25, 2026): a full AAPL/MSFT/NVDA run completed in ~166s with zero data gaps and all three briefings passing validation on the first attempt (~35–90s per briefing, ~6.4k prompt + ~1.5k completion tokens each). An earlier run during Build instability took 454s with two stream drops, both recovered by retry.

**Known limitation:** `AlpacaClient.get_relative_volume` treats the most recent daily bar as "today", so a run during market hours compares a partial day against full-day averages and understates relative volume. Runs so far were pre-market, where it's correct.

**Analyst agent is done and live-verified** (Sept 25, 2026): both analyst roles return valid, evidence-checked verdicts on real context. Per-call time is ~20–50s when Build is healthy, but under load Build drops connections before sending headers and a single call took 335s through four retries — budget minutes, not seconds, per cycle.

**Consistency test (Sept 25, 2026, 2–5 day horizon, temperature 0.2; "analyst_1" here is the since-replaced `gpt-oss-20b`):** each analyst ran 5× on identical frozen contexts for AAPL/MSFT/NVDA (33 calls, ~273k tokens, 71 min wall time with the two analysts in parallel). Findings that should shape the next steps:
- **Reliability is the biggest problem.** 7 of 30 analyst runs failed. `gpt-oss-20b` failed 5 of 15, every one after ~400s of exhausted connection retries, and its successful calls took 65–400s. `nemotron-3-super` failed 2 of 15: once malformed JSON (a missing quote; the correction turn reproduced the identical error at the identical character, so the correction turn is ineffective for this failure), and once a stream Build closed with no content and no finish_reason (now retried as a drop in `llm_client._read_stream`).
- **Same input, different recommendations.** 4 of 6 stock×analyst cells were mixed across repeats (e.g. MSFT: analyst_1 hold×3/buy×1, analyst_2 buy×3/hold×1). Only AAPL×analyst_2 (5/5 buy) and NVDA×analyst_1 (3/3 buy) were stable. A single run's verdict on a borderline stock is partly sampling noise.
- **Confidence scales differ by model.** analyst_1 means 0.61–0.69, analyst_2 0.66–0.74; within-cell stdev 0.03–0.07. A shared absolute 0.66 floor sits in the middle of gpt-oss's range, so it mostly measures which model is more cautious, not the evidence.
- **Grounding held up:** 115/115 evidence citations verified, zero wrong-index, and every successful run filled `data_concerns`.

**Model reliability check needed before the next agents:** in the Sept 25 screen, `z-ai/glm-5.3` (critic) and `moonshotai/kimi-k3` (technical) mostly failed on analyst-sized prompts, and `bias_1` shares gemma with `analyst_1`. Re-verify or reassign those roles before building them.

**Analyst cross-check is done** (Sept 25, 2026, `agents/cross_check.py`), wired into the CLI when both analysts run.

**Not built yet:** critic loop (it must handle the `critic` outcome: re-votes, abort if still unmatched), bias/technical agents, end-to-end pipeline orchestration, Oracle A1 deployment.

**Next step** per the spec's schedule (section 8): the critic loop (spec section 3, step 3) — but first re-test or reassign its model (`z-ai/glm-5.3` mostly failed in the Sept 25 screen). Hold vs avoid currently aborts under strict matching; that case wasn't explicitly decided and only matters for stocks already held. **Decided (Sept 25, 2026): strict matching** — the two analysts' `recommendation` values must be identical to continue; any difference (including buy vs hold) aborts, and a failed verdict always aborts. **Confidence floor dropped (Sept 25, 2026):** the consistency test showed the two models report confidence on different scales, so no confidence score gates the decision — only the recommendation match does. `confidence` stays in the verdict and trace as information only. **Also decided:** a buy-vs-hold split goes to the critic loop for re-votes instead of aborting immediately, and aborts if still unmatched afterwards; buy-vs-avoid and failed verdicts abort at once.

Live-run costs are still unmeasured: Build trial accounts carry a credit balance (~1,000, up to ~5,000) that drains per call independently of the 40 RPM ceiling. Spec section 9 has the remaining open questions (paper-execute vs. log-only, final stock list, demo output shape).
