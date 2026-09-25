# NVIDIA Claw Agent Challenge Paris: Market Analyst Desk

Working spec, compiled Sept 24, 2026. Free-tier limits and prices change, so re-check before relying on any figure.

## 1. Idea in one paragraph

A multi-agent "analyst desk" that acts as the advisor, so I don't have to invest myself. Specialist agents work in sequence and cross-check each other, so no single model's weakness decides the outcome. It is built from open-weight models: the team structure compensates for each model being weaker than a paid frontier model. It runs continuously on US stocks and logs every agent's reasoning.

**Draft one-liner for the submission (edit freely):** a team of independent open-weight agents that debates and cross-checks before recommending a US-stock portfolio, so the decision comes from structured checks and balances instead of one model's guess.

## 2. Challenge logistics

| Item | Detail |
|---|---|
| Format | Fully virtual, individual entry, no fixed schedule |
| Eligibility | Legal resident of France, 18+, individuals only (no corporate or institutional entries) |
| Deadline | Oct 2, 2026, 11:59 PM Pacific = about 8:59 to 9:59 AM Paris on Oct 3. Aim to submit by end of day Oct 1 or 2 Paris time |
| Entry | The Airtable form is the entry itself (a Luma sign-up alone isn't enough): https://airtable.com/appREoLM7BnGWxRzJ/pagLOxwVLzgiOaunm/form |
| Submission contents | Demo video or link to a site (60 to 90 seconds is ideal) plus a short description: what it does and why I built it |
| Winners | 2 winners, announced around Oct 6, chosen by NVIDIA judges (decisions final) |
| Prizes | 1st: GTC Berlin pass, DGX Spark Founders Edition, booth demo. 2nd: GTC Berlin pass, booth demo |
| Booth demo | Build-a-Claw Paris, Oct 13, 2026 (already accepted to the event) |

**Judging criteria:** (1) agent successfully deployed: runs end to end, non-trivial pipeline with multi-step reasoning, tool use, memory, long-running behavior; (2) innovation and creativity; (3) real-world value, including one sentence on who it helps.

**If I win:** confirm by email and return the signed release form within 7 days of the notification, or the prize is forfeited. The prize ships to the address given in the entry. Winner is responsible for taxes. Name appears on a public winners list.

**Terms to remember:**
- The rules say nothing on IP ownership or licensing of the entry. Anything shown in the entry may end up public.
- Keep the entry a clearly personal project (individual entries only).

## 3. Architecture

No model training. The agents use pretrained open-weight LLMs, so the work is deployment and engineering. The models don't know today's market, so all fresh information comes in through the data layer.

Pipeline:

1. **Macro/context agent.** Only fetches and passes context (macro data, filings, news, candidate stocks, price and volume data) to the analysts. Does no analysis itself.
2. **Two independent analyst agents.** Same task, different models, run separately, then compared.
   - If they contradict each other, the decision is aborted and the stock is not picked.
   - If they align, it moves on.
3. **Critic/feedback loop.** Agents challenge each other's reasoning over several rounds before anything is decided.
4. **Two bias/sentiment agents (redundancy).** Only run when the analysts align. They check for emotional or news-driven skew, trends, and stale news that may already be priced in.
5. **Technical expert agent.** Classic, book-based day-trading methods with multiple timeframes: start on the 4-hour chart for the global view, then drop to smaller charts for entry timing.
6. **Output.** The portfolio decision plus the full reasoning trail from every agent.

Every agent's input and output is logged with a timestamp, so the demo video can show the back-and-forth and not just a final buy or hold.

## 4. Models and rate limits

Use the NVIDIA Build endpoints (`https://integrate.api.nvidia.com/v1`, OpenAI-compatible). Implemented in `data_layer/llm_client.py`.

- **Catalog has drifted from this spec's original picks**, and drifted again mid-build: Qwen 3.5 (122B/397B) is no longer on Build at all. The first pass at `AGENT_MODELS` (picked from `docs.api.nvidia.com/nim/reference/...` pages) turned out to include two models that are genuinely dead — `z-ai/glm4.7` and `deepseek-ai/deepseek-v4-flash` both returned HTTP 410 Gone ("reached end of life") on a real call. Lesson: neither the marketing catalog page nor the per-model reference docs are reliable enough on their own — `GET /v1/models` on the live account plus an actual `chat_completion` call (some catalog-listed models 404 with "Not found for account" despite being listed) is the only way to know a model id is real *and* usable on a given account.
- **Current per-agent model assignment** (`AGENT_MODELS` in `llm_client.py`), one family per role for independence — every id below was live-verified with a real call against the account checked (Sept 2026):
  | Role | Model id | Notes |
  |---|---|---|
  | macro/context | `nvidia/nemotron-3.5-lightning-30b-a3b` | reasoning model, emits chain-of-thought |
  | analyst 1 | `google/gemma-4-31b-it` | non-reasoning; replaced `openai/gpt-oss-20b` on Sept 25 (failed 5/15 calls, 65–400s when it answered). Deterministic at temperature 0 |
  | analyst 2 | `nvidia/nemotron-3-super-120b-a12b` | reasoning model |
  | critic | `z-ai/glm-5.3` | reasoning model; original pick `z-ai/glm4.7` was 410 Gone. **Unreliable on Sept 25:** 1 of 2 analyst-sized calls answered (382s), the other exhausted retries — re-test before building the critic |
  | bias 1 | `google/gemma-4-31b-it` | **must be reassigned**: now also analyst 1, and a bias check by the same model would not be independent |
  | bias 2 | `mistralai/mistral-nemotron` | fast, non-reasoning, replies cleanly |
  | technical | `moonshotai/kimi-k3` | reasoning model; original pick `deepseek-ai/deepseek-v4-flash` was 410 Gone, and its live replacement `deepseek-ai/deepseek-v4.1-flash` failed with a consistent connection drop rather than answering. **Unreliable on Sept 25:** 0 of 2 analyst-sized calls answered (retries exhausted) — re-test before building the technical agent |
- **Several of these are reasoning models** (hidden chain-of-thought before visible content) — they need a generous `max_tokens` or they hit `finish_reason: "length"` with empty `content`, and `llm_client.py`'s `DEFAULT_TIMEOUT` is 120s (not the 15s used elsewhere in the data layer) because of how slow `gpt-oss-20b` in particular was.
- **Gateway 60s idle cutoff (found Sept 25, 2026):** Build closes any connection that sends no bytes for 60s, so non-streaming calls to reasoning models fail at exactly 60s. `llm_client.py` now streams by default. Streams also drop mid-response under load, and output from a degraded backend can turn into degenerate text, so agents retry and validate what they forward.
- **Thinking can be switched off on Nemotron models** via `chat_template_kwargs={"enable_thinking": false}`. The macro/context role uses it: with thinking on, the macro model spent its full 4,096-token budget reasoning (inside `content`) without producing a briefing; with it off, the same prompt returned a complete briefing in ~20–40s.
- **Free/trial tier limits:** confirmed ~40 requests/minute on the account checked (Sept 2026). Per-model limits for the others above aren't shown anywhere in the dashboard, so `llm_client.py` applies the same 40 RPM budget to every model by default (`DEFAULT_RPM_LIMIT`, overridable per model) as a conservative assumption, paced client-side on top of the existing 429 retry/backoff.
- **Credits, separate from RPM:** trial accounts get ~1,000 inference credits on signup (personal email), up to ~5,000 with a business email / "Request More". Credits deduct per call regardless of RPM headroom — a multi-day continuous run could exhaust credits before hitting any rate ceiling. Not yet measured against real usage.
- **Estimated load:** a few dozen LLM calls per stock cycle, well under 40 RPM unless loops get chatty.
- **Must have:** retry with backoff on HTTP 429 in every LLM call — done, `llm_client.py` calls go through the same `http_utils.request_with_retry` as every other data-layer client.
- **To check:** how credits are consumed over a multi-day run (Build dashboard) — no data yet since no live run has started.
- **Self-hosting:** not realistic on Oracle A1 (no GPU), so the endpoints are the route.

## 5. Data sources (US stocks only)

| Need | Source | Key facts |
|---|---|---|
| Price and volume, intraday, 4H | **Alpaca Market Data, free (Basic) plan** | US stocks and ETFs, IEX feed only, 200 historical calls/min, history since 2016, latest 15 minutes restricted on historical data, 30 websocket symbols. Build 4H bars by aggregating hourly bars (check the current SDK for native support) |
| Simulated execution | **Alpaca paper trading account** | Free API keys with a paper account. Lets agents place simulated orders for real P&L in the demo |
| Macro | **FRED** | Free API key, 800,000+ series, about 120 requests/min. Cache aggressively |
| Filings and fundamentals | **SEC EDGAR** | Free, no key, about 10 requests/second |
| Company news | **Finnhub, free plan** | Free company news with source timestamps. Sentiment is premium, so the bias agents score sentiment themselves. Free license is listed for personal use. Note: a GitHub issue reports the free plan rejecting US stock candle requests, so don't rely on Finnhub for prices |
| Optional news extra | **Alpha Vantage** | News with sentiment, but the free quota is 25 requests/day, so supplement only |
| Optional narrative layer | **Financial Times** | Already have access. Unverified whether it can be read programmatically |
| Avoid | Yahoo-based unofficial libraries | Fragile, risky for a demo |

**Paid Alpaca plan (not planned):** Algo Trader Plus, $99/month: all US exchanges, no 15-minute restriction, 10,000 calls/min, unlimited websockets. Older mentions of a $9 plan are outdated.

Design notes:
- The staleness check needs a timestamp on every news item, so favor sources that provide them.
- Cache everything that changes slowly.

## 6. Decision: stay on IEX-only volume, and say so

**Mention this in the submission description.** Suggested wording: volume figures come from the IEX exchange only (free Alpaca feed) and are not consolidated market volume.

What it means:
- IEX was about 4% of overall US equity volume and nearly 8% of exchange-traded volume in Q2 2026. Over half of US volume trades off-exchange (about 55% in Feb 2026).
- Absolute volume is too low: a stock with 50M shares traded may show about 2M.
- Relative volume still works: compare today's IEX volume with the same stock's recent IEX average.

Rules for the agents:
- Tell every agent explicitly that volume is IEX-only.
- Use relative volume versus a recent rolling baseline (weeks, not years). IEX's share has grown fast, so old baselines drift.
- Pick liquid stocks for the demo (thin slices of thin stocks are noisy).

## 7. Demo plan

- Portfolio of at least three US stocks (an idea, not a hard constraint).
- Runs about 2 days continuously on the Oracle A1 instance (via Termius).
- The video shows timestamped decisions, agent disagreement and agreement, and at least one aborted decision.
- Video length: 60 to 90 seconds.
- Add a note that this is a research demo (simulated or paper trading), not financial advice.

## 8. Suggested schedule (mine, not yet agreed)

Today is Sept 24. The live run needs to be underway with enough time left for 2 days of data, recording, and submitting.

| Days | Goal |
|---|---|
| Sept 25 to 26 | Data layer (Alpaca, FRED, EDGAR, Finnhub) plus macro agent plus first analyst, working end to end |
| Sept 27 | Second analyst and the cross-check/abort logic |
| Sept 28 | Bias checkers, technical expert, logging; deploy to Oracle A1 |
| Sept 29 to Oct 1 | Live run (about 2 days), fix problems as they appear |
| Oct 1 | Record demo video, write description |
| Oct 1 to 2 | Submit through the Airtable form (don't wait for the last hour) |

If time runs short, cut in this order: technical expert, then bias checkers. Keep the macro agent, two analysts, and cross-check.

## 9. Open questions

- Model-to-agent assignment is now decided (section 4) but not yet battle-tested against real traffic — revisit if any model turns out unavailable/slower than expected once agents are actually built.
- **Decided (Sept 25):** positions are long-only in real shares — no shorting, derivatives, leverage or negotiated deals.
- **Decided (Sept 25):** analyst cross-check uses strict matching — both recommendations must be identical (buy vs hold aborts). No confidence floor: tested and dropped, since the two models report confidence on different scales. A buy-vs-hold split goes to the critic loop for re-votes, and aborts if still unmatched; buy-vs-avoid or a failed verdict aborts at once.
- **Decided (Sept 25):** analyst horizon is the next 2–5 trading days, a forward projection judged only from data available now, so the ~2-day paper run can test the calls on camera.
- Should decisions be executed on the Alpaca paper account, or only logged?
- Final list of the three stocks?
- Is FT content usable programmatically, and is Barron's worth buying?
- Demo output: per-stock reasoning trail, multi-day log, or both?
- How Build credits get consumed over a multi-day run.