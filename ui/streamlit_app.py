"""Claw Agent Desk - run and inspect the desk by hand before automating it.

    .venv/Scripts/streamlit run ui/streamlit_app.py

Runs the same DeskPipeline as the CLI (agents/pipeline.py) and shows every
stage: the context each stock got, both analysts' verdicts with their
checked evidence, the cross-check, and each critic-loop round. The trace
viewer replays any past run from its JSONL trace without spending credits.
"""
from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# `streamlit run ui/streamlit_app.py` puts ui/ on sys.path, not the repo root.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import streamlit as st  # noqa: E402

from ui.support import (  # noqa: E402
    OUTCOME_COLORS,
    REC_COLORS,
    credential_status,
    event_row,
    load_user_env,
    parse_symbols,
    read_trace,
    summarize_trace,
)

# Credentials must be in the process before the settings are first read.
load_user_env()

from agents.critic_loop import CHALLENGE_AGREED_BUYS, MAX_ROUNDS  # noqa: E402
from agents.pipeline import DeskPipeline, ModelOutages, SymbolRun, default_trace_path  # noqa: E402
from agents.state import DeskState  # noqa: E402
from agents.trace import TraceLogger  # noqa: E402
from data_layer.llm_client import AGENT_MODELS, DEFAULT_MODEL_HEALTH, role_models  # noqa: E402

LOG_DIR = Path(os.environ.get("CLAW_DESK_LOG_DIR", REPO_ROOT / "logs"))
STATE_PATH = Path(os.environ.get("CLAW_DESK_STATE", REPO_ROOT / "state" / "desk_state.json"))
FULL, DATA_ONLY = "Full desk", "Data only (no Build credits)"

st.set_page_config(page_title="Claw Agent Desk", layout="wide")


# --- small rendering helpers ---

def rec_text(rec: str | None) -> str:
    return f":{REC_COLORS.get(rec, 'gray')}[**{(rec or 'failed').upper()}**]"


def outcome_text(outcome: str | None, rec: str | None = None) -> str:
    if outcome is None:
        return ":gray[no decision]"
    label = outcome.upper() + (f" ({rec})" if rec else "")
    return f":{OUTCOME_COLORS.get(outcome, 'gray')}[**{label}**]"


def model_note(v) -> str:
    if v.fallbacks:
        return f"`{v.model}` (backup - {', '.join(f['model'] for f in v.fallbacks)} failed first)"
    if getattr(v, "used_backup", False):
        return f"`{v.model}` (backup - primary `{v.primary_model}` failed recently or was excluded)"
    return f"`{v.model}`"


def render_context(ctx) -> None:
    price, rv = ctx.price or {}, ctx.relative_volume or {}
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Last price", price.get("last_close", "-"),
              f"{price.get('change_1d_pct')}% 1d" if price.get("change_1d_pct") is not None else None)
    c2.metric("20-day change", f"{price.get('change_20d_pct', '-')}%")
    rel = rv.get("relative_volume")
    c3.metric("Relative volume (IEX)", f"{rel:.2f}x" if isinstance(rel, (int, float)) else "-")
    c4.metric("News items (company)", len(ctx.news), f"{ctx.news_unrelated_dropped} unrelated dropped",
              delta_color="off")
    tech, val = ctx.technicals or {}, ctx.valuation or {}
    nxt = (ctx.earnings or {}).get("next_report") or {}
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("RSI (14)", tech.get("rsi_14", "-"))
    vs_spy = tech.get("change_20d_vs_spy_pp")
    c2.metric("20-day vs SPY", f"{vs_spy:+.1f} pp" if isinstance(vs_spy, (int, float)) else "-")
    pe = val.get("pe_ttm")
    c3.metric("P/E (TTM)", pe if pe is not None else "n/m" if val else "-")
    c4.metric("Next earnings", nxt.get("date", "-"),
              f"in {nxt['calendar_days_until']} days" if nxt else None, delta_color="off")
    if price.get("latest_bar_in_progress"):
        st.info(price.get("note", "Session in progress."))
    if rv.get("session_in_progress"):
        st.caption(f"Relative volume uses the last completed session; today so far: "
                   f"{rv.get('in_progress_volume')} shares on IEX.")
    if ctx.data_gaps:
        st.warning("Data gaps: " + "; ".join(ctx.data_gaps))
    if ctx.briefing:
        with st.expander(f"Briefing ({ctx.briefing_model})"):
            st.markdown(ctx.briefing)
    elif ctx.briefing_model is None:
        st.caption("No briefing (data-only run, or every briefing model failed - see data gaps).")
    with st.expander("Valuation, technicals, earnings (computed by the desk)"):
        for title, block in (("Valuation", ctx.valuation), ("Technicals", ctx.technicals),
                             ("Earnings", ctx.earnings)):
            if block:
                st.markdown(f"**{title}**")
                st.json({k: v for k, v in block.items() if k != "note"}, expanded=False)
                st.caption(block.get("note", ""))
    with st.expander("Fundamentals, macro, news, filings"):
        if ctx.fundamentals:
            st.dataframe([{"item": k, "value": v.get("value"), "period_end": v.get("period_end"),
                           "yoy %": v.get("yoy_pct_change"), "form": v.get("form")}
                          for k, v in ctx.fundamentals.items()], hide_index=True)
        if ctx.macro:
            st.dataframe([{"series": k, "label": v.get("label"), "latest": v.get("latest"), "date": v.get("date"),
                           "yoy": v.get("yoy_abs_change", v.get("yoy_pct_change"))}
                          for k, v in ctx.macro.items()], hide_index=True)
        if ctx.news:
            st.dataframe([{"#": n.get("index"), "age h": n.get("age_hours"), "headline": n.get("headline"),
                           "source": n.get("source")} for n in ctx.news], hide_index=True)
        if ctx.filings:
            st.dataframe([{"form": f.get("form"), "filed": f.get("filing_date"), "items": f.get("items"),
                           "url": f.get("url")} for f in ctx.filings], hide_index=True)
        st.caption("Limitations: " + " ".join(ctx.facts()["limitations"]))


def render_verdict(v) -> None:
    st.markdown(f"**{v.role}** · {model_note(v)}")
    if not v.ok:
        if v.call_failed:
            st.warning(f"Model unreachable (Build failure) - the stock is deferred and retried: {v.error}")
        else:
            st.error(f"Failed: {v.error}")
        return
    st.markdown(f"{rec_text(v.recommendation)} · confidence {v.confidence}")
    st.write(v.thesis)
    st.markdown("**Drivers**\n" + "\n".join(f"- {d}" for d in v.drivers))
    st.markdown("**Risks**\n" + "\n".join(f"- {r}" for r in v.risks))
    if v.data_concerns:
        st.markdown("**Data concerns**\n" + "\n".join(f"- {c}" for c in v.data_concerns))
    check = v.evidence_check
    st.caption(f"Evidence: {check.get('matches_source', 0)} match source, {check.get('wrong_index', 0)} wrong "
               f"index, {check.get('mismatch', 0)} mismatch, {check.get('unknown_path', 0)} unknown path")
    st.dataframe([{"fact": e.get("fact"), "value": str(e.get("value"))[:80], "status": e.get("status"),
                   "why": e.get("why")} for e in v.evidence], hide_index=True)
    if v.json_repairs:
        st.caption(f"JSON repaired: {'; '.join(v.json_repairs)}")


def render_critic_loop(loop) -> None:
    st.markdown(f"**Critic loop** ({loop.trigger}): {outcome_text(loop.outcome, loop.recommendation)} "
                f"- {loop.reason}")
    if loop.resume:
        kept = [name for name, have in (("the critique", loop.resume.get("critique")),
                                        (", ".join(loop.resume.get("revised") or {}) + " re-vote",
                                         loop.resume.get("revised"))) if have]
        st.caption(f"Resumes at round {loop.resume['round']}"
                   + (f", keeping {' and '.join(kept)}" if kept else "") + ".")
    for r in loop.rounds:
        critique = r["critique"]
        with st.expander(f"Round {r['round']}", expanded=True):
            if critique.get("error"):
                st.error(f"Critic failed: {critique['error']}")
                continue
            primary = critique.get("primary_model")
            backup = " (backup)" if primary and critique.get("model") != primary else ""
            st.markdown(f"**Critic** `{critique.get('model')}`{backup}: {critique.get('assessment')}")
            if critique.get("challenges"):
                st.dataframe([{"to": c["to"], "point": c["point"], "why": c["why"],
                               "fact": c.get("fact", ""), "status": c.get("status", "")}
                              for c in critique["challenges"]], hide_index=True)
            for role, v in r.get("verdicts", {}).items():
                st.markdown(f"**{role}** `{v['model']}`: {rec_text(v['previous_recommendation'])} → "
                            f"{rec_text(v['recommendation'])} · accepted {v['challenges_accepted']}, "
                            f"rejected {v['challenges_rejected']}")
                for resp in v.get("response_to_critique") or []:
                    verdict = {True: "accept", False: "reject"}.get(resp.get("accept"), "?")
                    st.caption(f"{verdict}: {resp.get('point') or ''} - {resp.get('reason') or ''}")
            if r.get("cross_check"):
                cc = r["cross_check"]
                st.markdown(f"Cross-check after round: {outcome_text(cc['outcome'], cc.get('recommendation'))}")


BIAS_COLORS = {"passed": "green", "vetoed": "red", "failed": "red", "deferred": "violet"}


def render_bias(gate) -> None:
    color = BIAS_COLORS.get(gate.outcome, "gray")
    st.markdown(f"**Bias gate:** :{color}[**{gate.outcome.upper()}**] - {gate.reason}")
    cols = st.columns(len(gate.checks) or 1)
    for col, (role, check) in zip(cols, gate.checks.items()):
        with col:
            st.markdown(f"**{role}** `{check.get('model')}`")
            if check.get("error"):
                st.warning(f"Failed: {check['error']}")
                continue
            st.markdown(f"{'flag' if check.get('verdict') == 'flag' else 'pass'} · news sentiment "
                        f"{check.get('news_sentiment')}")
            st.write(check.get("reason"))
            st.dataframe([{"check": name, "skewed": c.get("skewed"), "why": c.get("why")}
                          for name, c in (check.get("checks") or {}).items()], hide_index=True)


def render_technical(tech) -> None:
    color = {"passed": "green", "waiting": "orange", "failed": "red", "deferred": "violet"}.get(tech.outcome, "gray")
    st.markdown(f"**Technical** `{tech.model}`: :{color}[**{(tech.timing or tech.outcome).upper()}**] - "
                f"{tech.reason or tech.error}")
    if tech.timing:
        st.caption(f"4h trend {tech.trend_4h} · support {tech.support} · resistance {tech.resistance}")


def render_run(run: SymbolRun) -> None:
    st.markdown(f"### {run.symbol} - {outcome_text(run.outcome, run.recommendation)}")
    render_context(run.context)
    if run.verdicts:
        cols = st.columns(len(run.verdicts))
        for col, v in zip(cols, run.verdicts.values()):
            with col:
                render_verdict(v)
    if run.cross_check:
        cc = run.cross_check
        st.markdown(f"**Cross-check:** {outcome_text(cc.outcome, cc.recommendation)} - {cc.reason}")
    if run.critic_loop:
        render_critic_loop(run.critic_loop)
    if run.bias:
        render_bias(run.bias)
    if run.technical:
        render_technical(run.technical)


def summary_rows(runs: list[SymbolRun]) -> list[dict]:
    rows = []
    for run in runs:
        row = {"symbol": run.symbol}
        for role, v in run.verdicts.items():
            row[role] = (v.recommendation or "FAILED") + (" (backup)" if v.used_backup else "")
        row["cross-check"] = run.cross_check.outcome if run.cross_check else ""
        row["critic loop"] = (f"{run.critic_loop.outcome} after {len(run.critic_loop.rounds)} round(s)"
                              if run.critic_loop else "")
        row["final"] = (f"{run.outcome} {run.recommendation or ''}".strip() if run.outcome else "data only")
        row["bias gate"] = run.bias.outcome if run.bias else ""
        row["entry"] = (run.technical.timing or run.technical.outcome) if run.technical else ""
        row["data gaps"] = len(run.context.data_gaps)
        rows.append(row)
    return rows


# --- sidebar ---

with st.sidebar:
    st.header("Run settings")
    symbols_text = st.text_input("Symbols", "AAPL, MSFT, NVDA", help="Comma or space separated US tickers")
    mode = st.radio("Mode", [FULL, DATA_ONLY], help="Data only gathers context without any LLM call")
    data_only = mode == DATA_ONLY
    run_critic = st.checkbox("Run the critic loop", value=True, disabled=data_only)
    challenge_buys = st.checkbox("Challenge agreed buys", value=CHALLENGE_AGREED_BUYS, disabled=data_only,
                                 help="Also send an agreed buy through one critic round")
    max_rounds = st.number_input("Max critic rounds", min_value=1, max_value=4, value=MAX_ROUNDS,
                                 disabled=data_only)
    start = st.button("Run the desk", type="primary", width="stretch")
    st.caption("A full run takes minutes per stock. Don't change settings while it runs - "
               "Streamlit restarts the script on any interaction (finished stocks are kept).")

    st.divider()
    st.subheader("Credentials")
    for name, present in credential_status().items():
        st.markdown(f"{':green[set]' if present else ':red[missing]'} `{name}`")

    st.subheader("Models")
    cooling = [m for role in AGENT_MODELS for m in role_models(role) if DEFAULT_MODEL_HEALTH.is_cooling(m)]
    with st.expander("Line-up and backups"):
        st.dataframe([{"role": role, "primary": role_models(role)[0], "backups": ", ".join(role_models(role)[1:])}
                      for role in AGENT_MODELS], hide_index=True)
    if cooling:
        st.warning("Cooling (failed in the last 15 min, tried last): " + ", ".join(sorted(set(cooling))))


# --- main area ---

st.title("Claw Agent Desk")
st.caption("Research demo on public and paper-trading data - not financial advice. "
           "Volume figures are IEX-only (~4% of US volume); only relative volume is meaningful.")

run_tab, state_tab, trace_tab = st.tabs(["Run", "Desk state", "Trace viewer"])

with run_tab:
    if start:
        symbols, rejected = parse_symbols(symbols_text)
        if rejected:
            st.warning(f"Ignored: {', '.join(rejected)}")
        if not symbols:
            st.error("Enter at least one ticker.")
        else:
            missing = [n for n, ok in credential_status().items()
                       if not ok and (n != "NVIDIA_API_KEY" or not data_only)]
            if missing:
                st.error("Missing credentials: " + ", ".join(missing))
            else:
                trace = TraceLogger(default_trace_path(LOG_DIR))
                started = datetime.now(timezone.utc).isoformat()
                st.session_state["runs"] = []
                st.session_state["meta"] = {"trace": str(trace.path), "started": started, "symbols": symbols,
                                            "mode": mode, "run_critic": run_critic,
                                            "challenge_buys": challenge_buys, "max_rounds": int(max_rounds)}
                pipeline = DeskPipeline.from_settings(
                    trace=trace, use_llm=not data_only, run_critic=run_critic,
                    challenge_agreed_buys=challenge_buys, max_rounds=int(max_rounds),
                    outages=st.session_state.setdefault("outages", ModelOutages()),
                )
                t0 = time.time()
                with st.status(f"Running the desk on {', '.join(symbols)}...", expanded=True) as status:

                    def on_event(stage, symbol, payload):
                        if stage == "context":
                            status.update(label=f"{symbol}: context gathered")
                            st.write(f"**{symbol}** context: {len(payload.data_gaps)} data gaps, "
                                     f"briefing {'yes' if payload.briefing else 'no'}")
                        elif stage == "verdict":
                            st.write(f"**{symbol}** {payload.role} ({payload.model}): "
                                     f"{rec_text(payload.recommendation)}")
                        elif stage == "cross_check":
                            st.write(f"**{symbol}** cross-check: {outcome_text(payload.outcome, payload.recommendation)}")
                        elif stage == "critic_loop":
                            st.write(f"**{symbol}** critic loop: {outcome_text(payload.outcome, payload.recommendation)}")
                        elif stage == "bias":
                            st.write(f"**{symbol}** bias gate: {payload.outcome}")
                        elif stage == "technical":
                            st.write(f"**{symbol}** entry timing: {payload.timing or payload.outcome}")
                        elif stage == "done":
                            st.session_state["runs"].append(payload)

                    pipeline.run(symbols, on_event)
                    status.update(label=f"Done in {time.time() - t0:.0f}s", state="complete", expanded=False)
                st.session_state["meta"]["seconds"] = round(time.time() - t0)

    runs = st.session_state.get("runs") or []
    meta = st.session_state.get("meta") or {}
    deferred = [i for i, r in enumerate(runs) if r.deferred]
    if deferred and st.button(f"Retry the {len(deferred)} deferred stock(s)", type="primary"):
        trace = TraceLogger(meta.get("trace") or default_trace_path(LOG_DIR))
        pipeline = DeskPipeline.from_settings(
            trace=trace, run_critic=meta.get("run_critic", True),
            challenge_agreed_buys=meta.get("challenge_buys", CHALLENGE_AGREED_BUYS),
            max_rounds=meta.get("max_rounds", MAX_ROUNDS),
            outages=st.session_state.setdefault("outages", ModelOutages()),
        )
        with st.status("Retrying deferred stocks...", expanded=True) as status:
            for i in deferred:
                st.write(f"**{runs[i].symbol}**: resuming from the failed step")
                runs[i] = pipeline.resume_symbol(runs[i])
                st.write(f"**{runs[i].symbol}**: {outcome_text(runs[i].outcome, runs[i].recommendation)}")
            status.update(label="Retry finished", state="complete", expanded=False)
    if runs:
        st.subheader("Summary")
        st.dataframe(summary_rows(runs), hide_index=True)
        trace_path = Path(meta.get("trace", ""))
        if trace_path.exists():
            cost = summarize_trace(read_trace(trace_path), since=meta.get("started"))
            st.caption(f"{meta.get('mode')} · {meta.get('seconds', '?')}s · {cost['llm_calls']} LLM calls · "
                       f"{cost['total_tokens']:,} tokens · {cost['fallbacks']} fallbacks · "
                       f"{cost['llm_errors']} call errors · trace: {trace_path}")
        for tab, run in zip(st.tabs([r.symbol for r in runs]), runs):
            with tab:
                render_run(run)
    elif not start:
        st.info("Pick symbols and a mode in the sidebar, then run the desk.")

with state_tab:
    st.caption(f"What the scheduler (python -m agents.scheduler) remembers between cycles: {STATE_PATH}")
    if not STATE_PATH.exists():
        st.info("No state yet - the scheduler hasn't run on this machine.")
    else:
        desk = DeskState.load(STATE_PATH)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Positions (desk)", len(desk.positions))
        c2.metric("Deferred", len(desk.pending))
        c3.metric("Decisions logged", len(desk.decisions))
        c4.metric("Last decision session", desk.last_decision_session or "-")
        if desk.positions:
            st.subheader("Positions the desk opened")
            st.dataframe([{"symbol": s, **{k: v for k, v in p.items() if k != "reason"}, "reason": p.get("reason")}
                          for s, p in desk.positions.items()], hide_index=True)
        if desk.pending:
            st.subheader("Deferred (retried by the next pass)")
            st.dataframe([{"symbol": s, "session": p.get("session"), "attempts": p.get("attempts"),
                           "since": p.get("first_deferred"), "reason": p.get("reason")}
                          for s, p in desk.pending.items()], hide_index=True)
        if desk.outages:
            st.warning("Models down since: " + ", ".join(f"{m} ({t[:16]})" for m, t in desk.outages.items()))
        if desk.decisions:
            st.subheader("Decisions (newest first)")
            st.dataframe([{k: d.get(k) for k in ("session", "symbol", "outcome", "recommendation", "reason",
                                                  "decided_at")} for d in reversed(desk.decisions)],
                         hide_index=True)
        if desk.cycles:
            st.subheader("Passes (newest first)")
            st.dataframe([{"kind": c.get("kind"), "session": c.get("session"), "started": c.get("started"),
                           "decisions": len(c.get("decisions") or {}), "pending": ", ".join(c.get("pending") or []),
                           "orders": sum(1 for o in c.get("orders") or [] if "side" in o),
                           "llm calls": (c.get("llm") or {}).get("llm_calls"),
                           "llm errors": (c.get("llm") or {}).get("llm_errors"),
                           "dry run": c.get("dry_run")} for c in reversed(desk.cycles)], hide_index=True)

with trace_tab:
    files = sorted(LOG_DIR.glob("*.jsonl"), reverse=True) if LOG_DIR.exists() else []
    if not files:
        st.info(f"No traces yet in {LOG_DIR}.")
    else:
        chosen = st.selectbox("Trace file", files, format_func=lambda p: p.name)
        events = read_trace(chosen)
        cost = summarize_trace(events)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("LLM calls", cost["llm_calls"])
        c2.metric("Tokens", f"{cost['total_tokens']:,}")
        c3.metric("Call errors", cost["llm_errors"])
        c4.metric("Fallbacks", cost["fallbacks"])
        rows = [event_row(e) for e in events]
        f1, f2, f3 = st.columns(3)
        pick_symbols = f1.multiselect("Symbol", sorted({r["symbol"] for r in rows if r["symbol"]}))
        pick_agents = f2.multiselect("Agent", sorted({r["agent"] for r in rows if r["agent"]}))
        pick_events = f3.multiselect("Event", sorted({r["event"] for r in rows if r["event"]}))
        shown = [r for r in rows
                 if (not pick_symbols or r["symbol"] in pick_symbols)
                 and (not pick_agents or r["agent"] in pick_agents)
                 and (not pick_events or r["event"] in pick_events)]
        st.dataframe(shown, hide_index=True, width="stretch")
        st.download_button("Download trace", chosen.read_bytes(), file_name=chosen.name,
                           mime="application/x-ndjson")
