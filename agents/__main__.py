"""Run the macro/context agent, optionally followed by analysts, against live APIs.

    .venv/Scripts/python -m agents AAPL MSFT NVDA
    .venv/Scripts/python -m agents AAPL --no-llm                 # data only, no LLM calls
    .venv/Scripts/python -m agents AAPL --analysts analyst_1     # context + one analyst
    .venv/Scripts/python -m agents AAPL --analysts analyst_1 analyst_2
    .venv/Scripts/python -m agents AAPL --analysts analyst_1 analyst_2 --no-critic

Writes the trace to logs/trace-<UTC date>.jsonl. Without --analysts it
prints each symbol's analyst-facing context block; with them, a one-line
verdict per analyst per symbol. When both analyst_1 and analyst_2 ran it
also prints the cross-check outcome and, when that calls for it (a
buy/hold split or an agreed buy), each critic-loop round and the final
decision.
"""
from __future__ import annotations

import argparse

from .analyst_agent import AnalystVerdict
from .critic_loop import CriticLoopResult
from .pipeline import CROSS_CHECK_ROLES, DeskPipeline, default_trace_path
from .trace import TraceLogger


def _summary(v: AnalystVerdict) -> str:
    head = f"{v.symbol} {v.role} ({v.model})"
    if v.fallbacks:
        head += f" [backup: {', '.join(f['model'] for f in v.fallbacks)} failed first]"
    elif v.used_backup:
        head += f" [backup: primary {v.primary_model} not used - failed recently or excluded]"
    if not v.ok:
        return f"{head}: FAILED - {v.error}"
    check = v.evidence_check
    return (
        f"{head}: {v.recommendation.upper()} conf={v.confidence} | evidence "
        f"{check.get('matches_source', 0)} match source, {check.get('wrong_index', 0)} wrong index, "
        f"{check.get('mismatch', 0)} mismatch, {check.get('unknown_path', 0)} unknown path\n"
        f"  {v.thesis}"
    )


def _loop_summary(result: CriticLoopResult) -> str:
    lines = []
    for r in result.rounds:
        critique = r["critique"]
        if critique.get("error"):
            lines.append(f"  round {r['round']}: critic FAILED - {critique['error']}")
            continue
        per_role = {role: sum(1 for c in critique["challenges"] if c["to"] == role) for role in CROSS_CHECK_ROLES}
        votes = ", ".join(
            f"{role} {v['previous_recommendation']}->{v['recommendation'] or 'FAILED'}"
            f" (accepted {v['challenges_accepted']}, rejected {v['challenges_rejected']})"
            for role, v in r["verdicts"].items()
        )
        lines.append(f"  round {r['round']}: critic challenged {per_role}; re-votes: {votes}")
        primary = critique.get("primary_model")
        backup = " (backup)" if primary and critique.get("model") != primary else ""
        lines.append(f"    critic [{critique.get('model')}{backup}]: {critique['assessment']}")
    head = f"{result.symbol} critic loop ({result.trigger}): {result.outcome.upper()}"
    if result.recommendation:
        head += f" ({result.recommendation})"
    return "\n".join([f"{head} - {result.reason}", *lines])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("symbols", nargs="+")
    parser.add_argument("--no-llm", action="store_true", help="skip every LLM call")
    parser.add_argument(
        "--analysts", nargs="*", default=[], metavar="ROLE",
        help="analyst roles to run after the context agent, e.g. analyst_1 analyst_2",
    )
    parser.add_argument("--no-critic", action="store_true", help="stop at the cross-check")
    parser.add_argument("--no-bias", action="store_true", help="skip the bias gate on agreed buys")
    parser.add_argument("--no-technical", action="store_true", help="skip the entry-timing agent")
    parser.add_argument("--log-dir", default="logs")
    args = parser.parse_args()
    if args.no_llm and args.analysts:
        parser.error("--analysts needs the LLM; drop --no-llm")

    trace = TraceLogger(default_trace_path(args.log_dir))
    pipeline = DeskPipeline.from_settings(
        trace=trace,
        use_llm=not args.no_llm,
        analyst_roles=tuple(args.analysts),
        run_critic=not args.no_critic,
        run_bias=not args.no_bias,
        run_technical=not args.no_technical,
    )

    def on_event(stage: str, symbol: str, payload) -> None:
        if stage == "context" and not args.analysts:
            print(payload.to_prompt())
            print()
        elif stage == "verdict":
            print(_summary(payload), flush=True)
        elif stage == "cross_check":
            print(f"{symbol} cross-check: {payload.outcome.upper()}"
                  f"{f' ({payload.recommendation})' if payload.recommendation else ''} - {payload.reason}",
                  flush=True)
        elif stage == "critic_loop":
            print(_loop_summary(payload), flush=True)
        elif stage == "bias":
            print(f"{symbol} bias gate: {payload.outcome.upper()} - {payload.reason}", flush=True)
            for role, check in payload.checks.items():
                status = check.get("verdict") or f"FAILED ({check.get('error')})"
                print(f"  {role} [{check.get('model')}]: {status}, news sentiment "
                      f"{check.get('news_sentiment')} - {check.get('reason') or ''}", flush=True)
        elif stage == "technical":
            print(f"{symbol} technical [{payload.model}]: {(payload.timing or payload.outcome).upper()} - "
                  f"{payload.reason or payload.error} (4h trend {payload.trend_4h}, support {payload.support}, "
                  f"resistance {payload.resistance})", flush=True)

    pipeline.run(args.symbols, on_event)
    print(f"trace: {trace.path}")


if __name__ == "__main__":
    main()
