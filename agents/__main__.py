"""Run the macro/context agent, optionally followed by analysts, against live APIs.

    .venv/Scripts/python -m agents AAPL MSFT NVDA
    .venv/Scripts/python -m agents AAPL --no-llm                 # data only, no Build credits
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
from datetime import datetime, timezone
from pathlib import Path

from data_layer import AlpacaClient, DiskCache, EdgarClient, FinnhubClient, FredClient, get_settings
from data_layer.llm_client import LlmClient

from .analyst_agent import AnalystAgent, AnalystVerdict
from .critic_agent import CriticAgent
from .critic_loop import CriticLoopResult, needs_critic, run_critic_loop
from .cross_check import cross_check
from .macro_agent import MacroContextAgent
from .trace import TraceLogger

CROSS_CHECK_ROLES = ("analyst_1", "analyst_2")


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
    parser.add_argument("--log-dir", default="logs")
    args = parser.parse_args()
    if args.no_llm and args.analysts:
        parser.error("--analysts needs the LLM; drop --no-llm")

    settings = get_settings()
    cache = DiskCache(settings.cache_dir)
    trace = TraceLogger(
        Path(args.log_dir) / f"trace-{datetime.now(timezone.utc):%Y%m%d}.jsonl"
    )
    llm = None if args.no_llm else LlmClient(settings=settings)
    agent = MacroContextAgent(
        alpaca=AlpacaClient(settings=settings, cache=cache),
        fred=FredClient(settings=settings, cache=cache),
        edgar=EdgarClient(settings=settings, cache=cache),
        finnhub=FinnhubClient(settings=settings, cache=cache),
        llm=llm,
        trace=trace,
    )
    analysts = {role: AnalystAgent(llm, role, trace=trace) for role in args.analysts}
    critic = None if args.no_critic else CriticAgent(llm, trace=trace)

    for ctx in agent.run(args.symbols).values():
        if not analysts:
            print(ctx.to_prompt())
            print()
            continue
        verdicts = {}
        for role, analyst in analysts.items():
            # An analyst never runs on a model another analyst already used
            # (a backup could otherwise land two analysts on one model).
            verdicts[role] = analyst.analyze(ctx, exclude={v.model for v in verdicts.values()})
            print(_summary(verdicts[role]), flush=True)
        if not set(CROSS_CHECK_ROLES) <= verdicts.keys():
            continue
        result = cross_check(*(verdicts[r] for r in CROSS_CHECK_ROLES), trace=trace)
        print(f"{ctx.symbol} cross-check: {result.outcome.upper()}"
              f"{f' ({result.recommendation})' if result.recommendation else ''} - {result.reason}",
              flush=True)
        trigger = needs_critic(result)
        if critic is not None and trigger is not None:
            loop = run_critic_loop(
                ctx,
                {r: verdicts[r] for r in CROSS_CHECK_ROLES},
                {r: analysts[r] for r in CROSS_CHECK_ROLES},
                critic,
                trigger,
                trace=trace,
            )
            print(_loop_summary(loop), flush=True)
    print(f"trace: {trace.path}")


if __name__ == "__main__":
    main()
