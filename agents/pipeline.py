"""The desk's pipeline, run the same way by the CLI, the Streamlit app and
(later) the scheduler.

Per run: macro series and the market clock once, then per stock
    context (macro agent) -> analyst_1, analyst_2 -> cross_check
    -> critic loop when the cross-check calls for it.
Stocks are processed one at a time so callers can show progress as it
happens; `on_event(stage, symbol, payload)` is called after each stage with
stage one of "context", "verdict", "cross_check", "critic_loop", "done".

Nothing here decides anything itself - it only sequences the agents and
passes each one what the previous stage produced, including the models
earlier analysts actually used so an analyst never runs on another
analyst's model (see analyst_agent / critic_agent for the rules).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from data_layer import AlpacaClient, DiskCache, EdgarClient, FinnhubClient, FredClient, get_settings
from data_layer.config import Settings
from data_layer.llm_client import LlmClient

from .analyst_agent import AnalystAgent, AnalystVerdict
from .critic_agent import CriticAgent
from .critic_loop import CHALLENGE_AGREED_BUYS, MAX_ROUNDS, CriticLoopResult, needs_critic, run_critic_loop
from .cross_check import CrossCheckResult, cross_check
from .macro_agent import MacroContextAgent, StockContext
from .trace import TraceLogger

CROSS_CHECK_ROLES = ("analyst_1", "analyst_2")

EventCallback = Callable[[str, str, Any], None]


@dataclass
class SymbolRun:
    """Everything the desk produced for one stock in one run."""

    symbol: str
    context: StockContext
    verdicts: dict[str, AnalystVerdict] = field(default_factory=dict)
    cross_check: CrossCheckResult | None = None
    critic_loop: CriticLoopResult | None = None

    @property
    def outcome(self) -> str | None:
        """The desk's final word: the critic loop's if it ran, else the
        cross-check's; None when no cross-check ran (data only, one analyst)."""
        if self.critic_loop is not None:
            return self.critic_loop.outcome
        return self.cross_check.outcome if self.cross_check is not None else None

    @property
    def recommendation(self) -> str | None:
        if self.critic_loop is not None:
            return self.critic_loop.recommendation
        return self.cross_check.recommendation if self.cross_check is not None else None


def default_trace_path(log_dir: str | Path = "logs") -> Path:
    return Path(log_dir) / f"trace-{datetime.now(timezone.utc):%Y%m%d}.jsonl"


class DeskPipeline:
    def __init__(
        self,
        macro: MacroContextAgent,
        *,
        analysts: dict[str, AnalystAgent] | None = None,
        critic: CriticAgent | None = None,
        trace: TraceLogger | None = None,
        challenge_agreed_buys: bool = CHALLENGE_AGREED_BUYS,
        max_rounds: int = MAX_ROUNDS,
    ):
        self.macro = macro
        self.analysts = analysts or {}
        self.critic = critic
        self.trace = trace
        self.challenge_agreed_buys = challenge_agreed_buys
        self.max_rounds = max_rounds

    @classmethod
    def from_settings(
        cls,
        *,
        settings: Settings | None = None,
        trace: TraceLogger | None = None,
        use_llm: bool = True,
        analyst_roles: tuple[str, ...] = CROSS_CHECK_ROLES,
        run_critic: bool = True,
        challenge_agreed_buys: bool = CHALLENGE_AGREED_BUYS,
        max_rounds: int = MAX_ROUNDS,
    ) -> DeskPipeline:
        """The live desk: real data clients (sharing one on-disk cache) and,
        unless use_llm is False, the LLM agents with their default models and
        backups. use_llm=False is data only - no Build credits spent."""
        settings = settings or get_settings()
        cache = DiskCache(settings.cache_dir)
        llm = LlmClient(settings=settings) if use_llm else None
        macro = MacroContextAgent(
            alpaca=AlpacaClient(settings=settings, cache=cache),
            fred=FredClient(settings=settings, cache=cache),
            edgar=EdgarClient(settings=settings, cache=cache),
            finnhub=FinnhubClient(settings=settings, cache=cache),
            llm=llm,
            trace=trace,
        )
        analysts = {role: AnalystAgent(llm, role, trace=trace) for role in analyst_roles} if use_llm else {}
        critic = CriticAgent(llm, trace=trace) if (use_llm and run_critic) else None
        return cls(macro, analysts=analysts, critic=critic, trace=trace,
                   challenge_agreed_buys=challenge_agreed_buys, max_rounds=max_rounds)

    def run(self, symbols: list[str], on_event: EventCallback | None = None) -> list[SymbolRun]:
        emit = on_event or (lambda stage, symbol, payload: None)
        shared = self.macro.prepare()
        runs = []
        for symbol in symbols:
            ctx = self.macro.build_context(symbol, shared)
            emit("context", ctx.symbol, ctx)
            run = self.run_symbol(ctx, emit)
            emit("done", ctx.symbol, run)
            runs.append(run)
        return runs

    def run_symbol(self, ctx: StockContext, emit: EventCallback) -> SymbolRun:
        run = SymbolRun(symbol=ctx.symbol, context=ctx)
        for role, analyst in self.analysts.items():
            # An analyst never runs on a model another analyst already used
            # (a backup could otherwise land two analysts on one model).
            run.verdicts[role] = analyst.analyze(ctx, exclude={v.model for v in run.verdicts.values()})
            emit("verdict", ctx.symbol, run.verdicts[role])
        if not set(CROSS_CHECK_ROLES) <= run.verdicts.keys():
            return run
        pair = [run.verdicts[r] for r in CROSS_CHECK_ROLES]
        run.cross_check = cross_check(*pair, trace=self.trace)
        emit("cross_check", ctx.symbol, run.cross_check)
        trigger = needs_critic(run.cross_check, challenge_agreed_buys=self.challenge_agreed_buys)
        if self.critic is not None and trigger is not None:
            run.critic_loop = run_critic_loop(
                ctx,
                {r: run.verdicts[r] for r in CROSS_CHECK_ROLES},
                {r: self.analysts[r] for r in CROSS_CHECK_ROLES},
                self.critic,
                trigger,
                trace=self.trace,
                max_rounds=self.max_rounds,
            )
            emit("critic_loop", ctx.symbol, run.critic_loop)
        return run
