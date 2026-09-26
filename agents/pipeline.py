"""The desk's pipeline, run the same way by the CLI, the Streamlit app and
the scheduler.

Per run: macro series, the market clock and the benchmark once, then per
stock
    context (macro agent) -> analyst_1 + analyst_2 (in parallel)
    -> cross_check -> critic loop when the cross-check calls for it
    -> bias gate (bias_1, bias_2) when the desk agreed on buy
    -> technical agent (entry timing) when the bias gate passed.
Stocks are processed one at a time so callers can show progress as it
happens; `on_event(stage, symbol, payload)` is called after each stage with
stage one of "context", "verdict", "cross_check", "critic_loop", "bias",
"technical", "done".
Events are always emitted from the calling thread.

Decided Sept 26, 2026: a Build failure is not a decision. When an analyst's
model (or every critic candidate) can't be reached, the stock comes out
"deferred" with everything done so far kept, and `resume_symbol` picks it
up at the failed step - the scheduler does that on its next cycle. Each
analyst answers on its own primary model; its backups are allowed only after
the primary has been down for ANALYST_BACKUP_AFTER (tracked in
ModelOutages, which the scheduler keeps across cycles).

Nothing here decides anything itself - it only sequences the agents and
passes each one what the previous stage produced.
"""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from data_layer import AlpacaClient, DiskCache, EdgarClient, FinnhubClient, FredClient, get_settings
from data_layer.config import Settings
from data_layer.llm_client import LlmClient

from .analyst_agent import AnalystAgent, AnalystVerdict
from .bias_agent import BIAS_ROLES, BiasAgent, BiasGateResult, bias_gate
from .bias_agent import DEFERRED as BIAS_DEFERRED
from .critic_agent import CriticAgent, Critique
from .critic_loop import CHALLENGE_AGREED_BUYS, MAX_ROUNDS, CriticLoopResult, needs_critic, run_critic_loop
from .cross_check import AGREE, DEFERRED, CrossCheckResult, cross_check
from .macro_agent import MacroContextAgent, StockContext
from .technical_agent import DEFERRED as TECHNICAL_DEFERRED
from .technical_agent import TechnicalAgent, TechnicalResult
from .trace import TraceLogger

CROSS_CHECK_ROLES = ("analyst_1", "analyst_2")

# How long an analyst's primary must have been unreachable before its
# backups may answer instead of the stock being deferred again. Long enough
# that ordinary Build drops (minutes) never hand a decision to a backup.
ANALYST_BACKUP_AFTER = timedelta(hours=3)

EventCallback = Callable[[str, str, Any], None]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class ModelOutages:
    """When each model's current outage began: the first failed call since
    its last successful one (UTC, ISO). Serializable, so the scheduler can
    keep it across cycles and restarts."""

    down_since: dict[str, str] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def record(self, model: str, ok: bool, now: datetime) -> None:
        with self._lock:
            if ok:
                self.down_since.pop(model, None)
            else:
                self.down_since.setdefault(model, now.isoformat())

    def down_for(self, model: str, now: datetime) -> timedelta | None:
        since = self.down_since.get(model)
        return now - datetime.fromisoformat(since) if since else None

    def note(self, result: AnalystVerdict | Critique, now: datetime) -> None:
        """Learn from one agent result: every model that failed a call for it
        is down, the model that answered is up."""
        for failed in result.fallbacks:
            if failed.get("call_failed"):
                self.record(failed["model"], ok=False, now=now)
        if result.ok:
            self.record(result.model, ok=True, now=now)
        elif result.call_failed:
            self.record(result.model, ok=False, now=now)


@dataclass
class SymbolRun:
    """Everything the desk produced for one stock in one run."""

    symbol: str
    context: StockContext
    verdicts: dict[str, AnalystVerdict] = field(default_factory=dict)
    cross_check: CrossCheckResult | None = None
    critic_loop: CriticLoopResult | None = None
    # Only for an agreed buy (spec section 3, step 4).
    bias: BiasGateResult | None = None
    # Only for an agreed buy that passed the bias gate (step 5).
    technical: TechnicalResult | None = None

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

    @property
    def deferred(self) -> bool:
        """Waiting on a Build failure somewhere - the decision itself, or a
        gate on an agreed buy."""
        return (self.outcome == DEFERRED
                or (self.bias is not None and self.bias.outcome == BIAS_DEFERRED)
                or (self.technical is not None and self.technical.outcome == TECHNICAL_DEFERRED))

    def final_verdicts(self) -> dict[str, AnalystVerdict]:
        """The verdicts the decision rests on: the critic loop's last ones
        if it ran, else the first ones."""
        if self.critic_loop is not None and self.critic_loop.verdicts:
            return {role: AnalystVerdict(**v) for role, v in self.critic_loop.verdicts.items()}
        return dict(self.verdicts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "context": self.context.to_dict(),
            "verdicts": {role: v.to_dict() for role, v in self.verdicts.items()},
            "cross_check": self.cross_check.to_dict() if self.cross_check else None,
            "critic_loop": self.critic_loop.to_dict() if self.critic_loop else None,
            "bias": self.bias.to_dict() if self.bias else None,
            "technical": self.technical.to_dict() if self.technical else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SymbolRun:
        return cls(
            symbol=data["symbol"],
            context=StockContext(**data["context"]),
            verdicts={role: AnalystVerdict(**v) for role, v in data.get("verdicts", {}).items()},
            cross_check=CrossCheckResult(**data["cross_check"]) if data.get("cross_check") else None,
            critic_loop=CriticLoopResult(**data["critic_loop"]) if data.get("critic_loop") else None,
            bias=BiasGateResult(**data["bias"]) if data.get("bias") else None,
            technical=TechnicalResult(**data["technical"]) if data.get("technical") else None,
        )


def default_trace_path(log_dir: str | Path = "logs") -> Path:
    return Path(log_dir) / f"trace-{datetime.now(timezone.utc):%Y%m%d}.jsonl"


class DeskPipeline:
    def __init__(
        self,
        macro: MacroContextAgent,
        *,
        analysts: dict[str, AnalystAgent] | None = None,
        critic: CriticAgent | None = None,
        bias_agents: dict[str, BiasAgent] | None = None,
        technical: TechnicalAgent | None = None,
        trace: TraceLogger | None = None,
        challenge_agreed_buys: bool = CHALLENGE_AGREED_BUYS,
        max_rounds: int = MAX_ROUNDS,
        outages: ModelOutages | None = None,
        backup_after: timedelta = ANALYST_BACKUP_AFTER,
        now: Callable[[], datetime] = _utcnow,
    ):
        self.macro = macro
        self.analysts = analysts or {}
        self.critic = critic
        self.bias_agents = bias_agents or {}
        self.technical = technical
        self.trace = trace
        self.challenge_agreed_buys = challenge_agreed_buys
        self.max_rounds = max_rounds
        self.outages = outages if outages is not None else ModelOutages()
        self.backup_after = backup_after
        self._now = now

    @classmethod
    def from_settings(
        cls,
        *,
        settings: Settings | None = None,
        trace: TraceLogger | None = None,
        use_llm: bool = True,
        analyst_roles: tuple[str, ...] = CROSS_CHECK_ROLES,
        run_critic: bool = True,
        run_bias: bool = True,
        run_technical: bool = True,
        challenge_agreed_buys: bool = CHALLENGE_AGREED_BUYS,
        max_rounds: int = MAX_ROUNDS,
        outages: ModelOutages | None = None,
    ) -> DeskPipeline:
        """The live desk: real data clients (sharing one on-disk cache) and,
        unless use_llm is False, the LLM agents with their default models and
        backups. use_llm=False is data only - no LLM calls."""
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
        bias_agents = (
            {role: BiasAgent(llm, role, trace=trace) for role in BIAS_ROLES} if (use_llm and run_bias) else {}
        )
        technical = TechnicalAgent(llm, trace=trace) if (use_llm and run_technical) else None
        return cls(macro, analysts=analysts, critic=critic, bias_agents=bias_agents, technical=technical,
                   trace=trace, challenge_agreed_buys=challenge_agreed_buys, max_rounds=max_rounds,
                   outages=outages)

    def run(self, symbols: list[str], on_event: EventCallback | None = None, workers: int = 1) -> list[SymbolRun]:
        """One SymbolRun per symbol, in the order given. With workers > 1
        several stocks go through the desk at once - the scheduler does this
        so a full watchlist finishes before the open (Sept 26, 2026: 3 stocks
        took 41 minutes one after another on a slow Build morning). Events
        then come from worker threads, so interactive callers (the app) keep
        the default of 1."""
        emit = on_event or (lambda stage, symbol, payload: None)
        shared = self.macro.prepare()

        def one(symbol: str) -> SymbolRun:
            ctx = self.macro.build_context(symbol, shared)
            emit("context", ctx.symbol, ctx)
            run = self.run_symbol(ctx, emit)
            emit("done", ctx.symbol, run)
            return run

        if workers <= 1:
            return [one(symbol) for symbol in symbols]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(one, symbols))

    def allow_backup(self, role: str) -> bool:
        """Whether `role` may answer on a backup: only once its primary has
        been unreachable for backup_after."""
        analyst = self.analysts[role]
        down = self.outages.down_for(analyst.model, self._now())
        return len(analyst.models) > 1 and down is not None and down >= self.backup_after

    def _analyze(self, run: SymbolRun, roles: list[str], emit: EventCallback) -> None:
        """First verdicts for `roles`, in parallel. The analysts can't land
        on one model: each answers on its own primary, and a role's backups
        never overlap the other's list (tests/test_backups.py) - the
        cross-check still aborts if they ever did."""
        if not roles:
            return
        with ThreadPoolExecutor(max_workers=len(roles)) as pool:
            futures = {
                pool.submit(self.analysts[role].analyze, run.context, allow_backup=self.allow_backup(role)): role
                for role in roles
            }
            for future in as_completed(futures):
                role = futures[future]
                verdict = future.result()
                run.verdicts[role] = verdict
                self.outages.note(verdict, self._now())
                emit("verdict", run.symbol, verdict)
        # Keep the configured role order (as_completed returns them as they finish).
        run.verdicts = {role: run.verdicts[role] for role in self.analysts if role in run.verdicts}

    def run_symbol(self, ctx: StockContext, emit: EventCallback | None = None) -> SymbolRun:
        emit = emit or (lambda stage, symbol, payload: None)
        run = SymbolRun(symbol=ctx.symbol, context=ctx)
        self._analyze(run, list(self.analysts), emit)
        return self._decide(run, emit)

    def _decide(self, run: SymbolRun, emit: EventCallback) -> SymbolRun:
        """Cross-check the first verdicts, then the critic loop if called
        for, then the gates on an agreed buy (bias, then technical)."""
        if not set(CROSS_CHECK_ROLES) <= run.verdicts.keys():
            return run
        pair = [run.verdicts[r] for r in CROSS_CHECK_ROLES]
        run.cross_check = cross_check(*pair, trace=self.trace)
        emit("cross_check", run.symbol, run.cross_check)
        trigger = needs_critic(run.cross_check, challenge_agreed_buys=self.challenge_agreed_buys)
        if self.critic is not None and trigger is not None:
            run.critic_loop = self._critic_loop(run, trigger)
            emit("critic_loop", run.symbol, run.critic_loop)
        return self._bias(run, emit)

    def _bias(self, run: SymbolRun, emit: EventCallback, previous: BiasGateResult | None = None) -> SymbolRun:
        """The bias gate, for an agreed buy only (the one outcome that opens
        a position)."""
        if self.bias_agents and run.outcome == AGREE and run.recommendation == "buy":
            run.bias = bias_gate(run.context, run.final_verdicts(), self.bias_agents,
                                 previous=previous, trace=self.trace)
            emit("bias", run.symbol, run.bias)
        return self._technical(run, emit)

    def _technical(self, run: SymbolRun, emit: EventCallback) -> SymbolRun:
        """Entry timing, for an agreed buy that no bias gate stopped."""
        cleared = run.bias is None or run.bias.gate["passed"]
        if self.technical is not None and run.outcome == AGREE and run.recommendation == "buy" and cleared:
            run.technical = self.technical.time_entry(run.context)
            emit("technical", run.symbol, run.technical)
        return run

    def _critic_loop(self, run: SymbolRun, trigger: str, resume_from: CriticLoopResult | None = None):
        result = run_critic_loop(
            run.context,
            {r: run.verdicts[r] for r in CROSS_CHECK_ROLES},
            {r: self.analysts[r] for r in CROSS_CHECK_ROLES},
            self.critic,
            trigger,
            trace=self.trace,
            max_rounds=self.max_rounds,
            resume_from=resume_from,
        )
        now = self._now()
        for rnd in result.rounds:
            if isinstance(rnd.get("critique"), dict):
                self.outages.note(Critique(**rnd["critique"]), now)
        return result

    def resume_symbol(self, run: SymbolRun, emit: EventCallback | None = None) -> SymbolRun:
        """Continue a deferred stock from the step that failed: re-ask only
        the analysts that couldn't be reached (then decide as usual), pick
        the critic loop up at its pending round, or re-run the gate that was
        deferred (the bias gate keeps the check that came back). Anything
        not deferred is returned unchanged."""
        emit = emit or (lambda stage, symbol, payload: None)
        if run.cross_check is not None and run.cross_check.outcome == DEFERRED:
            retry = [r for r, v in run.verdicts.items() if not v.ok and v.call_failed]
            self._log("resume", {"symbol": run.symbol, "step": "analysts", "roles": retry})
            self._analyze(run, retry, emit)
            run.cross_check = None
            run.critic_loop = None
            return self._decide(run, emit)
        if run.critic_loop is not None and run.critic_loop.outcome == DEFERRED:
            self._log("resume", {"symbol": run.symbol, "step": "critic_loop",
                                 "round": (run.critic_loop.resume or {}).get("round")})
            run.critic_loop = self._critic_loop(run, run.critic_loop.trigger, resume_from=run.critic_loop)
            emit("critic_loop", run.symbol, run.critic_loop)
            return self._bias(run, emit)
        if run.bias is not None and run.bias.outcome == BIAS_DEFERRED:
            self._log("resume", {"symbol": run.symbol, "step": "bias"})
            return self._bias(run, emit, previous=run.bias)
        if run.technical is not None and run.technical.outcome == TECHNICAL_DEFERRED:
            self._log("resume", {"symbol": run.symbol, "step": "technical"})
            return self._technical(run, emit)
        return run

    def _log(self, event: str, payload: dict[str, Any]) -> None:
        if self.trace is not None:
            self.trace.log("pipeline", event, payload)
