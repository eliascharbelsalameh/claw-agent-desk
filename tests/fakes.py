"""Scripted stand-ins for the live clients and agents, shared by the CLI,
pipeline and app tests. Nothing here touches the network."""
import agents.pipeline as pipeline
from agents.analyst_agent import AnalystVerdict
from agents.bias_agent import BiasCheck
from agents.critic_agent import Critique
from agents.macro_agent import SharedContext, StockContext
from agents.technical_agent import PASSED, WAITING, TechnicalResult

FIRST = {"AAPL": ("buy", "hold"), "MSFT": ("buy", "buy"), "NVDA": ("hold", "hold")}
REVOTE = {"AAPL": ("hold", "hold"), "MSFT": ("buy", "buy")}


def _verdict(symbol, role, rec, **extra):
    """A verdict from lab-{role}/m; with error=..., a failed one (no content)."""
    if extra.get("error"):
        return AnalystVerdict(symbol=symbol, role=role, model=f"lab-{role}/m", generated_at="t", **extra)
    return AnalystVerdict(symbol=symbol, role=role, model=f"lab-{role}/m", generated_at="t",
                          recommendation=rec, confidence=0.7, thesis=f"{role} says {rec}",
                          drivers=["d"], risks=["r"], **extra)


class FakeMacro:
    def __init__(self, **kwargs):
        pass

    def prepare(self):
        return SharedContext(macro={}, gaps=[], clock=None)

    def build_context(self, symbol, shared):
        return StockContext(symbol=symbol, generated_at="t", macro=shared.macro)


class FakeAnalyst:
    excludes: dict = {}

    def __init__(self, llm, role, trace=None):
        self.role, self.model = role, f"lab-{role}/m"
        self.models = [self.model]
        self.index = 0 if role == "analyst_1" else 1

    def analyze(self, ctx, *, exclude=(), allow_backup=False):
        FakeAnalyst.excludes[(ctx.symbol, self.role)] = set(exclude)
        return _verdict(ctx.symbol, self.role, FIRST[ctx.symbol][self.index])

    def revise(self, ctx, own, other, *, assessment, challenges_to_me, challenges_to_other, review_round):
        return _verdict(ctx.symbol, self.role, REVOTE[ctx.symbol][self.index], review_round=review_round,
                        previous_recommendation=own.recommendation,
                        response_to_critique=[{"point": "p", "accept": True, "reason": "fair"}])


class FakeCritic:
    def __init__(self, llm, trace=None):
        pass

    def review(self, ctx, verdicts, review_round):
        return Critique(symbol=ctx.symbol, model="critic/m", review_round=review_round, generated_at="t",
                        assessment=f"{ctx.symbol} critique",
                        challenges=[{"to": "analyst_1", "point": "p", "why": "w"}])


class FakeBias:
    """Passes every buy, except the symbols in `flag` (a class attribute)."""
    flag: set = set()

    def __init__(self, llm, role, trace=None):
        self.role, self.model = role, f"bias-{role}/m"
        self.models = [self.model]

    def review(self, ctx, verdicts, *, exclude=()):
        flagged = ctx.symbol in FakeBias.flag
        return BiasCheck(symbol=ctx.symbol, role=self.role, model=self.model, generated_at="t",
                         verdict="flag" if flagged else "pass", news_sentiment=0.4,
                         checks={"news_driven": {"skewed": flagged, "why": "w"},
                                 "stale_news": {"skewed": False, "why": "w"},
                                 "trend_chasing": {"skewed": False, "why": "w"}},
                         reason=f"{self.role} {'flags' if flagged else 'passes'} {ctx.symbol}")


class FakeTechnical:
    """Enters every buy, except the symbols in `wait` (a class attribute)."""
    wait: set = set()

    def __init__(self, llm, trace=None):
        self.model = "tech/m"

    def time_entry(self, ctx):
        waiting = ctx.symbol in FakeTechnical.wait
        return TechnicalResult(symbol=ctx.symbol, model=self.model, generated_at="t",
                               outcome=WAITING if waiting else PASSED, timing="wait" if waiting else "enter",
                               trend_4h="up", support=90.0, resistance=110.0,
                               reason=f"{ctx.symbol} {'into resistance' if waiting else 'clean entry'}")


def patch_live_components(monkeypatch, tmp_path):
    """Replace every client and agent DeskPipeline.from_settings builds."""
    for name in ("AlpacaClient", "FredClient", "EdgarClient", "FinnhubClient", "LlmClient", "DiskCache"):
        monkeypatch.setattr(pipeline, name, lambda *a, **k: None)
    monkeypatch.setattr(pipeline, "get_settings", lambda: type("S", (), {"cache_dir": tmp_path})())
    monkeypatch.setattr(pipeline, "MacroContextAgent", FakeMacro)
    monkeypatch.setattr(pipeline, "AnalystAgent", FakeAnalyst)
    monkeypatch.setattr(pipeline, "CriticAgent", FakeCritic)
    monkeypatch.setattr(pipeline, "BiasAgent", FakeBias)
    monkeypatch.setattr(pipeline, "TechnicalAgent", FakeTechnical)
