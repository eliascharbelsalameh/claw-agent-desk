"""Scripted stand-ins for the live clients and agents, shared by the CLI,
pipeline and app tests. Nothing here touches the network."""
import agents.pipeline as pipeline
from agents.analyst_agent import AnalystVerdict
from agents.critic_agent import Critique
from agents.macro_agent import SharedContext, StockContext

FIRST = {"AAPL": ("buy", "hold"), "MSFT": ("buy", "buy"), "NVDA": ("hold", "hold")}
REVOTE = {"AAPL": ("hold", "hold"), "MSFT": ("buy", "buy")}


def _verdict(symbol, role, rec, **extra):
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
        self.index = 0 if role == "analyst_1" else 1

    def analyze(self, ctx, exclude=()):
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


def patch_live_components(monkeypatch, tmp_path):
    """Replace every client and agent DeskPipeline.from_settings builds."""
    for name in ("AlpacaClient", "FredClient", "EdgarClient", "FinnhubClient", "LlmClient", "DiskCache"):
        monkeypatch.setattr(pipeline, name, lambda *a, **k: None)
    monkeypatch.setattr(pipeline, "get_settings", lambda: type("S", (), {"cache_dir": tmp_path})())
    monkeypatch.setattr(pipeline, "MacroContextAgent", FakeMacro)
    monkeypatch.setattr(pipeline, "AnalystAgent", FakeAnalyst)
    monkeypatch.setattr(pipeline, "CriticAgent", FakeCritic)
