"""The Architecture and Flow diagrams: built from the code, valid DOT."""
from agents.critic_loop import MAX_ROUNDS
from agents.portfolio import HOLDING_SESSIONS
from data_layer.llm_client import AGENT_MODEL_BACKUPS, AGENT_MODELS
from ui.diagrams import (
    HANDOFFS,
    agent_rows,
    architecture_dot,
    architecture_legend,
    flow_dot,
    flow_legend,
    outcome_rows,
)


def _balanced(dot: str) -> bool:
    depth = 0
    for ch in dot:
        depth += {"{": 1, "}": -1}.get(ch, 0)
        if depth < 0:
            return False
    return depth == 0


def test_both_diagrams_are_dot_graphs_with_every_current_model():
    for dot in (architecture_dot(), flow_dot()):
        assert dot.startswith("digraph ") and dot.rstrip().endswith("}") and _balanced(dot)
    arch, flow = architecture_dot(), flow_dot()
    for role, model in AGENT_MODELS.items():
        assert model.split("/")[-1] in arch, role
    assert AGENT_MODEL_BACKUPS["analyst_1"][0].split("/")[-1] in arch  # analyst_1's stand-in is drawn
    for model in {m for r, m in AGENT_MODELS.items() if r != "analyst_1"}:
        assert model.split("/")[-1] in flow


def test_the_flow_has_bpmn_events_gateways_and_the_loops():
    flow = flow_dot()
    assert "start [" in flow and "shape=circle" in flow and "shape=doublecircle" in flow and "shape=diamond" in flow
    assert "end [" in flow and "penwidth=3.5" in flow
    assert "poll -> clock" in flow  # the scheduler's polling loop
    assert "xc2 -> critic" in flow and f"up to {MAX_ROUNDS} rounds" in flow  # the critic loop
    assert "resume -> analysts" in flow and "deferred -> save" in flow  # deferral and retry
    for outcome in ("o_buy", "o_wait", "o_veto", "o_failed", "o_hold", "o_abort"):
        assert f"{outcome} -> pf" in flow  # every decision reaches the portfolio
    assert f"held {HOLDING_SESSIONS} sessions" in flow


def test_labels_are_escaped_for_graphviz():
    # HTML-like labels: a raw & or < from a model name or text would break the graph
    for dot in (architecture_dot(), flow_dot()):
        assert " & " not in dot


def test_tables_and_legends():
    rows = agent_rows()
    assert [r["role"] for r in rows] == list(AGENT_MODELS)
    assert all(r["receives"] and r["returns"] for r in rows)
    assert next(r for r in rows if r["role"] == "analyst_2")["backup"] == "none"
    assert len(HANDOFFS) >= 8 and all(len(h) == 2 for h in HANDOFFS)
    outcomes = [r["outcome"] for r in outcome_rows()]
    assert outcomes[0] == "Buy" and "Deferred" in outcomes and "Abort" in outcomes
    assert "Start event" in flow_legend() and "LLM agent" in architecture_legend()
