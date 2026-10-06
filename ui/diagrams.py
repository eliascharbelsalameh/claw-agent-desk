"""The desk drawn for the app's Architecture and Flow tabs (and the demo).

Both diagrams are Graphviz DOT, rendered by st.graphviz_chart in the browser
(no extra dependency). Every model name and number in them comes from the
code - AGENT_MODELS, the backups, the loop and gate limits, the portfolio
rules, the schedule - so the pictures change when the desk does.
"""
from __future__ import annotations

from html import escape

from agents.bias_agent import BIAS_SOLO_AFTER
from agents.critic_agent import MAX_CHALLENGES_PER_ANALYST
from agents.critic_loop import MAX_ROUNDS
from agents.portfolio import HOLDING_SESSIONS, MAX_POSITIONS, POSITION_FRACTION
from agents.scheduler import (
    DECISION_TIME_ET,
    DEFAULT_WATCHLIST,
    POLL_SECONDS,
    RECHECK_TIME_ET,
    RETRY_EVERY,
    RETRY_UNTIL_ET,
    WORKERS,
)
from data_layer.llm_client import AGENT_MODEL_BACKUPS, AGENT_MODELS

# Fill and border per kind of block; the legend in the app uses the same.
KINDS = {
    "api": ("#dbeafe", "#2563eb", "API call"),
    "agent": ("#e4f4cc", "#4d7c0f", "LLM agent (NVIDIA Build)"),
    "rule": ("#fef3c7", "#b45309", "Deterministic rule"),
    "store": ("#ede9fe", "#6d28d9", "Stored data"),
    "runtime": ("#e0e7ff", "#4338ca", "Runtime"),
    "outcome": ("#f1f5f9", "#64748b", "Outcome"),
    "bought": ("#bbf7d0", "#15803d", "Buy, position"),
    "event": ("#fff7ed", "#c2410c", "Intermediate event (wait, deferral)"),
    "gateway": ("#ffffff", "#b45309", "Decision point"),
}

DEFER_COLOR = "#c2410c"  # deferral edges: a model couldn't be reached
LOOP_COLOR = "#4d7c0f"  # the critic loop's next round

HEADER = [
    'graph [bgcolor="#ffffff", pad="0.35", fontname="Helvetica", fontsize=12, newrank=true, compound=true];',
    'node [fontname="Helvetica", fontsize=11, shape=box, style="rounded,filled", penwidth=1.3, margin="0.16,0.09"];',
    'edge [fontname="Helvetica", fontsize=9, color="#64748b", fontcolor="#475569", arrowsize=0.7, penwidth=1.1];',
]


def model(role: str) -> str:
    return AGENT_MODELS[role].split("/")[-1]


def backups(role: str) -> list[str]:
    return [m.split("/")[-1] for m in AGENT_MODEL_BACKUPS.get(role, [])]


def _html(title: str, lines: tuple[str, ...]) -> str:
    small = "".join(f'<br/><font point-size="9">{escape(line)}</font>' for line in lines)
    return f"<<b>{escape(title)}</b>{small}>"


def _node(name: str, kind: str, title: str, *lines: str, **attrs: str) -> str:
    fill, border, _ = KINDS[kind]
    extra = {"shape": "cylinder"} if kind == "store" else {}
    extra.update(attrs)
    rest = "".join(f', {k}="{v}"' for k, v in extra.items())
    return f'{name} [label={_html(title, lines)}, fillcolor="{fill}", color="{border}"{rest}];'


def _event(name: str, text: str, kind: str = "intermediate") -> str:
    """BPMN-style events: start (thin circle), intermediate (double), end (thick)."""
    fill, border, _ = KINDS["event"]
    shape, pen = {"start": ("circle", "1.6"), "intermediate": ("doublecircle", "1.2"), "end": ("circle", "3.5")}[kind]
    if kind == "start":
        fill, border = "#dcfce7", "#15803d"
    if kind == "end":
        fill, border = "#f8fafc", "#334155"
    return (f'{name} [label="{text}", shape={shape}, style=filled, fillcolor="{fill}", color="{border}", '
            f'penwidth={pen}, fontsize=9, width=0.9, fixedsize=true];')


def _gate(name: str, text: str) -> str:
    fill, border, _ = KINDS["gateway"]
    return (f'{name} [label="{text}", shape=diamond, style=filled, fillcolor="{fill}", color="{border}", '
            f'fontsize=9, margin="0.02,0.02"];')


def _edge(a: str, b: str, label: str = "", **attrs: str) -> str:
    parts = [f'label="{label}"'] if label else []
    parts += [f"{k}={v}" if k in ("constraint",) else f'{k}="{v}"' for k, v in attrs.items()]
    return f"{a} -> {b}" + (f" [{', '.join(parts)}]" if parts else "") + ";"


def _cluster(name: str, title: str, lines: list[str], *, fill: str = "#f8fafc",
             style: str = "rounded,filled") -> list[str]:
    return ([f"subgraph cluster_{name} {{",
             f'label=<<b>{escape(title)}</b>>; style="{style}"; fillcolor="{fill}"; color="#cbd5e1"; '
             'fontsize=12; margin=12;']
            + lines + ["}"])


def _et(t) -> str:
    return f"{t.hour:02d}:{t.minute:02d} ET"


def _minutes(delta) -> int:
    return int(delta.total_seconds() // 60)


def architecture_dot() -> str:
    """Who talks to whom, and what they pass - the desk's components left to
    right, the runtime underneath."""
    a1_backup = backups("analyst_1")
    lines = ["digraph architecture {", *HEADER,
             'graph [rankdir=TB, nodesep="0.35", ranksep="0.45", splines=true];']
    lines += _cluster("sources", "Data sources · REST APIs", [
        _node("alpaca", "api", "Alpaca", "bars 1D · 4H · 1H, split-adjusted", "IEX relative volume · clock · calendar"),
        _node("fred", "api", "FRED", "rates · spreads · CPI"),
        _node("edgar", "api", "SEC EDGAR", "filings · US-GAAP fundamentals"),
        _node("finnhub", "api", "Finnhub", "company news · earnings calendar"),
    ])
    lines += _cluster("context", "1 · Context", [
        _node("facts", "rule", "Computed facts", "TTM P/E · P/S · market cap", "RSI · SMAs · ATR · vs SPY · earnings"),
        _node("macro", "agent", "Macro/context agent", model("macro") + " · thinking off", "neutral briefing, no opinions"),
        _node("packet", "outcome", "Context packet", "briefing + source facts + computed blocks", "limitations · data gaps",
              shape="note"),
    ])
    lines += _cluster("desk", "2–3 · Analysts and critic", [
        _node("a1", "agent", "Analyst 1", model("analyst_1"),
              *(["stand-in: " + a1_backup[0] + " (if unreachable)"] if a1_backup else [])),
        _node("a2", "agent", "Analyst 2", model("analyst_2")),
        _node("xc", "rule", "Cross-check", "strict match of the two calls"),
        _node("critic", "agent", "Critic", model("critic"), f"up to {MAX_ROUNDS} rounds · challenges, never votes"),
    ])
    lines += _cluster("gates", "4–5 · Gates on an agreed buy", [
        _node("b1", "agent", "Bias 1", model("bias_1"), "goes first; its pass clears"),
        _node("b2", "agent", "Bias 2", model("bias_2"), "asked only on a bias 1 flag"),
        _node("tech", "agent", "Technical agent", model("technical"), "4h trend → 1h entry"),
    ])
    lines += _cluster("output", "6 · Output", [
        _node("pf", "rule", "Portfolio rules", f"{POSITION_FRACTION:.0%} of equity · max {MAX_POSITIONS} positions",
              f"{HOLDING_SESSIONS}-session hold · long-only · no margin"),
        _node("paper", "api", "Alpaca paper account", "market orders with client_order_id"),
    ])
    lines += _cluster("runtime", "Runtime · Oracle A1", [
        _node("sched", "runtime", "Scheduler (systemd)", f"decision cycle {_et(DECISION_TIME_ET)} · {WORKERS} stocks at once",
              f"retries every {_minutes(RETRY_EVERY)} min until {_et(RETRY_UNTIL_ET)}"),
        _node("state", "store", "State file", "deferred runs · outages · positions", "decisions · passes · next pass"),
        _node("trace", "store", "Trace (JSONL)", "every agent's input and output"),
        _node("app", "runtime", "Streamlit app", "desk state · trace viewer · these diagrams"),
    ])
    lines += [
        "{rank=same; sched; edgar;}", "{rank=same; state; a1;}", "{rank=same; trace; b1;}", "{rank=same; app; pf;}",
        "{rank=same; a1; a2;}", "{rank=same; xc; critic;}",
        _edge("a1", "a2", "", style="invis"),
        _edge("alpaca", "facts", "daily bars, SPY"),
        _edge("edgar", "facts", "TTM figures"),
        _edge("finnhub", "facts", "earnings dates"),
        _edge("alpaca", "macro", "price, volume, clock"),
        _edge("fred", "macro", "macro series"),
        _edge("edgar", "macro", "filings"),
        _edge("finnhub", "macro", "company news"),
        _edge("facts", "packet", "computed blocks"),
        _edge("macro", "packet", "briefing"),
        _edge("packet", "a1", "same packet"),
        _edge("packet", "a2", "same packet"),
        _edge("a1", "xc", "verdict"),
        _edge("a2", "xc", "verdict"),
        _edge("xc", "critic", "split or agreed buy"),
        _edge("critic", "a1", f"≤{MAX_CHALLENGES_PER_ANALYST} challenges", style="dashed", constraint="false",
              tailport="n"),
        _edge("critic", "a2", f"≤{MAX_CHALLENGES_PER_ANALYST} challenges", style="dashed", constraint="false",
              tailport="n"),
        _edge("xc", "b1", "agreed buy"),
        _edge("b1", "b2", "flag"),
        _edge("b1", "tech", "pass"),
        _edge("b2", "tech", "pass"),
        _edge("tech", "pf", "enter"),
        _edge("pf", "paper", "orders", tailport="sw", headport="nw"),
        _edge("paper", "pf", "equity, positions", style="dotted", constraint="false", tailport="ne", headport="se"),
        _edge("sched", "macro", "starts each pass", style="dotted", color="#4338ca", constraint="false"),
        _edge("sched", "state", "reads · saves", style="dotted", color="#4338ca", dir="both"),
        _edge("xc", "trace", "every step logged", style="dotted", color="#6d28d9", ltail="cluster_desk",
              constraint="false"),
        _edge("state", "trace", "", style="invis"),
        _edge("trace", "app", "", style="dotted", color="#6d28d9"),
        _edge("state", "app", "", style="dotted", color="#6d28d9", constraint="false"),
        "}",
    ]
    return "\n".join(lines)


def flow_dot() -> str:
    """Every path a pass can take, top to bottom: events, API calls, agents,
    decision points and the loops (polling, the critic rounds, the retries).
    Orange dashed edges are deferrals: a model that couldn't be reached."""
    retry = _minutes(RETRY_EVERY)
    solo_h = int(BIAS_SOLO_AFTER.total_seconds() // 3600)
    a1_backup = backups("analyst_1")
    defer = {"style": "dashed", "color": DEFER_COLOR, "fontcolor": DEFER_COLOR}
    loop = {"color": LOOP_COLOR, "fontcolor": LOOP_COLOR, "penwidth": "1.6"}
    lines = ["digraph flow {", *HEADER, 'graph [rankdir=TB, nodesep="0.3", ranksep="0.4", splines=true];']
    lines += _cluster("sched", f"Scheduler · a systemd service, checking every {POLL_SECONDS} s", [
        _event("start", "Service\\nstarts", "start"),
        _node("clock", "api", "Read market clock", "Alpaca"),
        _gate("due", "What's\\ndue?"),
        _event("poll", f"wait\\n{POLL_SECONDS} s"),
        _node("prep", "api", "Shared data, once per pass",
              f"watchlist ({len(DEFAULT_WATCHLIST)}) + stocks held", "FRED macro · SPY bars"),
    ])
    stock = [
        _node("gather", "api", "Gather context", "Alpaca · SEC EDGAR · Finnhub"),
        _node("compute", "rule", "Computed facts", "valuation · technicals · earnings"),
        _node("brief", "agent", "Macro briefing", model("macro"), "rejected twice → facts only"),
        _node("analysts", "agent", "Analyst 1 and analyst 2, in parallel",
              f"{model('analyst_1')} · {model('analyst_2')}",
              *([f"analyst 1 unreachable → {a1_backup[0]} answers"] if a1_backup else [])),
        _gate("answered", "Both\\nverdicts in?"),
        _gate("xc", "Cross-check"),
        "{rank=same; gather; compute; brief;}",
    ]
    stock += _cluster("critic", f"Critic loop · up to {MAX_ROUNDS} rounds", [
        _node("critic", "agent", "Critic review", model("critic"), f"≤{MAX_CHALLENGES_PER_ANALYST} challenges each · no vote"),
        _node("revise", "agent", "Both analysts re-vote", "accept or reject each challenge"),
        _gate("xc2", "Cross-check\\nagain"),
    ])
    stock += _cluster("bias", "Bias gate", [
        _node("b1", "agent", "Bias 1 check", model("bias_1"), "news-driven · stale news · trend-chasing"),
        _gate("g1", "Bias 1"),
        _node("b2", "agent", "Bias 2 check", model("bias_2"), "the same three checks"),
        _gate("g2", "Both\\nchecks"),
    ])
    stock += _cluster("timing", "Entry timing", [
        _node("tech", "agent", "Technical agent", model("technical"), "4h trend → 1h entry"),
        _gate("gt", "Entry"),
    ])
    stock += [
        _node("o_buy", "bought", "Buy", "agreed · both gates passed"),
        _node("o_wait", "outcome", "No buy today", "a chart reason to wait"),
        _node("o_veto", "outcome", "Buy vetoed", "by the bias gate"),
        _node("o_failed", "outcome", "Gate failed", "an unusable answer"),
        _node("o_hold", "outcome", "Agreed hold or avoid", "a held stock is sold on avoid"),
        _node("o_abort", "outcome", "Abort", "no agreement · a held stock is kept"),
    ]
    lines += _cluster("stock", f"For each stock · {WORKERS} at a time", stock, fill="#ffffff", style="rounded,dashed")
    lines += _cluster("retry", "Deferred and retried", [
        _node("resume", "rule", "Resume each deferred stock", "at the step that failed, with the",
              "morning's context · dropped at", "the next session (analyzed afresh)"),
        _event("deferred", "Deferred"),
    ])
    lines += _cluster("recheck", f"Midday recheck · {_et(RECHECK_TIME_ET)}, once a session", [
        _node("rc_data", "api", "Fresh bars", "stocks the technical agent",
              "held back that morning (not held)"),
        _node("rc_tech", "agent", "Technical agent again", model("technical"), "no analyst, critic or bias call"),
        _gate("rc_gate", "Enter\\nnow?"),
    ])
    lines += _cluster("portfolio", "Portfolio · once per pass, after every stock", [
        _node("pf", "rule", "Plan orders", f"sell: held {HOLDING_SESSIONS} sessions, or agreed avoid",
              f"buy: {POSITION_FRACTION:.0%} of equity each · max {MAX_POSITIONS} · no margin"),
        _node("orders", "api", "Paper orders", "Alpaca · client_order_id, never twice",
              "fill at the open, or at once in market hours"),
        _node("o_pos", "bought", "Positions opened or closed"),
    ])
    lines += [
        _node("save", "store", "State file · trace", "decisions · positions · pending · next pass"),
        _event("end", "Pass\\ndone", "end"),
        _edge("start", "clock"),
        _edge("clock", "due"),
        _edge("due", "poll", "nothing"),
        _edge("poll", "clock", "", constraint="false"),
        _edge("due", "prep", f"decision cycle\\n{_et(DECISION_TIME_ET)}, once a session"),
        _edge("due", "resume", f"retry pass: stocks deferred\\nevery {retry} min until {_et(RETRY_UNTIL_ET)}", **defer),
        _edge("due", "rc_data", f"midday recheck\\n{_et(RECHECK_TIME_ET)}, once a session"),
        _edge("rc_data", "rc_tech"),
        _edge("rc_tech", "rc_gate"),
        _edge("rc_gate", "o_buy", "enter: the pick stands"),
        _edge("rc_gate", "o_wait", "wait · failed · unreachable", **defer),
        _edge("prep", "gather"),
        _edge("gather", "compute"),
        _edge("compute", "brief"),
        _edge("brief", "analysts", "context packet"),
        _edge("resume", "analysts", "re-enter at the failed step", **defer),
        _edge("analysts", "answered"),
        _edge("answered", "xc", "yes"),
        _edge("answered", "deferred", "an analyst unreachable", **defer),
        _edge("xc", "o_hold", "agree: hold / avoid"),
        _edge("xc", "o_abort", "buy vs avoid · hold vs avoid\\nan unusable verdict"),
        _edge("xc", "critic", "buy vs hold · agreed buy"),
        _edge("critic", "revise", "challenges"),
        _edge("revise", "xc2", "revised verdicts"),
        _edge("xc2", "critic", "still split:\\nanother round", constraint="false", **loop),
        _edge("xc2", "o_hold", "agree: hold / avoid"),
        _edge("xc2", "o_abort", f"split after round {MAX_ROUNDS}\\navoid vs buy or hold · unusable"),
        _edge("xc2", "b1", "agree: buy"),
        _edge("revise", "deferred", "critic or re-vote unreachable", **defer),
        _edge("b1", "g1"),
        _edge("g1", "tech", "pass: clears the gate"),
        _edge("g1", "b2", "flag, or no\\nusable answer"),
        _edge("b2", "g2"),
        _edge("g2", "tech", "bias 1 flags,\\nbias 2 passes"),
        _edge("g2", "o_veto", f"both flag, or bias 1\\nalone after {solo_h} h"),
        _edge("g2", "o_failed", "an answer\\nunusable"),
        _edge("g2", "deferred", "an agent unreachable", **defer),
        _edge("tech", "gt"),
        _edge("gt", "o_buy", "enter"),
        _edge("gt", "o_wait", "wait"),
        _edge("gt", "o_failed", "unusable"),
        _edge("gt", "deferred", "unreachable", **defer),
        *[_edge(o, "pf") for o in ("o_buy", "o_wait", "o_veto", "o_failed", "o_hold", "o_abort")],
        _edge("pf", "orders", "sells first, then buys"),
        _edge("pf", "save", "nothing to trade"),
        _edge("orders", "o_pos"),
        _edge("o_pos", "save"),
        _edge("deferred", "save", "kept in pending", **defer),
        _edge("save", "end"),
        "}",
    ]
    return "\n".join(lines)


def agent_rows() -> list[dict[str, str]]:
    """One row per LLM role: model, backups, what it gets and what it returns."""
    jobs = {
        "macro": ("the stock's data from the four APIs", "a neutral briefing (no opinions)"),
        "analyst_1": ("the context packet", "a verdict: buy / hold / avoid, thesis, drivers, risks, cited evidence"),
        "analyst_2": ("the same packet, independently", "a verdict, in the same shape"),
        "critic": ("the packet and both verdicts", f"an assessment and ≤{MAX_CHALLENGES_PER_ANALYST} challenges per "
                                                   "analyst, never its own recommendation"),
        "bias_1": ("the packet and the agreed buy", "news sentiment, three skew checks, pass or flag"),
        "bias_2": ("the same, when bias 1 flags", "the same shape"),
        "technical": ("the packet and 4h / 1h bars (asked again at midday for a wait)", "enter or wait, the 4h trend, support and resistance"),
    }
    return [{"role": role, "model": model(role), "backup": ", ".join(backups(role)) or "none",
             "receives": jobs[role][0], "returns": jobs[role][1]} for role in AGENT_MODELS]


# What travels along the architecture's arrows, in plain words.
HANDOFFS = [
    ("Data sources → context", "price and bars (daily, 4-hour, 1-hour), IEX relative volume, the market clock, FRED rates, "
                               "spreads and CPI, SEC filings and fundamentals, company news, earnings dates"),
    ("Context → both analysts", "one context packet: the neutral briefing, the full source facts (authoritative), the "
                                "computed valuation, technicals and earnings blocks, the desk's limitations, data gaps"),
    ("Analyst → cross-check", "a verdict: buy, hold or avoid; confidence (information only); thesis; drivers; risks; "
                              "evidence, each citing a path into the facts and checked against it; data concerns"),
    ("Cross-check → critic", "the trigger (a buy/hold split or an agreed buy) and both verdicts"),
    ("Critic ↔ analysts", "challenges with cited facts; back come revised verdicts that accept or reject each one"),
    ("Agreed buy → bias gate", "the final verdicts and the packet; back come news sentiment, three checks and pass/flag"),
    ("Bias gate → technical agent", "the passed buy, the packet and the 4-hour and 1-hour bars; back come enter/wait "
                                    "and the chart levels behind it"),
    ("Decision → portfolio → Alpaca", "outcome, reasons, price and gate results; out go market orders with ids built "
                                      "from the session, stock and side, so a retry never orders twice"),
    ("Everything → trace and state", "every agent's input and output, timestamped; deferred runs, positions, "
                                     "decisions and each pass's summary"),
]


def outcome_rows() -> list[dict[str, str]]:
    """Where a stock can end up in one pass, and what the portfolio does."""
    retry, solo_h = _minutes(RETRY_EVERY), int(BIAS_SOLO_AFTER.total_seconds() // 3600)
    return [
        {"outcome": "Buy", "when": "the analysts agree on buy (after the critic), the bias gate passes and the "
                                   "technical agent says enter",
         "portfolio": f"buys {POSITION_FRACTION:.0%} of equity in whole shares (max {MAX_POSITIONS} positions, "
                      f"no margin); sold after {HOLDING_SESSIONS} sessions or on an agreed avoid. Already held: "
                      "the holding period restarts"},
        {"outcome": "No buy today", "when": "the technical agent names a chart reason to wait",
         "portfolio": f"nothing yet: the technical agent is asked again at {_et(RECHECK_TIME_ET)} and buys on an "
                      "enter; otherwise the stock is analyzed afresh at the next decision cycle"},
        {"outcome": "Buy vetoed", "when": f"both bias agents flag the buy, or bias 1's flag after {solo_h} h "
                                          "without an answer from bias 2",
         "portfolio": "nothing"},
        {"outcome": "Gate failed", "when": "a bias or technical answer came back unusable",
         "portfolio": "nothing (a check that couldn't be done is not a pass)"},
        {"outcome": "Agreed hold or avoid", "when": "the analysts agree, first time or after the critic",
         "portfolio": "a stock the desk holds is sold on an agreed avoid; otherwise nothing"},
        {"outcome": "Abort", "when": f"buy vs avoid, hold vs avoid, an unusable verdict, or still split after "
                                     f"{MAX_ROUNDS} critic rounds",
         "portfolio": "nothing; a held stock is kept until its holding period ends"},
        {"outcome": "Deferred", "when": "a model couldn't be reached (Build outage), at any step",
         "portfolio": f"nothing yet: retried every {retry} min from the failed step until "
                      f"{_et(RETRY_UNTIL_ET)}, dropped at the next session"},
    ]


# --- legends, as HTML chips for st.markdown(..., unsafe_allow_html=True) ---

def _swatch(kind: str, *, radius: str = "4px", border: str = "2px solid", extra: str = "") -> str:
    fill, color, _ = KINDS[kind]
    return (f'<span style="display:inline-block;width:14px;height:14px;background:{fill};'
            f'border:{border} {color};border-radius:{radius};{extra}"></span>')


def _line(color: str, style: str) -> str:
    return f'<span style="display:inline-block;width:28px;border-top:2px {style} {color}"></span>'


def _legend(items: list[tuple[str, str]]) -> str:
    chips = "".join(f'<span style="display:inline-flex;align-items:center;gap:6px">{swatch}{escape(text)}</span>'
                    for swatch, text in items)
    return f'<div style="display:flex;flex-wrap:wrap;gap:6px 18px;font-size:0.85rem;margin:4px 0 10px">{chips}</div>'


def _kind(kind: str, **kw: str) -> tuple[str, str]:
    return _swatch(kind, **kw), KINDS[kind][2]


def architecture_legend() -> str:
    return _legend([_kind("api"), _kind("agent"), _kind("rule"), _kind("store", radius="3px 3px 7px 7px"),
                    _kind("runtime"), (_line("#64748b", "solid"), "data passed on"),
                    (_line("#64748b", "dashed"), "critic's challenges back"),
                    (_line("#6d28d9", "dotted"), "control, logging, reads")])


def flow_legend() -> str:
    start = ('<span style="display:inline-block;width:14px;height:14px;border-radius:50%;background:#dcfce7;'
             'border:2px solid #15803d"></span>')
    end = ('<span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:#f8fafc;'
           'border:3.5px solid #334155"></span>')
    return _legend([(start, "Start event"), _kind("event", radius="50%", border="3px double"),
                    (end, "End event"), _kind("gateway", radius="1px", extra="transform:rotate(45deg) scale(0.8)"),
                    _kind("api"), _kind("agent"), _kind("rule"), _kind("store", radius="3px 3px 7px 7px"),
                    _kind("outcome"), _kind("bought"), (_line(LOOP_COLOR, "solid"), "critic loop: another round"),
                    (_line(DEFER_COLOR, "dashed"), "deferral: a model couldn't be reached")])
