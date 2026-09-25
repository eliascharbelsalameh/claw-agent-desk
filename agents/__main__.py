"""Run the macro/context agent against live APIs.

    .venv/Scripts/python -m agents AAPL MSFT NVDA
    .venv/Scripts/python -m agents AAPL --no-llm      # data only, no Build credits

Writes the trace to logs/trace-<UTC date>.jsonl and prints each symbol's
analyst-facing context block.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from data_layer import AlpacaClient, DiskCache, EdgarClient, FinnhubClient, FredClient, get_settings
from data_layer.llm_client import LlmClient

from .macro_agent import MacroContextAgent
from .trace import TraceLogger


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("symbols", nargs="+")
    parser.add_argument("--no-llm", action="store_true", help="skip the LLM briefing")
    parser.add_argument("--log-dir", default="logs")
    args = parser.parse_args()

    settings = get_settings()
    cache = DiskCache(settings.cache_dir)
    trace = TraceLogger(
        Path(args.log_dir) / f"trace-{datetime.now(timezone.utc):%Y%m%d}.jsonl"
    )
    agent = MacroContextAgent(
        alpaca=AlpacaClient(settings=settings, cache=cache),
        fred=FredClient(settings=settings, cache=cache),
        edgar=EdgarClient(settings=settings, cache=cache),
        finnhub=FinnhubClient(settings=settings, cache=cache),
        llm=None if args.no_llm else LlmClient(settings=settings),
        trace=trace,
    )
    for ctx in agent.run(args.symbols).values():
        print(ctx.to_prompt())
        print()
    print(f"trace: {trace.path}")


if __name__ == "__main__":
    main()
