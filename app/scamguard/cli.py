"""Command-line entry point.

Usage: python -m scamguard.cli analyze 'text'

Runs the full pipeline on the given text and prints the decision and short advice.
Pass --json for the full CaseResult (decision, score, cues, advice and per-step trace).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from scamguard import config
from scamguard.llm import LLM
from scamguard.pipeline import analyze

_NO_KEY_MESSAGE = (
    "error: no Anthropic credentials found. Copy .env.example to .env and set "
    "ANTHROPIC_API_KEY (or export it), then retry. Tests and the eval --dry-run need no key."
)


def _load_calibration() -> list[float]:
    """Load the gate's calibration scores from config.CALIBRATION_PATH if it exists."""
    try:
        data = json.loads(config.CALIBRATION_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    scores = data.get("scores", data) if isinstance(data, dict) else data
    return [float(s) for s in scores] if isinstance(scores, list) else []


def _build_llm() -> LLM:
    from scamguard.llm import AnthropicLLM

    return AnthropicLLM()


def main(argv: list[str] | None = None, *, llm: LLM | None = None) -> int:
    parser = argparse.ArgumentParser(prog="scamguard")
    sub = parser.add_subparsers(dest="command", required=True)
    analyze_cmd = sub.add_parser("analyze", help="classify a forwarded message")
    analyze_cmd.add_argument("text", help="the message text to analyze")
    analyze_cmd.add_argument("--json", action="store_true", help="print the full CaseResult")
    args = parser.parse_args(argv)

    if args.command == "analyze":
        if llm is None and not (
            os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")
        ):
            print(_NO_KEY_MESSAGE, file=sys.stderr)
            return 1
        client = llm if llm is not None else _build_llm()
        result = analyze(args.text, llm=client, calibration_scores=_load_calibration())
        if args.json:
            print(result.model_dump_json(indent=2))
        else:
            print(f"Decision: {result.decision}  (score {result.score:.2f})")
            if result.cues:
                print("Cues: " + ", ".join(result.cues))
            print(f"Advice: {result.advice}")
        return 0

    return 2  # pragma: no cover - argparse requires a subcommand


if __name__ == "__main__":
    raise SystemExit(main())
