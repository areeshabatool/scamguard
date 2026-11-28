"""Produce the gate's calibration file from scam scores.

The conformal gate (scamguard.gate) releases "benign" only when a message's score is
strictly below the k-th smallest *calibration* score, where the calibration set is the
scores the system assigns to known scams. This script runs B3 (no calibration) over a
scam dataset and writes those scores to config.CALIBRATION_PATH, so the deployed gate and
the CLI can release benign at miss-rate budget config.ALPHA.

    python -m eval.calibrate --dry-run                 # offline smoke (FakeLLM)
    python -m eval.calibrate --dataset <scams.jsonl>   # live: needs ANTHROPIC_API_KEY

Writes {"scores": [...], "model": ..., "n": ..., "commit": ...}. Scores are numbers only,
never message text.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval.common import LLMFactory, dry_fake, git_commit, live_factory, read_jsonl, require_fields
from scamguard import config, pipeline
from scamguard.llm import default_model_name

FIELDS = ("id", "text", "label")


def scam_scores(rows: list[dict], factory: LLMFactory) -> list[float]:
    """Pre-gate score for each scam row (B3 with no calibration)."""
    scores: list[float] = []
    for row in rows:
        if row.get("label") != "scam":
            continue
        result = pipeline.analyze(row["text"], llm=factory(), calibration_scores=())
        scores.append(result.score)
    return scores


def run(*, dataset_path: Path, model: str, factory: LLMFactory) -> dict:
    rows = read_jsonl(dataset_path)
    require_fields(rows, FIELDS, dataset_path)
    scores = scam_scores(rows, factory)
    return {"scores": scores, "model": model, "n": len(scores), "commit": git_commit()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="eval.calibrate", description="Build calibration.json.")
    parser.add_argument("--dry-run", action="store_true", help="use a scripted FakeLLM, no API")
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=config.CALIBRATION_PATH)
    args = parser.parse_args(argv)

    dataset = args.dataset or (
        config.DATA_DIR / "samples" / "eval.jsonl" if args.dry_run else config.E3_DATASET
    )
    report = run(
        dataset_path=dataset,
        model="fake-model" if args.dry_run else default_model_name(),
        factory=dry_fake if args.dry_run else live_factory(default_model_name()),
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"calibration ({'dry-run' if args.dry_run else 'live'}) -> {args.out}  (n={report['n']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
