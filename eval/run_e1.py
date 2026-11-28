"""Experiment E1: compare B0-B3 on the synthetic eval set.

Reports recall, FPR, abstain rate, turns to detection, cost (tokens) and latency per
system, with bootstrap CIs, written to results/ as JSON tagged with the commit hash and
seeds. Use --dry-run to exercise the whole harness with a scripted FakeLLM (no API).

    python -m eval.run_e1            # real run (needs ANTHROPIC_API_KEY)
    python -m eval.run_e1 --dry-run  # offline, scripted
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean

from eval import metrics
from eval.common import (
    BOOTSTRAP_SEED,
    RESULTS_DIR,
    SAMPLES_DIR,
    display_path,
    dry_fake,
    git_commit,
    read_jsonl,
)
from eval.systems import B0, B1, B2, B3

DEFAULT_DATASET = SAMPLES_DIR / "eval.jsonl"
read_dataset = read_jsonl
_git_commit = git_commit
_dry_fake = dry_fake

__all__ = ["BOOTSTRAP_SEED", "DEFAULT_DATASET", "RESULTS_DIR", "main", "read_dataset", "run"]


def evaluate(system, dataset: list[dict], *, n_resamples: int) -> tuple[dict, list[str]]:
    y_true = [row["label"] for row in dataset]
    preds = [system.predict(row["text"]) for row in dataset]
    y_pred = [p.label for p in preds]

    rec_lo, rec_hi = metrics.bootstrap_ci(
        y_true, y_pred, metrics.recall, seed=BOOTSTRAP_SEED, n_resamples=n_resamples
    )
    fpr_lo, fpr_hi = metrics.bootstrap_ci(
        y_true, y_pred, metrics.false_positive_rate, seed=BOOTSTRAP_SEED, n_resamples=n_resamples
    )
    summary = {
        "recall": metrics.recall(y_true, y_pred),
        "recall_ci": [rec_lo, rec_hi],
        "fpr": metrics.false_positive_rate(y_true, y_pred),
        "fpr_ci": [fpr_lo, fpr_hi],
        "abstain_rate": metrics.abstain_rate(y_pred),
        "avg_turns": fmean(p.turns for p in preds),
        "avg_cost_tokens": fmean(p.cost_tokens for p in preds),
        "avg_latency_s": fmean(p.latency_s for p in preds),
    }
    return summary, y_pred


def run(*, dry_run: bool, dataset_path: Path, n_resamples: int) -> dict:
    dataset = read_dataset(dataset_path)
    y_true = [row["label"] for row in dataset]

    if dry_run:
        factory: object = _dry_fake
    else:
        from scamguard.llm import build_default_llm

        shared = build_default_llm()  # Claude by default, OpenAI if selected
        factory = lambda: shared  # noqa: E731 - share one client so the disk cache is reused

    systems = [B0(), B1(factory), B2(factory), B3(factory)]

    results: dict = {}
    preds_by: dict[str, list[str]] = {}
    for system in systems:
        summary, y_pred = evaluate(system, dataset, n_resamples=n_resamples)
        results[system.name] = summary
        preds_by[system.name] = y_pred

    paired = metrics.paired_difference(
        y_true,
        preds_by["B3-pipeline"],
        preds_by["B0-keyword"],
        metrics.recall,
        seed=BOOTSTRAP_SEED,
        n_resamples=n_resamples,
    )

    return {
        "experiment": "E1",
        "commit": _git_commit(),
        "dry_run": dry_run,
        "seeds": {"bootstrap": BOOTSTRAP_SEED},
        "dataset": {
            "path": display_path(dataset_path),
            "n": len(dataset),
            "n_scam": sum(t == "scam" for t in y_true),
            "n_benign": sum(t == "benign" for t in y_true),
        },
        "systems": results,
        "paired": {"B3_vs_B0_recall": paired},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="eval.run_e1", description="Run experiment E1.")
    parser.add_argument("--dry-run", action="store_true", help="use a scripted FakeLLM, no API")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--resamples", type=int, default=1000)
    args = parser.parse_args(argv)

    report = run(dry_run=args.dry_run, dataset_path=args.dataset, n_resamples=args.resamples)

    out = args.out
    if out is None:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out = RESULTS_DIR / ("e1_dryrun.json" if args.dry_run else "e1.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"E1 ({'dry-run' if args.dry_run else 'live'}) -> {out}")
    for name, s in report["systems"].items():
        print(
            f"  {name:20s} recall={s['recall']:.2f} fpr={s['fpr']:.2f} "
            f"abstain={s['abstain_rate']:.2f} turns={s['avg_turns']:.1f} "
            f"cost={s['avg_cost_tokens']:.0f}tok lat={s['avg_latency_s'] * 1000:.1f}ms"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
