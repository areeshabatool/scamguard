"""Experiment E4: does Scam Guard treat English and Roman Urdu the same?

Dataset (config.E4_DATASET, synthetic JSONL), items paired across languages by pair_id:

    {"id": ..., "pair_id": ..., "lang": "en" | "roman_urdu", "label": "scam" | "benign",
     "text": "..."}

Every pair_id must have exactly one "en" and one "roman_urdu" item with the same label.

B3 (config.MODEL) runs on every item with one calibration (config.CALIBRATION_PATH), so
both languages face the same gate threshold. Flagged = scam or suspicious.
  - Paired false-positive difference over benign pairs:
        mean(flagged(roman_urdu) - flagged(en)), 95% bootstrap CI resampling pairs.
    Positive means more false alarms on Roman Urdu.
  - Recall and FPR per language with 95% bootstrap CIs (resampling that language's items),
    plus the abstain rate per language.

Outputs: metrics only -> results/e4.json (or e4_dryrun.json); per-item labels by id ->
results/raw/e4_cases.jsonl (git-ignored).

    python -m eval.run_e4 --dry-run   # offline: data/samples/pairs.jsonl + FakeLLM
    python -m eval.run_e4             # live: needs ANTHROPIC_API_KEY, costs tokens
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from eval import metrics
from eval.common import (
    BOOTSTRAP_SEED,
    RESULTS_DIR,
    SAMPLES_DIR,
    LLMFactory,
    display_path,
    dry_fake,
    flagged,
    git_commit,
    live_factory,
    load_calibration,
    read_jsonl,
    require_fields,
    write_raw,
    write_report,
)
from eval.systems import B3
from scamguard import config, gate
from scamguard.llm import default_model_name

FIELDS = ("id", "pair_id", "lang", "label", "text")
LANGS = ("en", "roman_urdu")
DRY_DATASET = SAMPLES_DIR / "pairs.jsonl"


def build_pairs(rows: list[dict], path: Path) -> dict:
    """pair_id -> {"en": row, "roman_urdu": row}. Raises on malformed pairs (ids only)."""
    pairs: dict = {}
    problems: list[str] = []
    for row in rows:
        if row["lang"] not in LANGS:
            problems.append(f"id {row['id']}: lang must be one of {LANGS}")
            continue
        slot = pairs.setdefault(row["pair_id"], {})
        if row["lang"] in slot:
            problems.append(f"pair {row['pair_id']}: two '{row['lang']}' items")
        slot[row["lang"]] = row
    for pid, slot in pairs.items():
        if set(slot) != set(LANGS):
            problems.append(f"pair {pid}: needs one 'en' and one 'roman_urdu' item")
        elif slot["en"]["label"] != slot["roman_urdu"]["label"]:
            problems.append(f"pair {pid}: labels differ between languages")
    if problems:
        shown = "; ".join(problems[:5]) + (" ..." if len(problems) > 5 else "")
        raise ValueError(f"{path.name}: {shown}")
    return pairs


def _rate_with_ci(values: list[float], n_resamples: int) -> dict:
    lo, hi = metrics.bootstrap_mean_ci(values, seed=BOOTSTRAP_SEED, n_resamples=n_resamples)
    return {"value": metrics.mean_or_nan(values), "ci": [lo, hi], "n": len(values)}


def analyze_pairs(pairs: dict, preds: dict, *, n_resamples: int) -> dict:
    """preds maps item id -> predicted label."""

    def flag(row: dict) -> float:
        return 1.0 if flagged(preds[row["id"]]) else 0.0

    benign = [p for p in pairs.values() if p["en"]["label"] == "benign"]
    scam = [p for p in pairs.values() if p["en"]["label"] == "scam"]

    diffs = [flag(p["roman_urdu"]) - flag(p["en"]) for p in benign]
    by_lang = {}
    for lang in LANGS:
        items = [p[lang] for p in pairs.values()]
        by_lang[lang] = {
            "recall": _rate_with_ci([flag(p[lang]) for p in scam], n_resamples),
            "fpr": _rate_with_ci([flag(p[lang]) for p in benign], n_resamples),
            "abstain_rate": metrics.mean_or_nan(
                [1.0 if preds[r["id"]] == "unsure" else 0.0 for r in items]
            ),
        }
    return {
        "paired_fpr_difference": {
            "definition": "mean over benign pairs of flagged(roman_urdu) - flagged(en)",
            **_rate_with_ci(diffs, n_resamples),
        },
        "by_language": by_lang,
    }


def run(
    *,
    dataset_path: Path,
    model: str,
    factory: LLMFactory,
    calibration: Sequence[float],
    n_resamples: int,
    dry_run: bool = False,
) -> tuple[dict, list[dict]]:
    rows = read_jsonl(dataset_path)
    require_fields(rows, FIELDS, dataset_path)
    pairs = build_pairs(rows, dataset_path)

    system = B3(factory, calibration_scores=calibration)  # one threshold for both languages
    preds = {row["id"]: system.predict(row["text"]).label for row in rows}
    threshold = gate.conformal_threshold(calibration, config.ALPHA)

    report = {
        "experiment": "E4",
        "commit": git_commit(),
        "dry_run": dry_run,
        "model": model,
        "seeds": {"bootstrap": BOOTSTRAP_SEED},
        "dataset": {
            "path": display_path(dataset_path),
            "n_items": len(rows),
            "n_pairs": len(pairs),
            "n_benign_pairs": sum(p["en"]["label"] == "benign" for p in pairs.values()),
            "n_scam_pairs": sum(p["en"]["label"] == "scam" for p in pairs.values()),
        },
        "gate": {
            "alpha": config.ALPHA,
            "n_calibration": len(calibration),
            "threshold": threshold,
            "shared_across_languages": True,
        },
        **analyze_pairs(pairs, preds, n_resamples=n_resamples),
    }
    raw = [
        {
            "id": r["id"],
            "pair_id": r["pair_id"],
            "lang": r["lang"],
            "label": r["label"],
            "pred": preds[r["id"]],
        }
        for r in rows
    ]
    return report, raw


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="eval.run_e4", description="Run experiment E4.")
    parser.add_argument("--dry-run", action="store_true", help="use a scripted FakeLLM, no API")
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--resamples", type=int, default=1000)
    args = parser.parse_args(argv)

    dataset = args.dataset or (DRY_DATASET if args.dry_run else config.E4_DATASET)
    report, raw = run(
        dataset_path=dataset,
        model="fake-model" if args.dry_run else default_model_name(),
        factory=dry_fake if args.dry_run else live_factory(default_model_name()),
        calibration=load_calibration(),
        n_resamples=args.resamples,
        dry_run=args.dry_run,
    )
    out = args.out or RESULTS_DIR / ("e4_dryrun.json" if args.dry_run else "e4.json")
    write_report(report, out)
    raw_out = write_raw(raw, "e4_dryrun_cases.jsonl" if args.dry_run else "e4_cases.jsonl")

    d = report["paired_fpr_difference"]
    print(f"E4 ({'dry-run' if args.dry_run else 'live'}) -> {out} (cases: {raw_out})")
    print(f"  paired FPR diff (ru - en) = {d['value']:+.3f} [{d['ci'][0]:+.3f}, {d['ci'][1]:+.3f}]")
    for lang, block in report["by_language"].items():
        r = block["recall"]
        print(f"  {lang:10s} recall={r['value']:.2f} [{r['ci'][0]:.2f}, {r['ci'][1]:.2f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
