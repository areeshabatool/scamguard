"""Experiment E3: does the conformal gate keep the miss rate within alpha?

Dataset (config.E3_DATASET, synthetic JSONL):

    {"id": ..., "text": "...", "label": "scam" | "benign", "scam_type": "<type>"}  # scams only

1. Run B3 once over every item with no calibration and cache each item's pre-gate score
   and verdict (results/raw/e3_scores.jsonl: ids, labels and numbers only). The cache is
   reused while the dataset bytes and model are unchanged; --refresh forces a re-run.
   With no calibration the gate never releases benign, so the pre-gate verdict is read off
   the result: scam/suspicious pass through; a plain "unsure" means the guards said benign
   and the gate withheld it; model/schema failures and unreadable input stay "unsure".
2. Standard run, repeated over config.E3_N_SPLITS random splits: a random
   config.E3_CALIBRATION_FRACTION of the scams is the calibration set; the test set is the
   other scams plus every non-scam. The gate (gate.release_benign, alpha = config.ALPHA) is
   replayed on the cached scores. Per split:
     miss rate    = test scams released as benign / test scams
     abstain rate = test items ending "unsure" / test items
3. Held-out run: the same, but scams of config.E3_HELDOUT_SCAM_TYPE are never calibrated
   on; all of them are tested. Its miss rate is measured on that type alone.

Output: mean and spread across splits (std, 2.5/97.5 percentiles, min, max), and whether
the mean held-out miss rate exceeds alpha -> results/e3.json (or e3_dryrun.json).

    python -m eval.run_e3 --dry-run   # offline: data/samples/eval.jsonl + FakeLLM
    python -m eval.run_e3             # live: needs ANTHROPIC_API_KEY, costs tokens once
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from eval import common, metrics
from eval.common import (
    RESULTS_DIR,
    SAMPLES_DIR,
    LLMFactory,
    display_path,
    dry_fake,
    git_commit,
    live_factory,
    read_jsonl,
    require_fields,
    write_report,
)
from scamguard import config, gate, pipeline
from scamguard.llm import default_model_name
from scamguard.schemas import CaseResult

FIELDS = ("id", "text", "label")
DRY_DATASET = SAMPLES_DIR / "eval.jsonl"


# ---------------------------------------------------------------------------
# Step 1: score every item once, with a cache
# ---------------------------------------------------------------------------


def pre_gate(result: CaseResult) -> str:
    """Pre-gate verdict of a B3 result produced with no calibration (see module docstring)."""
    if result.decision in ("scam", "suspicious"):
        return result.decision
    clean = not result.reason and all(s.status != "error" for s in result.trace)
    return "benign" if result.decision == "unsure" and clean else "unsure"


def score_items(rows: list[dict], factory: LLMFactory) -> list[dict]:
    records = []
    for row in rows:
        result = pipeline.analyze(row["text"], llm=factory(), calibration_scores=())
        records.append(
            {
                "id": row["id"],
                "label": row["label"],
                "scam_type": row.get("scam_type"),
                "verdict": pre_gate(result),
                "score": result.score,
            }
        )
    return records


def _cache_key(dataset_path: Path, model: str) -> str:
    digest = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    return f"{digest}:{model}"


def cached_scores(
    rows: list[dict],
    *,
    dataset_path: Path,
    model: str,
    factory: LLMFactory,
    cache_path: Path,
    refresh: bool,
) -> tuple[list[dict], bool]:
    """Return (records, cache_hit). The cache holds ids, labels and scores, never text."""
    key = _cache_key(dataset_path, model)
    if not refresh and cache_path.exists():
        lines = cache_path.read_text(encoding="utf-8").splitlines()
        if lines and json.loads(lines[0]).get("key") == key:
            return [json.loads(line) for line in lines[1:] if line.strip()], True

    records = score_items(rows, factory)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    body = [json.dumps({"key": key})] + [json.dumps(r) for r in records]
    cache_path.write_text("\n".join(body) + "\n", encoding="utf-8")
    return records, False


# ---------------------------------------------------------------------------
# Steps 2-3: replay the gate over random splits
# ---------------------------------------------------------------------------


def gated_label(record: dict, calibration: list[float], alpha: float) -> str:
    """The label the deployed gate would release for this record."""
    if record["verdict"] != "benign":
        return record["verdict"]
    return "benign" if gate.release_benign(record["score"], calibration, alpha) else "unsure"


def _miss_rate(labels: list[str]) -> float:
    return metrics.mean_or_nan([1.0 if lab == "benign" else 0.0 for lab in labels])


def evaluate_split(
    calibration: list[dict], test: list[dict], alpha: float, *, miss_on: list[dict] | None = None
) -> dict:
    """Calibrate on `calibration` scams; measure on `test`. miss_on defaults to test scams."""
    cal_scores = [r["score"] for r in calibration]
    labels = {id(r): gated_label(r, cal_scores, alpha) for r in test}
    miss_pool = miss_on if miss_on is not None else [r for r in test if r["label"] == "scam"]
    return {
        "miss_rate": _miss_rate([labels[id(r)] for r in miss_pool]),
        "abstain_rate": metrics.mean_or_nan(
            [1.0 if v == "unsure" else 0.0 for v in labels.values()]
        ),
        "threshold": gate.conformal_threshold(cal_scores, alpha),
    }


def _n_calibration(n: int, fraction: float) -> int:
    return min(max(round(fraction * n), 1), n - 1) if n > 1 else n


def run_splits(
    records: list[dict],
    *,
    alpha: float,
    n_splits: int,
    fraction: float,
    seed: int,
    heldout: str | None = None,
) -> list[dict]:
    """One result dict per random split. With `heldout`, that scam type is never calibrated on."""
    scams = [r for r in records if r["label"] == "scam"]
    others = [r for r in records if r["label"] != "scam"]
    held = [r for r in scams if heldout is not None and r.get("scam_type") == heldout]
    pool = [r for r in scams if not (heldout is not None and r.get("scam_type") == heldout)]
    n_cal = _n_calibration(len(pool), fraction)

    rng = random.Random(seed)
    results = []
    for _ in range(n_splits):
        shuffled = pool[:]
        rng.shuffle(shuffled)
        cal, rest = shuffled[:n_cal], shuffled[n_cal:]
        if heldout is None:
            results.append(evaluate_split(cal, rest + others, alpha))
        else:
            split = evaluate_split(cal, rest + held + others, alpha, miss_on=held)
            split["miss_rate_other"] = evaluate_split(cal, rest, alpha)["miss_rate"]
            results.append(split)
    return results


def summarize_splits(splits: list[dict], keys: tuple[str, ...]) -> dict:
    out: dict = {}
    for key in keys:
        values = [s[key] for s in splits if s[key] is not None and s[key] == s[key]]
        out[key] = metrics.spread(values)
    out["splits_without_threshold"] = sum(s["threshold"] is None for s in splits)
    return out


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run(
    *,
    dataset_path: Path,
    model: str,
    factory: LLMFactory,
    cache_path: Path,
    refresh: bool = False,
    alpha: float | None = None,
    n_splits: int | None = None,
    fraction: float | None = None,
    seed: int | None = None,
    heldout: str | None = None,
    dry_run: bool = False,
) -> dict:
    alpha = config.ALPHA if alpha is None else alpha
    n_splits = config.E3_N_SPLITS if n_splits is None else n_splits
    fraction = config.E3_CALIBRATION_FRACTION if fraction is None else fraction
    seed = config.E3_SPLIT_SEED if seed is None else seed
    heldout = config.E3_HELDOUT_SCAM_TYPE if heldout is None else heldout

    rows = read_jsonl(dataset_path)
    require_fields(rows, FIELDS, dataset_path)
    scams = [r for r in rows if r["label"] == "scam"]
    require_fields(scams, ("scam_type",), dataset_path)
    types = sorted({r["scam_type"] for r in scams})
    if heldout not in types:
        raise ValueError(
            f"held-out scam type {heldout!r} (config.E3_HELDOUT_SCAM_TYPE) is not in the "
            f"dataset; scam types present: {types}"
        )

    records, cache_hit = cached_scores(
        rows,
        dataset_path=dataset_path,
        model=model,
        factory=factory,
        cache_path=cache_path,
        refresh=refresh,
    )

    split_args = dict(alpha=alpha, n_splits=n_splits, fraction=fraction, seed=seed)
    standard = run_splits(records, **split_args)
    held = run_splits(records, heldout=heldout, **split_args)

    standard_summary = summarize_splits(standard, ("miss_rate", "abstain_rate", "threshold"))
    held_summary = summarize_splits(
        held, ("miss_rate", "miss_rate_other", "abstain_rate", "threshold")
    )
    held_miss = [s["miss_rate"] for s in held if s["miss_rate"] == s["miss_rate"]]
    held_summary.update(
        {
            "scam_type": heldout,
            "n_heldout": sum(r.get("scam_type") == heldout for r in records),
            "exceeds_alpha": held_summary["miss_rate"]["mean"] > alpha,
            "share_of_splits_exceeding_alpha": metrics.mean_or_nan(
                [1.0 if m > alpha else 0.0 for m in held_miss]
            ),
        }
    )
    standard_summary["exceeds_alpha"] = standard_summary["miss_rate"]["mean"] > alpha

    n_scam = len(scams)
    return {
        "experiment": "E3",
        "commit": git_commit(),
        "dry_run": dry_run,
        "model": model,
        "seeds": {"splits": seed},
        "settings": {"alpha": alpha, "n_splits": n_splits, "calibration_fraction": fraction},
        "dataset": {
            "path": display_path(dataset_path),
            "n": len(rows),
            "n_scam": n_scam,
            "n_other": len(rows) - n_scam,
            "scam_types": types,
        },
        "scores_cache": {"path": display_path(cache_path), "hit": cache_hit},
        "pre_gate_verdicts": {
            v: sum(r["verdict"] == v for r in records)
            for v in ("scam", "suspicious", "benign", "unsure")
        },
        "standard": standard_summary,
        "heldout": held_summary,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="eval.run_e3", description="Run experiment E3.")
    parser.add_argument("--dry-run", action="store_true", help="use a scripted FakeLLM, no API")
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--refresh", action="store_true", help="ignore the cached scores")
    parser.add_argument("--splits", type=int, default=None)
    args = parser.parse_args(argv)

    dataset = args.dataset or (DRY_DATASET if args.dry_run else config.E3_DATASET)
    cache = common.RAW_DIR / ("e3_dryrun_scores.jsonl" if args.dry_run else "e3_scores.jsonl")
    report = run(
        dataset_path=dataset,
        model="fake-model" if args.dry_run else default_model_name(),
        factory=dry_fake if args.dry_run else live_factory(default_model_name()),
        cache_path=cache,
        refresh=args.refresh,
        n_splits=args.splits,
        dry_run=args.dry_run,
    )
    out = args.out or RESULTS_DIR / ("e3_dryrun.json" if args.dry_run else "e3.json")
    write_report(report, out)

    alpha = report["settings"]["alpha"]
    s, h = report["standard"], report["heldout"]
    print(f"E3 ({'dry-run' if args.dry_run else 'live'}) -> {out}  alpha={alpha}")
    print(
        f"  standard  miss={s['miss_rate']['mean']:.3f} (sd {s['miss_rate']['std']:.3f}) "
        f"abstain={s['abstain_rate']['mean']:.3f}"
    )
    print(
        f"  held-out '{h['scam_type']}' miss={h['miss_rate']['mean']:.3f} "
        f"(sd {h['miss_rate']['std']:.3f}) exceeds alpha: {h['exceeds_alpha']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
