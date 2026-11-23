"""Evaluation metrics: recall, false-positive rate, abstain rate, and bootstrap CIs.

Positive class = a true "scam" message. A prediction counts as flagged if it is "scam" or
"suspicious"; "unsure" is an abstention (not flagged, and reported separately). All
bootstrap functions take a fixed seed so results are reproducible.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from math import ceil, floor

Metric = Callable[[Sequence[str], Sequence[str]], float]

_FLAGGED = {"scam", "suspicious"}


def _is_scam(label: str) -> bool:
    return label == "scam"


def _flagged(pred: str) -> bool:
    return pred in _FLAGGED


def recall(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    """TP / P over true-scam messages. NaN if there are no positives in the sample."""
    positives = [p for t, p in zip(y_true, y_pred, strict=True) if _is_scam(t)]
    if not positives:
        return float("nan")
    return sum(_flagged(p) for p in positives) / len(positives)


def false_positive_rate(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    """FP / N over true-benign messages. NaN if there are no negatives in the sample."""
    negatives = [p for t, p in zip(y_true, y_pred, strict=True) if t == "benign"]
    if not negatives:
        return float("nan")
    return sum(_flagged(p) for p in negatives) / len(negatives)


def abstain_rate(y_pred: Sequence[str]) -> float:
    """Fraction of predictions that are 'unsure'."""
    if not y_pred:
        return float("nan")
    return sum(p == "unsure" for p in y_pred) / len(y_pred)


def _percentile(ordered: list[float], q: float) -> float:
    if not ordered:
        return float("nan")
    if len(ordered) == 1:
        return ordered[0]
    idx = q * (len(ordered) - 1)
    lo, hi = floor(idx), ceil(idx)
    return ordered[lo] * (1 - (idx - lo)) + ordered[hi] * (idx - lo)


def bootstrap_ci(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    metric: Metric,
    *,
    seed: int,
    n_resamples: int = 1000,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile bootstrap CI for a metric, reproducible under a fixed seed."""
    rng = random.Random(seed)
    n = len(y_true)
    stats: list[float] = []
    for _ in range(n_resamples):
        idx = [rng.randrange(n) for _ in range(n)]
        value = metric([y_true[i] for i in idx], [y_pred[i] for i in idx])
        if value == value:  # drop NaN resamples (no positives/negatives present)
            stats.append(value)
    stats.sort()
    return _percentile(stats, alpha / 2), _percentile(stats, 1 - alpha / 2)


def paired_difference(
    y_true: Sequence[str],
    y_pred_a: Sequence[str],
    y_pred_b: Sequence[str],
    metric: Metric,
    *,
    seed: int,
    n_resamples: int = 1000,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Paired bootstrap of metric(a) - metric(b): same resampled indices for both systems."""
    point = metric(y_true, y_pred_a) - metric(y_true, y_pred_b)
    rng = random.Random(seed)
    n = len(y_true)
    diffs: list[float] = []
    for _ in range(n_resamples):
        idx = [rng.randrange(n) for _ in range(n)]
        yt = [y_true[i] for i in idx]
        a = metric(yt, [y_pred_a[i] for i in idx])
        b = metric(yt, [y_pred_b[i] for i in idx])
        if a == a and b == b:
            diffs.append(a - b)
    diffs.sort()
    return {
        "difference": point,
        "ci_low": _percentile(diffs, alpha / 2),
        "ci_high": _percentile(diffs, 1 - alpha / 2),
    }
