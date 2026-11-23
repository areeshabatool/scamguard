"""Eval metric tests: recall, FPR, abstain, and reproducible bootstrap CIs."""

from eval import metrics


def test_recall_and_fpr_and_abstain() -> None:
    y_true = ["scam", "scam", "benign", "benign", "scam"]
    y_pred = ["scam", "benign", "scam", "benign", "unsure"]
    # positives: idx 0,1,4 -> flagged: scam, benign, unsure -> 1/3
    assert metrics.recall(y_true, y_pred) == 1 / 3
    # negatives: idx 2,3 -> flagged: scam, benign -> 1/2
    assert metrics.false_positive_rate(y_true, y_pred) == 1 / 2
    assert metrics.abstain_rate(y_pred) == 1 / 5


def test_suspicious_counts_as_flagged() -> None:
    assert metrics.recall(["scam"], ["suspicious"]) == 1.0
    assert metrics.false_positive_rate(["benign"], ["suspicious"]) == 1.0


def test_recall_nan_without_positives() -> None:
    value = metrics.recall(["benign", "benign"], ["benign", "scam"])
    assert value != value  # NaN


def test_bootstrap_ci_is_reproducible_and_brackets() -> None:
    y_true = ["scam"] * 10 + ["benign"] * 10
    y_pred = ["scam"] * 8 + ["benign"] * 2 + ["benign"] * 10  # recall 0.8
    ci1 = metrics.bootstrap_ci(y_true, y_pred, metrics.recall, seed=7, n_resamples=300)
    ci2 = metrics.bootstrap_ci(y_true, y_pred, metrics.recall, seed=7, n_resamples=300)
    assert ci1 == ci2  # same seed -> identical
    lo, hi = ci1
    assert 0.0 <= lo <= hi <= 1.0
    assert lo <= metrics.recall(y_true, y_pred) <= hi


def test_bootstrap_ci_changes_with_seed() -> None:
    y_true = ["scam"] * 10 + ["benign"] * 10
    y_pred = ["scam"] * 7 + ["benign"] * 3 + ["benign"] * 10
    a = metrics.bootstrap_ci(y_true, y_pred, metrics.recall, seed=1, n_resamples=300)
    b = metrics.bootstrap_ci(y_true, y_pred, metrics.recall, seed=2, n_resamples=300)
    assert a != b


def test_paired_difference() -> None:
    y_true = ["scam", "scam", "benign", "benign"]
    perfect = ["scam", "scam", "benign", "benign"]  # recall 1.0
    useless = ["benign", "benign", "benign", "benign"]  # recall 0.0
    result = metrics.paired_difference(
        y_true, perfect, useless, metrics.recall, seed=3, n_resamples=200
    )
    assert result["difference"] == 1.0
    assert result["ci_low"] <= result["difference"] <= result["ci_high"] or result["ci_low"] >= 0
