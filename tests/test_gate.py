"""Gate tests: conformal threshold, strict benign release, and the k < 1 rule."""

from math import floor

from scamguard.gate import conformal_k, conformal_threshold, decide, release_benign, unsure
from scamguard.schemas import Assessment, Decision


def _assess(verdict: str, score: float) -> Assessment:
    return Assessment(
        final_score=score, verdict=verdict, model_verdict=verdict, band_verdict=verdict
    )


# ---------- conformal maths ----------


def test_conformal_k_formula() -> None:
    assert conformal_k(9, 0.1) == floor(0.1 * 10) == 1
    assert conformal_k(19, 0.1) == 2
    assert conformal_k(5, 0.1) == 0  # k < 1


def test_threshold_is_kth_smallest() -> None:
    scores = [0.5, 0.1, 0.9, 0.3, 0.7, 0.2, 0.8, 0.4, 0.6]  # n = 9, alpha 0.2 -> k = 2
    # sorted: 0.1 0.2 0.3 ... ; 2nd smallest = 0.2
    assert conformal_threshold(scores, 0.2) == 0.2


def test_threshold_none_when_k_below_one() -> None:
    assert conformal_threshold([0.1, 0.2, 0.3], 0.1) is None  # k = floor(0.4) = 0


# ---------- release rule ----------


def test_release_requires_strictly_below_threshold() -> None:
    scores = [0.2, 0.4, 0.6, 0.8]  # n = 4, alpha 0.5 -> k = floor(2.5) = 2 -> threshold 0.4
    assert conformal_threshold(scores, 0.5) == 0.4
    assert release_benign(0.39, scores, 0.5) is True
    assert release_benign(0.4, scores, 0.5) is False  # equal is not below
    assert release_benign(0.5, scores, 0.5) is False


def test_never_release_when_k_below_one() -> None:
    assert release_benign(0.0, [0.9, 0.9, 0.9], 0.1) is False


# ---------- decide() ----------


def test_decide_releases_benign_below_threshold() -> None:
    scores = [0.3, 0.4, 0.5, 0.6]  # alpha 0.5 -> k 2 -> threshold 0.4
    d = decide(_assess("benign", 0.2), scores, alpha=0.5)
    assert isinstance(d, Decision)
    assert d.verdict == "benign" and d.released_benign is True and d.threshold == 0.4


def test_decide_benign_not_below_becomes_unsure() -> None:
    scores = [0.3, 0.4, 0.5, 0.6]
    d = decide(_assess("benign", 0.45), scores, alpha=0.5)
    assert d.verdict == "unsure" and d.released_benign is False


def test_decide_benign_unsure_when_k_below_one() -> None:
    d = decide(_assess("benign", 0.0), [0.9, 0.9, 0.9], alpha=0.1)
    assert d.verdict == "unsure"


def test_decide_passes_through_non_benign() -> None:
    scores = [0.1, 0.2, 0.3, 0.4]
    assert decide(_assess("scam", 0.95), scores, alpha=0.5).verdict == "scam"
    assert decide(_assess("suspicious", 0.5), scores, alpha=0.5).verdict == "suspicious"


def test_unsure_helper() -> None:
    d = unsure()
    assert d.verdict == "unsure" and d.released_benign is False and d.threshold is None
