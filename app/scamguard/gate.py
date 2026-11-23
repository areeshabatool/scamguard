"""Step 8 (CODE): calibrated gate.

Split-conformal calibration decides when "benign" may be released. Given n calibration
scores and a miss-rate budget alpha:

    k = floor(alpha * (n + 1))
    threshold = the k-th smallest calibration score
    release "benign" only if the message score is strictly below the threshold
    if k < 1, never release "benign"

Any verdict other than "benign" passes through unchanged. A benign verdict that is not
confidently below the threshold becomes "unsure", never benign (invariant 6).
"""

from __future__ import annotations

from collections.abc import Sequence
from math import floor

from scamguard import config
from scamguard.schemas import Assessment, Decision


def conformal_k(n: int, alpha: float) -> int:
    """k = floor(alpha * (n + 1)). k < 1 means benign can never be released."""
    return floor(alpha * (n + 1))


def conformal_threshold(scores: Sequence[float], alpha: float) -> float | None:
    """The k-th smallest calibration score, or None when k < 1 (never release benign)."""
    n = len(scores)
    k = conformal_k(n, alpha)
    if k < 1:
        return None
    ordered = sorted(scores)
    return ordered[min(k, n) - 1]  # k-th smallest (1-indexed); clamp guards alpha->1


def release_benign(score: float, calibration_scores: Sequence[float], alpha: float) -> bool:
    """True iff the score is strictly below the conformal threshold."""
    threshold = conformal_threshold(calibration_scores, alpha)
    return threshold is not None and score < threshold


def decide(
    assessment: Assessment,
    calibration_scores: Sequence[float],
    *,
    alpha: float | None = None,
) -> Decision:
    """Apply the gate to an Assessment and return the released Decision."""
    a = config.ALPHA if alpha is None else alpha
    threshold = conformal_threshold(calibration_scores, a)

    if assessment.verdict != "benign":
        return Decision(
            verdict=assessment.verdict,
            score=assessment.final_score,
            released_benign=False,
            threshold=threshold,
        )

    released = threshold is not None and assessment.final_score < threshold
    return Decision(
        verdict="benign" if released else "unsure",
        score=assessment.final_score,
        released_benign=released,
        threshold=threshold,
    )


def unsure(score: float = 1.0) -> Decision:
    """A fail-safe decision: any model or schema failure upstream yields unsure (invariant 6)."""
    return Decision(verdict="unsure", score=score, released_benign=False, threshold=None)
