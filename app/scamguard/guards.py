"""Step 7 (CODE): cue and risk check (the guards).

Three guarantees, all enforced by code so nothing in the message can weaken them:
  - Exact-substring cues (invariant 5): a reported cue is kept only if it is an exact
    substring of the sanitized message; everything else is dropped.
  - Final risk = max(model risk, tool risk, marker floor) (invariant 3): code risk and
    AI-address markers can only raise the score, never lower it.
  - Verdict bands: the score maps to a band, and the model's verdict is raised (never
    lowered) to that band.
"""

from __future__ import annotations

from collections.abc import Iterable

from scamguard import config
from scamguard.schemas import Assessment, Detection, SanitizedMessage, ToolReport

_RANK = {"benign": 0, "suspicious": 1, "scam": 2}
_BY_RANK = {rank: name for name, rank in _RANK.items()}


def filter_cues(cues: Iterable[str], text: str) -> list[str]:
    """Keep only cues that are exact substrings of the text; drop the rest, dedupe (invariant 5)."""
    kept: list[str] = []
    seen: set[str] = set()
    for cue in cues:
        if cue and cue in text and cue not in seen:
            seen.add(cue)
            kept.append(cue)
    return kept


def final_risk(model_score: float, tool_score: float, marker_floor: float = 0.0) -> float:
    """Final risk can only be raised by code: the max of model, tool and marker floor."""
    return max(model_score, tool_score, marker_floor)


def band_verdict(score: float) -> str:
    """Map a risk score to a verdict band using the calibrated thresholds in config."""
    if score >= config.SCAM_BAND:
        return "scam"
    if score >= config.SUSPICIOUS_BAND:
        return "suspicious"
    return "benign"


def raise_only(verdict: str, other: str) -> str:
    """Return the more severe of two verdicts (a verdict is never lowered)."""
    return _BY_RANK[max(_RANK[verdict], _RANK[other])]


def assess(
    detection: Detection, tool_report: ToolReport, sanitized: SanitizedMessage
) -> Assessment:
    """Combine the model verdict with the code checks into a final Assessment."""
    cues = filter_cues(detection.cues, sanitized.text)
    markers = list(sanitized.ai_address_markers) + list(sanitized.reassurance_markers)
    marker_floor = config.MARKER_RISK_FLOOR if markers else 0.0
    score = final_risk(detection.scam_score, tool_report.tool_score, marker_floor)
    band = band_verdict(score)
    verdict = raise_only(detection.verdict, band)
    return Assessment(
        final_score=score,
        verdict=verdict,
        model_verdict=detection.verdict,
        band_verdict=band,
        cues=cues,
        markers=markers,
    )
