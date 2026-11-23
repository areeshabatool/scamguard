"""Schema validation: bad model output must fail loudly so the gate can return unsure."""

import pytest
from pydantic import ValidationError

from scamguard.schemas import Detection, Facts


def test_detection_accepts_valid_output() -> None:
    d = Detection(verdict="scam", scam_score=0.9, cues=["send OTP"])
    assert d.verdict == "scam"
    assert d.cues == ["send OTP"]


@pytest.mark.parametrize("verdict", ["unsure", "safe", "SCAM", ""])
def test_detection_rejects_unknown_verdict(verdict: str) -> None:
    with pytest.raises(ValidationError):
        Detection(verdict=verdict, scam_score=0.5)


@pytest.mark.parametrize("score", [-0.01, 1.01, float("nan")])
def test_detection_rejects_score_outside_0_1(score: float) -> None:
    with pytest.raises(ValidationError):
        Detection(verdict="suspicious", scam_score=score)


def test_detection_score_bounds_inclusive() -> None:
    assert Detection(verdict="benign", scam_score=0.0).scam_score == 0.0
    assert Detection(verdict="scam", scam_score=1.0).scam_score == 1.0


def test_detection_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        Detection(verdict="benign", scam_score=0.1, final_verdict="benign")


def test_facts_defaults_and_extra_fields() -> None:
    f = Facts()
    assert f.urls == [] and f.phone_numbers == [] and f.claimed_sender is None
    with pytest.raises(ValidationError):
        Facts(urls=[], risk="low")
