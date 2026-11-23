"""Tests for the three model steps (extractor, detector, advisor). FakeLLM only."""

import json

import pytest
from pydantic import ValidationError

from scamguard.agents import MAX_CHECK_STEPS, SAFE_ACTIONS, advise, detect, extract
from scamguard.llm import FakeLLM, LLMError
from scamguard.sanitizer import TAG, sanitize
from scamguard.schemas import Advice, Detection, Facts


def msg(raw: str):
    return sanitize(raw)


OPEN = f"<{TAG}>"


# ---------- extractor ----------


def test_extractor_fills_facts() -> None:
    fake = FakeLLM(
        default={
            "urls": ["http://x.example/login"],
            "phone_numbers": ["0300-1234567"],
            "claimed_sender": "HBL",
            "requested_action": "verify account",
        }
    )
    facts = extract(fake, msg("Your HBL account is blocked, verify now"))
    assert isinstance(facts, Facts)
    assert facts.urls == ["http://x.example/login"]
    assert facts.claimed_sender == "HBL"
    # The message rode inside an <untrusted_message> block in the user turn (invariant 1).
    assert OPEN in fake.calls[0]["user"]


def test_extractor_defaults_when_optionals_missing() -> None:
    fake = FakeLLM(default={"urls": [], "phone_numbers": []})
    facts = extract(fake, msg("salam, kaise ho"))
    assert facts.claimed_sender is None
    assert facts.requested_action is None


# ---------- detector ----------


def test_detector_submits_directly() -> None:
    fake = FakeLLM(
        turns=[[("submit_verdict", {"verdict": "scam", "scam_score": 0.9, "cues": ["OTP"]})]]
    )
    det = detect(fake, msg("Send your OTP to claim prize"))
    assert isinstance(det, Detection)
    assert det.verdict == "scam" and det.scam_score == 0.9 and det.cues == ["OTP"]
    assert OPEN in fake.run_calls[0]["messages"][0]["content"]
    assert fake.run_calls[0]["tool_choice"] == "auto"


def test_detector_runs_check_domain_then_submits() -> None:
    official = frozenset({"hbl.com"})
    fake = FakeLLM(
        turns=[
            [("check_domain", {"url": "http://hbl.com.secure-login.net/verify"})],
            [("submit_verdict", {"verdict": "scam", "scam_score": 0.85})],
        ]
    )
    det = detect(fake, msg("login http://hbl.com.secure-login.net/verify"), official=official)
    assert det.verdict == "scam"
    assert len(fake.run_calls) == 2
    # The verifier result (a lookalike flag) was fed back as a tool_result before turn 2.
    fed_back = json.dumps(fake.run_calls[1]["messages"][-1])
    assert "lookalike" in fed_back


def test_detector_forces_submit_on_final_step() -> None:
    check = [("check_domain", {"url": "http://a.example"})]
    fake = FakeLLM(
        turns=[
            check,
            check,
            check,
            check,
            [("submit_verdict", {"verdict": "suspicious", "scam_score": 0.5})],
        ]
    )
    det = detect(fake, msg("visit http://a.example"), official=frozenset())
    assert det.verdict == "suspicious"
    assert len(fake.run_calls) == MAX_CHECK_STEPS + 1
    assert fake.run_calls[0]["tool_choice"] == "auto"
    # The last turn forces submit_verdict.
    assert fake.run_calls[-1]["tool_choice"] == {"type": "tool", "name": "submit_verdict"}


def test_detector_raises_when_verdict_never_submitted() -> None:
    check = [("check_domain", {"url": "http://a.example"})]
    fake = FakeLLM(turns=[check] * (MAX_CHECK_STEPS + 1))  # never submits, even when forced
    with pytest.raises(LLMError):
        detect(fake, msg("visit http://a.example"), official=frozenset())
    assert len(fake.run_calls) == MAX_CHECK_STEPS + 1


def test_detector_invalid_verdict_raises_validation_error() -> None:
    fake = FakeLLM(turns=[[("submit_verdict", {"verdict": "safe", "scam_score": 0.1})]])
    with pytest.raises(ValidationError):
        detect(fake, msg("hello"))


def test_detector_score_out_of_range_raises() -> None:
    fake = FakeLLM(turns=[[("submit_verdict", {"verdict": "scam", "scam_score": 1.5})]])
    with pytest.raises(ValidationError):
        detect(fake, msg("hello"))


# ---------- advisor ----------


def test_advisor_picks_ids_and_strips_contact_info() -> None:
    fake = FakeLLM(
        default={
            "action_ids": ["do_not_click", "verify_official", "not_a_real_id"],
            "message": "Click na karein, 0300-1234567 ya http://evil.example par rabta na karein.",
        }
    )
    adv = advise(
        fake, msg("win prize click http://evil.example"), Detection(verdict="scam", scam_score=0.9)
    )
    assert isinstance(adv, Advice)
    # Unknown id dropped; only allowlisted ids survive.
    assert adv.action_ids == ["do_not_click", "verify_official"]
    # No URL, domain or phone number remains in the phrased sentence (invariant 7).
    assert "http" not in adv.message
    assert "evil.example" not in adv.message
    assert not any(ch.isdigit() for ch in adv.message)
    assert adv.message  # still a non-empty sentence
    assert OPEN in fake.calls[0]["user"]


def test_advisor_drops_all_unknown_ids() -> None:
    fake = FakeLLM(default={"action_ids": ["nope", "also_nope"], "message": "Dhyan rakhein."})
    adv = advise(fake, msg("x"), Detection(verdict="benign", scam_score=0.0))
    assert adv.action_ids == []


def test_safe_actions_contain_no_urls_or_numbers() -> None:
    assert SAFE_ACTIONS  # non-empty fixed list
    for text in SAFE_ACTIONS.values():
        assert "http" not in text and "www" not in text
        assert not any(ch.isdigit() for ch in text)
