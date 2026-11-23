"""Pipeline wiring tests (FakeLLM only): the 9-step trace and the fail-safe behavior."""

from scamguard.llm import FakeLLM, LLMError
from scamguard.ocr import OCRResult
from scamguard.pipeline import Session, analyze

STEP_NAMES = [
    "message_in",
    "ocr",
    "sanitize",
    "extract",
    "verify",
    "detect",
    "guards",
    "gate",
    "advise",
]


def scam_fake(**extra) -> FakeLLM:
    return FakeLLM(
        by_tool={
            "submit_facts": {"urls": ["http://hbl.com.evil.net/login"], "phone_numbers": []},
            "give_advice": {"action_ids": ["do_not_click"], "message": "Click na karein."},
        },
        turns=[[("submit_verdict", {"verdict": "scam", "scam_score": 0.9, "cues": ["verify"]})]],
        **extra,
    )


def benign_fake(**extra) -> FakeLLM:
    return FakeLLM(
        by_tool={
            "submit_facts": {"urls": [], "phone_numbers": []},
            "give_advice": {"action_ids": ["verify_official"], "message": "Theek lagta hai."},
        },
        turns=[[("submit_verdict", {"verdict": "benign", "scam_score": 0.0})]],
        **extra,
    )


# ---------- happy paths and the trace ----------


def test_scam_case_full_trace() -> None:
    res = analyze("Your HBL account blocked, verify http://hbl.com.evil.net/login", llm=scam_fake())
    assert res.decision == "scam"
    assert res.advice
    assert res.cues == ["verify"]
    assert [t.name for t in res.trace] == STEP_NAMES
    assert all(t.status == "ok" for t in res.trace if t.name != "ocr")


def test_benign_released_with_calibration() -> None:
    # n=5, alpha=0.5 -> k=3 -> threshold = 3rd smallest = 0.7; score 0.0 < 0.7 releases benign.
    cal = [0.5, 0.6, 0.7, 0.8, 0.9]
    res = analyze("salam, kaise ho theek hai", llm=benign_fake(), calibration_scores=cal, alpha=0.5)
    assert res.decision == "benign"


def test_benign_becomes_unsure_without_calibration() -> None:
    # No calibration -> k < 1 -> benign can never be released (invariant 6).
    res = analyze("salam, kaise ho theek hai", llm=benign_fake())
    assert res.decision == "unsure"


# ---------- input guards (never call the model) ----------


def test_too_little_text_is_unsure_without_calling_model() -> None:
    fake = FakeLLM()
    res = analyze("hi", llm=fake)
    assert res.decision == "unsure"
    assert res.reason == "need_clearer_input"
    assert fake.calls == [] and fake.run_calls == []


def test_low_ocr_confidence_is_unsure() -> None:
    fake = FakeLLM()
    res = analyze(
        ocr=OCRResult(text="Your account is blocked, verify now", confidence=0.2), llm=fake
    )
    assert res.decision == "unsure"
    assert res.reason == "need_clearer_input"
    assert fake.calls == [] and fake.run_calls == []


def test_high_ocr_confidence_proceeds() -> None:
    res = analyze(
        ocr=OCRResult(text="HBL account blocked verify http://hbl.com.evil.net", confidence=0.95),
        llm=scam_fake(),
    )
    assert res.decision == "scam"
    assert res.trace[1].status == "ok"  # the ocr step ran


# ---------- fail safe: never benign on failure ----------


def test_extractor_failure_is_unsure() -> None:
    fake = FakeLLM(by_tool={"submit_facts": LLMError("boom")})
    res = analyze("a long enough message to pass the length check", llm=fake)
    assert res.decision == "unsure"
    assert res.reason == "model_or_schema_failure"


def test_detector_schema_failure_is_unsure() -> None:
    fake = FakeLLM(
        by_tool={"submit_facts": {"urls": [], "phone_numbers": []}},
        turns=[[("submit_verdict", {"verdict": "safe", "scam_score": 0.1})]],  # invalid verdict
    )
    res = analyze("a long enough message to pass the length check", llm=fake)
    assert res.decision == "unsure"


def test_advisor_failure_never_releases_benign() -> None:
    cal = [0.5, 0.6, 0.7, 0.8, 0.9]
    fake = FakeLLM(
        by_tool={"submit_facts": {"urls": [], "phone_numbers": []}, "give_advice": LLMError("x")},
        turns=[[("submit_verdict", {"verdict": "benign", "scam_score": 0.0})]],
    )
    res = analyze("salam, kaise ho theek hai", llm=fake, calibration_scores=cal, alpha=0.5)
    assert res.decision == "unsure"  # would have released benign, but the advisor failed
    assert res.advice  # safe fallback advice is still provided


def test_advisor_failure_keeps_scam_verdict() -> None:
    fake = FakeLLM(
        by_tool={"submit_facts": {"urls": [], "phone_numbers": []}, "give_advice": LLMError("x")},
        turns=[[("submit_verdict", {"verdict": "scam", "scam_score": 0.95})]],
    )
    res = analyze("a long enough scammy looking message", llm=fake)
    assert res.decision == "scam"
    assert res.action_ids  # fallback advice ids


# ---------- Session (in-memory multi-turn) ----------


def test_session_accumulates_results_in_memory() -> None:
    fake = FakeLLM(
        by_tool={
            "submit_facts": {"urls": [], "phone_numbers": []},
            "give_advice": {"action_ids": ["verify_official"], "message": "Dhyan rakhein."},
        },
        turns=[
            [("submit_verdict", {"verdict": "scam", "scam_score": 0.9})],
            [("submit_verdict", {"verdict": "suspicious", "scam_score": 0.5})],
        ],
    )
    session = Session(fake)
    r1 = session.analyze("first long enough message")
    r2 = session.analyze("second long enough message")
    assert session.count == 2
    assert (r1.decision, r2.decision) == ("scam", "suspicious")
    session.clear()
    assert session.count == 0
