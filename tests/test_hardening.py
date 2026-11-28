"""Tests for the hardening additions: reassurance markers and the OCR confidence helper."""

import sys
import types

from scamguard import guards, ocr
from scamguard.sanitizer import find_reassurance, sanitize
from scamguard.schemas import Detection, ToolReport

BENIGN_SAMPLES = [
    "Kal 5 baje milte hain, time pe aa jana.",
    "Your appointment is confirmed for Monday at 10am.",
    "Ammi ne kaha ghar jaldi aana, khana tayar hai.",
    "Meeting notes are attached, let me know your thoughts.",
    "Happy birthday! Hope you have a wonderful day today.",
    "Can you send me the grocery list for tonight please?",
    "The cricket match starts at 7, do not be late.",
]


# ---------- reassurance markers ----------


def test_reassurance_phrases_detected() -> None:
    for text in [
        "This is 100% genuine, not a scam.",
        "totally legit offer, claim now",
        "Bhai ye bilkul safe hai",
        "koi scam nahi hai, trust me",
    ]:
        assert find_reassurance(text), text


def test_reassurance_not_triggered_on_benign_messages() -> None:
    for text in BENIGN_SAMPLES:
        assert find_reassurance(text) == [], text


def test_sanitize_populates_reassurance_markers() -> None:
    s = sanitize("Claim your prize. This is 100% genuine, not a scam.")
    assert s.reassurance_markers


def test_reassurance_raises_risk_floor_in_guards() -> None:
    s = sanitize("Your prize is ready. This is 100% genuine, not a scam. Send details.")
    assessment = guards.assess(
        Detection(verdict="benign", scam_score=0.0), ToolReport(tool_score=0.0), s
    )
    assert assessment.final_score >= 0.30
    assert assessment.verdict != "benign"
    assert assessment.markers  # the phrase is recorded


# ---------- OCR confidence helper ----------


def test_average_confidence() -> None:
    assert ocr.average_confidence([90, 80, 100]) == (90 + 80 + 100) / 3 / 100
    assert ocr.average_confidence([-1, -1]) == 0.0  # unscored words ignored
    assert ocr.average_confidence([]) == 0.0
    assert ocr.average_confidence([200]) == 1.0  # clamped into [0, 1]


def test_tesseract_engine_uses_pytesseract(monkeypatch) -> None:
    fake = types.SimpleNamespace()
    fake.Output = types.SimpleNamespace(DICT="dict")

    def image_to_data(image, lang, output_type):
        return {"text": ["Send", "OTP", "", "now"], "conf": [90.0, 85.0, -1, 95.0]}

    fake.image_to_data = image_to_data
    monkeypatch.setitem(sys.modules, "pytesseract", fake)

    # Raw image bytes (as the web API passes) are opened with Pillow before OCR.
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (4, 4), "white").save(buf, "PNG")

    result = ocr.tesseract_engine(buf.getvalue())
    assert result.text == "Send OTP now"  # blank word dropped
    assert result.confidence == (90.0 + 85.0 + 95.0) / 3 / 100  # -1 ignored
