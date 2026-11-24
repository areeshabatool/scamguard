"""API tests with Starlette's TestClient and an injected FakeLLM (no real API)."""

import pytest
from starlette.testclient import TestClient

from scamguard import api
from scamguard.llm import FakeLLM
from scamguard.ocr import OCRResult


@pytest.fixture(autouse=True)
def _reset_app():
    api.app.state.llm = None
    api.app.state.ocr_engine = None
    api.app.state.calibration_scores = []
    api._sessions.clear()
    yield


@pytest.fixture
def client():
    return TestClient(api.app)


def scam_fake(**extra) -> FakeLLM:
    return FakeLLM(
        by_tool={
            "submit_facts": {"urls": ["http://hbl.com.evil.net"], "phone_numbers": []},
            "give_advice": {"action_ids": ["do_not_click"], "message": "Click na karein."},
        },
        turns=[[("submit_verdict", {"verdict": "scam", "scam_score": 0.9, "cues": ["verify"]})]],
        **extra,
    )


def test_health(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_index_serves_form(client: TestClient) -> None:
    resp = client.get("/")
    assert resp.status_code == 200
    assert "Scam Guard" in resp.text
    assert "<form" in resp.text


def test_analyze_text(client: TestClient) -> None:
    api.app.state.llm = scam_fake()
    resp = client.post("/analyze", data={"text": "Please verify http://hbl.com.evil.net now"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["decision"] == "scam"
    assert body["cues"] == ["verify"]
    assert [t["name"] for t in body["trace"]][0] == "message_in"
    assert body["advice"]


def test_analyze_requires_text_or_image(client: TestClient) -> None:
    resp = client.post("/analyze", data={})
    assert resp.status_code == 422


def test_analyze_image_without_engine_is_unsure(client: TestClient) -> None:
    api.app.state.llm = FakeLLM()  # must not be called; the read is zero-confidence
    resp = client.post(
        "/analyze",
        files={"image": ("shot.png", b"\x89PNG\r\n fake bytes", "image/png")},
    )
    assert resp.status_code == 200
    assert resp.json()["decision"] == "unsure"


def test_analyze_image_with_engine_runs_pipeline(client: TestClient) -> None:
    api.app.state.llm = scam_fake()
    api.app.state.ocr_engine = lambda data: OCRResult(
        text="HBL account blocked, verify http://hbl.com.evil.net", confidence=0.95
    )
    resp = client.post(
        "/analyze",
        files={"image": ("shot.png", b"bytes", "image/png")},
    )
    assert resp.status_code == 200
    assert resp.json()["decision"] == "scam"


def test_session_multi_turn(client: TestClient) -> None:
    api.app.state.llm = FakeLLM(
        by_tool={
            "submit_facts": {"urls": [], "phone_numbers": []},
            "give_advice": {"action_ids": ["verify_official"], "message": "Dhyan rakhein."},
        },
        turns=[
            [("submit_verdict", {"verdict": "scam", "scam_score": 0.9})],
            [("submit_verdict", {"verdict": "suspicious", "scam_score": 0.5})],
        ],
    )
    r1 = client.post("/analyze", data={"text": "first long enough message", "session_id": "s1"})
    r2 = client.post("/analyze", data={"text": "second long enough message", "session_id": "s1"})
    assert r1.json()["decision"] == "scam"
    assert r2.json()["decision"] == "suspicious"
    assert api._sessions["s1"].count == 2
