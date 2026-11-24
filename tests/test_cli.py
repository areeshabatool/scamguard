"""CLI tests (FakeLLM injected; no real API)."""

import json

import pytest

from scamguard import cli
from scamguard.llm import FakeLLM


def _fake() -> FakeLLM:
    return FakeLLM(
        by_tool={
            "submit_facts": {"urls": ["http://hbl.com.evil.net"], "phone_numbers": []},
            "give_advice": {"action_ids": ["do_not_click"], "message": "Click na karein."},
        },
        turns=[[("submit_verdict", {"verdict": "scam", "scam_score": 0.9, "cues": ["verify"]})]],
    )


def test_cli_analyze_prints_decision(capsys: pytest.CaptureFixture[str]) -> None:
    code = cli.main(["analyze", "Please verify http://hbl.com.evil.net now"], llm=_fake())
    out = capsys.readouterr().out
    assert code == 0
    assert "Decision: scam" in out
    assert "Advice:" in out


def test_cli_analyze_json(capsys: pytest.CaptureFixture[str]) -> None:
    code = cli.main(["analyze", "Please verify http://hbl.com.evil.net now", "--json"], llm=_fake())
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["decision"] == "scam"
    assert [t["name"] for t in payload["trace"]][0] == "message_in"


def test_cli_requires_subcommand() -> None:
    with pytest.raises(SystemExit):
        cli.main([])
