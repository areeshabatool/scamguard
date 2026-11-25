"""E1 harness tests: the systems behind predict() and the --dry-run runner (FakeLLM)."""

import json
from pathlib import Path

from eval import run_e1
from eval.systems import B0, B1, B2, B3, turns_to_detection
from scamguard.llm import FakeLLM


def _verdict_fake(verdict: str) -> FakeLLM:
    return FakeLLM(by_tool={"submit_verdict": {"verdict": verdict, "scam_score": 0.5}})


# ---------- systems behind predict() ----------


def test_b0_flags_keywords_without_a_model() -> None:
    b0 = B0()
    assert b0.predict("Send your OTP now").label == "scam"
    assert b0.predict("Kal milte hain").label == "benign"
    assert b0.predict("anything").turns == 0  # no model calls


def test_turns_to_detection_flags_at_the_scam_turn() -> None:
    convo = ["Hi beta kaise ho", "Sab theek hai?", "Send your OTP now to unlock account"]
    assert turns_to_detection(B0(), convo) == 3  # B0 flags once the OTP ask appears


def test_turns_to_detection_none_for_a_benign_chat() -> None:
    assert turns_to_detection(B0(), ["Kal milte hain", "Theek hai", "Chai peenge"]) is None


def test_b1_maps_model_verdict() -> None:
    pred = B1(lambda: _verdict_fake("scam")).predict("hello")
    assert pred.label == "scam" and pred.turns == 1


def test_b2_wraps_message_as_untrusted() -> None:
    fake = _verdict_fake("benign")
    pred = B2(lambda: fake).predict("some message text")
    assert pred.label == "benign"
    assert "<untrusted_message>" in fake.calls[0]["user"]


def test_single_prompt_schema_failure_is_unsure() -> None:
    bad = FakeLLM(by_tool={"submit_verdict": {"verdict": "safe", "scam_score": 0.1}})
    assert B1(lambda: bad).predict("x").label == "unsure"


def test_b3_runs_full_pipeline_turns() -> None:
    def fake() -> FakeLLM:
        return FakeLLM(
            by_tool={
                "submit_facts": {"urls": [], "phone_numbers": []},
                "give_advice": {"action_ids": ["do_not_click"], "message": "Click na karein."},
            },
            turns=[[("submit_verdict", {"verdict": "scam", "scam_score": 0.9})]],
        )

    pred = B3(fake).predict("a long enough scammy message")
    assert pred.label == "scam"
    assert pred.turns == 3  # extract + detect + advise


# ---------- the --dry-run runner ----------


def test_dry_run_writes_results_json(tmp_path: Path) -> None:
    out = tmp_path / "e1.json"
    code = run_e1.main(["--dry-run", "--out", str(out), "--resamples", "200"])
    assert code == 0
    report = json.loads(out.read_text(encoding="utf-8"))

    assert report["dry_run"] is True
    assert report["commit"]  # a hash or "unknown", never empty
    assert report["seeds"]["bootstrap"] == run_e1.BOOTSTRAP_SEED
    assert set(report["systems"]) == {
        "B0-keyword",
        "B1-single-prompt",
        "B2-hardened-prompt",
        "B3-pipeline",
    }
    for metrics_block in report["systems"].values():
        for key in (
            "recall",
            "fpr",
            "abstain_rate",
            "avg_turns",
            "avg_cost_tokens",
            "avg_latency_s",
        ):
            assert key in metrics_block
    assert report["systems"]["B0-keyword"]["avg_turns"] == 0
    assert report["systems"]["B3-pipeline"]["avg_turns"] == 3
    assert "B3_vs_B0_recall" in report["paired"]


def test_dry_run_does_not_store_message_content(tmp_path: Path) -> None:
    out = tmp_path / "e1.json"
    run_e1.main(["--dry-run", "--out", str(out), "--resamples", "50"])
    blob = out.read_text(encoding="utf-8")
    dataset = run_e1.read_dataset(run_e1.DEFAULT_DATASET)
    for row in dataset:
        assert row["text"] not in blob  # results carry metrics only, never the messages
