"""Tests for eval/failure_tags.py: cause classification, aggregation, CSV, dry-run."""

import csv
import json
from pathlib import Path

from eval import failure_tags as ft
from eval.failure_tags import (
    CaseSignals,
    TaggedCase,
    classify,
    count_table,
    tag,
    write_csv,
)


def sig(label: str, decision: str, **kw) -> CaseSignals:
    return CaseSignals(
        case_id=kw.pop("case_id", 1),
        system=kw.pop("system", "B3-pipeline"),
        label=label,
        decision=decision,
        **kw,
    )


# ---------- correct cases are not failures ----------


def test_correct_predictions_have_no_cause() -> None:
    assert classify(sig("scam", "scam")) is None
    assert classify(sig("scam", "suspicious")) is None
    assert classify(sig("benign", "benign")) is None
    assert classify(sig("benign", "unsure")) is None  # abstaining on benign is not a failure


# ---------- misses (scam not flagged) ----------


def test_ocr_error_takes_priority() -> None:
    assert classify(sig("scam", "unsure", ocr_confidence=0.2)) == ft.OCR_ERROR
    assert classify(sig("scam", "benign", ocr_confidence=0.1)) == ft.OCR_ERROR


def test_gate_abstention_on_unsure_miss() -> None:
    assert classify(sig("scam", "unsure", ocr_confidence=None)) == ft.GATE_ABSTENTION
    assert classify(sig("scam", "unsure", ocr_confidence=0.95)) == ft.GATE_ABSTENTION


def test_extraction_error_when_gold_signal_missed() -> None:
    s = sig("scam", "benign", gold_urls=("http://bad.example",), extracted_urls=())
    assert classify(s) == ft.EXTRACTION_ERROR
    s2 = sig("scam", "benign", gold_requests=("send otp",), extracted_requests=())
    assert classify(s2) == ft.EXTRACTION_ERROR
    s3 = sig("scam", "benign", gold_authority_claim="bank", extracted_authority=None)
    assert classify(s3) == ft.EXTRACTION_ERROR


def test_tool_gap_when_url_extracted_but_unflagged() -> None:
    s = sig(
        "scam",
        "benign",
        gold_urls=("http://bad.example",),
        extracted_urls=("http://bad.example",),
        tool_flags=(),
    )
    assert classify(s) == ft.TOOL_GAP


def test_model_misjudgement_when_everything_upstream_ok() -> None:
    s = sig("scam", "benign", gold_urls=(), extracted_urls=(), tool_flags=())
    assert classify(s) == ft.MODEL_MISJUDGEMENT


# ---------- false alarms (benign flagged) ----------


def test_false_alarm_over_extraction() -> None:
    s = sig("benign", "scam", gold_urls=(), extracted_urls=("http://y.example",))
    assert classify(s) == ft.EXTRACTION_ERROR


def test_false_alarm_tool_gap() -> None:
    s = sig("benign", "scam", tool_flags=("lookalike",))
    assert classify(s) == ft.TOOL_GAP


def test_false_alarm_model_misjudgement() -> None:
    assert classify(sig("benign", "suspicious")) == ft.MODEL_MISJUDGEMENT


def test_false_alarm_ocr_error() -> None:
    assert classify(sig("benign", "scam", ocr_confidence=0.1)) == ft.OCR_ERROR


# ---------- aggregation and CSV ----------


def test_count_table_aggregates_per_system_per_cause() -> None:
    tagged = [
        TaggedCase(sig("scam", "unsure", system="B3-pipeline"), ft.GATE_ABSTENTION),
        TaggedCase(sig("scam", "benign", system="B3-pipeline"), ft.MODEL_MISJUDGEMENT),
        TaggedCase(sig("benign", "scam", system="B1-single-prompt"), ft.MODEL_MISJUDGEMENT),
    ]
    table = count_table(tagged, systems=["B1-single-prompt", "B3-pipeline"])
    assert table["B3-pipeline"][ft.GATE_ABSTENTION] == 1
    assert table["B3-pipeline"][ft.MODEL_MISJUDGEMENT] == 1
    assert table["B3-pipeline"]["total"] == 2
    assert table["B1-single-prompt"][ft.MODEL_MISJUDGEMENT] == 1
    assert table["B1-single-prompt"][ft.TOOL_GAP] == 0


def test_tag_keeps_only_failures() -> None:
    signals = [sig("scam", "scam"), sig("scam", "unsure"), sig("benign", "benign")]
    tagged = tag(signals)
    assert len(tagged) == 1 and tagged[0].cause == ft.GATE_ABSTENTION


def test_write_csv_has_header_and_no_message_text(tmp_path: Path) -> None:
    tagged = [
        TaggedCase(
            sig("scam", "benign", case_id=7, gold_urls=("http://bad.example",)),
            ft.EXTRACTION_ERROR,
        )
    ]
    out = write_csv(tagged, tmp_path / "f.csv")
    rows = list(csv.reader(out.read_text(encoding="utf-8").splitlines()))
    assert rows[0] == list(ft._CSV_COLUMNS)
    assert rows[1][0] == "7" and rows[1][4] == ft.EXTRACTION_ERROR
    assert "http://bad.example" in rows[1]  # gold url kept for review


# ---------- the dry-run runner ----------


def _write_gold(path: Path) -> None:
    rows = [
        {
            "id": 1,
            "label": "scam",
            "text": "Send your OTP now to unlock your blocked account",
            "gold_urls": [],
            "gold_requests": ["send otp"],
            "gold_authority_claim": "bank",
        },
        {
            "id": 2,
            "label": "benign",
            "text": "Kal milte hain shaam ko chai peene",
            "gold_urls": [],
            "gold_requests": [],
            "gold_authority_claim": None,
        },
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")


def test_dry_run_writes_table_and_csv(tmp_path: Path) -> None:
    dataset = tmp_path / "gold.jsonl"
    _write_gold(dataset)
    out = tmp_path / "failure_tags.json"
    csv_out = tmp_path / "failure_tags.csv"

    code = ft.main(
        ["--dry-run", "--dataset", str(dataset), "--out", str(out), "--csv", str(csv_out)]
    )
    assert code == 0

    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["dry_run"] is True
    assert report["commit"]
    assert set(report["counts"]) == set(ft.SYSTEM_ORDER)
    for row in report["counts"].values():
        assert set(row) == {*ft.CAUSES, "total"}

    # The scripted FakeLLM always says scam, so the benign row is a false alarm for B1-B3.
    assert report["counts"]["B3-pipeline"]["total"] >= 1
    assert csv_out.exists()

    # No message text leaks into either output.
    blob = out.read_text(encoding="utf-8") + csv_out.read_text(encoding="utf-8")
    assert "Kal milte hain" not in blob
    assert "Send your OTP" not in blob
