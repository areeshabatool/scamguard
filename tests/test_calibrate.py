"""Tests for eval/calibrate.py (dry-run; no API)."""

import json
from pathlib import Path

from eval import calibrate


def test_calibrate_dry_run_writes_scam_scores(tmp_path: Path) -> None:
    dataset = tmp_path / "scams.jsonl"
    rows = [
        {"id": 1, "text": "Send your OTP now to unlock your blocked account", "label": "scam"},
        {"id": 2, "text": "You won a prize, click the link now to claim", "label": "scam"},
        {"id": 3, "text": "Kal milte hain shaam ko", "label": "benign"},  # skipped
    ]
    dataset.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    out = tmp_path / "calibration.json"

    code = calibrate.main(["--dry-run", "--dataset", str(dataset), "--out", str(out)])
    assert code == 0

    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["n"] == 2  # only the two scam rows were scored
    assert len(data["scores"]) == 2
    assert all(isinstance(s, (int, float)) for s in data["scores"])
    assert data["commit"]
