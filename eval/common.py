"""Shared helpers for the eval runners.

JSONL loading with field checks, the commit hash, output paths and LLM factories.
Error messages name files, line numbers and fields only, never message text: the eval
data may live in data/private/.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable, Iterable
from pathlib import Path

from scamguard import config
from scamguard.llm import LLM, FakeLLM

BOOTSTRAP_SEED = 1234
RESULTS_DIR = config.REPO_ROOT / "results"
RAW_DIR = RESULTS_DIR / "raw"  # per-case outputs; git-ignored
SAMPLES_DIR = config.DATA_DIR / "samples"

FLAGGED = frozenset({"scam", "suspicious"})

LLMFactory = Callable[[], LLM]


def flagged(label: str) -> bool:
    """A prediction counts as detected if it is scam or suspicious."""
    return label in FLAGGED


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"dataset not found: {path} (pass --dataset)")
    rows: list[dict] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            raise ValueError(f"{path.name}:{lineno}: not valid JSON") from None
    return rows


def require_fields(rows: Iterable[dict], fields: Iterable[str], path: Path) -> None:
    """Raise if any row lacks a field. Reports line numbers and field names only."""
    fields = list(fields)
    problems = [
        f"row {i}: missing {missing}"
        for i, row in enumerate(rows, start=1)
        if (missing := [f for f in fields if f not in row])
    ]
    if problems:
        shown = "; ".join(problems[:5]) + (" ..." if len(problems) > 5 else "")
        raise ValueError(f"{path.name}: expected fields {fields}: {shown}")


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=config.REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def display_path(path: Path) -> str:
    """Repo-relative path when possible (keeps local folder names out of results)."""
    try:
        return str(path.resolve().relative_to(config.REPO_ROOT))
    except ValueError:
        return path.name


def dry_fake() -> LLM:
    """A scripted FakeLLM for --dry-run: no network, deterministic plumbing check."""
    return FakeLLM(
        by_tool={
            "submit_verdict": {"verdict": "scam", "scam_score": 0.9, "cues": []},
            "submit_facts": {"urls": [], "phone_numbers": []},
            "give_advice": {"action_ids": ["do_not_click"], "message": "Click na karein."},
        },
        turns=[[("submit_verdict", {"verdict": "scam", "scam_score": 0.9})]],
    )


def live_factory(model: str) -> LLMFactory:
    """One shared client per model so its disk cache is reused across calls.

    Routes through build_default_llm, so the OpenAI backend is used when selected (a
    Claude model name is then ignored in favour of the OpenAI default model).
    """
    from scamguard.llm import build_default_llm

    shared = build_default_llm(model)
    return lambda: shared


def load_calibration(path: Path | None = None) -> list[float]:
    """The gate's calibration scores (a JSON list, or {"scores": [...]}). Missing -> []."""
    path = path or config.CALIBRATION_PATH
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    scores = data.get("scores", data) if isinstance(data, dict) else data
    return [float(s) for s in scores] if isinstance(scores, list) else []


def write_report(report: dict, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return out


def write_raw(rows: Iterable[dict], name: str) -> Path:
    """Per-case outputs (ids and labels, never text) to results/raw/ (git-ignored)."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out = RAW_DIR / name
    out.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return out
