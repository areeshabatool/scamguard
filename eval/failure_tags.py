"""Failure attribution for the eval: why did each miss or false alarm happen?

For every case that a system got wrong -- a miss (a scam not flagged) or a false alarm
(a benign flagged) -- assign one cause, using the dataset's gold fields (gold_urls,
gold_requests, gold_authority_claim) and the OCR confidence, plus the pipeline's own
intermediate signals (extracted facts, verifier flags):

  ocr_error          the input was a low-confidence OCR read (garbled before any model)
  gate_abstention    the system abstained ("unsure") instead of deciding (misses only)
  extraction_error   the extractor missed gold signals (miss) or invented them (false alarm)
  tool_gap           the deterministic verifier missed a bad URL, or flagged a benign one
  model_misjudgement the facts were right and the model still judged wrong

Outputs a count table per cause per system and a CSV of tagged cases for manual review.
The CSV carries ids, labels and signal summaries only -- never message text (eval data may
be private); reviewers join back to the dataset by id.
"""

from __future__ import annotations

import argparse
import csv
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from eval import common
from eval.systems import B0, B1, B2, B3
from scamguard import config, pipeline, verifier
from scamguard.ocr import OCRResult
from scamguard.sanitizer import sanitize

OCR_ERROR = "ocr_error"
EXTRACTION_ERROR = "extraction_error"
TOOL_GAP = "tool_gap"
MODEL_MISJUDGEMENT = "model_misjudgement"
GATE_ABSTENTION = "gate_abstention"
CAUSES = (OCR_ERROR, EXTRACTION_ERROR, TOOL_GAP, MODEL_MISJUDGEMENT, GATE_ABSTENTION)

GOLD_FIELDS = ("id", "label", "text", "gold_urls", "gold_requests", "gold_authority_claim")


@dataclass(frozen=True)
class CaseSignals:
    """Everything needed to attribute one (system, case) outcome. No message text."""

    case_id: Any
    system: str
    label: str  # gold label: "scam" | "benign"
    decision: str  # predicted: "scam" | "suspicious" | "benign" | "unsure"
    ocr_confidence: float | None = None
    gold_urls: tuple[str, ...] = ()
    gold_requests: tuple[str, ...] = ()
    gold_authority_claim: str | None = None
    extracted_urls: tuple[str, ...] = ()
    extracted_requests: tuple[str, ...] = ()
    extracted_authority: str | None = None
    tool_flags: tuple[str, ...] = ()
    tool_score: float = 0.0


@dataclass(frozen=True)
class TaggedCase:
    signals: CaseSignals
    cause: str


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def is_failure(label: str, decision: str) -> bool:
    """A miss (scam not flagged) or a false alarm (benign flagged)."""
    flagged = common.flagged(decision)
    return (label == "scam" and not flagged) or (label == "benign" and flagged)


def _extraction_incomplete(s: CaseSignals) -> bool:
    """A gold signal the message carried that the extractor failed to surface."""
    missing_url = any(u not in s.extracted_urls for u in s.gold_urls)
    missing_request = bool(s.gold_requests) and not s.extracted_requests
    missing_authority = bool(s.gold_authority_claim) and not s.extracted_authority
    return missing_url or missing_request or missing_authority


def _over_extracted(s: CaseSignals) -> bool:
    """A signal the extractor surfaced that the gold says is not there (false alarm)."""
    extra_url = any(u not in s.gold_urls for u in s.extracted_urls)
    extra_request = bool(s.extracted_requests) and not s.gold_requests
    extra_authority = bool(s.extracted_authority) and not s.gold_authority_claim
    return extra_url or extra_request or extra_authority


def classify(s: CaseSignals) -> str | None:
    """Return the failure cause for a wrong case, or None if the case was correct."""
    if not is_failure(s.label, s.decision):
        return None

    # A degraded OCR read is the root cause regardless of what followed.
    if s.ocr_confidence is not None and s.ocr_confidence < config.MIN_OCR_CONFIDENCE:
        return OCR_ERROR

    if s.label == "scam":  # a miss
        if s.decision == "unsure":
            return GATE_ABSTENTION
        if _extraction_incomplete(s):
            return EXTRACTION_ERROR
        gold_urls_seen = bool(s.gold_urls) and all(u in s.extracted_urls for u in s.gold_urls)
        if gold_urls_seen and not s.tool_flags:
            return TOOL_GAP  # the verifier had the URL but flagged nothing
        return MODEL_MISJUDGEMENT

    # a false alarm (benign flagged)
    if _over_extracted(s):
        return EXTRACTION_ERROR
    if s.tool_flags:
        return TOOL_GAP  # the verifier flagged a benign URL
    return MODEL_MISJUDGEMENT


def tag(signals: Iterable[CaseSignals]) -> list[TaggedCase]:
    """Keep only the failures and attach their cause."""
    tagged: list[TaggedCase] = []
    for s in signals:
        cause = classify(s)
        if cause is not None:
            tagged.append(TaggedCase(signals=s, cause=cause))
    return tagged


# ---------------------------------------------------------------------------
# Aggregation and output
# ---------------------------------------------------------------------------


def count_table(
    tagged: Iterable[TaggedCase], systems: Sequence[str] | None = None
) -> dict[str, dict[str, int]]:
    """counts[system][cause] -> int, with every cause present and a 'total' column."""
    tagged = list(tagged)
    if systems is None:
        systems = sorted({t.signals.system for t in tagged})
    table = {sys: dict.fromkeys((*CAUSES, "total"), 0) for sys in systems}
    for t in tagged:
        row = table.setdefault(t.signals.system, dict.fromkeys((*CAUSES, "total"), 0))
        row[t.cause] += 1
        row["total"] += 1
    return table


def format_count_table(table: dict[str, dict[str, int]]) -> str:
    """A fixed-width text rendering of the count table."""
    header = ["system", *CAUSES, "total"]
    widths = [max(len(header[0]), *(len(s) for s in table)) if table else len(header[0])]
    widths += [max(len(c), 5) for c in (*CAUSES, "total")]
    lines = ["  ".join(h.ljust(w) for h, w in zip(header, widths, strict=True))]
    for sysname, row in table.items():
        cells = [sysname, *(str(row[c]) for c in (*CAUSES, "total"))]
        lines.append("  ".join(c.ljust(w) for c, w in zip(cells, widths, strict=True)))
    return "\n".join(lines)


_CSV_COLUMNS = (
    "case_id",
    "system",
    "label",
    "decision",
    "cause",
    "ocr_confidence",
    "gold_urls",
    "extracted_urls",
    "gold_requests",
    "extracted_requests",
    "gold_authority_claim",
    "extracted_authority",
    "tool_flags",
    "tool_score",
)


def _join(values: Sequence[str]) -> str:
    return "|".join(values)


def write_csv(tagged: Iterable[TaggedCase], out: Path) -> Path:
    """Write tagged cases to a CSV for manual review (ids and signals only, no text)."""
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(_CSV_COLUMNS)
        for t in tagged:
            s = t.signals
            writer.writerow(
                [
                    s.case_id,
                    s.system,
                    s.label,
                    s.decision,
                    t.cause,
                    "" if s.ocr_confidence is None else f"{s.ocr_confidence:.2f}",
                    _join(s.gold_urls),
                    _join(s.extracted_urls),
                    _join(s.gold_requests),
                    _join(s.extracted_requests),
                    s.gold_authority_claim or "",
                    s.extracted_authority or "",
                    _join(s.tool_flags),
                    f"{s.tool_score:.2f}",
                ]
            )
    return out


# ---------------------------------------------------------------------------
# Signal collection (runs the systems over a gold dataset)
# ---------------------------------------------------------------------------


@dataclass
class _FactRecorder:
    """Wraps an LLM to capture the extractor's submit_facts output for one prediction."""

    inner: Any
    facts: dict = field(default_factory=dict)

    def call_tool(self, **kwargs: Any) -> Any:
        result = self.inner.call_tool(**kwargs)
        if str(kwargs.get("tool", {}).get("name")) == "submit_facts":
            self.facts = dict(result.data)
        return result

    def run_turn(self, **kwargs: Any) -> Any:
        return self.inner.run_turn(**kwargs)


def _gold(row: dict) -> dict[str, Any]:
    return {
        "gold_urls": tuple(row.get("gold_urls", ())),
        "gold_requests": tuple(row.get("gold_requests", ())),
        "gold_authority_claim": row.get("gold_authority_claim"),
    }


def _simple_signals(system: Any, row: dict) -> CaseSignals:
    """Signals for B0/B1/B2: only the decision is observable (no pipeline internals)."""
    prediction = system.predict(row["text"])
    return CaseSignals(
        case_id=row.get("id"),
        system=system.name,
        label=row["label"],
        decision=prediction.label,
        ocr_confidence=row.get("ocr_confidence"),
        **_gold(row),
    )


def _b3_signals(
    row: dict, llm_factory: common.LLMFactory, calibration: Sequence[float]
) -> CaseSignals:
    """Signals for B3: capture extracted facts and recompute the verifier's flags."""
    recorder = _FactRecorder(llm_factory())
    ocr_conf = row.get("ocr_confidence")
    if ocr_conf is not None:
        result = pipeline.analyze(
            ocr=OCRResult(text=row["text"], confidence=float(ocr_conf)),
            llm=recorder,
            calibration_scores=list(calibration),
        )
    else:
        result = pipeline.analyze(row["text"], llm=recorder, calibration_scores=list(calibration))
    report = verifier.verify(sanitize(row["text"]).text)
    facts = recorder.facts
    request = facts.get("requested_action")
    return CaseSignals(
        case_id=row.get("id"),
        system=B3.name,
        label=row["label"],
        decision=result.decision,
        ocr_confidence=ocr_conf,
        extracted_urls=tuple(facts.get("urls", ())),
        extracted_requests=(request,) if request else (),
        extracted_authority=facts.get("claimed_sender"),
        tool_flags=tuple(report.flags),
        tool_score=report.tool_score,
        **_gold(row),
    )


def collect_signals(
    rows: Sequence[dict], *, llm_factory: common.LLMFactory, calibration: Sequence[float]
) -> list[CaseSignals]:
    """Run B0-B3 over the dataset and return one CaseSignals per (system, case)."""
    b0, b1, b2 = B0(), B1(llm_factory), B2(llm_factory)
    signals: list[CaseSignals] = []
    for row in rows:
        signals.append(_simple_signals(b0, row))
        signals.append(_simple_signals(b1, row))
        signals.append(_simple_signals(b2, row))
        signals.append(_b3_signals(row, llm_factory, calibration))
    return signals


SYSTEM_ORDER = (B0.name, B1.name, B2.name, B3.name)


def run(*, dataset: Path, dry_run: bool) -> tuple[dict, list[TaggedCase]]:
    rows = common.read_jsonl(dataset)
    common.require_fields(rows, GOLD_FIELDS, dataset)
    llm_factory = common.dry_fake if dry_run else common.live_factory(config.MODEL)
    calibration = common.load_calibration()

    signals = collect_signals(rows, llm_factory=llm_factory, calibration=calibration)
    tagged = tag(signals)
    table = count_table(tagged, systems=SYSTEM_ORDER)
    report = {
        "analysis": "failure_tags",
        "commit": common.git_commit(),
        "dry_run": dry_run,
        "dataset": {"path": common.display_path(dataset), "n": len(rows)},
        "causes": list(CAUSES),
        "counts": table,
    }
    return report, tagged


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="eval.failure_tags", description="Tag eval failures.")
    parser.add_argument("--dataset", type=Path, default=config.E3_DATASET)
    parser.add_argument("--dry-run", action="store_true", help="use a scripted FakeLLM, no API")
    parser.add_argument("--out", type=Path, default=None, help="count-table JSON (results/)")
    parser.add_argument("--csv", type=Path, default=None, help="tagged cases CSV (results/raw/)")
    args = parser.parse_args(argv)

    report, tagged = run(dataset=args.dataset, dry_run=args.dry_run)

    out = args.out or (common.RESULTS_DIR / "failure_tags.json")
    csv_out = args.csv or (common.RAW_DIR / "failure_tags.csv")
    common.write_report(report, out)
    write_csv(tagged, csv_out)

    print(f"failure tags -> {out}  (cases: {len(tagged)})")
    print(format_count_table(report["counts"]))
    print(f"per-case CSV -> {csv_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
