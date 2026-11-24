"""Pipeline wiring for steps 1-9.

Message in -> OCR -> sanitizer -> extractor -> verifier -> detector
-> risk check (guards) -> gate -> advisor.

Fail safe (invariant 6): any model or schema failure yields "unsure", never "benign".
Too little text or a low-confidence OCR read yields "unsure" with a request for a clearer
image, before any model is called. Nothing the user forwards is stored by default
(invariant 8); the Session keeps results in memory only for the length of the chat.
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import ValidationError

from scamguard import config, gate, guards, verifier
from scamguard.agents import SAFE_ACTIONS, advise, detect, extract
from scamguard.llm import LLM, LLMError
from scamguard.ocr import OCRResult
from scamguard.sanitizer import sanitize
from scamguard.schemas import CaseResult, Decision, StepTrace

# Roman Urdu messages for the fail-safe paths.
CLEARER_IMAGE_ADVICE = "Message ya tasveer saaf nahi hai. Barah-e-karam saaf image ya text bhejein."
FAILURE_ADVICE = "Abhi faisla nahi ho saka. Ehtiyatan kisi link par click na karein."

_MODEL_STEPS = (LLMError, ValidationError)


def _fallback_advice(verdict: str) -> tuple[list[str], str]:
    """Safe advice from the fixed list when the advisor model is unavailable (invariant 7)."""
    ids = (
        ["do_not_click", "verify_official"]
        if verdict in ("scam", "suspicious")
        else ["verify_official"]
    )
    return ids, " ".join(SAFE_ACTIONS[i] for i in ids)


def analyze(
    text: str | None = None,
    *,
    ocr: OCRResult | None = None,
    llm: LLM,
    calibration_scores: Sequence[float] = (),
    alpha: float | None = None,
) -> CaseResult:
    """Run the full 9-step pipeline on a message (or an OCR result) and return a CaseResult."""
    trace: list[StepTrace] = []

    def step(n: int, name: str, status: str, detail: str = "") -> None:
        trace.append(StepTrace(step=n, name=name, status=status, detail=detail))

    # Step 1: message in.
    step(1, "message_in", "ok")

    # Step 2: OCR (only when an image was transcribed).
    if ocr is not None:
        text = ocr.text
        if ocr.confidence < config.MIN_OCR_CONFIDENCE:
            step(2, "ocr", "error", f"low confidence {ocr.confidence:.2f}")
            return _insufficient(trace)
        step(2, "ocr", "ok", f"confidence={ocr.confidence:.2f}")
    else:
        step(2, "ocr", "skipped")

    text = (text or "").strip()
    if len(text) < config.MIN_TEXT_CHARS:
        step(3, "sanitize", "skipped", "too little text")
        return _insufficient(trace)

    # Step 3: sanitize (strip hidden chars, wrap as untrusted data, flag AI-address markers).
    sanitized = sanitize(text)
    step(
        3,
        "sanitize",
        "ok",
        f"removed={sanitized.removed_chars} markers={len(sanitized.ai_address_markers)}",
    )

    # Step 4: extractor (MODEL).
    try:
        facts = extract(llm, sanitized)
    except _MODEL_STEPS as exc:
        step(4, "extract", "error", str(exc)[:80])
        return _model_failure(trace)
    step(4, "extract", "ok", f"urls={len(facts.urls)} phones={len(facts.phone_numbers)}")

    # Step 5: verifier (CODE) - parses URLs/phones, never fetches.
    report = verifier.verify(sanitized.text)
    step(5, "verify", "ok", f"tool_score={report.tool_score:.2f} flags={report.flags}")

    # Step 6: detector (MODEL) - a bounded tool loop.
    try:
        detection = detect(llm, sanitized)
    except _MODEL_STEPS as exc:
        step(6, "detect", "error", str(exc)[:80])
        return _model_failure(trace)
    step(6, "detect", "ok", f"verdict={detection.verdict} score={detection.scam_score:.2f}")

    # Step 7: cue and risk check (CODE guards).
    assessment = guards.assess(detection, report, sanitized)
    step(7, "guards", "ok", f"verdict={assessment.verdict} score={assessment.final_score:.2f}")

    # Step 8: calibrated gate (CODE).
    decision = gate.decide(assessment, calibration_scores, alpha=alpha)
    step(8, "gate", "ok", f"verdict={decision.verdict} released={decision.released_benign}")

    # Step 9: advisor (MODEL). A failure here must never leave a released "benign".
    try:
        advice = advise(llm, sanitized, detection)
        action_ids, advice_text = advice.action_ids, advice.message
        step(9, "advise", "ok", f"ids={action_ids}")
    except _MODEL_STEPS as exc:
        action_ids, advice_text = _fallback_advice(decision.verdict)
        step(9, "advise", "error", str(exc)[:80])
        if decision.verdict == "benign":
            decision = Decision(
                verdict="unsure",
                score=decision.score,
                released_benign=False,
                threshold=decision.threshold,
            )

    return CaseResult(
        decision=decision.verdict,
        score=decision.score,
        advice=advice_text,
        action_ids=action_ids,
        cues=assessment.cues,
        trace=trace,
    )


def _insufficient(trace: list[StepTrace]) -> CaseResult:
    return CaseResult(
        decision="unsure",
        score=0.0,
        advice=CLEARER_IMAGE_ADVICE,
        reason="need_clearer_input",
        trace=trace,
    )


def _model_failure(trace: list[StepTrace]) -> CaseResult:
    ids, text = _fallback_advice("suspicious")
    return CaseResult(
        decision="unsure",
        score=1.0,
        advice=text,
        action_ids=ids,
        reason="model_or_schema_failure",
        trace=trace,
    )


class Session:
    """A multi-turn chat. Holds results in memory only; nothing is persisted (invariant 8)."""

    def __init__(
        self,
        llm: LLM,
        *,
        calibration_scores: Sequence[float] = (),
        alpha: float | None = None,
    ) -> None:
        self.llm = llm
        self.calibration_scores = list(calibration_scores)
        self.alpha = alpha
        self.results: list[CaseResult] = []

    def analyze(self, text: str | None = None, *, ocr: OCRResult | None = None) -> CaseResult:
        result = analyze(
            text,
            ocr=ocr,
            llm=self.llm,
            calibration_scores=self.calibration_scores,
            alpha=self.alpha,
        )
        self.results.append(result)
        return result

    @property
    def count(self) -> int:
        return len(self.results)

    def clear(self) -> None:
        """Forget this chat's results (in-memory only)."""
        self.results.clear()
