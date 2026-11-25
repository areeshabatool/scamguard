"""Data shapes passed between pipeline steps.

Model outputs (Facts, Detection) are validated here; a validation failure upstream
must lead to "unsure", never "benign" (invariant 6). Unknown fields are rejected.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Facts(_Strict):
    """Step 4 output: structured fields the extractor pulled from the message."""

    urls: list[str] = Field(default_factory=list)
    phone_numbers: list[str] = Field(default_factory=list)
    claimed_sender: str | None = None
    requested_action: str | None = None


class Detection(_Strict):
    """Step 6 output: the detector's verdict, score and supporting cues."""

    verdict: Literal["benign", "suspicious", "scam"]
    scam_score: float = Field(ge=0.0, le=1.0)
    cues: list[str] = Field(default_factory=list)


class SanitizedMessage(_Strict):
    """Step 3 output: cleaned text, its tagged form for models, and risk markers.

    ai_address_markers are prompt-injection phrases aimed at the AI; reassurance_markers
    are scam "this is 100% safe" phrases aimed at the human. Both only raise risk (step 7).
    """

    text: str
    wrapped: str
    removed_chars: int = Field(ge=0)
    neutralized_tags: int = Field(ge=0)
    ai_address_markers: list[str] = Field(default_factory=list)
    reassurance_markers: list[str] = Field(default_factory=list)


class UrlFinding(_Strict):
    """One parsed URL and what the verifier found wrong with it (never fetched)."""

    url: str
    host: str | None = None
    registrable_domain: str | None = None
    official: bool = False
    flags: list[str] = Field(default_factory=list)
    score: float = Field(ge=0.0, le=1.0, default=0.0)


class ToolReport(_Strict):
    """Step 5 output: deterministic URL and phone checks and the tool risk score.

    tool_score is the max severity across findings; code risk can only raise the
    final risk, never lower it (invariant 3).
    """

    tool_score: float = Field(ge=0.0, le=1.0, default=0.0)
    urls: list[UrlFinding] = Field(default_factory=list)
    phone_numbers: list[str] = Field(default_factory=list)
    flags: list[str] = Field(default_factory=list)


class Advice(_Strict):
    """Step 9 output: chosen safe-action ids and one short Roman Urdu sentence.

    action_ids come from a fixed allowlist and the message only phrases them; the
    sentence must contain no URLs or phone numbers (invariant 7).
    """

    action_ids: list[str] = Field(default_factory=list)
    message: str = ""


class Assessment(_Strict):
    """Step 7 output: final risk and verdict after the code guards.

    final_score = max(model, tool, marker floor); the verdict is the model's verdict
    raised (never lowered) to the risk band (invariants 3 and 5).
    """

    final_score: float = Field(ge=0.0, le=1.0)
    verdict: Literal["benign", "suspicious", "scam"]
    model_verdict: Literal["benign", "suspicious", "scam"]
    band_verdict: Literal["benign", "suspicious", "scam"]
    cues: list[str] = Field(default_factory=list)
    markers: list[str] = Field(default_factory=list)


class Decision(_Strict):
    """Step 8 output: the released verdict after the calibrated gate.

    "benign" is released only when the score is strictly below the conformal threshold;
    otherwise the verdict is "unsure" (invariant 6). Non-benign verdicts pass through.
    """

    verdict: Literal["benign", "suspicious", "scam", "unsure"]
    score: float = Field(ge=0.0, le=1.0)
    released_benign: bool = False
    threshold: float | None = None


class StepTrace(_Strict):
    """One entry in the pipeline's per-step trace."""

    step: int
    name: str
    status: Literal["ok", "skipped", "error"]
    detail: str = ""


class CaseResult(_Strict):
    """The pipeline's output: the decision, short advice and a per-step trace."""

    decision: Literal["benign", "suspicious", "scam", "unsure"]
    score: float = Field(ge=0.0, le=1.0)
    advice: str = ""
    action_ids: list[str] = Field(default_factory=list)
    cues: list[str] = Field(default_factory=list)
    reason: str = ""
    trace: list[StepTrace] = Field(default_factory=list)
