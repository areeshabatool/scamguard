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
    """Step 3 output: cleaned text, its tagged form for models, and AI-address markers."""

    text: str
    wrapped: str
    removed_chars: int = Field(ge=0)
    neutralized_tags: int = Field(ge=0)
    ai_address_markers: list[str] = Field(default_factory=list)


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
