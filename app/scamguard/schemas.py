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
