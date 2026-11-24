"""Step 2 (CODE): OCR.

Turns a screenshot into plain text. The resulting text is untrusted, exactly like typed
input. A real OCR engine is pluggable (none is bundled); the pipeline consumes an
OCRResult and treats a low-confidence read as "unsure" with a request for a clearer image.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class OCRResult:
    """Transcribed text and the engine's confidence in it, in [0, 1]."""

    text: str
    confidence: float


def transcribe(image: Any, *, engine: Callable[[Any], OCRResult] | None = None) -> OCRResult:
    """Transcribe an image using the given engine. No engine is bundled (offline by default)."""
    if engine is None:
        raise RuntimeError("no OCR engine configured; pass engine=<callable returning OCRResult>")
    return engine(image)
