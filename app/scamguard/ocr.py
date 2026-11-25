"""Step 2 (CODE): OCR.

Turns a screenshot into plain text. The resulting text is untrusted, exactly like typed
input. A real OCR engine is pluggable (none is bundled by default); the pipeline consumes
an OCRResult and treats a low-confidence read as "unsure" with a request for a clearer
image. An optional Tesseract adapter is provided but not a hard dependency.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
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


def average_confidence(word_confidences: Sequence[float]) -> float:
    """Mean of per-word confidences (0-100 from Tesseract) as a [0, 1] score.

    Words Tesseract could not score (-1) are ignored. No readable words -> 0.0.
    """
    usable = [c for c in word_confidences if c is not None and c >= 0]
    if not usable:
        return 0.0
    return min(max(sum(usable) / len(usable) / 100.0, 0.0), 1.0)


def tesseract_engine(image: Any, *, lang: str = "eng+urd") -> OCRResult:
    """Optional adapter: transcribe with pytesseract (requires the tesseract binary).

    Install `pytesseract` and `pillow` and the system `tesseract` binary, then pass this as
    the engine, e.g. `api.app.state.ocr_engine = tesseract_engine`. Imported lazily so the
    package has no hard OCR dependency.
    """
    import pytesseract  # optional, not in requirements.txt

    data = pytesseract.image_to_data(image, lang=lang, output_type=pytesseract.Output.DICT)
    words = [w for w in data.get("text", []) if w and w.strip()]
    confs = [float(c) for c in data.get("conf", [])]
    return OCRResult(text=" ".join(words), confidence=average_confidence(confs))
