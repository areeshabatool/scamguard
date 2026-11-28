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


def _locate_tesseract() -> str | None:
    """Find the tesseract binary on PATH or in the usual Windows install locations."""
    import os
    import shutil

    exe = shutil.which("tesseract")
    if exe:
        return exe
    for path in (
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ):
        if os.path.exists(path):
            return path
    return None


def tesseract_available() -> bool:
    """True if pytesseract is importable and the tesseract binary can be found."""
    try:
        import pytesseract
    except ImportError:
        return False
    exe = _locate_tesseract()
    if exe is None:
        return False
    pytesseract.pytesseract.tesseract_cmd = exe
    return True


def tesseract_engine(image: Any, *, lang: str = "eng+urd") -> OCRResult:
    """Optional adapter: transcribe with pytesseract (requires the tesseract binary).

    Accepts a PIL image or raw image bytes (the web API passes bytes). Falls back to
    English if a requested language pack is missing (English also reads Roman Urdu, which
    is Latin script). Imported lazily so the package has no hard OCR dependency.
    """
    import pytesseract  # optional, not in requirements.txt

    exe = _locate_tesseract()
    if exe:
        try:
            pytesseract.pytesseract.tesseract_cmd = exe
        except AttributeError:
            pass  # a mock pytesseract (tests) may not expose .pytesseract.tesseract_cmd

    if isinstance(image, (bytes, bytearray)):
        import io

        from PIL import Image

        image = Image.open(io.BytesIO(image))

    try:
        data = pytesseract.image_to_data(image, lang=lang, output_type=pytesseract.Output.DICT)
    except pytesseract.TesseractError:
        # A requested language pack (e.g. urd) may not be installed; English still works.
        data = pytesseract.image_to_data(image, lang="eng", output_type=pytesseract.Output.DICT)

    words = [w for w in data.get("text", []) if w and w.strip()]
    confs = [float(c) for c in data.get("conf", [])]
    return OCRResult(text=" ".join(words), confidence=average_confidence(confs))
