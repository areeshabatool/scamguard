"""HTTP API.

Served with: uvicorn scamguard.api:app --reload

  GET  /health   -> {"status": "ok"}
  GET  /         -> the upload UI (web/index.html)
  POST /analyze  -> run the pipeline on `text` or an uploaded `image`, with an optional
                    `session_id` for multi-turn chats. Returns a CaseResult.

Message content is never logged or persisted (invariant 8). Uploaded bytes are read into
memory, transcribed and discarded; sessions keep results in memory only for the chat.
"""

from __future__ import annotations

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from scamguard import config
from scamguard.llm import LLM
from scamguard.ocr import OCRResult
from scamguard.pipeline import Session, analyze
from scamguard.schemas import CaseResult

app = FastAPI(title="Scam Guard", description="Classify forwarded messages as scam/benign.")

# In-memory only; nothing here is written to disk or survives a restart (invariant 8).
_sessions: dict[str, Session] = {}

_WEB_INDEX = config.REPO_ROOT / "web" / "index.html"


def _get_llm() -> LLM:
    llm = getattr(app.state, "llm", None)
    if llm is None:
        from scamguard.llm import build_default_llm

        llm = build_default_llm()
        app.state.llm = llm
    return llm


def _ocr_engine():
    """The OCR engine to use. An explicit app.state.ocr_engine (even None) is honoured;
    otherwise Tesseract is auto-enabled when it is installed, else None (images stay unsure).
    """
    if hasattr(app.state, "ocr_engine"):
        return app.state.ocr_engine
    from scamguard.ocr import tesseract_available, tesseract_engine

    return tesseract_engine if tesseract_available() else None


def _calibration() -> list[float]:
    """Calibration scores for the gate: an explicit app.state override, else the file."""
    override = getattr(app.state, "calibration_scores", None)
    if override is not None:
        return list(override)
    try:
        import json

        data = json.loads(config.CALIBRATION_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    scores = data.get("scores", data) if isinstance(data, dict) else data
    return [float(s) for s in scores] if isinstance(scores, list) else []


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    try:
        return HTMLResponse(_WEB_INDEX.read_text(encoding="utf-8"))
    except OSError as exc:
        raise HTTPException(status_code=404, detail="web UI not found") from exc


@app.post("/analyze", response_model=CaseResult)
async def analyze_endpoint(
    text: str | None = Form(default=None),
    image: UploadFile | None = File(default=None),
    session_id: str | None = Form(default=None),
) -> CaseResult:
    has_image = image is not None and bool(image.filename)
    has_text = bool(text and text.strip())
    if not has_text and not has_image:
        raise HTTPException(status_code=422, detail="provide text or an image")

    llm = _get_llm()

    ocr_result: OCRResult | None = None
    if has_image:
        data = await image.read()  # read into memory, never stored
        engine = _ocr_engine()
        if engine is not None:
            ocr_result = engine(data)
        elif not has_text:
            # No OCR engine and no typed text: an unreadable image -> "send a clearer input".
            ocr_result = OCRResult(text="", confidence=0.0)
        # else: image present but no OCR engine and text was typed -> analyze the typed text
        # rather than discarding it (don't let an unreadable attachment hide a real message).

    # Use the image's OCR text when we have an OCR result; otherwise the typed text.
    text_in = None if ocr_result is not None else text

    if session_id:
        session = _sessions.setdefault(session_id, Session(llm, calibration_scores=_calibration()))
        result = session.analyze(text_in, ocr=ocr_result)
    else:
        result = analyze(text_in, ocr=ocr_result, llm=llm, calibration_scores=_calibration())

    # Echo the transcribed screenshot text back to the UI for display (not stored).
    if ocr_result is not None and ocr_result.text.strip():
        result = result.model_copy(update={"transcript": ocr_result.text})
    return result


# Serve the cinematic front end (static files only) at /cinematic. Display-only: it calls
# the same /analyze endpoint and never touches the pipeline, guards, gate or eval code.
_CINEMATIC_DIR = config.REPO_ROOT / "web" / "cinematic"
if _CINEMATIC_DIR.is_dir():
    app.mount("/cinematic", StaticFiles(directory=str(_CINEMATIC_DIR), html=True), name="cinematic")
