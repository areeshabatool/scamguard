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
        from scamguard.llm import AnthropicLLM

        llm = AnthropicLLM()
        app.state.llm = llm
    return llm


def _calibration() -> list[float]:
    return list(getattr(app.state, "calibration_scores", []))


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
        engine = getattr(app.state, "ocr_engine", None)
        # No OCR engine configured -> a zero-confidence read -> "unsure, send clearer input".
        ocr_result = engine(data) if engine is not None else OCRResult(text="", confidence=0.0)

    # When an image is present we use its OCR text; otherwise the typed text.
    text_in = None if ocr_result is not None else text

    if session_id:
        session = _sessions.setdefault(session_id, Session(llm, calibration_scores=_calibration()))
        return session.analyze(text_in, ocr=ocr_result)

    return analyze(text_in, ocr=ocr_result, llm=llm, calibration_scores=_calibration())
