"""Central configuration.

Holds the calibrated benign threshold, the conformal alpha, model names and file paths
(e.g. data/official_domains.json). No other module hard-codes these values.
"""

from pathlib import Path

# app/scamguard/config.py -> parents[2] is the repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
OFFICIAL_DOMAINS_PATH = DATA_DIR / "official_domains.json"

# --- Model ---
# Every model call goes through llm.py and forces a single tool call for structured
# output. Forced tool_choice is rejected (HTTP 400) on the newest models (Opus 5.5,
# Sonnet 5.5, Fable 5.x), so the default is a model that still accepts it.
MODEL = "claude-opus-4-8"
MAX_TOKENS = 1024

# --- llm.py behavior ---
# Retries are owned by tenacity in llm.py (the SDK client is created with max_retries=0).
LLM_MAX_ATTEMPTS = 4
LLM_RETRY_INITIAL_SECONDS = 1.0
LLM_RETRY_MAX_SECONDS = 20.0
# JSON disk cache (keyed by model, system, user and tools). Under .cache/ (git-ignored).
LLM_CACHE_DIR = REPO_ROOT / ".cache" / "llm"

# --- Risk bands (step 7 guards) ---
# Final risk (in [0, 1]) maps to a verdict band; the model verdict can only be raised.
SUSPICIOUS_BAND = 0.30
SCAM_BAND = 0.70
# An AI-address marker in the message raises risk to at least this floor, never lowers it.
MARKER_RISK_FLOOR = SUSPICIOUS_BAND

# --- Conformal gate (step 8) ---
# Miss-rate budget for releasing "benign". k = floor(ALPHA * (n + 1)).
ALPHA = 0.10
# Optional calibration set of scam scores for the gate (a JSON list). Absent is fine.
CALIBRATION_PATH = DATA_DIR / "calibration.json"

# --- Pipeline input guards (steps 1-2) ---
# Below these, the case is "unsure" with a request for a clearer image (never benign).
MIN_TEXT_CHARS = 4
MIN_OCR_CONFIDENCE = 0.50
