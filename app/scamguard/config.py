"""Central configuration.

Holds the calibrated benign threshold, the conformal alpha, model names and file paths
(e.g. data/official_domains.json). No other module hard-codes these values.
"""

from pathlib import Path

# app/scamguard/config.py -> parents[2] is the repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
OFFICIAL_DOMAINS_PATH = DATA_DIR / "official_domains.json"
