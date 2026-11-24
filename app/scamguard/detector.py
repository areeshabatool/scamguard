"""Step 6 (MODEL): detector.

Asks the model, via llm.py, for a risk label, a model risk score and the cues it relied on.
Message text stays inside <untrusted_message> tags (invariant 1).

Implemented in agents.py; re-exported here as this step's entry point.
"""

from scamguard.agents import CHECK_DOMAIN_TOOL, SUBMIT_VERDICT_TOOL, detect

__all__ = ["CHECK_DOMAIN_TOOL", "SUBMIT_VERDICT_TOOL", "detect"]
