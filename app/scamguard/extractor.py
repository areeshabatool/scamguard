"""Step 4 (MODEL): extractor.

Asks the model, via llm.py, to pull structured fields (URLs, phone numbers, claimed sender,
requested action) out of the sanitized, tagged message. Output is schema-validated.

Implemented in agents.py; re-exported here as this step's entry point.
"""

from scamguard.agents import EXTRACT_TOOL, extract

__all__ = ["EXTRACT_TOOL", "extract"]
