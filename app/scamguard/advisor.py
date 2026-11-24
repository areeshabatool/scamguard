"""Step 9 (MODEL): advisor.

Selects safe actions from a fixed list; the model only phrases them in short Roman Urdu.
Advice never contains URLs or phone numbers (invariant 7).

Implemented in agents.py; re-exported here as this step's entry point.
"""

from scamguard.agents import SAFE_ACTIONS, advise

__all__ = ["SAFE_ACTIONS", "advise"]
