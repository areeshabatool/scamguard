"""Step 7 (CODE): cue and risk check.

Drops any reported cue that is not an exact substring of the message (invariant 5).
Final risk = max(model risk, tool risk); nothing in the message can lower it (invariant 3).

Implemented in guards.py; re-exported here as this step's entry point.
"""

from scamguard.guards import assess, band_verdict, filter_cues, final_risk, raise_only

__all__ = ["assess", "band_verdict", "filter_cues", "final_risk", "raise_only"]
