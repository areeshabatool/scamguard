"""Step 7 (CODE): cue and risk check.

Drops any reported cue that is not an exact substring of the message (invariant 5).
Final risk = max(model risk, tool risk); nothing in the message can lower it (invariant 3).
"""
