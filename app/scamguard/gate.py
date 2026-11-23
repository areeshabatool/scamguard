"""Step 8 (CODE): calibrated gate.

Releases "benign" only when final risk is below the calibrated threshold from config.py.
Any model or schema failure yields "unsure", never "benign" (invariant 6).
"""
