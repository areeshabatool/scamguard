"""Pipeline wiring for steps 1-9.

Message in -> OCR -> sanitizer -> extractor -> verifier -> detector
-> risk check -> gate -> advisor.
Nothing the user forwards is stored by default (invariant 8).
"""
