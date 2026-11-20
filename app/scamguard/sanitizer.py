"""Step 3 (CODE): sanitizer.

Strips hidden and zero-width characters before any model sees the text (invariant 2),
and wraps the message in <untrusted_message> tags, never in a system prompt (invariant 1).
"""
