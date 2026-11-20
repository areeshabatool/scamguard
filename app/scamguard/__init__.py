"""Scam Guard: classify a forwarded message or screenshot as scam, suspicious, benign or unsure.

Defensive tool. Input may be English, Roman Urdu or Urdu; advice is returned in short Roman Urdu.
The nine pipeline steps live in separate modules; only extractor, detector and advisor call a model.
"""
