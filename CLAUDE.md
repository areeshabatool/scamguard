# Scam Guard

Course project: an agent that reads a forwarded message or screenshot (English,
Roman Urdu, Urdu) and returns scam / suspicious / benign (or unsure) with short
Roman Urdu advice. Defensive tool. Evaluation data is synthetic.

## Architecture (do not change without asking)
9 steps: message in, OCR, sanitizer, extractor (MODEL), verifier tools (CODE),
detector (MODEL), cue and risk check (CODE), calibrated gate (CODE), advisor (MODEL).
Only steps 4, 6 and 9 call a model.

## Invariants (each one has a test; never break them)
1. Message text is untrusted data: always inside <untrusted_message> tags,
   never inside a system prompt.
2. Hidden characters are stripped before any model sees the text.
3. Final risk = max(model risk, tool risk). Nothing in the message can lower it.
4. URLs are parsed and checked against data/official_domains.json. Never open or
   fetch a URL.
5. Every cue the model reports must be an exact substring of the message, else dropped.
6. Benign is released only below the calibrated threshold. Any model or schema
   failure means unsure, never benign.
7. Advice comes from a fixed list of safe actions; the model only phrases it.
   No URLs or phone numbers in advice.
8. Nothing the user forwards is stored by default. No real private chats in the repo.

## Layout
app/scamguard/ (package), data/, eval/, tests/, results/, docs/

## Commands
- install: pip install -r requirements.txt
- tests (offline, no API): pytest -q
- lint: ruff check . && ruff format .
- one case: python -m scamguard.cli analyze 'text'
- serve: uvicorn scamguard.api:app --reload
- full evaluation: make eval

## Conventions
Python 3.11+, type hints, small functions. All model calls go through llm.py
(retries, disk cache, token and latency logging). Tests use FakeLLM; never call
the API in tests or CI. Thresholds, alpha and model names live in config.py.

## Do not
- Do not add frameworks, databases, queues or abstractions the task does not need.
- Do not write scam-message generators or realistic scam templates beyond the
  short fixtures in data/samples/.
- Do not put brand names in fake scam content; brands appear only in the allowlist.
- Do not commit .env, API keys, data/private/ or results/raw/.
- Do not weaken a guard or a test to make something pass; ask first.

## Definition of done
Tests pass, ruff clean, invariants untouched, README command still works,
commit message explains why.