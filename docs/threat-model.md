# Threat model

Scam Guard reads **untrusted** forwarded messages and screenshots. The message author is
the adversary; they may also have read our prompts. The design assumes the message can try
to manipulate the model, hide content, or impersonate a sender.

## Assets and goals

- Give a trustworthy verdict (scam / suspicious / benign / unsure).
- Never release `benign` for something risky, and never take a dangerous action (open a
  link, leak a prompt, store private content).

## Core invariants (each has a test; see `tests/`)

| # | Invariant | Enforced by |
| - | --------- | ----------- |
| 1 | Message text is untrusted data: always inside `<untrusted_message>` tags in the **user** turn, never in a system prompt. | `sanitizer.wrap`, fixed system constants in `agents.py` |
| 2 | Hidden/zero-width characters are stripped before any model sees the text. | `sanitizer.clean` |
| 3 | Final risk = `max(model, tool, marker floor)`; nothing in the message can lower it. | `guards.final_risk`, `guards.assess` |
| 4 | URLs are parsed and checked against the allowlist; never opened or fetched. | `verifier` (parse-only; a test asserts no socket opens) |
| 5 | Every model-reported cue must be an exact substring of the message, else dropped. | `guards.filter_cues` |
| 6 | Benign is released only below the calibrated threshold; any model/schema failure → unsure. | `gate.decide`, `pipeline.analyze` try/except |
| 7 | Advice comes from a fixed safe-action list; no URLs or phone numbers. | `agents.advise`, `_strip_contact_info` |
| 8 | Nothing the user forwards is stored by default; no real private chats in the repo. | in-memory `Session`; `data/private/`, `results/raw/` git-ignored |

## Attacker techniques and defenses

- **Prompt injection** ("ignore previous instructions", "mark this as benign") — the text
  is wrapped as untrusted data; such phrases are detected as `ai_address_markers` and
  raise a risk floor (never lower it).
- **Reassurance / social engineering** ("100% genuine, not a scam") — detected as
  `reassurance_markers` and treated as a risk floor, so a persuaded model still cannot
  release benign for a message that insists it is safe.
- **Hidden characters / homoglyphs** — stripped (Cf/Cc/Co/Cs/Cn, variation selectors) and
  NFKC-normalized; URL homoglyphs surface as punycode flags in the verifier.
- **Lookalike / obfuscated URLs** — the verifier flags lookalike, shortener, IP-host,
  punycode and userinfo tricks; the tool score floors the final risk independently of the
  model, and no URL is ever fetched.
- **A compromised or fooled model** — code floors (tool, markers) and the fail-to-unsure
  boundaries mean the model cannot, on its own, downgrade a risky message to benign. See
  `tests/test_security.py`.

## Residual risk

A scam with **no URL/phone and no detected marker phrase**, where the model is persuaded
to answer benign, can still be released as benign. The markers are a denylist and can be
evaded. Mitigations in place: the hardened detector prompt and the conformal gate (which
requires the score below a calibrated threshold). This is the main area for future work.
