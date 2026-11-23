"""Step 3 (CODE): sanitizer.

Strips hidden and zero-width characters before any model sees the text (invariant 2),
and wraps the message in <untrusted_message> tags, never in a system prompt (invariant 1).
Any copy of the tag inside the message is neutralized so the text cannot break out.
Phrases that address an AI are reported as markers; they can only raise risk, never lower it.
"""

import re
import unicodedata

from scamguard.schemas import SanitizedMessage

TAG = "untrusted_message"
NEUTRALIZED = "[tag removed]"

# Unicode categories with no visible meaning in a chat message:
# Cf format (zero-width, bidi controls, BOM, soft hyphen, tag chars U+E0000-E007F),
# Cc controls, Co private use, Cs surrogates, Cn unassigned.
_HIDDEN_CATEGORIES = {"Cf", "Cc", "Co", "Cs", "Cn"}
_KEEP = {"\n", "\t"}
# Invisible characters outside those categories: variation selectors and Hangul fillers.
_EXTRA_HIDDEN = re.compile("[︀-️\U000e0100-\U000e01efᅟᅠㅤﾠ]")

# Opening or closing tag in any case/spacing; the closing ">" is optional.
_TAG_PATTERN = re.compile(rf"<\s*/?\s*{TAG}\b[^<>]*>?", re.IGNORECASE)

_AI_ADDRESS_PATTERNS = [
    # English
    r"ignore\s+(?:all\s+)?(?:the\s+)?(?:previous|prior|above|earlier)\s+instructions?",
    r"disregard\s+(?:all\s+)?(?:the\s+)?(?:previous|prior|above|earlier)\s+instructions?",
    r"(?:you\s+are|you're|as)\s+an?\s+(?:ai|assistant|language\s+model|bot)\b",
    r"\b(?:dear|hey|hello)\s+(?:ai|assistant|bot|model)\b",
    r"system\s+prompt",
    r"(?:classify|mark|label|rate)\s+(?:this|it)(?:\s+message)?\s+as\s+(?:safe|benign|legit\w*)",
    # Roman Urdu
    r"(?:pichli|pehli|purani|upar\s+wali)\s+hidayat\s+(?:ko\s+)?(?:ignore|nazar\s*andaz)",
    r"hidayat\s+(?:ko\s+)?(?:ignore|nazar\s*andaz)\s+kar",
    r"(?:tum|aap)\s+(?:ek\s+)?(?:ai|bot|assistant)\s+(?:ho|hain)",
    r"\b(?:ai|bot|assistant)\s+(?:ko|se)\s+(?:kaho|bolo|batao)",
    r"(?:is|iss)\s+(?:message|msg)\s+ko\s+(?:safe|benign|theek)\s+(?:likho|batao|kaho|mark\s+karo)",
    r"(?:scam|fraud)\s+(?:nahi|nahin)\s+(?:likho|batao|kaho)",
]
_AI_ADDRESS = re.compile("|".join(f"(?:{p})" for p in _AI_ADDRESS_PATTERNS), re.IGNORECASE)


def _is_hidden(ch: str) -> bool:
    if ch in _KEEP:
        return False
    return unicodedata.category(ch) in _HIDDEN_CATEGORIES or bool(_EXTRA_HIDDEN.match(ch))


def _strip_hidden(text: str) -> tuple[str, int]:
    kept = "".join(ch for ch in text if not _is_hidden(ch))
    return kept, len(text) - len(kept)


def _clean_counted(text: str) -> tuple[str, int]:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Strip, normalize, strip again: NFKC can fold lookalikes into new hidden sequences.
    text, first = _strip_hidden(text)
    text, second = _strip_hidden(unicodedata.normalize("NFKC", text))
    return text, first + second


def clean(text: str) -> str:
    """NFKC-normalize and strip hidden characters. Idempotent."""
    return _clean_counted(text)[0]


def neutralize_tags(text: str) -> tuple[str, int]:
    """Replace any <untrusted_message> open/close tag inside the text."""
    return _TAG_PATTERN.subn(NEUTRALIZED, text)


def _tagged(body: str) -> str:
    return f"<{TAG}>\n{body}\n</{TAG}>"


def wrap(text: str) -> str:
    """Return the message as tagged untrusted data, for a user turn (never a system prompt)."""
    return _tagged(neutralize_tags(clean(text))[0])


def find_ai_address(text: str) -> list[str]:
    """Return phrases that address an AI, as exact substrings of the given text."""
    return [m.group(0) for m in _AI_ADDRESS.finditer(text)]


def sanitize(raw: str) -> SanitizedMessage:
    """Run the full step 3 on raw message text."""
    text, removed = _clean_counted(raw)
    body, count = neutralize_tags(text)
    return SanitizedMessage(
        text=text,
        wrapped=_tagged(body),
        removed_chars=removed,
        neutralized_tags=count,
        ai_address_markers=find_ai_address(text),
    )
