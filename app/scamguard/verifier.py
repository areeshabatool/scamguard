"""Step 5 (CODE): verifier tools.

Parses URLs and checks their domains against data/official_domains.json (invariant 4).
URLs are only ever parsed, never opened or fetched. Phone numbers are normalized to a
canonical Pakistani form. Produces a deterministic tool risk score in [0, 1]; this score
can only raise the final risk, never lower it (invariant 3).
"""

import ipaddress
import json
import re
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlsplit

from scamguard import config
from scamguard.schemas import ToolReport, UrlFinding

# ---------------------------------------------------------------------------
# Reason codes and their severities. tool_score is the max over all findings.
# ---------------------------------------------------------------------------
FLAG_USERINFO = "userinfo"  # http://bank.com@evil.com — the real host is hidden
FLAG_LOOKALIKE = "lookalike"  # typosquat / brand impersonation of an official domain
FLAG_IP_HOST = "ip_host"  # numeric host instead of a name
FLAG_PUNYCODE = "punycode"  # xn-- IDN host, a common homoglyph trick
FLAG_SHORTENER = "shortener"  # destination is hidden behind a link shortener

_SEVERITY: dict[str, float] = {
    FLAG_USERINFO: 0.9,
    FLAG_LOOKALIKE: 0.9,
    FLAG_IP_HOST: 0.8,
    FLAG_PUNYCODE: 0.8,
    FLAG_SHORTENER: 0.6,
}

# Common URL shorteners (destination is opaque, so always suspicious).
_SHORTENERS = frozenset(
    {
        "bit.ly",
        "tinyurl.com",
        "t.co",
        "goo.gl",
        "ow.ly",
        "buff.ly",
        "is.gd",
        "cutt.ly",
        "rb.gy",
        "shorturl.at",
        "rebrand.ly",
        "bit.do",
        "t.ly",
        "tiny.cc",
        "lnkd.in",
        "trib.al",
    }
)

# Two-label public suffixes we care about, so "foo.com.pk" -> registrable "foo.com.pk".
_TWO_LEVEL_SUFFIXES = frozenset(
    {
        "com.pk",
        "net.pk",
        "org.pk",
        "edu.pk",
        "gov.pk",
        "co.uk",
        "org.uk",
        "gov.uk",
        "com.au",
        "co.in",
    }
)

# URL candidates: explicit scheme, www. prefix, bare domain with a TLD, or a bare IPv4.
# A TLD is alphabetic, so a dotted date like "2024.01.05" never matches as a bare domain,
# and only four dotted octets match as an IP.
_URL_RE = re.compile(
    r"""(?<![\w@./-])(?:
        https?://[^\s<>"'`\]}),]+
      | www\.[^\s<>"'`\]}),]+
      | (?:\d{1,3}\.){3}\d{1,3}(?::\d+)?(?:[/?#][^\s<>"'`\]}),]*)?
      | (?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z]{2,}(?::\d+)?(?:[/?#][^\s<>"'`\]}),]*)?
    )""",
    re.IGNORECASE | re.VERBOSE,
)

# Pakistani mobile numbers: optional +92 / 0092 / 92 / 0 prefix, then a 3xx subscriber
# number of exactly ten digits. Separators between groups are optional, so 03xx-xxxxxxx,
# +92 3xx xxxxxxx and 03xxxxxxxxx all match. The leading boundary rejects longer digit
# runs, and the fixed 10-digit "3.........." shape rejects date-like strings.
_PHONE_RE = re.compile(
    r"(?<![\d+])(?:(?:\+92|0092|92)[\s.\-]?|0)3\d{2}[\s.\-]?\d{3}[\s.\-]?\d{4}(?!\d)"
)

_TRAILING = ".,;:!?)]}'\"`"


@lru_cache(maxsize=8)
def load_official_domains(path: str | None = None) -> frozenset[str]:
    """Load the allowlist of official registrable domains (lowercased)."""
    p = Path(path) if path else config.OFFICIAL_DOMAINS_PATH
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return frozenset()
    return frozenset(d.strip().lower().rstrip(".") for d in data.get("domains", []) if d.strip())


def extract_urls(text: str) -> list[str]:
    """Return URL-like substrings found in the text, trailing punctuation trimmed.

    Order is preserved and exact duplicates are collapsed. Nothing is fetched.
    """
    seen: set[str] = set()
    out: list[str] = []
    for m in _URL_RE.finditer(text):
        url = m.group(0).rstrip(_TRAILING)
        if not url or url in seen:
            continue
        seen.add(url)
        out.append(url)
    return out


def _registrable(host: str) -> str:
    labels = host.split(".")
    if len(labels) >= 3 and ".".join(labels[-2:]) in _TWO_LEVEL_SUFFIXES:
        return ".".join(labels[-3:])
    return ".".join(labels[-2:]) if len(labels) >= 2 else host


def _sld(registrable: str) -> str:
    """Second-level (brand) label, e.g. 'hbl' from 'hbl.com' or 'easypaisa.com.pk'."""
    return registrable.split(".", 1)[0]


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _is_lookalike(host: str, official: frozenset[str]) -> bool:
    """Heuristic impersonation / typosquat check against the allowlist.

    True when the host is not official but closely mimics an official domain: the brand
    label appears as a whole label or hyphen token, is a prefix of the host's own label,
    or the registrable domain is one edit away from an official one.
    """
    labels = host.split(".")
    cand_reg = _registrable(host)
    cand_sld = _sld(cand_reg)
    hyphen_tokens = set(cand_sld.split("-"))
    for off in official:
        off_sld = _sld(off)
        if cand_reg == off:
            return False  # exact official domain; handled elsewhere
        # Brand label reused under a different registrable domain (subdomain or other TLD).
        if off_sld in labels:
            return True
        # Brand label as a hyphen-delimited token, e.g. "hbl-verify".
        if off_sld in hyphen_tokens:
            return True
        # Longer brand used as a prefix, e.g. "easypaisa-login".
        if len(off_sld) >= 4 and cand_sld.startswith(off_sld) and cand_sld != off_sld:
            return True
        # One-character typo of the whole registrable domain, e.g. "hb1.com".
        if len(off_sld) >= 3 and _levenshtein(cand_reg, off) == 1:
            return True
    return False


def classify_url(url: str, official: frozenset[str]) -> UrlFinding:
    """Parse one URL and flag structural tricks. Never fetches anything."""
    try:
        parsed = urlsplit(url if "://" in url else f"http://{url}")
        host = (parsed.hostname or "").lower().rstrip(".")
        has_userinfo = parsed.username is not None
    except ValueError:
        # A malformed URL (e.g. stray brackets the message contained, which newer
        # urllib reads as a bad IPv6 host) is unparseable. Treat it as having no host
        # rather than letting it crash the pipeline (invariant: fail safe, never crash).
        return _finish(url, None, None, False, [])

    flags: list[str] = []
    if has_userinfo:
        flags.append(FLAG_USERINFO)

    if not host:
        return _finish(url, host or None, None, False, flags)

    is_ip = False
    try:
        ipaddress.ip_address(host)
        is_ip = True
    except ValueError:
        is_ip = False

    if is_ip:
        flags.append(FLAG_IP_HOST)
        return _finish(url, host, None, False, flags)

    registrable = _registrable(host)

    if any(label.startswith("xn--") for label in host.split(".")):
        flags.append(FLAG_PUNYCODE)

    is_official = host == "" or any(host == off or host.endswith("." + off) for off in official)

    if registrable in _SHORTENERS:
        flags.append(FLAG_SHORTENER)

    if not is_official and _is_lookalike(host, official):
        flags.append(FLAG_LOOKALIKE)

    return _finish(url, host, registrable, is_official, flags)


def _finish(
    url: str,
    host: str | None,
    registrable: str | None,
    official: bool,
    flags: list[str],
) -> UrlFinding:
    # De-duplicate flags while preserving order.
    seen: set[str] = set()
    flags = [f for f in flags if not (f in seen or seen.add(f))]
    score = 0.0 if official else max((_SEVERITY.get(f, 0.0) for f in flags), default=0.0)
    return UrlFinding(
        url=url,
        host=host,
        registrable_domain=registrable,
        official=official,
        flags=flags,
        score=score,
    )


def normalize_pk_phone(raw: str) -> str | None:
    """Return a matched Pakistani mobile number in canonical +92XXXXXXXXXX form, or None."""
    digits = re.sub(r"\D", "", raw)
    if digits.startswith("0092"):
        digits = digits[4:]
    elif digits.startswith("92") and len(digits) == 12:
        digits = digits[2:]
    elif digits.startswith("0") and len(digits) == 11:
        digits = digits[1:]
    if len(digits) == 10 and digits.startswith("3"):
        return "+92" + digits
    return None


def find_phone_numbers(text: str) -> list[str]:
    """Return normalized Pakistani mobile numbers, de-duplicated, in order of appearance."""
    out: list[str] = []
    seen: set[str] = set()
    for m in _PHONE_RE.finditer(text):
        num = normalize_pk_phone(m.group(0))
        if num and num not in seen:
            seen.add(num)
            out.append(num)
    return out


def verify(text: str, official: frozenset[str] | None = None) -> ToolReport:
    """Run the full step 5 on message text. Deterministic; never touches the network."""
    allowlist = official if official is not None else load_official_domains()
    findings = [classify_url(u, allowlist) for u in extract_urls(text)]
    phones = find_phone_numbers(text)

    tool_score = max((f.score for f in findings), default=0.0)
    agg_seen: set[str] = set()
    agg_flags = [
        f for find in findings for f in find.flags if not (f in agg_seen or agg_seen.add(f))
    ]
    return ToolReport(
        tool_score=tool_score,
        urls=findings,
        phone_numbers=phones,
        flags=agg_flags,
    )
