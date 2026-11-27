"""Verifier tests for invariant 4 (URLs parsed against the allowlist, never fetched).

Also covers the tool risk score, the required link cases (official, lookalike, shortener)
and that date-like strings are not mistaken for Pakistani phone numbers.
"""

import socket

import pytest

from scamguard.verifier import (
    FLAG_IP_HOST,
    FLAG_LOOKALIKE,
    FLAG_PUNYCODE,
    FLAG_SHORTENER,
    FLAG_USERINFO,
    classify_url,
    extract_urls,
    find_phone_numbers,
    normalize_pk_phone,
    verify,
)

# A fixed allowlist so tests do not depend on the shipped data/official_domains.json.
OFFICIAL = frozenset({"hbl.com", "easypaisa.com.pk", "nadra.gov.pk"})


# ---------- invariant 4: never fetch or open a URL ----------


def test_verify_never_opens_a_socket(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*args: object, **kwargs: object) -> None:
        raise AssertionError("verifier must never open a network connection")

    monkeypatch.setattr(socket, "socket", boom)
    monkeypatch.setattr(socket, "create_connection", boom)
    report = verify("Login now at http://192.168.0.1/hbl and https://bit.ly/x", OFFICIAL)
    assert report.tool_score > 0  # it still analyzed the URLs


# ---------- official links ----------


@pytest.mark.parametrize(
    "url",
    [
        "https://www.hbl.com/login",
        "https://hbl.com",
        "http://ibank.hbl.com/dashboard",
        "https://easypaisa.com.pk/app",
        "https://nadra.gov.pk/track",
    ],
)
def test_official_links_are_clean(url: str) -> None:
    finding = classify_url(url, OFFICIAL)
    assert finding.official is True
    assert finding.flags == []
    assert finding.score == 0.0


def test_official_link_gives_zero_tool_score() -> None:
    report = verify("Apna account yahan verify karein: https://www.hbl.com/verify", OFFICIAL)
    assert report.tool_score == 0.0
    assert report.urls[0].official is True


# ---------- lookalike links ----------


@pytest.mark.parametrize(
    "url",
    [
        "http://hbl.com.secure-login.net/verify",  # brand as subdomain of another domain
        "https://hbl-verify.com/login",  # brand as a hyphen token
        "http://hb1.com",  # one-character typosquat (l -> 1)
        "https://easypaisa-login.com/claim",  # brand used as a prefix
        "http://login.nadra.gov.pk.fake.co/track",  # gov brand under a foreign domain
    ],
)
def test_lookalike_links_are_flagged(url: str) -> None:
    finding = classify_url(url, OFFICIAL)
    assert finding.official is False
    assert FLAG_LOOKALIKE in finding.flags
    assert finding.score == pytest.approx(0.9)


def test_lookalike_raises_tool_score() -> None:
    report = verify("URGENT: http://hbl.com.secure-login.net/verify", OFFICIAL)
    assert report.tool_score == pytest.approx(0.9)
    assert FLAG_LOOKALIKE in report.flags


# ---------- shortener links ----------


@pytest.mark.parametrize(
    "url",
    [
        "https://bit.ly/3abcXYZ",
        "http://tinyurl.com/claim-prize",
        "https://t.co/abcd",
        "https://cutt.ly/xyz",
    ],
)
def test_shortener_links_are_flagged(url: str) -> None:
    finding = classify_url(url, OFFICIAL)
    assert FLAG_SHORTENER in finding.flags
    assert finding.score == pytest.approx(0.6)


# ---------- other URL tricks ----------


def test_ip_host_is_flagged() -> None:
    finding = classify_url("http://192.168.0.1/login", OFFICIAL)
    assert FLAG_IP_HOST in finding.flags
    assert finding.score == pytest.approx(0.8)


def test_punycode_host_is_flagged() -> None:
    finding = classify_url("http://xn--pple-43d.com/login", OFFICIAL)
    assert FLAG_PUNYCODE in finding.flags
    assert finding.score == pytest.approx(0.8)


def test_userinfo_trick_is_flagged() -> None:
    # The real host is evil.com; hbl.com is only the userinfo part.
    finding = classify_url("http://hbl.com@evil.com/login", OFFICIAL)
    assert finding.host == "evil.com"
    assert finding.official is False
    assert FLAG_USERINFO in finding.flags
    assert finding.score == pytest.approx(0.9)


def test_malformed_bracket_url_does_not_crash() -> None:
    # A stray bracket makes newer urllib read a bad IPv6 host and raise; the verifier
    # must treat it as unparseable, never crash (it fed a message's "[PLATFORM]" text).
    for bad in ("[PLATFORM].com/login", "http://[BANK]/verify", "http://a[b].com"):
        finding = classify_url(bad, OFFICIAL)
        assert finding.official is False
        assert finding.score == pytest.approx(0.0)


# ---------- scoring and extraction ----------


def test_no_url_means_zero_score() -> None:
    report = verify("Salam, kal 5 baje milte hain.", OFFICIAL)
    assert report.urls == []
    assert report.tool_score == 0.0


def test_tool_score_is_the_max_across_urls() -> None:
    text = "Safe: https://hbl.com and short: https://bit.ly/x"
    report = verify(text, OFFICIAL)
    assert len(report.urls) == 2
    assert report.tool_score == pytest.approx(0.6)  # shortener dominates the official link


def test_email_addresses_are_not_extracted_as_urls() -> None:
    assert extract_urls("Mujhe mail karein ali@example.com par") == []


def test_trailing_punctuation_is_trimmed() -> None:
    assert extract_urls("Visit https://hbl.com/login.") == ["https://hbl.com/login"]


# ---------- phone numbers vs date-like strings ----------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("03001234567", "+923001234567"),
        ("0300-1234567", "+923001234567"),
        ("0300 123 4567", "+923001234567"),
        ("+92 300 1234567", "+923001234567"),
        ("+923001234567", "+923001234567"),
        ("0092-300-1234567", "+923001234567"),
    ],
)
def test_pk_mobile_numbers_normalized(raw: str, expected: str) -> None:
    found = find_phone_numbers(f"Call me at {raw} please")
    assert found == [expected]


@pytest.mark.parametrize(
    "date",
    [
        "2024-01-05",
        "05/01/2024",
        "2024.01.05",
        "12-05-2024",
        "03-01-2024",  # a date that starts with 03 but is too short for a mobile number
        "Order #20240105 shipped",
        "Balance 4500.00 rupees",
    ],
)
def test_date_like_strings_are_not_phone_numbers(date: str) -> None:
    assert find_phone_numbers(f"Reference: {date}") == []


def test_landline_is_not_treated_as_mobile() -> None:
    # Only 03xx mobile numbers are normalized; a Lahore landline must not match.
    assert find_phone_numbers("Office: 042-35678901") == []


def test_normalize_pk_phone_rejects_non_mobile() -> None:
    assert normalize_pk_phone("12-05-2024") is None
    assert normalize_pk_phone("042-35678901") is None
    assert normalize_pk_phone("0300-1234567") == "+923001234567"


def test_phone_numbers_deduped_in_order() -> None:
    report = verify("Call 03001234567 or 0300-1234567 or 03217654321", OFFICIAL)
    assert report.phone_numbers == ["+923001234567", "+923217654321"]
