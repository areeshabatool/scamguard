"""Tests for the model gateway. No real API call is ever made (injected stub client)."""

import logging
from pathlib import Path
from types import SimpleNamespace

import anthropic
import httpx2
import pytest

import scamguard
from scamguard import config
from scamguard.llm import AnthropicLLM, FakeLLM, LLMError, cache_key

TOOL = {
    "name": "extract_facts",
    "description": "Pull structured fields from the message.",
    "input_schema": {
        "type": "object",
        "properties": {"urls": {"type": "array", "items": {"type": "string"}}},
        "required": ["urls"],
        "additionalProperties": False,
    },
}


# ---------- a minimal stand-in for anthropic.Anthropic ----------


def _message(tool_name, data, *, stop="tool_use", in_tok=12, out_tok=3, with_block=True):
    block = SimpleNamespace(type="tool_use", name=tool_name, input=data)
    return SimpleNamespace(
        content=[block] if with_block else [],
        stop_reason=stop,
        usage=SimpleNamespace(input_tokens=in_tok, output_tokens=out_tok),
    )


class _StubMessages:
    def __init__(self, script):
        self.script = list(script)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class _StubClient:
    def __init__(self, script):
        self.messages = _StubMessages(script)


def _llm(script, tmp_path: Path, **kw) -> AnthropicLLM:
    return AnthropicLLM(
        model=kw.get("model", "test-model"),
        max_tokens=kw.get("max_tokens", 99),
        cache_dir=tmp_path,
        client=_StubClient(script),
    )


# ---------- cache key ----------


def test_cache_key_is_deterministic() -> None:
    a = cache_key("m", "sys", "usr", [TOOL])
    b = cache_key("m", "sys", "usr", [TOOL])
    assert a == b and len(a) == 64


def test_cache_key_changes_with_inputs() -> None:
    base = cache_key("m", "sys", "usr", [TOOL])
    assert cache_key("m2", "sys", "usr", [TOOL]) != base
    assert cache_key("m", "sys2", "usr", [TOOL]) != base
    assert cache_key("m", "sys", "usr2", [TOOL]) != base


# ---------- AnthropicLLM (stubbed) ----------


def test_forces_tool_choice_and_parses_structured_output(tmp_path: Path) -> None:
    llm = _llm([_message("extract_facts", {"urls": ["x"]})], tmp_path)
    res = llm.call_tool(system="SYS", user="USER", tool=TOOL)

    assert res.data == {"urls": ["x"]}
    assert res.cached is False
    assert (res.input_tokens, res.output_tokens) == (12, 3)

    sent = llm.client.messages.calls[0]
    assert sent["tool_choice"] == {"type": "tool", "name": "extract_facts"}
    assert sent["model"] == "test-model" and sent["max_tokens"] == 99
    assert sent["system"] == "SYS"
    assert sent["messages"] == [{"role": "user", "content": "USER"}]
    assert sent["tools"][0]["name"] == "extract_facts"


def test_disk_cache_serves_second_call(tmp_path: Path) -> None:
    # Only one scripted response: a second real API call would raise IndexError.
    llm = _llm([_message("extract_facts", {"urls": ["x"]})], tmp_path)
    r1 = llm.call_tool(system="S", user="U", tool=TOOL)
    r2 = llm.call_tool(system="S", user="U", tool=TOOL)

    assert r1.cached is False and r2.cached is True
    assert r2.data == r1.data
    assert len(llm.client.messages.calls) == 1  # second served from disk
    assert list(tmp_path.glob("*.json"))  # a cache file was written


def test_refusal_raises_llm_error(tmp_path: Path) -> None:
    llm = _llm([_message("extract_facts", {}, stop="refusal", with_block=False)], tmp_path)
    with pytest.raises(LLMError):
        llm.call_tool(system="S", user="U", tool=TOOL)


def test_missing_tool_call_raises_llm_error(tmp_path: Path) -> None:
    llm = _llm([_message("extract_facts", {}, stop="end_turn", with_block=False)], tmp_path)
    with pytest.raises(LLMError):
        llm.call_tool(system="S", user="U", tool=TOOL)


def test_logs_tokens_and_latency(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    llm = _llm([_message("extract_facts", {"urls": []}, in_tok=12, out_tok=3)], tmp_path)
    with caplog.at_level(logging.INFO, logger="scamguard.llm"):
        llm.call_tool(system="S", user="U", tool=TOOL)
    assert any(
        "in=12" in r.getMessage() and "out=3" in r.getMessage() and "latency=" in r.getMessage()
        for r in caplog.records
    )


# ---------- tenacity retries ----------


def _req() -> httpx2.Request:
    return httpx2.Request("POST", "https://api.anthropic.com/v1/messages")


def test_retries_retryable_error_then_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Make backoff instant so the test is fast; _create reads these at call time.
    monkeypatch.setattr(config, "LLM_RETRY_INITIAL_SECONDS", 0.0)
    monkeypatch.setattr(config, "LLM_RETRY_MAX_SECONDS", 0.0)

    conn_err = anthropic.APIConnectionError(request=_req())
    llm = _llm([conn_err, _message("extract_facts", {"urls": ["ok"]})], tmp_path)
    res = llm.call_tool(system="S", user="U", tool=TOOL)

    assert res.data == {"urls": ["ok"]}
    assert len(llm.client.messages.calls) == 2  # one failure, one success


def test_does_not_retry_client_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "LLM_RETRY_INITIAL_SECONDS", 0.0)
    monkeypatch.setattr(config, "LLM_RETRY_MAX_SECONDS", 0.0)

    bad = anthropic.BadRequestError("bad", response=httpx2.Response(400, request=_req()), body=None)
    llm = _llm([bad], tmp_path)
    with pytest.raises(anthropic.BadRequestError):
        llm.call_tool(system="S", user="U", tool=TOOL)
    assert len(llm.client.messages.calls) == 1  # a 400 is not retried


# ---------- FakeLLM ----------


def test_fake_llm_responses_in_order() -> None:
    fake = FakeLLM(responses=[{"a": 1}, {"a": 2}])
    assert fake.call_tool(system="s", user="u", tool=TOOL).data == {"a": 1}
    assert fake.call_tool(system="s", user="u", tool=TOOL).data == {"a": 2}


def test_fake_llm_by_tool_and_default() -> None:
    fake = FakeLLM(by_tool={"extract_facts": {"urls": []}}, default={"fallback": True})
    assert fake.call_tool(system="s", user="u", tool=TOOL).data == {"urls": []}
    other = {"name": "detect", "description": "", "input_schema": {}}
    assert fake.call_tool(system="s", user="u", tool=other).data == {"fallback": True}


def test_fake_llm_callable() -> None:
    fake = FakeLLM(default=lambda system, user, tool: {"echo": user})
    assert fake.call_tool(system="s", user="hello", tool=TOOL).data == {"echo": "hello"}


def test_fake_llm_raises_scripted_exception() -> None:
    fake = FakeLLM(responses=[LLMError("boom")])
    with pytest.raises(LLMError):
        fake.call_tool(system="s", user="u", tool=TOOL)


def test_fake_llm_records_calls() -> None:
    fake = FakeLLM(default={"ok": True})
    fake.call_tool(system="SYS", user="USR", tool=TOOL)
    assert fake.calls == [{"system": "SYS", "user": "USR", "tool": "extract_facts"}]


def test_fake_llm_without_script_raises() -> None:
    with pytest.raises(LLMError):
        FakeLLM().call_tool(system="s", user="u", tool=TOOL)


# ---------- invariant: only llm.py talks to the API ----------


def test_only_llm_module_touches_the_api() -> None:
    pkg_dir = Path(scamguard.__file__).parent
    offenders = []
    for py in pkg_dir.glob("*.py"):
        if py.name == "llm.py":
            continue
        text = py.read_text(encoding="utf-8")
        if "import anthropic" in text or "messages.create" in text:
            offenders.append(py.name)
    assert offenders == [], f"these modules call the API directly: {offenders}"
