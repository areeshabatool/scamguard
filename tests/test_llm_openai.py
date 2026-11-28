"""OpenAI backend tests. A fake client stands in for the network, so these never call out.

Skipped entirely if the openai package is not installed (it is an opt-in backend).
"""

import json
from types import SimpleNamespace

import pytest

pytest.importorskip("openai")

from scamguard.agents import SUBMIT_VERDICT_TOOL  # noqa: E402
from scamguard.llm import (  # noqa: E402
    LLMError,
    OpenAILLM,
    _assistant_to_openai,
    _messages_to_openai,
    _tool_choice_to_openai,
    _tool_to_openai,
    build_default_llm,
    tool_result_block,
)

# ---- fake OpenAI client ----


def _tc(call_id: str, name: str, args: dict) -> SimpleNamespace:
    return SimpleNamespace(
        id=call_id, function=SimpleNamespace(name=name, arguments=json.dumps(args))
    )


def _response(*, content=None, tool_calls=None, in_tok=11, out_tok=7) -> SimpleNamespace:
    message = SimpleNamespace(content=content, tool_calls=tool_calls or [])
    choice = SimpleNamespace(message=message, finish_reason="tool_calls" if tool_calls else "stop")
    return SimpleNamespace(
        choices=[choice], usage=SimpleNamespace(prompt_tokens=in_tok, completion_tokens=out_tok)
    )


class _FakeClient:
    """Records the create() kwargs and returns scripted responses in order."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return self._responses.pop(0)


# ---- converters ----


def test_tool_converts_to_openai_function() -> None:
    out = _tool_to_openai(SUBMIT_VERDICT_TOOL)
    assert out["type"] == "function"
    assert out["function"]["name"] == "submit_verdict"
    assert out["function"]["parameters"] == SUBMIT_VERDICT_TOOL["input_schema"]


def test_tool_choice_mapping() -> None:
    assert _tool_choice_to_openai("auto") == "auto"
    assert _tool_choice_to_openai({"type": "tool", "name": "submit_verdict"}) == {
        "type": "function",
        "function": {"name": "submit_verdict"},
    }


def test_messages_convert_tool_use_and_results() -> None:
    assistant = [
        {"type": "tool_use", "id": "call_1", "name": "check_domain", "input": {"url": "x"}}
    ]
    messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": assistant},
        {"role": "user", "content": [tool_result_block("call_1", '{"ok": true}')]},
    ]
    out = _messages_to_openai("SYS", messages)
    assert out[0] == {"role": "system", "content": "SYS"}
    assert out[1] == {"role": "user", "content": "hello"}
    assert out[2]["role"] == "assistant"
    assert out[2]["tool_calls"][0]["id"] == "call_1"
    assert out[3] == {"role": "tool", "tool_call_id": "call_1", "content": '{"ok": true}'}


def test_assistant_without_tool_calls_keeps_string_content() -> None:
    assert _assistant_to_openai([{"type": "text", "text": "hi"}]) == {
        "role": "assistant",
        "content": "hi",
    }


# ---- call_tool ----


def test_call_tool_returns_tool_input(tmp_path) -> None:
    client = _FakeClient(
        [
            _response(
                tool_calls=[_tc("call_1", "submit_verdict", {"verdict": "scam", "scam_score": 0.9})]
            )
        ]
    )
    llm = OpenAILLM(client=client, cache_dir=tmp_path)
    result = llm.call_tool(system="s", user="u", tool=SUBMIT_VERDICT_TOOL)
    assert result.data == {"verdict": "scam", "scam_score": 0.9}
    assert result.input_tokens == 11 and result.output_tokens == 7
    # forced tool choice was sent
    assert client.calls[0]["tool_choice"] == {
        "type": "function",
        "function": {"name": "submit_verdict"},
    }


def test_call_tool_missing_tool_raises(tmp_path) -> None:
    client = _FakeClient([_response(content="no tool", tool_calls=[])])
    llm = OpenAILLM(client=client, cache_dir=tmp_path)
    with pytest.raises(LLMError):
        llm.call_tool(system="s", user="u", tool=SUBMIT_VERDICT_TOOL)


def test_call_tool_caches_second_call(tmp_path) -> None:
    client = _FakeClient(
        [
            _response(
                tool_calls=[_tc("c", "submit_verdict", {"verdict": "benign", "scam_score": 0.1})]
            )
        ]
    )
    llm = OpenAILLM(client=client, cache_dir=tmp_path)
    first = llm.call_tool(system="s", user="u", tool=SUBMIT_VERDICT_TOOL)
    second = llm.call_tool(system="s", user="u", tool=SUBMIT_VERDICT_TOOL)
    assert not first.cached and second.cached
    assert len(client.calls) == 1  # the second call was served from disk


# ---- run_turn ----


def test_run_turn_reports_tool_calls(tmp_path) -> None:
    client = _FakeClient(
        [_response(tool_calls=[_tc("call_9", "check_domain", {"url": "http://x"})])]
    )
    llm = OpenAILLM(client=client, cache_dir=tmp_path)
    turn = llm.run_turn(
        system="s", messages=[{"role": "user", "content": "u"}], tools=[SUBMIT_VERDICT_TOOL]
    )
    assert turn.tool_calls[0].name == "check_domain"
    assert turn.tool_calls[0].input == {"url": "http://x"}
    assert turn.stop_reason == "tool_use"
    # assistant_content echoes back as an Anthropic-style tool_use block
    assert turn.assistant_content[0]["type"] == "tool_use"


# ---- backend selection ----


def test_build_default_llm_prefers_openai_when_only_openai_key(monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("SCAMGUARD_LLM", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert isinstance(build_default_llm(), OpenAILLM)


def test_scamguard_llm_env_forces_choice(monkeypatch) -> None:
    monkeypatch.setenv("SCAMGUARD_LLM", "openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert isinstance(build_default_llm(), OpenAILLM)
