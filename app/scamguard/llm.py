"""Single gateway for every model call.

Provides retries, an on-disk response cache, and token and latency logging.
Also defines FakeLLM, the deterministic stand-in used by all tests and CI;
tests and CI never call the real API.

Two primitives, both of which force structured output through tools:
  - call_tool: one forced tool call, returns its input as a dict (extractor, advisor).
  - run_turn:  one turn of a multi-tool loop, returns the tool calls the model made
               (the detector drives this loop and feeds tool results back).
No other module may import anthropic or call the API directly.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from scamguard import config

log = logging.getLogger("scamguard.llm")

# A tool definition is {"name", "description", "input_schema"}; input is a JSON object.
Tool = Mapping[str, Any]
JsonObj = dict[str, Any]
# tool_choice is "auto"/"any"/"none" or {"type": "tool", "name": ...}.
ToolChoice = str | Mapping[str, Any]


class LLMError(RuntimeError):
    """Raised when the model returns no usable tool call (e.g. a refusal).

    Callers must turn this into "unsure", never "benign" (invariant 6).
    """


@dataclass(frozen=True)
class LLMResult:
    """A structured-output result (one forced tool) plus the metadata we log."""

    data: JsonObj
    model: str
    tool: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    cached: bool = False


@dataclass(frozen=True)
class ToolCall:
    """One tool_use block the model emitted during a turn."""

    id: str
    name: str
    input: JsonObj


@dataclass(frozen=True)
class LLMTurn:
    """One assistant turn of a tool loop."""

    tool_calls: list[ToolCall]
    assistant_content: list[Any]  # echo back into messages on the next turn
    text: str = ""
    stop_reason: str = ""
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    cached: bool = False


def tool_result_block(tool_use_id: str, content: str, *, is_error: bool = False) -> JsonObj:
    """Build a tool_result content block to send back in the next user turn."""
    block: JsonObj = {"type": "tool_result", "tool_use_id": tool_use_id, "content": content}
    if is_error:
        block["is_error"] = True
    return block


class LLM(Protocol):
    """The one interface callers use; AnthropicLLM and FakeLLM both satisfy it."""

    def call_tool(
        self, *, system: str, user: str, tool: Tool, model: str | None = None
    ) -> LLMResult: ...

    def run_turn(
        self,
        *,
        system: str,
        messages: list[JsonObj],
        tools: Sequence[Tool],
        tool_choice: ToolChoice = "auto",
    ) -> LLMTurn: ...


# ---------------------------------------------------------------------------
# Cache key
# ---------------------------------------------------------------------------


def cache_key(model: str, system: str, user: str, tools: Sequence[Tool]) -> str:
    """Stable sha256 over (model, system, user, tools). Order-independent for dict keys."""
    blob = json.dumps(
        {"model": model, "system": system, "user": user, "tools": list(tools)},
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Real client
# ---------------------------------------------------------------------------


class AnthropicLLM:
    """Forces tool calls through the Anthropic Messages API.

    Retries are handled here with tenacity; the SDK client is created with
    max_retries=0 so retry behavior lives in one place. call_tool results are
    cached on disk (the multi-step run_turn loop is stateful and not cached).
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        max_tokens: int | None = None,
        cache_dir: Path | None = None,
        client: Any | None = None,
    ) -> None:
        self.model = model or config.MODEL
        self.max_tokens = max_tokens or config.MAX_TOKENS
        self.cache_dir = cache_dir if cache_dir is not None else config.LLM_CACHE_DIR
        self._client = client  # injectable for tests; otherwise created lazily

    @property
    def client(self) -> Any:
        if self._client is None:
            import anthropic  # imported lazily so importing llm.py needs no API key

            self._client = anthropic.Anthropic(max_retries=0)
        return self._client

    # -- cache helpers --

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def _read_cache(self, key: str) -> JsonObj | None:
        path = self._cache_path(key)
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def _write_cache(self, key: str, payload: JsonObj) -> None:
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._cache_path(key).write_text(
                json.dumps(payload, ensure_ascii=False), encoding="utf-8"
            )
        except OSError as exc:  # a cache write must never break a request
            log.warning("llm cache write failed: %s", exc)

    # -- API call with tenacity retries (shared by call_tool and run_turn) --

    def _api_call(
        self,
        *,
        system: str,
        messages: list[JsonObj],
        tools: list[Tool],
        tool_choice: ToolChoice,
    ) -> Any:
        import anthropic
        from tenacity import (
            retry,
            retry_if_exception,
            stop_after_attempt,
            wait_exponential,
        )

        def _retryable(exc: BaseException) -> bool:
            if isinstance(
                exc,
                (
                    anthropic.APIConnectionError,
                    anthropic.APITimeoutError,
                    anthropic.RateLimitError,
                ),
            ):
                return True
            if isinstance(exc, anthropic.APIStatusError):
                return exc.status_code >= 500
            return False

        @retry(
            retry=retry_if_exception(_retryable),
            stop=stop_after_attempt(config.LLM_MAX_ATTEMPTS),
            wait=wait_exponential(
                multiplier=config.LLM_RETRY_INITIAL_SECONDS,
                max=config.LLM_RETRY_MAX_SECONDS,
            ),
            reraise=True,
        )
        def _do() -> Any:
            return self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system,
                messages=messages,
                tools=[dict(t) for t in tools],
                tool_choice=tool_choice,
            )

        return _do()

    @staticmethod
    def _usage(message: Any) -> tuple[int, int]:
        usage = getattr(message, "usage", None)
        return (
            int(getattr(usage, "input_tokens", 0) or 0),
            int(getattr(usage, "output_tokens", 0) or 0),
        )

    def call_tool(
        self, *, system: str, user: str, tool: Tool, model: str | None = None
    ) -> LLMResult:
        if model is not None:
            # A per-call model override would need its own instance; keep one model per client.
            raise ValueError("set the model on the AnthropicLLM instance, not per call")
        tool_name = str(tool["name"])
        key = cache_key(self.model, system, user, [tool])

        cached = self._read_cache(key)
        if cached is not None:
            log.info("llm cache hit model=%s tool=%s key=%s", self.model, tool_name, key[:12])
            return LLMResult(
                data=cached["data"],
                model=cached.get("model", self.model),
                tool=tool_name,
                input_tokens=cached.get("input_tokens", 0),
                output_tokens=cached.get("output_tokens", 0),
                latency_s=0.0,
                cached=True,
            )

        start = time.monotonic()
        message = self._api_call(
            system=system,
            messages=[{"role": "user", "content": user}],
            tools=[tool],
            tool_choice={"type": "tool", "name": tool_name},
        )
        latency = time.monotonic() - start

        if getattr(message, "stop_reason", None) == "refusal":
            raise LLMError(f"model refused the request (tool={tool_name})")

        data: JsonObj | None = None
        for block in message.content:
            if getattr(block, "type", None) == "tool_use" and block.name == tool_name:
                data = dict(block.input)
                break
        if data is None:
            raise LLMError(f"model returned no '{tool_name}' tool call")

        in_tok, out_tok = self._usage(message)
        log.info(
            "llm call model=%s tool=%s in=%d out=%d latency=%.3fs key=%s",
            self.model,
            tool_name,
            in_tok,
            out_tok,
            latency,
            key[:12],
        )

        self._write_cache(
            key,
            {
                "data": data,
                "model": self.model,
                "tool": tool_name,
                "input_tokens": in_tok,
                "output_tokens": out_tok,
            },
        )
        return LLMResult(
            data=data,
            model=self.model,
            tool=tool_name,
            input_tokens=in_tok,
            output_tokens=out_tok,
            latency_s=latency,
            cached=False,
        )

    def run_turn(
        self,
        *,
        system: str,
        messages: list[JsonObj],
        tools: Sequence[Tool],
        tool_choice: ToolChoice = "auto",
    ) -> LLMTurn:
        start = time.monotonic()
        message = self._api_call(
            system=system,
            messages=list(messages),
            tools=list(tools),
            tool_choice=tool_choice,
        )
        latency = time.monotonic() - start

        if getattr(message, "stop_reason", None) == "refusal":
            raise LLMError("model refused the request (detector turn)")

        tool_calls = [
            ToolCall(id=b.id, name=b.name, input=dict(b.input))
            for b in message.content
            if getattr(b, "type", None) == "tool_use"
        ]
        text = "".join(
            getattr(b, "text", "") for b in message.content if getattr(b, "type", None) == "text"
        )
        in_tok, out_tok = self._usage(message)
        log.info(
            "llm turn model=%s tools=%d calls=%d in=%d out=%d latency=%.3fs",
            self.model,
            len(tools),
            len(tool_calls),
            in_tok,
            out_tok,
            latency,
        )
        return LLMTurn(
            tool_calls=tool_calls,
            assistant_content=list(message.content),
            text=text,
            stop_reason=getattr(message, "stop_reason", "") or "",
            model=self.model,
            input_tokens=in_tok,
            output_tokens=out_tok,
            latency_s=latency,
            cached=False,
        )


# ---------------------------------------------------------------------------
# Optional OpenAI backend (opt-in; Claude stays the default)
# ---------------------------------------------------------------------------


def _tool_to_openai(tool: Tool) -> JsonObj:
    """Anthropic tool {name, description, input_schema} -> OpenAI function tool."""
    return {
        "type": "function",
        "function": {
            "name": str(tool["name"]),
            "description": str(tool.get("description", "")),
            "parameters": tool.get("input_schema", {"type": "object", "properties": {}}),
        },
    }


def _tool_choice_to_openai(choice: ToolChoice) -> Any:
    if isinstance(choice, str):
        return {"auto": "auto", "any": "required", "none": "none"}.get(choice, "auto")
    if isinstance(choice, Mapping) and choice.get("type") == "tool":
        return {"type": "function", "function": {"name": choice["name"]}}
    return "auto"


def _loads_args(raw: Any) -> JsonObj:
    if isinstance(raw, dict):
        return raw
    try:
        value = json.loads(raw or "{}")
    except (TypeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _assistant_to_openai(content: Any) -> JsonObj:
    """An echoed assistant turn (string or Anthropic-style blocks) -> OpenAI message."""
    if isinstance(content, str):
        return {"role": "assistant", "content": content}
    text_parts: list[str] = []
    tool_calls: list[JsonObj] = []
    for block in content or []:
        btype = block.get("type") if isinstance(block, Mapping) else getattr(block, "type", None)
        if btype == "text":
            text_parts.append(
                block.get("text", "") if isinstance(block, Mapping) else getattr(block, "text", "")
            )
        elif btype == "tool_use":
            bid = block["id"] if isinstance(block, Mapping) else block.id
            bname = block["name"] if isinstance(block, Mapping) else block.name
            binput = block["input"] if isinstance(block, Mapping) else block.input
            tool_calls.append(
                {
                    "id": bid,
                    "type": "function",
                    "function": {"name": bname, "arguments": json.dumps(dict(binput))},
                }
            )
    msg: JsonObj = {"role": "assistant"}
    text = "".join(text_parts)
    if tool_calls:
        msg["tool_calls"] = tool_calls
        msg["content"] = text or None
    else:
        msg["content"] = text or ""
    return msg


def _messages_to_openai(system: str, messages: list[JsonObj]) -> list[JsonObj]:
    """Convert Anthropic-style messages (as the pipeline builds them) to OpenAI chat form."""
    out: list[JsonObj] = [{"role": "system", "content": system}]
    for m in messages:
        role, content = m["role"], m["content"]
        if role == "assistant":
            out.append(_assistant_to_openai(content))
        elif isinstance(content, str):
            out.append({"role": "user", "content": content})
        else:  # a user turn carrying tool_result blocks
            for block in content:
                if not isinstance(block, Mapping):
                    continue
                if block.get("type") == "tool_result":
                    out.append(
                        {
                            "role": "tool",
                            "tool_call_id": block["tool_use_id"],
                            "content": str(block.get("content", "")),
                        }
                    )
                elif block.get("type") == "text":
                    out.append({"role": "user", "content": str(block.get("text", ""))})
    return out


class OpenAILLM:
    """Forces tool calls through the OpenAI Chat Completions API (opt-in backend).

    Same interface as AnthropicLLM. GPT-4o-mini accepts forced function calls, so the
    structured-output contract the pipeline relies on is preserved; nothing downstream
    changes. call_tool results are cached on disk; the stateful run_turn loop is not.
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        max_tokens: int | None = None,
        cache_dir: Path | None = None,
        client: Any | None = None,
    ) -> None:
        self.model = model or config.OPENAI_MODEL
        self.max_tokens = max_tokens or config.MAX_TOKENS
        self.cache_dir = cache_dir if cache_dir is not None else config.LLM_CACHE_DIR
        self._client = client

    @property
    def client(self) -> Any:
        if self._client is None:
            import openai  # imported lazily so importing llm.py needs no openai install

            self._client = openai.OpenAI(max_retries=0)
        return self._client

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def _read_cache(self, key: str) -> JsonObj | None:
        try:
            return json.loads(self._cache_path(key).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def _write_cache(self, key: str, payload: JsonObj) -> None:
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._cache_path(key).write_text(
                json.dumps(payload, ensure_ascii=False), encoding="utf-8"
            )
        except OSError as exc:
            log.warning("llm cache write failed: %s", exc)

    def _api_call(self, *, messages: list[JsonObj], tools: list[JsonObj], tool_choice: Any) -> Any:
        import openai
        from tenacity import (
            retry,
            retry_if_exception,
            stop_after_attempt,
            wait_exponential,
        )

        def _retryable(exc: BaseException) -> bool:
            if isinstance(
                exc,
                (openai.APIConnectionError, openai.APITimeoutError, openai.RateLimitError),
            ):
                return True
            if isinstance(exc, openai.APIStatusError):
                return exc.status_code >= 500
            return False

        @retry(
            retry=retry_if_exception(_retryable),
            stop=stop_after_attempt(config.LLM_MAX_ATTEMPTS),
            wait=wait_exponential(
                multiplier=config.LLM_RETRY_INITIAL_SECONDS, max=config.LLM_RETRY_MAX_SECONDS
            ),
            reraise=True,
        )
        def _do() -> Any:
            return self.client.chat.completions.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
            )

        return _do()

    @staticmethod
    def _usage(resp: Any) -> tuple[int, int]:
        usage = getattr(resp, "usage", None)
        return (
            int(getattr(usage, "prompt_tokens", 0) or 0),
            int(getattr(usage, "completion_tokens", 0) or 0),
        )

    def call_tool(
        self, *, system: str, user: str, tool: Tool, model: str | None = None
    ) -> LLMResult:
        if model is not None:
            raise ValueError("set the model on the OpenAILLM instance, not per call")
        tool_name = str(tool["name"])
        key = cache_key(self.model, system, user, [tool])

        cached = self._read_cache(key)
        if cached is not None:
            log.info("llm cache hit model=%s tool=%s key=%s", self.model, tool_name, key[:12])
            return LLMResult(
                data=cached["data"],
                model=cached.get("model", self.model),
                tool=tool_name,
                input_tokens=cached.get("input_tokens", 0),
                output_tokens=cached.get("output_tokens", 0),
                latency_s=0.0,
                cached=True,
            )

        start = time.monotonic()
        resp = self._api_call(
            messages=_messages_to_openai(system, [{"role": "user", "content": user}]),
            tools=[_tool_to_openai(tool)],
            tool_choice={"type": "function", "function": {"name": tool_name}},
        )
        latency = time.monotonic() - start

        message = resp.choices[0].message
        data: JsonObj | None = None
        for call in message.tool_calls or []:
            if call.function.name == tool_name:
                data = _loads_args(call.function.arguments)
                break
        if data is None:
            raise LLMError(f"model returned no '{tool_name}' tool call")

        in_tok, out_tok = self._usage(resp)
        log.info(
            "llm call model=%s tool=%s in=%d out=%d latency=%.3fs key=%s",
            self.model,
            tool_name,
            in_tok,
            out_tok,
            latency,
            key[:12],
        )
        self._write_cache(
            key,
            {
                "data": data,
                "model": self.model,
                "tool": tool_name,
                "input_tokens": in_tok,
                "output_tokens": out_tok,
            },
        )
        return LLMResult(
            data=data,
            model=self.model,
            tool=tool_name,
            input_tokens=in_tok,
            output_tokens=out_tok,
            latency_s=latency,
            cached=False,
        )

    def run_turn(
        self,
        *,
        system: str,
        messages: list[JsonObj],
        tools: Sequence[Tool],
        tool_choice: ToolChoice = "auto",
    ) -> LLMTurn:
        start = time.monotonic()
        resp = self._api_call(
            messages=_messages_to_openai(system, list(messages)),
            tools=[_tool_to_openai(t) for t in tools],
            tool_choice=_tool_choice_to_openai(tool_choice),
        )
        latency = time.monotonic() - start

        message = resp.choices[0].message
        tool_calls: list[ToolCall] = []
        assistant_content: list[Any] = []
        if message.content:
            assistant_content.append({"type": "text", "text": message.content})
        for call in message.tool_calls or []:
            inp = _loads_args(call.function.arguments)
            tool_calls.append(ToolCall(id=call.id, name=call.function.name, input=inp))
            assistant_content.append(
                {"type": "tool_use", "id": call.id, "name": call.function.name, "input": inp}
            )
        in_tok, out_tok = self._usage(resp)
        log.info(
            "llm turn model=%s tools=%d calls=%d in=%d out=%d latency=%.3fs",
            self.model,
            len(tools),
            len(tool_calls),
            in_tok,
            out_tok,
            latency,
        )
        return LLMTurn(
            tool_calls=tool_calls,
            assistant_content=assistant_content,
            text=message.content or "",
            stop_reason="tool_use" if tool_calls else "end_turn",
            model=self.model,
            input_tokens=in_tok,
            output_tokens=out_tok,
            latency_s=latency,
            cached=False,
        )


_DOTENV_LOADED = False


def _load_dotenv_once() -> None:
    """Load KEY=VALUE lines from a repo-root .env into os.environ (shell wins).

    Dependency-free. Only fills variables not already set. Disabled when
    SCAMGUARD_DISABLE_DOTENV is set (the test suite sets it, so a local .env never
    leaks into tests).
    """
    global _DOTENV_LOADED
    import os

    if _DOTENV_LOADED or os.environ.get("SCAMGUARD_DISABLE_DOTENV"):
        return
    _DOTENV_LOADED = True
    try:
        text = (config.REPO_ROOT / ".env").read_text(encoding="utf-8")
    except OSError:
        return
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


def have_credentials() -> bool:
    """True if any supported model credential is set (reads .env too)."""
    import os

    _load_dotenv_once()
    return bool(
        os.environ.get("ANTHROPIC_API_KEY")
        or os.environ.get("ANTHROPIC_AUTH_TOKEN")
        or os.environ.get("OPENAI_API_KEY")
    )


def _use_openai() -> bool:
    """Whether the OpenAI backend is active (reads SCAMGUARD_LLM and the keys / .env)."""
    import os

    _load_dotenv_once()
    choice = os.environ.get("SCAMGUARD_LLM", "").strip().lower()
    if choice == "openai":
        return True
    if choice == "anthropic":
        return False
    has_openai = bool(os.environ.get("OPENAI_API_KEY"))
    has_anthropic = bool(
        os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")
    )
    return has_openai and not has_anthropic


def active_backend() -> str:
    """'openai' or 'anthropic' — the backend build_default_llm would pick."""
    return "openai" if _use_openai() else "anthropic"


def default_model_name() -> str:
    """The model name the active backend uses by default (for report labels)."""
    return config.OPENAI_MODEL if _use_openai() else config.MODEL


def build_default_llm(model: str | None = None) -> LLM:
    """Pick the backend. Claude is the default; GPT-4o-mini is opt-in.

    SCAMGUARD_LLM=openai|anthropic forces a choice; otherwise OpenAI is used when
    OPENAI_API_KEY is set and no Anthropic key is. A Claude model name passed while the
    OpenAI backend is active is ignored (the OpenAI default model is used instead).
    """
    if _use_openai():
        if model is not None and not model.startswith("gpt"):
            model = None
        return OpenAILLM(model=model)
    return AnthropicLLM(model=model)


# ---------------------------------------------------------------------------
# Deterministic fake for tests and CI (never touches the network)
# ---------------------------------------------------------------------------

# A scripted call_tool response: the tool-input dict, an Exception to raise, or a
# callable(system, user, tool) -> dict.
Scripted = JsonObj | Exception | Callable[[str, str, Tool], JsonObj]
# A scripted run_turn: a list of (tool_name, input) pairs for that turn, or an Exception.
TurnScript = list[tuple[str, JsonObj]] | Exception


@dataclass
class FakeLLM:
    """Same interface as AnthropicLLM, backed by canned responses.

    call_tool: provide `responses` (consumed in order), a `by_tool` mapping, or a
    `default`. run_turn: provide `turns`, each a list of (tool_name, input) pairs
    (or an Exception to raise). Every call is recorded for assertions.
    """

    responses: list[Scripted] = field(default_factory=list)
    by_tool: dict[str, Scripted] = field(default_factory=dict)
    default: Scripted | None = None
    turns: list[TurnScript] = field(default_factory=list)
    model: str = "fake-model"
    calls: list[JsonObj] = field(default_factory=list)
    run_calls: list[JsonObj] = field(default_factory=list)

    def _resolve(self, system: str, user: str, tool: Tool) -> Scripted:
        name = str(tool["name"])
        if self.responses:
            return self.responses.pop(0)
        if name in self.by_tool:
            return self.by_tool[name]
        if self.default is not None:
            return self.default
        raise LLMError(f"FakeLLM has no scripted response for tool '{name}'")

    def call_tool(
        self, *, system: str, user: str, tool: Tool, model: str | None = None
    ) -> LLMResult:
        tool_name = str(tool["name"])
        self.calls.append({"system": system, "user": user, "tool": tool_name})

        scripted = self._resolve(system, user, tool)
        if isinstance(scripted, Exception):
            raise scripted
        data = scripted(system, user, tool) if callable(scripted) else scripted
        return LLMResult(data=dict(data), model=model or self.model, tool=tool_name, cached=False)

    def run_turn(
        self,
        *,
        system: str,
        messages: list[JsonObj],
        tools: Sequence[Tool],
        tool_choice: ToolChoice = "auto",
    ) -> LLMTurn:
        index = len(self.run_calls)
        self.run_calls.append(
            {
                "system": system,
                "messages": list(messages),
                "tools": [str(t["name"]) for t in tools],
                "tool_choice": tool_choice,
            }
        )
        if not self.turns:
            raise LLMError("FakeLLM has no scripted turn")
        spec = self.turns.pop(0)
        if isinstance(spec, Exception):
            raise spec

        tool_calls: list[ToolCall] = []
        content: list[Any] = []
        for i, (name, inp) in enumerate(spec):
            cid = f"toolu_{index}_{i}"
            tool_calls.append(ToolCall(id=cid, name=name, input=dict(inp)))
            content.append({"type": "tool_use", "id": cid, "name": name, "input": dict(inp)})
        return LLMTurn(
            tool_calls=tool_calls,
            assistant_content=content,
            text="",
            stop_reason="tool_use" if tool_calls else "end_turn",
            model=self.model,
        )
