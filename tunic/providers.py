"""OpenAI-compatible and Anthropic clients. Local calls are never streamed."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from urllib.parse import urlparse

from tunic.config import ConfigError, Settings, origin_of
from tunic.tools import Tool


class ProviderError(Exception):
    """The model endpoint failed. The message must not include a key."""


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str
    parsed: dict | None


@dataclass
class Completion:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    reasoning_content: str | None = None
    # Wire-level tool_calls, kept so a local server can see the same objects back.
    raw_tool_calls: list[dict] = field(default_factory=list)


def openai_tools(tools: list[Tool]) -> list[dict]:
    return [tool.openai_tool() for tool in tools]


def anthropic_tools(tools: list[Tool]) -> list[dict]:
    return [tool.anthropic_tool() for tool in tools]


def build_openai_payload(messages: list[dict], tools: list[Tool], settings: Settings) -> dict:
    payload = {
        "model": settings.model,
        "messages": messages,
        "stream": False,
        "max_tokens": settings.max_tokens,
    }
    # Some local servers fault on tool requests when extra sampling fields are set.
    # temperature and tool_choice are not required for that loop, so local
    # payloads omit them. Cloud requests keep both. stream stays false.
    if not settings.local:
        payload["temperature"] = settings.temperature
    if tools:
        payload["tools"] = openai_tools(tools)
        if not settings.local:
            payload["tool_choice"] = "auto"
    return payload


def build_anthropic_payload(messages: list[dict], tools: list[Tool], settings: Settings) -> dict:
    system_parts = [m["content"] for m in messages if m.get("role") == "system"]
    payload = {
        "model": settings.model,
        "max_tokens": settings.max_tokens,
        "temperature": settings.temperature,
        "stream": False,
        "messages": _anthropic_messages(messages),
    }
    if system_parts:
        payload["system"] = "\n\n".join(system_parts)
    if tools:
        payload["tools"] = anthropic_tools(tools)
    return payload


def _anthropic_messages(messages: list[dict]) -> list[dict]:
    converted: list[dict] = []
    for message in messages:
        role = message.get("role")
        if role == "system":
            continue
        if role == "user":
            converted.append({"role": "user", "content": message.get("content") or ""})
            continue
        if role == "assistant":
            blocks = []
            text = message.get("content") or ""
            if text:
                blocks.append({"type": "text", "text": text})
            for call in message.get("tool_calls") or []:
                fn = call.get("function") or {}
                raw_args = fn.get("arguments", "{}")
                try:
                    parsed = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                except json.JSONDecodeError:
                    parsed = {"_raw": raw_args}
                if not isinstance(parsed, dict):
                    parsed = {"_raw": parsed}
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": call.get("id") or "tool",
                        "name": fn.get("name") or call.get("name") or "",
                        "input": parsed,
                    }
                )
            converted.append({"role": "assistant", "content": blocks or ""})
            continue
        if role == "tool":
            converted.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": message.get("tool_call_id") or "",
                            "content": message.get("content") or "",
                        }
                    ],
                }
            )
    return converted


def replay_assistant(completion: Completion, settings: Settings) -> dict:
    message: dict[str, object] = {"role": "assistant", "content": completion.content or ""}
    if completion.raw_tool_calls:
        message["tool_calls"] = completion.raw_tool_calls
    # Qwen-family local servers (LM Studio) expect the reasoning channel echoed
    # back. Cloud OpenAI rejects the extra field, so it stays local-only.
    if settings.local and completion.reasoning_content is not None:
        message["reasoning_content"] = completion.reasoning_content
    return message


def tool_result_message(call: ToolCall, text: str, settings: Settings) -> dict:
    message = {
        "role": "tool",
        "tool_call_id": call.id,
        "content": text,
    }
    if settings.local:
        message["name"] = call.name
    return message


def bearer_token(settings: Settings, key: str | None) -> str:
    # Never forward a cloud key to a local or custom endpoint.
    if settings.local or not key:
        return "not-needed"
    return key


def complete(messages: list[dict], tools: list[Tool], settings: Settings, key: str | None) -> Completion:
    if not settings.model:
        raise ProviderError("no model selected")
    if settings.kind == "anthropic":
        return _complete_anthropic(messages, tools, settings, key)
    return _complete_openai(messages, tools, settings, key)


def _complete_openai(messages: list[dict], tools: list[Tool], settings: Settings, key: str | None) -> Completion:
    payload = build_openai_payload(messages, tools, settings)
    if payload.get("stream") is not False:
        raise ProviderError("refusing to stream; local tool-calls hang when streamed")
    url = settings.base_url.rstrip("/") + "/chat/completions"
    body = _post_json(
        url,
        payload,
        {
            "Authorization": "Bearer " + bearer_token(settings, key),
            "Content-Type": "application/json",
        },
        settings.http_timeout,
    )
    return _parse_openai(body)


def _complete_anthropic(messages: list[dict], tools: list[Tool], settings: Settings, key: str | None) -> Completion:
    if not key:
        raise ProviderError("anthropic: no API key. No request was sent.")
    payload = build_anthropic_payload(messages, tools, settings)
    if payload.get("stream") is not False:
        raise ProviderError("refusing to stream")
    url = settings.base_url.rstrip("/") + "/v1/messages"
    body = _post_json(
        url,
        payload,
        {
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        },
        settings.http_timeout,
    )
    return _parse_anthropic(body)


def _parse_openai(body: dict) -> Completion:
    try:
        message = body["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ProviderError("model response had no choices[0].message") from exc
    content = message.get("content") or ""
    if not isinstance(content, str):
        content = str(content)
    raw_calls = message.get("tool_calls") or []
    if not isinstance(raw_calls, list):
        raw_calls = []
    calls = []
    for raw in raw_calls:
        if not isinstance(raw, dict):
            continue
        fn = raw.get("function") or {}
        name = str(fn.get("name") or "")
        raw_args = fn.get("arguments")
        if isinstance(raw_args, str):
            arguments = raw_args
        else:
            arguments = json.dumps(raw_args or {})
        parsed = _parse_arguments(arguments)
        calls.append(
            ToolCall(
                id=str(raw.get("id") or name or "tool"),
                name=name,
                arguments=arguments,
                parsed=parsed,
            )
        )
    reasoning = message.get("reasoning_content") if "reasoning_content" in message else None
    return Completion(
        content=content,
        tool_calls=calls,
        reasoning_content=reasoning,
        raw_tool_calls=raw_calls,
    )


def _parse_anthropic(body: dict) -> Completion:
    blocks = body.get("content") or []
    texts = []
    calls = []
    raw_calls = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            texts.append(block.get("text") or "")
        elif block.get("type") == "tool_use":
            name = block.get("name") or ""
            parsed = block.get("input") if isinstance(block.get("input"), dict) else None
            arguments = json.dumps(parsed or {})
            call_id = str(block.get("id") or name or "tool")
            calls.append(ToolCall(id=call_id, name=name, arguments=arguments, parsed=parsed))
            raw_calls.append(
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": arguments},
                }
            )
    return Completion(content="".join(texts), tool_calls=calls, raw_tool_calls=raw_calls)


def _parse_arguments(arguments: str) -> dict | None:
    try:
        parsed = json.loads(arguments) if arguments else {}
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed


def _post_json(url: str, payload: dict, headers: dict, timeout: float) -> dict:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    host = urlparse(url).netloc or url
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise ProviderError(f"HTTP {exc.code} from {host}: {detail}") from None
    except urllib.error.URLError as exc:
        raise ProviderError(f"could not reach {host}: {exc.reason}") from None
    except TimeoutError as exc:
        raise ProviderError(f"timed out talking to {host}") from exc
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ProviderError(f"non-JSON response from {host}") from exc
    if not isinstance(parsed, dict):
        raise ProviderError(f"unexpected response from {host}")
    return parsed


def fetch_lmstudio_catalog(settings: Settings, timeout: float = 15) -> list[dict]:
    """GET the native models list. This does not load a model."""
    url = origin_of(settings.base_url) + "/api/v0/models"
    request = urllib.request.Request(
        url,
        headers={"Authorization": "Bearer not-needed"},
        method="GET",
    )
    host = urlparse(url).netloc or url
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:300]
        raise ProviderError(f"HTTP {exc.code} from {host}: {detail}") from None
    except urllib.error.URLError as exc:
        raise ProviderError(f"could not reach {host}: {exc.reason}") from None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ProviderError(f"non-JSON model list from {host}") from exc
    data = parsed.get("data") if isinstance(parsed, dict) else None
    if not isinstance(data, list):
        raise ProviderError(f"model list from {host} had no data array")
    return data


def select_lmstudio_model(settings: Settings, catalog: list[dict]) -> str:
    """Use an already-loaded model. Do not name an unloaded one unless asked."""
    loaded = [item for item in catalog if item.get("state") == "loaded" and item.get("id")]
    loaded_ids = [item["id"] for item in loaded]
    if settings.model:
        if settings.model in loaded_ids or settings.allow_load:
            return settings.model
        raise ConfigError(
            f"model {settings.model!r} is not loaded. Refusing to load it. "
            "Omit --model to use the loaded model, or pass --allow-load."
        )
    if not loaded:
        raise ConfigError("No model is loaded in LM Studio. Refusing to load one.")
    for item in loaded:
        caps = item.get("capabilities") or []
        if "tool_use" in caps:
            return item["id"]
    return loaded_ids[0]
