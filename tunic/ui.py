"""Interactive activity lines. Headless output stays machine-readable."""

from __future__ import annotations

import json
import re

_TOOL_CALL = re.compile(r"^tool-call id=\S+ name=(\S+) arguments=(.*)$", re.DOTALL)

_LABELS = {
    "bash": "run",
    "read_file": "read",
    "write_file": "write",
    "list_dir": "list",
}


class ActivityView:
    """Turn agent emit lines into a short feed of what is happening."""

    def __init__(self) -> None:
        self._expect_body = False

    def feed(self, text: str) -> list[str]:
        if text.startswith(("provider: ", "model: ")) or text == "stream: false":
            return []
        if text.startswith("nudge:"):
            return ["retrying — that was a description, not a tool call"]
        if text.startswith("stopped:"):
            return ["stopped —" + text[len("stopped:") :]]
        if text.startswith("assistant: "):
            body = text[len("assistant: ") :]
            return ["", body] if body else []
        match = _TOOL_CALL.match(text)
        if match:
            self._expect_body = True
            return [f"→ {_action(match.group(1), match.group(2))}"]
        if text.startswith("tool-result:"):
            return []
        if self._expect_body:
            self._expect_body = False
            preview = _preview(text)
            return [f"  {preview}"] if preview else []
        return [text]


def _action(name: str, arguments: str) -> str:
    label = _LABELS.get(name, name)
    data = _parse(arguments)
    if name == "bash":
        target = data.get("command")
    else:
        target = data.get("path")
    if not isinstance(target, str) or not target.strip():
        return label
    return f"{label} {_one_line(target, 72)}"


def _parse(arguments: str) -> dict:
    try:
        data = json.loads(arguments)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _preview(text: str) -> str:
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return ""
    first = _one_line(lines[0], 100)
    extra = len(lines) - 1
    if extra:
        return f"{first} (+{extra} lines)"
    return first


def _one_line(text: str, limit: int) -> str:
    flat = " ".join(text.split())
    if len(flat) <= limit:
        return flat
    return flat[: limit - 1] + "…"
