"""Interactive screen. Labeled boxes group what is already shown.

Headless output stays machine-readable.
"""

from __future__ import annotations

import json
import re

_TOOL_CALL = re.compile(r"^tool-call id=\S+ name=(\S+) arguments=(.*)$", re.DOTALL)
_ANSI = re.compile(r"\033\[[0-9;]*m")

# Shared floor so a short screen still has a real box, not a sliver.
SCREEN_WIDTH = 72
# Prefix for a horizontal rule inside a box. Not shown as text.
RULE = "\x1e"


def rule(label: str = "") -> str:
    return RULE + label

_LABELS = {
    "bash": "run",
    "read_file": "read",
    "write_file": "write",
    "list_dir": "list",
    "search": "search",
    "edit_file": "edit",
    "fetch": "fetch",
}


class ActivityView:
    """Turn agent emit lines into a short feed of what is happening."""

    def __init__(self) -> None:
        self._expect_body = False

    def feed(self, text: str) -> list[str]:
        if text.startswith(("provider: ", "model: ", "stream:")):
            return []
        if text.startswith("nudge:"):
            return ["retrying — that was a description, not a tool call"]
        if text.startswith("wrap:"):
            return ["last step — answering from what it has"]
        if text.startswith("step:"):
            return ["step " + text[len("step:") :].strip()]
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
            diff = [
                line
                for line in text.splitlines()
                if line.startswith(("+", "-")) and not line.startswith(("+++", "---"))
            ]
            if diff:
                shown = diff[:6]
                extra = len(diff) - 6
                rows = [f"  {line}" for line in shown]
                if extra > 0:
                    rows.append(f"  ... +{extra} more")
                return rows
            preview = _preview(text)
            return [f"  {preview}"] if preview else []
        return [text]


def _action(name: str, arguments: str) -> str:
    label = _LABELS.get(name, name)
    data = _parse(arguments)
    if name == "bash":
        target = data.get("command")
    elif name == "search":
        target = data.get("query")
        glob = data.get("glob")
        if isinstance(glob, str) and glob.strip() and isinstance(target, str):
            target = f"{target} in {glob}"
    elif name == "fetch":
        target = data.get("url")
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


def write_label(settings) -> str:
    """Whether a write or a shell may run without asking."""
    if settings.plan:
        return "off (plan mode)"
    if settings.yes:
        return "allowed"
    return "ask before a write or a shell"


def auto_label(settings) -> str:
    if settings.plan:
        return "off (plan mode)"
    return "on" if settings.yes else "off"


def paint(text: str, code: str, *, enabled: bool) -> str:
    """Wrap one line. The text itself stays contiguous so a search still matches."""
    if not enabled or not code or text == "":
        return text
    return f"\033[{code}m{text}\033[0m"


def tone_for(line: str) -> str:
    if line.startswith(("┌", "└", "├")):
        return "1" if "tunic " in line else "2"
    inner = line[1:].strip() if line.startswith("│") else line
    if inner.startswith("tunic "):
        return "1"
    if inner.startswith(("writes: ask", "writes: off", "plan: on", "auto: off")):
        return "33"
    if inner.startswith(("writes: allowed", "auto: on")):
        return "32"
    if inner.startswith(("project:", "provider:", "model:", "plan:", "writes:", "auto:", "steps:", "saved", "kind:", "base_url:", "home:", "stream:", "loaded:", "mutating ", "max_steps:")):
        return "36"
    if inner.startswith(("local:", "synthetic:", "note:")):
        return "33"
    if inner.startswith("/"):
        return "2"
    if inner[:1].isdigit() or inner.startswith(("q  ", "→")):
        return "36"
    return ""


def paint_block(text: str, *, enabled: bool) -> str:
    return "\n".join(paint(line, tone_for(line), enabled=enabled) for line in text.splitlines())


def _visible_len(text: str) -> int:
    return len(_ANSI.sub("", text))


def boxed(label: str, lines: list[str], *, min_width: int = SCREEN_WIDTH) -> str:
    """One labeled box. The words inside stay contiguous, including with color off."""
    visible_lines = [_rule_label(line) if _is_rule(line) else line for line in lines]
    width = max([_visible_len(line) for line in visible_lines] + [len(label) + 2, min_width])
    framed = [_box_top(label, width)]
    for line in lines:
        if _is_rule(line):
            framed.append(_box_rule(_rule_label(line), width))
        else:
            framed.append(_box_row(line, width))
    framed.append(_box_bottom(width))
    return "\n".join(framed)


def _is_rule(line: str) -> bool:
    return line.startswith(RULE)


def _rule_label(line: str) -> str:
    return line[len(RULE) :] if line.startswith(RULE) else line


def _box_top(label: str, width: int) -> str:
    return _box_edge("┌", "┐", label, width)


def _box_rule(label: str, width: int) -> str:
    return _box_edge("├", "┤", label, width)


def _box_edge(left: str, right: str, label: str, width: int) -> str:
    inner = width + 2
    if label:
        title = f" {label} "
        right_pad = inner - 1 - len(title)
        if right_pad < 1:
            title = f" {label} "
            right_pad = max(inner - 1 - len(title), 0)
        return left + "─" + title + ("─" * right_pad) + right
    return left + ("─" * inner) + right


def _box_row(text: str, width: int) -> str:
    visible = _visible_len(text)
    if visible <= width:
        return "│ " + text + (" " * (width - visible)) + " │"
    return "│ " + text


def _box_bottom(width: int) -> str:
    return "└" + ("─" * (width + 2)) + "┘"


class OpenBox:
    """Labeled box printed as the lines arrive. A long line keeps its words."""

    def __init__(self, label: str, width: int = SCREEN_WIDTH) -> None:
        self.label = label
        self.width = max(width, len(label) + 2)
        self.is_open = False
        self.streamed = False

    def start(self) -> str:
        self.is_open = True
        return _box_top(self.label, self.width)

    def row(self, text: str) -> str:
        return _box_row(text, self.width)

    def end(self) -> str:
        self.is_open = False
        return _box_bottom(self.width)


def format_slash_help() -> str:
    """Every interactive command, one meaning each, in one labeled box."""
    meaning = dict(SLASH_HELP)
    width = max(len(name) for name, _meaning in SLASH_HELP)
    rows: list[str] = []
    for group, names in SLASH_HELP_GROUPS:
        rows.append(rule(group))
        for name in names:
            rows.append(f"{name:<{width}}  {meaning[name]}")
    note = "A write or a shell command asks in plain language before it runs, unless /auto on, --yes, or config yes."
    return boxed("Commands", rows) + "\n" + note


def permission_question(name: str, arguments: dict | None = None) -> str:
    """Plain-language ask. The tool has not run yet."""
    data = arguments if isinstance(arguments, dict) else {}
    if name == "bash":
        command = data.get("command")
        if isinstance(command, str) and command.strip():
            return (
                "Run this shell command before continuing: "
                f"{_one_line(command, 72)}\nAllow it? [y/N] "
            )
        return "Run a shell command before continuing?\nAllow it? [y/N] "
    if name == "write_file":
        path = data.get("path")
        if isinstance(path, str) and path.strip():
            return f"Write {_one_line(path, 72)}? This changes the project.\nAllow it? [y/N] "
        return "Write a file? This changes the project.\nAllow it? [y/N] "
    if name == "edit_file":
        path = data.get("path")
        if isinstance(path, str) and path.strip():
            return f"Write a span in {_one_line(path, 72)}? This changes the project.\nAllow it? [y/N] "
        return "Write a span in a file? This changes the project.\nAllow it? [y/N] "
    return f"Allow {name} before it runs? [y/N] "


SLASH_HELP = (
    ("/exit", "leave this session"),
    ("/quit", "leave this session"),
    ("/help", "list these commands"),
    ("/settings", "choose a connection and model"),
    ("/config", "same as /settings"),
    ("/auto", "show whether writes and shell commands run without asking"),
    ("/auto on", "run writes and shell commands without asking"),
    ("/auto off", "ask before writes and shell commands"),
    ("/yes", "same as /auto, toggles when used alone"),
    ("/steps", "show the tool-step cap for a turn"),
    ("/steps N", "set how many tool calls a turn may make"),
    ("/continue", "run another round of tool steps after the cap"),
    ("/plan", "turn plan mode on or off (read-only)"),
    ("/go", "leave plan mode and execute the plan"),
    ("/compact", "shorten the remembered turns"),
    ("/sessions", "list saved sessions"),
    ("/undo", "restore the last write or edit"),
    ("/skill", "list skill notes"),
    ("/skill NAME", "print a skill note"),
    ("/model", "list local qwen3.8 and Synthetic models, then pick one"),
    ("/model ID", "save a different model for this connection"),
    ("/profile", "list saved profiles"),
    ("/profile NAME", "switch to a saved profile (local, synthetic, ...)"),
    ("/session NAME", "name this session so it can be resumed"),
)

SLASH_HELP_GROUPS = (
    ("session", ("/exit", "/quit", "/help", "/session NAME", "/sessions")),
    ("connection", ("/settings", "/config", "/model", "/model ID", "/profile", "/profile NAME")),
    ("tools", ("/auto", "/auto on", "/auto off", "/yes", "/steps", "/steps N", "/continue", "/plan", "/go", "/compact", "/undo", "/skill", "/skill NAME")),
)


def _one_line(text: str, limit: int) -> str:
    flat = " ".join(text.split())
    if len(flat) <= limit:
        return flat
    return flat[: limit - 1] + "…"
