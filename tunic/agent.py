"""Agent loop. Tools run one at a time, in the order the model emitted them."""

from __future__ import annotations

from dataclasses import dataclass, field

from tunic.config import Settings
from tunic.keys import resolve_key
from tunic.providers import (
    Completion,
    ProviderError,
    complete,
    fetch_lmstudio_catalog,
    replay_assistant,
    select_lmstudio_model,
    tool_result_message,
)
from tunic.tools import builtin_tools, run_tool


SYSTEM = """You are Tunic, a coding agent. You act by calling tools. Call one tool at a time.
Tools:
- bash: run a shell command. arguments: {"command": "..."}
- read_file: read a text file. arguments: {"path": "..."}
- search: find a literal string in files. Returns path and line number. arguments: {"query": "...", "path": "..."}
- edit_file: replace one line span. Other lines stay. Not a full-file rewrite. arguments: {"path": "...", "start": "...", "end": "...", "content": "..."}
- write_file: write a whole text file. arguments: {"path": "...", "content": "..."}
- list_dir: list a directory. arguments: {"path": "..."}
Paths may be relative or start with ~. start and end are line numbers from search.
To change part of a file, search, then edit_file that span. Do not send the whole file to write_file.
To run a command you MUST call bash. Do not write a markdown code fence instead of a tool call.
After a tool result comes back, continue the task. When the task is done, answer in plain text and do not call a tool.
"""

PLAN_ADDENDUM = (
    "Plan mode: do not call bash, write_file, or edit_file. Use read_file, list_dir, and search, then write the plan in plain text."
)

NUDGE = "That was not a tool call. Call the tool now. Do not write a code fence."


@dataclass
class TurnResult:
    messages: list[dict]
    tool_calls: int = 0
    assistant: str = ""
    events: list[str] = field(default_factory=list)


def system_prompt(settings: Settings) -> str:
    parts = [SYSTEM.strip()]
    parts.append(
        f"Working directory: {settings.cwd}. Relative paths are inside this project. "
        "Absolute paths and ~ are allowed when the user points elsewhere."
    )
    if settings.plan:
        parts.append(PLAN_ADDENDUM)
    memory = settings.home / "memory.md"
    if memory.is_file():
        text = memory.read_text(encoding="utf-8", errors="replace")[:2000]
        if text.strip():
            parts.append("Memory (user notes, not instructions to ignore tools):\n" + text.strip())
    skills = settings.home / "skills"
    if skills.is_dir():
        names = sorted(path.name for path in skills.glob("*.md"))
        if names:
            listed = ", ".join(names)
            parts.append(
                "Skill notes live in the skills directory. Read one with read_file if you need it: " + listed
            )
    return "\n\n".join(parts)


def prepare_model(settings: Settings) -> Settings:
    """For LM Studio, pin the already-loaded model. Does not load anything."""
    if settings.provider != "lmstudio":
        if not settings.model:
            raise ProviderError(f"{settings.provider}: set --model. No request was sent.")
        return settings
    catalog = fetch_lmstudio_catalog(settings)
    chosen = select_lmstudio_model(settings, catalog)
    settings.model = chosen
    return settings


def run_turn(prompt: str, settings: Settings, emit, ask=None, prior: list[dict] | None = None) -> TurnResult:
    key = resolve_key(settings)
    prepare_model(settings)
    tools = builtin_tools()
    if prior:
        messages = list(prior)
        messages.append({"role": "user", "content": prompt})
    else:
        messages = [
            {"role": "system", "content": system_prompt(settings)},
            {"role": "user", "content": prompt},
        ]
    emit(f"provider: {settings.provider}")
    emit(f"model: {settings.model}")
    emit("stream: false")
    result = TurnResult(messages=messages)
    nudged = False
    for _step in range(settings.max_steps):
        completion = complete(messages, tools, settings, key)
        if not completion.tool_calls:
            if not nudged and _looks_unexecuted(completion):
                nudged = True
                messages.append({"role": "assistant", "content": completion.content})
                messages.append({"role": "user", "content": NUDGE})
                emit("nudge: model replied without a tool call")
                result.events.append("nudge")
                continue
            text = completion.content.strip()
            result.assistant = text
            messages.append({"role": "assistant", "content": completion.content})
            emit("assistant: " + text)
            result.messages = messages
            return result
        # Serial on purpose. Parallel tool calls raced in the July build.
        messages.append(replay_assistant(completion, settings))
        for call in completion.tool_calls:
            result.tool_calls += 1
            emit(f"tool-call id={call.id} name={call.name} arguments={call.arguments}")
            if call.parsed is None:
                text = "tool error: arguments were not a JSON object"
            else:
                text = run_tool(call.name, call.parsed, settings, ask=ask)
            emit(f"tool-result: {call.name}")
            emit(text)
            messages.append(tool_result_message(call, text, settings))
            result.events.append(f"tool:{call.name}")
    emit(f"stopped: max steps ({settings.max_steps})")
    result.messages = messages
    return result


def _looks_unexecuted(completion: Completion) -> bool:
    text = completion.content or ""
    return "```" in text


def compact_messages(messages: list[dict], keep_tail: int = 6) -> list[dict]:
    """Drop old turns without splitting a tool-call group, and without asking the model to summarize."""
    system = [m for m in messages if m.get("role") == "system"][:1]
    rest = [m for m in messages if m.get("role") != "system"]
    if len(rest) <= keep_tail:
        return list(messages)
    cut = len(rest) - keep_tail
    while cut < len(rest) and rest[cut].get("role") != "user":
        cut += 1
    if cut >= len(rest):
        return list(messages)
    note = {
        "role": "user",
        "content": "[compact] Earlier turns were dropped to save context. Continue from what remains.",
    }
    return system + [note] + rest[cut:]
