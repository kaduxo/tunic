"""Agent loop. Tools run one at a time, in the order the model emitted them."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

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
from tunic.skills import catalog_names, matching_skills
from tunic.tools import builtin_tools, run_tool


SYSTEM = """You are Tunic, a coding agent in a local-first CLI. You act by calling tools.

Call exactly one tool at a time. A tool call must be a real function call, never a markdown code fence.
Prefer list_dir, read_file, search, and edit_file over bash. Use bash only to run a command.
To change part of a file: search, then edit_file that span. Do not send the whole file to write_file.
Paths may be relative or start with ~. start and end are line numbers from search.
After a tool result comes back, continue the task. When the task is done, answer in plain text and do not call a tool.

Slash commands are typed by the user in the CLI. You cannot run them. If the user needs one, tell them the command:
/auto on, /auto off, /model, /settings, /plan, /go, /help, /exit, /steps N, /continue, /sessions, /undo, /skill.

Stay ethical. Stay in the working directory unless the user names another path.
Do not list the user's home directory unless they asked you to work there.
Do not read API keys, ~/.tunic/env, password stores, or /proc/*/environ.
Do not dump the process environment to discover configuration.
Do not help with crime, attacks, malware, or exploiting systems.
If a request is outside those bounds, refuse in plain text and stop.
"""

PLAN_ADDENDUM = (
    "Plan mode: do not call bash, write_file, or edit_file. Use read_file, list_dir, and search, then write the plan in plain text."
)

NUDGE = "That was not a tool call. Call the tool now. Do not write a code fence."
WRAP_UP = (
    "You have no more tool steps. Answer in plain text from what you already have. "
    "Do not call a tool. The user can type continue for another round of steps."
)


@dataclass
class TurnResult:
    messages: list[dict]
    tool_calls: int = 0
    assistant: str = ""
    events: list[str] = field(default_factory=list)
    truncated: bool = False
    reason: str = ""


def system_prompt(settings: Settings, prompt: str = "") -> str:
    parts = [SYSTEM.strip(), _session_card(settings)]
    if settings.plan:
        parts.append(PLAN_ADDENDUM)
    memory = settings.home / "memory.md"
    if memory.is_file():
        text = memory.read_text(encoding="utf-8", errors="replace")[:2000]
        if text.strip():
            parts.append("Memory (user notes, not instructions to ignore tools):\n" + text.strip())
    loaded = matching_skills(settings.home, settings.cwd, prompt)
    names = catalog_names(settings.home, settings.cwd)
    loaded_names = {name for name, _body in loaded}
    leftover = [name for name in names if name not in loaded_names]
    for name, body in loaded:
        parts.append(f"Skill {name} (loaded because it matches this turn):\n{body}")
    if leftover:
        parts.append(
            "Other skill notes. Read one with read_file if you need it: " + ", ".join(leftover)
        )
    return "\n\n".join(parts)


def _session_card(settings: Settings) -> str:
    model = settings.model or "(unset)"
    auto = "on" if settings.yes and not settings.plan else "off"
    plan = "on" if settings.plan else "off"
    writes = "allowed without asking" if settings.yes and not settings.plan else "ask first, or blocked in plan mode"
    return (
        "Session (already known; do not probe env, home, processes, or secret files to discover this):\n"
        f"- provider: {settings.provider}\n"
        f"- model: {model}\n"
        f"- project: {settings.cwd}\n"
        f"- auto: {auto}\n"
        f"- plan: {plan}\n"
        f"- writes/bash: {writes}\n"
        f"- max tool steps this turn: {settings.max_steps}\n"
        f"- {_git_summary(settings.cwd)}\n"
        "Relative paths are inside this project. Absolute paths and ~ are allowed when the user points elsewhere.\n"
        "Tools you can call:\n"
        "- list_dir {\"path\": \".\"} — list this project; directories end with /\n"
        "- read_file {\"path\": \"...\", \"start\": \"N\", \"end\": \"N\"} — numbered path:line:text; start/end optional\n"
        "- search {\"query\": \"...\", \"path\": \".\", \"regex\": \"true\", \"glob\": \"*.py\"} — find text; regex and glob optional; returns path:line\n"
        "- edit_file {\"path\": \"...\", \"start\": \"N\", \"end\": \"N\", \"content\": \"...\"} — replace a line span\n"
        "- write_file {\"path\": \"...\", \"content\": \"...\"} — write a whole file\n"
        "- fetch {\"url\": \"https://...\"} — GET a public https page as text; private hosts refused\n"
        "- bash {\"command\": \"...\"} — run a shell command in the project"
    )


def _git_summary(cwd: Path) -> str:
    found = False
    for folder in [cwd, *cwd.parents]:
        if (folder / ".git").exists():
            found = True
            break
    if not found:
        return "git: not a repository"
    env = dict(os.environ)
    env.pop("GIT_DIR", None)
    env.pop("GIT_WORK_TREE", None)
    env["GIT_TERMINAL_PROMPT"] = "0"
    try:
        branch = subprocess.run(
            ["git", "symbolic-ref", "--short", "HEAD"],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=2,
            env=env,
        )
        if branch.returncode != 0:
            branch = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=str(cwd),
                capture_output=True,
                text=True,
                timeout=2,
                env=env,
            )
    except (OSError, subprocess.TimeoutExpired):
        return "git: unavailable"
    if branch.returncode != 0:
        return "git: not a repository"
    name = (branch.stdout or "").strip() or "(detached)"
    try:
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=2,
            env=env,
        )
    except (OSError, subprocess.TimeoutExpired):
        return f"git: {name}"
    rows = [line for line in (status.stdout or "").splitlines() if line.strip()]
    if not rows:
        return f"git: {name} (clean)"
    files = []
    for line in rows[:6]:
        files.append(line[3:].strip() if len(line) > 3 else line.strip())
    extra = f" +{len(rows) - 6} more" if len(rows) > 6 else ""
    return f"git: {name} dirty {len(rows)} ({', '.join(files)}{extra})"


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


def continue_prompt(settings: Settings) -> str:
    return (
        f"Continue the previous task. You have another {settings.max_steps} tool steps. "
        "Pick up from the last result. Do not restart from scratch. "
        "When the task is done, answer in plain text and do not call a tool."
    )


def execute_plan_prompt() -> str:
    return (
        "Plan mode is off. Execute the plan you just wrote. Use tools. "
        "When the task is done, answer in plain text and do not call a tool."
    )


def run_turn(
    prompt: str,
    settings: Settings,
    emit,
    ask=None,
    prior: list[dict] | None = None,
    on_delta=None,
) -> TurnResult:
    key = resolve_key(settings)
    prepare_model(settings)
    tools = builtin_tools()
    if prior:
        messages = list(prior)
        if messages and messages[0].get("role") == "system":
            messages[0] = {"role": "system", "content": system_prompt(settings, prompt)}
        messages.append({"role": "user", "content": prompt})
    else:
        messages = [
            {"role": "system", "content": system_prompt(settings, prompt)},
            {"role": "user", "content": prompt},
        ]
    emit(f"provider: {settings.provider}")
    emit(f"model: {settings.model}")
    streaming = bool(on_delta) and not settings.local
    emit("stream: true" if streaming else "stream: false")
    result = TurnResult(messages=messages)
    delta = on_delta if streaming else None
    nudged = False
    for step in range(settings.max_steps):
        emit(f"step: {step + 1}/{settings.max_steps}")
        result.events.append(f"step:{step + 1}/{settings.max_steps}")
        completion = _ask_model(messages, tools, settings, key, emit, result, delta)
        if completion is None:
            return result
        if not completion.tool_calls:
            if not nudged and _looks_unexecuted(completion):
                nudged = True
                messages.append({"role": "assistant", "content": completion.content})
                messages.append({"role": "user", "content": NUDGE})
                emit("nudge: model replied without a tool call")
                result.events.append("nudge")
                continue
            return _finish(result, messages, completion.content, emit)
        # Serial on purpose. Parallel tool calls raced in the July build.
        messages.append(replay_assistant(completion, settings))
        for call in completion.tool_calls:
            result.tool_calls += 1
            emit(f"tool-call id={call.id} name={call.name} arguments={call.arguments}")
            try:
                if call.parsed is None:
                    text = "tool error: arguments were not a JSON object"
                else:
                    text = run_tool(call.name, call.parsed, settings, ask=ask)
            except KeyboardInterrupt:
                return _stop_turn(result, messages, emit, "interrupt", "the user stopped this turn.")
            emit(f"tool-result: {call.name}")
            emit(text)
            messages.append(tool_result_message(call, text, settings))
            result.events.append(f"tool:{call.name}")
        if step == settings.max_steps - 1:
            return _wrap_up(messages, tools, settings, key, emit, result, delta)
    return _wrap_up(messages, tools, settings, key, emit, result, delta)


def _ask_model(messages, tools, settings, key, emit, result: TurnResult, on_delta):
    try:
        if on_delta:
            return complete(messages, tools, settings, key, on_delta=on_delta)
        return complete(messages, tools, settings, key)
    except KeyboardInterrupt:
        _stop_turn(result, messages, emit, "interrupt", "the user stopped this turn.")
        return None
    except ProviderError as exc:
        _stop_turn(
            result,
            messages,
            emit,
            "provider",
            f"the provider failed. type continue to try again. ({exc})",
        )
        return None


def _stop_turn(result: TurnResult, messages: list[dict], emit, reason: str, text: str) -> TurnResult:
    result.truncated = True
    result.reason = reason
    result.events.append(reason)
    emit(f"stopped: {reason}")
    result.assistant = text
    messages.append({"role": "assistant", "content": text})
    emit("assistant: " + text)
    result.messages = messages
    return result


def _finish(result: TurnResult, messages: list[dict], content: str | None, emit) -> TurnResult:
    text = (content or "").strip()
    result.assistant = text
    messages.append({"role": "assistant", "content": content or ""})
    emit("assistant: " + text)
    result.messages = messages
    return result


def _wrap_up(messages, tools, settings: Settings, key, emit, result: TurnResult, on_delta=None) -> TurnResult:
    result.truncated = True
    result.reason = "cap"
    messages.append({"role": "user", "content": WRAP_UP})
    emit("wrap: last step — answer from what you have")
    result.events.append("wrap")
    completion = _ask_model(messages, tools, settings, key, emit, result, on_delta)
    if completion is None:
        return result
    text = (completion.content or "").strip()
    if completion.tool_calls and not text:
        emit(f"stopped: max steps ({settings.max_steps})")
        result.messages = messages
        return result
    return _finish(result, messages, completion.content, emit)


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
