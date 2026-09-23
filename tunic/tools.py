"""Tools a small model can actually fill. Schemas are real JSON Schema objects."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from tunic.config import Settings


OUTPUT_CAP = 8000
_READ_ONLY = frozenset({"read_file", "list_dir", "search"})
_SKIP_DIRS = frozenset({".git", "__pycache__", ".venv", "node_modules", ".hg"})
_MAX_MATCHES = 40
_MAX_FILES = 2000
_MAX_FILE_BYTES = 1_000_000


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    schema: dict

    def openai_tool(self) -> dict:
        # The schema goes in function.parameters itself. Do not wrap it again
        # and do not rename it to inputSchema on this path. That drop is how
        # the July build made the model emit {}.
        schema = self.schema
        if schema.get("type") != "object" or "properties" not in schema:
            raise RuntimeError(f"{self.name} schema is not a JSON Schema object")
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": schema,
            },
        }

    def anthropic_tool(self) -> dict:
        schema = self.schema
        if schema.get("type") != "object" or "properties" not in schema:
            raise RuntimeError(f"{self.name} schema is not a JSON Schema object")
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": schema,
        }


def _object(properties: dict, required: list[str]) -> dict:
    # additionalProperties is valid JSON Schema. This LM Studio's tool grammar
    # was faulting, so local schemas stay at type + properties + required,
    # which is what the model must see. The omission is not a proven fix.
    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }


def builtin_tools() -> list[Tool]:
    return [
        Tool(
            name="bash",
            description="Run a shell command and return stdout, stderr, and the exit code.",
            schema=_object(
                {
                    "command": {
                        "type": "string",
                        "description": "The shell command to run.",
                    }
                },
                ["command"],
            ),
        ),
        Tool(
            name="read_file",
            description="Read a text file. Relative paths and ~ are allowed.",
            schema=_object(
                {
                    "path": {
                        "type": "string",
                        "description": "File path. Relative paths and ~ are allowed.",
                    }
                },
                ["path"],
            ),
        ),
        Tool(
            name="write_file",
            description="Write a whole text file, creating parent directories. Relative paths and ~ are allowed.",
            schema=_object(
                {
                    "path": {
                        "type": "string",
                        "description": "File path. Relative paths and ~ are allowed.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Full new contents of the file.",
                    },
                },
                ["path", "content"],
            ),
        ),
        Tool(
            name="edit_file",
            description="Replace one line span. Other lines stay unchanged. Not a full-file rewrite.",
            schema=_object(
                {
                    "path": {
                        "type": "string",
                        "description": "File path. Relative paths and ~ are allowed.",
                    },
                    "start": {
                        "type": "string",
                        "description": "First line of the span, 1-based, from search.",
                    },
                    "end": {
                        "type": "string",
                        "description": "Last line of the span, 1-based. Same as start to change one line.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Replacement text for that span only.",
                    },
                },
                ["path", "start", "end", "content"],
            ),
        ),
        Tool(
            name="list_dir",
            description="List entries in a directory. Relative paths and ~ are allowed.",
            schema=_object(
                {
                    "path": {
                        "type": "string",
                        "description": "Directory path. Use . for the working directory.",
                    }
                },
                ["path"],
            ),
        ),
        Tool(
            name="search",
            description="Search file contents for a literal string. Returns each match as path and line number.",
            schema=_object(
                {
                    "query": {
                        "type": "string",
                        "description": "Literal string to find. Not a regular expression.",
                    },
                    "path": {
                        "type": "string",
                        "description": "File or directory. Use . for the working directory.",
                    },
                },
                ["query", "path"],
            ),
        ),
    ]


def tools_by_name() -> dict[str, Tool]:
    return {tool.name: tool for tool in builtin_tools()}


def resolve_user_path(raw: str, cwd: Path) -> Path:
    """Expand ~ and resolve relative paths. Do not reject either form."""
    if raw is None or not str(raw).strip():
        raise ValueError("path is empty")
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        path = cwd / path
    return path.resolve()


def _cap(text: str) -> str:
    if len(text) <= OUTPUT_CAP:
        return text
    return text[:OUTPUT_CAP] + "\n[truncated]"


def _permission(name: str, settings: Settings, ask, arguments: dict | None) -> str | None:
    if name in _READ_ONLY:
        return None
    if settings.plan:
        return "permission denied: plan mode is read-only (bash, write_file, and edit_file are off)"
    if settings.yes:
        return None
    if ask is not None and _granted(ask, name, arguments or {}):
        return None
    if ask is None:
        return "permission denied: mutating tools need --yes, or run in a terminal and answer y"
    return "permission denied"


def _granted(ask, name: str, arguments: dict) -> bool:
    try:
        return bool(ask(name, arguments))
    except TypeError:
        return bool(ask(name))


def run_tool(name: str, arguments: dict | None, settings: Settings, ask=None) -> str:
    denied = _permission(name, settings, ask, arguments)
    if denied:
        return denied
    if not isinstance(arguments, dict):
        return f"tool error: {name} arguments were not a JSON object"
    try:
        if name == "bash":
            return _bash(arguments, settings)
        if name == "read_file":
            return _read(arguments, settings)
        if name == "write_file":
            return _write(arguments, settings)
        if name == "edit_file":
            return _edit(arguments, settings)
        if name == "list_dir":
            return _list(arguments, settings)
        if name == "search":
            return _search(arguments, settings)
    except ValueError as exc:
        return f"tool error: {exc}"
    except OSError as exc:
        return f"tool error: {exc.strerror or exc}"
    return f"tool error: unknown tool {name}"


def _bash(arguments: dict, settings: Settings) -> str:
    command = arguments.get("command")
    if not isinstance(command, str) or not command.strip():
        return "tool error: bash requires a command string"
    try:
        completed = subprocess.run(
            command,
            shell=True,
            cwd=str(settings.cwd),
            capture_output=True,
            text=True,
            timeout=30,
            executable="/bin/bash",
        )
    except subprocess.TimeoutExpired:
        return "tool error: bash timed out after 30s"
    stdout = completed.stdout or ""
    stderr = completed.stderr or ""
    parts = [f"exit={completed.returncode}", "stdout:", _cap(stdout).rstrip("\n")]
    if stderr.strip():
        parts.extend(["stderr:", _cap(stderr).rstrip("\n")])
    return "\n".join(parts)


def _read(arguments: dict, settings: Settings) -> str:
    path = resolve_user_path(str(arguments.get("path", "")), settings.cwd)
    if not path.is_file():
        return f"tool error: not a file: {path}"
    data = path.read_bytes()
    if b"\x00" in data[:8192]:
        return f"tool error: {path} looks binary"
    text = data.decode("utf-8", errors="replace")
    return _cap(text)


def _write(arguments: dict, settings: Settings) -> str:
    path = resolve_user_path(str(arguments.get("path", "")), settings.cwd)
    content = arguments.get("content")
    if not isinstance(content, str):
        return "tool error: write_file requires content as a string"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return f"wrote {path} ({len(content)} chars)"


def _list(arguments: dict, settings: Settings) -> str:
    path = resolve_user_path(str(arguments.get("path", "")), settings.cwd)
    if not path.is_dir():
        return f"tool error: not a directory: {path}"
    names = sorted(entry.name for entry in path.iterdir())
    if len(names) > 200:
        shown = names[:200]
        return "\n".join(shown) + f"\n[{len(names) - 200} more]"
    return "\n".join(names) if names else "(empty)"


def _line_number(value, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a line number")
    if isinstance(value, int):
        number = value
    elif isinstance(value, float) and value.is_integer():
        number = int(value)
    elif isinstance(value, str) and value.strip().isdigit():
        number = int(value.strip())
    else:
        raise ValueError(f"{label} must be a line number")
    if number < 1:
        raise ValueError(f"{label} must be >= 1")
    return number


def _split_keep(text: str) -> list[str]:
    """Split on \\n and keep each line's ending, including a last line with none."""
    if text == "":
        return []
    ends = text.endswith("\n")
    parts = text.split("\n")
    if ends:
        parts = parts[:-1]
    lines = [part + "\n" for part in parts[:-1]]
    if parts:
        lines.append(parts[-1] + ("\n" if ends else ""))
    return lines


def _replacement_lines(content: str) -> list[str]:
    if content == "":
        return []
    if not content.endswith("\n"):
        content += "\n"
    return [part + "\n" for part in content.split("\n")[:-1]]


def _edit(arguments: dict, settings: Settings) -> str:
    path = resolve_user_path(str(arguments.get("path", "")), settings.cwd)
    if not path.is_file():
        return f"tool error: not a file: {path}"
    content = arguments.get("content")
    if not isinstance(content, str):
        return "tool error: edit_file requires content as a string"
    try:
        start = _line_number(arguments.get("start"), "start")
        end_raw = arguments.get("end", None)
        end = start if end_raw in (None, "") else _line_number(end_raw, "end")
    except ValueError as exc:
        return f"tool error: {exc}"
    if end < start:
        return "tool error: end is before start"
    data = path.read_bytes()
    if b"\x00" in data[:8192]:
        return f"tool error: {path} looks binary"
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return f"tool error: {path} is not utf-8 text"
    lines = _split_keep(text)
    if start > len(lines) or end > len(lines):
        return f"tool error: line {start}-{end} is outside {path} ({len(lines)} lines)"
    # Splice the span. Lines outside it are copied, not rewritten by the model.
    new_lines = lines[: start - 1] + _replacement_lines(content) + lines[end:]
    path.write_text("".join(new_lines), encoding="utf-8")
    return f"edited {path} lines {start}-{end}"


def _display_path(path: Path, cwd: Path) -> str:
    try:
        rel = path.resolve().relative_to(cwd.resolve())
    except ValueError:
        return path.as_posix()
    text = rel.as_posix()
    return text or path.name


def _search_file(path: Path, query: str, cwd: Path, limit: int) -> tuple[list[str], bool]:
    try:
        if not path.is_file():
            return [], False
        if path.stat().st_size > _MAX_FILE_BYTES:
            return [], False
        data = path.read_bytes()
    except OSError:
        return [], False
    if b"\x00" in data[:8192]:
        return [], False
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return [], False
    shown: list[str] = []
    display = _display_path(path, cwd)
    for number, line in enumerate(text.splitlines(), start=1):
        if query not in line:
            continue
        preview = line if len(line) <= 200 else line[:200] + "…"
        if len(shown) >= limit:
            return shown, True
        shown.append(f"{display}:{number}:{preview}")
    return shown, False


def _iter_files(root: Path):
    if root.is_file():
        yield root
        return
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = sorted(name for name in dirnames if name not in _SKIP_DIRS)
        for name in sorted(filenames):
            yield Path(dirpath) / name


def _search(arguments: dict, settings: Settings) -> str:
    query = arguments.get("query")
    if not isinstance(query, str) or query == "":
        return "tool error: search requires a query string"
    if "\n" in query or "\r" in query:
        return "tool error: search query must be a single line"
    raw = arguments.get("path")
    if raw is None or raw == "":
        raw = "."
    if not isinstance(raw, str):
        return "tool error: search path must be a string"
    root = resolve_user_path(raw, settings.cwd)
    if not root.exists():
        return f"tool error: not found: {root}"
    if root.is_file() and root.stat().st_size > _MAX_FILE_BYTES:
        return f"tool error: {root} is too large to search"
    if not root.is_file() and not root.is_dir():
        return f"tool error: not a file or directory: {root}"
    matches: list[str] = []
    truncated = False
    seen = 0
    for file_path in _iter_files(root):
        seen += 1
        if seen > _MAX_FILES:
            truncated = True
            break
        found, hit = _search_file(file_path, query, settings.cwd, _MAX_MATCHES - len(matches))
        matches.extend(found)
        if hit or len(matches) >= _MAX_MATCHES:
            truncated = True
            break
    if not matches:
        return "no matches"
    text = "\n".join(matches)
    if truncated:
        text += "\n[more matches omitted]"
    return _cap(text)
