"""Tools a small model can actually fill. Schemas are real JSON Schema objects."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from tunic.config import Settings


OUTPUT_CAP = 8000


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
            description="Write a text file, creating parent directories. Relative paths and ~ are allowed.",
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
    if name in ("read_file", "list_dir"):
        return None
    if settings.plan:
        return "permission denied: plan mode is read-only (bash and write_file are off)"
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
        if name == "list_dir":
            return _list(arguments, settings)
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
