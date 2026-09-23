"""Session transcripts under ~/.tunic/sessions. No API keys are written."""

from __future__ import annotations

import json
import time
from pathlib import Path

from tunic.config import Settings


def sessions_dir(settings: Settings) -> Path:
    path = settings.home / "sessions"
    path.mkdir(parents=True, exist_ok=True)
    return path


def session_path(settings: Settings, name: str) -> Path:
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in name).strip("-")
    if not safe:
        raise ValueError("session name is empty")
    return sessions_dir(settings) / f"{safe}.jsonl"


def save_session(settings: Settings, name: str, messages: list[dict]) -> Path:
    path = session_path(settings, name)
    lines = [
        json.dumps(
            {
                "type": "meta",
                "provider": settings.provider,
                "model": settings.model,
                "saved_at": int(time.time()),
            }
        )
    ]
    for message in messages:
        stored = {
            "role": message.get("role"),
            "content": message.get("content"),
        }
        if message.get("tool_calls"):
            stored["tool_calls"] = message["tool_calls"]
        if message.get("tool_call_id"):
            stored["tool_call_id"] = message["tool_call_id"]
        if message.get("name"):
            stored["name"] = message["name"]
        lines.append(json.dumps(stored))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def load_session(settings: Settings, name: str) -> list[dict]:
    path = session_path(settings, name)
    if not path.is_file():
        return []
    messages = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        if item.get("type") == "meta":
            continue
        messages.append(item)
    return messages
