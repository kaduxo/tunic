"""Load matching skill notes into the system prompt. No secrets."""

from __future__ import annotations

import re
from pathlib import Path

_MAX_SKILLS = 2
_MAX_BODY = 1500
_STOP = frozenset(
    {
        "a",
        "an",
        "and",
        "for",
        "from",
        "how",
        "in",
        "of",
        "on",
        "or",
        "skill",
        "the",
        "this",
        "that",
        "to",
        "use",
        "with",
    }
)
_TOKEN = re.compile(r"[a-z0-9]+")


def skill_files(home: Path, cwd: Path) -> list[Path]:
    roots = [
        home / "skills",
        cwd / ".tunic" / "skills",
    ]
    found: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        if not root.is_dir():
            continue
        for path in sorted(root.glob("*.md")):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            found.append(path)
        for path in sorted(root.glob("*/SKILL.md")):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            found.append(path)
    return found


def catalog_names(home: Path, cwd: Path) -> list[str]:
    names = []
    for path in skill_files(home, cwd):
        names.append(_skill_meta(path)["name"])
    return names


def skill_by_name(home: Path, cwd: Path, name: str) -> tuple[str, str] | None:
    want = (name or "").strip().lower()
    if not want:
        return None
    for path in skill_files(home, cwd):
        meta = _skill_meta(path)
        if meta["name"].lower() == want:
            return meta["name"], meta["body"]
    return None


def matching_skills(home: Path, cwd: Path, prompt: str) -> list[tuple[str, str]]:
    """Return (name, body) for skills whose name or description matches the prompt."""
    hay = (prompt or "").lower()
    if not hay.strip():
        return []
    ranked: list[tuple[int, str, str]] = []
    for path in skill_files(home, cwd):
        meta = _skill_meta(path)
        score = _score(meta, hay)
        if score <= 0:
            continue
        ranked.append((score, meta["name"], meta["body"]))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    loaded: list[tuple[str, str]] = []
    for _rank, name, body in ranked[:_MAX_SKILLS]:
        loaded.append((name, body[:_MAX_BODY]))
    return loaded


def _skill_meta(path: Path) -> dict[str, str]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        text = ""
    front, body = _split_frontmatter(text)
    name = front.get("name") or (path.parent.name if path.name == "SKILL.md" else path.stem)
    description = front.get("description") or _first_line(body)
    return {"name": name, "description": description, "body": body.strip()}


def _split_frontmatter(text: str) -> tuple[dict[str, str], str]:
    if not text.startswith("---"):
        return {}, text
    rest = text[3:].lstrip("\n")
    end = rest.find("\n---")
    if end < 0:
        return {}, text
    raw = rest[:end]
    body = rest[end + 4 :].lstrip("\n")
    meta: dict[str, str] = {}
    for line in raw.splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        meta[key.strip().lower()] = value.strip().strip('"').strip("'")
    return meta, body


def _first_line(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip().lstrip("#").strip()
        if stripped:
            return stripped
    return ""


def _score(meta: dict[str, str], hay: str) -> int:
    score = 0
    name = meta["name"].lower()
    if name and name in hay:
        score += 5
    for token in _tokens(name + " " + meta["description"]):
        if token in hay:
            score += 1
    return score


def _tokens(text: str) -> list[str]:
    return [token for token in _TOKEN.findall(text.lower()) if token not in _STOP and len(token) >= 4]
