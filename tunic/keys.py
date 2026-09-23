"""Resolve cloud keys from the environment or a pass entry name. Never print them."""

from __future__ import annotations

import os
import subprocess

from tunic.config import PROVIDERS, ConfigError, Settings


class KeyMissing(ConfigError):
    """A cloud provider was selected and no key could be resolved."""


def _valid_pass_name(name: str) -> bool:
    if not name or name.startswith("-") or "\n" in name or "\x00" in name:
        return False
    return True


def pass_name_for(settings: Settings) -> str | None:
    spec = settings.spec
    env_name = spec.get("pass_env")
    if env_name:
        from_env = os.environ.get(env_name, "").strip()
        if from_env:
            return from_env
    if settings.pass_name:
        return settings.pass_name.strip()
    return None


def read_pass(name: str, runner=None) -> str:
    """Return the secret for a pass entry. Caller must not log the return value."""
    if not _valid_pass_name(name):
        raise KeyMissing(f"pass entry name is not usable: {name!r}")
    if runner is None:
        runner = subprocess.run
    try:
        completed = runner(
            ["pass", "show", name],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise KeyMissing("pass is not installed, so the configured pass entry cannot be read") from exc
    if completed.returncode != 0:
        raise KeyMissing(f"pass entry {name!r} could not be read")
    secret = (completed.stdout or "").splitlines()
    if not secret or not secret[0].strip():
        raise KeyMissing(f"pass entry {name!r} is empty")
    return secret[0].strip()


def resolve_key(settings: Settings, runner=None, account_resolver=None) -> str | None:
    """Return a cloud key, or None when this provider does not need one.

    Local providers return None. The HTTP client sends a dummy bearer token
    and must not forward a cloud key to a LAN server.
    """
    _ = account_resolver
    if not settings.needs_key:
        return None
    spec = settings.spec
    env_name = spec.get("env")
    if env_name:
        value = os.environ.get(env_name, "").strip()
        if value:
            return value
    name = pass_name_for(settings)
    if name:
        return read_pass(name, runner=runner)
    hint_env = env_name or "an API key env var"
    hint_pass = spec.get("pass_env") or "a pass-name env var"
    raise KeyMissing(
        f"{settings.provider}: no API key. Set {hint_env}, or set {hint_pass} "
        "to a pass entry name (the name, not the secret). "
        "A profile may also set \"pass\" to that name. No request was sent."
    )


def key_status(settings: Settings) -> str:
    """Human status that never includes the secret or a pass file's contents."""
    if not settings.needs_key:
        return f"{settings.provider}: no key required"
    spec = settings.spec
    env_name = spec.get("env")
    if env_name and os.environ.get(env_name, "").strip():
        return f"{settings.provider}: key configured via {env_name}"
    name = pass_name_for(settings)
    if name:
        if not _valid_pass_name(name):
            return f"{settings.provider}: pass entry name is not usable"
        return f"{settings.provider}: pass entry name configured ({name})"
    return f"{settings.provider}: key missing"


def cloud_provider_names() -> list[str]:
    return [name for name, spec in sorted(PROVIDERS.items()) if spec["needs_key"]]
