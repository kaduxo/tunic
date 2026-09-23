"""Paths, providers, and the on-disk config. No secrets live here."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse


# Local default. Override with --base-url, TUNIC_BASE_URL, or config.json.
LMSTUDIO_BASE_URL = "http://127.0.0.1:1234/v1"

LOCAL_PROVIDERS = frozenset({"lmstudio", "ollama", "vllm", "custom"})

# kind, default base url, whether a cloud key is required, env var, pass-name env var
PROVIDERS = {
    "lmstudio": {
        "kind": "openai",
        "base_url": LMSTUDIO_BASE_URL,
        "needs_key": False,
        "env": None,
        "pass_env": None,
    },
    "ollama": {
        "kind": "openai",
        "base_url": "http://127.0.0.1:11434/v1",
        "needs_key": False,
        "env": None,
        "pass_env": None,
    },
    "vllm": {
        "kind": "openai",
        "base_url": "http://127.0.0.1:8000/v1",
        "needs_key": False,
        "env": None,
        "pass_env": None,
    },
    "openai": {
        "kind": "openai",
        "base_url": "https://api.openai.com/v1",
        "needs_key": True,
        "env": "OPENAI_API_KEY",
        "pass_env": "TUNIC_OPENAI_PASS",
    },
    "xai": {
        "kind": "openai",
        "base_url": "https://api.x.ai/v1",
        "needs_key": True,
        "env": "XAI_API_KEY",
        "pass_env": "TUNIC_XAI_PASS",
    },
    "anthropic": {
        "kind": "anthropic",
        "base_url": "https://api.anthropic.com",
        "needs_key": True,
        "env": "ANTHROPIC_API_KEY",
        "pass_env": "TUNIC_ANTHROPIC_PASS",
    },
    "openrouter": {
        "kind": "openai",
        "base_url": "https://openrouter.ai/api/v1",
        "needs_key": True,
        "env": "OPENROUTER_API_KEY",
        "pass_env": "TUNIC_OPENROUTER_PASS",
    },
    "groq": {
        "kind": "openai",
        "base_url": "https://api.groq.com/openai/v1",
        "needs_key": True,
        "env": "GROQ_API_KEY",
        "pass_env": "TUNIC_GROQ_PASS",
    },
    "custom": {
        "kind": "openai",
        "base_url": "",
        "needs_key": False,
        "env": "TUNIC_CUSTOM_API_KEY",
        "pass_env": "TUNIC_CUSTOM_PASS",
    },
}


class ConfigError(Exception):
    """User-facing configuration problem. The message must never contain a secret."""


@dataclass
class Settings:
    provider: str = "lmstudio"
    model: str = ""
    base_url: str = LMSTUDIO_BASE_URL
    max_steps: int = 8
    max_tokens: int = 1024
    temperature: float = 0.2
    yes: bool = False
    plan: bool = False
    cwd: Path = field(default_factory=Path.cwd)
    session: str | None = None
    allow_load: bool = False
    no_color: bool = False
    pass_name: str | None = None
    auth: str | None = None
    profile: str | None = None
    home: Path = field(default_factory=lambda: Path.home() / ".tunic")
    http_timeout: float = 180.0

    @property
    def spec(self) -> dict:
        try:
            return PROVIDERS[self.provider]
        except KeyError as exc:
            known = ", ".join(sorted(PROVIDERS))
            raise ConfigError(f"unknown provider {self.provider!r}. Known: {known}") from exc

    @property
    def kind(self) -> str:
        return self.spec["kind"]

    @property
    def local(self) -> bool:
        return self.provider in LOCAL_PROVIDERS

    @property
    def needs_key(self) -> bool:
        return bool(self.spec["needs_key"])


def tunic_home() -> Path:
    override = os.environ.get("TUNIC_HOME")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".tunic"


def config_path() -> Path:
    return tunic_home() / "config.json"


def load_config_file() -> dict:
    path = config_path()
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"config is not JSON: {path}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"config must be a JSON object: {path}")
    return data


def ensure_home() -> Path:
    home = tunic_home()
    home.mkdir(parents=True, exist_ok=True)
    (home / "sessions").mkdir(exist_ok=True)
    (home / "skills").mkdir(exist_ok=True)
    path = home / "config.json"
    if not path.exists():
        path.write_text(
            json.dumps(
                {
                    "provider": "lmstudio",
                    "model": "",
                    "base_url": LMSTUDIO_BASE_URL,
                    "profiles": {
                        "local": {"provider": "lmstudio", "base_url": LMSTUDIO_BASE_URL},
                        "openai": {"provider": "openai", "pass": ""},
                        "xai": {"provider": "xai", "pass": ""},
                        "anthropic": {"provider": "anthropic", "pass": ""},
                    },
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    return home


def merged_provider(file_cfg: dict, profile_cfg: dict):
    if profile_cfg.get("provider"):
        return profile_cfg["provider"]
    return file_cfg.get("provider")


def scoped_value(key: str, provider: str, file_cfg: dict, profile_cfg: dict):
    """Profile fields apply to the selected profile. Top-level fields apply only to the saved provider."""
    if key in profile_cfg and profile_cfg[key] not in (None, ""):
        return profile_cfg[key]
    saved = file_cfg.get("provider") or "lmstudio"
    if provider == saved and file_cfg.get(key) not in (None, ""):
        return file_cfg[key]
    return None


def _pick(cli_value, env_name: str, file_value, default):
    if cli_value not in (None, ""):
        return cli_value
    env_value = os.environ.get(env_name)
    if env_value:
        return env_value
    if file_value not in (None, ""):
        return file_value
    return default


def resolve_settings(
    *,
    provider: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    profile: str | None = None,
    yes: bool = False,
    plan: bool = False,
    cwd: str | None = None,
    session: str | None = None,
    allow_load: bool = False,
    no_color: bool = False,
    max_steps: int | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
    home: Path | None = None,
) -> Settings:
    file_cfg = load_config_file()
    chosen_profile = profile or os.environ.get("TUNIC_PROFILE") or ""
    profile_cfg = {}
    if chosen_profile:
        profiles = file_cfg.get("profiles") or {}
        if not isinstance(profiles, dict) or chosen_profile not in profiles:
            raise ConfigError(f"unknown profile {chosen_profile!r}")
        raw = profiles[chosen_profile]
        if not isinstance(raw, dict):
            raise ConfigError(f"profile {chosen_profile!r} must be an object")
        profile_cfg = raw

    def merged(key: str):
        if key in profile_cfg and profile_cfg[key] not in (None, ""):
            return profile_cfg[key]
        if key in file_cfg and file_cfg[key] not in (None, ""):
            return file_cfg[key]
        return None

    name = _pick(provider, "TUNIC_PROVIDER", merged_provider(file_cfg, profile_cfg), "lmstudio")
    if name not in PROVIDERS:
        known = ", ".join(sorted(PROVIDERS))
        raise ConfigError(f"unknown provider {name!r}. Known: {known}")
    spec = PROVIDERS[name]
    # A saved base_url/model belongs to the provider it was saved for.
    # Otherwise `tunic --provider openai` would inherit the LM Studio URL
    # and could send a cloud key to the LAN.
    url = _pick(
        base_url,
        "TUNIC_BASE_URL",
        scoped_value("base_url", name, file_cfg, profile_cfg),
        spec["base_url"],
    )
    url = str(url or "").rstrip("/")
    if name == "custom" and not url:
        raise ConfigError("custom provider needs --base-url or TUNIC_BASE_URL")
    if name != "custom" and not url:
        url = spec["base_url"]

    pass_name = scoped_value("pass", name, file_cfg, profile_cfg)
    if isinstance(pass_name, str):
        pass_name = pass_name.strip() or None
    auth = None

    model_value = _pick(model, "TUNIC_MODEL", scoped_value("model", name, file_cfg, profile_cfg), "")

    work = Path(cwd).expanduser() if cwd else Path.cwd()
    if not work.is_dir():
        raise ConfigError(f"cwd is not a directory: {work}")

    steps = max_steps if max_steps is not None else int(merged("max_steps") or 8)
    tokens = max_tokens if max_tokens is not None else int(merged("max_tokens") or 1024)
    temp = temperature if temperature is not None else float(merged("temperature") or 0.2)
    if steps < 1:
        raise ConfigError("--max-steps must be >= 1")
    if tokens < 1:
        raise ConfigError("--max-tokens must be >= 1")

    return Settings(
        provider=name,
        model=str(model_value or ""),
        base_url=url,
        max_steps=steps,
        max_tokens=tokens,
        temperature=temp,
        yes=yes,
        plan=plan,
        cwd=work.resolve(),
        session=session,
        allow_load=allow_load,
        no_color=no_color or not _stdout_is_tty(),
        pass_name=pass_name,
        auth=auth,
        profile=chosen_profile or None,
        home=home or tunic_home(),
        http_timeout=float(os.environ.get("TUNIC_HTTP_TIMEOUT", "180")),
    )


def _stdout_is_tty() -> bool:
    try:
        return os.isatty(1)
    except OSError:
        return False


def origin_of(base_url: str) -> str:
    parsed = urlparse(base_url)
    if not parsed.scheme or not parsed.netloc:
        raise ConfigError(f"base url is not absolute: {base_url}")
    return f"{parsed.scheme}://{parsed.netloc}"


def _usable_model(model: str) -> str:
    text = (model or "").strip()
    if not text:
        return ""
    if any(ch.isspace() for ch in text) or text.startswith(("sk-", "eyJ")) or len(text) > 128:
        raise ConfigError("model id is not usable. No request was sent.")
    return text


def _usable_pass_name(name: str | None) -> str | None:
    if name is None:
        return None
    text = str(name).strip()
    if not text:
        return None
    if any(ch.isspace() for ch in text) or text.startswith(("sk-", "eyJ")) or len(text) > 80:
        raise ConfigError("pass entry name is not usable. Give the name, not the secret.")
    return text


def save_choice(
    home: Path,
    *,
    provider: str,
    model: str,
    pass_name: str | None = None,
    auth: str | None = None,
) -> None:
    """Write the user default. Never write a key or a token.

    ``auth`` is accepted and discarded. Cloud access is an env var or a pass
    entry name. This function does not read or write an account store.
    """
    _ = auth
    if provider not in PROVIDERS:
        known = ", ".join(sorted(PROVIDERS))
        raise ConfigError(f"unknown provider {provider!r}. Known: {known}")
    model_value = _usable_model(model)
    pass_value = _usable_pass_name(pass_name)
    home.mkdir(parents=True, exist_ok=True)
    path = home / "config.json"
    data: dict = {}
    if path.is_file():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ConfigError(f"config is not JSON: {path}") from exc
        if not isinstance(loaded, dict):
            raise ConfigError(f"config must be a JSON object: {path}")
        data = loaded
    previous = data.get("provider")
    previous_url = data.get("base_url")
    if previous == "lmstudio" and previous_url and provider != "lmstudio":
        profiles = data.get("profiles")
        if not isinstance(profiles, dict):
            profiles = {}
            data["profiles"] = profiles
        local = profiles.get("local")
        if not isinstance(local, dict):
            local = {"provider": "lmstudio"}
            profiles["local"] = local
        local["base_url"] = previous_url
        local["provider"] = "lmstudio"
    data["provider"] = provider
    data["model"] = model_value
    if provider == "lmstudio":
        kept = previous_url if previous == "lmstudio" and previous_url else None
        if not kept:
            profiles = data.get("profiles")
            local = profiles.get("local") if isinstance(profiles, dict) else None
            if isinstance(local, dict):
                kept = local.get("base_url")
        data["base_url"] = str(kept or PROVIDERS["lmstudio"]["base_url"]).rstrip("/")
    else:
        data["base_url"] = PROVIDERS[provider]["base_url"]
    if pass_value and PROVIDERS[provider]["needs_key"]:
        data["pass"] = pass_value
    else:
        data.pop("pass", None)
    data.pop("auth", None)
    text = json.dumps(data, indent=2) + "\n"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)
