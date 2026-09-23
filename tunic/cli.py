"""Command line. Headless `-p` is the path a small model is tested on."""

from __future__ import annotations

import argparse
import sys
import time

from tunic import __version__
from tunic.agent import compact_messages, run_turn
from tunic.config import PROVIDERS, ConfigError, ensure_home, resolve_settings, save_choice
from tunic.keys import KeyMissing, key_status
from tunic.providers import ProviderError, fetch_lmstudio_catalog
from tunic.session import load_session, save_session
from tunic.ui import ActivityView


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tunic",
        description="Local-first agentic CLI. Cloud keys are optional.",
    )
    parser.add_argument("command", nargs="?", choices=["doctor", "models"])
    parser.add_argument("-p", "--prompt", help="Run one headless turn and exit")
    parser.add_argument("--provider", help="lmstudio, ollama, vllm, openai, xai, anthropic, openrouter, groq, custom")
    parser.add_argument("--model", help="Model id. LM Studio defaults to the already-loaded model")
    parser.add_argument("--base-url", help="API base URL")
    parser.add_argument("--profile", help="Profile name from ~/.tunic/config.json")
    parser.add_argument("--yes", action="store_true", help="Allow bash and write_file without asking")
    parser.add_argument("--plan", action="store_true", help="Read-only plan turn")
    parser.add_argument("--cwd", help="Working directory for file and bash tools")
    parser.add_argument("--session", help="Resume or save a session name")
    parser.add_argument("--allow-load", action="store_true", help="Permit an LM Studio model that is not already loaded")
    parser.add_argument("--no-color", action="store_true")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--version", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.version:
        print(f"tunic {__version__}")
        return 0
    try:
        ensure_home()
        settings = resolve_settings(
            provider=args.provider,
            model=args.model,
            base_url=args.base_url,
            profile=args.profile,
            yes=args.yes,
            plan=args.plan,
            cwd=args.cwd,
            session=args.session,
            allow_load=args.allow_load,
            no_color=args.no_color,
            max_steps=args.max_steps,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
    except ConfigError as exc:
        print(f"tunic: {exc}", file=sys.stderr)
        return 2
    if args.command == "doctor":
        return _doctor(settings)
    if args.command == "models":
        return _models(settings)
    if args.prompt is not None:
        return _headless(settings, args.prompt)
    if not sys.stdin.isatty():
        prompt = sys.stdin.read().strip()
        if not prompt:
            parser.print_help()
            return 2
        return _headless(settings, prompt)
    return _repl(settings)


def _emit(settings, text: str) -> None:
    print(text, flush=True)


def _show(view: ActivityView, text: str) -> None:
    for line in view.feed(text):
        print(line, flush=True)


def _headless(settings, prompt: str) -> int:
    prior = None
    if settings.session:
        try:
            prior = load_session(settings, settings.session) or None
        except ValueError as exc:
            print(f"tunic: {exc}", file=sys.stderr)
            return 2
    try:
        result = run_turn(prompt, settings, emit=lambda line: _emit(settings, line), prior=prior)
    except (KeyMissing, ConfigError) as exc:
        print(f"tunic: {exc}", file=sys.stderr)
        return 2
    except ProviderError as exc:
        print(f"tunic: {exc}", file=sys.stderr)
        return 1
    name = settings.session or f"auto-{int(time.time())}"
    try:
        path = save_session(settings, name, result.messages)
    except ValueError as exc:
        print(f"tunic: {exc}", file=sys.stderr)
        return 2
    print(f"session: {path.name}")
    print(f"tool-calls: {result.tool_calls}")
    return 0


def _doctor(settings) -> int:
    print(f"tunic {__version__}")
    print(f"provider: {settings.provider}")
    print(f"kind: {settings.kind}")
    print(f"base_url: {settings.base_url}")
    print(f"model: {settings.model or '(unset)'}")
    print("stream: false")
    print(f"home: {settings.home}")
    print(key_status(settings))
    if settings.provider == "lmstudio":
        try:
            catalog = fetch_lmstudio_catalog(settings)
        except ProviderError as exc:
            print(f"lmstudio: unreachable ({exc})")
            return 1
        loaded = [item.get("id") for item in catalog if item.get("state") == "loaded"]
        print("loaded: " + (", ".join(str(item) for item in loaded) if loaded else "(none)"))
    print("mutating tools: " + ("allowed (--yes)" if settings.yes else "ask"))
    return 0


def _models(settings) -> int:
    if settings.provider != "lmstudio":
        print("tunic: models list is for the lmstudio provider", file=sys.stderr)
        return 2
    try:
        catalog = fetch_lmstudio_catalog(settings)
    except (ProviderError, ConfigError) as exc:
        print(f"tunic: {exc}", file=sys.stderr)
        return 1
    loaded = [item for item in catalog if item.get("state") == "loaded"]
    rest = [item for item in catalog if item.get("state") != "loaded"]
    for item in loaded + rest:
        state = item.get("state") or "unknown"
        print(f"{state}\t{item.get('id')}")
    return 0


def _repl(settings) -> int:
    print(f"tunic {__version__}", flush=True)
    print(f"project: {settings.cwd}", flush=True)
    print(f"provider: {settings.provider}", flush=True)
    print(f"model: {model_text(settings)}", flush=True)
    print("/settings to change the model    /help    /exit", flush=True)
    messages: list[dict] | None = None
    if settings.session:
        try:
            messages = load_session(settings, settings.session) or None
        except ValueError as exc:
            print(f"tunic: {exc}", file=sys.stderr)
            return 2
    try:
        import readline  # noqa: F401  — enables line editing when the module exists
    except ImportError:
        pass
    while True:
        try:
            line = input("tunic> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not line:
            continue
        if line in ("/exit", "/quit"):
            return 0
        if line == "/help":
            print("/exit  /plan  /compact  /model ID  /session NAME  /settings")
            continue
        if line in ("/settings", "/config"):
            _settings_mode(settings)
            continue
        if line == "/plan":
            settings.plan = not settings.plan
            print(f"plan: {'on' if settings.plan else 'off'}")
            continue
        if line == "/compact":
            if messages:
                messages = compact_messages(messages)
                print(f"messages: {len(messages)}")
            else:
                print("nothing to compact")
            continue
        if line.startswith("/model "):
            _save_model(settings, line.split(None, 1)[1].strip())
            continue
        if line.startswith("/session "):
            settings.session = line.split(None, 1)[1].strip()
            print(f"session: {settings.session}")
            continue
        ask = _interactive_ask if sys.stdin.isatty() else None
        view = ActivityView()
        print("thinking…", flush=True)
        try:
            result = run_turn(
                line,
                settings,
                emit=lambda text: _show(view, text),
                ask=ask,
                prior=messages,
            )
        except (KeyMissing, ConfigError) as exc:
            print(f"tunic: {exc}", file=sys.stderr)
            continue
        except ProviderError as exc:
            print(f"tunic: {exc}", file=sys.stderr)
            continue
        messages = result.messages
        if settings.session:
            try:
                save_session(settings, settings.session, messages)
            except ValueError as exc:
                print(f"tunic: {exc}", file=sys.stderr)


def _interactive_ask(name: str) -> bool:
    try:
        verb = {"bash": "run a command", "write_file": "write a file"}.get(name, name)
        answer = input(f"allow {verb}? [y/N] ")
    except EOFError:
        return False
    return answer.strip().lower() in ("y", "yes")


def format_banner(settings, model_line: str) -> str:
    return "\n".join(
        [
            f"tunic {__version__}",
            f"project: {settings.cwd}",
            f"provider: {settings.provider}",
            f"model: {model_line}",
            "/settings to change the model    /help    /exit",
        ]
    )


def model_text(settings, fetch=None) -> str:
    """Model line for the banner. Does not write the auto-picked id to disk."""
    if settings.provider != "lmstudio":
        return settings.model or "(not set)"
    fetch = fetch or fetch_lmstudio_catalog
    try:
        catalog = fetch(settings, timeout=3)
    except ProviderError:
        if settings.model:
            return f"{settings.model} (LM Studio did not answer)"
        return "unavailable (LM Studio did not answer)"
    loaded = []
    with_tools = None
    for item in catalog if isinstance(catalog, list) else []:
        if not isinstance(item, dict) or item.get("state") != "loaded" or not item.get("id"):
            continue
        loaded.append(item["id"])
        caps = item.get("capabilities") or []
        if with_tools is None and "tool_use" in caps:
            with_tools = item["id"]
    if settings.model:
        if settings.model in loaded:
            return str(settings.model)
        return f"{settings.model} (not loaded; will not load)"
    if not loaded:
        return "unavailable (no model loaded)"
    return str(with_tools or loaded[0])


def _ask_line(prompt: str) -> str | None:
    try:
        return input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return None


def _apply_saved(settings) -> None:
    fresh = resolve_settings(
        home=settings.home,
        cwd=str(settings.cwd),
        yes=settings.yes,
        plan=settings.plan,
        session=settings.session,
        allow_load=settings.allow_load,
        no_color=settings.no_color,
        max_steps=settings.max_steps,
        max_tokens=settings.max_tokens,
        temperature=settings.temperature,
        profile=settings.profile,
    )
    settings.provider = fresh.provider
    settings.model = fresh.model
    settings.base_url = fresh.base_url
    settings.pass_name = fresh.pass_name
    settings.auth = fresh.auth


def _settings_mode(settings) -> None:
    print("settings — saved for the next launch")
    if settings.profile:
        print("note: an active profile overrides this default on launch")
    print("  1  local model (LM Studio)")
    print("  2  API (OpenAI or Anthropic)")
    print("  3  grok / xAI")
    print("  q  back")
    choice = _ask_line("settings> ")
    if choice is None or choice in ("q", "quit", "back", ""):
        return
    if choice == "1":
        _save_local(settings)
    elif choice == "2":
        _save_api(settings)
    elif choice == "3":
        _save_xai(settings)
    else:
        print("tunic: unknown settings choice")


def _save_local(settings) -> None:
    # List LM Studio even if the session is currently on a cloud provider.
    # Do not send this GET to the cloud base URL.
    from dataclasses import replace

    probe = replace(
        settings,
        provider="lmstudio",
        base_url=PROVIDERS["lmstudio"]["base_url"],
        model="",
        auth=None,
        pass_name=None,
    )
    try:
        catalog = fetch_lmstudio_catalog(probe, timeout=15)
    except ProviderError as exc:
        print(f"tunic: {exc}")
        return
    loaded = [
        str(item.get("id"))
        for item in catalog
        if isinstance(item, dict) and item.get("state") == "loaded" and item.get("id")
    ]
    unloaded = [item for item in catalog if not (isinstance(item, dict) and item.get("state") == "loaded")]
    if unloaded:
        print(f"{len(unloaded)} not loaded (not selectable)")
    if not loaded:
        print("No model is loaded in LM Studio. Refusing to load one.")
        return
    for index, model_id in enumerate(loaded, 1):
        print(f"  {index}  {model_id}")
    raw = _ask_line("model> ")
    if raw is None or not raw.isdigit() or not (1 <= int(raw) <= len(loaded)):
        print("tunic: not a loaded model. Nothing saved.")
        return
    _commit_choice(settings, provider="lmstudio", model=loaded[int(raw) - 1], pass_name=None, auth=None)


def _save_api(settings) -> None:
    print("  1  openai")
    print("  2  anthropic")
    raw = _ask_line("api> ")
    if raw == "1":
        provider = "openai"
    elif raw == "2":
        provider = "anthropic"
    else:
        print("tunic: nothing saved.")
        return
    model = _ask_line("model id: ")
    if model is None:
        return
    pass_name = _ask_line("pass entry name (or empty): ")
    if pass_name is None:
        return
    _commit_choice(settings, provider=provider, model=model, pass_name=pass_name or None, auth=None)
    if settings.provider == provider and not settings.model:
        print(f"{provider}: a model id is required before a turn. No request was sent.")


def _save_xai(settings) -> None:
    model = _ask_line("model id: ")
    if model is None:
        return
    pass_name = _ask_line("pass entry name (or empty): ")
    if pass_name is None:
        return
    _commit_choice(settings, provider="xai", model=model, pass_name=pass_name or None, auth=None)
    if settings.provider == "xai" and not settings.model:
        print("xai: a model id is required before a turn. No request was sent.")


def _save_model(settings, model: str) -> None:
    if _commit_choice(
        settings,
        provider=settings.provider,
        model=model,
        pass_name=settings.pass_name,
        auth=settings.auth,
    ):
        print(f"model: {settings.model}")


def _commit_choice(settings, *, provider: str, model: str, pass_name, auth) -> bool:
    try:
        save_choice(settings.home, provider=provider, model=model, pass_name=pass_name, auth=auth)
    except ConfigError as exc:
        print(f"tunic: {exc}")
        return False
    _apply_saved(settings)
    print(format_banner(settings, model_text(settings)))
    return True


def known_providers() -> str:
    return ", ".join(sorted(PROVIDERS))
