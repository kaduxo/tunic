"""Command line. Headless `-p` is the path a small model is tested on."""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, replace

from tunic import __version__
from tunic.agent import compact_messages, continue_prompt, execute_plan_prompt, run_turn
from tunic.config import (
    DEFAULT_LOCAL_MODEL,
    DEFAULT_SYNTHETIC_MODEL,
    PROVIDERS,
    ConfigError,
    ensure_home,
    load_config_file,
    resolve_settings,
    save_choice,
    save_max_steps,
    save_yes,
)
from tunic.keys import KeyMissing, key_status, resolve_key
from tunic.providers import (
    ProviderError,
    catalog_ids,
    fetch_lmstudio_catalog,
    fetch_openai_catalog,
)
from tunic.session import list_sessions, load_session, persist_session
from tunic.skills import catalog_names, skill_by_name
from tunic.tools import undo_last
from tunic.ui import (
    ActivityView,
    OpenBox,
    SCREEN_WIDTH,
    boxed,
    format_slash_help,
    paint,
    auto_label,
    paint_block,
    permission_question,
    rule,
    write_label,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tunic",
        description="Local-first agentic CLI. Cloud keys are optional.",
    )
    parser.add_argument("command", nargs="?", choices=["doctor", "models"])
    parser.add_argument("-p", "--prompt", help="Run one headless turn and exit")
    parser.add_argument(
        "--provider",
        help="lmstudio, ollama, vllm, openai, xai, anthropic, openrouter, groq, synthetic, custom",
    )
    parser.add_argument("--model", help="Model id. LM Studio defaults to the already-loaded model")
    parser.add_argument("--base-url", help="API base URL")
    parser.add_argument("--profile", help="Profile name from ~/.tunic/config.json")
    parser.add_argument("--yes", action="store_true", help="Allow bash, write_file, and edit_file without asking")
    parser.add_argument(
        "--ask",
        action="store_true",
        help="Ask before bash, write_file, and edit_file, even if config has yes",
    )
    parser.add_argument("--plan", action="store_true", help="Read-only plan turn")
    parser.add_argument("--cwd", help="Working directory for file and bash tools")
    parser.add_argument("--session", help="Resume or save a session name")
    parser.add_argument("--allow-load", action="store_true", help="Permit an LM Studio model that is not already loaded")
    parser.add_argument("--no-color", action="store_true", help="Disable color")
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
            ask=args.ask,
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


def _show(view: ActivityView, text: str, settings, frame: OpenBox) -> None:
    if text.startswith("delta:"):
        _close_turn(settings, frame)
        sys.stdout.write(text[len("delta:") :])
        sys.stdout.flush()
        frame.streamed = True
        return
    if text.startswith("assistant: "):
        _close_turn(settings, frame)
        body = text[len("assistant: ") :]
        if frame.streamed:
            print(flush=True)
            return
        if body:
            print(flush=True)
            print(body, flush=True)
        return
    enabled = bool(settings is not None and not settings.no_color)
    for line in view.feed(text):
        if line == "":
            continue
        if line.startswith("→"):
            line = paint(line, "36", enabled=enabled)
        elif line.startswith("  "):
            line = paint(line, "2", enabled=enabled)
        _emit_turn_row(settings, frame, line)


def _working_line(settings) -> str:
    return paint("working…", "2", enabled=not settings.no_color)


def _paint_turn_row(settings, row: str) -> str:
    """Color only the borders. The words stay contiguous either way."""
    if settings.no_color or not row.startswith("│"):
        return row
    if row.endswith("│"):
        return paint("│", "2", enabled=True) + row[1:-1] + paint("│", "2", enabled=True)
    return paint("│", "2", enabled=True) + row[1:]


def _close_turn(settings, frame: OpenBox) -> None:
    if frame.is_open:
        print(paint(frame.end(), "2", enabled=not settings.no_color), flush=True)


def _emit_turn_row(settings, frame: OpenBox, text: str) -> None:
    """A row in the turn box. Reopens the box if a question closed it."""
    if not frame.is_open:
        print(paint(frame.start(), "2", enabled=not settings.no_color), flush=True)
    print(_paint_turn_row(settings, frame.row(text)), flush=True)


def _print_screen(settings, text: str) -> None:
    print(paint_block(text, enabled=not settings.no_color), flush=True)


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
    also = None if settings.session else f"auto-{int(time.time())}"
    try:
        path = persist_session(settings, result.messages, also=also)
    except ValueError as exc:
        print(f"tunic: {exc}", file=sys.stderr)
        return 2
    print(f"session: {path.name}")
    print(f"tool-calls: {result.tool_calls}")
    if result.truncated:
        print(f"continue: more steps available (cap {settings.max_steps})")
    return 0


def _doctor(settings) -> int:
    lines = [
        rule("connection"),
        f"provider: {settings.provider}",
        f"kind: {settings.kind}",
        f"base_url: {settings.base_url}",
        f"model: {settings.model or '(unset)'}",
        "stream: false",
        rule("paths"),
        f"home: {settings.home}",
        rule("status"),
        key_status(settings),
        f"auto: {'on' if settings.yes else 'off'}",
        f"max_steps: {settings.max_steps}",
        "mutating tools: " + ("allowed" if settings.yes else "ask"),
    ]
    if settings.provider == "lmstudio":
        try:
            catalog = fetch_lmstudio_catalog(settings)
        except ProviderError as exc:
            lines.append(f"lmstudio: unreachable ({exc})")
            _print_screen(settings, boxed(f"tunic {__version__}", lines))
            return 1
        loaded = [item.get("id") for item in catalog if item.get("state") == "loaded"]
        lines.append("loaded: " + (", ".join(str(item) for item in loaded) if loaded else "(none)"))
    _print_screen(settings, boxed(f"tunic {__version__}", lines))
    return 0


def _models(settings) -> int:
    options, notes = switchable_models(settings)
    for note in notes:
        print(note, file=sys.stderr)
    if not options:
        print("tunic: no models to list", file=sys.stderr)
        return 1
    for option in options:
        print(f"{option.group}\t{option.state}\t{option.model}")
    return 0


@dataclass
class ModelOption:
    group: str
    provider: str
    model: str
    state: str
    base_url: str | None = None


def _local_connection(settings):
    profiles = load_config_file(settings.home).get("profiles") or {}
    local = profiles.get("local") if isinstance(profiles, dict) else {}
    if not isinstance(local, dict):
        local = {}
    url = str(local.get("base_url") or PROVIDERS["lmstudio"]["base_url"]).rstrip("/")
    model = str(local.get("model") or DEFAULT_LOCAL_MODEL).strip() or DEFAULT_LOCAL_MODEL
    return replace(
        settings,
        provider="lmstudio",
        base_url=url,
        model=model,
        auth=None,
        pass_name=None,
    )


def _synthetic_connection(settings):
    return replace(
        settings,
        provider="synthetic",
        base_url=PROVIDERS["synthetic"]["base_url"],
        model=settings.model if settings.provider == "synthetic" else DEFAULT_SYNTHETIC_MODEL,
    )


def switchable_models(settings) -> tuple[list[ModelOption], list[str]]:
    """Local qwen3.8 plus Synthetic models, so /model can switch connections."""
    options: list[ModelOption] = []
    notes: list[str] = []
    seen: set[tuple[str, str]] = set()

    def add(option: ModelOption) -> None:
        key = (option.provider, option.model)
        if key in seen:
            return
        seen.add(key)
        options.append(option)

    local = _local_connection(settings)
    loaded_ids: list[str] = []
    try:
        catalog = fetch_lmstudio_catalog(local, timeout=3)
        loaded_ids = [
            str(item.get("id"))
            for item in catalog
            if isinstance(item, dict) and item.get("state") == "loaded" and item.get("id")
        ]
        for model_id in loaded_ids:
            add(
                ModelOption(
                    group="local",
                    provider="lmstudio",
                    model=model_id,
                    state="loaded",
                    base_url=local.base_url,
                )
            )
    except ProviderError as exc:
        notes.append(f"local: LM Studio did not answer ({exc})")
    if ("lmstudio", local.model) not in seen:
        add(
            ModelOption(
                group="local",
                provider="lmstudio",
                model=local.model,
                state="loaded" if local.model in loaded_ids else "saved",
                base_url=local.base_url,
            )
        )

    synthetic = _synthetic_connection(settings)
    try:
        key = resolve_key(synthetic) if synthetic.needs_key else None
        catalog = fetch_openai_catalog(synthetic, key, timeout=15)
        for model_id in catalog_ids(catalog, (DEFAULT_SYNTHETIC_MODEL,)):
            add(
                ModelOption(
                    group="synthetic",
                    provider="synthetic",
                    model=model_id,
                    state="available",
                    base_url=synthetic.base_url,
                )
            )
    except KeyMissing as exc:
        notes.append(str(exc))
        add(
            ModelOption(
                group="synthetic",
                provider="synthetic",
                model=DEFAULT_SYNTHETIC_MODEL,
                state="saved",
                base_url=synthetic.base_url,
            )
        )
    except ProviderError as exc:
        notes.append(f"synthetic: could not list models ({exc})")
        add(
            ModelOption(
                group="synthetic",
                provider="synthetic",
                model=DEFAULT_SYNTHETIC_MODEL,
                state="saved",
                base_url=synthetic.base_url,
            )
        )
    return options, notes


def listed_models(settings) -> list[tuple[str, str]]:
    options, notes = switchable_models(settings)
    if not options and notes:
        raise ProviderError(notes[0])
    return [(option.state, option.model) for option in options]


def _repl(settings) -> int:
    _print_screen(settings, format_banner(settings, model_text(settings)))
    print(flush=True)
    messages: list[dict] | None = None
    can_continue = False
    if settings.session:
        try:
            messages = load_session(settings, settings.session) or None
        except ValueError as exc:
            print(f"tunic: {exc}", file=sys.stderr)
            return 2
    history_path = _init_readline(settings)
    while True:
        try:
            line = input("tunic> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            _save_readline(history_path)
            return 0
        if not line:
            continue
        if line in ("/exit", "/quit"):
            _save_readline(history_path)
            return 0
        if line == "/help":
            _print_screen(settings, format_slash_help())
            continue
        if line in ("/settings", "/config"):
            _settings_mode(settings)
            continue
        if line == "/yes" or line.startswith("/yes ") or line == "/auto" or line.startswith("/auto "):
            _set_auto(settings, line)
            continue
        if line == "/steps" or line.startswith("/steps "):
            _set_steps(settings, line)
            continue
        if line == "/continue" or (line == "continue" and can_continue):
            if not messages:
                _print_screen(settings, boxed("steps", ["nothing to continue"]))
                continue
            line = continue_prompt(settings)
            # fall through to the turn
        elif line == "continue":
            pass
        if line == "/go":
            settings.plan = False
            if not messages:
                _print_screen(settings, boxed("session", ["plan: off", "nothing to execute"]))
                continue
            line = execute_plan_prompt()
            # fall through to the turn
        if line == "/plan":
            settings.plan = not settings.plan
            _print_screen(
                settings,
                boxed(
                    "session",
                    [
                        f"plan: {'on' if settings.plan else 'off'}",
                        f"auto: {auto_label(settings)}",
                        f"writes: {write_label(settings)}",
                        f"steps: {settings.max_steps}",
                    ],
                ),
            )
            continue
        if line == "/compact":
            if messages:
                messages = compact_messages(messages)
                _print_screen(settings, boxed("session", [f"messages: {len(messages)}"]))
            else:
                _print_screen(settings, boxed("session", ["nothing to compact"]))
            continue
        if line.startswith("/model "):
            _save_model(settings, line.split(None, 1)[1].strip())
            continue
        if line == "/model":
            _pick_model(settings)
            continue
        if line.startswith("/profile "):
            _use_profile(settings, line.split(None, 1)[1].strip())
            continue
        if line == "/profile":
            _list_profiles(settings)
            continue
        if line.startswith("/session "):
            settings.session = line.split(None, 1)[1].strip()
            _print_screen(settings, boxed("session", [f"session: {settings.session}"]))
            continue
        if line in ("/session", "/sessions"):
            _list_sessions(settings)
            continue
        if line == "/undo":
            _undo(settings)
            continue
        if line == "/skill" or line.startswith("/skill ") or line == "/skills":
            _skills(settings, line)
            continue
        if line.startswith("/"):
            _print_screen(settings, boxed("commands", ["unknown command", "/help lists them"]))
            continue
        view = ActivityView()
        frame = OpenBox("turn")

        def pause_for_ask(name, arguments=None):
            # The question is not a step. Close the box so it is not drawn through the frame.
            _close_turn(settings, frame)
            return _interactive_ask(name, arguments, settings)

        ask = pause_for_ask if sys.stdin.isatty() else None
        _emit_turn_row(settings, frame, _working_line(settings))

        def emit_line(text, view=view, frame=frame):
            _show(view, text, settings, frame)

        turn_kw = {}
        if not settings.local:
            turn_kw["on_delta"] = lambda chunk: emit_line("delta:" + chunk)
        try:
            result = run_turn(
                line,
                settings,
                emit=emit_line,
                ask=ask,
                prior=messages,
                **turn_kw,
            )
        except KeyboardInterrupt:
            _close_turn(settings, frame)
            print()
            can_continue = True
            _print_screen(settings, boxed("steps", ["stopped — type continue"]))
            continue
        except (KeyMissing, ConfigError) as exc:
            _close_turn(settings, frame)
            print(f"tunic: {exc}", file=sys.stderr)
            continue
        except ProviderError as exc:
            _close_turn(settings, frame)
            print(f"tunic: {exc}", file=sys.stderr)
            continue
        finally:
            _close_turn(settings, frame)
        messages = result.messages
        can_continue = result.truncated
        if result.truncated:
            if result.reason in {"interrupt", "provider"}:
                note = ["stopped — type continue"]
            else:
                note = [
                    f"budget spent ({settings.max_steps} steps)",
                    "type continue for another round",
                ]
            _print_screen(settings, boxed("steps", note))
        try:
            persist_session(settings, messages)
        except ValueError as exc:
            print(f"tunic: {exc}", file=sys.stderr)


def _interactive_ask(name: str, arguments: dict | None = None, settings=None) -> bool:
    question = permission_question(name, arguments)
    lines = question.splitlines()
    prompt = lines[-1] if lines else "Allow it? [y/N] "
    body = [line for line in lines[:-1] if line]
    try:
        if body and settings is not None:
            _print_screen(settings, boxed("permission", body))
            answer = input(prompt)
        else:
            answer = input(question)
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    return answer.strip().lower() in ("y", "yes")


def format_banner(settings, model_line: str) -> str:
    hint = "/settings  choose a connection    /help  commands    /exit  leave"
    body = boxed(
        f"tunic {__version__}",
        [
            f"project: {settings.cwd}",
            rule("connection"),
            f"provider: {settings.provider}",
            f"model: {model_line}",
            rule("session"),
            f"plan: {'on' if settings.plan else 'off'}",
            f"auto: {auto_label(settings)}",
            f"writes: {write_label(settings)}",
            f"steps: {settings.max_steps}",
        ],
        min_width=max(SCREEN_WIDTH, len(hint)),
    )
    return body + "\n" + hint


def format_status(settings, model_line: str) -> str:
    """Saved confirmation. Not a second copy of the launch screen."""
    return boxed(
        "saved",
        [
            f"provider: {settings.provider}",
            f"model: {model_line}",
            f"plan: {'on' if settings.plan else 'off'}",
            f"auto: {auto_label(settings)}",
            f"writes: {write_label(settings)}",
            f"steps: {settings.max_steps}",
        ],
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


def _announce_key(provider: str) -> None:
    env_name = PROVIDERS[provider].get("env")
    if env_name:
        print(f"{provider}: set {env_name}, or a pass entry name. The secret is not saved.")


def _set_auto(settings, line: str) -> None:
    parts = line.split()
    command = parts[0]
    if len(parts) == 1:
        if command == "/auto":
            _print_screen(
                settings,
                boxed(
                    "auto",
                    [
                        f"auto: {auto_label(settings)}",
                        f"writes: {write_label(settings)}",
                        "usage: /auto on    /auto off",
                    ],
                ),
            )
            return
        settings.yes = not settings.yes
    else:
        word = parts[1].strip().lower()
        if word in ("on", "true", "1"):
            settings.yes = True
        elif word in ("off", "false", "0"):
            settings.yes = False
        else:
            print("usage: /auto on    /auto off")
            return
    try:
        save_yes(settings.home, settings.yes)
    except ConfigError as exc:
        print(f"tunic: {exc}")
        return
    _print_screen(
        settings,
        boxed(
            "auto",
            [
                f"auto: {auto_label(settings)}",
                f"writes: {write_label(settings)}",
                "saved",
            ],
        ),
    )


def _init_readline(settings):
    try:
        import readline
    except ImportError:
        return None
    path = settings.home / "history"
    try:
        readline.read_history_file(str(path))
    except OSError:
        pass
    readline.set_history_length(1000)
    return path


def _save_readline(path) -> None:
    if path is None:
        return
    try:
        import readline
        path.parent.mkdir(parents=True, exist_ok=True)
        readline.write_history_file(str(path))
    except OSError:
        pass


def _list_sessions(settings) -> None:
    names = list_sessions(settings)
    if not names:
        _print_screen(settings, boxed("sessions", ["no saved sessions", "usage: /session NAME"]))
        return
    rows = [f"session: {name}" for name in names[:40]]
    if settings.session:
        rows.append(f"current: {settings.session}")
    rows.append("tunic --session last resumes the most recent")
    _print_screen(settings, boxed("sessions", rows))


def _undo(settings) -> None:
    text = undo_last(settings)
    _print_screen(settings, boxed("undo", [text]))


def _skills(settings, line: str) -> None:
    parts = line.split(None, 1)
    if len(parts) == 1:
        names = catalog_names(settings.home, settings.cwd)
        if not names:
            _print_screen(settings, boxed("skills", ["no skill notes in ~/.tunic/skills"]))
            return
        _print_screen(settings, boxed("skills", [f"skill: {name}" for name in names]))
        return
    found = skill_by_name(settings.home, settings.cwd, parts[1])
    if found is None:
        _print_screen(settings, boxed("skills", [f"unknown skill {parts[1].strip()}"]))
        return
    name, body = found
    rows = [f"skill: {name}"]
    rows.extend((body or "(empty)").splitlines()[:80] or ["(empty)"])
    _print_screen(settings, boxed("skills", rows))


def _set_steps(settings, line: str) -> None:
    from tunic.config import MAX_STEPS_CAP

    parts = line.split()
    if len(parts) == 1:
        _print_screen(
            settings,
            boxed(
                "steps",
                [
                    f"steps: {settings.max_steps}",
                    f"usage: /steps N    (1-{MAX_STEPS_CAP})",
                ],
            ),
        )
        return
    try:
        value = int(parts[1])
    except ValueError:
        print(f"usage: /steps N    (1-{MAX_STEPS_CAP})")
        return
    try:
        save_max_steps(settings.home, value)
    except ConfigError as exc:
        print(f"tunic: {exc}")
        return
    settings.max_steps = value
    _print_screen(
        settings,
        boxed(
            "steps",
            [
                f"steps: {settings.max_steps}",
                "saved",
            ],
        ),
    )


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
    lines = [
        "choose a connection. Saved for the next launch.",
    ]
    if settings.profile:
        lines.append("note: an active profile overrides this default on launch")
    lines.extend(
        [
            rule("local"),
            "  1  LM Studio     local, already loaded",
            "  4  ollama        local",
            "  5  vllm          local",
            "  8  custom URL",
            rule("cloud"),
            "  2  API           OpenAI or Anthropic",
            "  3  grok / xAI",
            "  6  openrouter",
            "  7  groq",
            "  9  synthetic     Qwen3.8 and other API models",
            rule(),
            "  q  back",
        ]
    )
    _print_screen(settings, boxed("settings", lines))
    choice = _ask_line("settings> ")
    if choice is None or choice in ("q", "quit", "back", ""):
        return
    if choice == "1":
        _save_local(settings)
    elif choice == "2":
        _save_api(settings)
    elif choice == "3":
        _save_xai(settings)
    elif choice == "4":
        _save_connection(settings, "ollama")
    elif choice == "5":
        _save_connection(settings, "vllm")
    elif choice == "6":
        _save_connection(settings, "openrouter")
    elif choice == "7":
        _save_connection(settings, "groq")
    elif choice == "8":
        _save_connection(settings, "custom")
    elif choice == "9":
        _save_synthetic(settings)
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
    rows = []
    if unloaded:
        rows.append(f"{len(unloaded)} not loaded (not selectable)")
    if not loaded:
        rows.append("No model is loaded in LM Studio. Refusing to load one.")
        _print_screen(settings, boxed("LM Studio", rows))
        return
    rows.append(rule("loaded"))
    for index, model_id in enumerate(loaded, 1):
        rows.append(f"  {index}  {model_id}")
    _print_screen(settings, boxed("LM Studio", rows))
    raw = _ask_line("model> ")
    if raw is None or not raw.isdigit() or not (1 <= int(raw) <= len(loaded)):
        print("tunic: not a loaded model. Nothing saved.")
        return
    _commit_choice(settings, provider="lmstudio", model=loaded[int(raw) - 1], pass_name=None, auth=None)


def _save_api(settings) -> None:
    _print_screen(
        settings,
        boxed(
            "API",
            [
                "  1  openai",
                "  2  anthropic",
            ],
        ),
    )
    raw = _ask_line("api> ")
    if raw == "1":
        provider = "openai"
    elif raw == "2":
        provider = "anthropic"
    else:
        print("tunic: nothing saved.")
        return
    _announce_key(provider)
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
    _announce_key("xai")
    model = _ask_line("model id: ")
    if model is None:
        return
    pass_name = _ask_line("pass entry name (or empty): ")
    if pass_name is None:
        return
    _commit_choice(settings, provider="xai", model=model, pass_name=pass_name or None, auth=None)
    if settings.provider == "xai" and not settings.model:
        print("xai: a model id is required before a turn. No request was sent.")


def _save_connection(settings, provider: str) -> None:
    spec = PROVIDERS[provider]
    base_url = None
    if provider in ("ollama", "vllm", "custom"):
        shown = spec["base_url"] or "required"
        raw = _ask_line(f"base url [{shown}]: ")
        if raw is None:
            return
        if raw:
            base_url = raw
        elif provider == "custom":
            print("tunic: custom needs a base url. Nothing saved.")
            return
    model = _ask_line("model id: ")
    if model is None:
        return
    pass_name = None
    if spec["needs_key"]:
        _announce_key(provider)
        pass_name = _ask_line("pass entry name (or empty): ")
        if pass_name is None:
            return
    if not _commit_choice(
        settings,
        provider=provider,
        model=model,
        pass_name=pass_name or None,
        auth=None,
        base_url=base_url,
    ):
        return
    if spec["needs_key"] and settings.provider == provider and not settings.model:
        print(f"{provider}: a model id is required before a turn. No request was sent.")


def _save_synthetic(settings) -> None:
    from dataclasses import replace

    _announce_key("synthetic")
    probe = replace(
        settings,
        provider="synthetic",
        base_url=PROVIDERS["synthetic"]["base_url"],
        model=settings.model or DEFAULT_SYNTHETIC_MODEL,
    )
    ids: list[str] = []
    try:
        key = resolve_key(probe)
    except KeyMissing:
        key = None
    if key:
        try:
            catalog = fetch_openai_catalog(probe, key, timeout=15)
            ids = catalog_ids(catalog, (DEFAULT_SYNTHETIC_MODEL,))
        except ProviderError as exc:
            print(f"tunic: could not list synthetic models ({exc})")
    rows: list[str] = []
    if ids:
        rows.append(rule("synthetic"))
        for index, model_id in enumerate(ids, 1):
            mark = "  (default)" if model_id == DEFAULT_SYNTHETIC_MODEL else ""
            rows.append(f"  {index}  {model_id}{mark}")
        rows.append("type a number, or a model id (hf:... or syn:...)")
    else:
        rows.append(f"default: {DEFAULT_SYNTHETIC_MODEL}")
        rows.append("type a model id (hf:... or syn:...), or enter for Qwen3.8")
    _print_screen(settings, boxed("synthetic", rows))
    raw = _ask_line("model> ")
    if raw is None:
        return
    if raw == "":
        model = DEFAULT_SYNTHETIC_MODEL
    elif raw.isdigit() and ids and 1 <= int(raw) <= len(ids):
        model = ids[int(raw) - 1]
    else:
        model = raw
    pass_name = _ask_line("pass entry name (or empty): ")
    if pass_name is None:
        return
    _commit_choice(settings, provider="synthetic", model=model, pass_name=pass_name or None, auth=None)
    if settings.provider == "synthetic" and not settings.model:
        print("synthetic: a model id is required before a turn. No request was sent.")


def _pick_model(settings) -> None:
    """List local and Synthetic models, then save the one that is chosen."""
    options, notes = switchable_models(settings)
    rows = list(notes)
    if not options:
        rows.append("no models to list. usage: /model ID")
        _print_screen(settings, boxed("models", rows or ["no models to list"]))
        return
    last_group = None
    for index, option in enumerate(options, 1):
        if option.group != last_group:
            rows.append(rule(option.group))
            last_group = option.group
        current = ""
        if option.provider == settings.provider and option.model == settings.model:
            current = "  (current)"
        rows.append(f"  {index}  {option.model}  {option.state}{current}")
    _print_screen(settings, boxed("models", rows))
    raw = _ask_line("model> ")
    if raw is None or raw == "":
        return
    if raw.isdigit() and 1 <= int(raw) <= len(options):
        _apply_option(settings, options[int(raw) - 1])
        return
    matched = _match_option(options, raw)
    if matched is not None:
        _apply_option(settings, matched)
        return
    _save_model(settings, raw)


def _match_option(options: list[ModelOption], raw: str) -> ModelOption | None:
    text = raw.strip()
    hits = [option for option in options if option.model == text]
    if len(hits) == 1:
        return hits[0]
    lowered = text.lower()
    hits = [option for option in options if option.model.lower() == lowered]
    if len(hits) == 1:
        return hits[0]
    return None


def _apply_option(settings, option: ModelOption) -> None:
    settings.profile = None
    pass_name = settings.pass_name if option.provider == "synthetic" else None
    _commit_choice(
        settings,
        provider=option.provider,
        model=option.model,
        pass_name=pass_name,
        auth=None,
        base_url=option.base_url,
    )


def _list_profiles(settings) -> None:
    from tunic.config import load_config_file

    profiles = load_config_file(settings.home).get("profiles") or {}
    if not isinstance(profiles, dict) or not profiles:
        _print_screen(settings, boxed("profiles", ["no profiles in ~/.tunic/config.json"]))
        return
    active = settings.profile or "(default)"
    rows = [f"active: {active}", rule("saved")]
    for name in sorted(profiles):
        spec = profiles[name] if isinstance(profiles[name], dict) else {}
        provider = spec.get("provider") or "?"
        model = spec.get("model") or ""
        suffix = f"  {model}" if model else ""
        mark = "  (active)" if name == settings.profile else ""
        rows.append(f"  {name}  {provider}{suffix}{mark}")
    rows.append("usage: /profile NAME")
    _print_screen(settings, boxed("profiles", rows))


def _use_profile(settings, name: str) -> None:
    from tunic.config import load_config_file

    profiles = load_config_file(settings.home).get("profiles") or {}
    if not isinstance(profiles, dict) or name not in profiles:
        print(f"tunic: unknown profile {name!r}")
        return
    settings.profile = name
    _apply_saved(settings)
    _print_screen(settings, format_status(settings, model_text(settings)))


def _save_model(settings, model: str) -> None:
    options, _notes = switchable_models(settings)
    matched = _match_option(options, model)
    if matched is not None:
        _apply_option(settings, matched)
        return
    # Model-only. Keep the URL already in use. Omitting it makes custom
    # refuse the save, and replaces a non-default ollama or vllm URL.
    base_url = settings.base_url if settings.provider in {"ollama", "vllm", "custom"} else None
    _commit_choice(
        settings,
        provider=settings.provider,
        model=model,
        pass_name=settings.pass_name,
        auth=settings.auth,
        base_url=base_url,
    )


def _commit_choice(settings, *, provider: str, model: str, pass_name, auth, base_url=None) -> bool:
    try:
        save_choice(
            settings.home,
            provider=provider,
            model=model,
            pass_name=pass_name,
            auth=auth,
            base_url=base_url,
            profile=settings.profile,
        )
    except ConfigError as exc:
        print(f"tunic: {exc}")
        return False
    _apply_saved(settings)
    _print_screen(settings, format_status(settings, model_text(settings)))
    return True


def known_providers() -> str:
    return ", ".join(sorted(PROVIDERS))
