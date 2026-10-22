"""Core constraints: schemas, serial tools, paths, missing keys, no secret leaks."""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import unittest
import urllib.request
from contextlib import redirect_stderr, redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import re
from pathlib import Path
from unittest.mock import patch

from tunic.agent import compact_messages, execute_plan_prompt, run_turn, system_prompt
from tunic.cli import main
from tunic.config import LMSTUDIO_BASE_URL, ConfigError, ensure_home, resolve_settings, save_choice, save_yes
from tunic.fetch import FetchError, public_https_url
from tunic.keys import KeyMissing, key_status, resolve_key
from tunic.providers import Completion, ProviderError, ToolCall, build_anthropic_payload, build_openai_payload, complete
from tunic.session import list_sessions, load_session
from tunic.tools import builtin_tools, resolve_user_path, run_tool, undo_last
from tunic.ui import ActivityView


def _strip_color(text: str) -> str:
    return re.sub(r"\033\[[0-9;]*m", "", text)


def _labeled_box(text: str, label: str) -> str:
    marker = f"─ {label} "
    start = text.index(marker)
    top = text.rfind("┌", 0, start)
    bottom = text.index("└", start)
    return text[top:bottom]


def _settings(**kwargs):
    home = kwargs.pop("home", None)
    if home is None:
        raise AssertionError("tests must pass home")
    return resolve_settings(home=home, **kwargs)


class SchemaTests(unittest.TestCase):
    def test_openai_schema_is_the_object_not_a_wrapper(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = _settings(provider="openai", model="m", home=Path(tmp), cwd=tmp)
            payload = build_openai_payload(
                [{"role": "user", "content": "hi"}],
                builtin_tools(),
                settings,
            )
        self.assertIs(payload["stream"], False)
        self.assertEqual(payload["tool_choice"], "auto")
        bash = payload["tools"][0]["function"]
        self.assertEqual(bash["name"], "bash")
        self.assertIn("command", bash["parameters"]["properties"])
        self.assertEqual(bash["parameters"]["type"], "object")
        self.assertNotIn("parameters", bash["parameters"])
        self.assertNotIn("inputSchema", bash)
        self.assertNotIn("input_schema", bash)
        encoded = json.dumps(payload)
        self.assertNotIn("inputSchema", encoded)
        self.assertNotIn("additionalProperties", encoded)

    def test_local_payload_omits_fields_that_crash_lmstudio(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = _settings(provider="lmstudio", model="m", home=Path(tmp), cwd=tmp)
            payload = build_openai_payload(
                [{"role": "user", "content": "hi"}],
                builtin_tools(),
                settings,
            )
        self.assertIs(payload["stream"], False)
        self.assertNotIn("tool_choice", payload)
        self.assertNotIn("temperature", payload)
        schema = payload["tools"][0]["function"]["parameters"]
        self.assertEqual(schema["type"], "object")
        self.assertIn("command", schema["properties"])
        self.assertNotIn("additionalProperties", schema)

    def test_anthropic_uses_input_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = _settings(provider="anthropic", model="m", home=Path(tmp), cwd=tmp)
            payload = build_anthropic_payload(
                [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}],
                builtin_tools(),
                settings,
            )
        self.assertIs(payload["stream"], False)
        self.assertEqual(payload["system"], "sys")
        tool = payload["tools"][0]
        self.assertIn("command", tool["input_schema"]["properties"])
        self.assertNotIn("parameters", tool)
        self.assertNotIn("inputSchema", tool)

    def test_default_local_url_is_loopback(self):
        self.assertEqual(LMSTUDIO_BASE_URL, "http://127.0.0.1:1234/v1")


class PathTests(unittest.TestCase):
    def test_relative_and_tilde_resolve(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            work = Path(tmp) / "work"
            home.mkdir()
            work.mkdir()
            relative = resolve_user_path("notes.txt", work)
            self.assertEqual(relative, (work / "notes.txt").resolve())
            with patch.dict(os.environ, {"HOME": str(home)}):
                tilde = resolve_user_path("~/notes.txt", work)
            self.assertEqual(tilde, (home / "notes.txt").resolve())

    def test_write_and_read_relative(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = _settings(provider="lmstudio", yes=True, home=Path(tmp), cwd=tmp)
            wrote = run_tool("write_file", {"path": "sub/a.txt", "content": "hello"}, settings)
            self.assertIn("wrote", wrote)
            text = run_tool("read_file", {"path": "sub/a.txt"}, settings)
            self.assertEqual(text, "sub/a.txt:1:hello")
            listed = run_tool("list_dir", {"path": "sub"}, settings)
            self.assertIn("a.txt", listed)

    def test_plan_and_ask_do_not_run_bash(self):
        with tempfile.TemporaryDirectory() as tmp:
            marker = Path(tmp) / "ran"
            command = f"touch {marker}"
            plan = _settings(provider="lmstudio", plan=True, yes=True, home=Path(tmp), cwd=tmp)
            denied = run_tool("bash", {"command": command}, plan)
            self.assertIn("permission denied", denied)
            self.assertFalse(marker.exists())
            ask = _settings(provider="lmstudio", yes=False, home=Path(tmp), cwd=tmp)
            denied = run_tool("bash", {"command": command}, ask)
            self.assertIn("permission denied", denied)
            self.assertFalse(marker.exists())


class SerialTests(unittest.TestCase):
    def test_tool_calls_do_not_overlap(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = _settings(
                provider="openai",
                model="m",
                yes=True,
                home=Path(tmp),
                cwd=tmp,
            )
            order = []
            in_flight = {"n": 0}

            def slow(name, arguments, settings, ask=None, reply=None):
                self.assertEqual(in_flight["n"], 0)
                in_flight["n"] += 1
                order.append(arguments["command"])
                time.sleep(0.05)
                in_flight["n"] -= 1
                return "ok"

            raw = [
                {"id": "1", "type": "function", "function": {"name": "bash", "arguments": "{\"command\": \"one\"}"}},
                {"id": "2", "type": "function", "function": {"name": "bash", "arguments": "{\"command\": \"two\"}"}},
            ]
            calls = [
                ToolCall(id="1", name="bash", arguments=raw[0]["function"]["arguments"], parsed={"command": "one"}),
                ToolCall(id="2", name="bash", arguments=raw[1]["function"]["arguments"], parsed={"command": "two"}),
            ]

            def fake_complete(messages, tools, settings, key, on_delta=None):
                if any(m.get("role") == "tool" for m in messages):
                    return Completion(content="done")
                return Completion(content="", tool_calls=calls, raw_tool_calls=raw)

            lines = []
            with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-secret", "TUNIC_HOME": tmp}):
                with patch("tunic.agent.complete", side_effect=fake_complete), patch(
                    "tunic.agent.run_tool", side_effect=slow
                ):
                    result = run_turn("go", settings, emit=lines.append)
            self.assertEqual(order, ["one", "two"])
            self.assertEqual(result.tool_calls, 2)
            self.assertEqual(result.assistant, "done")
            self.assertNotIn("sk-test-secret", "\n".join(lines))


class KeyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.home = self.tmp.name
        self.env = {
            "HOME": os.environ.get("HOME", self.home),
            "TUNIC_HOME": self.home,
            "PATH": os.environ.get("PATH", ""),
        }

    def tearDown(self):
        self.tmp.cleanup()

    def _cleared(self, extra=None):
        env = dict(self.env)
        if extra:
            env.update(extra)
        return patch.dict(os.environ, env, clear=True)

    def test_missing_cloud_keys_do_not_touch_the_network(self):
        cases = [
            ("openai", "OPENAI_API_KEY", "TUNIC_OPENAI_PASS"),
            ("xai", "XAI_API_KEY", "TUNIC_XAI_PASS"),
            ("anthropic", "ANTHROPIC_API_KEY", "TUNIC_ANTHROPIC_PASS"),
            ("synthetic", "SYNTHETIC_API_KEY", "TUNIC_SYNTHETIC_PASS"),
        ]
        for provider, env_name, pass_env in cases:
            with self._cleared():
                with patch("urllib.request.urlopen", side_effect=AssertionError("network")):
                    code = main(["--provider", provider, "-p", "hi"])
            self.assertEqual(code, 2, provider)
            with self._cleared():
                settings = _settings(provider=provider, model="m", home=Path(self.home), cwd=self.home)
                with self.assertRaises(KeyMissing) as caught:
                    resolve_key(settings)
            text = str(caught.exception)
            self.assertIn(env_name, text)
            self.assertIn(pass_env, text)
            self.assertIn("No request was sent", text)
            self.assertNotIn("sk-", text)

    def test_env_key_status_does_not_print_the_secret(self):
        secret = "sk-do-not-print"
        with self._cleared({"OPENAI_API_KEY": secret}):
            settings = _settings(provider="openai", model="m", home=Path(self.home), cwd=self.home)
            status = key_status(settings)
            self.assertIn("OPENAI_API_KEY", status)
            self.assertNotIn(secret, status)
            self.assertEqual(resolve_key(settings), secret)

    def test_pass_name_is_read_and_not_echoed(self):
        secret = "pass-secret-do-not-print"

        def runner(args, check, capture_output, text):
            self.assertEqual(args, ["pass", "show", "openai/api"])

            class Result:
                returncode = 0
                stdout = secret + "\n"
                stderr = "stderr-secret-do-not-print"

            return Result()

        with self._cleared({"TUNIC_OPENAI_PASS": "openai/api"}):
            settings = _settings(provider="openai", model="m", home=Path(self.home), cwd=self.home)
            self.assertEqual(resolve_key(settings, runner=runner), secret)
            status = key_status(settings)
            self.assertIn("openai/api", status)
            self.assertNotIn(secret, status)

    def test_pass_failure_hides_stderr(self):
        def runner(args, check, capture_output, text):
            class Result:
                returncode = 1
                stdout = ""
                stderr = "secret-from-pass-stderr"

            return Result()

        with self._cleared({"TUNIC_XAI_PASS": "missing/entry"}):
            settings = _settings(provider="xai", model="m", home=Path(self.home), cwd=self.home)
            with self.assertRaises(KeyMissing) as caught:
                resolve_key(settings, runner=runner)
        self.assertIn("missing/entry", str(caught.exception))
        self.assertNotIn("secret-from-pass-stderr", str(caught.exception))

    def test_home_env_supplies_synthetic_key(self):
        secret = "syn-do-not-print"
        Path(self.home).mkdir(parents=True, exist_ok=True)
        (Path(self.home) / "env").write_text(f"SYNTHETIC_API_KEY={secret}\n", encoding="utf-8")
        with self._cleared():
            import tunic.keys as keys

            keys._HOME_ENV_LOADED = False
            settings = _settings(provider="synthetic", model="hf:Qwen/Qwen3.8-27B", home=Path(self.home), cwd=self.home)
            self.assertEqual(resolve_key(settings), secret)
            status = key_status(settings)
            self.assertIn("SYNTHETIC_API_KEY", status)
            self.assertNotIn(secret, status)
            keys._HOME_ENV_LOADED = False

    def test_local_needs_no_key(self):
        with self._cleared():
            settings = _settings(provider="lmstudio", home=Path(self.home), cwd=self.home)
            self.assertIsNone(resolve_key(settings))
            self.assertIn("no key required", key_status(settings))

    def test_cloud_provider_does_not_inherit_local_base_url(self):
        cfg = {
            "provider": "lmstudio",
            "base_url": "http://127.0.0.1:9/v1",
            "model": "local-model",
        }
        (Path(self.home) / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
        with self._cleared():
            local = _settings(home=Path(self.home), cwd=self.home)
            cloud = _settings(provider="openai", home=Path(self.home), cwd=self.home)
            xai = _settings(provider="xai", home=Path(self.home), cwd=self.home)
            anthropic = _settings(provider="anthropic", home=Path(self.home), cwd=self.home)
        self.assertEqual(local.provider, "lmstudio")
        self.assertEqual(local.base_url, "http://127.0.0.1:9/v1")
        self.assertEqual(local.model, "local-model")
        self.assertEqual(cloud.base_url, "https://api.openai.com/v1")
        self.assertEqual(cloud.model, "")
        self.assertEqual(xai.base_url, "https://api.x.ai/v1")
        self.assertEqual(anthropic.base_url, "https://api.anthropic.com")
        self.assertNotIn("127.0.0.1:9", cloud.base_url)


class LoopTests(unittest.TestCase):
    def test_fake_server_tool_turn(self):
        payloads = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(length))
                payloads.append(body)
                has_tool = any(m.get("role") == "tool" for m in body["messages"])
                if not has_tool:
                    message = {
                        "role": "assistant",
                        "content": "",
                        "reasoning_content": "",
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {
                                    "name": "bash",
                                    "arguments": json.dumps({"command": "echo tunic-ok"}),
                                },
                            }
                        ],
                    }
                else:
                    message = {"role": "assistant", "content": "tunic-ok", "tool_calls": []}
                raw = json.dumps({"choices": [{"message": message}]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def log_message(self, format, *args):
                return

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        port = server.server_address[1]
        try:
            with tempfile.TemporaryDirectory() as tmp:
                settings = _settings(
                    provider="custom",
                    model="local-test",
                    base_url=f"http://127.0.0.1:{port}/v1",
                    yes=True,
                    temperature=0,
                    home=Path(tmp),
                    cwd=tmp,
                )
                lines = []
                result = run_turn("run echo", settings, emit=lines.append)
        finally:
            server.shutdown()
            server.server_close()
        text = "\n".join(lines)
        self.assertIn("tunic-ok", text)
        self.assertIn("tool-call", text)
        self.assertEqual(result.assistant, "tunic-ok")
        self.assertTrue(payloads)
        self.assertIs(payloads[0]["stream"], False)
        schema = payloads[0]["tools"][0]["function"]["parameters"]
        self.assertIn("command", schema["properties"])
        self.assertEqual(payloads[0]["tools"][0]["function"]["parameters"], schema)
        # The tool result was sent back before the model continued.
        self.assertTrue(any(m.get("role") == "tool" for m in payloads[1]["messages"]))
        self.assertIn("tunic-ok", payloads[1]["messages"][-1]["content"])

    def test_compact_does_not_split_a_tool_group(self):
        messages = [{"role": "system", "content": "s"}]
        for i in range(4):
            messages.append({"role": "user", "content": f"u{i}"})
            messages.append({"role": "assistant", "content": "", "tool_calls": [{"id": str(i)}]})
            messages.append({"role": "tool", "content": f"r{i}", "tool_call_id": str(i)})
        compacted = compact_messages(messages, keep_tail=3)
        self.assertEqual(compacted[0]["role"], "system")
        self.assertEqual(compacted[1]["content"].startswith("[compact]"), True)
        roles = [m["role"] for m in compacted[2:]]
        self.assertNotEqual(roles[0], "tool")

    def test_memory_is_capped_in_the_prompt(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            (home / "memory.md").write_text("remember-this " + ("x" * 5000), encoding="utf-8")
            settings = _settings(provider="lmstudio", home=home, cwd=tmp)
            prompt = system_prompt(settings)
            self.assertIn("remember-this", prompt)
            self.assertLess(len(prompt), 8000)


class CliTests(unittest.TestCase):
    def test_version_and_help_commands(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"TUNIC_HOME": tmp}):
                self.assertEqual(main(["--version"]), 0)

    def test_unknown_provider(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"TUNIC_HOME": tmp}):
                code = main(["--provider", "nope", "-p", "hi"])
        self.assertEqual(code, 2)


class SessionTests(unittest.TestCase):
    def _run(self, lines, home, cwd, fetch=None, extra_env=None, argv=None):
        fed = iter(lines)

        def fake_input(prompt=""):
            if prompt:
                print(prompt, end="")
            try:
                return next(fed)
            except StopIteration as exc:
                raise EOFError from exc

        stdout = io.StringIO()
        stderr = io.StringIO()
        env = {
            "TUNIC_HOME": str(home),
            "PATH": os.environ.get("PATH", ""),
            "HOME": os.environ.get("HOME", str(home)),
        }
        if extra_env:
            env.update(extra_env)
        old = os.getcwd()
        os.chdir(cwd)
        try:
            with patch.dict(os.environ, env, clear=True):
                with patch("sys.stdin.isatty", return_value=True), patch("builtins.input", side_effect=fake_input):
                    with redirect_stdout(stdout), redirect_stderr(stderr):
                        if fetch is None:
                            code = main(list(argv or []))
                        else:
                            with patch("tunic.cli.fetch_lmstudio_catalog", side_effect=fetch):
                                code = main(list(argv or []))
        finally:
            os.chdir(old)
        return code, stdout.getvalue(), stderr.getvalue()

    def test_session_banner_shows_project_provider_model(self):
        timeouts = []

        def fetch(settings, timeout=15):
            timeouts.append(timeout)
            return [{"id": "already-loaded", "state": "loaded", "capabilities": ["tool_use"]}]

        with tempfile.TemporaryDirectory() as tmp:
            with patch("urllib.request.urlopen", side_effect=AssertionError("network")):
                code, out, err = self._run(["/exit"], Path(tmp), Path(tmp), fetch=fetch)
        self.assertEqual(code, 0)
        self.assertIn(f"project: {Path(tmp).resolve()}", out)
        self.assertIn("provider: lmstudio", out)
        self.assertIn("model: already-loaded", out)
        self.assertIn("plan: off", out)
        self.assertIn("writes: ask before a write or a shell", out)
        self.assertEqual(out.count("tunic 0.1.0"), 1)
        self.assertIn("/settings", out)
        self.assertIn("tunic>", out)
        self.assertEqual(timeouts, [3])
        self.assertNotIn("provider=", out)
        self.assertEqual(err, "")

    def test_session_opens_when_catalog_is_down(self):
        def fetch(settings, timeout=15):
            raise ProviderError("down")

        with tempfile.TemporaryDirectory() as tmp:
            code, out, err = self._run(["/exit"], Path(tmp), Path(tmp), fetch=fetch)
        self.assertEqual(code, 0)
        self.assertIn("model: unavailable (LM Studio did not answer)", out)
        self.assertIn("tunic>", out)
        self.assertEqual(err, "")

    def test_settings_saves_xai_without_a_request(self):
        secret = "super-secret-value"

        def fetch(settings, timeout=15):
            return [{"id": "already-loaded", "state": "loaded", "capabilities": ["tool_use"]}]

        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            with patch("urllib.request.urlopen", side_effect=AssertionError("network")):
                code, out, err = self._run(
                    ["/settings", "3", "grok-test", "", "/exit"],
                    home,
                    home,
                    fetch=fetch,
                )
            text = (home / "config.json").read_text(encoding="utf-8")
            self.assertEqual(code, 0, err)
            self.assertIn("grok / xAI", out)
            self.assertNotIn(secret, out)
            self.assertNotIn(secret, text)
            self.assertNotIn("sk-", text)
            self.assertNotIn("auth", text)
            with patch.dict(os.environ, {"TUNIC_HOME": str(home), "HOME": str(home)}, clear=True):
                with redirect_stdout(io.StringIO()) as doctor_out:
                    doctor = main(["doctor"])
            doctor_text = doctor_out.getvalue()
            self.assertEqual(doctor, 0)
            self.assertIn("provider: xai", doctor_text)
            self.assertIn("model: grok-test", doctor_text)
            self.assertIn("key missing", doctor_text)
            self.assertNotIn(secret, doctor_text)

    def test_settings_saves_local_loaded_model_only(self):
        def fetch(settings, timeout=15):
            return [
                {"id": "already-loaded", "state": "loaded"},
                {"id": "sleeping", "state": "not-loaded"},
            ]

        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            code, out, err = self._run(["/settings", "1", "1", "/exit"], home, home, fetch=fetch)
            text = (home / "config.json").read_text(encoding="utf-8")
        self.assertEqual(code, 0, err)
        self.assertIn("not loaded (not selectable)", out)
        self.assertIn("already-loaded", out)
        self.assertIn('"provider": "lmstudio"', text)
        self.assertIn("already-loaded", text)
        self.assertNotIn("sleeping", text)
        self.assertIn("127.0.0.1:1234", text)

    def test_local_menu_does_not_query_the_cloud_url(self):
        seen = {}

        def fetch(settings, timeout=15):
            seen["base_url"] = settings.base_url
            return [{"id": "already-loaded", "state": "loaded"}]

        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            settings = _settings(provider="xai", model="grok-test", home=home, cwd=tmp)
            settings.base_url = "https://api.x.ai/v1"
            from tunic.cli import _save_local

            with patch("tunic.cli.fetch_lmstudio_catalog", side_effect=fetch), patch(
                "builtins.input", return_value="q"
            ):
                _save_local(settings)
        self.assertEqual(seen["base_url"], "http://127.0.0.1:1234/v1")
        self.assertNotIn("api.x.ai", seen["base_url"])

    def test_settings_saves_api_without_a_secret(self):
        secret = "sk-super-secret-value"

        def fetch(settings, timeout=15):
            return [{"id": "already-loaded", "state": "loaded", "capabilities": ["tool_use"]}]

        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            (home / "config.json").write_text(
                json.dumps(
                    {
                        "provider": "lmstudio",
                        "base_url": "http://127.0.0.1:9/v1",
                        "model": "local-model",
                        "profiles": {"openai": {"provider": "openai", "pass": "openai/api"}},
                    }
                ),
                encoding="utf-8",
            )
            code, out, err = self._run(
                ["/config", "2", "1", "gpt-test", "openai/api", "/exit"],
                home,
                home,
                fetch=fetch,
            )
            text = (home / "config.json").read_text(encoding="utf-8")
        self.assertEqual(code, 0, err)
        self.assertIn("provider: openai", out)
        self.assertIn("model: gpt-test", out)
        self.assertIn("openai/api", text)
        self.assertIn('"pass": "openai/api"', json.dumps(json.loads(text)["profiles"]["openai"]))
        self.assertNotIn(secret, text)
        self.assertNotIn("127.0.0.1:9", json.loads(text)["base_url"])
        self.assertEqual(json.loads(text)["base_url"], "https://api.openai.com/v1")

    def test_model_command_keeps_the_current_base_url(self):
        # The screen chooses a connection, then /model changes only the model.
        # Omitting the URL used to refuse custom and retarget ollama/vllm.
        fetch = lambda settings, timeout=15: [{"id": "already-loaded", "state": "loaded"}]
        sequences = (
            ("8", "http://127.0.0.1:9/v1", "local-test", "other-model", "custom"),
            ("4", "http://127.0.0.1:11435/v1", "llama", "other", "ollama"),
            ("5", "http://127.0.0.1:8001/v1", "m", "other", "vllm"),
        )
        for choice, url, first_model, next_model, provider in sequences:
            with self.subTest(provider=provider, path="settings-then-model"):
                with tempfile.TemporaryDirectory() as tmp:
                    home = Path(tmp)
                    code, out, err = self._run(
                        [
                            "/settings",
                            choice,
                            url,
                            first_model,
                            f"/model {next_model}",
                            "/exit",
                        ],
                        home,
                        home,
                        fetch=fetch,
                    )
                    saved = json.loads((home / "config.json").read_text(encoding="utf-8"))
                self.assertEqual(code, 0, err)
                self.assertEqual(saved["provider"], provider)
                self.assertEqual(saved["model"], next_model)
                self.assertEqual(saved["base_url"], url)
                self.assertIn(f"model: {next_model}", out)
                self.assertNotIn("Nothing saved", out)
                self.assertNotIn("needs a base URL", out)
            with self.subTest(provider=provider, path="saved-then-model"):
                with tempfile.TemporaryDirectory() as tmp:
                    home = Path(tmp)
                    save_choice(home, provider=provider, model=first_model, base_url=url)
                    code, out, err = self._run([f"/model {next_model}", "/exit"], home, home)
                    saved = json.loads((home / "config.json").read_text(encoding="utf-8"))
                self.assertEqual(code, 0, err)
                self.assertEqual(saved["model"], next_model)
                self.assertEqual(saved["base_url"], url)
                self.assertNotIn("Nothing saved", out)

    def test_settings_enter_means_the_url_shown_in_brackets(self):
        fetch = lambda settings, timeout=15: [{"id": "already-loaded", "state": "loaded"}]
        cases = (
            ("4", "ollama", "http://127.0.0.1:11434/v1", "http://127.0.0.1:11435/v1"),
            ("5", "vllm", "http://127.0.0.1:8000/v1", "http://127.0.0.1:8001/v1"),
        )
        for choice, provider, shown, typed in cases:
            with self.subTest(provider=provider):
                with tempfile.TemporaryDirectory() as tmp:
                    home = Path(tmp)
                    save_choice(home, provider=provider, model="kept", base_url=typed)
                    code, out, err = self._run(
                        ["/settings", choice, "", "after-enter", "/exit"],
                        home,
                        home,
                        fetch=fetch,
                    )
                    saved = json.loads((home / "config.json").read_text(encoding="utf-8"))
                self.assertEqual(code, 0, err)
                self.assertIn(f"base url [{shown}]:", out)
                self.assertEqual(saved["provider"], provider)
                self.assertEqual(saved["model"], "after-enter")
                self.assertEqual(saved["base_url"], shown)
                self.assertNotIn(typed, saved["base_url"])

    def test_settings_typed_url_replaces_the_saved_one(self):
        fetch = lambda settings, timeout=15: [{"id": "already-loaded", "state": "loaded"}]
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            save_choice(home, provider="ollama", model="llama", base_url="http://127.0.0.1:11435/v1")
            code, out, err = self._run(
                ["/settings", "4", "http://127.0.0.1:11436/v1", "replacement", "/exit"],
                home,
                home,
                fetch=fetch,
            )
            saved = json.loads((home / "config.json").read_text(encoding="utf-8"))
        self.assertEqual(code, 0, err)
        self.assertEqual(saved["model"], "replacement")
        self.assertEqual(saved["base_url"], "http://127.0.0.1:11436/v1")

    def test_model_command_does_not_retarget_a_cloud_url(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            save_choice(home, provider="openai", model="gpt-test", base_url="http://127.0.0.1:9/v1")
            code, out, err = self._run(["/model other-cloud", "/exit"], home, home)
            saved = json.loads((home / "config.json").read_text(encoding="utf-8"))
        self.assertEqual(code, 0, err)
        self.assertEqual(saved["provider"], "openai")
        self.assertEqual(saved["model"], "other-cloud")
        self.assertEqual(saved["base_url"], "https://api.openai.com/v1")
        self.assertNotIn("127.0.0.1:9", saved["base_url"])

    def test_config_alias_and_rejected_token_are_not_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            (home / "config.json").write_text("{}\n", encoding="utf-8")
            with self.assertRaises(Exception) as caught:
                save_choice(home, provider="openai", model="m", pass_name="sk-super-secret-value")
            text = (home / "config.json").read_text(encoding="utf-8")
        self.assertIn("not usable", str(caught.exception))
        self.assertNotIn("sk-super-secret-value", text)
        self.assertNotIn("sk-super-secret-value", str(caught.exception))


class SavedChoiceTests(unittest.TestCase):
    def test_saved_cloud_choice_does_not_touch_the_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            save_choice(home, provider="openai", model="gpt-test")
            env = {"TUNIC_HOME": str(home), "HOME": str(home), "PATH": os.environ.get("PATH", "")}
            with patch.dict(os.environ, env, clear=True):
                with patch("urllib.request.urlopen", side_effect=AssertionError("network")):
                    code = main(["-p", "hi"])
            self.assertEqual(code, 2)
            save_choice(home, provider="anthropic", model="claude-test")
            with patch.dict(os.environ, env, clear=True):
                with patch("urllib.request.urlopen", side_effect=AssertionError("network")):
                    with redirect_stderr(io.StringIO()) as err:
                        code = main(["-p", "hi"])
            self.assertEqual(code, 2)
            self.assertIn("ANTHROPIC_API_KEY", err.getvalue())
            self.assertIn("No request was sent", err.getvalue())
            self.assertNotIn("sk-", err.getvalue())
            save_choice(home, provider="xai", model="grok-test")
            with patch.dict(os.environ, env, clear=True):
                with patch("urllib.request.urlopen", side_effect=AssertionError("network")):
                    with redirect_stderr(io.StringIO()) as err:
                        code = main(["-p", "hi"])
            self.assertEqual(code, 2)
            self.assertIn("XAI_API_KEY", err.getvalue())
            self.assertIn("No request was sent", err.getvalue())
            self.assertNotIn("sk-", err.getvalue())
            self.assertNotIn("auth", (home / "config.json").read_text(encoding="utf-8"))

    def test_interactive_missing_key_stays_open(self):
        def fetch(settings, timeout=15):
            return [{"id": "already-loaded", "state": "loaded"}]

        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            save_choice(home, provider="xai", model="grok-test")
            with patch("urllib.request.urlopen", side_effect=AssertionError("network")):
                code, out, err = SessionTests()._run(
                    ["say hi", "/exit"],
                    home,
                    home,
                    fetch=fetch,
                )
        self.assertEqual(code, 0)
        self.assertIn("XAI_API_KEY", err)
        self.assertIn("No request was sent", err)
        self.assertNotIn("sk-", err)


class ChoiceUnitTests(unittest.TestCase):
    def test_save_round_trip_keeps_profiles_and_drops_local_url(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            (home / "config.json").write_text(
                json.dumps(
                    {
                        "provider": "lmstudio",
                        "base_url": "http://127.0.0.1:9/v1",
                        "model": "local-model",
                        "profiles": {"openai": {"provider": "openai", "pass": "openai/api"}},
                    }
                ),
                encoding="utf-8",
            )
            save_choice(home, provider="xai", model="grok-test", pass_name="xai/api")
            with patch.dict(os.environ, {"TUNIC_HOME": str(home)}, clear=True):
                settings = _settings(home=home, cwd=tmp)
            self.assertEqual(settings.provider, "xai")
            self.assertEqual(settings.model, "grok-test")
            self.assertEqual(settings.base_url, "https://api.x.ai/v1")
            self.assertEqual(settings.pass_name, "xai/api")
            self.assertIsNone(settings.auth)
            saved = json.loads((home / "config.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["profiles"]["openai"]["pass"], "openai/api")
            self.assertNotIn("super-secret-value", json.dumps(saved))
            self.assertNotIn("auth", saved)
            save_choice(home, provider="lmstudio", model="already-loaded")
            saved = json.loads((home / "config.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["provider"], "lmstudio")
            self.assertEqual(saved["base_url"], "http://127.0.0.1:9/v1")
            self.assertNotIn("api.x.ai", saved["base_url"])
            self.assertNotIn("auth", saved)
            self.assertNotIn("pass", saved)

    def test_prompt_names_the_project(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = _settings(provider="lmstudio", home=Path(tmp), cwd=tmp)
            prompt = system_prompt(settings)
        self.assertIn(str(Path(tmp).resolve()), prompt)
        self.assertIn("Relative paths are inside this project", prompt)

    def test_prompt_names_the_model_and_forbids_secret_hunting(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = _settings(
                provider="synthetic",
                model="hf:Qwen/Qwen3.8-27B",
                home=Path(tmp),
                cwd=tmp,
            )
            prompt = system_prompt(settings)
        self.assertIn("provider: synthetic", prompt)
        self.assertIn("model: hf:Qwen/Qwen3.8-27B", prompt)
        self.assertIn("do not probe env", prompt)
        self.assertIn("Stay ethical", prompt)
        self.assertIn("API keys", prompt)
        self.assertIn("list_dir", prompt)
        self.assertIn("edit_file", prompt)
        self.assertIn("/auto on", prompt)
        self.assertIn("You cannot run them", prompt)
        self.assertIn("auto: off", prompt)
        self.assertIn("max tool steps this turn:", prompt)
        self.assertIn("glob", prompt)
        self.assertIn("fetch", prompt)
        self.assertIn("ask", prompt)
        self.assertIn("Do not guess", prompt)
        self.assertIn("git:", prompt)

    def test_prompt_loads_a_matching_skill(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            skills = home / "skills"
            skills.mkdir()
            (skills / "search.md").write_text(
                "---\nname: search\ndescription: Find text in the project files\n---\nUse search, not bash grep.\n",
                encoding="utf-8",
            )
            (skills / "deploy.md").write_text("How to ship a release.\n", encoding="utf-8")
            settings = _settings(provider="lmstudio", home=home, cwd=tmp)
            loaded = system_prompt(settings, "search the repo for known-string")
            skipped = system_prompt(settings, "say hello")
        self.assertIn("Skill search", loaded)
        self.assertIn("Use search, not bash grep.", loaded)
        self.assertIn("deploy", loaded)
        self.assertNotIn("How to ship a release.", loaded)
        self.assertIn("search", skipped)
        self.assertNotIn("Use search, not bash grep.", skipped)

    def test_prompt_reflects_auto_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = _settings(provider="lmstudio", yes=True, home=Path(tmp), cwd=tmp)
            prompt = system_prompt(settings)
        self.assertIn("auto: on", prompt)
        self.assertIn("writes/bash: allowed without asking", prompt)

    def test_later_turn_refreshes_the_session_card(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = _settings(
                provider="custom",
                model="local-test",
                base_url="http://127.0.0.1:9/v1",
                yes=False,
                home=Path(tmp),
                cwd=tmp,
            )
            prior = [
                {"role": "system", "content": "stale system"},
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ]

            def fake_complete(messages, tools, settings, key, on_delta=None):
                self.assertEqual(messages[0]["role"], "system")
                self.assertIn("auto: on", messages[0]["content"])
                self.assertNotIn("stale system", messages[0]["content"])
                return Completion(content="ok")

            settings.yes = True
            with patch("tunic.agent.complete", side_effect=fake_complete):
                result = run_turn("next", settings, emit=lambda _line: None, prior=prior)
        self.assertEqual(result.assistant, "ok")

    def test_max_steps_asks_for_an_answer(self):
        raw = json.dumps({"command": "echo ok"})
        call = ToolCall(id="1", name="bash", arguments=raw, parsed={"command": "echo ok"})
        raw_call = {"id": "1", "type": "function", "function": {"name": "bash", "arguments": raw}}

        with tempfile.TemporaryDirectory() as tmp:
            settings = _settings(
                provider="custom",
                model="local-test",
                base_url="http://127.0.0.1:9/v1",
                yes=True,
                max_steps=2,
                home=Path(tmp),
                cwd=tmp,
            )

            def fake_complete(messages, tools, settings, key, on_delta=None):
                if any("no more tool steps" in str(m.get("content") or "") for m in messages):
                    return Completion(content="here is what I found")
                return Completion(content="", tool_calls=[call], raw_tool_calls=[raw_call])

            lines = []
            with patch("tunic.agent.complete", side_effect=fake_complete):
                result = run_turn("go", settings, emit=lines.append)
        self.assertEqual(result.assistant, "here is what I found")
        self.assertEqual(result.tool_calls, 2)
        self.assertIn("wrap", result.events)
        self.assertTrue(any(line.startswith("wrap:") for line in lines))
        self.assertNotIn("stopped: max steps", "\n".join(lines))
        self.assertTrue(result.truncated)
        self.assertTrue(any(line.startswith("step: 1/2") for line in lines))

    def test_continue_runs_another_round(self):
        raw = json.dumps({"command": "echo ok"})
        call = ToolCall(id="1", name="bash", arguments=raw, parsed={"command": "echo ok"})
        raw_call = {"id": "1", "type": "function", "function": {"name": "bash", "arguments": raw}}

        def fake_complete(messages, tools, settings, key, on_delta=None):
            texts = [str(m.get("content") or "") for m in messages]
            if any("Continue the previous task" in text for text in texts):
                return Completion(content="done after continue")
            if any("no more tool steps" in text for text in texts):
                return Completion(content="partial")
            return Completion(content="", tool_calls=[call], raw_tool_calls=[raw_call])

        def fetch(settings, timeout=15):
            return [{"id": "already-loaded", "state": "loaded"}]

        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            with patch("tunic.agent.complete", side_effect=fake_complete):
                code, out, err = SessionTests()._run(
                    ["do work", "continue", "/exit"],
                    home,
                    home,
                    fetch=fetch,
                    extra_env={"OPENAI_API_KEY": "test-key"},
                    argv=["--yes", "--provider", "custom", "--base-url", "http://127.0.0.1:9/v1", "--model", "m", "--max-steps", "1"],
                )
        self.assertEqual(code, 0, err)
        self.assertIn("partial", out)
        self.assertIn("type continue for another round", out)
        self.assertIn("done after continue", out)

    def test_cloud_default_steps_are_higher(self):
        with tempfile.TemporaryDirectory() as tmp:
            local = resolve_settings(provider="lmstudio", home=Path(tmp), cwd=tmp)
            cloud = resolve_settings(
                provider="synthetic",
                model="hf:Qwen/Qwen3.8-27B",
                home=Path(tmp),
                cwd=tmp,
            )
        self.assertEqual(local.max_steps, 16)
        self.assertEqual(cloud.max_steps, 32)

    def test_secret_files_are_denied_even_with_yes(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            env_path = home / "env"
            env_path.write_text("SYNTHETIC_API_KEY=secret-key\n", encoding="utf-8")
            settings = _settings(provider="lmstudio", yes=True, home=home, cwd=tmp)
            settings.home = home
            denied_read = run_tool("read_file", {"path": str(env_path)}, settings)
            denied_bash = run_tool(
                "bash",
                {"command": f"cat {env_path}"},
                settings,
            )
            denied_proc = run_tool(
                "bash",
                {"command": "tr '\\0' '\\n' < /proc/self/environ"},
                settings,
            )
        self.assertIn("permission denied", denied_read)
        self.assertIn("permission denied", denied_bash)
        self.assertIn("permission denied", denied_proc)
        self.assertNotIn("secret-key", denied_read)
        self.assertNotIn("secret-key", denied_bash)


class ActivityViewTests(unittest.TestCase):
    def test_hides_transport_lines_and_names_the_step(self):
        view = ActivityView()
        shown = []
        for line in (
            "provider: lmstudio",
            "model: local-model",
            "stream: false",
            'tool-call id=1 name=read_file arguments={"path": "README.md"}',
            "tool-result: read_file",
            "hello\nworld",
            "assistant: done",
        ):
            shown.extend(view.feed(line))
        self.assertEqual(shown[0], "→ read README.md")
        self.assertEqual(shown[1], "  hello (+1 lines)")
        self.assertEqual(shown[2:], ["", "done"])
        self.assertEqual(ActivityView().feed("step: 3/16"), ["step 3/16"])

    def test_bash_and_write_stay_one_line(self):
        view = ActivityView()
        self.assertEqual(
            view.feed('tool-call id=2 name=bash arguments={"command": "ls -la"}'),
            ["→ run ls -la"],
        )
        view.feed("tool-result: bash")
        self.assertEqual(view.feed("exit=0\nstdout:\nok"), ["  exit=0 (+2 lines)"])


class HelpTests(unittest.TestCase):
    def test_interactive_help_names_every_slash_command(self):
        from tunic.ui import SLASH_HELP

        def fetch(settings, timeout=15):
            return [{"id": "already-loaded", "state": "loaded", "capabilities": ["tool_use"]}]

        with tempfile.TemporaryDirectory() as tmp:
            code, out, err = SessionTests()._run(["/help", "/exit"], Path(tmp), Path(tmp), fetch=fetch)
        self.assertEqual(code, 0, err)
        for name, meaning in SLASH_HELP:
            self.assertIn(name, out)
            self.assertIn(meaning, out)
        self.assertNotIn("/exit  /plan", out)
        inside = _labeled_box(out, "Commands")
        self.assertIn("Commands", inside.splitlines()[0])
        for name, meaning in SLASH_HELP:
            self.assertIn(name, inside)
            self.assertIn(meaning, inside)

    def test_flag_help_names_flags_and_subcommands(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"TUNIC_HOME": tmp}):
                with redirect_stdout(io.StringIO()) as buf:
                    with self.assertRaises(SystemExit) as caught:
                        main(["--help"])
        self.assertEqual(caught.exception.code, 0)
        text = buf.getvalue()
        for name in (
            "doctor",
            "models",
            "--prompt",
            "--provider",
            "--model",
            "--base-url",
            "--profile",
            "--yes",
            "--ask",
            "--plan",
            "--cwd",
            "--session",
            "--allow-load",
            "--no-color",
            "--max-steps",
            "--max-tokens",
            "--temperature",
            "--version",
        ):
            self.assertIn(name, text)

    def test_settings_offers_every_connection(self):
        def fetch(settings, timeout=15):
            return [{"id": "already-loaded", "state": "loaded"}]

        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            with patch("urllib.request.urlopen", side_effect=AssertionError("network")):
                code, out, err = SessionTests()._run(
                    ["/settings", "6", "router-test", "", "/exit"],
                    home,
                    home,
                    fetch=fetch,
                )
            saved = json.loads((home / "config.json").read_text(encoding="utf-8"))
        self.assertEqual(code, 0, err)
        for label in ("ollama", "vllm", "openrouter", "groq", "custom URL", "grok / xAI", "synthetic"):
            self.assertIn(label, out)
        self.assertEqual(saved["provider"], "openrouter")
        self.assertEqual(saved["model"], "router-test")
        self.assertEqual(saved["base_url"], "https://openrouter.ai/api/v1")
        self.assertNotIn("auth", saved)
        self.assertEqual(out.count("tunic 0.1.0"), 1)
        self.assertIn("saved", out)
        self.assertIn("provider: openrouter", out)

    def test_permission_question_is_plain_language(self):
        from tunic.ui import permission_question

        shell = permission_question("bash", {"command": "ls -la"})
        write = permission_question("write_file", {"path": "notes.txt"})
        edit = permission_question("edit_file", {"path": "notes.txt", "start": "2"})
        self.assertIn("shell command", shell)
        self.assertIn("ls -la", shell)
        self.assertIn("Allow it?", shell)
        self.assertIn("notes.txt", write)
        self.assertIn("changes the project", write)
        self.assertIn("notes.txt", edit)
        self.assertIn("span", edit)
        self.assertIn("changes the project", edit)
        self.assertIn("Allow it?", edit)

    def test_no_color_turns_the_working_line_off(self):
        from tunic.cli import _working_line

        with tempfile.TemporaryDirectory() as tmp:
            plain = _settings(provider="lmstudio", home=Path(tmp), cwd=tmp)
            colored = _settings(provider="lmstudio", home=Path(tmp), cwd=tmp)
        plain.no_color = True
        colored.no_color = False
        self.assertEqual(_working_line(plain), "working…")
        self.assertIn("working…", _working_line(colored))
        self.assertNotEqual(_working_line(colored), "working…")


class ScreenTests(unittest.TestCase):
    def test_plan_mode_shows_writes_off_even_with_yes(self):
        from tunic.cli import format_banner

        def fetch(settings, timeout=15):
            return [{"id": "already-loaded", "state": "loaded"}]

        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            settings = _settings(provider="lmstudio", plan=True, yes=True, home=home, cwd=tmp)
            text = format_banner(settings, "already-loaded")
            code, out, err = SessionTests()._run(
                ["/exit"],
                home,
                home,
                fetch=fetch,
                argv=["--plan", "--yes"],
            )
        self.assertIn("plan: on", text)
        self.assertIn("writes: off (plan mode)", text)
        self.assertNotIn("writes: allowed", text)
        self.assertEqual(code, 0, err)
        self.assertIn("plan: on", out)
        self.assertIn("writes: off (plan mode)", out)
        self.assertEqual(out.count("tunic 0.1.0"), 1)

    def test_color_stays_out_of_the_words_and_no_color_strips_it(self):
        from tunic.cli import format_banner
        from tunic.ui import paint_block

        def fetch(settings, timeout=15):
            return [{"id": "already-loaded", "state": "loaded"}]

        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            settings = _settings(provider="lmstudio", home=home, cwd=tmp)
            banner = format_banner(settings, "already-loaded")
            plain = paint_block(banner, enabled=False)
            colored = paint_block(banner, enabled=True)
            self.assertNotIn("\033[", plain)
            self.assertIn("\033[", colored)
            self.assertIn("provider: lmstudio", colored)
            self.assertIn("writes: ask before a write or a shell", colored)
            with patch("tunic.config._stdout_is_tty", return_value=True):
                colored_code, colored_out, colored_err = SessionTests()._run(
                    ["/exit"], home, home, fetch=fetch
                )
                plain_code, plain_out, plain_err = SessionTests()._run(
                    ["/exit"], home, home, fetch=fetch, argv=["--no-color"]
                )
        self.assertEqual(colored_code, 0, colored_err)
        self.assertEqual(plain_code, 0, plain_err)
        self.assertIn("\033[", colored_out)
        self.assertIn("provider: lmstudio", colored_out)
        self.assertNotIn("\033[", plain_out)
        self.assertIn("plan: off", plain_out)
        self.assertIn("┌", plain_out)
        self.assertIn("└", plain_out)
        self.assertIn("┌", colored_out)
        self.assertEqual(_strip_color(colored_out), plain_out)

    def test_start_fields_sit_in_one_labeled_box(self):
        from tunic.cli import format_banner

        with tempfile.TemporaryDirectory() as tmp:
            settings = _settings(provider="lmstudio", home=Path(tmp), cwd=tmp)
            text = format_banner(settings, "already-loaded")
        self.assertEqual(text.count("tunic 0.1.0"), 1)
        self.assertEqual(text.count("┌"), 1)
        self.assertEqual(text.count("└"), 1)
        inside = _labeled_box(text, "tunic 0.1.0")
        self.assertIn("tunic 0.1.0", inside.splitlines()[0])
        for line in (
            f"project: {settings.cwd}",
            "provider: lmstudio",
            "model: already-loaded",
            "plan: off",
            "writes: ask before a write or a shell",
        ):
            self.assertIn(line, inside)
        self.assertGreater(text.index("/help"), text.index("└"))
        self.assertIn("├─ connection ", text)
        self.assertIn("├─ session ", text)
        self.assertEqual(text.count("┌"), 1)
        self.assertEqual(text.count("└"), 1)
        from tunic.ui import format_slash_help, paint_block

        colored = paint_block(text, enabled=True)
        self.assertNotIn("\033[", text)
        self.assertEqual(_strip_color(colored), text)
        help_text = format_slash_help()
        self.assertEqual(help_text.count("┌"), 1)
        self.assertIn("Commands", _labeled_box(help_text, "Commands").splitlines()[0])
        self.assertEqual(_strip_color(paint_block(help_text, enabled=True)), help_text)
        self.assertNotIn("\033[", help_text)

    def test_turn_box_holds_the_work_and_not_the_answer(self):
        from tunic.agent import TurnResult

        def fetch(settings, timeout=15):
            return [{"id": "already-loaded", "state": "loaded", "capabilities": ["tool_use"]}]

        def fake_turn(prompt, settings, emit, ask=None, prior=None, on_delta=None, reply=None):
            emit('tool-call id=1 name=read_file arguments={"path": "README.md"}')
            emit("tool-result: read_file")
            emit("# Title")
            emit("assistant: the heading is Title")
            return TurnResult(messages=[], assistant="the heading is Title")

        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            with patch("tunic.cli.run_turn", side_effect=fake_turn):
                with patch("tunic.config._stdout_is_tty", return_value=True):
                    code, colored, err = SessionTests()._run(["read it", "/exit"], home, home, fetch=fetch)
                    plain_code, plain, plain_err = SessionTests()._run(
                        ["read it", "/exit"], home, home, fetch=fetch, argv=["--no-color"]
                    )
        self.assertEqual(code, 0, err)
        self.assertEqual(plain_code, 0, plain_err)
        self.assertNotIn("\033[", plain)
        self.assertEqual(_strip_color(colored), plain)
        work = _labeled_box(plain, "turn")
        self.assertIn("working…", work)
        self.assertIn("→ read README.md", work)
        self.assertNotIn("the heading is Title", work)
        self.assertLess(plain.index("└", plain.index("─ turn ")), plain.index("the heading is Title"))
        self.assertIn("┌", plain)
        self.assertIn("└", plain)

    def test_turn_shows_working_then_the_step_then_the_answer(self):
        from tunic.agent import TurnResult

        def fetch(settings, timeout=15):
            return [{"id": "already-loaded", "state": "loaded", "capabilities": ["tool_use"]}]

        def fake_turn(prompt, settings, emit, ask=None, prior=None, on_delta=None, reply=None):
            emit('tool-call id=1 name=read_file arguments={"path": "README.md"}')
            emit("tool-result: read_file")
            emit("# Title")
            emit("assistant: the heading is Title")
            return TurnResult(messages=[], assistant="the heading is Title")

        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            with patch("tunic.cli.run_turn", side_effect=fake_turn):
                code, out, err = SessionTests()._run(["read it", "/exit"], home, home, fetch=fetch)
        self.assertEqual(code, 0, err)
        self.assertLess(out.index("working…"), out.index("→ read README.md"))
        self.assertLess(out.index("→ read README.md"), out.index("the heading is Title"))

    def test_write_asks_in_plain_language_before_it_runs(self):
        raw = json.dumps({"path": "notes.txt", "content": "ok"})
        call = ToolCall(
            id="1",
            name="write_file",
            arguments=raw,
            parsed={"path": "notes.txt", "content": "ok"},
        )
        raw_call = {"id": "1", "type": "function", "function": {"name": "write_file", "arguments": raw}}

        def fake_complete(messages, tools, settings, key, on_delta=None):
            if any(m.get("role") == "tool" for m in messages):
                return Completion(content="did not write")
            return Completion(content="", tool_calls=[call], raw_tool_calls=[raw_call])

        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            save_choice(home, provider="openai", model="gpt-test")
            with patch("urllib.request.urlopen", side_effect=AssertionError("network")):
                with patch("tunic.agent.complete", side_effect=fake_complete):
                    code, out, err = SessionTests()._run(
                        ["write a note", "n", "/exit"],
                        home,
                        home,
                        extra_env={"OPENAI_API_KEY": "test-key"},
                    )
            self.assertEqual(code, 0, err)
            self.assertIn("Write notes.txt?", out)
            self.assertIn("Allow it?", out)
            self.assertLess(out.index("working…"), out.index("→ write notes.txt"))
            self.assertLess(out.index("→ write notes.txt"), out.index("Allow it?"))
            asked = out.index("Allow it?")
            turn = out.index("─ turn ")
            self.assertIn("└", out[turn:asked])
            self.assertIn("permission denied", out)
            self.assertFalse((home / "notes.txt").exists())
            self.assertNotIn("test-key", out)
            self.assertNotIn("test-key", err)

    def test_yes_writes_without_asking(self):
        raw = json.dumps({"path": "notes.txt", "content": "ok"})
        call = ToolCall(
            id="1",
            name="write_file",
            arguments=raw,
            parsed={"path": "notes.txt", "content": "ok"},
        )
        raw_call = {"id": "1", "type": "function", "function": {"name": "write_file", "arguments": raw}}

        def fake_complete(messages, tools, settings, key, on_delta=None):
            if any(m.get("role") == "tool" for m in messages):
                return Completion(content="wrote it")
            return Completion(content="", tool_calls=[call], raw_tool_calls=[raw_call])

        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            save_choice(home, provider="openai", model="gpt-test")
            with patch("tunic.agent.complete", side_effect=fake_complete):
                code, out, err = SessionTests()._run(
                    ["write a note", "/exit"],
                    home,
                    home,
                    extra_env={"OPENAI_API_KEY": "test-key"},
                    argv=["--yes"],
                )
            self.assertEqual(code, 0, err)
            self.assertNotIn("Allow it?", out)
            self.assertIn("working…", out)
            self.assertIn("→ write notes.txt", out)
            self.assertIn("wrote it", out)
            self.assertEqual((home / "notes.txt").read_text(encoding="utf-8"), "ok")
            self.assertNotIn("test-key", out)

    def test_saved_yes_runs_without_asking(self):
        raw = json.dumps({"path": "notes.txt", "content": "ok"})
        call = ToolCall(
            id="1",
            name="write_file",
            arguments=raw,
            parsed={"path": "notes.txt", "content": "ok"},
        )
        raw_call = {"id": "1", "type": "function", "function": {"name": "write_file", "arguments": raw}}

        def fake_complete(messages, tools, settings, key, on_delta=None):
            if any(m.get("role") == "tool" for m in messages):
                return Completion(content="wrote it")
            return Completion(content="", tool_calls=[call], raw_tool_calls=[raw_call])

        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            save_choice(home, provider="openai", model="gpt-test")
            save_yes(home, True)
            with patch("tunic.agent.complete", side_effect=fake_complete):
                code, out, err = SessionTests()._run(
                    ["write a note", "/exit"],
                    home,
                    home,
                    extra_env={"OPENAI_API_KEY": "test-key"},
                )
            self.assertEqual(code, 0, err)
            self.assertNotIn("Allow it?", out)
            self.assertIn("writes: allowed", out)
            self.assertEqual((home / "notes.txt").read_text(encoding="utf-8"), "ok")

    def test_ask_overrides_saved_yes(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            save_yes(home, True)
            settings = resolve_settings(home=home, cwd=tmp, ask=True)
            self.assertFalse(settings.yes)

    def test_slash_yes_saves(self):
        def fetch(settings, timeout=15):
            return [{"id": "already-loaded", "state": "loaded"}]

        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            code, out, err = SessionTests()._run(
                ["/yes", "/exit"],
                home,
                home,
                fetch=fetch,
            )
            saved = json.loads((home / "config.json").read_text(encoding="utf-8"))
        self.assertEqual(code, 0, err)
        self.assertTrue(saved["yes"])
        self.assertIn("writes: allowed", out)
        self.assertIn("saved", out)

    def test_slash_auto_on_and_off(self):
        def fetch(settings, timeout=15):
            return [{"id": "already-loaded", "state": "loaded"}]

        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            code, out, err = SessionTests()._run(
                ["/auto on", "/auto", "/auto off", "/exit"],
                home,
                home,
                fetch=fetch,
            )
            saved = json.loads((home / "config.json").read_text(encoding="utf-8"))
        self.assertEqual(code, 0, err)
        self.assertFalse(saved["yes"])
        self.assertIn("auto: on", out)
        self.assertIn("auto: off", out)
        self.assertIn("usage: /auto on    /auto off", out)
        self.assertLess(out.index("auto: on"), out.index("usage: /auto on    /auto off"))

    def test_slash_steps_saves(self):
        def fetch(settings, timeout=15):
            return [{"id": "already-loaded", "state": "loaded"}]

        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            code, out, err = SessionTests()._run(
                ["/steps 24", "/exit"],
                home,
                home,
                fetch=fetch,
            )
            saved = json.loads((home / "config.json").read_text(encoding="utf-8"))
        self.assertEqual(code, 0, err)
        self.assertEqual(saved["max_steps"], 24)
        self.assertIn("steps: 24", out)
        self.assertIn("saved", out)

    def test_shell_asks_before_it_runs(self):
        raw = json.dumps({"command": "echo tunic-ok"})
        call = ToolCall(id="1", name="bash", arguments=raw, parsed={"command": "echo tunic-ok"})
        raw_call = {"id": "1", "type": "function", "function": {"name": "bash", "arguments": raw}}

        def fake_complete(messages, tools, settings, key, on_delta=None):
            if any(m.get("role") == "tool" for m in messages):
                return Completion(content="did not run")
            return Completion(content="", tool_calls=[call], raw_tool_calls=[raw_call])

        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            save_choice(home, provider="openai", model="gpt-test")
            with patch("tunic.agent.complete", side_effect=fake_complete):
                code, out, err = SessionTests()._run(
                    ["run it", "n", "/exit"],
                    home,
                    home,
                    extra_env={"OPENAI_API_KEY": "test-key"},
                )
        self.assertEqual(code, 0, err)
        self.assertIn("shell command", out)
        self.assertIn("echo tunic-ok", out)
        self.assertIn("Allow it?", out)
        self.assertLess(out.index("→ run echo tunic-ok"), out.index("Allow it?"))
        self.assertIn("permission denied", out)

    def test_custom_url_saves_and_a_secret_url_does_not(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            code, out, err = SessionTests()._run(
                ["/settings", "8", "http://127.0.0.1:9/v1", "local-test", "/exit"],
                home,
                home,
                fetch=lambda settings, timeout=15: [{"id": "already-loaded", "state": "loaded"}],
            )
            saved = json.loads((home / "config.json").read_text(encoding="utf-8"))
            self.assertEqual(code, 0, err)
            self.assertEqual(saved["provider"], "custom")
            self.assertEqual(saved["model"], "local-test")
            self.assertEqual(saved["base_url"], "http://127.0.0.1:9/v1")
            self.assertNotIn("pass", saved)
            self.assertEqual(out.count("tunic 0.1.0"), 1)
            with self.assertRaises(ConfigError) as caught:
                save_choice(
                    home,
                    provider="custom",
                    model="m",
                    base_url="http://user:super-secret-value@127.0.0.1:9/v1",
                )
            text = (home / "config.json").read_text(encoding="utf-8")
            self.assertNotIn("super-secret-value", str(caught.exception))
            self.assertNotIn("super-secret-value", text)
            self.assertEqual(json.loads(text)["base_url"], "http://127.0.0.1:9/v1")
            save_choice(home, provider="openai", model="gpt-test", base_url="http://127.0.0.1:9/v1")
            cloud = json.loads((home / "config.json").read_text(encoding="utf-8"))
            self.assertEqual(cloud["base_url"], "https://api.openai.com/v1")
            self.assertNotIn("127.0.0.1:9", cloud["base_url"])
            save_choice(home, provider="synthetic", model="hf:Qwen/Qwen3.8-27B")
            syn = json.loads((home / "config.json").read_text(encoding="utf-8"))
            self.assertEqual(syn["provider"], "synthetic")
            self.assertEqual(syn["model"], "hf:Qwen/Qwen3.8-27B")
            self.assertEqual(syn["base_url"], "https://api.synthetic.new/v1")

    def test_unknown_slash_is_not_sent_to_the_model(self):
        def fetch(settings, timeout=15):
            return [{"id": "already-loaded", "state": "loaded"}]

        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            with patch("tunic.cli.run_turn", side_effect=AssertionError("sent")):
                code, out, err = SessionTests()._run(["/nope", "/exit"], home, home, fetch=fetch)
        self.assertEqual(code, 0, err)
        self.assertIn("unknown command", out)
        self.assertIn("/help", out)

    def test_headless_emit_still_contains_tool_call(self):
        raw = json.dumps({"command": "echo tunic-ok"})
        call = ToolCall(id="1", name="bash", arguments=raw, parsed={"command": "echo tunic-ok"})
        raw_call = {"id": "1", "type": "function", "function": {"name": "bash", "arguments": raw}}

        def fake_complete(messages, tools, settings, key, on_delta=None):
            if any(m.get("role") == "tool" for m in messages):
                return Completion(content="tunic-ok")
            return Completion(content="", tool_calls=[call], raw_tool_calls=[raw_call])

        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"TUNIC_HOME": tmp, "HOME": tmp, "PATH": os.environ.get("PATH", "")}, clear=True):
                with patch("tunic.agent.complete", side_effect=fake_complete):
                    with redirect_stdout(io.StringIO()) as buf:
                        code = main(
                            [
                                "--provider",
                                "custom",
                                "--base-url",
                                "http://127.0.0.1:9/v1",
                                "--model",
                                "m",
                                "--yes",
                                "-p",
                                "hi",
                            ]
                        )
        text = buf.getvalue()
        self.assertEqual(code, 0)
        self.assertIn("tool-call", text)
        self.assertIn("stream: false", text)
        self.assertNotIn("working…", text)
        self.assertNotIn("\033[", text)
        self.assertNotIn("┌", text)
        self.assertNotIn("└", text)

    def test_settings_saves_synthetic_qwen(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            with patch("urllib.request.urlopen", side_effect=AssertionError("network")):
                code, out, err = SessionTests()._run(
                    ["/settings", "9", "", "", "/exit"],
                    home,
                    home,
                    fetch=lambda settings, timeout=15: [{"id": "already-loaded", "state": "loaded"}],
                )
            saved = json.loads((home / "config.json").read_text(encoding="utf-8"))
        self.assertEqual(code, 0, err)
        self.assertIn("synthetic", out)
        self.assertEqual(saved["provider"], "synthetic")
        self.assertEqual(saved["model"], "hf:Qwen/Qwen3.8-27B")
        self.assertEqual(saved["base_url"], "https://api.synthetic.new/v1")
        self.assertNotIn("auth", saved)

    def test_models_lists_synthetic_catalog(self):
        payload = {
            "data": [
                {"id": "hf:zai-org/GLM-4.7-Flash"},
                {"id": "hf:Qwen/Qwen3.8-27B"},
                {"id": "syn:large:text"},
            ]
        }

        class Response:
            def read(self):
                return json.dumps(payload).encode()

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        def fake_urlopen(request, timeout=15):
            self.assertIn("/models", request.full_url)
            self.assertNotIn("secret-key", request.full_url)
            return Response()

        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(
                os.environ,
                {
                    "TUNIC_HOME": tmp,
                    "HOME": tmp,
                    "PATH": os.environ.get("PATH", ""),
                    "SYNTHETIC_API_KEY": "secret-key",
                },
                clear=True,
            ):
                with patch("urllib.request.urlopen", side_effect=fake_urlopen):
                    with redirect_stdout(io.StringIO()) as buf, redirect_stderr(io.StringIO()) as err:
                        code = main(["--provider", "synthetic", "models"])
        text = buf.getvalue()
        self.assertEqual(code, 0, err.getvalue())
        self.assertIn("qwen3.8", text)
        self.assertIn("hf:Qwen/Qwen3.8-27B", text)
        self.assertIn("syn:large:text", text)
        self.assertLess(text.index("qwen3.8"), text.index("hf:Qwen/Qwen3.8-27B"))
        self.assertLess(text.index("hf:Qwen/Qwen3.8-27B"), text.index("hf:zai-org/GLM-4.7-Flash"))
        self.assertNotIn("secret-key", text)

    def test_model_slash_lists_and_saves(self):
        catalog = {
            "data": [
                {"id": "hf:Qwen/Qwen3.8-27B"},
                {"id": "syn:large:text"},
            ]
        }

        class Response:
            def read(self):
                return json.dumps(catalog).encode()

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            save_choice(home, provider="synthetic", model="hf:Qwen/Qwen3.8-27B")
            with patch("urllib.request.urlopen", return_value=Response()):
                code, out, err = SessionTests()._run(
                    ["/model", "3", "/exit"],
                    home,
                    home,
                    extra_env={"SYNTHETIC_API_KEY": "secret-key"},
                )
            saved = json.loads((home / "config.json").read_text(encoding="utf-8"))
        self.assertEqual(code, 0, err)
        self.assertIn("qwen3.8", out)
        self.assertIn("syn:large:text", out)
        self.assertEqual(saved["provider"], "synthetic")
        self.assertEqual(saved["model"], "syn:large:text")
        self.assertNotIn("secret-key", out)

    def test_model_slash_picks_local_qwen(self):
        catalog = {"data": [{"id": "hf:Qwen/Qwen3.8-27B"}]}

        class Response:
            def read(self):
                return json.dumps(catalog).encode()

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            save_choice(home, provider="synthetic", model="hf:Qwen/Qwen3.8-27B")
            with patch("urllib.request.urlopen", return_value=Response()):
                code, out, err = SessionTests()._run(
                    ["/model", "1", "/exit"],
                    home,
                    home,
                    extra_env={"SYNTHETIC_API_KEY": "secret-key"},
                )
            saved = json.loads((home / "config.json").read_text(encoding="utf-8"))
        self.assertEqual(code, 0, err)
        self.assertIn("qwen3.8", out)
        self.assertEqual(saved["provider"], "lmstudio")
        self.assertEqual(saved["model"], "qwen3.8")
        self.assertEqual(saved["base_url"], "http://127.0.0.1:1234/v1")

    def test_profile_switches_to_synthetic(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            with patch.dict(os.environ, {"TUNIC_HOME": str(home), "HOME": str(home)}):
                ensure_home()
            save_choice(home, provider="lmstudio", model="local-qwen")
            code, out, err = SessionTests()._run(
                ["/profile", "/profile synthetic", "/exit"],
                home,
                home,
                fetch=lambda settings, timeout=15: [{"id": "local-qwen", "state": "loaded"}],
            )
            saved = json.loads((home / "config.json").read_text(encoding="utf-8"))
        self.assertEqual(code, 0, err)
        self.assertIn("synthetic", out)
        self.assertIn("hf:Qwen/Qwen3.8-27B", out)
        self.assertEqual(saved["profiles"]["synthetic"]["provider"], "synthetic")


class SpanSearchTests(unittest.TestCase):
    def test_search_returns_path_and_line_number(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            nested = root / "pkg"
            nested.mkdir()
            (nested / "notes.txt").write_text("first\nknown-string\nthird\n", encoding="utf-8")
            (root / "other.txt").write_text("nothing here\n", encoding="utf-8")
            git = root / ".git"
            git.mkdir()
            (git / "hidden.txt").write_text("known-string\n", encoding="utf-8")
            settings = _settings(provider="lmstudio", plan=True, home=root, cwd=tmp)
            found = run_tool("search", {"query": "known-string", "path": "."}, settings)
        self.assertNotIn("permission denied", found)
        self.assertIn("pkg/notes.txt:2:", found)
        self.assertIn("known-string", found)
        self.assertNotIn("other.txt", found)
        self.assertNotIn(".git", found)

    def test_search_regex_and_glob(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "keep.py").write_text("alpha_one\n", encoding="utf-8")
            (root / "skip.txt").write_text("alpha_one\n", encoding="utf-8")
            settings = _settings(provider="lmstudio", plan=True, home=root, cwd=tmp)
            found = run_tool(
                "search",
                {"query": "alpha_[a-z]+", "path": ".", "regex": "true", "glob": "*.py"},
                settings,
            )
            skipped = run_tool(
                "search",
                {"query": "alpha_[a-z]+", "path": ".", "glob": "*.py"},
                settings,
            )
        self.assertIn("keep.py:1:", found)
        self.assertNotIn("skip.txt", found)
        self.assertEqual(skipped, "no matches")

    def test_edit_changes_one_line_and_leaves_the_rest(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "notes.txt"
            path.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
            settings = _settings(provider="lmstudio", yes=True, home=Path(tmp), cwd=tmp)
            edited = run_tool(
                "edit_file",
                {"path": "notes.txt", "start": "2", "end": "2", "content": "BETA"},
                settings,
            )
            text = path.read_text(encoding="utf-8")
        self.assertNotIn("permission denied", edited)
        self.assertNotIn("tool error", edited)
        self.assertEqual(text, "alpha\nBETA\ngamma\n")
        lines = text.splitlines()
        self.assertEqual(lines[0], "alpha")
        self.assertEqual(lines[2], "gamma")
        self.assertNotEqual(text, "BETA")

    def test_span_write_asks_unless_yes_and_plan_refuses(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "notes.txt"
            original = "alpha\nbeta\ngamma\n"
            path.write_text(original, encoding="utf-8")
            args = {"path": "notes.txt", "start": "2", "end": "2", "content": "BETA"}
            asked = []

            def ask(name, arguments):
                asked.append((name, path.read_text(encoding="utf-8")))
                return False

            denied_settings = _settings(provider="lmstudio", home=Path(tmp), cwd=tmp)
            denied = run_tool("edit_file", args, denied_settings, ask=ask)
            self.assertEqual(asked, [("edit_file", original)])
            self.assertIn("permission denied", denied)
            self.assertEqual(path.read_text(encoding="utf-8"), original)

            def refuse(name, arguments):
                raise AssertionError("plan mode asked")

            plan = _settings(provider="lmstudio", plan=True, yes=True, home=Path(tmp), cwd=tmp)
            refused = run_tool("edit_file", args, plan, ask=refuse)
            self.assertIn("permission denied", refused)
            self.assertEqual(path.read_text(encoding="utf-8"), original)

            def boom(name, arguments):
                raise AssertionError("yes should not ask")

            allowed = _settings(provider="lmstudio", yes=True, home=Path(tmp), cwd=tmp)
            edited = run_tool("edit_file", args, allowed, ask=boom)
            self.assertNotIn("permission denied", edited)
            self.assertEqual(path.read_text(encoding="utf-8"), "alpha\nBETA\ngamma\n")

    def test_request_uses_default_max_tokens_not_1024(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            env = {"TUNIC_HOME": tmp, "HOME": tmp, "PATH": os.environ.get("PATH", "")}
            with patch.dict(os.environ, env, clear=True):
                settings = resolve_settings(provider="lmstudio", model="m", home=home, cwd=tmp)
                payload = build_openai_payload(
                    [{"role": "user", "content": "hi"}],
                    builtin_tools(),
                    settings,
                )
            self.assertGreaterEqual(settings.max_tokens, 4096)
            self.assertNotEqual(settings.max_tokens, 1024)
            self.assertEqual(payload["max_tokens"], settings.max_tokens)
            self.assertGreaterEqual(payload["max_tokens"], 4096)
            self.assertNotEqual(payload["max_tokens"], 1024)
            self.assertIs(payload["stream"], False)
            self.assertEqual(payload["tools"][0]["function"]["name"], "bash")
            names = [tool["function"]["name"] for tool in payload["tools"]]
            for name in ("bash", "read_file", "write_file", "list_dir", "search", "edit_file", "fetch", "ask"):
                self.assertIn(name, names)
            with patch.dict(os.environ, env, clear=True):
                short = resolve_settings(
                    provider="lmstudio",
                    model="m",
                    home=home,
                    cwd=tmp,
                    max_tokens=128,
                )
            self.assertEqual(short.max_tokens, 128)

    def test_original_four_tools_still_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = _settings(provider="lmstudio", yes=True, home=Path(tmp), cwd=tmp)
            wrote = run_tool("write_file", {"path": "sub/a.txt", "content": "hello"}, settings)
            read = run_tool("read_file", {"path": "sub/a.txt"}, settings)
            listed = run_tool("list_dir", {"path": "sub"}, settings)
            ran = run_tool("bash", {"command": "printf tunic-ok"}, settings)
        self.assertIn("wrote", wrote)
        self.assertEqual(read, "sub/a.txt:1:hello")
        self.assertIn("a.txt", listed)
        self.assertIn("tunic-ok", ran)
        self.assertIn("exit=0", ran)


class HarnessTests(unittest.TestCase):
    def test_read_file_is_numbered_and_can_span(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "notes.txt"
            path.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
            settings = _settings(provider="lmstudio", home=Path(tmp), cwd=tmp)
            whole = run_tool("read_file", {"path": "notes.txt"}, settings)
            span = run_tool("read_file", {"path": "notes.txt", "start": "2", "end": "3"}, settings)
        self.assertEqual(whole, "notes.txt:1:alpha\nnotes.txt:2:beta\nnotes.txt:3:gamma")
        self.assertEqual(span, "notes.txt:2:beta\nnotes.txt:3:gamma")

    def test_list_dir_marks_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "pkg").mkdir()
            (root / "readme.txt").write_text("x\n", encoding="utf-8")
            settings = _settings(provider="lmstudio", home=root, cwd=tmp)
            listed = run_tool("list_dir", {"path": "."}, settings)
        self.assertIn("pkg/", listed)
        self.assertIn("readme.txt", listed)
        self.assertNotIn("readme.txt/", listed)

    def test_write_shows_diff_and_undo_restores(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "notes.txt"
            path.write_text("old\n", encoding="utf-8")
            settings = _settings(provider="lmstudio", yes=True, home=Path(tmp), cwd=tmp)
            wrote = run_tool("write_file", {"path": "notes.txt", "content": "new\n"}, settings)
            self.assertIn("-old", wrote.replace(" ", ""))
            self.assertIn("+new", wrote.replace(" ", ""))
            self.assertEqual(path.read_text(encoding="utf-8"), "new\n")
            undone = undo_last(settings)
            self.assertIn("restored", undone)
            self.assertEqual(path.read_text(encoding="utf-8"), "old\n")

    def test_fetch_refuses_private_urls(self):
        with self.assertRaises(FetchError):
            public_https_url("http://example.com")
        with self.assertRaises(FetchError):
            public_https_url("https://127.0.0.1/")
        with self.assertRaises(FetchError):
            public_https_url("https://localhost/secret")
        with self.assertRaises(FetchError):
            public_https_url("https://192.168.1.10/x")
        with self.assertRaises(FetchError):
            public_https_url("https://169.254.169.254/latest")
        with tempfile.TemporaryDirectory() as tmp:
            settings = _settings(provider="lmstudio", plan=True, home=Path(tmp), cwd=tmp)
            denied = run_tool("fetch", {"url": "https://127.0.0.1/"}, settings)
        self.assertIn("tool error", denied)
        self.assertIn("public", denied.lower() + denied)

    def test_fetch_strips_html(self):
        class Resp:
            headers = {"Content-Type": "text/html"}

            def read(self, n=-1):
                data = b"<html><body><h1>Hello</h1><script>secret()</script><p>World</p></body></html>"
                if n is None or n < 0:
                    return data
                return data[:n]

            def geturl(self):
                return "https://example.org/page"

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        class Opener:
            def open(self, req, timeout=15):
                return Resp()

        with tempfile.TemporaryDirectory() as tmp:
            settings = _settings(provider="lmstudio", plan=True, home=Path(tmp), cwd=tmp)
            with patch("tunic.fetch._host_blocked", return_value=False):
                with patch("urllib.request.build_opener", return_value=Opener()):
                    text = run_tool("fetch", {"url": "https://example.org/page"}, settings)
        self.assertIn("Hello", text)
        self.assertIn("World", text)
        self.assertNotIn("secret()", text)

    def test_git_card_names_the_branch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            subprocess = __import__("subprocess")
            env = {key: value for key, value in os.environ.items() if key not in {"GIT_DIR", "GIT_WORK_TREE"}}
            env["GIT_TERMINAL_PROMPT"] = "0"
            created = subprocess.run(
                ["git", "init", "-b", "tunic-test"],
                cwd=root,
                capture_output=True,
                text=True,
                env=env,
            )
            if created.returncode != 0:
                subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True, env=env)
                subprocess.run(
                    ["git", "symbolic-ref", "HEAD", "refs/heads/tunic-test"],
                    cwd=root,
                    check=True,
                    capture_output=True,
                    env=env,
                )
            self.assertTrue((root / ".git").exists(), created.stderr)
            (root / "dirty.txt").write_text("x\n", encoding="utf-8")
            settings = _settings(provider="lmstudio", home=root, cwd=str(root))
            prompt = system_prompt(settings)
        self.assertIn("git: tunic-test", prompt)
        self.assertIn("dirty", prompt)
        self.assertIn("dirty.txt", prompt)

    def test_interrupt_keeps_the_conversation(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = _settings(
                provider="custom",
                model="local-test",
                base_url="http://127.0.0.1:9/v1",
                home=Path(tmp),
                cwd=tmp,
            )

            def boom(*args, **kwargs):
                raise KeyboardInterrupt

            with patch("tunic.agent.complete", side_effect=boom):
                result = run_turn("go", settings, emit=lambda _line: None)
        self.assertTrue(result.truncated)
        self.assertEqual(result.reason, "interrupt")
        self.assertTrue(result.messages)
        self.assertIn("stopped", result.assistant)

    def test_slash_go_executes_the_plan(self):
        seen = []

        def fetch(settings, timeout=15):
            return [{"id": "already-loaded", "state": "loaded"}]

        def fake_turn(prompt, settings, emit, ask=None, prior=None, on_delta=None, reply=None):
            seen.append((prompt, settings.plan))
            emit("assistant: ran the plan")
            from tunic.agent import TurnResult

            return TurnResult(messages=[{"role": "user", "content": "plan it"}, {"role": "assistant", "content": "1. do it"}], assistant="ran the plan")

        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            with patch("tunic.cli.run_turn", side_effect=fake_turn):
                code, out, err = SessionTests()._run(
                    ["first", "/plan", "/go", "/exit"],
                    home,
                    home,
                    fetch=fetch,
                )
        self.assertEqual(code, 0, err)
        self.assertEqual(seen[0][0], "first")
        self.assertIn("Execute the plan", seen[1][0])
        self.assertFalse(seen[1][1])
        self.assertIn(execute_plan_prompt()[:20], seen[1][0])

    def test_sessions_auto_save_last(self):
        def fetch(settings, timeout=15):
            return [{"id": "already-loaded", "state": "loaded"}]

        def fake_turn(prompt, settings, emit, ask=None, prior=None, on_delta=None, reply=None):
            emit("assistant: ok")
            from tunic.agent import TurnResult

            return TurnResult(
                messages=[{"role": "user", "content": prompt}, {"role": "assistant", "content": "ok"}],
                assistant="ok",
            )

        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            with patch("tunic.cli.run_turn", side_effect=fake_turn):
                code, out, err = SessionTests()._run(["hello there", "/sessions", "/exit"], home, home, fetch=fetch)
            names = list_sessions(_settings(provider="lmstudio", home=home, cwd=tmp))
            loaded = load_session(_settings(provider="lmstudio", home=home, cwd=tmp), "last")
        self.assertEqual(code, 0, err)
        self.assertIn("last", names)
        self.assertIn("session: last", out)
        self.assertTrue(any(m.get("content") == "hello there" for m in loaded))

    def test_skill_slash_lists_and_prints(self):
        def fetch(settings, timeout=15):
            return [{"id": "already-loaded", "state": "loaded"}]

        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            skills = home / "skills"
            skills.mkdir()
            (skills / "search.md").write_text("Use search, not bash grep.\n", encoding="utf-8")
            code, out, err = SessionTests()._run(["/skill", "/skill search", "/exit"], home, home, fetch=fetch)
        self.assertEqual(code, 0, err)
        self.assertIn("skill: search", out)
        self.assertIn("Use search, not bash grep.", out)

    def test_retries_one_502_then_succeeds(self):
        hits = {"n": 0}

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", "0"))
                self.rfile.read(length)
                hits["n"] += 1
                if hits["n"] == 1:
                    self.send_response(502)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                raw = json.dumps({"choices": [{"message": {"role": "assistant", "content": "after-retry"}}]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def log_message(self, format, *args):
                return

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        port = server.server_address[1]
        try:
            with tempfile.TemporaryDirectory() as tmp:
                settings = _settings(
                    provider="custom",
                    model="local-test",
                    base_url=f"http://127.0.0.1:{port}/v1",
                    home=Path(tmp),
                    cwd=tmp,
                )
                result = run_turn("hi", settings, emit=lambda _line: None)
        finally:
            server.shutdown()
            server.server_close()
        self.assertEqual(hits["n"], 2)
        self.assertEqual(result.assistant, "after-retry")

    def test_cloud_streams_answer_deltas(self):
        chunks = []
        streamed = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(length))
                streamed.append(body.get("stream"))
                payload = (
                    b'data: {"choices":[{"delta":{"content":"hel"}}]}\n\n'
                    b'data: {"choices":[{"delta":{"content":"lo"}}]}\n\n'
                    b"data: [DONE]\n\n"
                )
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, format, *args):
                return

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        port = server.server_address[1]
        try:
            with tempfile.TemporaryDirectory() as tmp:
                settings = _settings(
                    provider="openai",
                    model="gpt-test",
                    base_url=f"http://127.0.0.1:{port}/v1",
                    home=Path(tmp),
                    cwd=tmp,
                )
                with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
                    completion = complete(
                        [{"role": "user", "content": "hi"}],
                        [],
                        settings,
                        "test-key",
                        on_delta=chunks.append,
                    )
        finally:
            server.shutdown()
            server.server_close()
        self.assertEqual(streamed, [True])
        self.assertEqual(completion.content, "hello")
        self.assertEqual(chunks, ["hel", "lo"])

    def test_activity_view_shows_write_diff(self):
        view = ActivityView()
        view.feed('tool-call id=1 name=write_file arguments={"path": "a.txt"}')
        view.feed("tool-result: write_file")
        shown = view.feed("wrote a.txt (3 chars)\n--- a.txt\n+++ a.txt\n-old\n+new")
        self.assertEqual(shown, ["  -old", "  +new"])

    def test_ask_runs_in_plan_mode_and_headless_has_no_user(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = _settings(provider="lmstudio", plan=True, yes=True, home=Path(tmp), cwd=tmp)

            def refuse(name, arguments):
                raise AssertionError("ask is not a mutating permission prompt")

            answered = run_tool(
                "ask",
                {"question": "which file?", "choices": "a.txt\nb.txt"},
                plan,
                ask=refuse,
                reply=lambda question, choices: choices[1],
            )
            headless = run_tool(
                "ask",
                {"question": "which file?", "choices": "a.txt | b.txt"},
                plan,
            )
        self.assertEqual(answered, "b.txt")
        self.assertIn("no user to ask", headless)
        self.assertIn("a.txt", headless)
        self.assertIn("b.txt", headless)

    def test_ask_pauses_the_turn_for_a_numbered_choice(self):
        def fetch(settings, timeout=15):
            return [{"id": "already-loaded", "state": "loaded"}]

        def fake_complete(messages, tools, settings, key, on_delta=None):
            if any(m.get("role") == "tool" for m in messages):
                return Completion(content="picked " + (messages[-1].get("content") or ""), tool_calls=[])
            return Completion(
                content="",
                tool_calls=[
                    ToolCall(
                        id="ask-1",
                        name="ask",
                        arguments='{"question": "which file?", "choices": "a.txt\\nb.txt"}',
                        parsed={"question": "which file?", "choices": "a.txt\nb.txt"},
                    )
                ],
            )

        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            with patch("tunic.agent.complete", side_effect=fake_complete):
                code, out, err = SessionTests()._run(
                    ["need a file", "2", "/exit"],
                    home,
                    home,
                    fetch=fetch,
                    argv=["--provider", "custom", "--base-url", "http://127.0.0.1:9/v1", "--model", "m"],
                )
        self.assertEqual(code, 0, err)
        self.assertIn("which file?", out)
        self.assertIn("  1  a.txt", out)
        self.assertIn("  2  b.txt", out)
        self.assertIn("picked b.txt", out)
        self.assertIn("ask>", out)


if __name__ == "__main__":
    unittest.main()
