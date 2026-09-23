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
from pathlib import Path
from unittest.mock import patch

from tunic.agent import compact_messages, run_turn, system_prompt
from tunic.cli import main
from tunic.config import LMSTUDIO_BASE_URL, resolve_settings, save_choice
from tunic.keys import KeyMissing, key_status, resolve_key
from tunic.providers import Completion, ProviderError, ToolCall, build_anthropic_payload, build_openai_payload
from tunic.tools import builtin_tools, resolve_user_path, run_tool


ROOT = Path(__file__).resolve().parents[1]


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
            self.assertEqual(text, "hello")
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

            def slow(name, arguments, settings, ask=None):
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

            def fake_complete(messages, tools, settings, key):
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
    def _run(self, lines, home, cwd, fetch=None, extra_env=None):
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
                            code = main([])
                        else:
                            with patch("tunic.cli.fetch_lmstudio_catalog", side_effect=fetch):
                                code = main([])
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


if __name__ == "__main__":
    unittest.main()
