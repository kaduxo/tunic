# Tunic

Local-first agentic CLI. A small model drives the tools. Cloud providers are optional.

## Why it exists

Every agentic CLI assumes a frontier cloud model — one that can juggle parallel tool calls, long streams, and deep tool chains. A 35B-class local model breaks under those assumptions: it over-calls, invents tools, or leaves an open stream hanging on your GPU box. Tunic is built the other way around: one tool at a time, the provider's real schema, no stream, and a hard refusal if the model you named isn't actually loaded. Built for daily use on a home LM Studio server — solid enough that it's the CLI I run my own ops through.

## Install

```bash
python3 -m venv .venv
.venv/bin/pip install -e .
```

Or put `bin/tunic` on `PATH`. `tonic` is the same command.

## Local model

Default provider is LM Studio at `http://127.0.0.1:1234/v1`.

If the server is on another machine, set the URL once:

```bash
tunic --base-url http://127.0.0.1:1234/v1 --provider lmstudio doctor
```

`/settings` then `1` saves the loaded model as the user default in `~/.tunic/config.json`. A saved local URL is kept. Tunic does not load a model. If you name one that is not loaded, it refuses. `--allow-load` overrides that.

`stream` is always false.

## Use

```bash
cd /path/to/project
tonic
```

The first screen puts the project, the connection, the model, and whether plan mode or write permission is on in one labeled box. The version is on that box, once. It is not printed again after a save. `/help` puts the slash commands in a labeled box. A turn puts the working line and each step in a labeled box; the answer comes after that box. A write or a shell command asks before it runs unless you pass `--yes`. `--no-color` turns color off and still prints the boxes.

`search` returns each match as a path and a line number. `edit_file` replaces that span and leaves the other lines. It asks first, the same as a full-file write, unless `--yes`. Plan mode still refuses it.

```bash
tunic -p "read README.md and say the first heading"
tunic --yes -p "create notes.txt with the line ok"
```

`/exit` leaves. `/settings` lists every connection the CLI accepts: LM Studio, Ollama, vLLM, OpenAI, Anthropic, xAI, OpenRouter, Groq, and a custom URL. A write or a shell command asks first unless you pass `--yes` or answer `y`.

## Cloud providers

OpenAI, Anthropic, xAI, OpenRouter, and Groq are optional. Each needs an API key in the environment, or a `pass` entry name. The name is saved. The secret is not.

```bash
export OPENAI_API_KEY=...
tunic --provider openai --model gpt-4.1-mini -p "say ok"
```

Missing key, no request:

```text
tunic: openai: no API key. Set OPENAI_API_KEY, or set TUNIC_OPENAI_PASS to a pass entry name (the name, not the secret). A profile may also set "pass" to that name. No request was sent.
```

Same shape for `xai` / `XAI_API_KEY` / `TUNIC_XAI_PASS`, `anthropic` / `ANTHROPIC_API_KEY` / `TUNIC_ANTHROPIC_PASS`, `openrouter` / `OPENROUTER_API_KEY` / `TUNIC_OPENROUTER_PASS`, and `groq` / `GROQ_API_KEY` / `TUNIC_GROQ_PASS`.

## Four failure modes this build refuses

1. Tools run one at a time. A batch is not parallel.
2. Tool schemas are the provider's real schema, not a wrapper.
3. `stream` is false. A local server must not be left on an open stream.
4. Paths are lenient: relative, `~`, and absolute. Relative paths stay in the launch directory.

## License

MIT — see [LICENSE](LICENSE).
