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

Local calls never stream. Cloud providers stream the final answer.

## Use

```bash
cd /path/to/project
tonic
```

The first screen puts the project, the connection, the model, and whether plan mode or write permission is on in one labeled box, with section rules between those groups. `/settings`, `/model`, `/profile`, `/help`, and `tunic doctor` use the same boxed layout. The version is on that box, once. It is not printed again after a save. `/help` puts the slash commands in a labeled box. A turn puts the working line and each step in a labeled box; the answer comes after that box. A write or a shell command asks before it runs unless auto mode is on (`/auto on`, `--yes`, or `"yes": true` in `~/.tunic/config.json`). `/auto off` or `--ask` turns asking back on. `--no-color` turns color off and still prints the boxes.

A turn may make 16 tool calls on a local provider, or 32 on Synthetic and other cloud providers. `/steps N` saves a cap. If the budget is spent, Tunic asks for a wrap-up answer; type `continue` for another round. Ctrl+C stops the turn and keeps the session. The turn box shows `step 3/16`. Matching notes in `~/.tunic/skills` are loaded into that turn; `/skill` lists them.

The session card includes the git branch and a short dirty summary. Interactive turns auto-save as `last` (`tunic --session last`). `/sessions` lists saved transcripts. `/go` leaves plan mode and executes the plan. Prompt history is stored in `~/.tunic/history`. One HTTP 502/503/504 is retried once.

```bash
tunic -p "read README.md and say the first heading"
tunic --yes -p "create notes.txt with the line ok"
```

## Tools

Tools run one at a time. Plan mode keeps the read-only tools and refuses the rest.

| Tool | What it does |
| --- | --- |
| `list_dir` | List a directory. Directories end with `/`. Path `.` is the working directory. |
| `read_file` | Read a text file as `path:line:text`. Optional `start` and `end` (1-based) read a span. |
| `search` | Find text. Returns `path:line:` for `edit_file`. Optional `regex` and `glob` (for example `*.py`). |
| `edit_file` | Replace one line span from `search`. Other lines stay. Prints a short diff. `/undo` restores it. |
| `write_file` | Create or replace a whole text file. Prints a short diff. `/undo` restores it. |
| `fetch` | GET a public https page as text. HTML is stripped. Localhost, private, and credentialed URLs are refused. |
| `ask` | Ask the user a question mid-turn when a path, name, constraint, or choice is missing. Optional `choices` (one per line or `\|`). Works in plan mode. Do not guess. |
| `bash` | Run a shell command in the project directory. Prefer the file tools for reads and edits. |

`write_file`, `edit_file`, and `bash` ask before they run unless auto mode is on. Plan mode refuses them. `ask` still runs in plan mode. Secret paths (`~/.tunic/env`, `/proc/*/environ`) stay blocked even with auto on. Headless `-p` has no person to answer, so `ask` returns that and the model continues with what it has.

## Commands

| Command | What it does |
| --- | --- |
| `/help` | List these commands. |
| `/exit`, `/quit` | Leave the session. |
| `/settings`, `/config` | Choose a connection and model. LM Studio, Ollama, vLLM, OpenAI, Anthropic, xAI, OpenRouter, Groq, Synthetic, or a custom URL. |
| `/model` | List local qwen3.8 (LM Studio) and Synthetic models, then pick one. |
| `/model ID` | Save a different model for this connection. |
| `/profile` | List saved profiles. |
| `/profile NAME` | Switch a saved profile (`local`, `synthetic`, …). |
| `/auto`, `/yes` | Show whether writes and shell commands run without asking. |
| `/auto on` | Run writes and shell commands without asking. Saved as the default. |
| `/auto off` | Ask before writes and shell commands. |
| `/plan` | Turn plan mode on or off (read-only). |
| `/go` | Leave plan mode and execute the plan just written. |
| `/steps` | Show the tool-step cap. |
| `/steps N` | Set how many tool calls a turn may make (1–64). |
| `/continue` | Run another round of tool steps after the cap. |
| `/compact` | Shorten the remembered turns. |
| `/session NAME` | Name this session so it can be resumed. |
| `/sessions` | List saved sessions. `tunic --session last` resumes the most recent. |
| `/undo` | Restore the last write or edit. |
| `/skill` | List skill notes in `~/.tunic/skills`. |
| `/skill NAME` | Print a skill note. |

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

Same shape for `xai` / `XAI_API_KEY` / `TUNIC_XAI_PASS`, `anthropic` / `ANTHROPIC_API_KEY` / `TUNIC_ANTHROPIC_PASS`, `openrouter` / `OPENROUTER_API_KEY` / `TUNIC_OPENROUTER_PASS`, `groq` / `GROQ_API_KEY` / `TUNIC_GROQ_PASS`, and `synthetic` / `SYNTHETIC_API_KEY` / `TUNIC_SYNTHETIC_PASS`.

Synthetic is OpenAI-compatible at `https://api.synthetic.new/v1`. Default model is `hf:Qwen/Qwen3.8-27B`. Switch with `/model`, `tunic models`, or `/settings` then `9`. Aliases such as `syn:large:text` work too.

```bash
export SYNTHETIC_API_KEY=...
# or put SYNTHETIC_API_KEY in ~/.tunic/env (mode 600)
tunic --provider synthetic --model hf:Qwen/Qwen3.8-27B -p "say ok"
tunic --profile synthetic
```

## Four failure modes this build refuses

1. Tools run one at a time. A batch is not parallel.
2. Tool schemas are the provider's real schema, not a wrapper.
3. Local calls never stream. A local server must not be left on an open stream. Cloud providers stream the final answer.
4. Paths are lenient: relative, `~`, and absolute. Relative paths stay in the launch directory.

## License

MIT — see [LICENSE](LICENSE).
