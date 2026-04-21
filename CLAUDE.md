# ochat (formerly ollama-chat)

Simple TUI chat client for Ollama with custom system prompt support. Named "ollama-chat" for the original Ollama-only scope, but now supports multiple backends (Ollama, OpenAI-compatible, llama.cpp). Package and binary are `ochat`.

## Project structure

```
ochat/
├── ochat.py                    # Entry point
├── ochat.sh                    # Bash launcher (symlinked to ~/bin/ochat)
├── ochat/                      # Package
│   ├── __init__.py             # NullHandler logging
│   ├── app.py                  # OChat class (backend abstraction), main()
│   ├── commands.py             # CommandsMixin (_handle_*)
│   ├── generation.py           # GenerationMixin (streaming, _generate_response, _chat_call delegates to backend)
│   ├── widgets.py              # Message, ChatContainer, CommandSuggester
│   ├── config.py               # Configuration management, personalities, backend field
│   ├── chat.tcss               # Textual CSS styles
│   └── backend/                # Backend abstraction layer
│       ├── __init__.py         # BackendProtocol, create_backend factory, AutoBackend
│       ├── base.py             # BackendProtocol interface
│       ├── ollama.py           # OllamaBackend (ollama-python client)
│       ├── openai.py           # OpenAIBackend (openai client)
│       └── llama_cpp.py        # LlamaCppBackend (llama.cpp /v1 API)
├── system_instructions.json    # LLM instructions for commands (compact, impersonate, impersonate_short)
├── personalities/              # Bundled personality templates (copied on first run)
│   ├── default.md
│   ├── creative.md
│   └── storyteller.md
├── requirements.txt            # Dependencies: textual, ollama, openai
└── .venv/                      # Python 3.11 virtual environment
```

## Configuration

Config stored in `~/.config/ochat/`:
- `config.conf` - INI format settings (host, verify_ssl, model, num_ctx, personality, streaming, append_local_prompt, auto_suggest, config_name)
- `*.conf` - Named config profiles (backups created via setup wizard or `--as-default`)
- `personalities/` - System prompt templates (`.md` files)

## Backend modes

The app supports four backend modes (configured in `config.conf` `[defaults] backend`):

1. **Ollama** (default): Uses the Ollama Python client (`ollama` package). Supports model listing, `num_ctx`, `model_options`, and real context token tracking.

2. **OpenAI**: Uses the `openai` Python client with custom `base_url`. Works with LM Studio, llama.cpp, vLLM, etc. Limitations:
   - No interactive model selection (model listing is used for connection testing only — must configure model name manually)
   - `num_ctx` and `model_options` are ignored
   - Context usage tracking is disabled (percentages and warnings don't apply)
   - Setup wizard won't work (edit config.conf manually)

3. **Llama.cpp**: Uses llama.cpp server's `/v1/chat/completions` and `/info` endpoints. Supports `num_ctx` via `/info`, `include_usage` for real token tracking, and `stream_options` for streaming.

4. **Auto** (new default option): Tries Ollama → llama.cpp → OpenAI in sequence. Uses `BackendProtocol` abstraction — all API calls go through `self.backend`.

Backend is created via `_create_backend()` in `app.py`. The `backend_type` field in config determines which backend is instantiated. All generation commands use `self.backend` via `BackendProtocol` interface (`chat()`, `list_models()`, `get_info()`).

Detection happens at startup in `_show_greeting()` based on `self.backend_type`. The greeting shows the backend type: `Connected (Ollama)`, `Connected (OpenAI)`, `Connected (llama.cpp)`, or `Connected (auto)`.

## Usage

```bash
ochat                              # Use saved config
ochat -m <model>                   # Override model
ochat -c <num_ctx>                 # Override context size
ochat -s <file>                    # Use specific system prompt file
ochat -C / --config                # Interactive setup wizard (modify current)
ochat --new                        # Create a new named config profile
ochat --use-config <name>          # Use named config for this session
ochat --use-config <name> --as-default  # Switch to named config as new default
ochat -d                               # Enable debug logging
```

## Multiple configs

Named config profiles allow switching between different setups (models, personalities, etc.):

1. **Create new config**: `--new` creates a fresh named profile from scratch
2. **Or backup existing**: Run `-C`, then backup when prompted
3. **Use temporarily**: `--use-config mistral-creative` loads for one session only
4. **Switch default**: `--use-config mistral-creative --as-default` backs up current config and switches
5. **Switch in-app**: `/config` lists profiles, `/config <n>` switches and restarts

When switching with `--as-default`:
- Current `config.conf` is backed up to `{config_name}.conf` (prompts for name if unset from CLI, defaults to "config-default" from TUI)
- Asks confirmation before overwriting existing backup (CLI only)
- The chosen config becomes the new `config.conf`

Config name is shown in greeting: `config: mistral-creative · Connected`

## Keyboard shortcuts

- **Ctrl+C** - Clear input
- **Ctrl+D** - Quit
- **Ctrl+L** - Clear chat history
- **Ctrl+O** - Toggle streaming mode (saved to config)
- **Escape** - Cancel generation / Quit

## Slash commands

- `/help`, `/h` - Show help
- `/clear`, `/c` - Clear chat history
- `/retry`, `/r` - Regenerate last response
- `/copy` - Copy last response to clipboard
- `/context`, `/ctx` - Show context info
- `/prompt` - Show system prompt
- `/sys <msg>`, `/system <msg>` - Inject a system message into conversation
- `/model`, `/m` - Show current model
- `/personality`, `/p` - List personalities
- `/p <n>` - Switch to personality by number/name
- `/config` - List config profiles
- `/config <n>` - Switch to config by number/name (restarts app)
- `/impersonate`, `/imp` - Generate user response suggestion (long-form)
- `/imps` - Short impersonate (suggestion-length, under 15 words)
- `/suggest` - Toggle auto-suggest after responses
- `/project` - Toggle local prompt append
- `/stats`, `/st` - Show generation statistics (TTFT, t/s, tokens)
- `/compact` - Summarize conversation to free context

## System instructions

LLM instructions for commands (compact, impersonate, impersonate_short) are stored in `system_instructions.json` at the project root. Users can edit this file to customize the instructions. The app exits with an error if the file is missing or invalid JSON.

## Personalities

System prompt templates in `~/.config/ochat/personalities/`. Bundled personalities (default, creative, storyteller) are copied on first run.

**Local prompt append** (`append_local_prompt`):
- ON: system prompt + local `agent.md`/`system.md` appended
- OFF: system prompt only, local files ignored

## Impersonate & auto-suggest

`/impersonate` or `/imp` generates a suggested user response using the LLM:
- Useful for roleplay, brainstorming, or getting the LLM's perspective on what to ask next
- LLM generates what the user might say next (long-form)
- Result is placed in input field (not sent automatically)
- User can edit or send directly

`/imps` is the short variant (under 15 words), same workflow.

**Auto-suggest** (`auto_suggest` config option, default ON):
- After each LLM response, a short suggestion is generated in the background
- Appears as placeholder text in the input field
- **Tab** accepts the suggestion into the input
- Typing anything else replaces it naturally
- If `append_local_prompt` is ON and a project file exists, auto-suggest also fires on startup (suggesting a first message based on project context)
- Toggle with `/suggest` command
- Uses `impersonate_short` instruction from `system_instructions.json`

**Implementation note**: Auto-suggest uses the input placeholder (not Textual's `_suggestion` reactive) because Textual's `Suggester` mechanism skips empty inputs entirely (`_watch_value` clears `_suggestion` when value is falsy). The placeholder approach is the only way to show ghost text in an empty Textual Input.

## Tech stack

- Python 3.11 (required for type syntax)
- Textual for TUI
- ollama-python for Ollama API
- openai for OpenAI-compatible API (llama.cpp, LM Studio, vLLM, etc.)
- Backend abstraction layer (`BackendProtocol`) for unified API access

## Debugging

Logging is disabled by default. Enable with `-d` flag:
```bash
ochat -d
```

Log files are written to temp (auto-cleaned after 7 days):
- **Unix/Mac**: `/tmp/ochat-YYYYMMDD-HHMMSS.log`
- **Windows**: `%TEMP%/ochat-YYYYMMDD-HHMMSS.log`

Captures: app start, commands, errors, Textual exceptions.

Logging uses the `"ochat"` logger hierarchy. Each module uses `_log = logging.getLogger(__name__)` (gives `ochat.app`, `ochat.commands`, etc.). The `-d` flag configures the parent `"ochat"` logger with a FileHandler.

## Gotchas

- **ollama-python global client**: Never use `ollama.chat()`, `ollama.list()` etc. (module-level functions). The library creates its default client at import time using `OLLAMA_HOST` env var — setting the env var after `import ollama` has no effect. Always use an explicit `ollama.Client(host=...)` instance (`self.client` in `OllamaBackend`). Host priority: config.conf `[server] host` > `OLLAMA_HOST` env var > `http://localhost:11434`.

- **SSL verification**: `verify_ssl` in `[server]` controls SSL certificate verification. Only written to config when `false` (default is `true`/absent). When `false`, `ollama.Client` gets `verify=False` and `openai.OpenAI` gets a custom `httpx.Client(verify=False)`. The setup wizard auto-detects SSL errors on HTTPS hosts and offers to disable verification.

- **Backend abstraction**: All backend operations go through `BackendProtocol` interface (`chat()`, `list_models()`, `get_info()`, `extract_chunk()`, `extract_result()`, `initialize()`). I/O methods are all async; chat streams yield an async iterator. The `_chat_call()` helper in `GenerationMixin` awaits `self.backend.chat(...)` directly — no mode branching, no `to_thread`. `AutoBackend` tries Ollama→llama.cpp→OpenAI sequentially inside `initialize()` (locked + idempotent), called from every entry point. Each backend wraps a native async client (`ollama.AsyncClient` for Ollama, `openai.AsyncOpenAI` for OpenAI/llama.cpp).

- **Backend-specific features**: `num_ctx` is passed through at construction (`OllamaBackend(num_ctx=...)`) or from `/info` (llama.cpp). Context tracking uses real token counts from API responses when available (`include_usage` for llama.cpp, `usage` field for Ollama), otherwise falls back to ~4 chars/token approximation. OpenAI mode has no context tracking.

- **AutoBackend missing fields**: `AutoBackend.context_tokens` and `AutoBackend.n_ctx` delegate to `_detected_backend`. If `context_tokens` is unavailable, context info shows `~` prefix.

- **Streaming iteration**: `_consume_chunks` iterates the stream via `async for chunk in stream` directly — no thread hops, no sentinels. Cancellation propagates naturally as `asyncio.CancelledError`. Graceful in-flight cancel (Escape) still uses `self._generation_cancelled` flag checked inside the loop.

- **Markdown render throttling**: `Message` extends Textual's `Markdown`, which makes mistune re-parse the full buffer on every `update()`. Per-chunk updates are O(n²) at high token rates. `_consume_chunks` throttles streaming updates to ~50ms (via `time.monotonic()`); the final render is unconditional once the stream ends. Status-bar updates stay per-chunk (cheap).

- **Generation lock**: Generation uses the `_generating_lock` async context manager as the single source of truth for `is_generating` + input-disable state. Background tasks (auto-suggest) are spawned **after** the lock releases so the user can type while the suggestion runs. Do not set `self.is_generating` manually.

## Generation modes

Both streaming and non-streaming modes use `stream=True` under the hood:

- **Streaming**: shows text live as tokens arrive
- **Non-streaming**: buffers tokens, shows `⠋ thinking... 3.2s (42 chunks)` progress, then displays the full response prefixed with `*thought for 3.2s*`

Both modes start with a "waiting for first token" spinner and track TTFT.

## Context tracking

- Token count is approximated (~4 chars/token) or uses real token counts from API responses when available (Ollama: `usage` field, llama.cpp: `include_usage`)
- Context tracking is disabled in OpenAI mode (`_context_pct()` returns 0) since `num_ctx` is not applicable
- Context usage percentage shown in `/context` and `/stats`
- Status bar shows `ctx: N% remaining` when usage exceeds 85%
- Status bar shows `⚠ Context length probably exceeded` when over 100%
- One-shot system message warning at 80% usage suggesting `/compact`
- OChat class renamed from OllamaChat, `backend_type` in config determines backend

## Code quality

Run linters before committing (from venv):

```bash
pylint ochat/ ochat.py
pyflakes ochat/ ochat.py
radon cc ochat/ ochat.py -s -a
```

**Rules**: no D or F in cyclomatic complexity. C is tolerable for orchestrator functions. Target A/B for new code. Ignore pylint `E0401` (import-error) — false positives from venv-only deps.

## Notes for future development

- No conversation persistence yet (could add /save, /load commands)
- No multi-turn editing (could add message editing)
- No multiline input (Input widget limitation, would need TextArea)
