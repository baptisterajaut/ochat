# ollama-chat

*Made with Claude to talk with non-Claude.*

A setup-and-forget Ollama chat TUI. One screen, no sub-menus, plenty of creature comforts.

![Python](https://img.shields.io/badge/python-3.11+-blue)
![License](https://img.shields.io/badge/license-public%20domain-brightgreen)
![Vibe](https://img.shields.io/badge/vibe-coded-ff69b4)

![ochat screenshot](.github/ochat.jpg)

## Why?

I wanted something like [Open WebUI](https://github.com/open-webui/open-webui) but in terminal. A chat interface without needing a full browser eating 8GB of RAM just to exist.

It started as a bare-bones wrapper, but grew into something I actually enjoy using daily — switchable personalities, auto-suggest, impersonate mode, context tracking, multiple config profiles. Still not Open WebUI, but not bare-bones either.

I also wanted a TUI as pleasant to use as [Claude Code](https://claude.com/claude-code) but for local Ollama models.

This project was entirely vibe-coded with Claude. It does exactly what I need.

## Features

- Clean TUI with streaming responses
- Switchable personalities (system prompts)
- Multiple config profiles (easily switch between setups)
- Project-specific prompts (auto-loads `agent.md`/`system.md` from current directory)
- Slash commands (`/help`, `/retry`, `/personality`, `/config`, `/impersonate`, etc.)
- Keyboard shortcuts (Ctrl+O toggle streaming, Escape to cancel, etc.)
- Auto-suggest: after each response, a short suggestion appears in the input (Tab to accept)
- Impersonate mode: LLM suggests what you'd say next (`/imp` long-form, `/imps` short)
- Context tracking with usage warnings and `/compact` to summarize conversation
- Generation stats (TTFT, tokens/s) in status bar and `/stats`
- Persistent configuration
- Pass-through for any Ollama model option (temperature, top_p, etc.) via config file
- OpenAI-compatible API fallback (LM Studio, llama.cpp, vLLM, etc.) - **untested**

## Non-features

- It's just a client. No plan to have it start Ollama by itself or llama-cpp-python or anything
- Made for Ollama only, OpenAI fallback is there "just in case" and is not tested
- No conversation persistence yet (could use `agent.md` generation as lightweight conversation memory in the future)
- No multi-model conversations
- No model templates (chat templates are handled by Ollama/your server)
- No multiline input (TextArea doesn't support suggesters, and I prefer a single-line input with autocomplete over multiline without it — or making Claude reinvent the wheel and turning this codebase from "only Claude and God understand it" to "only God understands it")
- No RAG, no agents, no tools

## Installation

```bash
git clone https://github.com/baptisterajaut/ollama-chat
cd ollama-chat

# Optional: symlink to your bin
ln -s "$(pwd)/ochat.sh" ~/bin/ochat
```

The virtual environment and dependencies are created automatically on first run.

### Windows (untested)

```cmd
git clone https://github.com/baptisterajaut/ollama-chat
cd ollama-chat
ochat.bat
```

Windows support is experimental and untested.

## Usage

```bash
ochat                    # Start chatting (first run launches setup wizard)
ochat -C                 # Configure (host, model, personality)
ochat -m llama3.2        # Override model
ochat --new              # Create a new named config profile
ochat --use-config NAME  # Use a named config for this session
ochat --use-config NAME --as-default  # Switch to named config permanently
ochat -d                 # Enable debug logging
ochat --help             # Show options
```

### Keyboard shortcuts

| Key | Action |
|-----|--------|
| `Ctrl+C` | Clear input |
| `Ctrl+D` | Quit |
| `Ctrl+L` | Clear chat |
| `Ctrl+O` | Toggle streaming |
| `Escape` | Cancel generation / Quit |
| `Tab` | Accept suggestion / autocomplete command |

### Slash commands

| Command | Action |
|---------|--------|
| `/help` | Show help |
| `/retry` | Regenerate last response |
| `/copy` | Copy last response to clipboard |
| `/clear` | Clear chat history |
| `/personality` | List/switch personalities |
| `/config` | List/switch config profiles (restarts app) |
| `/impersonate` | Generate suggested user response (long-form) |
| `/imps` | Short impersonate (under 15 words) |
| `/suggest` | Toggle auto-suggest after responses |
| `/project` | Toggle project prompt merge |
| `/prompt` | Show current system prompt |
| `/sys <msg>` | Inject a system message (alias: `/system`) |
| `/model` | Show current model |
| `/stats` | Show generation statistics (TTFT, t/s, tokens) |
| `/compact` | Summarize conversation to free context |
| `/context` | Show context info |

## Configuration

Config lives in `~/.config/ollama-chat/`:
- `config.conf` - Default settings (host, model, context size, etc.)
- `*.conf` - Named config profiles
- `personalities/` - System prompt templates (`.md` files)

### Multiple config profiles

You can create and switch between different configurations (different models, personalities, settings):

```bash
ochat --new                          # Create a new profile from scratch
ochat --use-config my-creative       # Use profile for this session only
ochat --use-config my-creative --as-default  # Make it the new default
```

Or switch in-app with `/config`. When switching with `--as-default`, your current config is backed up automatically.

Bundled personalities (copied on first run):
- `default` - Helpful, concise assistant
- `creative` - Brainstorming and unconventional ideas
- `storyteller` - Narrative and creative writing

### Advanced model options

You can add any Ollama model parameter in `config.conf`. These are passed directly to the API without validation:

```ini
[model_options]
temperature = 0.7
top_p = 0.9
top_k = 40
min_p = 0.05
repeat_penalty = 1.1
# Any other Ollama option works too
```

See [Ollama docs](https://docs.ollama.com/modelfile#valid-parameters-and-values) for available options. Not exposed in the setup wizard - edit the config file manually.

### OpenAI-compatible mode (untested)

If Ollama isn't available at the configured host, the app falls back to OpenAI-compatible API mode. This should work with LM Studio, llama.cpp server, vLLM, text-generation-inference, and other servers exposing the standard `/v1/chat/completions` endpoint.

**Limitations in OpenAI mode:**
- No interactive model selection (model listing is used for connection testing only — you must configure the model name manually)
- `num_ctx` and `model_options` are ignored (server-side settings apply)
- Context usage tracking is disabled (percentages and warnings don't apply)
- Setup wizard won't work (configure `config.conf` manually)

**To use with LM Studio:**
```ini
[server]
host = http://localhost:1234

[defaults]
model = your-loaded-model-name
```

### Self-signed certificates (homelab)

For HTTPS hosts with self-signed certificates, the setup wizard will detect SSL errors and offer to disable verification. You can also set it manually in `config.conf`:

```ini
[server]
host = https://my-homelab:11434
verify_ssl = false
```

The greeting will show "Connected (OpenAI mode)" when using this fallback.

## Debugging

Logging is disabled by default. Use `-d` to enable:

```bash
ochat -d
```

Log files are written to temp (auto-cleaned after 7 days):
- **Unix/Mac**: `/tmp/ollama-chat-YYYYMMDD-HHMMSS.log`
- **Windows**: `%TEMP%\ollama-chat-YYYYMMDD-HHMMSS.log`

## Code quality

Average cyclomatic complexity: **B (5.1)** — no D or F.

<details>
<summary>Functions rated B and C (radon cc)</summary>

| Function | File | CC | Grade |
|----------|------|----|-------|
| `_handle_command` | commands.py | 20 | C |
| `_generate_response` | generation.py | 16 | C |
| `main` | app.py | 14 | C |
| `run_setup` | config.py | 13 | C |
| `_handle_config_command` | commands.py | 12 | C |
| `_handle_compact` | commands.py | 12 | C |
| `_handle_personality_command` | commands.py | 10 | B |
| `_handle_project_toggle` | commands.py | 9 | B |
| `_status_text` | app.py | 9 | B |
| `load_config` | config.py | 9 | B |
| `switch_config_to_default` | config.py | 9 | B |
| `_connect_ollama` | config.py | 9 | B |
| `_consume_chunks` | generation.py | 8 | B |
| `_save_new_profile` | config.py | 8 | B |
| `_handle_retry` | commands.py | 7 | B |
| `_show_greeting` | app.py | 7 | B |
| `on_input_submitted` | app.py | 7 | B |
| `load_system_prompt` | config.py | 7 | B |
| `_select_numbered` | config.py | 7 | B |
| `save_config` | config.py | 6 | B |
| `_handle_impersonate` | commands.py | 6 | B |
| `_handle_impersonate_short` | commands.py | 6 | B |
| `CommandSuggester` | widgets.py | 6 | B |

</details>

## Alternatives

If you want a more full-featured Ollama TUI, check out [parllama](https://github.com/paulrobello/parllama) and [oterm](https://github.com/ggozad/oterm). Both are capable and well-maintained. Their busier interfaces weren't to my taste — this project exists because I wanted a clean, single-screen interface with no panels or sidebars to manage.

## License

Public domain. You can't copyright AI-generated code anyway.

---

## "Vibe coding is awful, LLMs steal real code, you should've coded it yourself"

*Frankly my dear, I don't give a damn.*
