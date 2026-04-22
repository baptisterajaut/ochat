"""Configuration management for ochat."""

import configparser
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

_BACKUP_NAME_RE = re.compile(r"^[\w.-]+$")

import ollama

# Config paths
OLD_CONFIG_DIR = Path.home() / ".config" / "ollama-chat"
NEW_CONFIG_DIR = Path.home() / ".config" / "ochat"
CONFIG_DIR = NEW_CONFIG_DIR

# Config file
CONFIG_FILE = CONFIG_DIR / "config.conf"
PERSONALITIES_DIR = CONFIG_DIR / "personalities"


def check_config_migration() -> None:
    """Migrate from old config dir (~/.config/ollama-chat) to new one (~/.config/ochat)."""
    if OLD_CONFIG_DIR.exists() and not NEW_CONFIG_DIR.exists():
        shutil.move(str(OLD_CONFIG_DIR), str(NEW_CONFIG_DIR))
        print(f"Migrated config from {OLD_CONFIG_DIR} to {NEW_CONFIG_DIR}")
    elif OLD_CONFIG_DIR.exists() and NEW_CONFIG_DIR.exists():
        print("Warning: old config dir detected alongside new one, please check it manually")


# Bundled personalities (shipped with the app)
BUNDLED_PERSONALITIES_DIR = Path(__file__).parent.parent / "personalities"

# Default config values (single source of truth)
DEFAULT_CONFIG = {
    "host": "",  # filled by get_default_host() at runtime
    "model": "llama3.2",
    "num_ctx": 4096,
    "personality": "default",
    "append_local_prompt": True,
    "streaming": True,
    "verify_ssl": True,
    "auto_suggest": True,
    "thinking_enabled": True,
    "model_options": {},
    "config_name": "",
    "backend": "ollama",
}


def get_default_host(backend: str = "ollama") -> str:
    """Get default host from environment or fallback."""
    if backend == "ollama":
        env_host = os.environ.get("OLLAMA_HOST")
        if env_host:
            return env_host
        return "http://localhost:11434"
    return "http://localhost:8080"


def ensure_personalities_dir() -> None:
    """Ensure personalities directory exists with bundled personalities."""
    PERSONALITIES_DIR.mkdir(parents=True, exist_ok=True)

    # Copy bundled personalities if user folder is empty
    existing = list(PERSONALITIES_DIR.glob("*.md"))
    if not existing and BUNDLED_PERSONALITIES_DIR.exists():
        for src in BUNDLED_PERSONALITIES_DIR.glob("*.md"):
            dst = PERSONALITIES_DIR / src.name
            if not dst.exists():
                dst.write_text(src.read_text())


def list_personalities() -> list[str]:
    """List available personality names (without .md extension)."""
    ensure_personalities_dir()
    return sorted([
        p.stem for p in PERSONALITIES_DIR.glob("*.md")
        if p.is_file()
    ])


def load_personality(name: str) -> str | None:
    """Load a personality by name."""
    ensure_personalities_dir()
    path = PERSONALITIES_DIR / f"{name}.md"
    if path.exists():
        return path.read_text().strip()
    return None


def list_configs() -> list[str]:
    """List available config files (without .conf extension)."""
    return sorted([
        p.stem for p in CONFIG_DIR.glob("*.conf")
        if p.is_file() and p.stem != "config"
    ])


def load_config(config_file: Path | None = None) -> dict:
    """Load configuration from file."""
    config = {**DEFAULT_CONFIG, "host": get_default_host()}

    file_to_load = config_file or CONFIG_FILE
    if file_to_load.exists():
        parser = configparser.ConfigParser()
        parser.read(file_to_load)

        config["host"] = parser.get("server", "host", fallback=config["host"])
        config["verify_ssl"] = parser.getboolean("server", "verify_ssl", fallback=config["verify_ssl"])
        config["model"] = parser.get("defaults", "model", fallback=config["model"])
        config["num_ctx"] = parser.getint("defaults", "num_ctx", fallback=config["num_ctx"])
        config["personality"] = parser.get("defaults", "personality", fallback=config["personality"])
        config["append_local_prompt"] = parser.getboolean("defaults", "append_local_prompt", fallback=config["append_local_prompt"])
        config["streaming"] = parser.getboolean("defaults", "streaming", fallback=config["streaming"])
        config["auto_suggest"] = parser.getboolean("defaults", "auto_suggest", fallback=config["auto_suggest"])
        config["thinking_enabled"] = parser.getboolean("defaults", "thinking_enabled", fallback=config["thinking_enabled"])
        config["config_name"] = parser.get("defaults", "config_name", fallback=config["config_name"])
        config["backend"] = parser.get("defaults", "backend", fallback=config["backend"])

        # Load model options (empty string = not set, any key allowed)
        if parser.has_section("model_options"):
            model_options = {}
            for key in parser.options("model_options"):
                val = parser.get("model_options", key).strip()
                if val:
                    # Try to convert: bool, int, float, fallback to string
                    if val.lower() in ("true", "false"):
                        model_options[key] = val.lower() == "true"
                    else:
                        try:
                            model_options[key] = int(val)
                        except ValueError:
                            try:
                                model_options[key] = float(val)
                            except ValueError:
                                model_options[key] = val
            config["model_options"] = model_options

    return config


def save_config(
    host: str,
    model: str,
    num_ctx: int,
    personality: str = "default",
    append_local_prompt: bool = True,
    streaming: bool = True,
    model_options: dict | None = None,
    config_name: str = "",
    config_file: Path | None = None,
    verify_ssl: bool = True,
    auto_suggest: bool = True,
    thinking_enabled: bool = True,
    backend: str = "ollama",
) -> None:
    """Save configuration to file."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    parser = configparser.ConfigParser()
    server = {"host": host}
    if not verify_ssl:
        server["verify_ssl"] = "false"
    parser["server"] = server
    defaults = {
        "model": model,
        "num_ctx": str(num_ctx),
        "personality": personality,
        "append_local_prompt": str(append_local_prompt).lower(),
        "streaming": str(streaming).lower(),
        "auto_suggest": str(auto_suggest).lower(),
        "thinking_enabled": str(thinking_enabled).lower(),
        "backend": backend,
    }
    if config_name:
        defaults["config_name"] = config_name
    parser["defaults"] = defaults

    # Advanced model options (pass-through, any key allowed)
    if model_options is None:
        model_options = {}
    parser["model_options"] = {k: str(v) for k, v in model_options.items()}

    file_to_save = config_file or CONFIG_FILE
    with open(file_to_save, "w", encoding="utf-8") as f:
        parser.write(f)


def save_config_dict(config: dict, config_file: Path | None = None, **overrides) -> None:
    """Save config from a dict, with optional field overrides."""
    merged = {**config, **overrides}
    save_config(
        host=merged["host"],
        model=merged["model"],
        num_ctx=merged["num_ctx"],
        personality=merged.get("personality", "default"),
        append_local_prompt=merged.get("append_local_prompt", True),
        streaming=merged.get("streaming", True),
        model_options=merged.get("model_options"),
        config_name=merged.get("config_name", ""),
        config_file=config_file,
        verify_ssl=merged.get("verify_ssl", True),
        auto_suggest=merged.get("auto_suggest", True),
        thinking_enabled=merged.get("thinking_enabled", True),
        backend=merged.get("backend", "ollama"),
    )


def update_config(**overrides) -> None:
    """Update specific fields in config, preserving other settings."""
    config = load_config()
    save_config_dict(config, **overrides)


def switch_config_to_default(new_config_name: str, interactive: bool = True) -> tuple[bool, str]:
    """Switch a named config to be the default config.conf.

    Backs up current config.conf to {config_name}.conf first.
    If interactive=True (CLI), prompts for backup name if unset.
    If interactive=False (TUI), silently defaults to "config-default".
    """
    new_config_file = CONFIG_DIR / f"{new_config_name}.conf"
    if not new_config_file.exists():
        return False, f"Config '{new_config_name}' not found"

    if CONFIG_FILE.exists():
        current_config = load_config()
        current_name = current_config.get("config_name", "")

        if not current_name:
            if interactive:
                while True:
                    current_name = input("Name for current config backup [config-default]: ").strip()
                    if not current_name:
                        current_name = "config-default"
                    if _BACKUP_NAME_RE.match(current_name):
                        break
                    print("Invalid name: use only letters, digits, '_', '-', '.'")
                    current_name = ""
            if not current_name:
                current_name = "config-default"

            backup_file = CONFIG_DIR / f"{current_name}.conf"

            if interactive and backup_file.exists():
                confirm = input(f"'{current_name}.conf' exists. Overwrite? [y/N]: ").strip().lower()
                if confirm not in ("y", "yes"):
                    return False, "Cancelled"
        else:
            # Named config: overwrite its backup silently (it's saving "back home")
            backup_file = CONFIG_DIR / f"{current_name}.conf"

        save_config_dict(current_config, config_file=backup_file, config_name=current_name)
        print(f"Backed up current config to {backup_file.name}")

    shutil.copy(new_config_file, CONFIG_FILE)
    print(f"Switched to config '{new_config_name}'")

    return True, f"Now using '{new_config_name}'"


def load_project_prompt() -> str | None:
    """Load project prompt file from current directory."""
    for name in ["agent.md", "system.md", "system.txt", "AGENT.md"]:
        p = Path(name)
        if p.exists():
            return p.read_text(encoding="utf-8").strip()
    return None


def load_system_prompt(
    path: str | None,
    personality: str | None = None,
    append_local_prompt: bool = True,
) -> tuple[str | None, str | None]:
    """Load system prompt.

    Modes:
    - If -s path provided: use that file alone
    - If append_local_prompt=True: personality + project file (agent.md, etc.)
    - If append_local_prompt=False: personality alone

    Returns:
        (system_prompt_content, personality_name or None)
    """
    if path:
        p = Path(path)
        if not p.exists():
            print(f"Error: system prompt file not found: {path}", file=sys.stderr)
            sys.exit(1)
        return p.read_text(encoding="utf-8").strip(), None

    ensure_personalities_dir()
    personality_name = personality or "default"
    personality_content = load_personality(personality_name)
    if not personality_content:
        personality_content = load_personality("default")
        personality_name = "default"

    if append_local_prompt:
        project_prompt = load_project_prompt()
        if project_prompt:
            combined = f"{personality_content}\n\n---\n\n{project_prompt}"
            return combined, personality_name

    return personality_content, personality_name


def _backup_config_interactive(config: dict, default_name: str = "") -> None:
    """Prompt for backup name, check overwrite, and save. Used by run_setup."""
    backup_name = default_name
    if not backup_name:
        while True:
            backup_name = input("Name for current config backup [config-default]: ").strip()
            if not backup_name:
                backup_name = "config-default"
            if _BACKUP_NAME_RE.match(backup_name):
                break
            print("Invalid name: use only letters, digits, '_', '-', '.'")
            backup_name = ""

    backup_file = CONFIG_DIR / f"{backup_name}.conf"

    if backup_file.exists():
        confirm = input(f"'{backup_name}.conf' exists. Overwrite? [y/N]: ").strip().lower()
        if confirm not in ("y", "yes"):
            print("Backup skipped")
            return

    save_config_dict(config, config_file=backup_file, config_name=backup_name)
    print(f"Backed up to {backup_file.name}")


def _is_ssl_error(host: str, exc: Exception) -> bool:
    if not host.startswith("https"):
        return False
    msg = str(exc).upper()
    return any(kw in msg for kw in ("SSL", "CERTIFICATE", "VERIFY"))


def _prompt_disable_ssl() -> bool:
    ans = input(
        "SSL certificate error. Ignore verification? (homelab only) [y/N]: "
    ).strip().lower()
    return ans in ("y", "yes")


def _backend_probe(host: str, backend_type: str):
    """Return (list_fn, extract_ids) for a backend type. list_fn takes a verify bool."""
    if backend_type == "ollama":
        def list_fn(verify: bool):
            return ollama.Client(host=host, verify=verify).list()
        return list_fn, lambda resp: [m.model for m in resp.models]

    if backend_type in ("openai", "llama_cpp"):
        import httpx
        import openai
        base_url = f"{host.rstrip('/')}/v1"
        api_key = "llama.cpp" if backend_type == "llama_cpp" else "not-needed"
        def list_fn(verify: bool):
            kwargs = {"base_url": base_url, "api_key": api_key}
            if not verify:
                kwargs["http_client"] = httpx.Client(verify=False)
            return openai.OpenAI(**kwargs).models.list()
        return list_fn, lambda resp: [m.id for m in resp.data]

    print(f"Unknown backend type: {backend_type}")
    sys.exit(1)


def __connect_backend(host: str, backend_type: str, verify_ssl: bool) -> tuple[list[str], bool]:
    """Connect to backend, handle SSL retry. Returns (models, verify_ssl)."""
    list_fn, extract = _backend_probe(host, backend_type)
    try:
        return extract(list_fn(verify_ssl)), verify_ssl
    except Exception as e:
        if _is_ssl_error(host, e) and _prompt_disable_ssl():
            try:
                return extract(list_fn(False)), False
            except Exception as e2:
                print(f"Connection error: {e2}")
                sys.exit(1)
        print(f"Connection error: {e}")
        sys.exit(1)


def _select_numbered(label: str, items: list[str], default_idx: int,
                     markers: dict[int, str] | None = None) -> str:
    """Display a numbered list and prompt for selection. Returns chosen item."""
    print(f"\nAvailable {label}:")
    for i, item in enumerate(items, 1):
        marker = (markers or {}).get(i, "")
        print(f"  {i}. {item}{marker}")

    while True:
        choice = input(f"\nSelect {label.rstrip('s')} [{default_idx}]: ").strip()
        if not choice:
            choice = str(default_idx)
        try:
            idx = int(choice)
            if 1 <= idx <= len(items):
                return items[idx - 1]
            print(f"Invalid choice. Enter a number between 1 and {len(items)}")
        except ValueError:
            print("Enter a valid number")


def _input_positive_int(prompt: str, default: int) -> int:
    """Prompt for a positive integer with a default value."""
    while True:
        raw = input(f"\n{prompt} [{default}]: ").strip()
        if not raw:
            return default
        try:
            val = int(raw)
            if val > 0:
                return val
            print("Size must be positive")
        except ValueError:
            print("Enter a valid number")


def run_setup(create_new: bool = False) -> None:
    """Run interactive setup wizard."""
    if create_new:
        print("ochat - Create new config profile\n")
        config = {**DEFAULT_CONFIG, "host": get_default_host()}
        config_existed = False
        old_config_name = ""
    else:
        print("ochat configuration\n")
        config_existed = CONFIG_FILE.exists()
        config = load_config()
        old_config_name = config.get("config_name", "")

    # 1. Ask for backend type
    backends = ["ollama", "openai", "llama_cpp"]
    default_backend = config.get("backend", "ollama")
    backend_idx = backends.index(default_backend) + 1 if default_backend in backends else 1
    backend = _select_numbered("backends", backends, backend_idx)

    # 2. Ask for host
    if create_new:
        default_host = get_default_host(backend)
    else:
        default_host = config["host"]
    raw_host = input(f"Host [{default_host}]: ").strip() or default_host
    if not raw_host.startswith(("http://", "https://")):
        raw_host = "http://" + raw_host
    host = raw_host

    # 3. Connect and select model
    verify_ssl = config.get("verify_ssl", True)
    print(f"\nConnecting to {backend.capitalize()}...")
    models, verify_ssl = __connect_backend(host, backend, verify_ssl)

    if not models:
        print("No models found. Install one with 'ollama pull <model>'")
        sys.exit(1)

    default_idx = 1
    if config["model"] in models:
        default_idx = models.index(config["model"]) + 1
    model = _select_numbered("models", models, default_idx)

  # 3. Ask for context size
    if backend == "llama_cpp":
        num_ctx = _input_positive_int("Context size", config["num_ctx"])
    elif backend == "openai":
        print("\nContext size: N/A (server-determined)")
        num_ctx = config["num_ctx"]
    else:
        num_ctx = _input_positive_int("Context size", config["num_ctx"])

    # 4. Select personality
    ensure_personalities_dir()
    personalities = list_personalities()

    default_p_idx = 1
    if config["personality"] in personalities:
        default_p_idx = personalities.index(config["personality"]) + 1
    markers = {
        i: " (current)" for i, p in enumerate(personalities, 1)
        if p == config["personality"]
    }
    personality = _select_numbered("personalities", personalities, default_p_idx, markers)

    # 5. Edit personality?
    edit_prompt = input(f"\nEdit personality '{personality}'? [y/N]: ").strip().lower()
    if edit_prompt in ("y", "yes"):
        editor = os.environ.get("EDITOR", "nano")
        subprocess.run([editor, str(PERSONALITIES_DIR / f"{personality}.md")], check=False)

    # 6. Append local prompt?
    print("\nAppend local prompt (agent.md, system.md from current directory)?")
    print("  If yes: system prompt + local file merged")
    print("  If no: system prompt only")
    default_yn = "Y/n" if config["append_local_prompt"] else "y/N"
    project_choice = input(f"Append local prompt? [{default_yn}]: ").strip().lower()
    if project_choice in ("y", "yes"):
        append_local_prompt = True
    elif project_choice in ("n", "no"):
        append_local_prompt = False
    else:
        append_local_prompt = config["append_local_prompt"]

    # Common kwargs for save
    save_kwargs = dict(host=host, model=model, num_ctx=num_ctx,
                       personality=personality, append_local_prompt=append_local_prompt,
                       verify_ssl=verify_ssl, backend=backend)

    if create_new:
        _save_new_profile(config, save_kwargs)
    else:
        _save_updated_config(config, save_kwargs, config_existed, old_config_name)

    print(f"Personalities in {PERSONALITIES_DIR}/")


def _save_new_profile(config: dict, save_kwargs: dict) -> None:
    """Handle save flow for creating a new named config profile."""
    while True:
        new_name = input("\nName for this config profile: ").strip()
        if not new_name:
            print("Name required")
            continue
        if new_name == "config":
            print("Cannot use 'config' as name (reserved)")
            continue
        break

    apply_default = input("\nApply this config as default? [y/N]: ").strip().lower()

    if apply_default in ("y", "yes"):
        if CONFIG_FILE.exists():
            current_config = load_config()
            _backup_config_interactive(current_config, current_config.get("config_name", ""))
        save_config_dict(config, config_name=new_name, **save_kwargs)
        print(f"\nConfig '{new_name}' saved as default")
    else:
        new_file = CONFIG_DIR / f"{new_name}.conf"
        if new_file.exists():
            confirm = input(f"'{new_name}.conf' exists. Overwrite? [y/N]: ").strip().lower()
            if confirm not in ("y", "yes"):
                print("Cancelled")
                return
        save_config_dict(config, config_file=new_file, config_name=new_name, **save_kwargs)
        print(f"\nConfig saved to {new_file}")
        print(f"Use with: ochat --use-config {new_name}")


def _save_updated_config(config: dict, save_kwargs: dict,
                         config_existed: bool, old_config_name: str) -> None:
    """Handle save flow for updating existing config."""
    if config_existed:
        backup_choice = input("\nBackup current config before saving? [y/N]: ").strip().lower()
        if backup_choice in ("y", "yes"):
            _backup_config_interactive(config, old_config_name)

    save_config_dict(config, **save_kwargs)
    print(f"\nConfiguration saved to {CONFIG_FILE}")
