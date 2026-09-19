"""Local secret storage kept outside versioned configuration."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Mapping

SECRET_FIELDS = {
    "LUMINA_SESSION_SECRET", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "LLM_API_KEY",
    "BRAVE_SEARCH_KEY", "SERPAPI_KEY", "ELEVENLABS_API_KEY", "TELEGRAM_TOKEN",
    "WHATSAPP_TWILIO_SID", "WHATSAPP_TWILIO_TOKEN", "WHATSAPP_WEBHOOK_SECRET",
    "HOME_ASSISTANT_TOKEN", "BODY_BRIDGE_TOKEN",
}


def secret_path() -> Path:
    return Path(".venv") / ".env"


def _read(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    result: Dict[str, str] = {}
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                result[key.strip()] = value.strip().strip('"').strip("'")
    except OSError:
        pass
    return result


def _scrub_legacy_file() -> None:
    legacy_path = Path(".env")
    if not legacy_path.exists():
        return
    try:
        lines = []
        for line in legacy_path.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                key = line.split("=", 1)[0].strip()
                if key in SECRET_FIELDS:
                    line = f"{key}="
            lines.append(line)
        legacy_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    except OSError:
        pass


def write_secrets(values: Mapping[str, str]) -> None:
    target = secret_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    existing = _read(target)
    existing.update({key: str(value) for key, value in values.items() if key in SECRET_FIELDS and value})
    lines = ["# PandoraBOX local secrets - never commit this file"]
    lines.extend(f"{key}={value}" for key, value in sorted(existing.items()) if key in SECRET_FIELDS)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    try:
        target.chmod(0o600)
    except OSError:
        pass


def load_secrets() -> Dict[str, str]:
    target = secret_path()
    secrets = _read(target)
    legacy = _read(Path(".env"))
    changed = False
    for key in SECRET_FIELDS:
        if not secrets.get(key) and legacy.get(key):
            secrets[key] = legacy[key]
            changed = True
    if changed or (secrets and not target.exists()):
        write_secrets(secrets)
    _scrub_legacy_file()
    return secrets


def resolve(name: str, value: object = "") -> str:
    if isinstance(value, str) and value.startswith("@env:"):
        name = value[5:].strip() or name
    secrets = load_secrets()
    return os.environ.get(name) or secrets.get(name, "") or (str(value) if value and not str(value).startswith("@env:") else "")


def store(name: str, value: object) -> None:
    if name in SECRET_FIELDS and value:
        write_secrets({name: str(value)})
        os.environ[name] = str(value)


def config_reference(name: str, value: object = "") -> str:
    return f"@env:{name}" if value or load_secrets().get(name) else ""
