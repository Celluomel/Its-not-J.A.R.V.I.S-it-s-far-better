"""
core/bootstrap.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Security bootstrap — must run before any other module reads os.environ.

Creates .env from .env.example if missing, loads it into os.environ,
generates LUMINA_SESSION_SECRET if absent, and restricts file permissions.

Called as the very first statement in app.py:

    from core.bootstrap import bootstrap_security
    bootstrap_security()
"""
"""Robot Agent Pro - Main Application with Vision Integration"""
# ══════════════════════════════════════════════════════════════════════════════
#  SECURITY BOOTSTRAP — runs before everything else
#  Ensures .env exists, LUMINA_SESSION_SECRET is set, and env vars are loaded
# ══════════════════════════════════════════════════════════════════════════════
import os
import sys
import platform
import secrets
import shutil
from pathlib import Path
from secret_store import load_secrets

def _bootstrap_security():
    """
    First-run security setup.
    - Creates .env from .env.example if missing
    - Generates LUMINA_SESSION_SECRET if not set
    - Loads .env into os.environ on all platforms
    - Prints clear guidance on what was done
    """
    _HERE    = Path(__file__).parent
    _ENV     = _HERE / ".env"
    _EXAMPLE = _HERE / ".env.example"

    created_env = False

    # Secrets migrated to .venv/.env take precedence over the legacy root
    # .env and are exposed only through the process environment.
    for key, value in load_secrets().items():
        if value and key not in os.environ:
            os.environ[key] = value

    # ── 1. Create .env from .env.example if missing ───────────────────────────
    if not _ENV.exists():
        if _EXAMPLE.exists():
            shutil.copy(_EXAMPLE, _ENV)
            created_env = True
            print("\n✅ Created .env from .env.example")
        else:
            # No example either — create a minimal .env
            _ENV.write_text(
                "# Lumina secrets — generated automatically\n"
                "LUMINA_SESSION_SECRET=\n"
                "OPENAI_API_KEY=\n"
                "ANTHROPIC_API_KEY=\n"
                "ELEVENLABS_API_KEY=\n"
                "TELEGRAM_TOKEN=\n"
            )
            created_env = True
            print("\n✅ Created fresh .env (no .env.example found)")

    # ── 2. Load .env into os.environ ──────────────────────────────────────────
    _loaded = 0
    try:
        lines = _ENV.read_text(encoding="utf-8").splitlines()
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip('"').strip("'")  # strip surrounding quotes
            if key and val and key not in os.environ:
                os.environ[key] = val
                _loaded += 1
        if _loaded:
            print(f"✅ Loaded {_loaded} variable(s) from .env")
    except Exception as e:
        print(f"⚠️  Could not read .env: {e}")

    # ── 3. Generate LUMINA_SESSION_SECRET if missing ──────────────────────────
    if not os.environ.get("LUMINA_SESSION_SECRET"):
        secret = secrets.token_hex(32)
        os.environ["LUMINA_SESSION_SECRET"] = secret

        # Write it back into .env so it persists across restarts
        try:
            text = _ENV.read_text(encoding="utf-8")
            if "LUMINA_SESSION_SECRET=" in text:
                # Replace the blank line
                lines = text.splitlines()
                new_lines = []
                for line in lines:
                    if line.strip().startswith("LUMINA_SESSION_SECRET="):
                        new_lines.append(f"LUMINA_SESSION_SECRET={secret}")
                    else:
                        new_lines.append(line)
                _ENV.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
            else:
                # Append it
                with open(_ENV, "a", encoding="utf-8") as f:
                    f.write(f"\nLUMINA_SESSION_SECRET={secret}\n")

            print(f"🔐 Generated LUMINA_SESSION_SECRET and saved to .env")
        except Exception as e:
            print(f"⚠️  Could not save session secret to .env: {e}")
            print(f"   Set it manually: LUMINA_SESSION_SECRET={secret}")
    else:
        print("🔐 LUMINA_SESSION_SECRET already set")

    # ── 4. Platform-specific guidance (printed once on first run) ────────────
    if created_env:
        _os = platform.system()
        print("\n" + "─"*60)
        print("🛡️  SECURITY SETUP COMPLETE")
        print("─"*60)
        print(f"   OS detected : {_os}")
        print(f"   .env location: {_ENV}")
        print("")
        if _os == "Windows":
            print("   To start Lumina in future (Windows):")
            print("   > python app.py")
            print("   (env vars load automatically on startup now)")
            print("")
            print("   To add API keys later, edit .env in a text editor:")
            print(f"   > notepad {_ENV}")
        else:
            print("   To start Lumina in future (Linux/macOS):")
            print("   > python app.py")
            print("   (env vars load automatically on startup now)")
            print("")
            print("   To add API keys later, edit .env:")
            print(f"   > nano {_ENV}")
        print("")
        print("   API keys to add when needed:")
        print("   TELEGRAM_TOKEN      — for Telegram connector")
        print("   ELEVENLABS_API_KEY  — for premium TTS voice")
        print("   OPENAI_API_KEY      — if using OpenAI instead of LM Studio")
        print("─"*60 + "\n")

    # ── 5. Restrict .env file permissions on Linux/macOS ─────────────────────
    if platform.system() != "Windows":
        try:
            _ENV.chmod(0o600)   # owner read/write only
        except Exception:
            pass

_bootstrap_security()
