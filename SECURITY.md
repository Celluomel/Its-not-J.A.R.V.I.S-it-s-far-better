# Lumina Security Hardening Guide

## What's protected by default (after this update)

| Vector | Protection |
|---|---|
| Prompt injection | 20+ regex patterns block role-switch, system-prompt leakage, DAN attempts |
| Rate limiting | 20 requests / 60s per user — prevents abuse and runaway loops |
| Network exposure | Bound to `127.0.0.1` only — never reachable from LAN or internet |
| Path traversal | user_id sanitised before any file path use |
| Session hijacking | NiceGUI session secret via env var |
| Hardcoded secrets | Startup audit warns if API keys are in config.json |
| LLM response leakage | System prompt markers stripped from all responses |
| Messaging user_id | External platform IDs sanitised before file system use |

---

## Recommended: move secrets out of config.json

Never store API keys in `config.json` — it is world-readable on most systems.

**Step 1** — create a `.env` file (never commit this):
```bash
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
ELEVENLABS_API_KEY=...
BRAVE_SEARCH_KEY=...
SERPAPI_KEY=...
TELEGRAM_TOKEN=...
LUMINA_SESSION_SECRET=<random-32-char-string>
```

**Step 2** — load it before starting Lumina:
```bash
# Linux / macOS
export $(cat .env | xargs) && python app.py

# Or with python-dotenv installed:
pip install python-dotenv --break-system-packages
```

**Step 3** — remove keys from config.json, leave them blank:
```json
{
  "OPENAI_API_KEY": "",
  "ANTHROPIC_API_KEY": "",
  "ELEVENLABS_API_KEY": ""
}
```

The app reads env vars first (already implemented in settings_manager.py).

---

## File permissions

Restrict access to sensitive data directories:
```bash
chmod 700 data/
chmod 600 config.json
chmod 600 data/security/audit.log
```

On Windows, right-click → Properties → Security → restrict to your user only.

---

## Session secret

Set a strong random session secret so NiceGUI sessions can't be forged:
```bash
export LUMINA_SESSION_SECRET=$(python3 -c "import secrets; print(secrets.token_hex(32))")
```

Add this to your `.env` file so it persists across restarts.

---

## Network access

Lumina binds to `127.0.0.1` by default — only accessible from the local machine.

If you need LAN access (e.g. from another device on your network):
```python
# In app.py — change ONLY if you understand the risk:
host='0.0.0.0'   # accessible from LAN
```

If you expose to LAN, add a reverse proxy with authentication (nginx + basic auth, 
or Tailscale for zero-config private networking).

**Never expose port 8080 to the internet directly.**

---

## Telegram / WhatsApp security

- Telegram bot token: treat like a password — store in env var only
- Restrict your bot to specific chat IDs in `messaging_manager.py`:

```python
ALLOWED_TELEGRAM_CHAT_IDS = {123456789, 987654321}  # your chat IDs

async def _on_message(update, context):
    if update.effective_chat.id not in ALLOWED_TELEGRAM_CHAT_IDS:
        return  # silently ignore unknown chats
```

- Twilio webhook: verify the `X-Twilio-Signature` header on incoming requests

---

## Audit log

Security events are logged to `data/security/audit.log`:
```
2025-03-06T14:22:01  INJECTION_ATTEMPT         user_123              pattern: 'ignore previous instructions'
2025-03-06T14:25:33  RATE_LIMITED              user_123              too many requests
2025-03-06T14:30:00  HARDCODED_SECRET          config                OPENAI_API_KEY
```

Review this log periodically. The orchestrator dashboard shows the last 8 events in real time.

---

## LLM-specific risks

**Prompt injection via memory** — if Lumina stores malicious content in memory 
(e.g. from web research) and later injects it into the system prompt, it could 
affect behaviour. Mitigation: the sanitiser runs on user input, but web-scraped 
content bypasses it by design. Monitor web research results in the Research page.

**Model exfiltration** — locally-run models (LM Studio) have no exfiltration risk. 
Cloud APIs (OpenAI, Anthropic) send conversation history to external servers. 
Use local models for sensitive conversations.

**Jailbreak via TTS/STT** — voice input goes through Whisper before the sanitiser.
The sanitiser runs on the transcribed text so injection attempts via voice are caught.
