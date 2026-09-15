"""
managers/messaging_manager.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Bidirectional connector between Lumina and external messaging platforms.

Supported platforms
───────────────────
  Telegram   — python-telegram-bot  (polling, no public URL needed)
  WhatsApp   — Twilio WhatsApp API  (webhook, needs public URL or ngrok)

Message flow
────────────
  User → Platform → Webhook/Polling → receive_message()
       → Orchestrator → PersonaBridge → response text
       → send_message() → Platform → User

User identity
─────────────
  Each platform user is mapped to a Lumina internal user_id so that
  memory, relational history, and persona work across platforms.

  Mapping stored in:  data/messaging/user_map.json
  Name map stored in: data/messaging/name_map.json

WhatsApp setup (Twilio sandbox — free, 5 min)
─────────────────────────────────────────────
  1. Sign up at https://www.twilio.com/try-twilio
  2. Go to Messaging → Try it out → Send a WhatsApp message
  3. Sandbox number is shown (usually +14155238886)
  4. Your phone: WhatsApp the sandbox number with the join code shown
  5. In config.json set:
       WHATSAPP_TWILIO_SID   = "AC..."
       WHATSAPP_TWILIO_TOKEN = "..."
       WHATSAPP_FROM         = "whatsapp:+14155238886"
  6. Run brain.py — exposes POST /webhook/whatsapp
  7. Expose port with ngrok:  ngrok http 8765
  8. In Twilio Sandbox settings → "When a message comes in":
       https://YOUR-NGROK-URL/webhook/whatsapp

Telegram setup (5 min)
──────────────────────
  1. Open Telegram → search @BotFather → /newbot
  2. Copy the token into config.json:  TELEGRAM_TOKEN = "..."
  3. Run brain.py — Lumina will start polling automatically
  4. Find your bot in Telegram and start chatting
"""

import asyncio
import re
import json
import hmac
import hashlib
import logging
import urllib.parse
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

USER_MAP_PATH = Path("data/messaging/user_map.json")
NAME_MAP_PATH = Path("data/messaging/name_map.json")


class MessagingManager:
    """
    Bidirectional Telegram + WhatsApp connector.

    Parameters
    ----------
    orchestrator : AutonomousOrchestrator
    settings     : AppSettings
    """

    def __init__(self, orchestrator: Any, settings: Any):
        self._orch     = orchestrator
        self._cfg      = settings
        self._user_map: Dict[str, str] = {}   # "platform:ext_id" → lumina_user_id
        self._name_map: Dict[str, str] = {}   # lumina_user_id → display name
        self._pending:  Dict[str, asyncio.Queue] = {}  # user_id → response queue

        self._tg_app = None    # python-telegram-bot Application
        self._twilio = None    # Twilio Client

        self._load_maps()
        logger.info("📲 MessagingManager initialised")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  Lifecycle
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def start(self) -> None:
        await self._start_telegram()
        self._init_whatsapp()
        logger.info("📲 MessagingManager started")

    async def stop(self) -> None:
        if self._tg_app:
            try:
                await self._tg_app.stop()
                await self._tg_app.shutdown()
            except Exception as e:
                logger.debug(f"Telegram stop: {e}")
        logger.info("📲 MessagingManager stopped")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  Core message routing
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def receive_message(
        self,
        platform:         str,
        external_user_id: str,
        text:             str,
        display_name:     str = "",
    ) -> Optional[str]:
        """
        Called when a message arrives from any platform.
        Returns the response text (so webhook handlers can send it back inline).
        """
        lumina_user = self._resolve_user(platform, external_user_id, display_name)
        logger.info(f"📲 [{platform}] {display_name or external_user_id} → {lumina_user}: {text[:80]}")

        # ── Handle built-in commands ──────────────────────────────────────
        cmd_response = self._handle_command(text, lumina_user, display_name)
        if cmd_response is not None:
            return cmd_response

        # ── Route through orchestrator ────────────────────────────────────
        # Create a per-user queue to receive the response
        q: asyncio.Queue = asyncio.Queue()
        self._pending[lumina_user] = q

        self._orch.notify_external(platform, lumina_user, text)

        # Wait for response
        response = await self._wait_for_response(lumina_user, timeout=60.0)
        self._pending.pop(lumina_user, None)

        return response or "I'm thinking — give me a moment. 🤔"

    async def send_message(self, platform: str, external_user_id: str, text: str) -> None:
        """Dispatch a text message to the appropriate platform."""
        if not text:
            return
        if platform == "telegram":
            await self._send_telegram(external_user_id, text)
        elif platform == "whatsapp":
            await self._send_whatsapp(external_user_id, text)
        else:
            logger.warning(f"Unknown platform: {platform!r}")

    def deliver_response(self, lumina_user_id: str, text: str) -> None:
        """
        Called by the orchestrator execution layer when Lumina has finished
        generating a response for a messaging user.
        """
        q = self._pending.get(lumina_user_id)
        if q:
            q.put_nowait(text)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  Built-in commands (!register, !name, !forget, !status)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _handle_command(self, text: str, lumina_user: str, display_name: str) -> Optional[str]:
        """
        Returns a string response if the message is a command, else None.

        Commands
        ────────
        !register <your name>   — tell Lumina your name
        !name                   — ask what name Lumina knows you by
        !forget                 — remove your name mapping
        !status                 — brain health summary
        !help                   — list commands
        """
        t = text.strip()
        if not t.startswith("!"):
            return None

        parts = t.split(None, 1)
        cmd   = parts[0].lower()
        arg   = parts[1].strip() if len(parts) > 1 else ""

        if cmd == "!register":
            if not arg:
                return "Usage: !register <your name>\nExample: !register Alice"
            self._name_map[lumina_user] = arg
            self._save_maps()
            return (
                f"✅ Got it! I'll call you *{arg}* from now on.\n"
                f"Your Lumina ID: `{lumina_user}`\n\n"
                f"You can now chat normally — just send me a message!"
            )

        if cmd == "!name":
            name = self._name_map.get(lumina_user, display_name or "unknown")
            return f"I know you as: *{name}*\nYour Lumina ID: `{lumina_user}`"

        if cmd == "!forget":
            self._name_map.pop(lumina_user, None)
            self._save_maps()
            return "✅ I've cleared your name. Use !register <name> to set it again."

        if cmd == "!status":
            from core.state import state
            orch = getattr(state, "orchestrator", None)
            cycles = orch.status().get("cycle_count", "—") if orch else "—"
            return (
                f"🧠 Lumina brain status\n"
                f"  Ready: {getattr(state, 'ready', False)}\n"
                f"  Orch cycles: {cycles}\n"
                f"  Your ID: {lumina_user}\n"
                f"  Known as: {self._name_map.get(lumina_user, 'not set — use !register')}"
            )

        if cmd == "!help":
            return (
                "📋 *Lumina commands*\n\n"
                "  !register <name>  — tell me your name\n"
                "  !name             — what name I know you by\n"
                "  !forget           — clear your name\n"
                "  !status           — brain health\n"
                "  !help             — this message\n\n"
                "Or just chat normally — no commands needed!"
            )

        return None   # unknown command → pass to LLM

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  Twilio WhatsApp webhook  (called from brain.py FastAPI route)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def handle_whatsapp_webhook(
        self,
        form_data: dict,
        x_twilio_signature: str = "",
        raw_url: str = "",
    ) -> str:
        """
        Processes an inbound Twilio WhatsApp webhook POST.
        Returns a TwiML XML string to send back to Twilio.

        form_data keys Twilio sends:
          From        — "whatsapp:+1234567890"
          Body        — message text
          ProfileName — display name the user set in WhatsApp
        """
        # ── Optional signature validation ─────────────────────────────────
        secret = getattr(self._cfg, "WHATSAPP_WEBHOOK_SECRET", "")
        if secret and raw_url and x_twilio_signature:
            if not self._validate_twilio_signature(secret, raw_url, form_data, x_twilio_signature):
                logger.warning("⚠️  WhatsApp webhook: invalid Twilio signature — rejected")
                return self._twiml("")   # empty response, don't echo back

        from_raw     = form_data.get("From", "")
        body         = form_data.get("Body", "").strip()
        profile_name = form_data.get("ProfileName", "")

        # Strip "whatsapp:" prefix for our internal key
        from_number = from_raw.replace("whatsapp:", "")

        if not body:
            return self._twiml("")

        response = await self.receive_message(
            platform="whatsapp",
            external_user_id=from_number,
            text=body,
            display_name=profile_name,
        )

        return self._twiml(response or "")

    @staticmethod
    def _twiml(text: str) -> str:
        """Wrap response text in TwiML <Response><Message> envelope."""
        safe = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        return f'<?xml version="1.0" encoding="UTF-8"?><Response><Message>{safe}</Message></Response>'

    @staticmethod
    def _validate_twilio_signature(
        auth_token: str, url: str, params: dict, signature: str
    ) -> bool:
        """
        Validate Twilio's X-Twilio-Signature header.
        https://www.twilio.com/docs/usage/webhooks/webhooks-security
        """
        try:
            s = url + "".join(f"{k}{v}" for k, v in sorted(params.items()))
            expected = hmac.new(
                auth_token.encode(), s.encode(), hashlib.sha1
            ).digest()
            import base64
            return hmac.compare_digest(base64.b64encode(expected).decode(), signature)
        except Exception:
            return False

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  Telegram
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def _start_telegram(self) -> None:
        token = getattr(self._cfg, "TELEGRAM_TOKEN", "")
        if not token:
            logger.info("📲 Telegram: no token configured — skipped")
            return
        try:
            from telegram.ext import Application, MessageHandler, CommandHandler, filters
            from telegram import Update
            from telegram.ext import ContextTypes

            app = Application.builder().token(token).build()

            async def _on_message(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
                if not (update.message and update.message.text):
                    return
                chat_id = str(update.effective_chat.id)
                name    = update.effective_user.full_name if update.effective_user else ""
                text    = update.message.text

                response = await self.receive_message("telegram", chat_id, text, name)
                if response:
                    await self._send_telegram(chat_id, response)

            async def _on_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
                name = update.effective_user.first_name if update.effective_user else "there"
                await update.message.reply_text(
                    f"Hi {name}! 👋 I'm Lumina.\n\n"
                    f"Use *!register {name}* so I remember your name, "
                    f"or just start chatting!\n\nSend *!help* for all commands.",
                    parse_mode="Markdown",
                )

            app.add_handler(CommandHandler("start", _on_start))
            app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, _on_message))

            await app.initialize()
            await app.start()
            asyncio.create_task(app.updater.start_polling())
            self._tg_app = app
            logger.info("✅ Telegram connector started — polling")
        except ImportError:
            logger.warning("📲 Telegram: python-telegram-bot not installed.\n"
                           "   pip install python-telegram-bot --break-system-packages")
        except Exception as e:
            logger.error(f"Telegram start failed: {e}")

    async def _send_telegram(self, chat_id: str, text: str) -> None:
        if not self._tg_app:
            return
        # Telegram max message length is 4096 chars
        chunks = [text[i:i+4000] for i in range(0, len(text), 4000)]
        for chunk in chunks:
            try:
                await self._tg_app.bot.send_message(chat_id=int(chat_id), text=chunk)
            except Exception as e:
                logger.error(f"Telegram send failed: {e}")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  WhatsApp (Twilio)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _init_whatsapp(self) -> None:
        sid   = getattr(self._cfg, "WHATSAPP_TWILIO_SID",   "")
        token = getattr(self._cfg, "WHATSAPP_TWILIO_TOKEN", "")
        if not (sid and token):
            logger.info("📲 WhatsApp: no Twilio credentials — skipped\n"
                        "   Add WHATSAPP_TWILIO_SID / WHATSAPP_TWILIO_TOKEN to config.json")
            return
        try:
            from twilio.rest import Client
            self._twilio = Client(sid, token)
            logger.info("✅ WhatsApp (Twilio) connector initialised")
        except ImportError:
            logger.warning("📲 WhatsApp: twilio package not installed.\n"
                           "   pip install twilio --break-system-packages")
        except Exception as e:
            logger.error(f"WhatsApp init failed: {e}")

    async def _send_whatsapp(self, to_number: str, text: str) -> None:
        if not self._twilio:
            return
        from_number = getattr(self._cfg, "WHATSAPP_FROM", "whatsapp:+14155238886")
        if not to_number.startswith("whatsapp:"):
            to_number = f"whatsapp:{to_number}"
        # WhatsApp via Twilio: 1600 char limit per message
        chunks = [text[i:i+1500] for i in range(0, len(text), 1500)]
        for chunk in chunks:
            try:
                await asyncio.to_thread(
                    self._twilio.messages.create,
                    body=chunk,
                    from_=from_number,
                    to=to_number,
                )
            except Exception as e:
                logger.error(f"WhatsApp send failed: {e}")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  User identity resolution
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _resolve_user(self, platform: str, external_id: str, display_name: str = "") -> str:
        """
        Map platform:external_id → Lumina internal user_id.
        Creates a stable mapping on first contact and persists it.
        """
        safe_plat = re.sub(r"[^a-z0-9]", "", platform.lower())[:16]
        safe_ext  = re.sub(r"[^a-zA-Z0-9_\-]", "_", str(external_id))[:48]
        key       = f"{safe_plat}:{safe_ext}"

        if key not in self._user_map:
            # First contact — create stable ID
            lumina_id = f"msg_{safe_plat}_{safe_ext}"
            self._user_map[key] = lumina_id
            # Auto-set display name from platform profile if available
            if display_name and lumina_id not in self._name_map:
                self._name_map[lumina_id] = display_name
            self._save_maps()
            logger.info(f"📲 New {platform} user: {display_name or external_id} → {lumina_id}")

        return self._user_map[key]

    def get_user_name(self, lumina_user_id: str) -> str:
        """Return the display name Lumina uses for this user."""
        return self._name_map.get(lumina_user_id, lumina_user_id)

    def list_users(self) -> list:
        """Return all registered messaging users."""
        return [
            {
                "lumina_id":    lumina_id,
                "platform_key": platform_key,
                "name":         self._name_map.get(lumina_id, "—"),
            }
            for platform_key, lumina_id in self._user_map.items()
        ]

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  Response waiting
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def _wait_for_response(self, user_id: str, timeout: float = 60.0) -> Optional[str]:
        """
        Wait for the orchestrator to produce a response for this user.
        Uses a dedicated asyncio.Queue per user for clean fan-out.
        Falls back to polling the execution layer's persona_queue.
        """
        # Strategy 1: per-user queue (set up by receive_message)
        q = self._pending.get(user_id)
        if q:
            try:
                return await asyncio.wait_for(q.get(), timeout=timeout)
            except asyncio.TimeoutError:
                pass

        # Strategy 2: poll execution layer queue (legacy path)
        persona_queue = getattr(getattr(self._orch, "execution", None), "persona_queue", None)
        if persona_queue is None:
            return None

        deadline = asyncio.get_event_loop().time() + timeout
        held = []
        try:
            while asyncio.get_event_loop().time() < deadline:
                try:
                    item = await asyncio.wait_for(persona_queue.get(), timeout=2.0)
                    if isinstance(item, dict) and item.get("user_id") == user_id:
                        # Drain held items back
                        for h in held:
                            await persona_queue.put(h)
                        return item.get("text", "")
                    held.append(item)
                except asyncio.TimeoutError:
                    pass
        finally:
            for h in held:
                await persona_queue.put(h)

        return None

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  Persistence
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _load_maps(self) -> None:
        for path, attr in [(USER_MAP_PATH, "_user_map"), (NAME_MAP_PATH, "_name_map")]:
            try:
                if path.exists():
                    setattr(self, attr, json.loads(path.read_text(encoding="utf-8")))
            except Exception as e:
                logger.debug(f"Map load skipped ({path.name}): {e}")

    def _save_maps(self) -> None:
        for path, attr in [(USER_MAP_PATH, "_user_map"), (NAME_MAP_PATH, "_name_map")]:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(getattr(self, attr), indent=2), encoding="utf-8")
            except Exception as e:
                logger.debug(f"Map save failed ({path.name}): {e}")
