"""LLM Manager - Ollama, LM Studio, OpenAI providers with ABC base."""
import logging
import json
import requests
import threading
from typing import Optional, List, Dict, Iterator
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

# Max conversation history entries kept in memory (each entry = 1 message).
# Older messages are dropped to keep requests small and fast.
_HISTORY_MAX = 20  # 10 user + 10 assistant turns


# ─────────────────────────────────────────────
#  Abstract base
# ─────────────────────────────────────────────
class BaseLLMProvider(ABC):
    def __init__(self, model: str, **kwargs):
        self.model = model
        self.extra = kwargs

    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        pass

    @abstractmethod
    def generate_stream(
        self, prompt: str, system_prompt: Optional[str] = None, **kwargs
    ) -> Iterator[str]:
        """Yield response tokens as they arrive."""
        pass

    def generate_with_history(
        self,
        prompt: str,
        history: List[Dict],
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """Build a full message list from history + current prompt and generate."""
        messages: List[Dict] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(history)
        messages.append({"role": "user", "content": prompt})
        return self._generate_messages(messages, **kwargs)

    def generate_with_history_stream(
        self,
        prompt: str,
        history: List[Dict],
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Iterator[str]:
        """Streaming version of generate_with_history.
        Auto-trims history to fit within the configured context window.
        """
        try:
            from managers.settings_manager import config as _cfg
            ctx_limit = max(getattr(_cfg, 'LLM_CTX', 4096), 512)
        except Exception:
            ctx_limit = 4096
        chars_budget = int((ctx_limit - 800) * 3.5)
        fixed_cost = (len(system_prompt) if system_prompt else 0) + len(prompt)
        trimmed = list(history)
        while trimmed and (fixed_cost + sum(len(m.get("content","")) for m in trimmed)) > chars_budget:
            trimmed = trimmed[2:]
        if len(trimmed) < len(history):
            import logging as _l
            _l.getLogger(__name__).debug(
                f"Context guard: dropped {len(history)-len(trimmed)} msgs to fit {ctx_limit}-token window")
        messages: List[Dict] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(trimmed)
        messages.append({"role": "user", "content": prompt})
        yield from self._generate_messages_stream(messages, **kwargs)

    @abstractmethod
    def _generate_messages(self, messages: List[Dict], **kwargs) -> str:
        """Send a pre-built messages list to the provider and return the reply."""
        pass

    @abstractmethod
    def _generate_messages_stream(self, messages: List[Dict], **kwargs) -> Iterator[str]:
        """Streaming version of _generate_messages."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        pass


# ─────────────────────────────────────────────
#  Providers
# ─────────────────────────────────────────────
class OllamaProvider(BaseLLMProvider):
    def __init__(self, model: str = "llama3", base_url: str = "http://localhost:11434", **kwargs):
        super().__init__(model, **kwargs)
        self.base_url = base_url.rstrip('/')

    def is_available(self) -> bool:
        try:
            return requests.get(f"{self.base_url}/api/tags", timeout=3).status_code == 200
        except Exception:
            return False

    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        messages: List[Dict] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return self._generate_messages(messages, **kwargs)

    def generate_stream(
        self, prompt: str, system_prompt: Optional[str] = None, **kwargs
    ) -> Iterator[str]:
        messages: List[Dict] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        yield from self._generate_messages_stream(messages, **kwargs)

    def _generate_messages(self, messages: List[Dict], **kwargs) -> str:
        try:
            r = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model":    kwargs.get("model", self.model),
                    "messages": messages,
                    "stream":   False,
                    "options": {
                        "temperature": kwargs.get("temperature", 0.7),
                        # Agent is instructed to reply in max 2 sentences —
                        # 100 tokens is ample and avoids wasted generation time.
                        "num_predict": kwargs.get("max_tokens", 600)
                    }
                },
                timeout=90
            )
            if r.status_code == 200:
                return r.json()["message"]["content"]
            return f"Error: Ollama status {r.status_code}"
        except Exception as e:
            logger.error(f"Ollama error: {e}")
            return f"Error: {e}"

    def _generate_messages_stream(self, messages: List[Dict], **kwargs) -> Iterator[str]:
        """Stream tokens from Ollama's /api/chat endpoint."""
        try:
            with requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model":    kwargs.get("model", self.model),
                    "messages": messages,
                    "stream":   True,
                    "options": {
                        "temperature": kwargs.get("temperature", 0.7),
                        "num_predict": kwargs.get("max_tokens", 600)
                    }
                },
                stream=True,
                timeout=90
            ) as r:
                if r.status_code != 200:
                    yield f"Error: Ollama status {r.status_code}"
                    return
                for raw_line in r.iter_lines():
                    if not raw_line:
                        continue
                    try:
                        chunk = json.loads(raw_line)
                        token = chunk.get("message", {}).get("content", "")
                        if token:
                            yield token
                        if chunk.get("done"):
                            break
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logger.error(f"Ollama stream error: {e}")
            yield f"Error: {e}"

    def generate_with_vision(
        self,
        prompt: str,
        image_base64: str,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Dict:
        """Generate response with vision model (e.g., llava)"""
        vision_model = model or "llava:latest"

        messages: List[Dict] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({
            "role": "user",
            "content": prompt,
            "images": [image_base64]
        })

        try:
            r = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": vision_model,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": kwargs.get("temperature", 0.7),
                        "num_predict": kwargs.get("max_tokens", 600)
                    }
                },
                timeout=120
            )
            if r.status_code == 200:
                return {"text": r.json()["message"]["content"], "model": vision_model, "success": True}
            return {"text": f"Error: Ollama status {r.status_code}", "success": False}
        except Exception as e:
            logger.error(f"Ollama vision error: {e}")
            return {"text": f"Error: {e}", "success": False}


class LMStudioProvider(BaseLLMProvider):
    def __init__(self, model: str = "local-model", base_url: str = "http://localhost:1234/v1", **kwargs):
        super().__init__(model, **kwargs)
        self.base_url = base_url.rstrip('/')

    def is_available(self) -> bool:
        try:
            return requests.get(f"{self.base_url}/models", timeout=3).status_code == 200
        except Exception:
            return False

    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        messages: List[Dict] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return self._generate_messages(messages, **kwargs)

    def generate_stream(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> Iterator[str]:
        messages: List[Dict] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        yield from self._generate_messages_stream(messages, **kwargs)

    def _generate_messages(self, messages: List[Dict], **kwargs) -> str:
        try:
            request_body = {
                "model":       kwargs.get("model", self.model),
                "messages":    messages,
                "temperature": kwargs.get("temperature", 0.7),
                "max_tokens":  kwargs.get("max_tokens", 600)
            }
            # Some reasoning-capable local models can stop after emitting only
            # a hidden reasoning channel. Structured background tasks need a
            # normal assistant content channel, so callers may opt out locally.
            if "reasoning_format" in kwargs:
                request_body["reasoning_format"] = kwargs["reasoning_format"]
            if kwargs.get("json_mode"):
                request_body["response_format"] = {"type": "json_object"}
            r = requests.post(
                f"{self.base_url}/chat/completions",
                json=request_body,
                timeout=60
            )
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"]
            return f"Error: LM Studio status {r.status_code}"
        except Exception as e:
            logger.error(f"LMStudio error: {e}")
            return f"Error: {e}"

    def _generate_messages_stream(self, messages: List[Dict], **kwargs) -> Iterator[str]:
        try:
            emitted = 0
            finish_reason = None
            with requests.post(
                f"{self.base_url}/chat/completions",
                json={
                    "model":       kwargs.get("model", self.model),
                    "messages":    messages,
                    "temperature": kwargs.get("temperature", 0.7),
                    "max_tokens":  kwargs.get("max_tokens", 600),
                    "stream":      True,
                },
                stream=True,
                timeout=60
            ) as r:
                if r.status_code != 200:
                    yield f"Error: LM Studio status {r.status_code}"
                    return
                for raw_line in r.iter_lines():
                    if not raw_line:
                        continue
                    line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
                    if line.startswith("data: "):
                        line = line[6:]
                    if line.strip() == "[DONE]":
                        break
                    try:
                        chunk = json.loads(line)
                        choice = chunk["choices"][0]
                        finish_reason = choice.get("finish_reason") or finish_reason
                        delta = choice.get("delta", {})
                        reasoning = (
                            delta.get("reasoning_content")
                            or delta.get("reasoning")
                            or choice.get("reasoning_content")
                            or ""
                        )
                        if reasoning:
                            yield {"type": "reasoning", "text": reasoning}
                        token = delta.get("content", "")
                        if token:
                            emitted += len(token)
                            yield token
                        # Preserve provider completion metadata so the chat
                        # bridge can distinguish a clean stop from max_tokens.
                        if finish_reason:
                            yield {"type": "finish", "reason": finish_reason}
                    except (json.JSONDecodeError, KeyError):
                        continue
            if emitted == 0:
                logger.warning(
                    "LMStudio stream completed without content (finish_reason=%s, messages=%d)",
                    finish_reason,
                    len(messages),
                )
        except Exception as e:
            logger.error(f"LMStudio stream error: {e}")
            yield f"Error: {e}"


    def generate_with_vision(
        self,
        prompt: str,
        image_base64: str,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Dict:
        """
        Send an image to a vision-capable model via LM Studio (OpenAI-compatible).

        Format: content array with text + image_url (data URI).
        Works with LLaVA, Qwen-VL, InternVL, Pixtral, and any OpenAI-vision-compatible
        model loaded in LM Studio.
        """
        vision_model = model or self.model
        messages: List[Dict] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_base64}",
                        "detail": "low",   # low = faster + cheaper; change to "high" for detail
                    }
                },
                {
                    "type": "text",
                    "text": prompt,
                }
            ]
        })
        try:
            r = requests.post(
                f"{self.base_url}/chat/completions",
                json={
                    "model":       vision_model,
                    "messages":    messages,
                    "temperature": kwargs.get("temperature", 0.5),
                    "max_tokens":  kwargs.get("max_tokens", 300),
                },
                timeout=120,
            )
            if r.status_code == 200:
                text = r.json()["choices"][0]["message"]["content"]
                return {"text": text, "model": vision_model, "success": True}
            return {"text": f"Error: LM Studio status {r.status_code}", "success": False}
        except Exception as e:
            logger.error(f"LMStudio vision error: {e}")
            return {"text": f"Error: {e}", "success": False}


class OpenAIProvider(BaseLLMProvider):
    def __init__(self, model: str = "gpt-3.5-turbo", api_key: Optional[str] = None, **kwargs):
        super().__init__(model, **kwargs)
        self.api_key  = api_key
        self.base_url = "https://api.openai.com/v1"

    def is_available(self) -> bool:
        return bool(self.api_key)

    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        messages: List[Dict] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return self._generate_messages(messages, **kwargs)

    def generate_stream(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> Iterator[str]:
        messages: List[Dict] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        yield from self._generate_messages_stream(messages, **kwargs)

    def _generate_messages(self, messages: List[Dict], **kwargs) -> str:
        if not self.api_key:
            return "Error: OpenAI API key not configured"
        try:
            r = requests.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model":       self.model,
                    "messages":    messages,
                    "temperature": kwargs.get("temperature", 0.7),
                    "max_tokens":  kwargs.get("max_tokens", 600)
                },
                timeout=60
            )
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"]
            return f"Error: OpenAI status {r.status_code}"
        except Exception as e:
            logger.error(f"OpenAI error: {e}")
            return f"Error: {e}"

    def _generate_messages_stream(self, messages: List[Dict], **kwargs) -> Iterator[str]:
        if not self.api_key:
            yield "Error: OpenAI API key not configured"
            return
        try:
            with requests.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model":       self.model,
                    "messages":    messages,
                    "temperature": kwargs.get("temperature", 0.7),
                    "max_tokens":  kwargs.get("max_tokens", 600),
                    "stream":      True,
                },
                stream=True,
                timeout=60
            ) as r:
                if r.status_code != 200:
                    yield f"Error: OpenAI status {r.status_code}"
                    return
                for raw_line in r.iter_lines():
                    if not raw_line:
                        continue
                    line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
                    if line.startswith("data: "):
                        line = line[6:]
                    if line.strip() == "[DONE]":
                        break
                    try:
                        chunk = json.loads(line)
                        token = chunk["choices"][0].get("delta", {}).get("content", "")
                        if token:
                            yield token
                    except (json.JSONDecodeError, KeyError):
                        continue
        except Exception as e:
            logger.error(f"OpenAI stream error: {e}")
            yield f"Error: {e}"


# ─────────────────────────────────────────────
#  Manager wrapper
# ─────────────────────────────────────────────
class LLMManager:
    def __init__(self, provider: BaseLLMProvider, text_model: str = ""):
        self.provider    = provider
        # text_model: lighter model used for text-only turns.
        # Falls back to provider.model when empty.
        # Embedding models can be configured separately for FAISS/Cognee, but
        # must never receive conversational chat requests. Older settings
        # sometimes copied the embedding model into TEXT_MODEL, which makes
        # LM Studio appear to hang or return no assistant content.
        embedding_hint = (text_model or '').lower()
        if any(marker in embedding_hint for marker in ('embedding', 'embed-', 'nomic-embed')):
            logger.warning("TEXT_MODEL=%r looks like an embedding model; using the chat model instead.", text_model)
            self.text_model = provider.model
        else:
            self.text_model = text_model or provider.model
        self.history: List[Dict] = []
        # LM Studio/Ollama may serialize inference internally.  Keep shared
        # provider calls coordinated and give an interactive turn priority.
        self._provider_lock = threading.Lock()
        self._chat_active = threading.Event()

        # Attach session manager for persistence + compression
        from managers.session_manager import get_session_manager
        self._session = get_session_manager()

    def begin_interactive_turn(self) -> None:
        """Reserve the LLM for a user turn, including prompt preparation."""
        self._chat_active.set()

    def end_interactive_turn(self) -> None:
        """Release the reservation after the complete user turn finishes."""
        self._chat_active.clear()

    def _trim_history(self):
        """
        Notify session manager of new exchange (triggers save + compress).
        No longer silently drops turns — session_manager compresses instead.
        The raw history is kept at MAX_RAW_TURNS by session_manager._compress().
        """
        self._session.on_exchange_complete(self.history)
        # Safety cap: if session manager hasn't trimmed yet, apply hard limit
        if len(self.history) > _HISTORY_MAX:
            self.history = self.history[-_HISTORY_MAX:]

    def record_exchange(self, prompt: str, response: str) -> None:
        """Record a locally resolved turn without invoking the provider."""
        self.history.extend([
            {"role": "user", "content": str(prompt or "")},
            {"role": "assistant", "content": str(response or "")},
        ])
        self._trim_history()

    def generate_bare(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> str:
        """Call the LLM directly — NO history read, NO history write.

        Used by Lumina's internal cognitive tasks (dream cycles, learning cycles,
        emotional analysis, identity reflection) so they never pollute the user's
        visible chat history.  Temperature and max_tokens are fully forwarded.
        Default max_tokens=1000 (vs the 100-token robot-agent default).
        """
        return self.generate_bare_result(
            prompt, system_prompt=system_prompt, **kwargs
        )["text"]

    def generate_bare_result(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, str]:
        """Run background inference and preserve why no text was returned.

        ``generate_bare`` historically represented both provider failures and
        deliberate contention skips as an empty string. Cognitive planners
        need that distinction so an occupied camera/chat model does not count
        as failed reasoning.
        """
        kwargs.setdefault("max_tokens", 1000)
        messages: List[Dict] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        # Internal cognitive work is history-free and may use the optional
        # lighter conversational model. Embedding models are rejected by the
        # constructor and fall back to the main chat model.
        kwargs.setdefault("model", self.text_model)
        # Reasoning-capable local models can spend a short autonomous call in
        # the hidden reasoning channel and return an empty final content field.
        # Background cognition needs a usable result, not hidden reasoning;
        # LM Studio supports this explicit opt-out and other providers ignore
        # the provider-specific kwarg in their existing adapters.
        kwargs.setdefault("reasoning_format", "none")
        # Proactive cognition must never compete with an active user turn.
        # Skipping is preferable to delaying the chat or filling its model
        # queue with an obsolete background thought.
        if self._chat_active.is_set():
            logger.debug("Deferring background LLM task during an interactive turn.")
            return {
                "text": "",
                "status": "deferred",
                "reason": "interactive turn active",
            }
        if not self._provider_lock.acquire(False):
            logger.debug("Deferring background LLM task while the provider is busy.")
            return {
                "text": "",
                "status": "deferred",
                "reason": "provider busy",
            }
        try:
            text = self.provider._generate_messages(messages, **kwargs) or ""
            stripped = str(text).strip()
            if not stripped:
                return {
                    "text": "",
                    "status": "empty",
                    "reason": "provider returned no assistant content",
                }
            if stripped.lower().startswith("error"):
                return {
                    "text": str(text),
                    "status": "error",
                    "reason": stripped[:160],
                }
            return {"text": str(text), "status": "ok", "reason": ""}
        finally:
            self._provider_lock.release()

    def generate_interactive_analysis(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> str:
        """Run a bounded, history-free analysis inside the active user turn.

        Unlike ``generate_bare``, this path is intentionally allowed while the
        interactive marker is set. It acquires the same provider lock before
        the visible stream starts, so background work remains excluded and no
        two provider calls can overlap.
        """
        kwargs.setdefault("max_tokens", 480)
        kwargs.setdefault("temperature", 0.1)
        kwargs.setdefault("model", self.text_model)
        messages: List[Dict] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        with self._provider_lock:
            return self.provider._generate_messages(messages, **kwargs)

    def recover_final_response(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> str:
        """Regenerate a missing final channel after a reasoning-only stream."""
        kwargs.setdefault("max_tokens", 600)
        kwargs.setdefault("temperature", 0.45)
        kwargs.setdefault("model", self.text_model)
        prior_history = self.history
        if (
            len(prior_history) >= 2
            and prior_history[-2].get("role") == "user"
            and prior_history[-1].get("role") == "assistant"
        ):
            prior_history = prior_history[:-2]
        recovery_prompt = (
            prompt
            + "\n\nWrite the actual answer to the user now. Return only the "
              "user-facing answer. Do not output planning notes, self-correction, "
              "a final check, private reasoning, or think tags."
        )
        with self._provider_lock:
            response = self.provider.generate_with_history(
                recovery_prompt, prior_history, system_prompt, **kwargs
            )
        if self.history and self.history[-1].get("role") == "assistant":
            self.history[-1]["content"] = response
        return response

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        use_vision_model: bool = False,
        **kwargs
    ) -> str:
        """Generate a full response (non-streaming).

        When use_vision_model=False (default) the lighter TEXT_MODEL is used.
        Pass use_vision_model=True for turns where LLaVA context is needed.
        """
        model = self.provider.model if use_vision_model else self.text_model
        self._trim_history()
        with self._provider_lock:
            response = self.provider.generate_with_history(
                prompt, self.history, system_prompt, model=model, **kwargs
            )
        self.history.extend([
            {"role": "user",      "content": prompt},
            {"role": "assistant", "content": response},
        ])
        return response

    def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        use_vision_model: bool = False,
        **kwargs
    ) -> Iterator[str]:
        """Stream response tokens.  Accumulates them for history on completion.

        When use_vision_model=False the lighter TEXT_MODEL is used so plain
        text turns never wait for LLaVA's heavier inference stack.
        """
        model = self.provider.model if use_vision_model else self.text_model
        self._trim_history()
        accumulated = []
        owns_chat_marker = not self._chat_active.is_set()
        if owns_chat_marker:
            self._chat_active.set()
        try:
            for token in self.provider.generate_with_history_stream(
                prompt, self.history, system_prompt, model=model, **kwargs
            ):
                if isinstance(token, dict):
                    yield token
                    continue
                accumulated.append(token)
                yield token
        finally:
            if owns_chat_marker:
                self._chat_active.clear()
        full_response = "".join(accumulated)
        # Native reasoning events are excluded above. Remove tagged private
        # channels from the persisted fallback as well.
        import re
        full_response = re.sub(
            r"<think>[\s\S]*?(?:</think>|$)|<analysis>[\s\S]*?(?:</analysis>|$)",
            "", full_response, flags=re.IGNORECASE,
        ).strip()
        self.history.extend([
            {"role": "user",      "content": prompt},
            {"role": "assistant", "content": full_response},
        ])

    def generate_with_vision(
        self,
        prompt: str,
        image_base64: str,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Dict:
        """Generate response with vision support (if provider supports it)."""
        if hasattr(self.provider, "generate_with_vision"):
            # Vision perception is a background task.  It must never queue
            # ahead of an interactive chat turn or hold the provider lock
            # while the UI is waiting for speech/text.
            if self._chat_active.is_set() or not self._provider_lock.acquire(False):
                logger.debug("Skipping background vision call while the interactive chat is active.")
                return {"text": "", "success": False, "skipped": True}
            try:
                return self.provider.generate_with_vision(
                    prompt, image_base64, model, system_prompt, **kwargs
                )
            finally:
                self._provider_lock.release()
        return {"text": "Vision not supported by current provider", "success": False}

    def is_available(self) -> bool:
        return self.provider.is_available()

    def clear_history(self):
        self.history.clear()


# ─────────────────────────────────────────────
#  Factory
# ─────────────────────────────────────────────
PROVIDER_MAP = {
    "ollama":   OllamaProvider,
    "lmstudio": LMStudioProvider,
    "openai":   OpenAIProvider,
}


def create_llm_manager(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    **kwargs
) -> LLMManager:
    from managers.settings_manager import config
    provider   = provider or config.LLM_PROVIDER
    model      = model    or config.LLM_MODEL
    text_model = getattr(config, "TEXT_MODEL", "") or model

    if provider not in PROVIDER_MAP:
        raise ValueError(f"Unknown LLM provider: '{provider}'. Valid: {list(PROVIDER_MAP)}")

    if provider in ("ollama", "lmstudio"):
        kwargs.setdefault("base_url", config.LLM_BASE_URL)
    elif provider == "openai":
        kwargs.setdefault("api_key", config.OPENAI_API_KEY)

    instance = PROVIDER_MAP[provider](model=model, **kwargs)
    manager = LLMManager(instance, text_model=text_model)
    logger.info(
        f"LLM manager created: {provider}/{model} "
        f"(effective text model: {manager.text_model})"
    )
    return manager


logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  Abstract base
