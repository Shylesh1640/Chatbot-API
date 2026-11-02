"""Core chatbot logic: per-conversation history + context window management.

Context strategy (two layers):

1. Rolling summary (primary)
   Once a conversation holds more than SUMMARIZE_TRIGGER messages, everything except
   the newest MAX_RECENT_MESSAGES is folded into a running summary by a cheap LLM call.
   The summary is prepended to the system prompt on every later request, so facts such
   as the user's name or goals survive long after the raw messages are gone.

2. Token-budget truncation (safety net)
   Before every request the prompt size is estimated (~4 chars/token). If it still exceeds
   MAX_CONTEXT_TOKENS (huge messages, or the summariser failed), the oldest messages are
   dropped until it fits. The newest user message is always kept.
"""
from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field

import groq
from groq import Groq

import config


class ChatError(Exception):
    """An error with a message that is safe to show directly to the end user."""

    def __init__(self, message: str, status: int = 502):
        super().__init__(message)
        self.message = message
        self.status = status


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


@dataclass
class Conversation:
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    messages: list[dict] = field(default_factory=list)  # {"role": "user"|"assistant", "content": str}
    summary: str = ""
    lock: threading.Lock = field(default_factory=threading.Lock)


class Chatbot:
    def __init__(self, client: Groq | None = None):
        if client is None:
            if not config.GROQ_API_KEY:
                raise ChatError(
                    "GROQ_API_KEY is not set. Copy .env.example to .env and add your key.", 500
                )
            client = Groq(
                api_key=config.GROQ_API_KEY,
                timeout=config.REQUEST_TIMEOUT,
                max_retries=config.MAX_RETRIES,
            )
        self.client = client
        self._conversations: dict[str, Conversation] = {}
        self._registry_lock = threading.Lock()

    # ---------- conversation store ----------
    def get(self, conversation_id: str | None) -> Conversation:
        with self._registry_lock:
            if conversation_id and conversation_id in self._conversations:
                return self._conversations[conversation_id]
            convo = Conversation()
            self._conversations[convo.id] = convo
            return convo

    def reset(self, conversation_id: str) -> None:
        with self._registry_lock:
            self._conversations.pop(conversation_id, None)

    # ---------- public API ----------
    def send(self, conversation_id: str | None, user_text: str) -> tuple[str, Conversation]:
        user_text = (user_text or "").strip()
        if not user_text:
            raise ChatError("Please type a message first.", 400)
        if len(user_text) > config.MAX_USER_CHARS:
            raise ChatError(
                f"That message is too long ({len(user_text)} characters). "
                f"The limit is {config.MAX_USER_CHARS}.",
                400,
            )

        convo = self.get(conversation_id)
        with convo.lock:  # one in-flight request per conversation keeps history consistent
            convo.messages.append({"role": "user", "content": user_text})
            try:
                self._compact(convo)
                prompt = self._build_prompt(convo)
                reply = self._complete(config.MODEL, prompt, temperature=0.6, max_tokens=700)
            except ChatError:
                convo.messages.pop()  # roll back so a failed turn doesn't pollute history
                raise
            convo.messages.append({"role": "assistant", "content": reply})
            return reply, convo

    # ---------- context management ----------
    def _system_message(self, convo: Conversation) -> dict:
        content = config.SYSTEM_PROMPT
        if convo.summary:
            content += (
                "\n\nSummary of the earlier part of this conversation "
                "(treat as trusted memory):\n" + convo.summary
            )
        return {"role": "system", "content": content}

    def _compact(self, convo: Conversation) -> None:
        """Fold older messages into the running summary once history gets long."""
        if len(convo.messages) <= config.SUMMARIZE_TRIGGER:
            return
        keep = config.MAX_RECENT_MESSAGES
        old, recent = convo.messages[:-keep], convo.messages[-keep:]
        # Keep the window starting on a user turn so roles alternate cleanly.
        while recent and recent[0]["role"] != "user":
            old.append(recent.pop(0))
        try:
            convo.summary = self._summarise(convo.summary, old)
        except ChatError:
            # Summariser failed: fall through to plain truncation (old messages are dropped
            # but the previous summary is kept). Better to forget than to crash the chat.
            pass
        convo.messages = recent

    def _summarise(self, previous: str, old: list[dict]) -> str:
        transcript = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in old)
        prompt = [
            {
                "role": "system",
                "content": (
                    "You compress chat history into a short memory note. Keep: the user's name, "
                    "goals, skill level, code/errors discussed, decisions and open questions. "
                    "Drop pleasantries. Max 150 words. Output only the note."
                ),
            },
            {
                "role": "user",
                "content": f"Existing note:\n{previous or '(none)'}\n\nNew messages:\n{transcript}\n\nUpdated note:",
            },
        ]
        return self._complete(config.SUMMARY_MODEL, prompt, temperature=0.2, max_tokens=300)

    def _build_prompt(self, convo: Conversation) -> list[dict]:
        system = self._system_message(convo)
        history = list(convo.messages)
        budget = config.MAX_CONTEXT_TOKENS - estimate_tokens(system["content"])

        def cost(msgs):
            return sum(estimate_tokens(m["content"]) + 4 for m in msgs)

        # Drop oldest messages until under budget, but always keep the latest user message.
        while len(history) > 1 and cost(history) > budget:
            history.pop(0)
        while len(history) > 1 and history[0]["role"] != "user":
            history.pop(0)
        return [system, *history]

    # ---------- LLM call + error mapping ----------
    def _complete(self, model: str, messages: list[dict], temperature: float, max_tokens: int) -> str:
        try:
            resp = self.client.chat.completions.create(
                model=model, messages=messages, temperature=temperature, max_tokens=max_tokens
            )
            text = (resp.choices[0].message.content or "").strip()
        except groq.APITimeoutError:
            raise ChatError("The AI service took too long to respond. Please try again.", 504)
        except groq.APIConnectionError:
            raise ChatError("Can't reach the AI service. Check your internet connection and retry.", 503)
        except groq.RateLimitError:
            raise ChatError("Rate limit reached. Wait a few seconds and send your message again.", 429)
        except groq.AuthenticationError:
            raise ChatError("The API key was rejected. Check GROQ_API_KEY in your .env file.", 500)
        except groq.BadRequestError as e:
            raise ChatError(f"The AI service rejected this request: {getattr(e, 'message', e)}", 400)
        except groq.APIStatusError as e:
            raise ChatError(f"The AI service returned an error (HTTP {e.status_code}). Try again shortly.", 502)
        except (IndexError, AttributeError):
            raise ChatError("The AI service returned an unexpected response. Please retry.", 502)
        if not text:
            raise ChatError("The AI service returned an empty reply. Please retry.", 502)
        return text
