# Kodi — a multi-turn Groq chatbot

A programming-tutor chatbot ("Kodi") built on the Groq API with a Flask backend, a small web UI, a CLI, and real context-window management.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env        # then edit .env and paste your key from https://console.groq.com/keys
python test_connection.py   # sanity check: should print a short greeting

python app.py               # web UI at http://127.0.0.1:5000
# or
python cli.py               # terminal chat
```

## Requirement checklist

| Requirement | Where |
|---|---|
| Backend endpoint calling an LLM | `POST /api/chat` in `app.py` → `Chatbot.send` in `chatbot.py` |
| Per-conversation multi-turn history | `Conversation` objects keyed by `conversation_id` (in memory) |
| Context window management | Rolling summary + token-budget truncation (`chatbot.py`, top docstring) |
| System prompt with role/personality | `SYSTEM_PROMPT` in `config.py` (Kodi, a coding tutor) |
| Error handling | Timeouts, connection errors, rate limits, bad keys, empty replies mapped to friendly messages; failed turns are rolled back; UI shows a Retry button |
| Keys in env vars, not git | `.env` is git-ignored; `.env.example` is the template |
| Frontend / CLI | `static/index.html`, `cli.py` |

## How context management works

1. Each request sends: system prompt (+ running summary) + recent messages.
2. When a conversation exceeds `SUMMARIZE_TRIGGER` (14) messages, everything but the newest `MAX_RECENT_MESSAGES` (8) is summarised by `GROQ_SUMMARY_MODEL` and merged into the running summary, which is added to the system prompt.
3. As a safety net, the prompt is trimmed oldest-first to fit `MAX_CONTEXT_TOKENS` (token count estimated at ~4 chars/token). If the summariser call itself fails, the bot falls back to plain truncation instead of erroring.

All values are tunable through environment variables (see `.env.example`).

## API

```
POST /api/chat   {"message": "...", "conversation_id": "optional"}
  200 → {"reply": "...", "conversation_id": "...", "meta": {...}}
  4xx/5xx → {"error": "human-readable message"}
POST /api/reset  {"conversation_id": "..."}
GET  /api/health
```

## Tests (no API key needed)

```bash
python -m unittest discover tests
```

## Notes

- Conversations live in memory; restarting the server clears them. Swap `Chatbot._conversations` for Redis/SQLite for persistence.
- Never commit `.env`. If a key leaks, rotate it in the Groq console.
