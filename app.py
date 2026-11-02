"""Flask backend: POST /api/chat, POST /api/reset, GET /api/health + static frontend."""
from flask import Flask, jsonify, request, send_from_directory

import config
from chatbot import Chatbot, ChatError

app = Flask(__name__, static_folder="static", static_url_path="/static")

_bot: Chatbot | None = None


def bot() -> Chatbot:
    global _bot
    if _bot is None:
        _bot = Chatbot()  # raises ChatError if the key is missing
    return _bot


@app.get("/")
def index():
    return send_from_directory("static", "index.html")


@app.get("/api/health")
def health():
    return jsonify(ok=True, model=config.MODEL, key_configured=bool(config.GROQ_API_KEY))


@app.post("/api/chat")
def chat():
    data = request.get_json(silent=True) or {}
    try:
        reply, convo = bot().send(data.get("conversation_id"), data.get("message", ""))
    except ChatError as e:
        return jsonify(error=e.message), e.status
    except Exception:  # never leak a stack trace or break the UI
        app.logger.exception("Unhandled error in /api/chat")
        return jsonify(error="Something went wrong on the server. Please try again."), 500
    return jsonify(
        reply=reply,
        conversation_id=convo.id,
        meta={"messages_in_window": len(convo.messages), "has_summary": bool(convo.summary)},
    )


@app.post("/api/reset")
def reset():
    data = request.get_json(silent=True) or {}
    try:
        if data.get("conversation_id"):
            bot().reset(data["conversation_id"])
    except ChatError:
        pass
    return jsonify(ok=True)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
