"""Terminal interface. Run: python cli.py   (commands: /reset, /quit)"""
from chatbot import Chatbot, ChatError


def main():
    try:
        bot = Chatbot()
    except ChatError as e:
        print(f"Setup error: {e.message}")
        return
    convo_id = None
    print("Kodi — programming tutor.  /reset starts over, /quit exits.\n")
    while True:
        try:
            text = input("you > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if text in ("/quit", "/exit"):
            break
        if text == "/reset":
            if convo_id:
                bot.reset(convo_id)
            convo_id = None
            print("(conversation cleared)\n")
            continue
        try:
            reply, convo = bot.send(convo_id, text)
            convo_id = convo.id
            print(f"\nkodi > {reply}\n")
        except ChatError as e:
            print(f"\n[!] {e.message}\n")


if __name__ == "__main__":
    main()
