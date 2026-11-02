"""Step 2 of the guide: one message in, one reply out. Confirms your key works."""
import os

from dotenv import load_dotenv
from groq import Groq

load_dotenv()
key = os.getenv("GROQ_API_KEY")
if not key:
    raise SystemExit("GROQ_API_KEY missing. Copy .env.example to .env and set it.")

resp = Groq(api_key=key).chat.completions.create(
    model=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
    messages=[{"role": "user", "content": "Say hello in five words."}],
)
print(resp.choices[0].message.content)
