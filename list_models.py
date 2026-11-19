"""Print the Gemini models your API key can use for text generation.

Run:  python list_models.py
"""
import os

from dotenv import load_dotenv
from google import genai

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY", "").strip()
if not api_key:
    raise SystemExit("GEMINI_API_KEY is missing from .env")

client = genai.Client(api_key=api_key)
rows = []
for model in client.models.list():
    actions = getattr(model, "supported_actions", None) or []
    name = (model.name or "").removeprefix("models/")
    if "generateContent" in actions and name.startswith("gemini"):
        rows.append((name, model.display_name or ""))

for name, display in sorted(rows):
    print(f"{name:<45} {display}")
print(f"\n{len(rows)} models support generateContent.")
