import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
resp = client.models.generate_content(
    model=os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"),
    contents="Reply with exactly: OK"
)
print(resp.text)