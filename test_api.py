import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

print("✅ Available EMBEDDING models:")
for model in client.models.list():
    if "embedContent" in model.supported_actions:
        print(" →", model.name)

print("\n✅ Available LLM models:")
for model in client.models.list():
    if "generateContent" in model.supported_actions:
        print(" →", model.name)