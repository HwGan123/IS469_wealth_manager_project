from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
print(f"API Key loaded: {api_key is not None}")
if api_key:
    print(f"Key starts with: {api_key[:10]}...")
else:
    print("API Key NOT found!")