import os
from dotenv import load_dotenv
load_dotenv()
from groq import Groq

try:
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    models = client.models.list()
    print("Available Groq models:")
    for m in models.data[:10]:
        print(f"  - {m.id}")
except Exception as e:
    print(f"Error: {e}")
