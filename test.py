import google.generativeai as genai
import os

from dotenv import load_dotenv
load_dotenv(override=True)

print('AIzaSyDsSAYG-DqdhQ9Q7oV0tCe6Jh-00BA1bIc')
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
print(GEMINI_API_KEY)
print(f"GEMINI_API_KEY : {GEMINI_API_KEY}")

def initialize_gemini(api_key):
    """Initializes the Gemini API."""
    try:
        genai.configure(api_key=api_key)
        print(f'api_key:{api_key}')
        print("Gemini API initialized successfully.")
    except Exception as e:
        print(f"Gemini API initialization failed: {e}")
        exit()
initialize_gemini(GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-2.5-flash')
response = model.generate_content("你好")
print(response)