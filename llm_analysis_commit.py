import requests
import pandas as pd
from datetime import datetime
import google.generativeai as genai
import json
import os
import time

# --- Constants ---
BASE_URL = "https://api.github.com"
CACHE_FILE = "llm_analysis_cache.json"

from dotenv import load_dotenv
load_dotenv(override=True)

# --- API Keys and Headers ---
TOKEN = os.getenv("GITHUB_TOKEN")
HEADERS = {"Authorization": f"token {TOKEN}"} if TOKEN else {}
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
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

# --- Cache Management ---
def load_cache():
    """Loads the analysis cache from a JSON file."""
    if not os.path.exists(CACHE_FILE):
        return {}
    try:
        with open(CACHE_FILE, 'r', encoding='utf-8') as f:
            print(f"成功載入快取:{CACHE_FILE}")
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}

def save_cache(cache_data):
    """Saves the analysis cache to a JSON file."""
    try:
        with open(CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=4)
            print(f"成功保存快取:{cache_data}")
    except IOError as e:
        print(f"Error saving cache: {e}")

# --- LLM Analysis with Cache ---
def get_llm_analysis(commit_message, cache, max_retries=3, initial_backoff=5):
    """
    Analyzes a commit message using Gemini LLM, with caching and retry mechanisms.
    Takes the cache dictionary as an argument.
    """
    if commit_message in cache:
        return cache[commit_message]

    retries = 0
    backoff_time = initial_backoff
    max_retries = 5
    while retries < max_retries:
        try:
            initialize_gemini(GEMINI_API_KEY)
            model = genai.GenerativeModel('gemini-2.5-flash')
            prompt = f"""
            請分析以下加密貨幣專案的 Git Commit 訊息，並提供簡潔的總結分析，
            以及對該加密貨幣幣價的潛在影響（上漲、下跌、無明顯影響）。
            請以繁體中文回答，並以以下 JSON 格式輸出：

            ```json
            {{
              "LLM 總結分析": "Commit 訊息的總結分析。",
              "對幣價的影響": "對幣價的潛在影響 (上漲/下跌/無明顯影響)。"
            }}
            ```

            Commit 訊息：
            {commit_message}
            """
            print("調用LLM API前先延遲1秒")
            print(f"開始分析commit message:{str(commit_message)[:50]}")
            time.sleep(1)
            response = model.generate_content(prompt)
            print(f"response:{response}")
            json_str = response.text.replace("```json", "").replace("```", "").strip()
            analysis_result = json.loads(json_str)
            print(f"分析結果:{str(analysis_result)[:50]}")
            
            cache[commit_message] = analysis_result
            return analysis_result,cache

        except Exception as e:
            if "429" in str(e) or "ResourceExhausted" in str(e):
                retries += 1
                if retries < max_retries:
                    time.sleep(backoff_time)
                    backoff_time *= 2
                else:
                    return {"LLM 總結分析": "分析失敗 (速率限制)", "對幣價的影響": "未知"}
            else:
                return {"LLM 總結分析": "分析失敗 (未知錯誤)", "對幣價的影響": "未知"}

    return {"LLM 總結分析": "分析失敗 (最終)", "對幣價的影響": "未知"}

# --- Standalone Execution Logic (for testing) ---

def get_latest_commits(org, repo, num_commits=10):
    """Fetches the latest commits from a specific GitHub repository."""
    commits_endpoint = f"/repos/{org}/{repo}/commits"
    params = {"per_page": num_commits}
    try:
        response = requests.get(f"{BASE_URL}{commits_endpoint}", headers=HEADERS, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching from GitHub API: {e}")
        return None

def initialize_gemini(api_key):
    """Initializes the Gemini API."""
    try:
        genai.configure(api_key=api_key)
        print(f'api_key:{api_key}')
        print("Gemini API initialized successfully.")
    except Exception as e:
        print(f"Gemini API initialization failed: {e}")
        exit()

if __name__ == "__main__":
    if not GEMINI_API_KEY or "YOUR_DEFAULT_TOKEN" in GEMINI_API_KEY:
        print("Error: Gemini API key not found or is a default value.")
        exit()
    
    initialize_gemini(GEMINI_API_KEY)
    
    org = "ethereum"
    repo = "go-ethereum"
    print(f"Fetching latest 10 commits from {org}/{repo} for testing...")
    commits = get_latest_commits(org, repo, num_commits=10)

    if commits:
        for commit in commits:
            message = commit["commit"]["message"]
            analysis = get_llm_analysis(message)
            print(f"--- Result for commit: {message[:30]}... ---")
            print(analysis)
            print("-------------------------------------\n")
            time.sleep(1)
    else:
        print("Could not fetch commits.")
