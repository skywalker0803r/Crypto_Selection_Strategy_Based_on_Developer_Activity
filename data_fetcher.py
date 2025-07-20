import requests
import pandas as pd
from datetime import datetime, timedelta
import time

def get_coingecko_id_list(top_n, currency, predefined_cryptos):
    """
    Fetches a list of top N cryptocurrency IDs by market capitalization from CoinGecko API,
    excluding those already present in PREDEFINED_CRYPTOS.
    """
    print(f"--- Fetching top {top_n} cryptocurrency IDs from CoinGecko (excluding predefined) ---")
    url = f"https://api.coingecko.com/api/v3/coins/markets"
    params = {
        "vs_currency": currency,
        "order": "market_cap_desc",
        "per_page": min(top_n + len(predefined_cryptos) * 2, 250),
        "page": 1
    }
    print(f"DEBUG(CoinGecko API): Request URL: {url}")
    print(f"DEBUG(CoinGecko API): Request Params: {params}")

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        predefined_ids = set(predefined_cryptos.keys())
        filtered_ids = []
        for coin in data:
            if coin['id'] not in predefined_ids:
                filtered_ids.append(coin['id'])
            if len(filtered_ids) >= top_n:
                break
        
        filtered_ids.sort()
        print(f"Successfully fetched {len(filtered_ids)} CoinGecko IDs (top {top_n} excluding predefined).")
        return filtered_ids
    except requests.exceptions.RequestException as e:
        error_message = f"Error: Failed to fetch CoinGecko ID list: {e}"
        if 'response' in locals() and hasattr(response, 'text'):
            error_message += f"\nResponse content: {response.text}"
        print(error_message)
        return []

def get_crypto_prices(crypto_id, currency, start_date, end_date):
    print(f"\n--- Starting to fetch {crypto_id.upper()} price data ({currency.upper()}) ---")
    start_timestamp = int(start_date.timestamp())
    end_timestamp = int(end_date.timestamp())
    url = f"https://api.coingecko.com/api/v3/coins/{crypto_id}/market_chart/range?vs_currency={currency}&from={start_timestamp}&to={end_timestamp}"
    print(f"DEBUG(Price API): Request URL: {url}")

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        if not data or 'prices' not in data or not data['prices']:
            print(f"Warning: No price data fetched for {crypto_id.upper()} from CoinGecko. Check CoinGecko ID or date range.")
            return pd.Series(dtype='float64')

        prices = []
        for price_data in data['prices']:
            timestamp, price = price_data
            prices.append({'date': datetime.fromtimestamp(timestamp / 1000), 'price': price})

        df = pd.DataFrame(prices)
        df['date'] = pd.to_datetime(df['date']).dt.floor('D')
        
        # Group by date and calculate the mean price to handle duplicates
        daily_prices = df.groupby('date')['price'].mean()
        
        daily_prices = daily_prices.sort_index()
        print(f"Successfully fetched and consolidated {len(df)} price data points into {len(daily_prices)} daily prices.")
        return daily_prices

    except requests.exceptions.HTTPError as e:
        print(f"Error: HTTP error occurred while fetching {crypto_id.upper()} price: {e} (Status code: {response.status_code if 'response' in locals() else 'N/A'})")
        return pd.Series(dtype='float64')
    except requests.exceptions.RequestException as e:
        print(f"Error: Failed to connect to CoinGecko API: {e}")
        return pd.Series(dtype='float64')
    except Exception as e:
        print(f"Error: An unknown error occurred while fetching {crypto_id.upper()} price: {e}")
        return pd.Series(dtype='float64')

def get_github_commits(owner, repo, start_date, end_date, headers):
    print(f"\n--- Starting to fetch GitHub Commit data for {owner}/{repo} ---")
    commits_data = []
    page = 1
    per_page = 100

    api_start_date = (start_date - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    api_end_date = (end_date + timedelta(days=1)).replace(hour=23, minute=59, second=59, microsecond=999999)

    since_date_str = api_start_date.isoformat(timespec='seconds') + 'Z'
    until_date_str = api_end_date.isoformat(timespec='seconds') + 'Z'
    print(f"DEBUG(GitHub API): Search range: from {since_date_str} to {until_date_str}")

    try:
        while True:
            url = f"https://api.github.com/repos/{owner}/{repo}/commits?per_page={per_page}&page={page}&since={since_date_str}&until={until_date_str}"
            print(f"DEBUG(GitHub API): Request URL: {url}")
            response = requests.get(url, headers=headers)
            response.raise_for_status()

            commits = response.json()
            print(f"DEBUG(GitHub API): Received {len(commits)} commits for page {page}.")

            if not commits:
                print(f"DEBUG(GitHub API): No more commits found for page {page}, breaking loop.")
                break

            for commit in commits:
                commit_date_str = commit['commit']['author']['date']
                commit_date = datetime.strptime(commit_date_str, '%Y-%m-%dT%H:%M:%SZ')
                commit_message = commit['commit']['message']
                if start_date.replace(hour=0, minute=0, second=0, microsecond=0) <= commit_date.replace(hour=0, minute=0, second=0, microsecond=0) <= end_date.replace(hour=0, minute=0, second=0, microsecond=0):
                    commits_data.append({'date': commit_date, 'message': commit_message})

            if len(commits) < per_page:
                print(f"DEBUG(GitHub API): Less than {per_page} commits received, assuming last page.")
                break

            page += 1
            time.sleep(0.1)

        if not commits_data:
            print(f"Warning: No Commit data fetched from GitHub repository {owner}/{repo} within the specified date range ({start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}), despite expanded search range.")
            return pd.DataFrame(columns=['date', 'message'])

        df = pd.DataFrame(commits_data)
        df['date'] = pd.to_datetime(df['date']).dt.floor('D')
        print(f"Successfully fetched {len(df)} GitHub Commit data points.")
        return df

    except requests.exceptions.HTTPError as e:
        print(f"Error: HTTP error occurred while fetching GitHub Commits: {e} (Status code: {response.status_code if 'response' in locals() else 'N/A'})")
        if 'response' in locals() and response.status_code == 404:
            print("Please check if GitHub Owner and Repository names are correct.")
        elif 'response' in locals() and response.status_code == 403:
            print(f"GitHub API rate limit might have been reached (Status code: {response.status_code}). Consider setting GITHUB_TOKEN.")
        return pd.Series(dtype='int64')
    except requests.exceptions.RequestException as e:
        print(f"Error: Failed to connect to GitHub API: {e}")
        return pd.Series(dtype='int64')
    except Exception as e:
        print(f"Error: An unknown error occurred while fetching GitHub Commits: {e}")
        return pd.Series(dtype='int64')
