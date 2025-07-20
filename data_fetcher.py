import requests
import pandas as pd
from datetime import datetime, timedelta
import time

def get_binance_trading_pairs(top_n, currency, predefined_cryptos):
    """
    Fetches a list of top N cryptocurrency trading pairs from Binance API,
    excluding those already present in PREDEFINED_CRYPTOS.
    """
    print(f"--- Fetching top {top_n} cryptocurrency trading pairs from Binance (excluding predefined) ---")
    url = "https://api.binance.com/api/v3/exchangeInfo"

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        predefined_symbols = set(predefined_cryptos.keys())
        filtered_symbols = []
        
        # Filter for USDT pairs and sort by volume (Binance API doesn't provide market cap directly)
        # This is a simplified approach; a more robust solution might involve fetching 24hr ticker data
        # to get actual volumes and then sorting. For now, we'll just get all USDT pairs.
        
        usdt_pairs = [s['symbol'] for s in data['symbols'] if s['quoteAsset'] == currency.upper() and s['status'] == 'TRADING']
        
        for symbol in usdt_pairs:
            if symbol not in predefined_symbols:
                filtered_symbols.append(symbol)
            if len(filtered_symbols) >= top_n:
                break
        
        filtered_symbols.sort()
        print(f"Successfully fetched {len(filtered_symbols)} Binance trading pairs (top {top_n} excluding predefined).")
        return filtered_symbols
    except requests.exceptions.RequestException as e:
        error_message = f"Error: Failed to fetch Binance trading pairs: {e}"
        if 'response' in locals() and hasattr(response, 'text'):
            error_message += f"\nResponse content: {response.text}"
        print(error_message)
        return []

def get_crypto_prices(symbol, currency, start_date, end_date):
    print(f"\n--- Starting to fetch {symbol.upper()} price data ({currency.upper()}) ---")
    
    # Binance API expects milliseconds for timestamps
    start_timestamp_ms = int(start_date.timestamp() * 1000)
    end_timestamp_ms = int(end_date.timestamp() * 1000)

    # Binance API endpoint for historical klines (candlestick data)
    url = "https://api.binance.com/api/v3/klines"
    
    # Interval for daily data
    interval = "1d" 
    
    params = {
        "symbol": symbol.upper(),
        "interval": interval,
        "startTime": start_timestamp_ms,
        "endTime": end_timestamp_ms,
        "limit": 1000 # Max 1000 data points per request
    }
    print(f"DEBUG(Price API): Request URL: {url}")
    print(f"DEBUG(Price API): Request Params: {params}")

    all_klines = []
    while True:
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            klines = response.json()

            if not klines:
                break
            
            all_klines.extend(klines)
            
            # If we received less than the limit, it means we've fetched all data
            if len(klines) < params["limit"]:
                break
            
            # For the next request, start from the end of the last fetched data point
            params["startTime"] = klines[-1][0] + 1 # Add 1 millisecond to avoid duplicate
            
            time.sleep(0.1) # Be kind to the API
            
        except requests.exceptions.HTTPError as e:
            print(f"Error: HTTP error occurred while fetching {symbol.upper()} price: {e} (Status code: {response.status_code if 'response' in locals() else 'N/A'})")
            return pd.Series(dtype='float64')
        except requests.exceptions.RequestException as e:
            print(f"Error: Failed to connect to Binance API: {e}")
            return pd.Series(dtype='float64')
        except Exception as e:
            print(f"Error: An unknown error occurred while fetching {symbol.upper()} price: {e}")
            return pd.Series(dtype='float64')

    if not all_klines:
        print(f"Warning: No price data fetched for {symbol.upper()} from Binance. Check symbol or date range.")
        return pd.Series(dtype='float64')

    prices = []
    for kline in all_klines:
        timestamp_ms = kline[0]
        close_price = float(kline[4]) # Close price is at index 4
        prices.append({'date': datetime.fromtimestamp(timestamp_ms / 1000), 'price': close_price})

    df = pd.DataFrame(prices)
    df['date'] = pd.to_datetime(df['date']).dt.floor('D')
    
    # Group by date and calculate the mean price to handle duplicates (though 1d interval should prevent this)
    daily_prices = df.groupby('date')['price'].mean()
    
    daily_prices = daily_prices.sort_index()
    print(f"Successfully fetched and consolidated {len(df)} price data points into {len(daily_prices)} daily prices.")
    return daily_prices

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
