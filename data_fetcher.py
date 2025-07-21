import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import json
import os

CACHE_DIR = "cache"
GITHUB_COMMITS_CACHE_FILE = os.path.join(CACHE_DIR, "github_commits_cache.json")

def _load_cache():
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
    if os.path.exists(GITHUB_COMMITS_CACHE_FILE):
        with open(GITHUB_COMMITS_CACHE_FILE, 'r') as f:
            try:
                cache_data = json.load(f)
                print(f"DEBUG(Cache): Successfully loaded cache from {GITHUB_COMMITS_CACHE_FILE}. Contains {len(cache_data)} top-level entries.")
                return cache_data
            except json.JSONDecodeError as e:
                print(f"ERROR(Cache): Failed to decode JSON from {GITHUB_COMMITS_CACHE_FILE}: {e}. Returning empty cache.")
                return {}
    print(f"DEBUG(Cache): Cache file {GITHUB_COMMITS_CACHE_FILE} not found. Returning empty cache.")
    return {}

def _save_cache(data):
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
    with open(GITHUB_COMMITS_CACHE_FILE, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"DEBUG(Cache): Successfully saved cache to {GITHUB_COMMITS_CACHE_FILE}.")

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

    cache = _load_cache()
    repo_key = f"{owner}/{repo}"
    if repo_key not in cache:
        cache[repo_key] = {} # Initialize cache for this repo

    repo_cache = cache[repo_key]
    print(f"DEBUG(GitHub API): Initial cache for {repo_key}: {len(repo_cache)} entries.")

    requested_dates = set()
    current_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
    while current_date <= end_date.replace(hour=0, minute=0, second=0, microsecond=0):
        requested_dates.add(current_date.strftime('%Y-%m-%d'))
        current_date += timedelta(days=1)
    print(f"DEBUG(GitHub API): Requested dates: {sorted(list(requested_dates))}")

    available_dates = set(repo_cache.keys())
    print(f"DEBUG(GitHub API): Available dates in cache: {sorted(list(available_dates))}")

    missing_dates_str = sorted(list(requested_dates - available_dates))

    all_commits_for_range = []

    # Fetch missing data
    if missing_dates_str:
        print(f"DEBUG(GitHub API): Missing data for dates: {missing_dates_str}")
        
        # Group missing dates into contiguous blocks for API calls
        contiguous_blocks = []
        if missing_dates_str:
            current_block_start = datetime.strptime(missing_dates_str[0], '%Y-%m-%d')
            current_block_end = current_block_start
            for i in range(1, len(missing_dates_str)):
                date = datetime.strptime(missing_dates_str[i], '%Y-%m-%d')
                if date == current_block_end + timedelta(days=1):
                    current_block_end = date
                else:
                    contiguous_blocks.append((current_block_start, current_block_end))
                    current_block_start = date
                    current_block_end = date
            contiguous_blocks.append((current_block_start, current_block_end))

        for block_start, block_end in contiguous_blocks:
            print(f"DEBUG(GitHub API): Fetching block from {block_start.strftime('%Y-%m-%d')} to {block_end.strftime('%Y-%m-%d')}")
            
            # Expand search range slightly for API to ensure all commits are caught
            api_start_date = (block_start - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
            api_end_date = (block_end + timedelta(days=1)).replace(hour=23, minute=59, second=59, microsecond=999999)

            since_date_str = api_start_date.isoformat(timespec='seconds') + 'Z'
            until_date_str = api_end_date.isoformat(timespec='seconds') + 'Z'
            print(f"DEBUG(GitHub API): API search range: from {since_date_str} to {until_date_str}")

            page = 1
            per_page = 100
            
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
                        # If no commits are returned for a block, ensure all dates in that block are marked as checked
                        temp_date = block_start.replace(hour=0, minute=0, second=0, microsecond=0)
                        while temp_date <= block_end.replace(hour=0, minute=0, second=0, microsecond=0):
                            day_str = temp_date.strftime('%Y-%m-%d')
                            if day_str not in repo_cache: # Only add if not already present (e.g., from a previous page)
                                repo_cache[day_str] = [] # Mark as checked with no commits
                                print(f"DEBUG(GitHub API): Marked {day_str} as no commits in cache.")
                            temp_date += timedelta(days=1)
                        break

                    for commit in commits:
                        commit_date_str = commit['commit']['author']['date']
                        commit_date_obj = datetime.strptime(commit_date_str, '%Y-%m-%dT%H:%M:%SZ')
                        commit_day_str = commit_date_obj.strftime('%Y-%m-%d')
                        
                        # Only store commits within the requested block's date range
                        if block_start.replace(hour=0, minute=0, second=0, microsecond=0) <= commit_date_obj.replace(hour=0, minute=0, second=0, microsecond=0) <= block_end.replace(hour=0, minute=0, second=0, microsecond=0):
                            if commit_day_str not in repo_cache:
                                repo_cache[commit_day_str] = []
                            # Store the raw commit data or a simplified version
                            repo_cache[commit_day_str].append({
                                'date': commit_date_str, # Store as string for JSON
                                'message': commit['commit']['message']
                            })
                            print(f"DEBUG(GitHub API): Added commit for {commit_day_str} to cache.")

                    if len(commits) < per_page:
                        print(f"DEBUG(GitHub API): Less than {per_page} commits received, assuming last page.")
                        # After fetching all pages for a block, ensure all dates in that block are marked as checked
                        temp_date = block_start.replace(hour=0, minute=0, second=0, microsecond=0)
                        while temp_date <= block_end.replace(hour=0, minute=0, second=0, microsecond=0):
                            day_str = temp_date.strftime('%Y-%m-%d')
                            if day_str not in repo_cache: # Only add if not already present (e.g., if no commits were found for this specific day)
                                repo_cache[day_str] = [] # Mark as checked with no commits
                                print(f"DEBUG(GitHub API): Marked {day_str} as no commits in cache (end of block)." )
                            temp_date += timedelta(days=1)
                        break

                    page += 1
                    time.sleep(0.1) # Be kind to the API

            except requests.exceptions.HTTPError as e:
                print(f"Error: HTTP error occurred while fetching GitHub Commits: {e} (Status code: {response.status_code if 'response' in locals() else 'N/A'})")
                if 'response' in locals() and response.status_code == 404:
                    print("Please check if GitHub Owner and Repository names are correct.")
                elif 'response' in locals() and response.status_code == 403:
                    print(f"GitHub API rate limit might have been reached (Status code: {response.status_code}). Consider setting GITHUB_TOKEN.")
                # For now, return empty DataFrame on error for missing data
                return pd.DataFrame(columns=['date', 'message'])
            except requests.exceptions.RequestException as e:
                print(f"Error: Failed to connect to GitHub API: {e}")
                return pd.DataFrame(columns=['date', 'message'])
            except Exception as e:
                print(f"Error: An unknown error occurred while fetching GitHub Commits: {e}")
                return pd.DataFrame(columns=['date', 'message'])
    else:
        print("DEBUG(GitHub API): All requested dates are already in cache.")

    # Consolidate all commits for the requested range from cache
    current_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
    while current_date <= end_date.replace(hour=0, minute=0, second=0, microsecond=0):
        day_str = current_date.strftime('%Y-%m-%d')
        if day_str in repo_cache:
            for commit_data in repo_cache[day_str]:
                # Convert date string back to datetime object for DataFrame
                commit_data_copy = commit_data.copy()
                commit_data_copy['date'] = datetime.strptime(commit_data_copy['date'], '%Y-%m-%dT%H:%M:%SZ')
                all_commits_for_range.append(commit_data_copy)
        current_date += timedelta(days=1)

    if not all_commits_for_range:
        print(f"Warning: No Commit data found for {owner}/{repo} within the specified date range ({start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}).")
        df = pd.DataFrame(columns=['date', 'message'])
    else:
        df = pd.DataFrame(all_commits_for_range)
        df['date'] = pd.to_datetime(df['date']).dt.floor('D') # Ensure date is floored to day
        # Filter to ensure strict adherence to requested start_date and end_date
        df = df[(df['date'] >= start_date.replace(hour=0, minute=0, second=0, microsecond=0)) &
                (df['date'] <= end_date.replace(hour=0, minute=0, second=0, microsecond=0))]
        print(f"Successfully consolidated {len(df)} GitHub Commit data points for the requested range.")

    _save_cache(cache) # Save the updated cache

    return df
