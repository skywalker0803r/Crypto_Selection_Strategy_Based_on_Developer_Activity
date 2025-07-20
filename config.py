import os

# --- Configuration ---
DEFAULT_CRYPTO_CURRENCY = 'usd'
DEFAULT_DAYS = 364 # Default date range in days
INITIAL_CAPITAL = 10000 # Initial capital for backtesting
COMMISSION_RATE = 0.001 # Transaction commission rate, e.g., 0.1%

# Retaining PREDEFINED_CRYPTOS as requested
PREDEFINED_CRYPTOS = {
    "ethereum": {"binance_symbol": "ETHUSDT", "github_owner": "ethereum", "github_repo": "go-ethereum"},
    "bitcoin": {"binance_symbol": "BTCUSDT", "github_owner": "bitcoin", "github_repo": "bitcoin"},
    "solana": {"binance_symbol": "SOLUSDT", "github_owner": "solana-labs", "github_repo": "solana"},
    "polkadot": {"binance_symbol": "DOTUSDT", "github_owner": "paritytech", "github_repo": "polkadot-sdk"},
    "cardano": {"binance_symbol": "ADAUSDT", "github_owner": "input-output-hk", "github_repo": "cardano-node"},
}

from dotenv import load_dotenv
import os

load_dotenv()

# --- GitHub Token (Sensitive, should be handled securely in production) ---
# IMPORTANT: Replace with your actual GitHub Token or use environment variables for production.
GITHUB_TOKEN = os.environ.get('GITHUB_TOKEN')
headers = {'Authorization': f'token {GITHUB_TOKEN}'} if GITHUB_TOKEN else {}
