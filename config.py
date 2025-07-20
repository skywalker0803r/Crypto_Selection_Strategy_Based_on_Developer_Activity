import os

# --- Configuration ---
DEFAULT_CRYPTO_CURRENCY = 'usd'
DEFAULT_DAYS = 364 # Default date range in days
INITIAL_CAPITAL = 10000 # Initial capital for backtesting
COMMISSION_RATE = 0.001 # Transaction commission rate, e.g., 0.1%

# Retaining PREDEFINED_CRYPTOS as requested
PREDEFINED_CRYPTOS = {
    "ethereum": {"coingecko_id": "ethereum", "github_owner": "ethereum", "github_repo": "go-ethereum"},
    "bitcoin": {"coingecko_id": "bitcoin", "github_owner": "bitcoin", "github_repo": "bitcoin"},
    "solana": {"coingecko_id": "solana", "github_owner": "solana-labs", "github_repo": "solana"},
    "polkadot": {"coingecko_id": "polkadot", "github_owner": "paritytech", "github_repo": "polkadot-sdk"},
    "cardano": {"coingecko_id": "cardano", "github_owner": "input-output-hk", "github_repo": "cardano-node"},
}

from dotenv import load_dotenv
import os

load_dotenv()

# --- GitHub Token (Sensitive, should be handled securely in production) ---
# IMPORTANT: Replace with your actual GitHub Token or use environment variables for production.
GITHUB_TOKEN = os.environ.get('GITHUB_TOKEN')
headers = {'Authorization': f'token {GITHUB_TOKEN}'} if GITHUB_TOKEN else {}
