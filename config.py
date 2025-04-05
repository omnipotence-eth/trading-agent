# config.py
from dotenv import load_dotenv
import os

load_dotenv()

# Twitter/X API credentials
X_API_KEY = os.getenv('X_API_KEY')
X_API_SECRET = os.getenv('X_API_SECRET')
X_ACCESS_TOKEN = os.getenv('X_ACCESS_TOKEN')
X_ACCESS_TOKEN_SECRET = os.getenv('X_ACCESS_TOKEN_SECRET')

# Grok API key
GROK_API_KEY = os.getenv('GROK_API_KEY')

# News API key
NEWSAPI_KEY = os.getenv('NEWSAPI_KEY')

# Finnhub API key
FINNHUB_API_KEY = os.getenv('FINNHUB_API_KEY')

# Validate required environment variables
def validate_config():
    required_vars = [
        'X_API_KEY',
        'X_API_SECRET',
        'X_ACCESS_TOKEN',
        'X_ACCESS_TOKEN_SECRET',
        'GROK_API_KEY',
        'FINNHUB_API_KEY'
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

# Validate configuration on import
validate_config()