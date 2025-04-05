# twitter_client.py
import tweepy
from config import X_API_KEY, X_API_SECRET, X_ACCESS_TOKEN, X_ACCESS_TOKEN_SECRET
from logger import setup_logger

logger = setup_logger()

# Initialize Twitter API client
auth = tweepy.OAuth1UserHandler(X_API_KEY, X_API_SECRET, X_ACCESS_TOKEN, X_ACCESS_TOKEN_SECRET)
api = tweepy.API(auth)

def get_twitter_api():
    """Get the Twitter API client."""
    return api

def validate_credentials():
    """Validate Twitter API credentials."""
    try:
        api.verify_credentials()
        logger.info("Twitter API credentials validated successfully")
        return True
    except Exception as e:
        logger.error(f"Twitter API credentials validation failed: {e}")
        return False 