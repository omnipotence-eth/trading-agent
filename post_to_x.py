# post_to_x.py
from twitter_client import get_twitter_api
from logger import setup_logger

logger = setup_logger()
api = get_twitter_api()

def post_tweet(tweet):
    """Post a tweet to X and return the tweet object."""
    try:
        posted_tweet = api.update_status(tweet)
        logger.info(f"Posted tweet: {tweet}")
        return posted_tweet
    except Exception as e:
        logger.error(f"Error posting tweet: {e}")
        return None