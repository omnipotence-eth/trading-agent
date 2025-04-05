# analytics.py
import csv
import threading
from twitter_client import get_twitter_api
from logger import setup_logger
import time
from datetime import datetime
import os

logger = setup_logger()
api = get_twitter_api()

def collect_analytics(tweet_id, tweet):
    """Start a background thread to collect analytics after 24 hours."""
    thread = threading.Thread(target=_collect_analytics_thread, args=(tweet_id, tweet))
    thread.daemon = True  # Thread will exit when main program exits
    thread.start()
    logger.info(f"Started analytics collection thread for tweet ID {tweet_id}")

def _collect_analytics_thread(tweet_id, tweet):
    """Background thread to collect analytics after 24 hours."""
    try:
        time.sleep(86400)  # Wait 24 hours
        tweet_data = api.get_status(tweet_id)
        likes = tweet_data.favorite_count
        retweets = tweet_data.retweet_count
        timestamp = datetime.now().isoformat()
        
        file_exists = os.path.exists('analytics.csv')
        with open('analytics.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['tweet', 'likes', 'retweets', 'timestamp'])
            writer.writerow([tweet, likes, retweets, timestamp])
        logger.info(f"Collected analytics for tweet ID {tweet_id}")
    except Exception as e:
        logger.error(f"Error collecting analytics: {e}")