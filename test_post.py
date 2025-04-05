# test_post.py
import sys
import time
from trend_monitor import get_trends, get_news
from content_gen import generate_tweet
from logger import setup_logger
from twitter_client import validate_credentials
from config import GROK_API_KEY, NEWSAPI_KEY

logger = setup_logger()

def validate_environment():
    """Validate that all required API keys are set."""
    if not GROK_API_KEY:
        logger.error("GROK_API_KEY is not set in .env file")
        return False
    
    if NEWSAPI_KEY == "your_newsapi_key":
        logger.warning("NEWSAPI_KEY is not set in .env file, will use default news")
    
    if not validate_credentials():
        logger.error("Twitter API credentials validation failed")
        return False
    
    return True

def test_content_generation(num_samples=5):
    """Generate multiple tweet samples for testing."""
    try:
        # Validate environment
        if not validate_environment():
            logger.error("Environment validation failed, exiting")
            sys.exit(1)
        
        # Fetch topics
        logger.info("Fetching topics...")
        trends = get_trends()
        news = get_news()
        
        # Combine trends and news for more variety
        all_topics = trends + news
        logger.info(f"Found {len(all_topics)} topics")
        
        # Generate multiple tweets
        logger.info(f"Generating {num_samples} tweet samples...")
        print("\n" + "="*50)
        print("TWEET SAMPLES")
        print("="*50 + "\n")
        
        for i in range(min(num_samples, len(all_topics))):
            topic = all_topics[i]
            logger.info(f"Generating tweet for topic: {topic}")
            
            tweet = generate_tweet(topic)
            
            print(f"Topic: {topic}")
            print(f"Tweet: {tweet}")
            print(f"Length: {len(tweet)} characters")
            print("-"*50 + "\n")
            
            # Small delay between generations to avoid rate limiting
            time.sleep(2)
        
        print("="*50)
        print("TEST COMPLETE")
        print("="*50)
        
    except Exception as e:
        logger.error(f"Unexpected error in test function: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test_content_generation() 