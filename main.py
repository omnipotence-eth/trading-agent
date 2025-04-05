# main.py
import random
import time
import sys
from datetime import datetime
from trend_monitor import get_trends, get_news
from content_gen import generate_tweet
from post_to_x import post_tweet
from analytics import collect_analytics
from logger import setup_logger
from twitter_client import validate_credentials
from config import GROK_API_KEY, NEWSAPI_KEY
from utils.rate_limiter import rate_limit
from utils.health_check import health_checker
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

logger = setup_logger()
scheduler = BackgroundScheduler()

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

@rate_limit("twitter_api")
def post_trading_update():
    """Post a trading update with rate limiting."""
    try:
        # Run health check
        health_status = health_checker.run_health_check()
        if health_status["status"] != "healthy":
            logger.error(f"Health check failed: {health_status}")
            return
        
        # Fetch topics
        logger.info("Fetching topics...")
        trends = get_trends()
        news = get_news()
        topic = trends[0] if trends else news[0]
        logger.info(f"Selected topic: {topic}")
        
        # Generate tweet
        logger.info("Generating tweet...")
        tweet = generate_tweet(topic)
        
        # Post tweet
        logger.info("Posting tweet...")
        posted_tweet = post_tweet(tweet)
        
        if posted_tweet:
            tweet_id = posted_tweet._json['id_str']
            logger.info(f"Tweet posted successfully with ID: {tweet_id}")
            collect_analytics(tweet_id, tweet)
            logger.info("Analytics collection started in background")
        else:
            logger.error("Tweet posting failed")
            
    except Exception as e:
        logger.error(f"Error in post_trading_update: {str(e)}")
        # Notify admin of failure (implement your notification method)

def setup_scheduler():
    """Setup the scheduler for periodic tasks."""
    # Schedule trading updates
    scheduler.add_job(
        post_trading_update,
        CronTrigger(hour='9-16', minute='*/30'),  # Every 30 minutes during market hours
        id='trading_update'
    )
    
    # Schedule health checks
    scheduler.add_job(
        health_checker.run_health_check,
        CronTrigger(minute='*/5'),  # Every 5 minutes
        id='health_check'
    )
    
    scheduler.start()
    logger.info("Scheduler started")

def main():
    """Main function to run the social agent."""
    try:
        # Validate environment
        if not validate_environment():
            logger.error("Environment validation failed, exiting")
            sys.exit(1)
        
        # Setup scheduler
        setup_scheduler()
        
        # Keep the main thread alive
        while True:
            time.sleep(60)
            
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        scheduler.shutdown()
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error in main function: {str(e)}")
        scheduler.shutdown()
        sys.exit(1)

if __name__ == "__main__":
    main()