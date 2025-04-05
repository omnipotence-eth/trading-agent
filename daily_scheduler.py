import schedule
import time
from datetime import datetime
import pytz
from content_gen import generate_market_analysis, generate_bible_verse, generate_humanitarian_insight
from post_to_x import post_tweet
from logger import setup_logger
import requests

logger = setup_logger()

def get_central_time():
    """Get current time in Central Time zone."""
    central = pytz.timezone('US/Central')
    return datetime.now(central)

def post_pre_market_analysis():
    """Post pre-market analysis thread."""
    try:
        # Only post on weekdays
        if get_central_time().weekday() >= 5:  # 5 = Saturday, 6 = Sunday
            logger.info("Skipping pre-market analysis on weekend")
            return
            
        logger.info("Generating pre-market analysis...")
        from technical_analysis import generate_analysis_report
        analysis_report = generate_analysis_report()
        
        if analysis_report:
            # Generate pre-market analysis thread
            headers = {
                "Authorization": f"Bearer {GROK_API_KEY}",
                "Content-Type": "application/json"
            }
            
            # Format analysis data for the prompt
            analysis_context = "Pre-Market Analysis:\n\n"
            analysis_context += f"Market Sentiment: {analysis_report['market_summary']['bullish_count']} Bullish, {analysis_report['market_summary']['bearish_count']} Bearish, {analysis_report['market_summary']['neutral_count']} Neutral\n\n"
            analysis_context += "Top Stock Suggestions:\n"
            for suggestion in analysis_report['top_suggestions']:
                analysis_context += f"\n{suggestion['symbol']} (Score: {suggestion['score']}/10):\n"
                analysis_context += f"Current Price: ${suggestion['current_price']:.2f}\n"
                analysis_context += f"Sentiment: {suggestion['sentiment'].capitalize()}\n"
                for point in suggestion['key_points']:
                    analysis_context += f"- {point}\n"
            
            data = {
                "model": "grok-2-latest",
                "messages": [
                    {
                        "role": "system",
                        "content": """You are a professional market analyst. Generate a pre-market analysis thread focusing on:
                        1. Overnight market developments and their implications
                        2. Technical analysis insights for key stocks
                        3. Mathematical analysis of trends and patterns
                        4. Support and resistance levels
                        5. Volume analysis and market sentiment
                        6. Trading opportunities with risk management
                        Format as a thread with 4-5 tweets. Each tweet should focus on a specific aspect. Keep it professional and data-driven. Do not use hashtags or emojis."""
                    },
                    {
                        "role": "user",
                        "content": f"Generate a pre-market analysis thread using this data:\n{analysis_context}\nFocus on technical analysis, mathematical reasoning, and actionable insights. Keep it professional and informative."
                    }
                ],
                "stream": False,
                "temperature": 0.7
            }
            
            response = requests.post(
                "https://api.x.ai/v1/chat/completions",
                headers=headers,
                json=data
            )
            
            if response.status_code == 200:
                thread = response.json()["choices"][0]["message"]["content"].strip()
                # Clean and post each tweet in the thread
                from content_gen import clean_content
                tweets = [clean_content(tweet) for tweet in thread.split("\n\n")]
                for tweet in tweets:
                    post_tweet(tweet)
                    time.sleep(2)  # Small delay between thread tweets
                logger.info("Pre-market analysis thread posted successfully")
            else:
                logger.error(f"API request failed with status {response.status_code}: {response.text}")
        else:
            logger.error("Failed to generate pre-market analysis")
    except Exception as e:
        logger.error(f"Error posting pre-market analysis: {e}")

def post_market_analysis():
    """Post daily market analysis thread."""
    try:
        # Only post on weekdays
        if get_central_time().weekday() >= 5:  # 5 = Saturday, 6 = Sunday
            logger.info("Skipping market analysis on weekend")
            return
            
        logger.info("Generating daily market analysis...")
        thread = generate_market_analysis()
        if thread:
            for tweet in thread:
                post_tweet(tweet)
                time.sleep(2)  # Small delay between thread tweets
            logger.info("Market analysis thread posted successfully")
        else:
            logger.error("Failed to generate market analysis")
    except Exception as e:
        logger.error(f"Error posting market analysis: {e}")

def post_bible_verse():
    """Post daily Bible verse."""
    try:
        logger.info("Generating daily Bible verse...")
        verse = generate_bible_verse()
        if verse:
            post_tweet(verse)
            logger.info("Bible verse posted successfully")
        else:
            logger.error("Failed to generate Bible verse")
    except Exception as e:
        logger.error(f"Error posting Bible verse: {e}")

def post_humanitarian_insight():
    """Post daily humanitarian insight."""
    try:
        logger.info("Generating daily humanitarian insight...")
        insight = generate_humanitarian_insight()
        if insight:
            post_tweet(insight)
            logger.info("Humanitarian insight posted successfully")
        else:
            logger.error("Failed to generate humanitarian insight")
    except Exception as e:
        logger.error(f"Error posting humanitarian insight: {e}")

def setup_schedule():
    """Setup daily posting schedule."""
    # Pre-market analysis at 7:30 AM CT (30 minutes before market open)
    schedule.every().day.at("07:30").do(post_pre_market_analysis)
    
    # Market analysis at 8:15 AM CT (15 minutes before market open)
    schedule.every().day.at("08:15").do(post_market_analysis)
    
    # Bible verse at 6:30 AM CT (early morning inspiration)
    schedule.every().day.at("06:30").do(post_bible_verse)
    
    # Humanitarian insight at 12:00 PM CT (lunch time engagement)
    schedule.every().day.at("12:00").do(post_humanitarian_insight)
    
    logger.info("Daily posting schedule setup completed for Central Time")

def run_scheduler():
    """Run the scheduler."""
    setup_schedule()
    logger.info(f"Starting scheduler in Central Time. Current time: {get_central_time().strftime('%Y-%m-%d %H:%M:%S %Z')}")
    
    while True:
        try:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
        except Exception as e:
            logger.error(f"Error in scheduler: {e}")
            time.sleep(300)  # Wait 5 minutes before retrying

if __name__ == "__main__":
    run_scheduler() 