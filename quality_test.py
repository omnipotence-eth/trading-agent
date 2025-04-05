# quality_test.py
import sys
import time
import random
from trend_monitor import get_trends, get_news
from content_gen import generate_tweet
from logger import setup_logger
from twitter_client import validate_credentials
from config import GROK_API_KEY, NEWSAPI_KEY

logger = setup_logger()

def validate_environment():
    """Validate that all required API keys are set."""
    if not GROK_API_KEY or GROK_API_KEY.startswith("xai-"):
        logger.error("GROK_API_KEY is not set or invalid in .env file")
        return False
    
    if NEWSAPI_KEY == "your_newsapi_key":
        logger.warning("NEWSAPI_KEY is not set in .env file, will use default news")
    
    if not validate_credentials():
        logger.error("Twitter API credentials validation failed")
        return False
    
    return True

def get_fallback_topics():
    """Return a list of fallback topics for testing when API calls fail."""
    return [
        "Market Analysis: Tech Sector Trends",
        "Trading Strategy: Risk Management",
        "Investment Opportunities: Emerging Markets",
        "Financial Planning: Retirement Goals",
        "Economic Indicators: GDP Growth"
    ]

def evaluate_tweet_quality(tweet):
    """Evaluate the quality of a tweet based on various metrics."""
    # Length check
    length_score = 1.0 if 100 <= len(tweet) <= 240 else 0.5 if len(tweet) < 100 else 0.8
    
    # Engagement potential indicators
    engagement_indicators = [
        "?",  # Questions
        "!",  # Exclamations
        "data",  # Data-driven
        "analysis",  # Analysis
        "insight",  # Insights
        "trend",  # Trends
        "opportunity",  # Opportunities
        "strategy",  # Strategies
        "breakthrough",  # Breakthroughs
        "innovation"  # Innovations
    ]
    
    engagement_score = sum(1 for indicator in engagement_indicators if indicator.lower() in tweet.lower()) / len(engagement_indicators)
    
    # Professional tone indicators
    professional_indicators = [
        "market",
        "trading",
        "investment",
        "finance",
        "economy",
        "growth",
        "performance",
        "strategy",
        "analysis",
        "trend"
    ]
    
    professional_score = sum(1 for indicator in professional_indicators if indicator.lower() in tweet.lower()) / len(professional_indicators)
    
    # Calculate overall score
    overall_score = (length_score * 0.3) + (engagement_score * 0.4) + (professional_score * 0.3)
    
    return {
        "length_score": length_score,
        "engagement_score": engagement_score,
        "professional_score": professional_score,
        "overall_score": overall_score
    }

def generate_fallback_tweet(topic):
    """Generate a fallback tweet when the API is not available."""
    templates = [
        f"🔍 Analysis: {topic} shows promising trends for investors. Key metrics indicate strong potential for growth. What's your take on this development?",
        f"📊 Market Update: {topic} presents unique opportunities. Data suggests significant market movements ahead. Stay informed!",
        f"💡 Insight: {topic} reveals interesting patterns. Strategic approach could yield positive results. How are you positioning yourself?",
        f"📈 Trading Alert: {topic} indicates potential breakthrough. Market analysis suggests favorable conditions. What's your strategy?",
        f"🎯 Investment Focus: {topic} demonstrates strong fundamentals. Expert analysis points to promising outcomes. Are you prepared?"
    ]
    return random.choice(templates)

def test_content_quality(num_samples=5):
    """Generate and evaluate multiple tweet samples for quality testing."""
    try:
        # Validate environment
        if not validate_environment():
            logger.warning("Environment validation failed, using fallback content for testing")
            topics = get_fallback_topics()
        else:
            # Fetch topics
            logger.info("Fetching topics...")
            try:
                trends = get_trends()
                news = get_news()
                topics = trends + news
            except Exception as e:
                logger.warning(f"Error fetching topics: {e}")
                topics = get_fallback_topics()
        
        logger.info(f"Using {len(topics)} topics for testing")
        
        # Generate and evaluate multiple tweets
        logger.info(f"Generating and evaluating {num_samples} tweet samples...")
        print("\n" + "="*70)
        print("QUALITY TEST RESULTS")
        print("="*70 + "\n")
        
        results = []
        
        for i in range(min(num_samples, len(topics))):
            topic = topics[i]
            logger.info(f"Generating tweet for topic: {topic}")
            
            try:
                tweet = generate_tweet(topic)
            except Exception as e:
                logger.warning(f"Error generating tweet: {e}")
                tweet = generate_fallback_tweet(topic)
            
            quality_scores = evaluate_tweet_quality(tweet)
            results.append((topic, tweet, quality_scores))
            
            print(f"Topic: {topic}")
            print(f"Tweet: {tweet}")
            print(f"Length: {len(tweet)} characters")
            print(f"Quality Scores:")
            print(f"  - Length: {quality_scores['length_score']:.2f}")
            print(f"  - Engagement: {quality_scores['engagement_score']:.2f}")
            print(f"  - Professional: {quality_scores['professional_score']:.2f}")
            print(f"  - Overall: {quality_scores['overall_score']:.2f}")
            print("-"*70 + "\n")
            
            # Small delay between generations to avoid rate limiting
            time.sleep(2)
        
        # Calculate average scores
        avg_length = sum(r[2]['length_score'] for r in results) / len(results)
        avg_engagement = sum(r[2]['engagement_score'] for r in results) / len(results)
        avg_professional = sum(r[2]['professional_score'] for r in results) / len(results)
        avg_overall = sum(r[2]['overall_score'] for r in results) / len(results)
        
        print("="*70)
        print("AVERAGE QUALITY SCORES")
        print(f"  - Length: {avg_length:.2f}")
        print(f"  - Engagement: {avg_engagement:.2f}")
        print(f"  - Professional: {avg_professional:.2f}")
        print(f"  - Overall: {avg_overall:.2f}")
        print("="*70)
        
        # Find the best tweet
        best_tweet = max(results, key=lambda x: x[2]['overall_score'])
        print("\nBEST TWEET:")
        print(f"Topic: {best_tweet[0]}")
        print(f"Tweet: {best_tweet[1]}")
        print(f"Overall Score: {best_tweet[2]['overall_score']:.2f}")
        print("="*70)
        
    except Exception as e:
        logger.error(f"Unexpected error in quality test function: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test_content_quality() 