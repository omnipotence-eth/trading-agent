# grok_test.py
import sys
from content_gen import generate_tweet
from logger import setup_logger
from config import GROK_API_KEY

logger = setup_logger()

def test_grok_api():
    """Test the Grok API with a single topic and display detailed results."""
    print("\n" + "="*70)
    print("GROK API TEST")
    print("="*70)
    
    # Check API key format
    if not GROK_API_KEY.startswith("xai-"):
        logger.error("Invalid API key format. Key should start with 'xai-'")
        return False
    
    logger.info("API key format validation passed")
    
    # Test topic
    test_topic = "Bitcoin price movement and market sentiment analysis"
    
    try:
        print(f"\nGenerating tweet for topic: {test_topic}")
        print("-"*70)
        
        # Generate tweet
        tweet = generate_tweet(test_topic)
        
        print("\nGenerated Tweet:")
        print("-"*70)
        print(tweet)
        print("-"*70)
        print(f"Length: {len(tweet)} characters")
        
        # Basic content checks
        checks = {
            "Length": len(tweet) <= 280,
            "Topic relevance": any(word.lower() in tweet.lower() 
                                 for word in ["bitcoin", "price", "market", "analysis"]),
            "Professional tone": any(word.lower() in tweet.lower() 
                                   for word in ["analysis", "trend", "data", "market"]),
            "Engagement elements": any(char in tweet for char in ["?", "!", "📈", "🔍", "💡"])
        }
        
        print("\nContent Checks:")
        print("-"*70)
        for check, passed in checks.items():
            status = "✅ Passed" if passed else "❌ Failed"
            print(f"{check}: {status}")
        
        all_passed = all(checks.values())
        print("\nOverall Test Result:", "✅ Passed" if all_passed else "❌ Failed")
        print("="*70)
        
        return all_passed
        
    except Exception as e:
        logger.error(f"Error during Grok API test: {e}")
        print("\nTest Failed with error:")
        print(f"❌ {str(e)}")
        print("="*70)
        return False

if __name__ == "__main__":
    test_grok_api() 