# style_test.py
import sys
import time
from content_gen import generate_tweet
from logger import setup_logger
from config import GROK_API_KEY

logger = setup_logger()

def test_different_styles():
    """Test different tweet styles to help tailor the agent's output."""
    print("\n" + "="*70)
    print("GROK API STYLE TEST")
    print("="*70)
    
    # Check API key format
    if not GROK_API_KEY.startswith("xai-"):
        logger.error("Invalid API key format. Key should start with 'xai-'")
        return False
    
    logger.info("API key format validation passed")
    
    # Test topics
    topics = [
        "Bitcoin price movement and market sentiment analysis",
        "Tesla stock performance and electric vehicle market trends",
        "Federal Reserve interest rate decision impact on markets"
    ]
    
    # Different style prompts
    styles = [
        {
            "name": "Data-Driven Analysis",
            "system_prompt": "You are a data-driven financial analyst. Focus on specific metrics, percentages, and technical indicators. Use precise language and include exact numbers when possible. End with a question about the data implications.",
            "user_prompt": "Write a concise, data-focused tweet (under 280 characters) about {topic}. Include specific metrics, percentages, and technical indicators. Use emojis sparingly and end with a question about the data implications."
        },
        {
            "name": "Market Commentary",
            "system_prompt": "You are a market commentator providing insights on market movements. Focus on trends, patterns, and potential implications. Use a conversational yet professional tone. Include a thought-provoking question.",
            "user_prompt": "Write a concise market commentary tweet (under 280 characters) about {topic}. Focus on trends and patterns. Use a conversational yet professional tone and end with a thought-provoking question."
        },
        {
            "name": "Educational Insight",
            "system_prompt": "You are a financial educator explaining market concepts. Focus on teaching a key concept or strategy related to the topic. Use clear, accessible language while maintaining professionalism. End with a question to encourage learning.",
            "user_prompt": "Write a concise educational tweet (under 280 characters) about {topic}. Explain a key concept or strategy. Use clear, accessible language and end with a question to encourage learning."
        },
        {
            "name": "Breaking News",
            "system_prompt": "You are a financial news reporter covering breaking market developments. Focus on the latest developments, their immediate impact, and potential next steps. Use a journalistic tone with appropriate urgency. End with a question about implications.",
            "user_prompt": "Write a concise breaking news tweet (under 280 characters) about {topic}. Focus on the latest developments and their immediate impact. Use a journalistic tone and end with a question about implications."
        },
        {
            "name": "Technical Analysis",
            "system_prompt": "You are a technical analyst focusing on chart patterns and indicators. Use specific technical terms and reference key price levels, support/resistance, and chart patterns. End with a question about the technical setup.",
            "user_prompt": "Write a concise technical analysis tweet (under 280 characters) about {topic}. Reference specific price levels, support/resistance, and chart patterns. Use appropriate technical terms and end with a question about the setup."
        }
    ]
    
    try:
        for topic in topics:
            print(f"\n\nTOPIC: {topic}")
            print("="*70)
            
            for style in styles:
                print(f"\nSTYLE: {style['name']}")
                print("-"*70)
                
                # Generate tweet with custom style
                tweet = generate_tweet_with_style(topic, style)
                
                print(f"Tweet: {tweet}")
                print(f"Length: {len(tweet)} characters")
                print("-"*70)
                
                # Small delay between generations to avoid rate limiting
                time.sleep(2)
        
        print("\n" + "="*70)
        print("STYLE TEST COMPLETED")
        print("="*70)
        
        return True
        
    except Exception as e:
        logger.error(f"Error during style test: {e}")
        print("\nTest Failed with error:")
        print(f"❌ {str(e)}")
        print("="*70)
        return False

def generate_tweet_with_style(topic, style):
    """Generate a tweet with a specific style."""
    import requests
    import json
    
    headers = {
        "Authorization": f"Bearer {GROK_API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "grok-2-latest",
        "messages": [
            {
                "role": "system",
                "content": style["system_prompt"]
            },
            {
                "role": "user",
                "content": style["user_prompt"].format(topic=topic)
            }
        ],
        "stream": False,
        "temperature": 0.7
    }
    
    try:
        response = requests.post(
            "https://api.x.ai/v1/chat/completions",
            headers=headers,
            json=data
        )
        
        if response.status_code == 200:
            tweet = response.json()["choices"][0]["message"]["content"].strip()
            if len(tweet) > 280:
                tweet = tweet[:277] + "..."
            return tweet
        else:
            logger.error(f"API request failed with status {response.status_code}: {response.text}")
            raise Exception(f"API request failed: {response.text}")
            
    except Exception as e:
        logger.error(f"Error generating tweet: {e}")
        return f"Error generating {style['name']} tweet for {topic}"

if __name__ == "__main__":
    test_different_styles() 