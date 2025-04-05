# trend_monitor.py
import requests
from bs4 import BeautifulSoup
from config import NEWSAPI_KEY
from logger import setup_logger
from twitter_client import get_twitter_api

logger = setup_logger()
api = get_twitter_api()

def get_trends():
    """Get top 5 trends from X using the API."""
    try:
        # Get trends for Worldwide
        trends = api.get_place_trends(1)  # 1 is the WOEID for worldwide
        trend_list = [trend['name'] for trend in trends[0]['trends'][:5]]
        logger.info(f"Fetched trends: {trend_list}")
        return trend_list or ["trading"]
    except Exception as e:
        logger.error(f"Error fetching trends: {e}")
        return ["trading"]

def get_news():
    """Fetch top 5 business news headlines from NewsAPI."""
    if NEWSAPI_KEY == "your_newsapi_key":
        logger.warning("NewsAPI key not set, using default news")
        return ["market update"]
        
    url = f'https://newsapi.org/v2/top-headlines?category=business&apiKey={NEWSAPI_KEY}'
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        news = response.json()['articles']
        news_titles = [article['title'] for article in news[:5]]
        logger.info(f"Fetched news: {news_titles}")
        return news_titles or ["market update"]
    except Exception as e:
        logger.error(f"Error fetching news: {e}")
        return ["market update"]