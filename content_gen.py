# content_gen.py
import os
import requests
import json
import re
import time
from datetime import datetime, timedelta
import pytz
from logger import setup_logger
from config import GROK_API_KEY, FINNHUB_API_KEY
from technical_analysis import generate_analysis_report

logger = setup_logger()

def get_crypto_data():
    """Get cryptocurrency market data."""
    try:
        crypto_symbols = ["BTCUSD", "ETHUSD", "DOGEUSD", "SOLUSD", "ADAUSD"]
        crypto_data = {}
        base_url = "https://finnhub.io/api/v1/crypto/candle"
        
        # Get current timestamp
        now = datetime.now()
        end_timestamp = int(now.timestamp())
        start_timestamp = int((now - timedelta(days=1)).timestamp())
        
        for symbol in crypto_symbols:
            params = {
                "token": FINNHUB_API_KEY,
                "symbol": f"BINANCE:{symbol}",
                "resolution": "D",  # Daily resolution
                "from": start_timestamp,
                "to": end_timestamp
            }
            
            response = requests.get(base_url, params=params)
            if response.status_code == 200:
                data = response.json()
                if data.get('s') == 'ok':
                    current_price = data['c'][-1]
                    previous_price = data['o'][-1]
                    
                    change_percent = ((current_price - previous_price) / previous_price) * 100
                    
                    crypto_data[symbol] = {
                        "price": f"${current_price:.2f}",
                        "change_percent": f"{change_percent:.2f}%",
                        "24h_high": f"${max(data['h']):.2f}",
                        "24h_low": f"${min(data['l']):.2f}",
                        "volume": f"${data['v'][-1]:,.0f}"
                    }
            
            time.sleep(0.5)  # Rate limiting
            
        return crypto_data
    except Exception as e:
        logger.error(f"Error fetching crypto data: {e}")
        return None

def get_market_news():
    """Get relevant market news."""
    try:
        base_url = "https://finnhub.io/api/v1/news"
        params = {
            "token": FINNHUB_API_KEY,
            "category": "general"
        }
        
        response = requests.get(base_url, params=params)
        if response.status_code == 200:
            news_data = response.json()
            # Filter and format relevant news
            relevant_news = []
            for news in news_data[:5]:  # Get top 5 news items
                relevant_news.append({
                    "headline": news.get('headline'),
                    "summary": news.get('summary'),
                    "source": news.get('source')
                })
            return relevant_news
        return None
    except Exception as e:
        logger.error(f"Error fetching market news: {e}")
        return None

def get_market_data():
    """Get real-time market data using Finnhub API."""
    try:
        # Get major indices and Texas-focused stocks
        symbols = {
            # Major Indices
            "SPY": "S&P 500 ETF",
            "DIA": "Dow Jones ETF",
            "QQQ": "Nasdaq ETF",
            # Energy Sector
            "XLE": "Energy Sector ETF",
            "USO": "US Oil ETF",
            # Texas-based Companies
            "XOM": "Exxon Mobil",
            "CVX": "Chevron",
            "PSX": "Phillips 66",
            "KMI": "Kinder Morgan",
            "EOG": "EOG Resources",
            # Tech Companies with Texas Presence
            "TSLA": "Tesla",
            "AMD": "AMD",
            "ORCL": "Oracle"
        }
        
        market_data = {}
        base_url = "https://finnhub.io/api/v1"
        
        for symbol in symbols:
            # Get quote data
            quote_params = {
                "token": FINNHUB_API_KEY,
                "symbol": symbol
            }
            
            response = requests.get(f"{base_url}/quote", params=quote_params)
            if response.status_code == 200:
                data = response.json()
                if data:
                    current_price = data.get("c", "N/A")
                    previous_close = data.get("pc", "N/A")
                    
                    # Calculate percentage change
                    if current_price != "N/A" and previous_close != "N/A" and previous_close != 0:
                        change_percent = ((current_price - previous_close) / previous_close) * 100
                        change_percent_str = f"{change_percent:.2f}%"
                    else:
                        change_percent_str = "N/A"
                    
                    market_data[symbol] = {
                        "name": symbols[symbol],
                        "price": f"${current_price:.2f}" if current_price != "N/A" else "N/A",
                        "change_percent": change_percent_str,
                        "high": data.get("h", "N/A"),
                        "low": data.get("l", "N/A"),
                        "volume": data.get("v", "N/A")
                    }
            
            time.sleep(0.5)  # Rate limiting
            
        return market_data
    except Exception as e:
        logger.error(f"Error fetching market data: {e}")
        return None

def clean_content(text):
    """Remove hashtags and emojis from text."""
    # Remove hashtags
    text = re.sub(r'#\w+', '', text)
    # Remove emojis
    text = re.sub(r'[\U00010000-\U0010ffff]', '', text)
    # Clean up extra spaces
    text = ' '.join(text.split())
    return text.strip()

def generate_market_analysis():
    """Generate a daily market analysis thread with real-time data."""
    try:
        market_data = get_market_data()
        crypto_data = get_crypto_data()
        news_data = get_market_news()
        
        # Get technical analysis and suggestions
        analysis_report = generate_analysis_report()
        
        # Format market data for the prompt
        market_context = "Current Market Data:\n\n"
        
        # Traditional Markets
        market_context += "Traditional Markets:\n"
        if market_data:
            # Group by sector
            sectors = {
                "Major Indices": ["SPY", "DIA", "QQQ"],
                "Energy Sector": ["XLE", "USO", "XOM", "CVX", "PSX", "KMI", "EOG"],
                "Technology": ["TSLA", "AMD", "ORCL"]
            }
            
            for sector, symbols in sectors.items():
                market_context += f"\n{sector}:\n"
                for symbol in symbols:
                    if symbol in market_data:
                        data = market_data[symbol]
                        market_context += f"{symbol} ({data['name']}): {data['price']} ({data['change_percent']})\n"
        
        # Crypto Markets
        market_context += "\nCryptocurrency Markets:\n"
        if crypto_data:
            for symbol, data in crypto_data.items():
                market_context += f"{symbol}: {data['price']} ({data['change_percent']}) | 24h Volume: {data['volume']}\n"
        
        # Recent News
        market_context += "\nRecent Market News:\n"
        if news_data:
            for news in news_data:
                market_context += f"- {news['headline']} (Source: {news['source']})\n"
        
        # Technical Analysis and Suggestions
        if analysis_report:
            market_context += "\nTechnical Analysis Summary:\n"
            market_context += f"Market Sentiment: {analysis_report['market_summary']['bullish_count']} Bullish, {analysis_report['market_summary']['bearish_count']} Bearish, {analysis_report['market_summary']['neutral_count']} Neutral\n\n"
            market_context += "Top Stock Suggestions:\n"
            for suggestion in analysis_report['top_suggestions']:
                market_context += f"\n{suggestion['symbol']} (Score: {suggestion['score']}/10):\n"
                market_context += f"Current Price: ${suggestion['current_price']:.2f}\n"
                market_context += f"Sentiment: {suggestion['sentiment'].capitalize()}\n"
                for point in suggestion['key_points']:
                    market_context += f"- {point}\n"
        
        headers = {
            "Authorization": f"Bearer {GROK_API_KEY}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "grok-2-latest",
            "messages": [
                {
                    "role": "system",
                    "content": """You are a professional market analyst based in Texas. Generate a comprehensive market analysis thread. Focus on:
                    1. Key market movements and their impact on Texas industries (especially energy, technology, and agriculture)
                    2. Oil and gas sector performance and implications for Texas energy companies
                    3. Cryptocurrency market trends and their relationship with traditional markets
                    4. Regional economic implications for Texas businesses and investors
                    5. Trading opportunities relevant to Central Time Zone traders
                    6. Integration of relevant news events with market movements
                    7. Technical analysis insights and stock suggestions
                    8. Mathematical analysis of trends and patterns
                    Format as a thread with 5-6 tweets. Each tweet should focus on a specific aspect. Keep it professional and data-driven. Do not use hashtags or emojis."""
                },
                {
                    "role": "user",
                    "content": f"Generate a market analysis thread using this comprehensive data:\n{market_context}\nFocus on implications for Texas-based investors and regional economic impact. Include technical analysis insights and mathematical reasoning for stock suggestions. Keep it professional and informative."
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
            # Clean each tweet in the thread
            return [clean_content(tweet) for tweet in thread.split("\n\n")]
        else:
            logger.error(f"API request failed with status {response.status_code}: {response.text}")
            return None
            
    except Exception as e:
        logger.error(f"Error generating market analysis: {e}")
        return None

def generate_bible_verse():
    """Generate a relevant Bible verse based on current news."""
    try:
        headers = {
            "Authorization": f"Bearer {GROK_API_KEY}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "grok-2-latest",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a spiritual advisor. Select a Bible verse that is relevant to current events and provides positive, uplifting guidance. Keep it tasteful and inspiring. Do not use hashtags or emojis."
                },
                {
                    "role": "user",
                    "content": "Generate a tweet with a relevant Bible verse for today, considering current events. Include the verse reference and a brief, positive reflection. Do not use hashtags or emojis."
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
            return clean_content(response.json()["choices"][0]["message"]["content"].strip())
        else:
            logger.error(f"API request failed with status {response.status_code}: {response.text}")
            return None
            
    except Exception as e:
        logger.error(f"Error generating Bible verse: {e}")
        return None

def generate_humanitarian_insight():
    """Generate valuable information for humanity."""
    try:
        headers = {
            "Authorization": f"Bearer {GROK_API_KEY}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "grok-2-latest",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a humanitarian advisor. Generate informative content that provides value to humanity. Focus on positive developments, solutions to global challenges, or educational insights. Do not use hashtags or emojis."
                },
                {
                    "role": "user",
                    "content": "Generate a tweet with valuable information for humanity. Focus on positive developments, solutions, or educational insights. Do not use hashtags or emojis."
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
            return clean_content(response.json()["choices"][0]["message"]["content"].strip())
        else:
            logger.error(f"API request failed with status {response.status_code}: {response.text}")
            return None
            
    except Exception as e:
        logger.error(f"Error generating humanitarian insight: {e}")
        return None

def generate_tweet(topic):
    """Generate a single informative tweet."""
    try:
        headers = {
            "Authorization": f"Bearer {GROK_API_KEY}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "grok-2-latest",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a professional content creator. Generate informative, engaging tweets that provide value to readers. Focus on clarity, accuracy, and professionalism. Do not use hashtags or emojis."
                },
                {
                    "role": "user",
                    "content": f"Write a concise, informative tweet about {topic}. Focus on providing valuable information. Keep it professional and engaging. Do not use hashtags or emojis."
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
            return clean_content(response.json()["choices"][0]["message"]["content"].strip())
        else:
            logger.error(f"API request failed with status {response.status_code}: {response.text}")
            return None
            
    except Exception as e:
        logger.error(f"Error generating tweet: {e}")
        return None