from content_gen import get_market_data, get_crypto_data, get_market_news, generate_market_analysis
from logger import setup_logger

logger = setup_logger()

def format_volume(volume):
    """Format volume number with commas."""
    try:
        return f"{int(volume):,}" if volume != "N/A" else "N/A"
    except:
        return str(volume)

def test_market_data():
    """Test comprehensive market data collection."""
    print("\n" + "="*70)
    print("COMPREHENSIVE MARKET DATA TEST")
    print("="*70)
    
    # Test Traditional Markets
    print("\nTRADITIONAL MARKETS")
    print("-"*70)
    market_data = get_market_data()
    
    if market_data:
        sectors = {
            "Major Indices": ["SPY", "DIA", "QQQ"],
            "Energy Sector": ["XLE", "USO", "XOM", "CVX", "PSX", "KMI", "EOG"],
            "Technology": ["TSLA", "AMD", "ORCL"]
        }
        
        for sector, symbols in sectors.items():
            print(f"\n{sector}:")
            print("-"*30)
            for symbol in symbols:
                if symbol in market_data:
                    data = market_data[symbol]
                    print(f"{symbol} ({data['name']}):")
                    print(f"  Price: {data['price']}")
                    print(f"  Change: {data['change_percent']}")
                    print(f"  High: ${data['high']}")
                    print(f"  Low: ${data['low']}")
                    print(f"  Volume: {format_volume(data['volume'])}")
    else:
        print("Failed to fetch market data")
    
    # Test Crypto Markets
    print("\nCRYPTOCURRENCY MARKETS")
    print("-"*70)
    crypto_data = get_crypto_data()
    
    if crypto_data:
        for symbol, data in crypto_data.items():
            print(f"\n{symbol}:")
            print(f"  Price: {data['price']}")
            print(f"  Change: {data['change_percent']}")
            print(f"  24h High: {data['24h_high']}")
            print(f"  24h Low: {data['24h_low']}")
            print(f"  Volume: {data['volume']}")
    else:
        print("Failed to fetch crypto data")
    
    # Test Market News
    print("\nMARKET NEWS")
    print("-"*70)
    news_data = get_market_news()
    
    if news_data:
        for i, news in enumerate(news_data, 1):
            print(f"\nNews {i}:")
            print(f"Headline: {news['headline']}")
            print(f"Source: {news['source']}")
            print(f"Summary: {news['summary'][:150]}...")
    else:
        print("Failed to fetch market news")
    
    # Test Market Analysis Generation
    print("\nMARKET ANALYSIS THREAD")
    print("-"*70)
    analysis = generate_market_analysis()
    
    if analysis:
        for i, tweet in enumerate(analysis, 1):
            print(f"\nTweet {i}:")
            print(tweet)
            print(f"Length: {len(tweet)} characters")
    else:
        print("Failed to generate market analysis")

if __name__ == "__main__":
    test_market_data() 