import requests
import json
from config import FINNHUB_API_KEY
from logger import setup_logger

logger = setup_logger()

def test_finnhub_api():
    """Test Finnhub API connection and data retrieval."""
    print("\n" + "="*70)
    print("FINNHUB API TEST")
    print("="*70)
    
    # Test API key
    if not FINNHUB_API_KEY:
        print("Error: Finnhub API key not found in environment variables")
        return
    
    # Test symbols
    symbols = ["SPY", "DIA", "QQQ", "XLE", "USO"]
    
    for symbol in symbols:
        print(f"\nTesting {symbol} quote data:")
        print("-"*50)
        
        try:
            # Get quote data
            response = requests.get(
                f"https://finnhub.io/api/v1/quote",
                params={"token": FINNHUB_API_KEY, "symbol": symbol}
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"Status: Success (HTTP {response.status_code})")
                print(f"Current price: ${data.get('c', 'N/A')}")
                print(f"Previous close: ${data.get('pc', 'N/A')}")
                print(f"High: ${data.get('h', 'N/A')}")
                print(f"Low: ${data.get('l', 'N/A')}")
                print(f"Open: ${data.get('o', 'N/A')}")
                
                # Calculate percentage change
                current = data.get('c')
                previous = data.get('pc')
                if current and previous and previous != 0:
                    change_percent = ((current - previous) / previous) * 100
                    print(f"Change: {change_percent:.2f}%")
            else:
                print(f"Error: API request failed with status {response.status_code}")
                print(f"Response: {response.text}")
                
        except Exception as e:
            print(f"Error: {e}")
    
    print("\n" + "="*70)
    print("TEST COMPLETED")
    print("="*70)

if __name__ == "__main__":
    test_finnhub_api() 