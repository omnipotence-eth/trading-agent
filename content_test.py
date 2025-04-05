from content_gen import generate_market_analysis, generate_bible_verse, generate_humanitarian_insight
from logger import setup_logger

logger = setup_logger()

def test_content_generation():
    """Test all content types."""
    print("\n" + "="*70)
    print("CONTENT GENERATION TEST")
    print("="*70)
    
    # Test Market Analysis
    print("\nMARKET ANALYSIS THREAD")
    print("-"*70)
    market_thread = generate_market_analysis()
    if market_thread:
        for i, tweet in enumerate(market_thread, 1):
            print(f"\nTweet {i}:")
            print(tweet)
            print(f"Length: {len(tweet)} characters")
    else:
        print("Failed to generate market analysis")
    
    # Test Bible Verse
    print("\nBIBLE VERSE")
    print("-"*70)
    verse = generate_bible_verse()
    if verse:
        print(verse)
        print(f"Length: {len(verse)} characters")
    else:
        print("Failed to generate Bible verse")
    
    # Test Humanitarian Insight
    print("\nHUMANITARIAN INSIGHT")
    print("-"*70)
    insight = generate_humanitarian_insight()
    if insight:
        print(insight)
        print(f"Length: {len(insight)} characters")
    else:
        print("Failed to generate humanitarian insight")

if __name__ == "__main__":
    test_content_generation() 