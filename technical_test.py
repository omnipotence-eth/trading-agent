from technical_analysis import get_real_time_data, analyze_stock, get_market_suggestions, generate_analysis_report
from logger import setup_logger

logger = setup_logger()

def test_real_time_data():
    """Test real-time data retrieval."""
    logger.info("Testing real-time data retrieval...")
    symbol = "AAPL"
    data = get_real_time_data(symbol)
    
    if data:
        logger.info(f"Successfully retrieved real-time data for {symbol}")
        logger.info(f"Current price: ${data['current_price']}")
        logger.info(f"Previous close: ${data['previous_close']}")
        logger.info(f"High: ${data['high']}")
        logger.info(f"Low: ${data['low']}")
        logger.info(f"Volume: {data['volume']}")
        return True
    else:
        logger.error(f"Failed to retrieve real-time data for {symbol}")
        return False

def test_stock_analysis():
    """Test stock analysis functionality."""
    logger.info("Testing stock analysis...")
    symbol = "AAPL"
    analysis = analyze_stock(symbol)
    
    if analysis:
        logger.info(f"Successfully analyzed {symbol}")
        logger.info(f"Trend: {analysis['trend']}")
        logger.info(f"Price change: {analysis['change_percent']}%")
        logger.info(f"Volume trend: {analysis['volume_trend']}")
        logger.info(f"Volatility: {analysis['volatility']}%")
        return True
    else:
        logger.error(f"Failed to analyze {symbol}")
        return False

def test_market_suggestions():
    """Test market suggestions generation."""
    logger.info("Testing market suggestions...")
    suggestions = get_market_suggestions()
    
    if suggestions:
        logger.info("Successfully generated market suggestions")
        for suggestion in suggestions[:3]:  # Show top 3 suggestions
            logger.info(f"\nSymbol: {suggestion['symbol']}")
            logger.info(f"Score: {suggestion['score']}")
            logger.info(f"Current price: ${suggestion['analysis']['current_price']}")
            logger.info(f"Trend: {suggestion['analysis']['trend']}")
        return True
    else:
        logger.error("Failed to generate market suggestions")
        return False

def test_analysis_report():
    """Test analysis report generation."""
    logger.info("Testing analysis report generation...")
    report = generate_analysis_report()
    
    if report:
        logger.info("Successfully generated analysis report")
        logger.info(f"Timestamp: {report['timestamp']}")
        logger.info("\nMarket Summary:")
        logger.info(f"Bullish stocks: {report['market_summary']['bullish_count']}")
        logger.info(f"Bearish stocks: {report['market_summary']['bearish_count']}")
        logger.info(f"Neutral stocks: {report['market_summary']['neutral_count']}")
        
        logger.info("\nTop Suggestions:")
        for suggestion in report['top_suggestions']:
            logger.info(f"\nSymbol: {suggestion['symbol']}")
            logger.info(f"Score: {suggestion['score']}")
            logger.info(f"Sentiment: {suggestion['sentiment']}")
            logger.info("Key Points:")
            for point in suggestion['key_points']:
                logger.info(f"- {point}")
        return True
    else:
        logger.error("Failed to generate analysis report")
        return False

def main():
    """Run all tests."""
    logger.info("Starting technical analysis tests...")
    
    tests = [
        ("Real-time Data Retrieval", test_real_time_data),
        ("Stock Analysis", test_stock_analysis),
        ("Market Suggestions", test_market_suggestions),
        ("Analysis Report", test_analysis_report)
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\nRunning {test_name} test...")
        try:
            result = test_func()
            results.append((test_name, result))
            logger.info(f"{test_name} test {'passed' if result else 'failed'}")
        except Exception as e:
            logger.error(f"Error in {test_name} test: {str(e)}")
            results.append((test_name, False))
    
    logger.info("\nTest Results Summary:")
    for test_name, result in results:
        logger.info(f"{test_name}: {'✓' if result else '✗'}")
    
    all_passed = all(result for _, result in results)
    logger.info(f"\nAll tests {'passed' if all_passed else 'failed'}")

if __name__ == "__main__":
    main() 