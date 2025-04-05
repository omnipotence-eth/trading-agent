"""Test configuration file."""

from datetime import datetime, timedelta

# Current timestamp for consistent testing
CURRENT_TIME = datetime.now()

MOCK_FINNHUB_API_KEY = "test_api_key"

# Mock API responses
MOCK_QUOTE_RESPONSE = {
    'c': 150.0,  # Current price
    'h': 155.0,  # High price
    'l': 145.0,  # Low price
    'o': 148.0,  # Open price
    'pc': 147.0,  # Previous close
    't': 1619712000,  # Timestamp
    'd': 2.15,    # Change
    'dp': 0.43    # Percent change
}

MOCK_CANDLES_RESPONSE = {
    'c': [150.0] * 56,  # Close prices
    'h': [155.0] * 56,  # High prices
    'l': [145.0] * 56,  # Low prices
    'o': [148.0] * 56,  # Open prices
    's': 'ok',
    't': [1619712000 + i * 86400 for i in range(56)],  # Timestamps
    'v': [1000000] * 56  # Volumes
}

MOCK_ANALYSIS_REPORT = {
    'timestamp': datetime.now(),
    'top_suggestions': [
        {
            'symbol': 'SPY',
            'score': 0.85,
            'sentiment': 'bullish',
            'current_price': 505.28,
            'key_points': [
                'Strong upward trend',
                'RSI indicates momentum',
                'MACD shows bullish crossover'
            ]
        },
        {
            'symbol': 'QQQ',
            'score': 0.82,
            'sentiment': 'bullish',
            'current_price': 428.15,
            'key_points': [
                'Breaking resistance',
                'High volume support',
                'Positive technical indicators'
            ]
        }
    ],
    'market_summary': {
        'bullish_count': 8,
        'bearish_count': 3,
        'neutral_count': 4,
        'overall_sentiment': 'bullish',
        'key_trends': [
            'Technology sector leading',
            'Momentum building',
            'Volume supporting uptrend'
        ]
    }
}

MOCK_REAL_TIME_DATA = {
    'current_price': 150.0,
    'change': 2.15,
    'percent_change': 0.43,
    'volume': 1000000,
    'high': 155.0,
    'low': 145.0,
    'open': 148.0,
    'previous_close': 147.0,
    'timestamp': datetime.now(),
    'technical_indicators': {
        'volatility': {'value': 20.5, 'interpretation': 'moderate'},
        'rsi': {'value': 65.4, 'interpretation': 'overbought'},
        'macd': {
            'macd': 2.15,
            'signal': 1.95,
            'histogram': 0.20
        },
        'bollinger_bands': {
            'upper': 160.0,
            'middle': 150.0,
            'lower': 140.0
        },
        'stochastic': {
            'k': 65.0,
            'd': 60.0
        },
        'adx': {'value': 25.0, 'interpretation': 'weak trend'},
        'ichimoku': {
            'tenkan': 154.50,
            'kijun': 153.75,
            'senkou_a': 155.00,
            'senkou_b': 152.50,
            'chikou': 156.25
        },
        'fibonacci_levels': {
            '0.0': 145.0,
            '0.236': 147.36,
            '0.382': 148.82,
            '0.5': 150.0,
            '0.618': 151.18,
            '0.786': 152.73,
            '1.0': 155.0,
            '1.618': 161.18,
            '2.618': 171.18
        },
        'pivot_points': {
            'pp': 150.0,
            'r1': 152.5,
            's1': 147.5,
            'r2': 155.0,
            's2': 145.0,
            'r3': 157.5,
            's3': 142.5
        },
        'mfi': {'value': 58.5, 'interpretation': 'neutral'},
        'cmf': {'value': 0.15, 'interpretation': 'bullish'},
        'obv': {'value': 15000000, 'interpretation': 'bullish'},
        'williams_r': {'value': -46.34, 'interpretation': 'neutral'},
        'cci': {'value': 31.58, 'interpretation': 'neutral'},
        'aroon': {
            'up': 65.0,
            'down': 35.0,
            'oscillator': 30.0,
            'interpretation': 'bullish'
        },
        'volume_profile': {
            'value_area_high': 155.0,
            'value_area_low': 145.0,
            'value_area_volume': 800000,
            'poc': 150.0
        },
        'market_profile': {
            'value_area': {
                'high': 155.0,
                'low': 145.0,
                'volume': 800000
            },
            'poc': 150.0,
            'distribution': 'normal'
        }
    }
}

MOCK_TRADING_SUGGESTIONS = {
    'market_conditions': {
        'trend': 'bullish',
        'strength': 'strong',
        'volatility': 'moderate',
        'support_levels': [500.0, 495.0],
        'resistance_levels': [510.0, 515.0]
    },
    'options_strategies': [
        {
            'name': 'Bull Call Spread',
            'description': 'Bullish strategy with defined risk',
            'strikes': [500, 510],
            'expiration': '30-45 days',
            'position_size': 10
        }
    ],
    'stock_picks': [
        {
            'symbol': 'SPY',
            'action': 'buy',
            'entry_price': 505.28,
            'stop_loss': 495.0,
            'take_profit': 515.0
        }
    ],
    'risk_management': {
        'position_size': 100,
        'stop_loss': 495.0,
        'take_profit': 515.0
    }
} 