import unittest
from unittest.mock import patch, MagicMock
from trading_suggestions import (
    generate_trading_suggestions,
    format_trading_suggestions_for_twitter,
    calculate_strike_prices,
    calculate_position_size,
    calculate_risk_reward_ratio,
    generate_dynamic_options_strategies
)
from technical_analysis import (
    get_real_time_data,
    generate_analysis_report,
    calculate_volatility,
    calculate_rsi,
    calculate_macd,
    calculate_bollinger_bands,
    calculate_adx,
    calculate_ichimoku,
    calculate_fibonacci_levels,
    calculate_pivot_points,
    calculate_money_flow_index,
    calculate_chaikin_money_flow,
    calculate_obv,
    calculate_stochastic,
    calculate_williams_r,
    calculate_cci,
    calculate_aroon,
    calculate_volume_profile,
    calculate_market_profile
)
from test_config import (
    MOCK_QUOTE_RESPONSE,
    MOCK_CANDLES_RESPONSE,
    MOCK_ANALYSIS_REPORT,
    MOCK_REAL_TIME_DATA,
    MOCK_TRADING_SUGGESTIONS
)
from logger import setup_logger
import time
import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

logger = setup_logger()

# Set up mock API key for testing
os.environ['FINNHUB_API_KEY'] = 'test_api_key'

class TestTradingSuggestions(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.symbol = "SPY"
        self.current_price = 505.28
        self.volatility = 0.20
        self.sentiment = "bullish"
        
    def test_calculate_strike_prices(self):
        """Test strike price calculation."""
        strikes = calculate_strike_prices(
            self.current_price,
            self.volatility,
            self.sentiment
        )
        
        self.assertIsInstance(strikes, dict)
        self.assertIn('buy_strike', strikes)
        self.assertIn('sell_strike', strikes)
        self.assertGreater(strikes['sell_strike'], strikes['buy_strike'])
        self.assertGreater(strikes['sell_strike'], self.current_price)
        self.assertLess(strikes['buy_strike'], self.current_price)
        
    def test_calculate_position_size(self):
        """Test position size calculation."""
        account_size = 100000
        risk_per_trade = 1.0
        stop_loss = self.current_price * 0.95
        
        position_size = calculate_position_size(
            account_size,
            risk_per_trade,
            stop_loss,
            self.current_price
        )
        
        self.assertIsInstance(position_size, int)
        self.assertGreater(position_size, 0)
        
    def test_calculate_risk_reward_ratio(self):
        """Test risk-reward ratio calculation."""
        entry_price = self.current_price
        target_price = entry_price * 1.05
        stop_loss = entry_price * 0.95
        
        ratio = calculate_risk_reward_ratio(
            entry_price,
            target_price,
            stop_loss
        )
        
        self.assertIsInstance(ratio, float)
        self.assertGreater(ratio, 0)
        
    def test_generate_dynamic_options_strategies(self):
        """Test options strategy generation."""
        technical_data = {
            'rsi': 65,
            'macd': {'macd': 0.5, 'signal': 0.3},
            'adx': 30,
            'mfi': 60,
            'stochastic': {'k': 75, 'd': 70},
            'williams_r': -30,
            'cci': 100,
            'aroon': {'up': 60, 'down': 40}
        }
        
        strategies = generate_dynamic_options_strategies(
            self.symbol,
            self.current_price,
            self.sentiment,
            self.volatility,
            technical_data
        )
        
        self.assertIsInstance(strategies, list)
        self.assertGreater(len(strategies), 0)
        
        for strategy in strategies:
            self.assertIn('strategy', strategy)
            self.assertIn('description', strategy)
            self.assertIn('entry', strategy)
            self.assertIn('exit_conditions', strategy)
            
    @patch('trading_suggestions.get_real_time_data')
    def test_generate_trading_suggestions(self, mock_get_real_time_data):
        """Test trading suggestions generation."""
        # Set up mock data
        mock_get_real_time_data.return_value = {
            "current_price": 505.28,
            "change": 2.15,
            "percent_change": 0.43,
            "high": 510.0,
            "low": 500.0,
            "open": 503.0,
            "previous_close": 503.13,
            "volume": 1000000,
            "timestamp": datetime.now(),
            "technical_indicators": {
                "volatility": {"value": 0.20, "interpretation": "moderate"},
                "rsi": {"value": 65.4, "interpretation": "overbought"},
                "macd": {
                    "macd": 2.15,
                    "signal": 1.95,
                    "histogram": 0.20
                },
                "bollinger_bands": {
                    "upper": 510.0,
                    "middle": 505.0,
                    "lower": 500.0
                },
                "adx": {"value": 25.0, "interpretation": "weak trend"},
                "ichimoku": {
                    "tenkan": 506.0,
                    "kijun": 504.0,
                    "senkou_a": 508.0,
                    "senkou_b": 502.0,
                    "chikou": 507.0
                },
                "fibonacci": {
                    "0.0": 500.0,
                    "0.236": 501.18,
                    "0.382": 502.09,
                    "0.5": 502.5,
                    "0.618": 502.91,
                    "0.786": 503.93,
                    "1.0": 505.0,
                    "1.618": 507.59,
                    "2.618": 512.95
                },
                "pivot_points": {
                    "pp": 504.38,
                    "r1": 506.75,
                    "s1": 502.0,
                    "r2": 508.13,
                    "s2": 500.63,
                    "r3": 509.5,
                    "s3": 499.25
                },
                "mfi": {"value": 58.5, "interpretation": "neutral"},
                "cmf": {"value": 0.15, "interpretation": "bullish"},
                "obv": {"value": 15000000, "interpretation": "bullish"},
                "stochastic": {
                    "k": 65.0,
                    "d": 60.0
                },
                "williams_r": {"value": -46.34, "interpretation": "neutral"},
                "cci": {"value": 125.0, "interpretation": "bullish"},
                "aroon": {
                    "up": 65.0,
                    "down": 35.0
                },
                "volume_profile": {
                    "value_area": {
                        "high": 508.0,
                        "low": 502.0
                    },
                    "poc": 505.0
                },
                "market_profile": {
                    "value_area": {
                        "high": 508.0,
                        "low": 502.0
                    },
                    "poc": 505.0
                }
            }
        }
        
        suggestions = generate_trading_suggestions("SPY")
        
        self.assertIsInstance(suggestions, dict)
        self.assertIn('market_conditions', suggestions)
        self.assertIn('options_strategies', suggestions)
        self.assertIn('stock_picks', suggestions)
        self.assertIn('risk_management', suggestions)
        
        # Check market conditions
        market_conditions = suggestions['market_conditions']
        self.assertIn('trend', market_conditions)
        self.assertIn('strength', market_conditions)
        self.assertIn('volatility', market_conditions)
        self.assertIn('support_levels', market_conditions)
        self.assertIn('resistance_levels', market_conditions)
        
        # Check options strategies
        options_strategies = suggestions['options_strategies']
        self.assertIsInstance(options_strategies, list)
        if options_strategies:
            strategy = options_strategies[0]
            self.assertIn('strategy', strategy)
            self.assertIn('description', strategy)
            self.assertIn('entry', strategy)
            
        # Check stock picks
        stock_picks = suggestions['stock_picks']
        self.assertIsInstance(stock_picks, list)
        if stock_picks:
            pick = stock_picks[0]
            self.assertIn('symbol', pick)
            self.assertIn('action', pick)
            self.assertIn('entry_price', pick)
            self.assertIn('stop_loss', pick)
            self.assertIn('take_profit', pick)
            
        # Check risk management
        risk_management = suggestions['risk_management']
        self.assertIn('stop_loss', risk_management)
        self.assertIn('take_profit', risk_management)
        self.assertIn('position_size', risk_management)
        
    def test_format_trading_suggestions_for_twitter(self):
        """Test formatting trading suggestions for Twitter."""
        suggestions = {
            "market_conditions": {
                "trend": "bullish",
                "strength": "strong",
                "volatility": "moderate",
                "support_levels": [500.0, 495.0],
                "resistance_levels": [510.0, 515.0]
            },
            "options_strategies": [
                {
                    "strategy": "Bull Call Spread",
                    "description": "Bullish strategy with defined risk",
                    "entry": {
                        "buy_call": {
                            "strike": 505.0,
                            "expiration": "30-45 days",
                            "position_size": 10
                        },
                        "sell_call": {
                            "strike": 510.0,
                            "expiration": "30-45 days",
                            "position_size": 10
                        }
                    },
                    "exit_conditions": {
                        "profit_target": 515.0,
                        "stop_loss": 500.0,
                        "risk_reward_ratio": 2.0
                    }
                }
            ],
            "stock_picks": [
                {
                    "symbol": "SPY",
                    "action": "buy",
                    "entry_price": 505.28,
                    "stop_loss": 495.0,
                    "take_profit": 515.0
                }
            ],
            "risk_management": {
                "position_size": 100,
                "stop_loss": 495.0,
                "take_profit": 515.0
            }
        }
        
        formatted = format_trading_suggestions_for_twitter(suggestions)
        
        self.assertIsInstance(formatted, str)
        self.assertLessEqual(len(formatted), 280)  # Twitter character limit
        
        # Check if the tweet contains all required sections or is truncated
        if len(formatted) < 280:
            self.assertIn("📊 Market Analysis:", formatted)
            self.assertIn("🎯 Options Strategies:", formatted)
            self.assertIn("💎 Stock Picks:", formatted)
            self.assertIn("⚠️ Risk Management:", formatted)
            self.assertIn("DISCLAIMER", formatted)
        else:
            # If tweet is truncated, check that it ends with "..."
            self.assertTrue(formatted.endswith("..."))

class TestTechnicalAnalysis(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.symbol = "SPY"
        # Generate more realistic price data with some movement
        base_price = 150.0
        self.prices = []
        self.highs = []
        self.lows = []
        self.volumes = []
        
        # Create 56 days of data with some trend and volatility
        for i in range(56):
            # Add a slight upward trend and some random movement
            price = base_price + (i * 0.1) + ((-1) ** i) * 0.5
            high = price + 1.0
            low = price - 1.0
            volume = 1000000 + ((-1) ** i) * 100000
            
            self.prices.append(price)
            self.highs.append(high)
            self.lows.append(low)
            self.volumes.append(volume)
        
        # Ensure lists are numpy arrays for calculations
        self.prices = np.array(self.prices)
        self.highs = np.array(self.highs)
        self.lows = np.array(self.lows)
        self.volumes = np.array(self.volumes)
        
    def test_calculate_volatility(self):
        """Test volatility calculation."""
        volatility = calculate_volatility(self.prices)
        self.assertIsInstance(volatility, dict)
        self.assertIn('annual_vol', volatility)
        self.assertIn('daily_vol', volatility)
        self.assertIn('high', volatility)
        self.assertIn('low', volatility)
        self.assertGreater(volatility['annual_vol'], 0)
        self.assertGreater(volatility['daily_vol'], 0)
        self.assertGreater(volatility['high'], volatility['low'])
        
    def test_calculate_rsi(self):
        """Test RSI calculation."""
        rsi = calculate_rsi(self.prices)
        self.assertIsInstance(rsi, float)
        self.assertGreaterEqual(rsi, 0)
        self.assertLessEqual(rsi, 100)
        
    def test_calculate_macd(self):
        """Test MACD calculation."""
        macd = calculate_macd(self.prices)
        self.assertIsInstance(macd, dict)
        self.assertIn('macd', macd)
        self.assertIn('signal', macd)
        self.assertIn('histogram', macd)
        self.assertIsInstance(macd['macd'], float)
        self.assertIsInstance(macd['signal'], float)
        self.assertIsInstance(macd['histogram'], float)
        
    def test_calculate_bollinger_bands(self):
        """Test Bollinger Bands calculation."""
        bb = calculate_bollinger_bands(self.prices)
        self.assertIsInstance(bb, dict)
        self.assertIn('upper', bb)
        self.assertIn('middle', bb)
        self.assertIn('lower', bb)
        self.assertGreater(bb['upper'], bb['middle'])
        self.assertLess(bb['lower'], bb['middle'])
        
    def test_calculate_adx(self):
        """Test ADX calculation."""
        adx = calculate_adx(self.highs, self.lows, self.prices)
        self.assertIsInstance(adx, float)
        self.assertGreaterEqual(adx, 0)
        self.assertLessEqual(adx, 100)
        
    def test_calculate_ichimoku(self):
        """Test Ichimoku Cloud calculation."""
        ichimoku = calculate_ichimoku(self.highs, self.lows, self.prices)
        self.assertIsInstance(ichimoku, dict)
        self.assertIn('tenkan', ichimoku)
        self.assertIn('kijun', ichimoku)
        self.assertIn('senkou_a', ichimoku)
        self.assertIn('senkou_b', ichimoku)
        self.assertIn('chikou', ichimoku)
        
    def test_calculate_fibonacci_levels(self):
        """Test Fibonacci levels calculation."""
        fib_levels = calculate_fibonacci_levels(max(self.highs), min(self.lows))
        self.assertIsInstance(fib_levels, dict)
        self.assertIn('0.0', fib_levels)
        self.assertIn('0.236', fib_levels)
        self.assertIn('0.382', fib_levels)
        self.assertIn('0.5', fib_levels)
        self.assertIn('0.618', fib_levels)
        self.assertIn('0.786', fib_levels)
        self.assertIn('1.0', fib_levels)
        self.assertIn('1.618', fib_levels)
        self.assertIn('2.618', fib_levels)
        
    def test_calculate_pivot_points(self):
        """Test pivot points calculation."""
        pivot_points = calculate_pivot_points(self.highs[-1], self.lows[-1], self.prices[-1])
        self.assertIsInstance(pivot_points, dict)
        self.assertIn('pp', pivot_points)
        self.assertIn('r1', pivot_points)
        self.assertIn('s1', pivot_points)
        self.assertIn('r2', pivot_points)
        self.assertIn('s2', pivot_points)
        self.assertIn('r3', pivot_points)
        self.assertIn('s3', pivot_points)
        
    def test_calculate_money_flow_index(self):
        """Test Money Flow Index calculation."""
        mfi = calculate_money_flow_index(self.highs, self.lows, self.prices, self.volumes)
        self.assertIsInstance(mfi, float)
        self.assertGreaterEqual(mfi, 0)
        self.assertLessEqual(mfi, 100)
        
    def test_calculate_chaikin_money_flow(self):
        """Test Chaikin Money Flow calculation."""
        cmf = calculate_chaikin_money_flow(self.highs, self.lows, self.prices, self.volumes)
        self.assertIsInstance(cmf, float)
        self.assertGreaterEqual(cmf, -1)
        self.assertLessEqual(cmf, 1)
        
    def test_calculate_obv(self):
        """Test On-Balance Volume calculation."""
        obv = calculate_obv(self.prices, self.volumes)
        self.assertIsInstance(obv, float)
        self.assertNotEqual(obv, 0)
        
    def test_calculate_stochastic(self):
        """Test Stochastic Oscillator calculation."""
        stoch = calculate_stochastic(self.highs, self.lows, self.prices)
        self.assertIsInstance(stoch, dict)
        self.assertIn('k', stoch)
        self.assertIn('d', stoch)
        self.assertGreaterEqual(stoch['k'], 0)
        self.assertLessEqual(stoch['k'], 100)
        self.assertGreaterEqual(stoch['d'], 0)
        self.assertLessEqual(stoch['d'], 100)
        
    def test_calculate_williams_r(self):
        """Test Williams %R calculation."""
        highs = [100, 105, 103, 107, 110, 108, 112, 115, 113, 116, 114, 118, 120, 119, 121]
        lows = [95, 98, 97, 100, 102, 101, 105, 107, 106, 109, 108, 111, 113, 112, 115]
        closes = [97, 102, 100, 105, 107, 106, 110, 112, 111, 114, 113, 116, 118, 117, 119]
        
        williams_r = calculate_williams_r(highs, lows, closes)
        
        self.assertIsInstance(williams_r, dict)
        self.assertIn('value', williams_r)
        self.assertIn('interpretation', williams_r)
        self.assertGreaterEqual(williams_r['value'], -100)
        self.assertLessEqual(williams_r['value'], 0)
        
    def test_calculate_cci(self):
        """Test Commodity Channel Index calculation."""
        cci = calculate_cci(self.highs, self.lows, self.prices)
        
        self.assertIsInstance(cci, dict)
        self.assertIn('value', cci)
        self.assertIn('interpretation', cci)
        self.assertIsInstance(cci['value'], float)
        self.assertIsInstance(cci['interpretation'], str)
        self.assertIn(cci['interpretation'], ['overbought', 'oversold', 'neutral'])
        
    def test_calculate_aroon(self):
        """Test Aroon Oscillator calculation."""
        aroon = calculate_aroon(self.highs, self.lows)
        self.assertIsInstance(aroon, dict)
        self.assertIn('up', aroon)
        self.assertIn('down', aroon)
        self.assertIn('oscillator', aroon)
        self.assertGreaterEqual(aroon['up'], 0)
        self.assertLessEqual(aroon['up'], 100)
        self.assertGreaterEqual(aroon['down'], 0)
        self.assertLessEqual(aroon['down'], 100)
        self.assertEqual(aroon['oscillator'], aroon['up'] - aroon['down'])
        
    def test_calculate_volume_profile(self):
        """Test Volume Profile calculation."""
        volume_profile = calculate_volume_profile(self.highs, self.lows, self.prices, self.volumes)
        self.assertIsInstance(volume_profile, dict)
        self.assertIn('poc', volume_profile)
        self.assertIn('value_area_high', volume_profile)
        self.assertIn('value_area_low', volume_profile)
        
    def test_calculate_market_profile(self):
        """Test Market Profile calculation."""
        market_profile = calculate_market_profile(self.highs, self.lows, self.prices, self.volumes)
        self.assertIsInstance(market_profile, dict)
        self.assertIn('value_area', market_profile)
        self.assertIn('poc', market_profile)
        
    @patch('finnhub.Client')
    def test_get_real_time_data(self, mock_finnhub):
        """Test real-time data retrieval."""
        mock_client = MagicMock()
        mock_client.quote.return_value = MOCK_QUOTE_RESPONSE
        mock_client.stock_candles.return_value = MOCK_CANDLES_RESPONSE
        mock_finnhub.return_value = mock_client
        
        data = get_real_time_data(self.symbol)
        self.assertIsNotNone(data)
        self.assertIn('current_price', data)
        self.assertIn('change', data)
        self.assertIn('percent_change', data)
        self.assertIn('high', data)
        self.assertIn('low', data)
        self.assertIn('open', data)
        self.assertIn('previous_close', data)
        self.assertIn('volume', data)
        self.assertIn('timestamp', data)
        self.assertIn('technical_indicators', data)
        
    @patch('technical_analysis.get_real_time_data')
    def test_generate_analysis_report(self, mock_data):
        """Test analysis report generation."""
        mock_data.return_value = MOCK_REAL_TIME_DATA
        
        report = generate_analysis_report()
        self.assertIsNotNone(report)
        self.assertIn('timestamp', report)
        self.assertIn('top_suggestions', report)
        self.assertIn('market_summary', report)

if __name__ == '__main__':
    unittest.main() 