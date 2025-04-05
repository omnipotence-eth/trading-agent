from technical_analysis import (
    generate_analysis_report, get_real_time_data, calculate_volatility,
    calculate_rsi, calculate_macd, calculate_bollinger_bands, calculate_adx,
    calculate_ichimoku, calculate_fibonacci_levels, calculate_pivot_points,
    calculate_money_flow_index, calculate_chaikin_money_flow, calculate_obv,
    calculate_stochastic, calculate_williams_r, calculate_cci, calculate_aroon,
    calculate_volume_profile, calculate_market_profile
)
from datetime import datetime, timedelta
import json
from typing import Dict, List, Optional, Any, Union, Tuple
from logger import setup_logger
import time
import math
import numpy as np
from scipy import stats
import pandas as pd
import requests
from config import FINNHUB_API_KEY
import finnhub
import os
import concurrent.futures
from functools import lru_cache

logger = setup_logger()

# Cache for trading suggestions with a 15-minute TTL
_suggestions_cache = {}
_cache_ttl = 900  # 15 minutes in seconds
_max_cache_size = 100  # Maximum number of symbols to cache

def _get_cached_suggestions() -> Optional[Dict[str, Any]]:
    """Get cached trading suggestions if they exist and are not expired."""
    current_time = time.time()
    if 'suggestions' in _suggestions_cache:
        cache_data = _suggestions_cache['suggestions']
        if current_time - cache_data['timestamp'] < _cache_ttl:
            logger.info("Using cached trading suggestions")
            return cache_data['data']
    return None

def _update_suggestions_cache(suggestions: Dict[str, Any]) -> None:
    """Update the trading suggestions cache."""
    _suggestions_cache['suggestions'] = {
        'data': suggestions,
        'timestamp': time.time()
    }

def calculate_strike_prices(current_price: float, volatility: float, sentiment: str) -> Dict[str, float]:
    """
    Calculate strike prices based on current price, volatility, and sentiment.
    Uses a more sophisticated approach based on standard deviation and market sentiment.
    
    Formula:
    Upper Strike = Current Price + (Z * Daily Standard Deviation * Current Price)
    Lower Strike = Current Price - (Z * Daily Standard Deviation * Current Price)
    
    where:
    - Z is the Z-score based on sentiment (higher for stronger sentiment)
    - Daily Standard Deviation = Annualized Volatility / sqrt(252)
    """
    # Convert annualized volatility to daily standard deviation
    daily_std = volatility / math.sqrt(252)
    
    # Determine Z-score based on sentiment
    if sentiment == "bullish":
        z_score = 1.5  # More aggressive for bullish sentiment
    elif sentiment == "bearish":
        z_score = 1.5  # More aggressive for bearish sentiment
    else:
        z_score = 1.0  # Neutral sentiment
        
    # Calculate strike distances
    strike_distance = z_score * daily_std * current_price
    
    # Calculate strikes
    upper_strike = current_price + strike_distance
    lower_strike = current_price - strike_distance
    
    return {
        'buy_strike': round(lower_strike, 2),
        'sell_strike': round(upper_strike, 2)
    }

def calculate_position_size(account_size: float, risk_per_trade: float, stop_loss: float, current_price: float) -> int:
    """
    Calculate position size based on risk management principles.
    
    Formula:
    Position Size = (Account Size * Risk Per Trade) / (Stop Loss Distance * Current Price)
    
    where:
    - Account Size is the total trading capital
    - Risk Per Trade is the maximum percentage of account to risk per trade
    - Stop Loss Distance is the distance from entry to stop loss
    """
    risk_amount = account_size * (risk_per_trade / 100)
    stop_loss_distance = abs(current_price - stop_loss)
    
    if stop_loss_distance <= 0:
        return 1  # Minimum position size
        
    position_size = max(1, int(risk_amount / (stop_loss_distance * current_price)))
    return position_size

def calculate_risk_reward_ratio(entry_price: float, target_price: float, stop_loss: float) -> float:
    """
    Calculate the risk-reward ratio for a trade.
    
    Formula:
    Risk-Reward Ratio = (Target Price - Entry Price) / (Entry Price - Stop Loss)
    """
    if entry_price == stop_loss:
        return 0.0
        
    reward = abs(target_price - entry_price)
    risk = abs(entry_price - stop_loss)
    
    return round(reward / risk, 2)

def generate_dynamic_options_strategies(symbol: str, current_price: float, sentiment: str, volatility: float, technical_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Generate dynamic options strategies based on market data and technical indicators.
    Uses a more sophisticated approach incorporating multiple technical indicators and market conditions.
    """
    strategies = []
    
    # Calculate strike prices
    strike_prices = calculate_strike_prices(current_price, volatility, sentiment)
    
    # Get technical indicators with default values
    rsi = technical_data.get('rsi', 50)
    macd = technical_data.get('macd', {})
    macd_value = macd.get('macd', 0)
    macd_signal = macd.get('signal', 0)
    adx = technical_data.get('adx', 25)  # Default to trend threshold
    mfi = technical_data.get('mfi', 50)
    stochastic_k = technical_data.get('stochastic', {}).get('k', 50)
    stochastic_d = technical_data.get('stochastic', {}).get('d', 50)
    williams_r = technical_data.get('williams_r', -50)
    cci = technical_data.get('cci', 0)
    aroon_up = technical_data.get('aroon', {}).get('up', 50)
    aroon_down = technical_data.get('aroon', {}).get('down', 50)
    
    # Always generate at least one strategy based on sentiment
    if sentiment == "bearish":
        strategies.append({
            "strategy": "Bear Put Spread",
            "description": "Bearish strategy with defined risk",
            "entry": {
                "buy_put": {
                    "strike": strike_prices['sell_strike'],
                    "expiration": "30-45 days",
                    "position_size": calculate_position_size(100000, 1, strike_prices['sell_strike'] * 1.05, current_price)
                },
                "sell_put": {
                    "strike": strike_prices['buy_strike'],
                    "expiration": "30-45 days",
                    "position_size": calculate_position_size(100000, 1, strike_prices['buy_strike'] * 0.95, current_price)
                }
            },
            "exit_conditions": {
                "profit_target": strike_prices['sell_strike'] * 0.95,
                "stop_loss": strike_prices['sell_strike'] * 1.05,
                "risk_reward_ratio": calculate_risk_reward_ratio(current_price, strike_prices['sell_strike'] * 0.95, strike_prices['sell_strike'] * 1.05)
            }
        })
    elif sentiment == "bullish":
        strategies.append({
            "strategy": "Bull Call Spread",
            "description": "Bullish strategy with defined risk",
            "entry": {
                "buy_call": {
                    "strike": strike_prices['buy_strike'],
                    "expiration": "30-45 days",
                    "position_size": calculate_position_size(100000, 1, strike_prices['buy_strike'] * 0.95, current_price)
                },
                "sell_call": {
                    "strike": strike_prices['sell_strike'],
                    "expiration": "30-45 days",
                    "position_size": calculate_position_size(100000, 1, strike_prices['sell_strike'] * 1.05, current_price)
                }
            },
            "exit_conditions": {
                "profit_target": strike_prices['buy_strike'] * 1.05,
                "stop_loss": strike_prices['buy_strike'] * 0.95,
                "risk_reward_ratio": calculate_risk_reward_ratio(current_price, strike_prices['buy_strike'] * 1.05, strike_prices['buy_strike'] * 0.95)
            }
        })
    
    # Add Iron Condor for high volatility scenarios
    if volatility > 30:
        strategies.append({
            "strategy": "Iron Condor",
            "description": "Neutral strategy for high volatility",
            "entry": {
                "sell_put": {
                    "strike": strike_prices['buy_strike'] * 0.90,
                    "expiration": "30-45 days",
                    "position_size": calculate_position_size(100000, 1, strike_prices['buy_strike'] * 0.85, current_price)
                },
                "buy_put": {
                    "strike": strike_prices['buy_strike'] * 0.85,
                    "expiration": "30-45 days",
                    "position_size": calculate_position_size(100000, 1, strike_prices['buy_strike'] * 0.80, current_price)
                },
                "sell_call": {
                    "strike": strike_prices['sell_strike'] * 1.10,
                    "expiration": "30-45 days",
                    "position_size": calculate_position_size(100000, 1, strike_prices['sell_strike'] * 1.05, current_price)
                },
                "buy_call": {
                    "strike": strike_prices['sell_strike'] * 1.05,
                    "expiration": "30-45 days",
                    "position_size": calculate_position_size(100000, 1, strike_prices['sell_strike'] * 1.00, current_price)
                }
            },
            "exit_conditions": {
                "profit_target": current_price * 1.02,
                "stop_loss": current_price * 0.98,
                "risk_reward_ratio": calculate_risk_reward_ratio(current_price, current_price * 1.02, current_price * 0.98)
            }
        })
    
    return strategies

def generate_trading_suggestions(symbol: str = "SPY") -> Dict[str, Any]:
    """
    Generate trading suggestions based on technical analysis and market conditions.
    Returns a dictionary containing market analysis, options strategies, and risk management.
    """
    try:
        # Check cache first
        cached_suggestions = _get_cached_suggestions()
        if cached_suggestions:
            return cached_suggestions
            
        # Get real-time data
        data = get_real_time_data(symbol)
        if not data:
            logger.error(f"Failed to get real-time data for {symbol}")
            return {
                "market_conditions": {
                    "trend": "unknown",
                    "strength": "unknown",
                    "volatility": "unknown",
                    "support_levels": [],
                    "resistance_levels": []
                },
                "options_strategies": [],
                "stock_picks": [],
                "risk_management": {
                    "stop_loss": None,
                    "take_profit": None,
                    "position_size": None
                }
            }
            
        # Extract technical indicators
        indicators = data.get('technical_indicators', {})
        current_price = data.get('current_price', 0)
        
        # Calculate volatility and sentiment
        volatility = indicators.get('volatility', {}).get('value', 0.2)
        rsi = indicators.get('rsi', {}).get('value', 50)
        macd = indicators.get('macd', {})
        macd_value = macd.get('macd', 0)
        macd_signal = macd.get('signal', 0)
        
        # Determine market sentiment
        sentiment = "bullish" if rsi > 50 and macd_value > macd_signal else "bearish"
        
        # Calculate strike prices
        strikes = calculate_strike_prices(current_price, volatility, sentiment)
        
        # Generate options strategies
        strategies = generate_dynamic_options_strategies(
            symbol,
            current_price,
            sentiment,
            volatility,
            indicators
        )
        
        # Calculate position size and risk management
        account_size = 100000  # Example account size
        risk_per_trade = 1.0  # 1% risk per trade
        stop_loss = current_price * 0.95 if sentiment == "bullish" else current_price * 1.05
        
        position_size = calculate_position_size(
            account_size,
            risk_per_trade,
            stop_loss,
            current_price
        )
        
        # Prepare suggestions
        suggestions = {
            "market_conditions": {
                "trend": sentiment,
                "strength": "strong" if abs(rsi - 50) > 20 else "weak",
                "volatility": "high" if volatility > 0.3 else "low",
                "support_levels": [strikes['buy_strike']],
                "resistance_levels": [strikes['sell_strike']]
            },
            "options_strategies": strategies,
            "stock_picks": [{
                "symbol": symbol,
                "action": "buy" if sentiment == "bullish" else "sell",
                "entry_price": current_price,
                "stop_loss": stop_loss,
                "take_profit": current_price * 1.1 if sentiment == "bullish" else current_price * 0.9
            }],
            "risk_management": {
                "stop_loss": stop_loss,
                "take_profit": current_price * 1.1 if sentiment == "bullish" else current_price * 0.9,
                "position_size": position_size
            }
        }
        
        # Update cache
        _update_suggestions_cache(suggestions)
        
        return suggestions
        
    except Exception as e:
        logger.error(f"Error generating trading suggestions: {str(e)}")
        return {
            "market_conditions": {
                "trend": "unknown",
                "strength": "unknown",
                "volatility": "unknown",
                "support_levels": [],
                "resistance_levels": []
            },
            "options_strategies": [],
            "stock_picks": [],
            "risk_management": {
                "stop_loss": None,
                "take_profit": None,
                "position_size": None
            }
        }

def format_trading_suggestions_for_twitter(suggestions: Dict[str, Any]) -> str:
    """
    Format trading suggestions for Twitter with proper sections and emojis.
    """
    if not suggestions:
        return "Unable to generate trading suggestions at this time."
        
    # Market Analysis Section (essential)
    market_analysis = f"📊 Market Analysis:\n"
    market_analysis += f"Trend: {suggestions['market_conditions']['trend'].upper()}\n"
    market_analysis += f"Strength: {suggestions['market_conditions']['strength']}\n"
    if 'volatility' in suggestions['market_conditions']:
        market_analysis += f"Volatility: {suggestions['market_conditions']['volatility']}\n"
    
    # Support/Resistance Levels (optional)
    if 'support_levels' in suggestions['market_conditions'] and suggestions['market_conditions']['support_levels']:
        market_analysis += f"Support: {', '.join(map(str, suggestions['market_conditions']['support_levels']))}\n"
    if 'resistance_levels' in suggestions['market_conditions'] and suggestions['market_conditions']['resistance_levels']:
        market_analysis += f"Resistance: {', '.join(map(str, suggestions['market_conditions']['resistance_levels']))}\n"
    
    # Options Strategies Section (essential)
    options_strategies = "🎯 Options Strategies:\n"
    for strategy in suggestions['options_strategies']:
        strategy_text = f"• {strategy['strategy']}: {strategy['description']}\n"
        if len(options_strategies + strategy_text) <= 100:  # Keep within reasonable length
            options_strategies += strategy_text
    
    # Stock Picks Section (optional)
    stock_picks = "💎 Stock Picks:\n"
    for pick in suggestions['stock_picks']:
        pick_text = f"• {pick['symbol']}: {pick['action'].upper()} @ {pick['entry_price']}\n"
        if len(stock_picks + pick_text) <= 80:  # Keep within reasonable length
            stock_picks += pick_text
    
    # Risk Management Section (essential)
    risk_management = "⚠️ Risk Management:\n"
    risk_management += f"Size: {suggestions['risk_management']['position_size']} | "
    risk_management += f"SL: {suggestions['risk_management']['stop_loss']} | "
    risk_management += f"TP: {suggestions['risk_management']['take_profit']}"
    
    # Disclaimer (essential)
    disclaimer = "\n⚠️ DYOR. Not financial advice."
    
    # Build tweet with essential sections first
    tweet = market_analysis + "\n"
    tweet += options_strategies + "\n"  # Make Options Strategies essential
    tweet += risk_management + disclaimer
    
    # Add stock picks if there's room
    if len(tweet) + len(stock_picks) + 2 <= 277:  # Leave room for "..."
        tweet = tweet[:-len(disclaimer)] + "\n" + stock_picks + disclaimer
    
    # Ensure tweet is within Twitter's character limit
    if len(tweet) > 280:
        tweet = tweet[:277] + "..."
    
    return tweet.strip()

if __name__ == "__main__":
    # Test the functions
    suggestions = format_trading_suggestions_for_twitter(generate_trading_suggestions())
    if suggestions:
        print("\nTrading Suggestions:")
        print(suggestions)
    else:
        print("Failed to generate trading suggestions") 