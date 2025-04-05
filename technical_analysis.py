import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import requests
import time
from typing import Dict, List, Optional, Any, Union, Tuple
from config import FINNHUB_API_KEY
from logger import setup_logger
import math
import finnhub
import os
from scipy import stats
from statsmodels.tsa.stattools import adfuller
import json
import concurrent.futures
from functools import lru_cache

logger = setup_logger()

# Cache for API responses with a 5-minute TTL
_api_cache = {}
_cache_ttl = 300  # 5 minutes in seconds
_max_cache_size = 100  # Maximum number of symbols to cache

# Rate limiting settings
_rate_limit = 60  # requests per minute
_request_times = []

def _clean_cache() -> None:
    """Clean expired cache entries."""
    current_time = time.time()
    expired_keys = [k for k, v in _api_cache.items() 
                   if current_time - v['timestamp'] > _cache_ttl]
    for k in expired_keys:
        del _api_cache[k]
    
    # If still too many entries, remove oldest
    if len(_api_cache) > _max_cache_size:
        sorted_cache = sorted(_api_cache.items(), key=lambda x: x[1]['timestamp'])
        for k, _ in sorted_cache[:-(_max_cache_size)]:
            del _api_cache[k]

def _check_rate_limit() -> None:
    """Check and enforce rate limiting."""
    current_time = time.time()
    _request_times.append(current_time)
    
    # Remove requests older than 1 minute
    _request_times[:] = [t for t in _request_times if current_time - t <= 60]
    
    if len(_request_times) > _rate_limit:
        sleep_time = 60 - (current_time - _request_times[0])
        if sleep_time > 0:
            time.sleep(sleep_time)
            return _check_rate_limit()  # Recursive check after sleeping

def calculate_volatility(prices: List[float]) -> Dict[str, float]:
    """
    Calculate volatility using Parkinson's High-Low Range volatility estimator.
    This method is more robust than standard deviation as it:
    1. Captures extreme price movements better
    2. Is less sensitive to outliers
    3. Provides a more accurate estimate of true volatility
    
    Formula:
    σ = sqrt(1/(4*log(2)) * (log(high/low))^2 / n) * sqrt(252) * 100
    
    where:
    - high/low are the highest/lowest prices in the period
    - n is the number of periods
    - 252 is the number of trading days in a year
    
    Returns a dictionary with:
    - annual_vol: Annualized volatility as a percentage
    - daily_vol: Daily volatility as a percentage
    - high: Highest price in the period
    - low: Lowest price in the period
    """
    # Convert to numpy array if not already
    prices_arr = np.array(prices)
    
    if len(prices_arr) < 2:
        return {
            "annual_vol": 0.0,
            "daily_vol": 0.0,
            "high": 0.0,
            "low": 0.0
        }
        
    high = np.max(prices_arr)
    low = np.min(prices_arr)
    
    if low <= 0:
        return {
            "annual_vol": 0.0,
            "daily_vol": 0.0,
            "high": high,
            "low": low
        }
        
    # Calculate daily volatility
    daily_vol = math.sqrt(1/(4*math.log(2)) * (math.log(high/low))**2 / len(prices_arr))
    
    # Annualize and convert to percentage
    annual_vol = daily_vol * math.sqrt(252) * 100
    
    return {
        "annual_vol": round(annual_vol, 2),
        "daily_vol": round(daily_vol * 100, 2),
        "high": round(high, 2),
        "low": round(low, 2)
    }

def calculate_rsi(prices: List[float], period: int = 14) -> float:
    """
    Calculate Relative Strength Index (RSI).
    RSI measures the speed and magnitude of recent price changes to evaluate overbought or oversold conditions.
    """
    if len(prices) < period + 1:
        return 50.0  # Default neutral value
        
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    
    if avg_loss == 0:
        return 100.0
        
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return round(rsi, 2)

def calculate_macd(prices: List[float], fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, float]:
    """
    Calculate Moving Average Convergence Divergence (MACD).
    MACD is a trend-following momentum indicator that shows the relationship between two moving averages.
    """
    if len(prices) < slow + signal:
        return {'macd': 0.0, 'signal': 0.0, 'histogram': 0.0}
        
    exp1 = pd.Series(prices).ewm(span=fast, adjust=False).mean()
    exp2 = pd.Series(prices).ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    
    return {
        'macd': round(macd.iloc[-1], 2),
        'signal': round(signal_line.iloc[-1], 2),
        'histogram': round(histogram.iloc[-1], 2)
    }

def calculate_bollinger_bands(prices: List[float], period: int = 20, std_dev: float = 2.0) -> Dict[str, float]:
    """
    Calculate Bollinger Bands.
    Bollinger Bands consist of a middle band (SMA) and upper/lower bands that are standard deviations away from the middle band.
    """
    if len(prices) < period:
        # For insufficient data, return default values that maintain the expected relationship
        avg_price = sum(prices) / len(prices) if prices else 0.0
        return {
            'upper': avg_price * 1.02,
            'middle': avg_price,
            'lower': avg_price * 0.98
        }
        
    prices_series = pd.Series(prices)
    middle_band = prices_series.rolling(window=period).mean().iloc[-1]
    std = prices_series.rolling(window=period).std().iloc[-1]
    
    upper_band = middle_band + (std * std_dev)
    lower_band = middle_band - (std * std_dev)
    
    # Ensure the bands maintain their relationship
    if not (upper_band > middle_band > lower_band):
        middle_band = prices_series.iloc[-1]
        std = abs(prices_series.pct_change().std() * middle_band)
        upper_band = middle_band + (std * std_dev)
        lower_band = middle_band - (std * std_dev)
    
    return {
        'upper': round(upper_band, 2),
        'middle': round(middle_band, 2),
        'lower': round(lower_band, 2)
    }

def calculate_adx(high: List[float], low: List[float], close: List[float], period: int = 14) -> float:
    """
    Calculate Average Directional Index (ADX).
    ADX measures trend strength regardless of direction.
    """
    if len(close) < period + 1:
        return 0.0
        
    # Calculate True Range
    tr = []
    for i in range(1, len(close)):
        tr.append(max(
            high[i] - low[i],
            abs(high[i] - close[i-1]),
            abs(low[i] - close[i-1])
        ))
    
    # Calculate Directional Movement
    plus_dm = []
    minus_dm = []
    for i in range(1, len(close)):
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        
        if up_move > down_move and up_move > 0:
            plus_dm.append(up_move)
        else:
            plus_dm.append(0)
            
        if down_move > up_move and down_move > 0:
            minus_dm.append(down_move)
        else:
            minus_dm.append(0)
    
    # Calculate ADX
    tr_smooth = pd.Series(tr).rolling(window=period).mean()
    plus_di = 100 * pd.Series(plus_dm).rolling(window=period).mean() / tr_smooth
    minus_di = 100 * pd.Series(minus_dm).rolling(window=period).mean() / tr_smooth
    
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(window=period).mean()
    
    return round(adx.iloc[-1], 2)

def calculate_ichimoku(high: List[float], low: List[float], close: List[float]) -> Dict[str, float]:
    """
    Calculate Ichimoku Cloud indicators.
    Ichimoku Cloud is a comprehensive trend analysis tool that identifies support/resistance levels and trend direction.
    """
    if len(close) < 52:  # Need at least 52 periods for all calculations
        return {
            "tenkan": 0.0,
            "kijun": 0.0,
            "senkou_a": 0.0,
            "senkou_b": 0.0,
            "chikou": 0.0
        }
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan = (max(high[-9:]) + min(low[-9:])) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun = (max(high[-26:]) + min(low[-26:])) / 2
    
    # Senkou Span A (Leading Span A): (Conversion Line + Base Line)/2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
    senkou_b = (max(high[-52:]) + min(low[-52:])) / 2
    
    # Chikou Span (Lagging Span): Close price shifted back 26 periods
    chikou = close[-26] if len(close) >= 26 else close[0]
    
    return {
        "tenkan": round(tenkan, 2),
        "kijun": round(kijun, 2),
        "senkou_a": round(senkou_a, 2),
        "senkou_b": round(senkou_b, 2),
        "chikou": round(chikou, 2)
    }

def calculate_fibonacci_levels(high: float, low: float) -> Dict[str, float]:
    """
    Calculate Fibonacci retracement and extension levels.
    Fibonacci levels are used to identify potential support and resistance levels.
    """
    diff = high - low
    
    return {
        "0.0": round(low, 2),
        "0.236": round(low + diff * 0.236, 2),
        "0.382": round(low + diff * 0.382, 2),
        "0.5": round(low + diff * 0.5, 2),
        "0.618": round(low + diff * 0.618, 2),
        "0.786": round(low + diff * 0.786, 2),
        "1.0": round(high, 2),
        "1.618": round(high + diff * 0.618, 2),
        "2.618": round(high + diff * 1.618, 2)
    }

def calculate_pivot_points(high: float, low: float, close: float) -> Dict[str, float]:
    """
    Calculate pivot points and support/resistance levels.
    Pivot points are used to identify potential support and resistance levels.
    """
    pp = (high + low + close) / 3
    
    r1 = 2 * pp - low
    s1 = 2 * pp - high
    r2 = pp + (high - low)
    s2 = pp - (high - low)
    r3 = high + 2 * (pp - low)
    s3 = low - 2 * (high - pp)
    
    return {
        "pp": round(pp, 2),
        "r1": round(r1, 2),
        "s1": round(s1, 2),
        "r2": round(r2, 2),
        "s2": round(s2, 2),
        "r3": round(r3, 2),
        "s3": round(s3, 2)
    }

def calculate_money_flow_index(high: List[float], low: List[float], close: List[float], volume: List[float], period: int = 14) -> float:
    """
    Calculate Money Flow Index (MFI).
    MFI is a volume-weighted RSI that measures buying and selling pressure.
    """
    if len(close) < period + 1 or len(volume) < period + 1:
        return 50.0  # Default neutral value
        
    # Calculate typical price
    typical_price = [(h + l + c) / 3 for h, l, c in zip(high, low, close)]
    
    # Calculate money flow
    money_flow = [tp * vol for tp, vol in zip(typical_price, volume)]
    
    # Calculate positive and negative money flow
    positive_flow = []
    negative_flow = []
    
    for i in range(1, len(typical_price)):
        if typical_price[i] > typical_price[i-1]:
            positive_flow.append(money_flow[i])
            negative_flow.append(0)
        else:
            positive_flow.append(0)
            negative_flow.append(money_flow[i])
    
    # Calculate MFI
    positive_mf = pd.Series(positive_flow).rolling(window=period).sum()
    negative_mf = pd.Series(negative_flow).rolling(window=period).sum()
    
    mfi = 100 - (100 / (1 + positive_mf / negative_mf))
    
    return round(mfi.iloc[-1], 2)

def calculate_chaikin_money_flow(high: List[float], low: List[float], close: List[float], volume: List[float], period: int = 20) -> float:
    """
    Calculate Chaikin Money Flow (CMF).
    CMF measures buying and selling pressure based on volume and price action.
    """
    if len(close) < period or len(volume) < period:
        return 0.0  # Default neutral value
        
    # Calculate Money Flow Multiplier
    mfm = [(close[i] - low[i] - (high[i] - close[i])) / (high[i] - low[i]) if high[i] != low[i] else 0 
           for i in range(len(close))]
    
    # Calculate Money Flow Volume
    mfv = [mfm[i] * volume[i] for i in range(len(close))]
    
    # Calculate CMF
    cmf = pd.Series(mfv).rolling(window=period).sum() / pd.Series(volume).rolling(window=period).sum()
    
    return round(cmf.iloc[-1], 2)

def calculate_obv(close: List[float], volume: List[float]) -> float:
    """
    Calculate On-Balance Volume (OBV).
    OBV is a cumulative volume indicator that relates volume to price change.
    """
    if len(close) < 2 or len(volume) < 2:
        return 0.0
        
    obv = [0.0]  # Initialize with float
    for i in range(1, len(close)):
        if close[i] > close[i-1]:
            obv.append(obv[-1] + float(volume[i]))
        elif close[i] < close[i-1]:
            obv.append(obv[-1] - float(volume[i]))
        else:
            obv.append(obv[-1])
    
    # Return the rate of change of OBV
    obv_roc = (obv[-1] - obv[-min(20, len(obv))]) / abs(obv[-min(20, len(obv))] + 1e-10) * 100.0
    
    return round(obv_roc, 2)

def calculate_stochastic(high: List[float], low: List[float], close: List[float], k_period: int = 14, d_period: int = 3) -> Dict[str, float]:
    """
    Calculate Stochastic Oscillator.
    The Stochastic Oscillator is a momentum indicator comparing a particular closing price to a range of prices over time.
    """
    if len(close) < k_period:
        return {'k': 50.0, 'd': 50.0}  # Default neutral values
        
    # Calculate %K
    low_min = pd.Series(low).rolling(window=k_period).min()
    high_max = pd.Series(high).rolling(window=k_period).max()
    
    k = 100 * ((pd.Series(close) - low_min) / (high_max - low_min))
    
    # Calculate %D
    d = k.rolling(window=d_period).mean()
    
    return {
        'k': round(k.iloc[-1], 2),
        'd': round(d.iloc[-1], 2)
    }

def calculate_williams_r(high: List[float], low: List[float], close: List[float], period: int = 14) -> Dict[str, float]:
    """
    Calculate Williams %R.
    Williams %R measures overbought and oversold levels similar to the Stochastic Oscillator.
    Returns a dictionary with the Williams %R value and its interpretation.
    """
    if len(close) < period:
        return {
            "value": -50.0,  # Default neutral value
            "interpretation": "neutral"
        }
        
    highest_high = pd.Series(high).rolling(window=period).max()
    lowest_low = pd.Series(low).rolling(window=period).min()
    
    wr = -100 * (highest_high - pd.Series(close)) / (highest_high - lowest_low)
    wr_value = round(wr.iloc[-1], 2)
    
    # Interpret the Williams %R value
    if wr_value <= -80:
        interpretation = "oversold"
    elif wr_value >= -20:
        interpretation = "overbought"
    else:
        interpretation = "neutral"
    
    return {
        "value": wr_value,
        "interpretation": interpretation
    }

def calculate_cci(high: List[float], low: List[float], close: List[float], period: int = 20) -> Dict[str, Any]:
    """
    Calculate Commodity Channel Index (CCI).
    CCI measures the current price level relative to an average price level over a given period.
    Returns a dictionary with the CCI value and its interpretation.
    """
    if len(close) < period:
        return {
            "value": 0.0,  # Default neutral value
            "interpretation": "neutral"
        }
        
    # Calculate Typical Price
    tp = [(h + l + c) / 3 for h, l, c in zip(high, low, close)]
    
    # Calculate Simple Moving Average of Typical Price
    sma_tp = pd.Series(tp).rolling(window=period).mean()
    
    # Calculate Mean Deviation
    md = pd.Series([abs(tp[i] - sma_tp[i]) for i in range(len(tp))]).rolling(window=period).mean()
    
    # Calculate CCI
    cci = (pd.Series(tp) - sma_tp) / (0.015 * md)
    cci_value = round(cci.iloc[-1], 2)
    
    # Interpret the CCI value
    if cci_value > 100:
        interpretation = "overbought"
    elif cci_value < -100:
        interpretation = "oversold"
    else:
        interpretation = "neutral"
    
    return {
        "value": cci_value,
        "interpretation": interpretation
    }

def calculate_aroon(high: List[float], low: List[float], period: int = 25) -> Dict[str, float]:
    """
    Calculate Aroon Oscillator.
    Aroon Oscillator measures the time between highs and lows over a time period.
    Returns a dictionary with 'up' and 'down' values.
    """
    if len(high) < period or len(low) < period:
        return {"up": 0.0, "down": 0.0}  # Default neutral values
        
    # Convert to numpy arrays for efficient computation
    high_arr = np.array(high)
    low_arr = np.array(low)
    
    # Calculate Aroon Up
    aroon_up = []
    for i in range(period, len(high) + 1):
        high_period = high_arr[i-period:i]
        days_since_high = period - np.argmax(high_period) - 1
        aroon_up.append(((period - days_since_high) / period) * 100)
    
    # Calculate Aroon Down
    aroon_down = []
    for i in range(period, len(low) + 1):
        low_period = low_arr[i-period:i]
        days_since_low = period - np.argmin(low_period) - 1
        aroon_down.append(((period - days_since_low) / period) * 100)
    
    # Calculate Aroon Oscillator
    aroon_oscillator = [up - down for up, down in zip(aroon_up, aroon_down)]
    
    return {
        "up": round(aroon_up[-1], 2),
        "down": round(aroon_down[-1], 2),
        "oscillator": round(aroon_oscillator[-1], 2)
    }

def calculate_volume_profile(high: List[float], low: List[float], close: List[float], volume: List[float], num_bins: int = 10) -> Dict[str, float]:
    """
    Calculate Volume Profile.
    Volume Profile shows the distribution of volume at different price levels.
    """
    if len(close) < 2 or len(volume) < 2:
        return {"poc": 0.0, "value_area_high": 0.0, "value_area_low": 0.0}
        
    # Create price bins
    price_min = min(low)
    price_max = max(high)
    bin_size = (price_max - price_min) / num_bins
    
    # Initialize volume profile
    volume_profile = {i: 0 for i in range(num_bins)}
    
    # Calculate volume profile
    for i in range(len(close)):
        price = close[i]
        bin_index = min(int((price - price_min) / bin_size), num_bins - 1)
        volume_profile[bin_index] += volume[i]
    
    # Find Point of Control (POC)
    poc_bin = max(volume_profile, key=volume_profile.get)
    poc_price = price_min + (poc_bin + 0.5) * bin_size
    
    # Calculate Value Area (70% of volume)
    total_volume = sum(volume_profile.values())
    value_area_volume = total_volume * 0.7
    
    # Sort bins by volume
    sorted_bins = sorted(volume_profile.items(), key=lambda x: x[1], reverse=True)
    
    # Find Value Area
    value_area_volume_sum = 0
    value_area_bins = []
    
    for bin_index, bin_volume in sorted_bins:
        value_area_volume_sum += bin_volume
        value_area_bins.append(bin_index)
        
        if value_area_volume_sum >= value_area_volume:
            break
    
    value_area_high = price_min + (max(value_area_bins) + 1) * bin_size
    value_area_low = price_min + min(value_area_bins) * bin_size
    
    return {
        "poc": round(poc_price, 2),
        "value_area_high": round(value_area_high, 2),
        "value_area_low": round(value_area_low, 2)
    }

def calculate_market_profile(high: List[float], low: List[float], close: List[float], volume: List[float], num_bins: int = 10) -> Dict[str, Any]:
    """
    Calculate Market Profile.
    Market Profile shows the distribution of price action over time.
    """
    if len(close) < 2 or len(volume) < 2:
        return {"value_area": {"high": 0.0, "low": 0.0}, "poc": 0.0}
        
    # Create price bins
    price_min = min(low)
    price_max = max(high)
    bin_size = (price_max - price_min) / num_bins
    
    # Initialize market profile
    market_profile = {i: {"volume": 0, "time": 0} for i in range(num_bins)}
    
    # Calculate market profile
    for i in range(len(close)):
        price = close[i]
        bin_index = min(int((price - price_min) / bin_size), num_bins - 1)
        market_profile[bin_index]["volume"] += volume[i]
        market_profile[bin_index]["time"] += 1
    
    # Find Point of Control (POC)
    poc_bin = max(market_profile, key=lambda x: market_profile[x]["volume"])
    poc_price = price_min + (poc_bin + 0.5) * bin_size
    
    # Calculate Value Area (70% of volume)
    total_volume = sum(profile["volume"] for profile in market_profile.values())
    value_area_volume = total_volume * 0.7
    
    # Sort bins by volume
    sorted_bins = sorted(market_profile.items(), key=lambda x: x[1]["volume"], reverse=True)
    
    # Find Value Area
    value_area_volume_sum = 0
    value_area_bins = []
    
    for bin_index, bin_data in sorted_bins:
        value_area_volume_sum += bin_data["volume"]
        value_area_bins.append(bin_index)
        
        if value_area_volume_sum >= value_area_volume:
            break
    
    value_area_high = price_min + (max(value_area_bins) + 1) * bin_size
    value_area_low = price_min + min(value_area_bins) * bin_size
    
    return {
        "value_area": {
            "high": round(value_area_high, 2),
            "low": round(value_area_low, 2)
        },
        "poc": round(poc_price, 2)
    }

@lru_cache(maxsize=100)
def get_real_time_data(symbol: str) -> Optional[Dict[str, Any]]:
    """Get real-time stock data with caching and rate limiting."""
    try:
        # Check cache first
        current_time = time.time()
        if symbol in _api_cache:
            cache_data = _api_cache[symbol]
            if current_time - cache_data['timestamp'] < _cache_ttl:
                logger.info(f"Using cached data for {symbol}")
                return cache_data['data']
        
        # Clean cache if needed
        _clean_cache()
        
        # Check rate limit
        _check_rate_limit()
        
        # Initialize Finnhub client
        finnhub_client = finnhub.Client(api_key=os.getenv('FINNHUB_API_KEY'))
        
        # Get quote data
        quote = finnhub_client.quote(symbol)
        if not quote or 'c' not in quote:
            logger.error(f"Failed to get quote data for {symbol}")
            return None
        
        # Get additional technical indicators
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        # Get historical data for technical analysis
        hist_data = finnhub_client.stock_candles(
            symbol,
            'D',
            int(start_date.timestamp()),
            int(end_date.timestamp())
        )
        
        if hist_data['s'] != 'ok' or not hist_data['c'] or len(hist_data['c']) < 14:
            logger.error(f"Failed to get historical data for {symbol}")
            return None
            
        # Calculate technical indicators
        closes = hist_data['c']
        highs = hist_data['h']
        lows = hist_data['l']
        volumes = hist_data['v'] if 'v' in hist_data else [0] * len(closes)
        
        # Calculate basic indicators
        volatility = calculate_volatility(closes)
        rsi = calculate_rsi(closes)
        macd = calculate_macd(closes)
        bollinger_bands = calculate_bollinger_bands(closes)
        adx = calculate_adx(highs, lows, closes)
        ichimoku = calculate_ichimoku(highs, lows, closes)
        fib_levels = calculate_fibonacci_levels(max(highs), min(lows))
        pivot_points = calculate_pivot_points(highs[-1], lows[-1], closes[-1])
        
        # Calculate advanced indicators
        mfi = calculate_money_flow_index(highs, lows, closes, volumes)
        cmf = calculate_chaikin_money_flow(highs, lows, closes, volumes)
        obv = calculate_obv(closes, volumes)
        stochastic = calculate_stochastic(highs, lows, closes)
        williams_r = calculate_williams_r(highs, lows, closes)
        cci = calculate_cci(highs, lows, closes)
        aroon = calculate_aroon(highs, lows)
        
        # Calculate volume and market profile
        volume_profile = calculate_volume_profile(highs, lows, closes, volumes)
        market_profile = calculate_market_profile(highs, lows, closes, volumes)
        
        # Prepare response
        data = {
            "current_price": quote['c'],
            "change": quote['d'],
            "percent_change": quote['dp'],
            "high": quote['h'],
            "low": quote['l'],
            "open": quote['o'],
            "previous_close": quote['pc'],
            "volume": quote.get('v', 0),
            "timestamp": datetime.now(),
            "technical_indicators": {
                "volatility": volatility,
                "rsi": rsi,
                "macd": macd,
                "bollinger_bands": bollinger_bands,
                "adx": adx,
                "ichimoku": ichimoku,
                "fibonacci": fib_levels,
                "pivot_points": pivot_points,
                "mfi": mfi,
                "cmf": cmf,
                "obv": obv,
                "stochastic": stochastic,
                "williams_r": williams_r,
                "cci": cci,
                "aroon": aroon,
                "volume_profile": volume_profile,
                "market_profile": market_profile
            }
        }
        
        # Update cache
        _api_cache[symbol] = {
            'data': data,
            'timestamp': current_time
        }
        
        return data
        
    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {str(e)}")
        return None

def analyze_stock(symbol: str) -> Optional[Dict[str, Any]]:
    """Perform comprehensive stock analysis using real-time data."""
    try:
        data = get_real_time_data(symbol)
        if not data:
            return None
            
        # Calculate basic metrics
        current_price = data['current_price']
        previous_close = data['previous_close']
        high = data['high']
        low = data['low']
        volume = data.get('volume', 0)
        
        # Get technical indicators
        indicators = data['technical_indicators']
        
        # Initialize analysis dict with default values
        analysis = {
            "symbol": symbol,
            "current_price": current_price,
            "previous_close": previous_close,
            "price_change": None,
            "change_percent": None,
            "high": high,
            "low": low,
            "volume": volume,
            "volatility": indicators['volatility'],
            "trend": "neutral",
            "volume_trend": "unknown",  # Default to unknown if no volume data
            "timestamp": data['timestamp'].strftime("%Y-%m-%d %H:%M:%S"),
            "technical_indicators": {
                "rsi": indicators['rsi'],
                "macd": indicators['macd'],
                "bollinger_bands": indicators['bollinger_bands'],
                "adx": indicators['adx'],
                "mfi": indicators['mfi'],
                "stochastic": indicators['stochastic'],
                "williams_r": indicators['williams_r'],
                "cci": indicators['cci'],
                "aroon": indicators['aroon']
            }
        }
        
        # Calculate price change and percentage
        if current_price is not None and previous_close is not None and previous_close != 0:
            price_change = current_price - previous_close
            change_percent = (price_change / previous_close) * 100
            analysis.update({
                "price_change": price_change,
                "change_percent": change_percent
            })
            
        # Determine trend using multiple indicators
        trend_signals = []
        
        # Price trend
        if current_price is not None and previous_close is not None:
            price_trend = "bullish" if current_price > previous_close else "bearish" if current_price < previous_close else "neutral"
            trend_signals.append(price_trend)
            
        # RSI trend
        rsi = indicators['rsi'].get('value', 50)
        if rsi > 70:
            trend_signals.append("bearish")  # Overbought
        elif rsi < 30:
            trend_signals.append("bullish")  # Oversold
        else:
            trend_signals.append("neutral")
            
        # MACD trend
        macd = indicators['macd']['macd']
        signal = indicators['macd']['signal']
        if macd > signal:
            trend_signals.append("bullish")
        elif macd < signal:
            trend_signals.append("bearish")
        else:
            trend_signals.append("neutral")
            
        # Stochastic trend
        k = indicators['stochastic']['k']
        d = indicators['stochastic']['d']
        if k > 80 and d > 80:
            trend_signals.append("bearish")  # Overbought
        elif k < 20 and d < 20:
            trend_signals.append("bullish")  # Oversold
        else:
            trend_signals.append("neutral")
            
        # Determine overall trend
        bullish_count = trend_signals.count("bullish")
        bearish_count = trend_signals.count("bearish")
        
        if bullish_count > bearish_count:
            analysis["trend"] = "bullish"
        elif bearish_count > bullish_count:
            analysis["trend"] = "bearish"
        else:
            analysis["trend"] = "neutral"
            
        # Volume analysis
        if volume > 0:
            volume_trend = "increasing" if volume > 1000000 else "decreasing" if volume < 500000 else "stable"
            analysis["volume_trend"] = volume_trend
        
        return analysis
    except Exception as e:
        logger.error(f"Error analyzing stock {symbol}: {str(e)}")
        return None

def get_market_suggestions() -> Optional[List[Dict[str, Any]]]:
    """Generate market suggestions based on real-time analysis."""
    try:
        # List of stocks to analyze
        symbols = [
            "SPY", "QQQ", "DIA",  # Major indices
            "AAPL", "MSFT", "GOOGL", "AMZN", "META",  # Tech giants
            "XOM", "CVX", "COP",  # Energy
            "JPM", "BAC", "GS",  # Financials
            "JNJ", "PFE", "MRK"  # Healthcare
        ]
        
        # Use concurrent execution for faster analysis
        suggestions = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            future_to_symbol = {executor.submit(analyze_stock, symbol): symbol for symbol in symbols}
            
            for future in concurrent.futures.as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    analysis = future.result()
                    if analysis:
                        # Score the stock based on real-time metrics
                        score = 0
                        
                        # Trend analysis (0-3 points)
                        if analysis['trend'] == "bullish":
                            score += 3
                        elif analysis['trend'] == "neutral":
                            score += 1
                            
                        # Volume analysis (0-2 points)
                        if analysis['volume_trend'] == "increasing":
                            score += 2
                        elif analysis['volume_trend'] == "stable":
                            score += 1
                            
                        # Volatility analysis (0-2 points)
                        if analysis['volatility'] is not None:
                            if 1 <= analysis['volatility'] <= 5:  # Healthy volatility
                                score += 2
                            elif 5 < analysis['volatility'] <= 10:  # Moderate volatility
                                score += 1
                                
                        # Price change analysis (0-3 points)
                        if analysis['change_percent'] is not None:
                            if 0 < analysis['change_percent'] <= 2:  # Healthy gain
                                score += 3
                            elif -2 <= analysis['change_percent'] < 0:  # Small loss
                                score += 1
                            elif analysis['change_percent'] > 2:  # Strong gain
                                score += 2
                                
                        # Technical indicator analysis (0-5 points)
                        indicators = analysis.get('technical_indicators', {})
                        
                        # RSI analysis
                        rsi = indicators.get('rsi', 50)
                        if 30 <= rsi <= 70:  # Healthy RSI
                            score += 1
                            
                        # MACD analysis
                        macd = indicators.get('macd', {}).get('macd', 0)
                        signal = indicators.get('macd', {}).get('signal', 0)
                        if macd > signal:  # Bullish MACD
                            score += 1
                            
                        # Stochastic analysis
                        k = indicators.get('stochastic', {}).get('k', 50)
                        d = indicators.get('stochastic', {}).get('d', 50)
                        if 20 <= k <= 80 and 20 <= d <= 80:  # Healthy stochastic
                            score += 1
                            
                        # ADX analysis
                        adx = indicators.get('adx', 0)
                        if adx > 25:  # Strong trend
                            score += 1
                            
                        # Aroon analysis
                        aroon_up = indicators.get('aroon', {}).get('up', 50)
                        aroon_down = indicators.get('aroon', {}).get('down', 50)
                        if aroon_up > aroon_down:  # Bullish aroon
                            score += 1
                        
                        suggestions.append({
                            "symbol": symbol,
                            "score": score,
                            "analysis": analysis
                        })
                except Exception as e:
                    logger.error(f"Error processing {symbol}: {str(e)}")
        
        # Sort by score and return top suggestions
        suggestions.sort(key=lambda x: x['score'], reverse=True)
        return suggestions[:5]  # Return top 5 suggestions
        
    except Exception as e:
        logger.error(f"Error generating market suggestions: {str(e)}")
        return None

def generate_analysis_report() -> Optional[Dict[str, Any]]:
    """Generate a comprehensive market analysis report."""
    try:
        # List of stocks to analyze
        symbols = [
            "SPY", "QQQ", "DIA",  # Major indices
            "AAPL", "MSFT", "GOOGL", "AMZN", "META",  # Tech giants
            "XOM", "CVX", "COP",  # Energy
            "JPM", "BAC", "GS",  # Financials
            "JNJ", "PFE", "MRK"  # Healthcare
        ]
        
        # Analyze each stock
        analyses = []
        for symbol in symbols:
            analysis = analyze_stock(symbol)
            if analysis:
                # Calculate score based on technical indicators
                score = 0.0
                indicators = analysis.get('technical_indicators', {})
                
                # RSI score (0-1)
                rsi = indicators.get('rsi', {}).get('value', 50)
                if 40 <= rsi <= 60:
                    score += 0.5  # Neutral
                elif 30 <= rsi <= 70:
                    score += 0.3  # Moderate
                
                # MACD score (0-1)
                macd = indicators.get('macd', {})
                if macd.get('macd', 0) > macd.get('signal', 0):
                    score += 0.5  # Bullish
                
                # Stochastic score (0-1)
                stoch = indicators.get('stochastic', {})
                k = stoch.get('k', 50)
                d = stoch.get('d', 50)
                if 20 <= k <= 80 and 20 <= d <= 80:
                    score += 0.5  # Healthy range
                
                # Williams %R score (0-1)
                williams = indicators.get('williams_r', {}).get('value', -50)
                if -80 <= williams <= -20:
                    score += 0.5  # Neutral range
                
                # CCI score (0-1)
                cci = indicators.get('cci', {}).get('value', 0)
                if -100 <= cci <= 100:
                    score += 0.5  # Normal range
                
                # ADX score (0-1)
                adx = indicators.get('adx', {}).get('value', 20)
                if adx > 25:
                    score += 0.5  # Strong trend
                
                # Normalize score to 0-1 range
                score = min(1.0, score / 4.0)
                
                # Add key points based on indicators
                key_points = []
                if analysis['trend'] == 'bullish':
                    key_points.append('Bullish trend detected')
                elif analysis['trend'] == 'bearish':
                    key_points.append('Bearish trend detected')
                
                if rsi > 70:
                    key_points.append('Overbought on RSI')
                elif rsi < 30:
                    key_points.append('Oversold on RSI')
                
                if macd.get('macd', 0) > macd.get('signal', 0):
                    key_points.append('MACD bullish crossover')
                elif macd.get('macd', 0) < macd.get('signal', 0):
                    key_points.append('MACD bearish crossover')
                
                analyses.append({
                    'symbol': symbol,
                    'score': round(score, 2),
                    'sentiment': analysis['trend'],
                    'current_price': analysis['current_price'],
                    'key_points': key_points[:3]  # Top 3 points
                })
        
        if not analyses:
            logger.error("No valid analyses generated")
            return None
            
        # Sort analyses by score
        analyses.sort(key=lambda x: x['score'], reverse=True)
        
        # Count sentiments
        sentiments = [a['sentiment'] for a in analyses]
        bullish_count = sentiments.count('bullish')
        bearish_count = sentiments.count('bearish')
        neutral_count = sentiments.count('neutral')
        
        # Determine overall sentiment
        if bullish_count > bearish_count:
            overall_sentiment = 'bullish'
        elif bearish_count > bullish_count:
            overall_sentiment = 'bearish'
        else:
            overall_sentiment = 'neutral'
        
        # Identify key trends
        key_trends = []
        if bullish_count > len(analyses) * 0.6:
            key_trends.append('Strong bullish momentum')
        elif bearish_count > len(analyses) * 0.6:
            key_trends.append('Strong bearish pressure')
            
        if neutral_count > len(analyses) * 0.4:
            key_trends.append('Market consolidation')
            
        # Add sector-specific trends
        tech_sentiment = [a['sentiment'] for a in analyses if a['symbol'] in ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']]
        if tech_sentiment.count('bullish') > len(tech_sentiment) * 0.6:
            key_trends.append('Technology sector leading')
        
        # Prepare report
        report = {
            'timestamp': datetime.now(),
            'top_suggestions': analyses[:5],  # Top 5 stocks
            'market_summary': {
                'bullish_count': bullish_count,
                'bearish_count': bearish_count,
                'neutral_count': neutral_count,
                'overall_sentiment': overall_sentiment,
                'key_trends': key_trends[:3]  # Top 3 trends
            }
        }
        
        return report
        
    except Exception as e:
        logger.error(f"Error generating analysis report: {str(e)}")
        return None

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