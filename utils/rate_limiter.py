"""Rate limiter utility for API calls."""
import time
from functools import wraps
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

class RateLimiter:
    def __init__(self, calls_per_minute: int = 60):
        self.calls_per_minute = calls_per_minute
        self.calls: Dict[str, list] = {}
        
    def _cleanup_old_calls(self, key: str) -> None:
        """Remove calls older than 1 minute."""
        current_time = time.time()
        self.calls[key] = [t for t in self.calls.get(key, []) 
                          if current_time - t < 60]
    
    def wait_if_needed(self, key: str) -> None:
        """Wait if rate limit would be exceeded."""
        self._cleanup_old_calls(key)
        
        if key not in self.calls:
            self.calls[key] = []
            
        if len(self.calls[key]) >= self.calls_per_minute:
            sleep_time = 60 - (time.time() - self.calls[key][0])
            if sleep_time > 0:
                logger.info(f"Rate limit reached for {key}, waiting {sleep_time:.2f} seconds")
                time.sleep(sleep_time)
                self._cleanup_old_calls(key)
        
        self.calls[key].append(time.time())

# Global rate limiter instance
rate_limiter = RateLimiter()

def rate_limit(key: str):
    """Decorator to apply rate limiting to functions."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            rate_limiter.wait_if_needed(key)
            return func(*args, **kwargs)
        return wrapper
    return decorator 