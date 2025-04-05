"""Health check utility for monitoring application health."""
import os
import psutil
import time
from datetime import datetime
from typing import Dict, Any
import logging
import json

logger = logging.getLogger(__name__)

class HealthCheck:
    def __init__(self, log_file: str = "health_checks.json"):
        self.log_file = log_file
        self.start_time = time.time()
        self.last_check = datetime.now()
        
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics."""
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        return {
            "cpu_percent": process.cpu_percent(),
            "memory_usage_mb": memory_info.rss / 1024 / 1024,
            "disk_usage_percent": psutil.disk_usage('/').percent,
            "uptime_seconds": time.time() - self.start_time
        }
    
    def check_api_health(self) -> Dict[str, bool]:
        """Check health of external APIs."""
        # Add your API health checks here
        return {
            "twitter_api": True,  # Replace with actual check
            "finnhub_api": True,  # Replace with actual check
            "news_api": True      # Replace with actual check
        }
    
    def run_health_check(self) -> Dict[str, Any]:
        """Run a complete health check."""
        try:
            metrics = self.get_system_metrics()
            api_health = self.check_api_health()
            
            health_data = {
                "timestamp": datetime.now().isoformat(),
                "status": "healthy",
                "system_metrics": metrics,
                "api_health": api_health
            }
            
            # Log health check
            self._log_health_check(health_data)
            
            return health_data
            
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return {
                "timestamp": datetime.now().isoformat(),
                "status": "unhealthy",
                "error": str(e)
            }
    
    def _log_health_check(self, health_data: Dict[str, Any]) -> None:
        """Log health check data to file."""
        try:
            # Read existing logs
            if os.path.exists(self.log_file):
                with open(self.log_file, 'r') as f:
                    logs = json.load(f)
            else:
                logs = []
            
            # Add new log
            logs.append(health_data)
            
            # Keep only last 1000 logs
            if len(logs) > 1000:
                logs = logs[-1000:]
            
            # Write back to file
            with open(self.log_file, 'w') as f:
                json.dump(logs, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to log health check: {str(e)}")

# Global health check instance
health_checker = HealthCheck() 