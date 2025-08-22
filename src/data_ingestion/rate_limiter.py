"""Adaptive rate limiter for NCBI API compliance with exponential backoff."""

import asyncio
import time
import random
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from collections import deque

from ..config import get_logger


logger = get_logger(__name__)


@dataclass
class RateLimitMetrics:
    """Metrics for rate limiting performance."""
    total_requests: int = 0
    successful_requests: int = 0
    rate_limited_requests: int = 0
    error_requests: int = 0
    total_wait_time: float = 0.0
    current_rate: float = 0.0
    adaptive_adjustments: int = 0
    last_reset_time: Optional[float] = None
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage."""
        if self.total_requests == 0:
            return 0.0
        return (self.successful_requests / self.total_requests) * 100
    
    @property
    def average_wait_time(self) -> float:
        """Calculate average wait time per request."""
        if self.total_requests == 0:
            return 0.0
        return self.total_wait_time / self.total_requests


class AdaptiveRateLimiter:
    """Adaptive rate limiter with NCBI API compliance and exponential backoff."""
    
    def __init__(self,
                 default_rate: float = 3.0,  # requests per second
                 api_key_rate: float = 10.0,  # requests per second with API key
                 has_api_key: bool = False,
                 max_burst: int = 100,
                 backoff_base: float = 1.0,
                 backoff_max: float = 60.0,
                 jitter_factor: float = 0.1,
                 adaptation_factor: float = 0.8):
        """Initialize the adaptive rate limiter.
        
        Args:
            default_rate: Default requests per second without API key
            api_key_rate: Requests per second with API key
            has_api_key: Whether API key is available
            max_burst: Maximum queued requests for burst handling
            backoff_base: Base backoff time in seconds
            backoff_max: Maximum backoff time in seconds
            jitter_factor: Jitter factor for randomizing delays (0-1)
            adaptation_factor: Factor for adaptive rate reduction (0-1)
        """
        self.default_rate = default_rate
        self.api_key_rate = api_key_rate
        self.has_api_key = has_api_key
        self.max_burst = max_burst
        self.backoff_base = backoff_base
        self.backoff_max = backoff_max
        self.jitter_factor = jitter_factor
        self.adaptation_factor = adaptation_factor
        
        # Current rate (starts at configured rate)
        self.current_rate = api_key_rate if has_api_key else default_rate
        self.original_rate = self.current_rate
        
        # Rate limiting state
        self._lock = asyncio.Lock()
        self._last_request_time = 0.0
        self._request_times = deque(maxlen=int(self.current_rate * 10))  # Track last 10 seconds
        self._consecutive_errors = 0
        self._backoff_until = 0.0
        
        # Metrics
        self.metrics = RateLimitMetrics()
        
        logger.info(f"Initialized AdaptiveRateLimiter: rate={self.current_rate}/s, "
                   f"api_key={has_api_key}, max_burst={max_burst}")
    
    async def acquire(self) -> None:
        """Acquire permission to make a request with rate limiting."""
        async with self._lock:
            now = time.time()
            
            # Check if we're in backoff period
            if now < self._backoff_until:
                wait_time = self._backoff_until - now
                logger.debug(f"Waiting {wait_time:.2f}s for backoff period to end")
                await asyncio.sleep(wait_time)
                now = time.time()
            
            # Clean old request times (older than 1 second)
            cutoff_time = now - 1.0
            while self._request_times and self._request_times[0] < cutoff_time:
                self._request_times.popleft()
            
            # Check if we need to wait for rate limit
            if len(self._request_times) >= self.current_rate:
                # Calculate wait time based on oldest request in the window
                oldest_request = self._request_times[0]
                wait_time = 1.0 - (now - oldest_request)
                
                if wait_time > 0:
                    # Add jitter to prevent thundering herd
                    jitter = random.uniform(0, wait_time * self.jitter_factor)
                    total_wait = wait_time + jitter
                    
                    logger.debug(f"Rate limit reached, waiting {total_wait:.2f}s")
                    await asyncio.sleep(total_wait)
                    now = time.time()
            
            # Record this request
            self._request_times.append(now)
            self._last_request_time = now
            self.metrics.total_requests += 1
    
    def record_success(self) -> None:
        """Record a successful request."""
        self.metrics.successful_requests += 1
        self._consecutive_errors = 0
        
        # Gradually increase rate back to original if it was reduced
        if self.current_rate < self.original_rate:
            self.current_rate = min(
                self.original_rate,
                self.current_rate * 1.1  # Increase by 10%
            )
            logger.debug(f"Rate increased to {self.current_rate:.2f}/s after success")
    
    def record_error(self, status_code: int) -> None:
        """Record an error response and adjust rate limiting.
        
        Args:
            status_code: HTTP status code of the error
        """
        self.metrics.error_requests += 1
        
        if status_code == 429:  # Too Many Requests
            self.metrics.rate_limited_requests += 1
            self._consecutive_errors += 1
            
            # Reduce rate more aggressively for rate limit errors
            self.current_rate *= self.adaptation_factor
            self.current_rate = max(0.1, self.current_rate)  # Minimum 0.1 req/s
            self.metrics.adaptive_adjustments += 1
            
            logger.warning(f"Rate limited (429), reduced rate to {self.current_rate:.2f}/s")
            
        elif 500 <= status_code < 600:  # Server errors
            self._consecutive_errors += 1
            
            # Moderate rate reduction for server errors
            if self._consecutive_errors >= 3:
                self.current_rate *= 0.9  # Reduce by 10%
                self.current_rate = max(0.5, self.current_rate)
                self.metrics.adaptive_adjustments += 1
                
                logger.warning(f"Multiple server errors, reduced rate to {self.current_rate:.2f}/s")
        
        # Set backoff period for consecutive errors
        if self._consecutive_errors >= 2:
            backoff_time = min(
                self.backoff_base * (2 ** (self._consecutive_errors - 2)),
                self.backoff_max
            )
            
            # Add jitter to backoff
            jitter = random.uniform(0, backoff_time * self.jitter_factor)
            total_backoff = backoff_time + jitter
            
            self._backoff_until = time.time() + total_backoff
            logger.warning(f"Setting backoff period: {total_backoff:.2f}s after {self._consecutive_errors} consecutive errors")
    
    async def wait_for_reset(self, retry_after: int) -> None:
        """Wait for rate limit reset based on Retry-After header.
        
        Args:
            retry_after: Seconds to wait from Retry-After header
        """
        # Cap the wait time to prevent excessive delays
        wait_time = min(retry_after, self.backoff_max)
        
        # Add jitter
        jitter = random.uniform(0, wait_time * self.jitter_factor)
        total_wait = wait_time + jitter
        
        logger.info(f"Waiting {total_wait:.2f}s for rate limit reset (Retry-After: {retry_after})")
        
        self._backoff_until = time.time() + total_wait
        self.metrics.total_wait_time += total_wait
        self.metrics.last_reset_time = time.time()
        
        await asyncio.sleep(total_wait)
    
    def get_current_rate(self) -> float:
        """Get the current rate limit in requests per second.
        
        Returns:
            float: Current rate limit
        """
        return self.current_rate
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive rate limiting metrics.
        
        Returns:
            Dict[str, Any]: Rate limiting metrics and statistics
        """
        now = time.time()
        
        # Calculate current request rate over last second
        cutoff_time = now - 1.0
        recent_requests = sum(1 for t in self._request_times if t >= cutoff_time)
        
        return {
            'configured_rate': self.original_rate,
            'current_rate': self.current_rate,
            'actual_rate_last_second': recent_requests,
            'has_api_key': self.has_api_key,
            'consecutive_errors': self._consecutive_errors,
            'backoff_remaining': max(0, self._backoff_until - now),
            'queue_size': len(self._request_times),
            'max_burst': self.max_burst,
            'metrics': {
                'total_requests': self.metrics.total_requests,
                'successful_requests': self.metrics.successful_requests,
                'rate_limited_requests': self.metrics.rate_limited_requests,
                'error_requests': self.metrics.error_requests,
                'success_rate': self.metrics.success_rate,
                'total_wait_time': self.metrics.total_wait_time,
                'average_wait_time': self.metrics.average_wait_time,
                'adaptive_adjustments': self.metrics.adaptive_adjustments,
                'last_reset_time': self.metrics.last_reset_time
            }
        }
    
    def reset_rate(self) -> None:
        """Reset rate to original configured value."""
        old_rate = self.current_rate
        self.current_rate = self.original_rate
        self._consecutive_errors = 0
        self._backoff_until = 0.0
        
        logger.info(f"Rate limit reset from {old_rate:.2f}/s to {self.current_rate:.2f}/s")
    
    def update_api_key_status(self, has_api_key: bool) -> None:
        """Update API key status and adjust rates accordingly.
        
        Args:
            has_api_key: Whether API key is now available
        """
        if has_api_key != self.has_api_key:
            self.has_api_key = has_api_key
            self.original_rate = self.api_key_rate if has_api_key else self.default_rate
            self.current_rate = self.original_rate
            
            logger.info(f"API key status updated: {has_api_key}, new rate: {self.current_rate}/s")