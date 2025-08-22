"""Rate limiting and throttling system for API endpoints."""

import time
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, deque
import threading
import logging

from fastapi import HTTPException, Request, status
from pydantic import BaseModel

from ..config import get_logger

logger = get_logger(__name__)


class RateLimitRule(BaseModel):
    """Rate limit rule configuration."""
    requests_per_minute: int
    requests_per_hour: int
    requests_per_day: int
    burst_size: int = 10  # Maximum burst requests
    burst_window_seconds: int = 60  # Burst window duration


class RateLimitInfo(BaseModel):
    """Rate limit information."""
    limit: int
    remaining: int
    reset_time: datetime
    retry_after: Optional[int] = None


class RateLimiter:
    """Rate limiter with sliding window and burst handling."""
    
    def __init__(self):
        """Initialize rate limiter."""
        self.rules = self._setup_default_rules()
        self.request_history = defaultdict(lambda: {
            'minute': deque(),
            'hour': deque(),
            'day': deque(),
            'burst': deque()
        })
        self.lock = threading.RLock()
        
        # Cleanup task
        self._last_cleanup = time.time()
        self._cleanup_interval = 300  # 5 minutes
        
        logger.info("Rate limiter initialized")
    
    def _setup_default_rules(self) -> Dict[str, RateLimitRule]:
        """Setup default rate limiting rules."""
        return {
            # General API endpoints
            'default': RateLimitRule(
                requests_per_minute=60,
                requests_per_hour=1000,
                requests_per_day=10000,
                burst_size=10,
                burst_window_seconds=60
            ),
            
            # Prediction endpoints (more restrictive)
            'predictions': RateLimitRule(
                requests_per_minute=30,
                requests_per_hour=500,
                requests_per_day=5000,
                burst_size=5,
                burst_window_seconds=60
            ),
            
            # Batch predictions (very restrictive)
            'batch_predictions': RateLimitRule(
                requests_per_minute=5,
                requests_per_hour=50,
                requests_per_day=200,
                burst_size=2,
                burst_window_seconds=120
            ),
            
            # Clinical endpoints
            'clinical': RateLimitRule(
                requests_per_minute=20,
                requests_per_hour=300,
                requests_per_day=2000,
                burst_size=3,
                burst_window_seconds=60
            ),
            
            # Admin endpoints (less restrictive for authenticated users)
            'admin': RateLimitRule(
                requests_per_minute=100,
                requests_per_hour=2000,
                requests_per_day=20000,
                burst_size=20,
                burst_window_seconds=60
            ),
            
            # Health checks (very permissive)
            'health': RateLimitRule(
                requests_per_minute=120,
                requests_per_hour=3600,
                requests_per_day=86400,
                burst_size=30,
                burst_window_seconds=30
            )
        }
    
    def _get_client_id(self, request: Request) -> str:
        """Get client identifier for rate limiting."""
        # Try to get user ID from request state (set by auth middleware)
        if hasattr(request.state, 'user') and request.state.user:
            return f"user:{request.state.user.user_id}"
        
        # Try to get API key from headers
        api_key = request.headers.get('X-API-Key')
        if api_key:
            return f"api_key:{api_key[:8]}..."  # Truncated for logging
        
        # Fall back to IP address
        client_ip = request.client.host if request.client else "unknown"
        forwarded_for = request.headers.get('X-Forwarded-For')
        if forwarded_for:
            client_ip = forwarded_for.split(',')[0].strip()
        
        return f"ip:{client_ip}"
    
    def _get_endpoint_category(self, request: Request) -> str:
        """Determine endpoint category for rate limiting."""
        path = request.url.path.lower()
        
        if '/health' in path or '/status' in path:
            return 'health'
        elif '/admin' in path:
            return 'admin'
        elif '/predictions' in path:
            if 'batch' in path:
                return 'batch_predictions'
            else:
                return 'predictions'
        elif any(clinical in path for clinical in ['/drug-gene', '/risk-scoring', '/dosing', '/clinical-support']):
            return 'clinical'
        else:
            return 'default'
    
    def _cleanup_old_requests(self):
        """Clean up old request history entries."""
        current_time = time.time()
        
        # Only cleanup every few minutes
        if current_time - self._last_cleanup < self._cleanup_interval:
            return
        
        with self.lock:
            cutoff_times = {
                'minute': current_time - 60,
                'hour': current_time - 3600,
                'day': current_time - 86400,
                'burst': current_time - 300  # 5 minutes for burst
            }
            
            for client_id, history in list(self.request_history.items()):
                for window, cutoff_time in cutoff_times.items():
                    window_history = history[window]
                    
                    # Remove old entries
                    while window_history and window_history[0] < cutoff_time:
                        window_history.popleft()
                
                # Remove empty client histories
                if all(not history[window] for window in history):
                    del self.request_history[client_id]
            
            self._last_cleanup = current_time
            logger.debug(f"Cleaned up rate limit history for {len(self.request_history)} clients")
    
    def _check_rate_limit(self, client_id: str, rule: RateLimitRule) -> Tuple[bool, RateLimitInfo]:
        """Check if request is within rate limits."""
        current_time = time.time()
        
        with self.lock:
            history = self.request_history[client_id]
            
            # Check different time windows
            windows = {
                'minute': (60, rule.requests_per_minute),
                'hour': (3600, rule.requests_per_hour),
                'day': (86400, rule.requests_per_day),
                'burst': (rule.burst_window_seconds, rule.burst_size)
            }
            
            for window_name, (window_seconds, limit) in windows.items():
                window_history = history[window_name]
                cutoff_time = current_time - window_seconds
                
                # Remove old entries
                while window_history and window_history[0] < cutoff_time:
                    window_history.popleft()
                
                # Check if limit exceeded
                if len(window_history) >= limit:
                    # Calculate retry after time
                    if window_history:
                        oldest_request = window_history[0]
                        retry_after = int(oldest_request + window_seconds - current_time) + 1
                    else:
                        retry_after = window_seconds
                    
                    return False, RateLimitInfo(
                        limit=limit,
                        remaining=0,
                        reset_time=datetime.fromtimestamp(current_time + retry_after),
                        retry_after=retry_after
                    )
            
            # If we get here, request is allowed
            # Add current request to all windows
            for window_name in windows:
                history[window_name].append(current_time)
            
            # Calculate remaining requests (use most restrictive window)
            remaining = min(
                rule.requests_per_minute - len(history['minute']),
                rule.requests_per_hour - len(history['hour']),
                rule.requests_per_day - len(history['day'])
            )
            
            return True, RateLimitInfo(
                limit=rule.requests_per_minute,  # Show minute limit as primary
                remaining=max(0, remaining),
                reset_time=datetime.fromtimestamp(current_time + 60)  # Next minute reset
            )
    
    async def check_rate_limit(self, request: Request) -> None:
        """Check rate limit for incoming request (FastAPI dependency)."""
        # Cleanup old entries periodically
        self._cleanup_old_requests()
        
        # Get client identifier and endpoint category
        client_id = self._get_client_id(request)
        category = self._get_endpoint_category(request)
        
        # Get rate limit rule
        rule = self.rules.get(category, self.rules['default'])
        
        # Check rate limit
        allowed, limit_info = self._check_rate_limit(client_id, rule)
        
        # Add rate limit headers to response (will be added by middleware)
        if not hasattr(request.state, 'rate_limit_info'):
            request.state.rate_limit_info = {}
        
        request.state.rate_limit_info = {
            'limit': limit_info.limit,
            'remaining': limit_info.remaining,
            'reset': int(limit_info.reset_time.timestamp()),
            'category': category
        }
        
        if not allowed:
            logger.warning(
                f"Rate limit exceeded for {client_id} on {category} endpoint",
                extra={
                    'client_id': client_id,
                    'category': category,
                    'limit': limit_info.limit,
                    'retry_after': limit_info.retry_after
                }
            )
            
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={
                    "error": "RATE_LIMIT_EXCEEDED",
                    "message": f"Rate limit exceeded for {category} endpoints",
                    "limit": limit_info.limit,
                    "retry_after": limit_info.retry_after,
                    "reset_time": limit_info.reset_time.isoformat()
                },
                headers={
                    "Retry-After": str(limit_info.retry_after),
                    "X-RateLimit-Limit": str(limit_info.limit),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(limit_info.reset_time.timestamp()))
                }
            )
        
        logger.debug(
            f"Rate limit check passed for {client_id} on {category}",
            extra={
                'client_id': client_id,
                'category': category,
                'remaining': limit_info.remaining
            }
        )
    
    def get_rate_limit_status(self, client_id: str, category: str = 'default') -> RateLimitInfo:
        """Get current rate limit status for a client."""
        rule = self.rules.get(category, self.rules['default'])
        allowed, limit_info = self._check_rate_limit(client_id, rule)
        return limit_info
    
    def reset_rate_limit(self, client_id: str) -> bool:
        """Reset rate limit for a specific client."""
        with self.lock:
            if client_id in self.request_history:
                del self.request_history[client_id]
                logger.info(f"Reset rate limit for client: {client_id}")
                return True
            return False
    
    def update_rule(self, category: str, rule: RateLimitRule) -> None:
        """Update rate limiting rule for a category."""
        self.rules[category] = rule
        logger.info(f"Updated rate limit rule for category: {category}")
    
    def get_rules(self) -> Dict[str, RateLimitRule]:
        """Get all rate limiting rules."""
        return self.rules.copy()
    
    def get_client_stats(self, client_id: str) -> Dict[str, Any]:
        """Get statistics for a specific client."""
        with self.lock:
            if client_id not in self.request_history:
                return {
                    'client_id': client_id,
                    'requests_last_minute': 0,
                    'requests_last_hour': 0,
                    'requests_last_day': 0,
                    'burst_requests': 0
                }
            
            history = self.request_history[client_id]
            
            return {
                'client_id': client_id,
                'requests_last_minute': len(history['minute']),
                'requests_last_hour': len(history['hour']),
                'requests_last_day': len(history['day']),
                'burst_requests': len(history['burst'])
            }
    
    def get_global_stats(self) -> Dict[str, Any]:
        """Get global rate limiting statistics."""
        with self.lock:
            total_clients = len(self.request_history)
            total_requests = {
                'minute': sum(len(history['minute']) for history in self.request_history.values()),
                'hour': sum(len(history['hour']) for history in self.request_history.values()),
                'day': sum(len(history['day']) for history in self.request_history.values())
            }
            
            return {
                'total_clients': total_clients,
                'total_requests_last_minute': total_requests['minute'],
                'total_requests_last_hour': total_requests['hour'],
                'total_requests_last_day': total_requests['day'],
                'rules': {category: {
                    'requests_per_minute': rule.requests_per_minute,
                    'requests_per_hour': rule.requests_per_hour,
                    'requests_per_day': rule.requests_per_day,
                    'burst_size': rule.burst_size
                } for category, rule in self.rules.items()}
            }
    
    def add_rate_limit_headers_middleware(self):
        """Middleware to add rate limit headers to responses."""
        async def add_headers(request: Request, call_next):
            response = await call_next(request)
            
            # Add rate limit headers if available
            if hasattr(request.state, 'rate_limit_info'):
                info = request.state.rate_limit_info
                response.headers["X-RateLimit-Limit"] = str(info['limit'])
                response.headers["X-RateLimit-Remaining"] = str(info['remaining'])
                response.headers["X-RateLimit-Reset"] = str(info['reset'])
                response.headers["X-RateLimit-Category"] = info['category']
            
            return response
        
        return add_headers