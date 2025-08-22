"""Tests for AdaptiveRateLimiter."""

import pytest
import asyncio
import time
from unittest.mock import patch

from src.data_ingestion.rate_limiter import AdaptiveRateLimiter, RateLimitMetrics


class TestRateLimitMetrics:
    """Test cases for RateLimitMetrics."""
    
    def test_init_default_values(self):
        """Test initialization with default values."""
        metrics = RateLimitMetrics()
        
        assert metrics.total_requests == 0
        assert metrics.successful_requests == 0
        assert metrics.rate_limited_requests == 0
        assert metrics.error_requests == 0
        assert metrics.total_wait_time == 0.0
        assert metrics.current_rate == 0.0
        assert metrics.adaptive_adjustments == 0
        assert metrics.last_reset_time is None
    
    def test_success_rate_calculation(self):
        """Test success rate calculation."""
        metrics = RateLimitMetrics()
        
        # No requests
        assert metrics.success_rate == 0.0
        
        # Some requests
        metrics.total_requests = 100
        metrics.successful_requests = 80
        assert metrics.success_rate == 80.0
        
        # All successful
        metrics.successful_requests = 100
        assert metrics.success_rate == 100.0
    
    def test_average_wait_time_calculation(self):
        """Test average wait time calculation."""
        metrics = RateLimitMetrics()
        
        # No requests
        assert metrics.average_wait_time == 0.0
        
        # Some requests
        metrics.total_requests = 10
        metrics.total_wait_time = 5.0
        assert metrics.average_wait_time == 0.5


class TestAdaptiveRateLimiter:
    """Test cases for AdaptiveRateLimiter."""
    
    def test_init_default_values(self):
        """Test initialization with default values."""
        limiter = AdaptiveRateLimiter()
        
        assert limiter.default_rate == 3.0
        assert limiter.api_key_rate == 10.0
        assert limiter.has_api_key is False
        assert limiter.current_rate == 3.0  # Should use default rate
        assert limiter.original_rate == 3.0
        assert limiter.max_burst == 100
        assert limiter.backoff_base == 1.0
        assert limiter.backoff_max == 60.0
        assert limiter.jitter_factor == 0.1
        assert limiter.adaptation_factor == 0.8
    
    def test_init_with_api_key(self):
        """Test initialization with API key."""
        limiter = AdaptiveRateLimiter(has_api_key=True)
        
        assert limiter.has_api_key is True
        assert limiter.current_rate == 10.0  # Should use API key rate
        assert limiter.original_rate == 10.0
    
    def test_init_custom_values(self):
        """Test initialization with custom values."""
        limiter = AdaptiveRateLimiter(
            default_rate=5.0,
            api_key_rate=15.0,
            has_api_key=True,
            max_burst=50,
            backoff_base=2.0,
            backoff_max=120.0,
            jitter_factor=0.2,
            adaptation_factor=0.7
        )
        
        assert limiter.default_rate == 5.0
        assert limiter.api_key_rate == 15.0
        assert limiter.current_rate == 15.0
        assert limiter.max_burst == 50
        assert limiter.backoff_base == 2.0
        assert limiter.backoff_max == 120.0
        assert limiter.jitter_factor == 0.2
        assert limiter.adaptation_factor == 0.7
    
    @pytest.mark.asyncio
    async def test_acquire_no_rate_limit(self):
        """Test acquire when no rate limiting is needed."""
        limiter = AdaptiveRateLimiter(default_rate=10.0)  # High rate limit
        
        start_time = time.time()
        
        # Should not wait for first few requests
        for _ in range(5):
            await limiter.acquire()
        
        elapsed = time.time() - start_time
        assert elapsed < 0.1  # Should be very fast
        
        assert len(limiter._request_times) == 5
        assert limiter.metrics.total_requests == 5
    
    @pytest.mark.asyncio
    async def test_acquire_with_rate_limit(self):
        """Test acquire when rate limiting is triggered."""
        limiter = AdaptiveRateLimiter(default_rate=2.0)  # Low rate limit
        
        start_time = time.time()
        
        # First 2 requests should be fast
        await limiter.acquire()
        await limiter.acquire()
        
        # Third request should wait
        await limiter.acquire()
        
        elapsed = time.time() - start_time
        assert elapsed >= 0.4  # Should wait at least some time
        
        assert limiter.metrics.total_requests == 3
    
    @pytest.mark.asyncio
    async def test_acquire_with_backoff(self):
        """Test acquire when in backoff period."""
        limiter = AdaptiveRateLimiter()
        
        # Set backoff period
        limiter._backoff_until = time.time() + 0.2
        
        start_time = time.time()
        await limiter.acquire()
        elapsed = time.time() - start_time
        
        assert elapsed >= 0.15  # Should wait for backoff
    
    def test_record_success(self):
        """Test recording successful requests."""
        limiter = AdaptiveRateLimiter(default_rate=5.0)
        limiter.current_rate = 3.0  # Reduced rate
        limiter._consecutive_errors = 2
        
        limiter.record_success()
        
        assert limiter.metrics.successful_requests == 1
        assert limiter._consecutive_errors == 0
        assert limiter.current_rate > 3.0  # Should increase rate
    
    def test_record_error_rate_limit(self):
        """Test recording rate limit errors (429)."""
        limiter = AdaptiveRateLimiter(default_rate=5.0)
        original_rate = limiter.current_rate
        
        limiter.record_error(429)
        
        assert limiter.metrics.error_requests == 1
        assert limiter.metrics.rate_limited_requests == 1
        assert limiter._consecutive_errors == 1
        assert limiter.current_rate < original_rate  # Should reduce rate
        assert limiter.metrics.adaptive_adjustments == 1
    
    def test_record_error_server_error(self):
        """Test recording server errors (5xx)."""
        limiter = AdaptiveRateLimiter(default_rate=5.0)
        
        # First few server errors shouldn't reduce rate
        limiter.record_error(500)
        limiter.record_error(502)
        original_rate = limiter.current_rate
        
        # Third consecutive error should reduce rate
        limiter.record_error(503)
        
        assert limiter.metrics.error_requests == 3
        assert limiter._consecutive_errors == 3
        assert limiter.current_rate < original_rate
    
    def test_record_error_backoff_period(self):
        """Test backoff period setting after consecutive errors."""
        limiter = AdaptiveRateLimiter()
        
        # First error - no backoff
        limiter.record_error(500)
        assert limiter._backoff_until == 0.0
        
        # Second error - should set backoff
        limiter.record_error(500)
        assert limiter._backoff_until > time.time()
    
    @pytest.mark.asyncio
    async def test_wait_for_reset(self):
        """Test waiting for rate limit reset."""
        limiter = AdaptiveRateLimiter()
        
        start_time = time.time()
        await limiter.wait_for_reset(1)  # Wait 1 second
        elapsed = time.time() - start_time
        
        assert elapsed >= 0.8  # Should wait close to 1 second (with jitter)
        assert limiter.metrics.total_wait_time > 0
        assert limiter.metrics.last_reset_time is not None
        assert limiter._backoff_until > time.time()
    
    @pytest.mark.asyncio
    async def test_wait_for_reset_capped(self):
        """Test wait for reset is capped at backoff_max."""
        limiter = AdaptiveRateLimiter(backoff_max=2.0)
        
        start_time = time.time()
        await limiter.wait_for_reset(10)  # Request 10 seconds, should be capped
        elapsed = time.time() - start_time
        
        assert elapsed < 3.0  # Should be capped at backoff_max + jitter
    
    def test_get_current_rate(self):
        """Test getting current rate."""
        limiter = AdaptiveRateLimiter(default_rate=5.0)
        
        assert limiter.get_current_rate() == 5.0
        
        limiter.current_rate = 3.0
        assert limiter.get_current_rate() == 3.0
    
    def test_get_metrics(self):
        """Test getting comprehensive metrics."""
        limiter = AdaptiveRateLimiter(
            default_rate=5.0,
            api_key_rate=10.0,
            has_api_key=True
        )
        
        # Add some test data
        limiter.metrics.total_requests = 100
        limiter.metrics.successful_requests = 90
        limiter._consecutive_errors = 2
        limiter._backoff_until = time.time() + 5.0
        
        metrics = limiter.get_metrics()
        
        assert metrics['configured_rate'] == 10.0
        assert metrics['current_rate'] == 10.0
        assert metrics['has_api_key'] is True
        assert metrics['consecutive_errors'] == 2
        assert metrics['backoff_remaining'] > 0
        assert metrics['max_burst'] == 100
        assert metrics['metrics']['total_requests'] == 100
        assert metrics['metrics']['successful_requests'] == 90
        assert metrics['metrics']['success_rate'] == 90.0
    
    def test_reset_rate(self):
        """Test resetting rate to original value."""
        limiter = AdaptiveRateLimiter(default_rate=5.0)
        
        # Modify state
        limiter.current_rate = 2.0
        limiter._consecutive_errors = 3
        limiter._backoff_until = time.time() + 10.0
        
        limiter.reset_rate()
        
        assert limiter.current_rate == 5.0
        assert limiter._consecutive_errors == 0
        assert limiter._backoff_until == 0.0
    
    def test_update_api_key_status(self):
        """Test updating API key status."""
        limiter = AdaptiveRateLimiter(
            default_rate=3.0,
            api_key_rate=10.0,
            has_api_key=False
        )
        
        assert limiter.current_rate == 3.0
        
        # Enable API key
        limiter.update_api_key_status(True)
        
        assert limiter.has_api_key is True
        assert limiter.current_rate == 10.0
        assert limiter.original_rate == 10.0
        
        # Disable API key
        limiter.update_api_key_status(False)
        
        assert limiter.has_api_key is False
        assert limiter.current_rate == 3.0
        assert limiter.original_rate == 3.0
    
    @pytest.mark.asyncio
    async def test_concurrent_acquire(self):
        """Test concurrent acquire calls."""
        limiter = AdaptiveRateLimiter(default_rate=5.0)
        
        async def make_request():
            await limiter.acquire()
            return time.time()
        
        # Make multiple concurrent requests
        start_time = time.time()
        tasks = [make_request() for _ in range(10)]
        results = await asyncio.gather(*tasks)
        
        # All requests should complete
        assert len(results) == 10
        assert limiter.metrics.total_requests == 10
        
        # Requests should be spread out due to rate limiting
        total_time = max(results) - min(results)
        assert total_time >= 1.0  # Should take at least 1 second for 10 requests at 5/sec
    
    @pytest.mark.asyncio
    async def test_jitter_in_waits(self):
        """Test that jitter is applied to wait times."""
        limiter = AdaptiveRateLimiter(jitter_factor=0.5)  # High jitter for testing
        
        # Record multiple wait times
        wait_times = []
        for _ in range(5):
            start = time.time()
            await limiter.wait_for_reset(1)
            wait_times.append(time.time() - start)
        
        # Wait times should vary due to jitter
        assert len(set(f"{t:.2f}" for t in wait_times)) > 1  # Should have different values
    
    def test_request_times_cleanup(self):
        """Test that old request times are cleaned up."""
        limiter = AdaptiveRateLimiter()
        
        # Add old request times
        old_time = time.time() - 2.0
        limiter._request_times.extend([old_time] * 5)
        
        # Current request times
        current_time = time.time()
        limiter._request_times.extend([current_time] * 3)
        
        # Trigger cleanup by calling acquire (mocked to avoid actual waiting)
        with patch('time.time', return_value=current_time):
            with patch('asyncio.sleep'):
                asyncio.run(limiter.acquire())
        
        # Old times should be cleaned up
        assert len(limiter._request_times) <= 4  # 3 current + 1 new