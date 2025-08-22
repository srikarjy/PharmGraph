"""API monitoring and analytics system with metrics collection and alerting."""

import time
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from collections import defaultdict, deque
import threading
import json
import logging
from dataclasses import dataclass, field
from enum import Enum

from fastapi import Request, Response
import psutil

from ..config import get_logger

logger = get_logger(__name__)


class AlertLevel(str, Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class MetricPoint:
    """Single metric data point."""
    timestamp: datetime
    value: float
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class Alert:
    """System alert."""
    alert_id: str
    level: AlertLevel
    message: str
    metric: str
    value: float
    threshold: float
    timestamp: datetime
    resolved: bool = False
    resolved_at: Optional[datetime] = None


@dataclass
class APIMetrics:
    """API performance metrics."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    average_response_time_ms: float = 0.0
    p95_response_time_ms: float = 0.0
    p99_response_time_ms: float = 0.0
    requests_per_second: float = 0.0
    error_rate: float = 0.0
    uptime_seconds: float = 0.0


@dataclass
class SystemMetrics:
    """System resource metrics."""
    cpu_usage_percent: float = 0.0
    memory_usage_percent: float = 0.0
    memory_usage_mb: float = 0.0
    disk_usage_percent: float = 0.0
    network_bytes_sent: int = 0
    network_bytes_received: int = 0
    active_connections: int = 0


class MetricsCollector:
    """Collects and stores metrics data."""
    
    def __init__(self, retention_hours: int = 24):
        """Initialize metrics collector.
        
        Args:
            retention_hours: How long to retain metrics data
        """
        self.retention_hours = retention_hours
        self.retention_seconds = retention_hours * 3600
        
        # Metrics storage
        self.metrics = defaultdict(lambda: deque(maxlen=10000))
        self.response_times = deque(maxlen=1000)  # Last 1000 response times
        self.request_counts = defaultdict(int)
        self.error_counts = defaultdict(int)
        
        # System metrics
        self.system_metrics_history = deque(maxlen=1440)  # 24 hours at 1-minute intervals
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Start time for uptime calculation
        self.start_time = time.time()
        
        logger.info("Metrics collector initialized")
    
    def record_request(self, method: str, endpoint: str, status_code: int, 
                      response_time_ms: float, user_id: Optional[str] = None):
        """Record API request metrics."""
        timestamp = datetime.utcnow()
        
        with self.lock:
            # Record response time
            self.response_times.append(response_time_ms)
            
            # Record request count
            key = f"{method}:{endpoint}"
            self.request_counts[key] += 1
            self.request_counts["total"] += 1
            
            # Record errors
            if status_code >= 400:
                self.error_counts[key] += 1
                self.error_counts["total"] += 1
            
            # Store detailed metrics
            labels = {
                "method": method,
                "endpoint": endpoint,
                "status_code": str(status_code),
                "user_id": user_id or "anonymous"
            }
            
            self.metrics["request_count"].append(
                MetricPoint(timestamp, 1, labels)
            )
            
            self.metrics["response_time"].append(
                MetricPoint(timestamp, response_time_ms, labels)
            )
            
            if status_code >= 400:
                self.metrics["error_count"].append(
                    MetricPoint(timestamp, 1, labels)
                )
    
    def record_prediction(self, model_name: str, processing_time_ms: float, 
                         success: bool, user_id: Optional[str] = None):
        """Record ML prediction metrics."""
        timestamp = datetime.utcnow()
        
        with self.lock:
            labels = {
                "model": model_name,
                "success": str(success),
                "user_id": user_id or "anonymous"
            }
            
            self.metrics["prediction_count"].append(
                MetricPoint(timestamp, 1, labels)
            )
            
            self.metrics["prediction_time"].append(
                MetricPoint(timestamp, processing_time_ms, labels)
            )
            
            if not success:
                self.metrics["prediction_error"].append(
                    MetricPoint(timestamp, 1, labels)
                )
    
    def record_system_metrics(self):
        """Record system resource metrics."""
        timestamp = datetime.utcnow()
        
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_mb = memory.used / (1024 * 1024)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            
            # Network stats
            network = psutil.net_io_counters()
            
            with self.lock:
                system_metrics = SystemMetrics(
                    cpu_usage_percent=cpu_percent,
                    memory_usage_percent=memory_percent,
                    memory_usage_mb=memory_mb,
                    disk_usage_percent=disk_percent,
                    network_bytes_sent=network.bytes_sent,
                    network_bytes_received=network.bytes_recv,
                    active_connections=len(psutil.net_connections())
                )
                
                self.system_metrics_history.append((timestamp, system_metrics))
                
                # Store as individual metrics
                self.metrics["cpu_usage"].append(
                    MetricPoint(timestamp, cpu_percent)
                )
                
                self.metrics["memory_usage"].append(
                    MetricPoint(timestamp, memory_percent)
                )
                
                self.metrics["disk_usage"].append(
                    MetricPoint(timestamp, disk_percent)
                )
                
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
    
    def get_api_metrics(self, time_window_minutes: int = 60) -> APIMetrics:
        """Get API performance metrics for a time window."""
        cutoff_time = datetime.utcnow() - timedelta(minutes=time_window_minutes)
        
        with self.lock:
            # Filter metrics by time window
            recent_requests = [
                point for point in self.metrics["request_count"]
                if point.timestamp >= cutoff_time
            ]
            
            recent_response_times = [
                point.value for point in self.metrics["response_time"]
                if point.timestamp >= cutoff_time
            ]
            
            recent_errors = [
                point for point in self.metrics["error_count"]
                if point.timestamp >= cutoff_time
            ]
            
            # Calculate metrics
            total_requests = len(recent_requests)
            successful_requests = total_requests - len(recent_errors)
            failed_requests = len(recent_errors)
            
            if recent_response_times:
                avg_response_time = sum(recent_response_times) / len(recent_response_times)
                sorted_times = sorted(recent_response_times)
                p95_index = int(0.95 * len(sorted_times))
                p99_index = int(0.99 * len(sorted_times))
                p95_response_time = sorted_times[p95_index] if p95_index < len(sorted_times) else 0
                p99_response_time = sorted_times[p99_index] if p99_index < len(sorted_times) else 0
            else:
                avg_response_time = 0.0
                p95_response_time = 0.0
                p99_response_time = 0.0
            
            requests_per_second = total_requests / (time_window_minutes * 60) if time_window_minutes > 0 else 0
            error_rate = (failed_requests / total_requests) if total_requests > 0 else 0
            uptime_seconds = time.time() - self.start_time
            
            return APIMetrics(
                total_requests=total_requests,
                successful_requests=successful_requests,
                failed_requests=failed_requests,
                average_response_time_ms=avg_response_time,
                p95_response_time_ms=p95_response_time,
                p99_response_time_ms=p99_response_time,
                requests_per_second=requests_per_second,
                error_rate=error_rate,
                uptime_seconds=uptime_seconds
            )
    
    def get_system_metrics(self) -> Optional[SystemMetrics]:
        """Get latest system metrics."""
        with self.lock:
            if self.system_metrics_history:
                return self.system_metrics_history[-1][1]
            return None
    
    def get_metric_history(self, metric_name: str, time_window_minutes: int = 60) -> List[MetricPoint]:
        """Get metric history for a time window."""
        cutoff_time = datetime.utcnow() - timedelta(minutes=time_window_minutes)
        
        with self.lock:
            return [
                point for point in self.metrics[metric_name]
                if point.timestamp >= cutoff_time
            ]
    
    def cleanup_old_metrics(self):
        """Clean up old metrics data."""
        cutoff_time = datetime.utcnow() - timedelta(seconds=self.retention_seconds)
        
        with self.lock:
            for metric_name, points in self.metrics.items():
                # Remove old points
                while points and points[0].timestamp < cutoff_time:
                    points.popleft()
            
            # Clean up system metrics
            while (self.system_metrics_history and 
                   self.system_metrics_history[0][0] < cutoff_time):
                self.system_metrics_history.popleft()


class AlertManager:
    """Manages system alerts and notifications."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        """Initialize alert manager.
        
        Args:
            metrics_collector: Metrics collector instance
        """
        self.metrics_collector = metrics_collector
        self.alerts = deque(maxlen=1000)  # Keep last 1000 alerts
        self.alert_rules = self._setup_default_alert_rules()
        self.alert_callbacks = []
        self.lock = threading.RLock()
        
        logger.info("Alert manager initialized")
    
    def _setup_default_alert_rules(self) -> Dict[str, Dict[str, Any]]:
        """Setup default alerting rules."""
        return {
            "high_error_rate": {
                "metric": "error_rate",
                "threshold": 0.05,  # 5% error rate
                "comparison": "greater_than",
                "level": AlertLevel.WARNING,
                "message": "High error rate detected: {value:.2%}"
            },
            "critical_error_rate": {
                "metric": "error_rate",
                "threshold": 0.10,  # 10% error rate
                "comparison": "greater_than",
                "level": AlertLevel.CRITICAL,
                "message": "Critical error rate detected: {value:.2%}"
            },
            "high_response_time": {
                "metric": "p95_response_time_ms",
                "threshold": 1000,  # 1 second
                "comparison": "greater_than",
                "level": AlertLevel.WARNING,
                "message": "High response time detected: {value:.0f}ms"
            },
            "critical_response_time": {
                "metric": "p95_response_time_ms",
                "threshold": 2000,  # 2 seconds
                "comparison": "greater_than",
                "level": AlertLevel.CRITICAL,
                "message": "Critical response time detected: {value:.0f}ms"
            },
            "high_cpu_usage": {
                "metric": "cpu_usage_percent",
                "threshold": 80,  # 80% CPU
                "comparison": "greater_than",
                "level": AlertLevel.WARNING,
                "message": "High CPU usage detected: {value:.1f}%"
            },
            "critical_cpu_usage": {
                "metric": "cpu_usage_percent",
                "threshold": 95,  # 95% CPU
                "comparison": "greater_than",
                "level": AlertLevel.CRITICAL,
                "message": "Critical CPU usage detected: {value:.1f}%"
            },
            "high_memory_usage": {
                "metric": "memory_usage_percent",
                "threshold": 85,  # 85% memory
                "comparison": "greater_than",
                "level": AlertLevel.WARNING,
                "message": "High memory usage detected: {value:.1f}%"
            },
            "critical_memory_usage": {
                "metric": "memory_usage_percent",
                "threshold": 95,  # 95% memory
                "comparison": "greater_than",
                "level": AlertLevel.CRITICAL,
                "message": "Critical memory usage detected: {value:.1f}%"
            }
        }
    
    def check_alerts(self):
        """Check all alert rules and trigger alerts if needed."""
        # Get current metrics
        api_metrics = self.metrics_collector.get_api_metrics(time_window_minutes=5)
        system_metrics = self.metrics_collector.get_system_metrics()
        
        # Combine metrics for checking
        current_values = {
            "error_rate": api_metrics.error_rate,
            "p95_response_time_ms": api_metrics.p95_response_time_ms,
            "requests_per_second": api_metrics.requests_per_second
        }
        
        if system_metrics:
            current_values.update({
                "cpu_usage_percent": system_metrics.cpu_usage_percent,
                "memory_usage_percent": system_metrics.memory_usage_percent,
                "disk_usage_percent": system_metrics.disk_usage_percent
            })
        
        # Check each alert rule
        for rule_name, rule in self.alert_rules.items():
            metric_name = rule["metric"]
            
            if metric_name not in current_values:
                continue
            
            current_value = current_values[metric_name]
            threshold = rule["threshold"]
            comparison = rule["comparison"]
            
            # Check if alert condition is met
            alert_triggered = False
            
            if comparison == "greater_than" and current_value > threshold:
                alert_triggered = True
            elif comparison == "less_than" and current_value < threshold:
                alert_triggered = True
            elif comparison == "equals" and current_value == threshold:
                alert_triggered = True
            
            if alert_triggered:
                self._trigger_alert(rule_name, rule, current_value)
    
    def _trigger_alert(self, rule_name: str, rule: Dict[str, Any], value: float):
        """Trigger an alert."""
        alert_id = f"{rule_name}_{int(time.time())}"
        
        alert = Alert(
            alert_id=alert_id,
            level=rule["level"],
            message=rule["message"].format(value=value),
            metric=rule["metric"],
            value=value,
            threshold=rule["threshold"],
            timestamp=datetime.utcnow()
        )
        
        with self.lock:
            self.alerts.append(alert)
        
        # Log alert
        log_level = logging.WARNING if alert.level in [AlertLevel.WARNING, AlertLevel.INFO] else logging.ERROR
        logger.log(log_level, f"ALERT [{alert.level.value.upper()}]: {alert.message}")
        
        # Call alert callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")
    
    def add_alert_callback(self, callback: Callable[[Alert], None]):
        """Add alert callback function."""
        self.alert_callbacks.append(callback)
    
    def get_active_alerts(self, level: Optional[AlertLevel] = None) -> List[Alert]:
        """Get active (unresolved) alerts."""
        with self.lock:
            alerts = [alert for alert in self.alerts if not alert.resolved]
            
            if level:
                alerts = [alert for alert in alerts if alert.level == level]
            
            return sorted(alerts, key=lambda a: a.timestamp, reverse=True)
    
    def get_alert_history(self, hours: int = 24) -> List[Alert]:
        """Get alert history for specified hours."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        with self.lock:
            return [
                alert for alert in self.alerts
                if alert.timestamp >= cutoff_time
            ]
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert."""
        with self.lock:
            for alert in self.alerts:
                if alert.alert_id == alert_id and not alert.resolved:
                    alert.resolved = True
                    alert.resolved_at = datetime.utcnow()
                    logger.info(f"Resolved alert: {alert_id}")
                    return True
            return False


class APIMonitor:
    """Main API monitoring system."""
    
    def __init__(self):
        """Initialize API monitor."""
        self.metrics_collector = MetricsCollector()
        self.alert_manager = AlertManager(self.metrics_collector)
        
        # Background tasks
        self._monitoring_task = None
        self._cleanup_task = None
        self._running = False
        
        logger.info("API monitor initialized")
    
    async def start_monitoring(self):
        """Start background monitoring tasks."""
        if self._running:
            return
        
        self._running = True
        
        # Start system metrics collection
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        # Start cleanup task
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        logger.info("API monitoring started")
    
    async def stop_monitoring(self):
        """Stop background monitoring tasks."""
        self._running = False
        
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        logger.info("API monitoring stopped")
    
    async def _monitoring_loop(self):
        """Background monitoring loop."""
        while self._running:
            try:
                # Collect system metrics
                self.metrics_collector.record_system_metrics()
                
                # Check alerts
                self.alert_manager.check_alerts()
                
                # Wait for next collection
                await asyncio.sleep(60)  # Collect every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(60)
    
    async def _cleanup_loop(self):
        """Background cleanup loop."""
        while self._running:
            try:
                # Clean up old metrics
                self.metrics_collector.cleanup_old_metrics()
                
                # Wait for next cleanup
                await asyncio.sleep(3600)  # Cleanup every hour
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")
                await asyncio.sleep(3600)
    
    def record_request(self, request: Request, response: Response, 
                      processing_time_ms: float):
        """Record API request for monitoring."""
        # Extract request info
        method = request.method
        endpoint = request.url.path
        status_code = response.status_code
        user_id = getattr(request.state, 'user_id', None)
        
        # Record metrics
        self.metrics_collector.record_request(
            method, endpoint, status_code, processing_time_ms, user_id
        )
    
    def record_prediction(self, model_name: str, processing_time_ms: float, 
                         success: bool, user_id: Optional[str] = None):
        """Record ML prediction for monitoring."""
        self.metrics_collector.record_prediction(
            model_name, processing_time_ms, success, user_id
        )
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for monitoring dashboard."""
        api_metrics = self.metrics_collector.get_api_metrics()
        system_metrics = self.metrics_collector.get_system_metrics()
        active_alerts = self.alert_manager.get_active_alerts()
        
        return {
            "api_metrics": {
                "total_requests": api_metrics.total_requests,
                "successful_requests": api_metrics.successful_requests,
                "failed_requests": api_metrics.failed_requests,
                "error_rate": api_metrics.error_rate,
                "average_response_time_ms": api_metrics.average_response_time_ms,
                "p95_response_time_ms": api_metrics.p95_response_time_ms,
                "requests_per_second": api_metrics.requests_per_second,
                "uptime_seconds": api_metrics.uptime_seconds
            },
            "system_metrics": {
                "cpu_usage_percent": system_metrics.cpu_usage_percent if system_metrics else 0,
                "memory_usage_percent": system_metrics.memory_usage_percent if system_metrics else 0,
                "memory_usage_mb": system_metrics.memory_usage_mb if system_metrics else 0,
                "disk_usage_percent": system_metrics.disk_usage_percent if system_metrics else 0,
                "active_connections": system_metrics.active_connections if system_metrics else 0
            },
            "alerts": {
                "active_count": len(active_alerts),
                "critical_count": len([a for a in active_alerts if a.level == AlertLevel.CRITICAL]),
                "warning_count": len([a for a in active_alerts if a.level == AlertLevel.WARNING]),
                "recent_alerts": [
                    {
                        "id": alert.alert_id,
                        "level": alert.level.value,
                        "message": alert.message,
                        "timestamp": alert.timestamp.isoformat()
                    }
                    for alert in active_alerts[:10]  # Last 10 alerts
                ]
            },
            "timestamp": datetime.utcnow().isoformat()
        }