"""Production monitoring and validation for pharmacogenomics pipeline."""

import asyncio
import time
import psutil
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum

from ..config import get_logger, get_config
from ..data_ingestion.data_models import PaperMetadata
from ..quality.quality_scorer import QualityScoreComponents
from ..storage.database import get_database_manager, db_session_scope
from ..storage.models import PaperModel, QualityMetrics


logger = get_logger(__name__)


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class PerformanceMetrics:
    """Real-time performance metrics."""
    papers_per_minute: float = 0.0
    avg_processing_time: float = 0.0
    success_rate: float = 100.0
    error_rate: float = 0.0
    
    # Resource utilization
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    memory_usage_mb: float = 0.0
    
    # API metrics
    api_calls_per_minute: float = 0.0
    api_success_rate: float = 100.0
    avg_api_response_time: float = 0.0
    
    # Database metrics
    db_connections_active: int = 0
    avg_db_query_time: float = 0.0
    db_success_rate: float = 100.0
    
    # Quality metrics
    avg_quality_score: float = 0.0
    high_quality_percentage: float = 0.0
    quality_score_std: float = 0.0


@dataclass
class QualityValidationResult:
    """Data quality validation result."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    completeness_score: float = 100.0
    consistency_score: float = 100.0


@dataclass
class Alert:
    """System alert."""
    level: AlertLevel
    message: str
    timestamp: datetime
    component: str
    metric_name: str
    current_value: Any
    threshold: Any
    resolved: bool = False


class PipelineMonitor:
    """Real-time pipeline performance tracking and validation."""
    
    def __init__(self, 
                 monitoring_interval: float = 30.0,
                 alert_thresholds: Dict[str, Any] = None,
                 enable_resource_monitoring: bool = True):
        """Initialize the pipeline monitor.
        
        Args:
            monitoring_interval: Seconds between monitoring cycles
            alert_thresholds: Custom alert thresholds
            enable_resource_monitoring: Enable system resource monitoring
        """
        self.monitoring_interval = monitoring_interval
        self.enable_resource_monitoring = enable_resource_monitoring
        
        # Default alert thresholds
        self.alert_thresholds = {
            'error_rate_warning': 5.0,  # %
            'error_rate_critical': 15.0,  # %
            'success_rate_warning': 90.0,  # %
            'success_rate_critical': 80.0,  # %
            'cpu_usage_warning': 80.0,  # %
            'cpu_usage_critical': 95.0,  # %
            'memory_usage_warning': 80.0,  # %
            'memory_usage_critical': 95.0,  # %
            'api_response_time_warning': 5.0,  # seconds
            'api_response_time_critical': 10.0,  # seconds
            'quality_score_warning': 60.0,  # minimum average
            'quality_score_critical': 40.0,  # minimum average
            'papers_per_minute_warning': 10.0,  # minimum rate
            'papers_per_minute_critical': 5.0,  # minimum rate
        }
        
        if alert_thresholds:
            self.alert_thresholds.update(alert_thresholds)
        
        # Monitoring state
        self.is_monitoring = False
        self.monitoring_thread = None
        self.metrics_history = deque(maxlen=1000)  # Keep last 1000 measurements
        self.active_alerts = []
        self.alert_callbacks: List[Callable[[Alert], None]] = []
        
        # Performance tracking
        self.processing_times = deque(maxlen=100)
        self.api_response_times = deque(maxlen=100)
        self.db_query_times = deque(maxlen=100)
        self.quality_scores = deque(maxlen=1000)
        
        # Error tracking
        self.error_counts = defaultdict(int)
        self.error_history = deque(maxlen=1000)
        
        # Validation rules
        self.validation_rules = self._initialize_validation_rules()
        
        logger.info("PipelineMonitor initialized")
    
    def start_monitoring(self):
        """Start real-time monitoring."""
        if self.is_monitoring:
            logger.warning("Monitoring is already running")
            return
        
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        logger.info(f"Started pipeline monitoring (interval: {self.monitoring_interval}s)")
    
    def stop_monitoring(self):
        """Stop real-time monitoring."""
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        
        logger.info("Stopped pipeline monitoring")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                # Collect metrics
                metrics = self._collect_metrics()
                self.metrics_history.append(metrics)
                
                # Check for alerts
                self._check_alerts(metrics)
                
                # Log metrics periodically
                if len(self.metrics_history) % 10 == 0:  # Every 10 cycles
                    self._log_metrics_summary(metrics)
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {str(e)}")
            
            time.sleep(self.monitoring_interval)
    
    def _collect_metrics(self) -> PerformanceMetrics:
        """Collect current performance metrics."""
        metrics = PerformanceMetrics()
        
        # Calculate processing rate
        if self.processing_times:
            recent_times = list(self.processing_times)[-10:]  # Last 10 measurements
            metrics.avg_processing_time = sum(recent_times) / len(recent_times)
            
            # Estimate papers per minute based on recent performance
            if metrics.avg_processing_time > 0:
                metrics.papers_per_minute = 60.0 / metrics.avg_processing_time
        
        # Calculate success/error rates
        if self.error_history:
            recent_errors = [e for e in self.error_history 
                           if e['timestamp'] > datetime.utcnow() - timedelta(minutes=5)]
            total_recent = len([e for e in self.error_history 
                              if e['timestamp'] > datetime.utcnow() - timedelta(minutes=5)])
            
            if total_recent > 0:
                metrics.error_rate = (len(recent_errors) / total_recent) * 100
                metrics.success_rate = 100 - metrics.error_rate
        
        # System resource metrics
        if self.enable_resource_monitoring:
            try:
                metrics.cpu_usage = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                metrics.memory_usage = memory.percent
                metrics.memory_usage_mb = memory.used / (1024 * 1024)
            except Exception as e:
                logger.warning(f"Failed to collect system metrics: {str(e)}")
        
        # API metrics
        if self.api_response_times:
            recent_api_times = list(self.api_response_times)[-10:]
            metrics.avg_api_response_time = sum(recent_api_times) / len(recent_api_times)
            
            # Calculate API calls per minute
            api_calls_last_minute = len([t for t in self.api_response_times 
                                       if time.time() - t < 60])
            metrics.api_calls_per_minute = api_calls_last_minute
        
        # Database metrics
        if self.db_query_times:
            recent_db_times = list(self.db_query_times)[-10:]
            metrics.avg_db_query_time = sum(recent_db_times) / len(recent_db_times)
        
        # Quality metrics
        if self.quality_scores:
            recent_scores = list(self.quality_scores)[-100:]  # Last 100 papers
            metrics.avg_quality_score = sum(recent_scores) / len(recent_scores)
            
            # Calculate quality distribution
            high_quality = sum(1 for score in recent_scores if score >= 75)
            metrics.high_quality_percentage = (high_quality / len(recent_scores)) * 100
            
            # Calculate standard deviation
            mean = metrics.avg_quality_score
            variance = sum((score - mean) ** 2 for score in recent_scores) / len(recent_scores)
            metrics.quality_score_std = variance ** 0.5
        
        return metrics
    
    def _check_alerts(self, metrics: PerformanceMetrics):
        """Check metrics against thresholds and generate alerts."""
        current_time = datetime.utcnow()
        
        # Error rate alerts
        if metrics.error_rate >= self.alert_thresholds['error_rate_critical']:
            self._create_alert(AlertLevel.CRITICAL, "Error rate critical", 
                             "pipeline", "error_rate", metrics.error_rate, 
                             self.alert_thresholds['error_rate_critical'])
        elif metrics.error_rate >= self.alert_thresholds['error_rate_warning']:
            self._create_alert(AlertLevel.WARNING, "Error rate elevated", 
                             "pipeline", "error_rate", metrics.error_rate, 
                             self.alert_thresholds['error_rate_warning'])
        
        # Success rate alerts
        if metrics.success_rate <= self.alert_thresholds['success_rate_critical']:
            self._create_alert(AlertLevel.CRITICAL, "Success rate critical", 
                             "pipeline", "success_rate", metrics.success_rate, 
                             self.alert_thresholds['success_rate_critical'])
        elif metrics.success_rate <= self.alert_thresholds['success_rate_warning']:
            self._create_alert(AlertLevel.WARNING, "Success rate low", 
                             "pipeline", "success_rate", metrics.success_rate, 
                             self.alert_thresholds['success_rate_warning'])
        
        # Resource alerts
        if self.enable_resource_monitoring:
            if metrics.cpu_usage >= self.alert_thresholds['cpu_usage_critical']:
                self._create_alert(AlertLevel.CRITICAL, "CPU usage critical", 
                                 "system", "cpu_usage", metrics.cpu_usage, 
                                 self.alert_thresholds['cpu_usage_critical'])
            elif metrics.cpu_usage >= self.alert_thresholds['cpu_usage_warning']:
                self._create_alert(AlertLevel.WARNING, "CPU usage high", 
                                 "system", "cpu_usage", metrics.cpu_usage, 
                                 self.alert_thresholds['cpu_usage_warning'])
            
            if metrics.memory_usage >= self.alert_thresholds['memory_usage_critical']:
                self._create_alert(AlertLevel.CRITICAL, "Memory usage critical", 
                                 "system", "memory_usage", metrics.memory_usage, 
                                 self.alert_thresholds['memory_usage_critical'])
            elif metrics.memory_usage >= self.alert_thresholds['memory_usage_warning']:
                self._create_alert(AlertLevel.WARNING, "Memory usage high", 
                                 "system", "memory_usage", metrics.memory_usage, 
                                 self.alert_thresholds['memory_usage_warning'])
        
        # API performance alerts
        if metrics.avg_api_response_time >= self.alert_thresholds['api_response_time_critical']:
            self._create_alert(AlertLevel.CRITICAL, "API response time critical", 
                             "api", "response_time", metrics.avg_api_response_time, 
                             self.alert_thresholds['api_response_time_critical'])
        elif metrics.avg_api_response_time >= self.alert_thresholds['api_response_time_warning']:
            self._create_alert(AlertLevel.WARNING, "API response time slow", 
                             "api", "response_time", metrics.avg_api_response_time, 
                             self.alert_thresholds['api_response_time_warning'])
        
        # Quality alerts
        if metrics.avg_quality_score <= self.alert_thresholds['quality_score_critical']:
            self._create_alert(AlertLevel.CRITICAL, "Quality score critical", 
                             "quality", "avg_score", metrics.avg_quality_score, 
                             self.alert_thresholds['quality_score_critical'])
        elif metrics.avg_quality_score <= self.alert_thresholds['quality_score_warning']:
            self._create_alert(AlertLevel.WARNING, "Quality score low", 
                             "quality", "avg_score", metrics.avg_quality_score, 
                             self.alert_thresholds['quality_score_warning'])
        
        # Performance alerts
        if metrics.papers_per_minute <= self.alert_thresholds['papers_per_minute_critical']:
            self._create_alert(AlertLevel.CRITICAL, "Processing rate critical", 
                             "performance", "papers_per_minute", metrics.papers_per_minute, 
                             self.alert_thresholds['papers_per_minute_critical'])
        elif metrics.papers_per_minute <= self.alert_thresholds['papers_per_minute_warning']:
            self._create_alert(AlertLevel.WARNING, "Processing rate low", 
                             "performance", "papers_per_minute", metrics.papers_per_minute, 
                             self.alert_thresholds['papers_per_minute_warning'])
    
    def _create_alert(self, level: AlertLevel, message: str, component: str, 
                     metric_name: str, current_value: Any, threshold: Any):
        """Create and process an alert."""
        # Check if similar alert already exists
        existing_alert = next((alert for alert in self.active_alerts 
                             if alert.component == component and 
                             alert.metric_name == metric_name and 
                             not alert.resolved), None)
        
        if existing_alert:
            # Update existing alert
            existing_alert.current_value = current_value
            existing_alert.timestamp = datetime.utcnow()
        else:
            # Create new alert
            alert = Alert(
                level=level,
                message=message,
                timestamp=datetime.utcnow(),
                component=component,
                metric_name=metric_name,
                current_value=current_value,
                threshold=threshold
            )
            
            self.active_alerts.append(alert)
            
            # Notify callbacks
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    logger.error(f"Alert callback failed: {str(e)}")
            
            logger.log(
                level.name.upper() if level != AlertLevel.INFO else "INFO",
                f"ALERT [{level.value.upper()}] {component}.{metric_name}: {message} "
                f"(current: {current_value}, threshold: {threshold})"
            )
    
    def _log_metrics_summary(self, metrics: PerformanceMetrics):
        """Log periodic metrics summary."""
        logger.info(
            f"Pipeline Metrics - "
            f"Rate: {metrics.papers_per_minute:.1f} papers/min, "
            f"Success: {metrics.success_rate:.1f}%, "
            f"Avg Quality: {metrics.avg_quality_score:.1f}, "
            f"CPU: {metrics.cpu_usage:.1f}%, "
            f"Memory: {metrics.memory_usage:.1f}%"
        )
    
    def record_processing_time(self, processing_time: float):
        """Record processing time for a batch or paper."""
        self.processing_times.append(processing_time)
    
    def record_api_response_time(self, response_time: float):
        """Record API response time."""
        self.api_response_times.append(response_time)
    
    def record_db_query_time(self, query_time: float):
        """Record database query time."""
        self.db_query_times.append(query_time)
    
    def record_quality_score(self, quality_score: float):
        """Record quality score for monitoring."""
        self.quality_scores.append(quality_score)
    
    def record_error(self, error_type: str, error_message: str, component: str):
        """Record an error for monitoring."""
        self.error_counts[error_type] += 1
        self.error_history.append({
            'type': error_type,
            'message': error_message,
            'component': component,
            'timestamp': datetime.utcnow()
        })
    
    def validate_paper_data(self, paper: PaperMetadata, 
                          quality_scores: QualityScoreComponents) -> QualityValidationResult:
        """Validate paper data quality."""
        result = QualityValidationResult(is_valid=True)
        
        # Run validation rules
        for rule_name, rule_func in self.validation_rules.items():
            try:
                rule_result = rule_func(paper, quality_scores)
                if not rule_result['valid']:
                    result.is_valid = False
                    if rule_result['severity'] == 'error':
                        result.errors.append(f"{rule_name}: {rule_result['message']}")
                    else:
                        result.warnings.append(f"{rule_name}: {rule_result['message']}")
            except Exception as e:
                logger.error(f"Validation rule {rule_name} failed: {str(e)}")
                result.warnings.append(f"{rule_name}: Validation rule failed")
        
        # Calculate completeness score
        result.completeness_score = self._calculate_completeness_score(paper)
        
        # Calculate consistency score
        result.consistency_score = self._calculate_consistency_score(paper, quality_scores)
        
        return result
    
    def _initialize_validation_rules(self) -> Dict[str, Callable]:
        """Initialize data validation rules."""
        return {
            'required_fields': self._validate_required_fields,
            'quality_score_range': self._validate_quality_score_range,
            'publication_date': self._validate_publication_date,
            'journal_name': self._validate_journal_name,
            'duplicate_detection': self._validate_duplicate_detection,
        }
    
    def _validate_required_fields(self, paper: PaperMetadata, 
                                quality_scores: QualityScoreComponents) -> Dict[str, Any]:
        """Validate required fields are present."""
        missing_fields = []
        
        if not paper.pmid:
            missing_fields.append('pmid')
        if not paper.title:
            missing_fields.append('title')
        if not paper.abstract:
            missing_fields.append('abstract')
        
        if missing_fields:
            return {
                'valid': False,
                'severity': 'error',
                'message': f"Missing required fields: {', '.join(missing_fields)}"
            }
        
        return {'valid': True, 'message': 'All required fields present'}
    
    def _validate_quality_score_range(self, paper: PaperMetadata, 
                                    quality_scores: QualityScoreComponents) -> Dict[str, Any]:
        """Validate quality scores are in valid range."""
        if not (0 <= quality_scores.total_score <= 100):
            return {
                'valid': False,
                'severity': 'error',
                'message': f"Quality score {quality_scores.total_score} out of range [0-100]"
            }
        
        if not (0 <= quality_scores.confidence_level <= 1):
            return {
                'valid': False,
                'severity': 'error',
                'message': f"Confidence level {quality_scores.confidence_level} out of range [0-1]"
            }
        
        return {'valid': True, 'message': 'Quality scores in valid range'}
    
    def _validate_publication_date(self, paper: PaperMetadata, 
                                 quality_scores: QualityScoreComponents) -> Dict[str, Any]:
        """Validate publication date is reasonable."""
        if not paper.publication_date:
            return {
                'valid': True,
                'severity': 'warning',
                'message': 'Publication date missing'
            }
        
        # Check if date is in the future
        if paper.publication_date > datetime.utcnow():
            return {
                'valid': False,
                'severity': 'warning',
                'message': 'Publication date is in the future'
            }
        
        # Check if date is too old (before 1950)
        if paper.publication_date.year < 1950:
            return {
                'valid': False,
                'severity': 'warning',
                'message': f'Publication date {paper.publication_date.year} seems too old'
            }
        
        return {'valid': True, 'message': 'Publication date valid'}
    
    def _validate_journal_name(self, paper: PaperMetadata, 
                             quality_scores: QualityScoreComponents) -> Dict[str, Any]:
        """Validate journal name standardization."""
        if not paper.journal or not paper.journal.title:
            return {
                'valid': True,
                'severity': 'warning',
                'message': 'Journal name missing'
            }
        
        # Check for common journal name issues
        journal_name = paper.journal.title.lower()
        
        if len(journal_name) < 3:
            return {
                'valid': False,
                'severity': 'warning',
                'message': 'Journal name too short'
            }
        
        if journal_name.count('.') > 5:  # Too many abbreviations
            return {
                'valid': True,
                'severity': 'warning',
                'message': 'Journal name may need standardization'
            }
        
        return {'valid': True, 'message': 'Journal name valid'}
    
    def _validate_duplicate_detection(self, paper: PaperMetadata, 
                                    quality_scores: QualityScoreComponents) -> Dict[str, Any]:
        """Check for potential duplicates."""
        try:
            # Check database for existing paper
            db_manager = get_database_manager()
            with db_session_scope() as session:
                existing = session.query(PaperModel).filter_by(pmid=paper.pmid).first()
                if existing:
                    return {
                        'valid': False,
                        'severity': 'warning',
                        'message': f'Paper {paper.pmid} already exists in database'
                    }
        except Exception as e:
            logger.warning(f"Duplicate check failed: {str(e)}")
        
        return {'valid': True, 'message': 'No duplicates detected'}
    
    def _calculate_completeness_score(self, paper: PaperMetadata) -> float:
        """Calculate data completeness score."""
        total_fields = 10
        present_fields = 0
        
        if paper.pmid:
            present_fields += 1
        if paper.title:
            present_fields += 1
        if paper.abstract:
            present_fields += 1
        if paper.authors:
            present_fields += 1
        if paper.journal:
            present_fields += 1
        if paper.publication_date:
            present_fields += 1
        if paper.doi:
            present_fields += 1
        if paper.mesh_terms:
            present_fields += 1
        if paper.keywords:
            present_fields += 1
        if paper.language:
            present_fields += 1
        
        return (present_fields / total_fields) * 100
    
    def _calculate_consistency_score(self, paper: PaperMetadata, 
                                   quality_scores: QualityScoreComponents) -> float:
        """Calculate data consistency score."""
        consistency_score = 100.0
        
        # Check title-abstract consistency (basic check)
        if paper.title and paper.abstract:
            title_words = set(paper.title.lower().split())
            abstract_words = set(paper.abstract.lower().split())
            
            # Check if title words appear in abstract
            title_in_abstract = len(title_words.intersection(abstract_words))
            if title_in_abstract / len(title_words) < 0.3:  # Less than 30% overlap
                consistency_score -= 10
        
        # Check author-affiliation consistency
        if paper.authors:
            authors_with_affiliation = sum(1 for author in paper.authors if author.affiliation)
            if authors_with_affiliation / len(paper.authors) < 0.5:  # Less than 50% have affiliations
                consistency_score -= 5
        
        # Check MeSH terms relevance (basic check)
        if paper.mesh_terms and paper.title:
            # This is a simplified check - in practice, you'd use more sophisticated NLP
            title_lower = paper.title.lower()
            relevant_mesh = sum(1 for term in paper.mesh_terms 
                              if term.descriptor_name.lower() in title_lower)
            if relevant_mesh == 0 and len(paper.mesh_terms) > 0:
                consistency_score -= 5
        
        return max(0.0, consistency_score)
    
    def get_current_metrics(self) -> Optional[PerformanceMetrics]:
        """Get current performance metrics."""
        if self.metrics_history:
            return self.metrics_history[-1]
        return None
    
    def get_metrics_history(self, minutes: int = 60) -> List[PerformanceMetrics]:
        """Get metrics history for specified time period."""
        cutoff_time = datetime.utcnow() - timedelta(minutes=minutes)
        return [metrics for metrics in self.metrics_history 
                if hasattr(metrics, 'timestamp') and metrics.timestamp > cutoff_time]
    
    def get_active_alerts(self) -> List[Alert]:
        """Get currently active alerts."""
        return [alert for alert in self.active_alerts if not alert.resolved]
    
    def resolve_alert(self, alert_id: int):
        """Mark an alert as resolved."""
        if 0 <= alert_id < len(self.active_alerts):
            self.active_alerts[alert_id].resolved = True
            logger.info(f"Alert {alert_id} resolved")
    
    def add_alert_callback(self, callback: Callable[[Alert], None]):
        """Add callback for alert notifications."""
        self.alert_callbacks.append(callback)
    
    def get_error_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get error summary for specified time period."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        recent_errors = [error for error in self.error_history 
                        if error['timestamp'] > cutoff_time]
        
        error_by_type = defaultdict(int)
        error_by_component = defaultdict(int)
        
        for error in recent_errors:
            error_by_type[error['type']] += 1
            error_by_component[error['component']] += 1
        
        return {
            'total_errors': len(recent_errors),
            'error_by_type': dict(error_by_type),
            'error_by_component': dict(error_by_component),
            'error_rate': len(recent_errors) / max(1, hours) if recent_errors else 0
        }