"""A/B testing framework for model comparison with statistical significance testing."""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import random
import threading
from enum import Enum
import logging

# Statistical testing
try:
    from scipy import stats
    from scipy.stats import chi2_contingency, ttest_ind, mannwhitneyu
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from ..config import get_logger
from .model_server import ModelServer, PredictionRequest, PredictionResponse


logger = get_logger(__name__)


class ExperimentStatus(Enum):
    """Experiment status enumeration."""
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class TrafficSplitStrategy(Enum):
    """Traffic splitting strategy."""
    RANDOM = "random"
    WEIGHTED = "weighted"
    STICKY = "sticky"  # User-based consistency
    GRADUAL = "gradual"  # Gradual rollout


@dataclass
class ExperimentVariant:
    """A/B test experiment variant."""
    name: str
    model_name: str
    model_version: str
    traffic_percentage: float
    description: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentConfig:
    """A/B test experiment configuration."""
    experiment_id: str
    name: str
    description: str
    variants: List[ExperimentVariant]
    traffic_split_strategy: TrafficSplitStrategy = TrafficSplitStrategy.RANDOM
    success_metrics: List[str] = field(default_factory=lambda: ['accuracy', 'latency'])
    minimum_sample_size: int = 1000
    confidence_level: float = 0.95
    statistical_power: float = 0.8
    minimum_effect_size: float = 0.05
    max_duration_days: int = 30
    auto_stop_on_significance: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    created_by: str = "system"


@dataclass
class ExperimentMetrics:
    """Metrics for an experiment variant."""
    variant_name: str
    total_requests: int = 0
    successful_predictions: int = 0
    failed_predictions: int = 0
    total_latency_ms: float = 0.0
    accuracy_scores: List[float] = field(default_factory=list)
    conversion_events: int = 0  # Custom success events
    custom_metrics: Dict[str, List[float]] = field(default_factory=dict)
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_requests == 0:
            return 0.0
        return self.successful_predictions / self.total_requests
    
    @property
    def average_latency_ms(self) -> float:
        """Calculate average latency."""
        if self.successful_predictions == 0:
            return 0.0
        return self.total_latency_ms / self.successful_predictions
    
    @property
    def average_accuracy(self) -> float:
        """Calculate average accuracy."""
        if not self.accuracy_scores:
            return 0.0
        return np.mean(self.accuracy_scores)
    
    @property
    def conversion_rate(self) -> float:
        """Calculate conversion rate."""
        if self.total_requests == 0:
            return 0.0
        return self.conversion_events / self.total_requests


@dataclass
class StatisticalTestResult:
    """Result of statistical significance test."""
    test_name: str
    metric: str
    variant_a: str
    variant_b: str
    statistic: float
    p_value: float
    is_significant: bool
    confidence_level: float
    effect_size: float
    interpretation: str


@dataclass
class ExperimentResult:
    """Complete experiment result."""
    experiment_id: str
    status: ExperimentStatus
    start_time: datetime
    end_time: Optional[datetime]
    duration_hours: float
    variant_metrics: Dict[str, ExperimentMetrics]
    statistical_tests: List[StatisticalTestResult]
    winner: Optional[str] = None
    confidence: Optional[float] = None
    recommendation: Optional[str] = None


class TrafficSplitter:
    """Handles traffic splitting for A/B tests."""
    
    def __init__(self, strategy: TrafficSplitStrategy = TrafficSplitStrategy.RANDOM):
        """Initialize traffic splitter.
        
        Args:
            strategy: Traffic splitting strategy
        """
        self.strategy = strategy
        self.user_assignments = {}  # For sticky assignments
        self.lock = threading.Lock()
    
    def assign_variant(self, experiment_config: ExperimentConfig,
                      user_id: Optional[str] = None,
                      request_metadata: Optional[Dict[str, Any]] = None) -> ExperimentVariant:
        """Assign a variant to a request.
        
        Args:
            experiment_config: Experiment configuration
            user_id: Optional user ID for sticky assignment
            request_metadata: Optional request metadata
            
        Returns:
            Assigned experiment variant
        """
        if self.strategy == TrafficSplitStrategy.RANDOM:
            return self._random_assignment(experiment_config)
        
        elif self.strategy == TrafficSplitStrategy.WEIGHTED:
            return self._weighted_assignment(experiment_config)
        
        elif self.strategy == TrafficSplitStrategy.STICKY:
            return self._sticky_assignment(experiment_config, user_id)
        
        elif self.strategy == TrafficSplitStrategy.GRADUAL:
            return self._gradual_assignment(experiment_config)
        
        else:
            return self._random_assignment(experiment_config)
    
    def _random_assignment(self, config: ExperimentConfig) -> ExperimentVariant:
        """Random assignment with equal probability."""
        return random.choice(config.variants)
    
    def _weighted_assignment(self, config: ExperimentConfig) -> ExperimentVariant:
        """Weighted assignment based on traffic percentages."""
        rand_val = random.random() * 100
        cumulative = 0.0
        
        for variant in config.variants:
            cumulative += variant.traffic_percentage
            if rand_val <= cumulative:
                return variant
        
        # Fallback to last variant
        return config.variants[-1]
    
    def _sticky_assignment(self, config: ExperimentConfig, 
                          user_id: Optional[str]) -> ExperimentVariant:
        """Sticky assignment that maintains consistency per user."""
        if not user_id:
            return self._weighted_assignment(config)
        
        with self.lock:
            key = f"{config.experiment_id}:{user_id}"
            
            if key in self.user_assignments:
                variant_name = self.user_assignments[key]
                # Find variant by name
                for variant in config.variants:
                    if variant.name == variant_name:
                        return variant
            
            # New assignment
            variant = self._weighted_assignment(config)
            self.user_assignments[key] = variant.name
            return variant
    
    def _gradual_assignment(self, config: ExperimentConfig) -> ExperimentVariant:
        """Gradual rollout assignment."""
        # Simple implementation: increase treatment percentage over time
        days_running = (datetime.now() - config.created_at).days
        max_days = config.max_duration_days
        
        if len(config.variants) != 2:
            return self._weighted_assignment(config)
        
        control_variant = config.variants[0]
        treatment_variant = config.variants[1]
        
        # Gradually increase treatment percentage
        progress = min(days_running / max_days, 1.0)
        treatment_percentage = treatment_variant.traffic_percentage * progress
        
        if random.random() * 100 < treatment_percentage:
            return treatment_variant
        else:
            return control_variant


class StatisticalAnalyzer:
    """Performs statistical analysis for A/B tests."""
    
    def __init__(self, confidence_level: float = 0.95):
        """Initialize statistical analyzer.
        
        Args:
            confidence_level: Confidence level for statistical tests
        """
        if not SCIPY_AVAILABLE:
            logger.warning("SciPy not available, statistical tests will be limited")
        
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
    
    def compare_success_rates(self, metrics_a: ExperimentMetrics,
                            metrics_b: ExperimentMetrics) -> StatisticalTestResult:
        """Compare success rates between two variants using chi-square test.
        
        Args:
            metrics_a: Metrics for variant A
            metrics_b: Metrics for variant B
            
        Returns:
            Statistical test result
        """
        if not SCIPY_AVAILABLE:
            return self._dummy_test_result("success_rate", metrics_a.variant_name, metrics_b.variant_name)
        
        # Create contingency table
        successes_a = metrics_a.successful_predictions
        failures_a = metrics_a.total_requests - successes_a
        successes_b = metrics_b.successful_predictions
        failures_b = metrics_b.total_requests - successes_b
        
        contingency_table = np.array([
            [successes_a, failures_a],
            [successes_b, failures_b]
        ])
        
        # Perform chi-square test
        try:
            chi2, p_value, dof, expected = chi2_contingency(contingency_table)
            
            # Calculate effect size (Cramér's V)
            n = contingency_table.sum()
            effect_size = np.sqrt(chi2 / (n * (min(contingency_table.shape) - 1)))
            
            is_significant = p_value < self.alpha
            
            # Interpretation
            rate_a = metrics_a.success_rate
            rate_b = metrics_b.success_rate
            
            if is_significant:
                if rate_b > rate_a:
                    interpretation = f"Variant B has significantly higher success rate ({rate_b:.3f} vs {rate_a:.3f})"
                else:
                    interpretation = f"Variant A has significantly higher success rate ({rate_a:.3f} vs {rate_b:.3f})"
            else:
                interpretation = f"No significant difference in success rates ({rate_a:.3f} vs {rate_b:.3f})"
            
            return StatisticalTestResult(
                test_name="Chi-square test",
                metric="success_rate",
                variant_a=metrics_a.variant_name,
                variant_b=metrics_b.variant_name,
                statistic=chi2,
                p_value=p_value,
                is_significant=is_significant,
                confidence_level=self.confidence_level,
                effect_size=effect_size,
                interpretation=interpretation
            )
            
        except Exception as e:
            logger.error(f"Chi-square test failed: {e}")
            return self._dummy_test_result("success_rate", metrics_a.variant_name, metrics_b.variant_name)
    
    def compare_continuous_metric(self, values_a: List[float], values_b: List[float],
                                metric_name: str, variant_a: str, variant_b: str) -> StatisticalTestResult:
        """Compare continuous metrics using t-test or Mann-Whitney U test.
        
        Args:
            values_a: Values for variant A
            values_b: Values for variant B
            metric_name: Name of the metric
            variant_a: Name of variant A
            variant_b: Name of variant B
            
        Returns:
            Statistical test result
        """
        if not SCIPY_AVAILABLE or not values_a or not values_b:
            return self._dummy_test_result(metric_name, variant_a, variant_b)
        
        try:
            # Check for normality (simplified)
            if len(values_a) > 8 and len(values_b) > 8:
                # Use Shapiro-Wilk test for normality
                _, p_norm_a = stats.shapiro(values_a[:5000])  # Limit for performance
                _, p_norm_b = stats.shapiro(values_b[:5000])
                
                # If both are normally distributed, use t-test
                if p_norm_a > 0.05 and p_norm_b > 0.05:
                    statistic, p_value = ttest_ind(values_a, values_b)
                    test_name = "Independent t-test"
                else:
                    # Use Mann-Whitney U test for non-normal data
                    statistic, p_value = mannwhitneyu(values_a, values_b, alternative='two-sided')
                    test_name = "Mann-Whitney U test"
            else:
                # Small samples, use Mann-Whitney U
                statistic, p_value = mannwhitneyu(values_a, values_b, alternative='two-sided')
                test_name = "Mann-Whitney U test"
            
            # Calculate effect size (Cohen's d for t-test, r for Mann-Whitney)
            mean_a = np.mean(values_a)
            mean_b = np.mean(values_b)
            
            if test_name == "Independent t-test":
                pooled_std = np.sqrt(((len(values_a) - 1) * np.var(values_a, ddof=1) + 
                                    (len(values_b) - 1) * np.var(values_b, ddof=1)) / 
                                   (len(values_a) + len(values_b) - 2))
                effect_size = (mean_b - mean_a) / pooled_std if pooled_std > 0 else 0.0
            else:
                # For Mann-Whitney, use r = Z / sqrt(N)
                n_total = len(values_a) + len(values_b)
                z_score = stats.norm.ppf(1 - p_value / 2)  # Approximate
                effect_size = z_score / np.sqrt(n_total)
            
            is_significant = p_value < self.alpha
            
            # Interpretation
            if is_significant:
                if mean_b > mean_a:
                    interpretation = f"Variant B has significantly higher {metric_name} ({mean_b:.3f} vs {mean_a:.3f})"
                else:
                    interpretation = f"Variant A has significantly higher {metric_name} ({mean_a:.3f} vs {mean_b:.3f})"
            else:
                interpretation = f"No significant difference in {metric_name} ({mean_a:.3f} vs {mean_b:.3f})"
            
            return StatisticalTestResult(
                test_name=test_name,
                metric=metric_name,
                variant_a=variant_a,
                variant_b=variant_b,
                statistic=statistic,
                p_value=p_value,
                is_significant=is_significant,
                confidence_level=self.confidence_level,
                effect_size=effect_size,
                interpretation=interpretation
            )
            
        except Exception as e:
            logger.error(f"Statistical test failed: {e}")
            return self._dummy_test_result(metric_name, variant_a, variant_b)
    
    def _dummy_test_result(self, metric: str, variant_a: str, variant_b: str) -> StatisticalTestResult:
        """Create dummy test result when statistical tests are not available."""
        return StatisticalTestResult(
            test_name="Descriptive comparison",
            metric=metric,
            variant_a=variant_a,
            variant_b=variant_b,
            statistic=0.0,
            p_value=1.0,
            is_significant=False,
            confidence_level=self.confidence_level,
            effect_size=0.0,
            interpretation="Statistical testing not available"
        )
    
    def calculate_sample_size(self, baseline_rate: float, minimum_effect_size: float,
                            power: float = 0.8, alpha: float = 0.05) -> int:
        """Calculate required sample size for A/B test.
        
        Args:
            baseline_rate: Baseline conversion/success rate
            minimum_effect_size: Minimum detectable effect size
            power: Statistical power (1 - β)
            alpha: Type I error rate
            
        Returns:
            Required sample size per variant
        """
        if not SCIPY_AVAILABLE:
            # Simple approximation
            return max(1000, int(1 / (minimum_effect_size ** 2) * 16))
        
        try:
            # Use normal approximation for proportions
            z_alpha = stats.norm.ppf(1 - alpha / 2)
            z_beta = stats.norm.ppf(power)
            
            p1 = baseline_rate
            p2 = baseline_rate + minimum_effect_size
            
            p_pooled = (p1 + p2) / 2
            
            numerator = (z_alpha * np.sqrt(2 * p_pooled * (1 - p_pooled)) + 
                        z_beta * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
            
            denominator = (p2 - p1) ** 2
            
            sample_size = int(np.ceil(numerator / denominator))
            
            return max(100, sample_size)  # Minimum 100 samples
            
        except Exception as e:
            logger.error(f"Sample size calculation failed: {e}")
            return 1000  # Default fallback


class ABTestExperiment:
    """A/B test experiment manager."""
    
    def __init__(self, config: ExperimentConfig, model_server: ModelServer):
        """Initialize A/B test experiment.
        
        Args:
            config: Experiment configuration
            model_server: Model server for predictions
        """
        self.config = config
        self.model_server = model_server
        self.status = ExperimentStatus.DRAFT
        self.start_time = None
        self.end_time = None
        
        # Initialize components
        self.traffic_splitter = TrafficSplitter(config.traffic_split_strategy)
        self.statistical_analyzer = StatisticalAnalyzer(config.confidence_level)
        
        # Metrics tracking
        self.variant_metrics = {
            variant.name: ExperimentMetrics(variant_name=variant.name)
            for variant in config.variants
        }
        
        self.lock = threading.Lock()
        
        logger.info(f"A/B test experiment initialized: {config.experiment_id}")
    
    def start(self) -> bool:
        """Start the experiment.
        
        Returns:
            True if started successfully
        """
        with self.lock:
            if self.status != ExperimentStatus.DRAFT:
                logger.warning(f"Cannot start experiment in status: {self.status}")
                return False
            
            self.status = ExperimentStatus.RUNNING
            self.start_time = datetime.now()
            
            logger.info(f"A/B test experiment started: {self.config.experiment_id}")
            return True
    
    def stop(self) -> bool:
        """Stop the experiment.
        
        Returns:
            True if stopped successfully
        """
        with self.lock:
            if self.status != ExperimentStatus.RUNNING:
                logger.warning(f"Cannot stop experiment in status: {self.status}")
                return False
            
            self.status = ExperimentStatus.COMPLETED
            self.end_time = datetime.now()
            
            logger.info(f"A/B test experiment stopped: {self.config.experiment_id}")
            return True
    
    def process_request(self, request: PredictionRequest,
                       user_id: Optional[str] = None) -> Tuple[PredictionResponse, str]:
        """Process a prediction request through the A/B test.
        
        Args:
            request: Prediction request
            user_id: Optional user ID for sticky assignment
            
        Returns:
            Tuple of (prediction response, assigned variant name)
        """
        if self.status != ExperimentStatus.RUNNING:
            # Use default model if experiment not running
            response = self.model_server.predict(request)
            return response, "default"
        
        # Assign variant
        assigned_variant = self.traffic_splitter.assign_variant(
            self.config, user_id, request.metadata
        )
        
        # Update request with variant model
        variant_request = PredictionRequest(
            request_id=request.request_id,
            features=request.features,
            model_name=assigned_variant.model_name,
            model_version=assigned_variant.model_version,
            return_probabilities=request.return_probabilities,
            metadata=request.metadata
        )
        
        # Make prediction
        response = self.model_server.predict(variant_request)
        
        # Update metrics
        self._update_metrics(assigned_variant.name, response)
        
        return response, assigned_variant.name
    
    def _update_metrics(self, variant_name: str, response: PredictionResponse) -> None:
        """Update metrics for a variant.
        
        Args:
            variant_name: Name of the variant
            response: Prediction response
        """
        with self.lock:
            metrics = self.variant_metrics[variant_name]
            
            metrics.total_requests += 1
            
            if response.error is None:
                metrics.successful_predictions += 1
                metrics.total_latency_ms += response.processing_time_ms
                
                # Add accuracy if available (would need ground truth)
                # metrics.accuracy_scores.append(accuracy)
            else:
                metrics.failed_predictions += 1
    
    def record_conversion(self, variant_name: str, user_id: str,
                         custom_metrics: Optional[Dict[str, float]] = None) -> None:
        """Record a conversion event for a variant.
        
        Args:
            variant_name: Name of the variant
            user_id: User ID
            custom_metrics: Optional custom metrics
        """
        with self.lock:
            if variant_name in self.variant_metrics:
                metrics = self.variant_metrics[variant_name]
                metrics.conversion_events += 1
                
                if custom_metrics:
                    for metric_name, value in custom_metrics.items():
                        if metric_name not in metrics.custom_metrics:
                            metrics.custom_metrics[metric_name] = []
                        metrics.custom_metrics[metric_name].append(value)
    
    def analyze_results(self) -> ExperimentResult:
        """Analyze experiment results and determine winner.
        
        Returns:
            Experiment result with statistical analysis
        """
        logger.info(f"Analyzing results for experiment: {self.config.experiment_id}")
        
        statistical_tests = []
        
        # Compare all pairs of variants
        variant_names = list(self.variant_metrics.keys())
        
        for i in range(len(variant_names)):
            for j in range(i + 1, len(variant_names)):
                variant_a = variant_names[i]
                variant_b = variant_names[j]
                
                metrics_a = self.variant_metrics[variant_a]
                metrics_b = self.variant_metrics[variant_b]
                
                # Compare success rates
                success_test = self.statistical_analyzer.compare_success_rates(metrics_a, metrics_b)
                statistical_tests.append(success_test)
                
                # Compare latency if we have data
                if (metrics_a.successful_predictions > 0 and metrics_b.successful_predictions > 0):
                    latency_a = [metrics_a.average_latency_ms] * metrics_a.successful_predictions
                    latency_b = [metrics_b.average_latency_ms] * metrics_b.successful_predictions
                    
                    latency_test = self.statistical_analyzer.compare_continuous_metric(
                        latency_a, latency_b, "latency", variant_a, variant_b
                    )
                    statistical_tests.append(latency_test)
        
        # Determine winner
        winner = None
        confidence = None
        
        if len(variant_names) == 2:
            # Binary A/B test
            success_test = next((t for t in statistical_tests if t.metric == "success_rate"), None)
            
            if success_test and success_test.is_significant:
                metrics_a = self.variant_metrics[success_test.variant_a]
                metrics_b = self.variant_metrics[success_test.variant_b]
                
                if metrics_b.success_rate > metrics_a.success_rate:
                    winner = success_test.variant_b
                else:
                    winner = success_test.variant_a
                
                confidence = 1 - success_test.p_value
        
        # Generate recommendation
        recommendation = None
        if winner:
            winner_metrics = self.variant_metrics[winner]
            recommendation = f"Deploy {winner} (success rate: {winner_metrics.success_rate:.3f})"
        else:
            recommendation = "No clear winner detected, consider running longer or with more traffic"
        
        # Calculate duration
        duration_hours = 0.0
        if self.start_time:
            end_time = self.end_time or datetime.now()
            duration_hours = (end_time - self.start_time).total_seconds() / 3600
        
        return ExperimentResult(
            experiment_id=self.config.experiment_id,
            status=self.status,
            start_time=self.start_time,
            end_time=self.end_time,
            duration_hours=duration_hours,
            variant_metrics=self.variant_metrics.copy(),
            statistical_tests=statistical_tests,
            winner=winner,
            confidence=confidence,
            recommendation=recommendation
        )
    
    def should_stop_early(self) -> Tuple[bool, str]:
        """Check if experiment should stop early.
        
        Returns:
            Tuple of (should_stop, reason)
        """
        if not self.config.auto_stop_on_significance:
            return False, ""
        
        # Check minimum sample size
        total_requests = sum(metrics.total_requests for metrics in self.variant_metrics.values())
        if total_requests < self.config.minimum_sample_size:
            return False, "Minimum sample size not reached"
        
        # Check for statistical significance
        result = self.analyze_results()
        
        significant_tests = [t for t in result.statistical_tests if t.is_significant]
        
        if significant_tests:
            return True, f"Statistical significance achieved: {len(significant_tests)} significant tests"
        
        # Check maximum duration
        if self.start_time:
            duration_days = (datetime.now() - self.start_time).days
            if duration_days >= self.config.max_duration_days:
                return True, "Maximum duration reached"
        
        return False, ""
    
    def get_status_summary(self) -> Dict[str, Any]:
        """Get experiment status summary.
        
        Returns:
            Status summary
        """
        total_requests = sum(metrics.total_requests for metrics in self.variant_metrics.values())
        
        variant_summaries = {}
        for name, metrics in self.variant_metrics.items():
            variant_summaries[name] = {
                'requests': metrics.total_requests,
                'success_rate': metrics.success_rate,
                'average_latency_ms': metrics.average_latency_ms,
                'conversions': metrics.conversion_events,
                'conversion_rate': metrics.conversion_rate
            }
        
        duration_hours = 0.0
        if self.start_time:
            end_time = self.end_time or datetime.now()
            duration_hours = (end_time - self.start_time).total_seconds() / 3600
        
        return {
            'experiment_id': self.config.experiment_id,
            'status': self.status.value,
            'duration_hours': duration_hours,
            'total_requests': total_requests,
            'variants': variant_summaries,
            'should_stop_early': self.should_stop_early()
        }