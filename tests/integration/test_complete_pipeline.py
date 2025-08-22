"""Complete integration tests for the pharmacogenomics pipeline."""

import pytest
import asyncio
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any
import tempfile
import os

from src.processing.pipeline_orchestrator import PharmacogenomicsPipeline, PipelineStatus, PipelineStage
from src.processing.pipeline_monitor import PipelineMonitor, AlertLevel
from src.processing.performance_optimizer import PerformanceOptimizer
from src.config.config_manager import ConfigManager
from src.storage.database import get_database_manager, db_session_scope
from src.storage.models import PaperModel, QualityMetrics


class TestCompletePipeline:
    """Complete end-to-end pipeline integration tests."""
    
    @pytest.fixture
    def config_manager(self):
        """Create test configuration manager."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ConfigManager(config_dir=temp_dir)
            # Set test configuration
            config.ncbi.email = "test@example.com"
            config.ncbi.api_key = "test_api_key"
            config.database.url = "sqlite:///:memory:"
            yield config
    
    @pytest.fixture
    def pipeline(self, config_manager):
        """Create test pipeline."""
        return PharmacogenomicsPipeline(
            email="test@example.com",
            api_key="test_api_key",
            batch_size=50,
            max_concurrent=3
        )
    
    @pytest.fixture
    def monitor(self):
        """Create test monitor."""
        return PipelineMonitor(
            monitoring_interval=1.0,  # Fast monitoring for tests
            enable_resource_monitoring=False  # Disable for CI
        )
    
    @pytest.fixture
    def optimizer(self):
        """Create test optimizer."""
        return PerformanceOptimizer(
            max_memory_mb=512,  # Lower limit for tests
            max_concurrent_api=3,
            target_batch_size=25
        )
    
    @pytest.mark.asyncio
    async def test_full_pipeline_small_dataset(self, pipeline, monitor, optimizer):
        """Test complete pipeline with small dataset."""
        # Test queries for pharmacogenomics
        test_queries = [
            "CYP2D6 AND pharmacogenomics",
            "warfarin AND dosing"
        ]
        
        # Start monitoring
        monitor.start_monitoring()
        
        try:
            # Run pipeline
            report = await pipeline.run_full_pipeline(
                search_queries=test_queries,
                max_papers=100  # Small dataset for testing
            )
            
            # Verify pipeline completed
            assert report.status == PipelineStatus.COMPLETED
            assert report.papers_collected > 0
            assert report.papers_stored >= 0
            assert report.success_rate >= 0
            
            # Verify timing metrics
            assert report.total_time_minutes > 0
            assert report.papers_per_minute >= 0
            
            # Verify quality metrics
            assert 0 <= report.avg_quality_score <= 100
            assert report.high_quality_papers >= 0
            assert report.medium_quality_papers >= 0
            assert report.low_quality_papers >= 0
            
            # Verify error handling
            assert report.total_errors >= 0
            assert isinstance(report.error_breakdown, dict)
            
            # Check monitoring metrics
            current_metrics = monitor.get_current_metrics()
            if current_metrics:
                assert current_metrics.papers_per_minute >= 0
                assert 0 <= current_metrics.success_rate <= 100
            
        finally:
            monitor.stop_monitoring()
    
    @pytest.mark.asyncio
    async def test_pipeline_error_recovery(self, pipeline):
        """Test pipeline error recovery and resumption."""
        # Test with invalid query to trigger errors
        test_queries = ["invalid_query_that_should_fail_12345"]
        
        report = await pipeline.run_full_pipeline(
            search_queries=test_queries,
            max_papers=10
        )
        
        # Pipeline should handle errors gracefully
        assert report.status in [PipelineStatus.COMPLETED, PipelineStatus.FAILED]
        assert report.total_errors >= 0
        
        # Test resumption capability
        pipeline_state = pipeline.get_pipeline_status()
        if pipeline_state and pipeline_state.status == PipelineStatus.FAILED:
            # Attempt to resume (should handle gracefully)
            try:
                resumed_report = await pipeline.resume_failed_pipeline(pipeline_state.pipeline_id)
                assert resumed_report is not None
            except Exception as e:
                # Resumption might fail, but should be handled gracefully
                assert "not found" in str(e) or "not in failed state" in str(e)
    
    @pytest.mark.asyncio
    async def test_batch_processing_performance(self, pipeline, optimizer):
        """Test batch processing performance optimization."""
        # Create test PMIDs
        test_pmids = [f"1234567{i:02d}" for i in range(50)]
        
        start_time = time.time()
        
        # Process batch with optimization
        result = await pipeline.process_paper_batch(test_pmids)
        
        processing_time = time.time() - start_time
        
        # Verify batch processing results
        assert isinstance(result, dict)
        assert "processed" in result
        assert "batch_time" in result
        assert result["batch_time"] > 0
        
        # Performance should be reasonable
        assert processing_time < 60  # Should complete within 60 seconds
        
        # Check optimization metrics
        metrics = optimizer.get_optimization_metrics()
        assert metrics.memory_usage_mb > 0
        assert metrics.papers_per_second >= 0
    
    @pytest.mark.asyncio
    async def test_quality_validation_pipeline(self, pipeline, monitor):
        """Test quality validation throughout pipeline."""
        test_queries = ["pharmacogenomics AND validation"]
        
        # Start monitoring with validation
        monitor.start_monitoring()
        
        try:
            report = await pipeline.run_full_pipeline(
                search_queries=test_queries,
                max_papers=50
            )
            
            # Verify quality metrics are reasonable
            if report.papers_stored > 0:
                assert 0 <= report.avg_quality_score <= 100
                
                # Quality distribution should add up
                total_quality_papers = (report.high_quality_papers + 
                                      report.medium_quality_papers + 
                                      report.low_quality_papers)
                assert total_quality_papers <= report.papers_stored
            
            # Check for quality-related alerts
            alerts = monitor.get_active_alerts()
            quality_alerts = [alert for alert in alerts if alert.component == "quality"]
            
            # Should not have critical quality alerts for small dataset
            critical_quality_alerts = [alert for alert in quality_alerts 
                                     if alert.level == AlertLevel.CRITICAL]
            assert len(critical_quality_alerts) == 0
            
        finally:
            monitor.stop_monitoring()
    
    @pytest.mark.asyncio
    async def test_database_integrity(self, pipeline):
        """Test database integrity after pipeline execution."""
        test_queries = ["CYP2D6 AND test"]
        
        # Initialize database
        db_manager = get_database_manager()
        db_manager.initialize()
        
        # Get initial counts
        with db_session_scope() as session:
            initial_paper_count = session.query(PaperModel).count()
            initial_quality_count = session.query(QualityMetrics).count()
        
        # Run pipeline
        report = await pipeline.run_full_pipeline(
            search_queries=test_queries,
            max_papers=25
        )
        
        # Check database integrity
        with db_session_scope() as session:
            final_paper_count = session.query(PaperModel).count()
            final_quality_count = session.query(QualityMetrics).count()
            
            # Should have added papers
            papers_added = final_paper_count - initial_paper_count
            quality_added = final_quality_count - initial_quality_count
            
            # Papers and quality metrics should match
            assert papers_added == quality_added
            assert papers_added == report.papers_stored
            
            # Verify referential integrity
            if papers_added > 0:
                # Check that all papers have corresponding quality metrics
                papers = session.query(PaperModel).limit(10).all()
                for paper in papers:
                    quality_metric = session.query(QualityMetrics).filter_by(pmid=paper.pmid).first()
                    if quality_metric:  # Some papers might not have quality metrics due to processing errors
                        assert quality_metric.pmid == paper.pmid
                        assert 0 <= quality_metric.total_score <= 100
    
    @pytest.mark.asyncio
    async def test_monitoring_and_alerting(self, monitor):
        """Test monitoring and alerting system."""
        # Configure strict thresholds for testing
        monitor.alert_thresholds.update({
            'error_rate_warning': 1.0,  # Very low threshold
            'success_rate_warning': 99.0,  # Very high threshold
        })
        
        monitor.start_monitoring()
        
        try:
            # Simulate some errors
            for i in range(5):
                monitor.record_error("test_error", f"Test error {i}", "test_component")
                await asyncio.sleep(0.1)
            
            # Wait for monitoring cycle
            await asyncio.sleep(2.0)
            
            # Check that alerts were generated
            alerts = monitor.get_active_alerts()
            assert len(alerts) > 0
            
            # Check error summary
            error_summary = monitor.get_error_summary(hours=1)
            assert error_summary['total_errors'] >= 5
            assert 'test_error' in error_summary['error_by_type']
            assert 'test_component' in error_summary['error_by_component']
            
        finally:
            monitor.stop_monitoring()
    
    @pytest.mark.asyncio
    async def test_performance_optimization(self, optimizer):
        """Test performance optimization features."""
        # Test cache functionality
        cache_manager = optimizer.cache_manager
        
        # Test journal score caching
        test_journal = "Test Journal"
        test_score = 85.5
        
        # Should be cache miss first time
        cached_score = cache_manager.get_journal_score(test_journal)
        assert cached_score is None
        
        # Cache the score
        cache_manager.cache_journal_score(test_journal, test_score)
        
        # Should be cache hit second time
        cached_score = cache_manager.get_journal_score(test_journal)
        assert cached_score == test_score
        
        # Test cache statistics
        cache_stats = cache_manager.get_cache_stats()
        assert cache_stats['journal_scores']['hits'] >= 1
        assert cache_stats['journal_scores']['misses'] >= 1
        
        # Test memory optimization
        memory_optimizer = optimizer.memory_optimizer
        initial_memory = memory_optimizer.check_memory_usage()
        
        # Force memory optimization
        memory_optimizer.optimize_memory(force=True)
        
        # Memory usage should be tracked
        assert initial_memory > 0
        
        # Test batch optimization
        batch_optimizer = optimizer.batch_optimizer
        test_items = list(range(500))
        
        batches = batch_optimizer.optimize_batch_size(test_items)
        assert len(batches) > 1
        assert sum(len(batch) for batch in batches) == len(test_items)
    
    @pytest.mark.asyncio
    async def test_load_testing_simulation(self, pipeline, monitor, optimizer):
        """Simulate load testing with larger dataset."""
        # This test simulates processing a larger dataset
        test_queries = [
            "pharmacogenomics",
            "CYP2D6",
            "warfarin dosing"
        ]
        
        monitor.start_monitoring()
        
        try:
            start_time = time.time()
            
            # Run pipeline with larger dataset
            report = await pipeline.run_full_pipeline(
                search_queries=test_queries,
                max_papers=200  # Larger dataset
            )
            
            total_time = time.time() - start_time
            
            # Performance assertions
            assert total_time < 600  # Should complete within 10 minutes
            assert report.papers_per_minute > 0
            
            # Memory usage should be reasonable
            metrics = optimizer.get_optimization_metrics()
            assert metrics.memory_usage_mb < 1024  # Less than 1GB
            
            # Success rate should be reasonable
            assert report.success_rate >= 50  # At least 50% success rate
            
            # Check monitoring detected the load
            current_metrics = monitor.get_current_metrics()
            if current_metrics:
                assert current_metrics.papers_per_minute > 0
            
        finally:
            monitor.stop_monitoring()
    
    def test_pipeline_configuration(self, pipeline):
        """Test pipeline configuration and initialization."""
        # Test pipeline initialization
        assert pipeline.email == "test@example.com"
        assert pipeline.api_key == "test_api_key"
        assert pipeline.batch_size == 50
        assert pipeline.max_concurrent == 3
        
        # Test pipeline state management
        assert pipeline.current_pipeline is None
        assert len(pipeline.pipeline_history) == 0
        
        # Test configuration snapshot
        config_snapshot = pipeline._capture_config_snapshot()
        assert config_snapshot['email'] == "test@example.com"
        assert config_snapshot['has_api_key'] is True
        assert config_snapshot['batch_size'] == 50
    
    def test_monitor_configuration(self, monitor):
        """Test monitor configuration and thresholds."""
        # Test default thresholds
        assert monitor.alert_thresholds['error_rate_warning'] == 5.0
        assert monitor.alert_thresholds['success_rate_critical'] == 80.0
        
        # Test monitoring state
        assert not monitor.is_monitoring
        assert monitor.monitoring_thread is None
        
        # Test validation rules
        assert len(monitor.validation_rules) > 0
        assert 'required_fields' in monitor.validation_rules
        assert 'quality_score_range' in monitor.validation_rules
    
    def test_optimizer_configuration(self, optimizer):
        """Test optimizer configuration and components."""
        # Test component initialization
        assert optimizer.memory_optimizer is not None
        assert optimizer.cache_manager is not None
        assert optimizer.concurrency_manager is not None
        assert optimizer.batch_optimizer is not None
        assert optimizer.db_optimizer is not None
        
        # Test initial metrics
        assert optimizer.total_papers_processed == 0
        assert optimizer.start_time > 0
        
        # Test performance report structure
        report = optimizer.get_performance_report()
        assert 'optimization_metrics' in report
        assert 'cache_performance' in report
        assert 'resource_utilization' in report
        assert 'overall_stats' in report


@pytest.mark.integration
class TestProductionScenarios:
    """Test production-like scenarios."""
    
    @pytest.mark.asyncio
    async def test_production_pipeline_simulation(self):
        """Simulate production pipeline execution."""
        # This test would run against real NCBI APIs in a production environment
        # For CI/CD, we'll use mock data or skip if no API credentials
        
        pipeline = PharmacogenomicsPipeline(
            email=os.getenv("NCBI_EMAIL", "test@example.com"),
            api_key=os.getenv("NCBI_API_KEY"),
            batch_size=100,
            max_concurrent=5
        )
        
        monitor = PipelineMonitor(monitoring_interval=10.0)
        optimizer = PerformanceOptimizer()
        
        # Production-like queries
        production_queries = [
            "CYP2D6 AND pharmacogenomics AND (machine learning OR artificial intelligence)",
            "warfarin AND dosing AND precision medicine",
            "pharmacogenomics AND biomarkers AND clinical decision support"
        ]
        
        if os.getenv("NCBI_EMAIL") and os.getenv("NCBI_API_KEY"):
            # Only run with real credentials
            monitor.start_monitoring()
            
            try:
                report = await pipeline.run_full_pipeline(
                    search_queries=production_queries,
                    max_papers=500  # Production-like volume
                )
                
                # Production performance expectations
                assert report.total_time_minutes < 30  # Should complete within 30 minutes
                assert report.papers_per_minute >= 10  # At least 10 papers per minute
                assert report.success_rate >= 80  # At least 80% success rate
                
                # Quality expectations
                assert report.avg_quality_score >= 40  # Reasonable quality
                assert report.papers_stored > 0  # Should store some papers
                
            finally:
                monitor.stop_monitoring()
        else:
            pytest.skip("Production test requires NCBI credentials")
    
    @pytest.mark.asyncio
    async def test_error_recovery_scenarios(self):
        """Test various error recovery scenarios."""
        pipeline = PharmacogenomicsPipeline(
            email="test@example.com",
            api_key="invalid_key",  # Intentionally invalid
            batch_size=50
        )
        
        # Test network error handling
        try:
            report = await pipeline.run_full_pipeline(
                search_queries=["test query"],
                max_papers=10
            )
            
            # Should handle errors gracefully
            assert report.total_errors >= 0
            
        except Exception as e:
            # Should not raise unhandled exceptions
            assert "authentication" in str(e).lower() or "network" in str(e).lower()
    
    def test_scalability_configuration(self):
        """Test configuration for different scale scenarios."""
        # Small scale configuration
        small_pipeline = PharmacogenomicsPipeline(
            batch_size=50,
            max_concurrent=3
        )
        
        # Large scale configuration
        large_pipeline = PharmacogenomicsPipeline(
            batch_size=200,
            max_concurrent=10
        )
        
        # Verify configurations
        assert small_pipeline.batch_size == 50
        assert large_pipeline.batch_size == 200
        assert small_pipeline.max_concurrent == 3
        assert large_pipeline.max_concurrent == 10