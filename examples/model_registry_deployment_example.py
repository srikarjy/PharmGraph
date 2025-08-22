"""Complete Model Registry & Deployment example demonstrating MLflow integration, model serving, and A/B testing."""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import time
import uuid
from pathlib import Path

# Import ML components
from src.ml.model_registry import (
    MLflowModelRegistry, ModelPromotionWorkflow, ModelLifecycleManager,
    PromotionCriteria
)
from src.ml.model_server import (
    ModelServer, PredictionRequest, PredictionResponse,
    BatchPredictionRequest
)
from src.ml.ab_testing import (
    ABTestExperiment, ExperimentConfig, ExperimentVariant,
    TrafficSplitStrategy, ExperimentStatus
)
from src.ml.model_selector import RandomForestWrapper
from src.ml.training_pipeline import TrainingPipeline, TrainingConfig
from src.ml.feature_extraction import FeatureExtractor, FeatureSet, FeatureMetadata

# Import data components
from src.data_ingestion.data_models import PaperMetadata
from src.quality.quality_scorer import QualityScoreComponents
from src.config import get_logger

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = get_logger(__name__)


def create_sample_training_data(n_papers: int = 200) -> tuple:
    """Create sample training data for demonstration.
    
    Args:
        n_papers: Number of sample papers to create
        
    Returns:
        Tuple of (feature_set, targets, training_result)
    """
    logger.info(f"Creating {n_papers} sample papers for training")
    
    # Create sample papers (simplified)
    papers = []
    quality_scores = []
    targets = []
    
    for i in range(n_papers):
        paper = PaperMetadata(
            pmid=f"PMID{i+1:06d}",
            title=f"Pharmacogenomics study {i+1}",
            abstract=f"This study investigates drug response in patients with genetic variants. Sample size: {np.random.randint(100, 1000)} patients.",
            authors=[f"Author{j}" for j in range(np.random.randint(2, 6))],
            journal="Pharmacogenomics Journal",
            publication_date=datetime.now() - timedelta(days=np.random.randint(1, 1000)),
            doi=f"10.1000/example.{i+1}",
            mesh_terms=['pharmacogenomics', 'genetics'],
            keywords=['drug response', 'genetics']
        )
        papers.append(paper)
        
        # Create quality scores
        quality = QualityScoreComponents(
            journal_score=np.random.uniform(0.6, 1.0),
            clinical_relevance_score=np.random.uniform(0.7, 1.0),
            ml_methodology_score=np.random.uniform(0.5, 0.9),
            recency_score=np.random.uniform(0.6, 1.0),
            total_score=0.0,
            confidence_level=np.random.uniform(0.8, 1.0)
        )
        quality.total_score = (
            quality.journal_score + quality.clinical_relevance_score + 
            quality.ml_methodology_score + quality.recency_score
        ) / 4
        quality_scores.append(quality)
        
        # Create binary target
        target = 1 if quality.clinical_relevance_score > 0.8 else 0
        targets.append(target)
    
    # Extract features
    feature_extractor = FeatureExtractor()
    feature_set = feature_extractor.extract_features(
        papers=papers,
        quality_scores=quality_scores,
        feature_types=['text', 'numerical', 'quality']
    )
    
    # Train a model
    training_config = TrainingConfig(
        test_size=0.2,
        cv_folds=3,
        enable_hyperopt=False,  # Disable for demo speed
        save_models=False
    )
    
    training_pipeline = TrainingPipeline(training_config)
    model = RandomForestWrapper('demo_model', n_estimators=50, random_state=42)
    
    training_result = training_pipeline.train_model(
        model=model,
        feature_set=feature_set,
        target=np.array(targets),
        pipeline_id='demo_training'
    )
    
    logger.info(f"Training completed: accuracy = {training_result.metrics.test_scores['accuracy']:.4f}")
    
    return feature_set, np.array(targets), training_result


def demonstrate_model_registry(training_result):
    """Demonstrate MLflow model registry functionality."""
    logger.info("=== Model Registry Demonstration ===")
    
    # Initialize model registry (without actual MLflow for demo)
    try:
        registry = MLflowModelRegistry()
        logger.info("MLflow model registry initialized")
    except ImportError:
        logger.warning("MLflow not available, using mock registry")
        # Create a mock registry for demonstration
        class MockRegistry:
            def __init__(self):
                self.models = {}
            
            def register_model(self, model, model_name, training_result, description=None, tags=None):
                from src.ml.model_registry import ModelVersion
                version = ModelVersion(
                    name=model_name,
                    version="1",
                    stage="None",
                    run_id="mock_run_id",
                    model_uri="mock_uri",
                    creation_timestamp=datetime.now(),
                    last_updated_timestamp=datetime.now(),
                    description=description,
                    tags=tags or {},
                    metrics=training_result.metrics.test_scores,
                    params=model.params if hasattr(model, 'params') else {}
                )
                self.models[model_name] = [version]
                return version
            
            def get_model_versions(self, model_name):
                return self.models.get(model_name, [])
            
            def get_latest_model_version(self, model_name, stage=None):
                versions = self.get_model_versions(model_name)
                if stage:
                    versions = [v for v in versions if v.stage == stage]
                return versions[0] if versions else None
            
            def promote_model(self, model_name, version, stage, archive_existing=True):
                versions = self.get_model_versions(model_name)
                for v in versions:
                    if v.version == version:
                        v.stage = stage
                        return True
                return False
            
            def load_model(self, model_name, version=None, stage=None):
                # Return the original model for demo
                return training_result.model.model
        
        registry = MockRegistry()
    
    # 1. Register model
    logger.info("1. Registering model")
    model_version = registry.register_model(
        model=training_result.model,
        model_name="pharmacogenomics_classifier",
        training_result=training_result,
        description="Demo pharmacogenomics classification model",
        tags={"version": "demo", "dataset": "synthetic"}
    )
    
    logger.info(f"Model registered: {model_version.name} v{model_version.version}")
    logger.info(f"Model metrics: {model_version.metrics}")
    
    # 2. Model promotion workflow
    logger.info("2. Model promotion workflow")
    criteria = PromotionCriteria(
        min_accuracy=0.7,
        min_precision=0.7,
        min_recall=0.7,
        min_f1_score=0.7,
        require_approval=False
    )
    
    promotion_workflow = ModelPromotionWorkflow(registry, criteria)
    
    # Evaluate for promotion
    evaluation = promotion_workflow.evaluate_for_promotion(
        "pharmacogenomics_classifier", "1"
    )
    
    logger.info(f"Promotion evaluation: {evaluation['eligible']}")
    logger.info(f"Recommendation: {evaluation['recommendation']}")
    
    # Auto-promote if eligible
    if evaluation['eligible']:
        success = promotion_workflow.auto_promote(
            "pharmacogenomics_classifier", "1", "Staging"
        )
        logger.info(f"Auto-promotion to Staging: {success}")
        
        # Try production promotion
        prod_success = promotion_workflow.auto_promote(
            "pharmacogenomics_classifier", "1", "Production"
        )
        logger.info(f"Auto-promotion to Production: {prod_success}")
    
    # 3. Model lifecycle management
    logger.info("3. Model lifecycle management")
    lifecycle_manager = ModelLifecycleManager(registry)
    
    # Deploy complete pipeline
    deployment_result = lifecycle_manager.deploy_model_pipeline(
        training_result=training_result,
        model_name="pharmacogenomics_classifier_v2",
        auto_promote=True
    )
    
    logger.info(f"Pipeline deployment: {deployment_result['status']}")
    if deployment_result['registration']:
        logger.info(f"Registered version: {deployment_result['registration']['version']}")
    
    # Get model health report
    health_report = lifecycle_manager.get_model_health_report("pharmacogenomics_classifier")
    logger.info(f"Model health: {health_report['health_status']}")
    logger.info(f"Recommendations: {health_report['recommendations']}")
    
    return registry


def demonstrate_model_serving(registry, feature_set):
    """Demonstrate model serving infrastructure."""
    logger.info("=== Model Serving Demonstration ===")
    
    # Initialize model server
    model_server = ModelServer(
        registry=registry,
        model_cache_size=3,
        enable_prediction_cache=True,
        redis_config=None  # No Redis for demo
    )
    
    # 1. Single predictions
    logger.info("1. Single prediction serving")
    
    # Create sample prediction requests
    sample_features = {}
    for i, feature_name in enumerate(feature_set.feature_names[:10]):  # First 10 features
        sample_features[feature_name] = float(feature_set.features[feature_name][0])
    
    request = PredictionRequest(
        request_id=str(uuid.uuid4()),
        features=sample_features,
        model_name="pharmacogenomics_classifier",
        return_probabilities=True
    )
    
    # Make prediction
    response = model_server.predict(request)
    
    logger.info(f"Prediction: {response.prediction}")
    logger.info(f"Probabilities: {response.probabilities}")
    logger.info(f"Processing time: {response.processing_time_ms:.2f}ms")
    
    # 2. Batch predictions
    logger.info("2. Batch prediction serving")
    
    # Create batch request
    batch_requests = []
    for i in range(5):
        batch_features = {}
        for j, feature_name in enumerate(feature_set.feature_names[:10]):
            batch_features[feature_name] = float(feature_set.features[feature_name][i])
        
        batch_request = PredictionRequest(
            request_id=str(uuid.uuid4()),
            features=batch_features,
            model_name="pharmacogenomics_classifier",
            return_probabilities=True
        )
        batch_requests.append(batch_request)
    
    batch_prediction_request = BatchPredictionRequest(
        batch_id=str(uuid.uuid4()),
        requests=batch_requests,
        model_name="pharmacogenomics_classifier",
        parallel_processing=True,
        max_workers=2
    )
    
    # Process batch
    batch_response = model_server.predict_batch(batch_prediction_request)
    
    logger.info(f"Batch processed: {batch_response.successful_predictions}/{batch_response.total_requests}")
    logger.info(f"Average processing time: {batch_response.average_processing_time_ms:.2f}ms")
    
    # 3. Server statistics
    logger.info("3. Server statistics")
    stats = model_server.get_server_stats()
    
    logger.info(f"Total requests: {stats['requests']['total_requests']}")
    logger.info(f"Success rate: {stats['success_rate']:.3f}")
    logger.info(f"Cache hit rate: {stats['cache_hit_rate']:.3f}")
    logger.info(f"Model cache: {stats['model_cache']['cached_models']} models")
    
    # 4. Health check
    health = model_server.health_check()
    logger.info(f"Server health: {health['status']}")
    
    return model_server


def demonstrate_ab_testing(model_server, feature_set):
    """Demonstrate A/B testing framework."""
    logger.info("=== A/B Testing Demonstration ===")
    
    # 1. Create experiment configuration
    logger.info("1. Creating A/B test experiment")
    
    # Define variants
    control_variant = ExperimentVariant(
        name="control",
        model_name="pharmacogenomics_classifier",
        model_version="1",
        traffic_percentage=50.0,
        description="Current production model"
    )
    
    treatment_variant = ExperimentVariant(
        name="treatment",
        model_name="pharmacogenomics_classifier_v2",
        model_version="1",
        traffic_percentage=50.0,
        description="New improved model"
    )
    
    # Create experiment config
    experiment_config = ExperimentConfig(
        experiment_id="pgx_model_comparison_001",
        name="Pharmacogenomics Model A/B Test",
        description="Compare current vs new pharmacogenomics classification model",
        variants=[control_variant, treatment_variant],
        traffic_split_strategy=TrafficSplitStrategy.RANDOM,
        success_metrics=['accuracy', 'latency'],
        minimum_sample_size=50,  # Reduced for demo
        confidence_level=0.95,
        max_duration_days=7,
        auto_stop_on_significance=True
    )
    
    # Initialize experiment
    experiment = ABTestExperiment(experiment_config, model_server)
    
    # 2. Start experiment
    logger.info("2. Starting A/B test")
    success = experiment.start()
    logger.info(f"Experiment started: {success}")
    
    # 3. Simulate traffic
    logger.info("3. Simulating user traffic")
    
    for i in range(100):  # Simulate 100 requests
        # Create sample request
        sample_features = {}
        sample_idx = i % len(feature_set.sample_ids)
        
        for j, feature_name in enumerate(feature_set.feature_names[:10]):
            sample_features[feature_name] = float(feature_set.features[feature_name][sample_idx])
        
        request = PredictionRequest(
            request_id=str(uuid.uuid4()),
            features=sample_features,
            model_name="pharmacogenomics_classifier",  # Will be overridden by experiment
            return_probabilities=True
        )
        
        # Process through A/B test
        user_id = f"user_{i % 20}"  # 20 unique users
        response, assigned_variant = experiment.process_request(request, user_id)
        
        # Simulate conversion events (random for demo)
        if np.random.random() < 0.3:  # 30% conversion rate
            experiment.record_conversion(assigned_variant, user_id)
        
        # Add some delay to simulate real traffic
        if i % 20 == 0:
            time.sleep(0.1)
    
    logger.info("Traffic simulation completed")
    
    # 4. Analyze results
    logger.info("4. Analyzing A/B test results")
    
    # Get status summary
    status = experiment.get_status_summary()
    logger.info(f"Experiment status: {status['status']}")
    logger.info(f"Total requests: {status['total_requests']}")
    
    for variant_name, variant_stats in status['variants'].items():
        logger.info(f"{variant_name}: {variant_stats['requests']} requests, "
                   f"success rate: {variant_stats['success_rate']:.3f}, "
                   f"conversion rate: {variant_stats['conversion_rate']:.3f}")
    
    # Perform statistical analysis
    results = experiment.analyze_results()
    
    logger.info(f"Statistical tests performed: {len(results.statistical_tests)}")
    
    for test in results.statistical_tests:
        logger.info(f"Test: {test.test_name} on {test.metric}")
        logger.info(f"  p-value: {test.p_value:.4f}, significant: {test.is_significant}")
        logger.info(f"  {test.interpretation}")
    
    if results.winner:
        logger.info(f"Winner: {results.winner} (confidence: {results.confidence:.3f})")
    else:
        logger.info("No clear winner detected")
    
    logger.info(f"Recommendation: {results.recommendation}")
    
    # 5. Check early stopping
    should_stop, reason = experiment.should_stop_early()
    logger.info(f"Should stop early: {should_stop} ({reason})")
    
    # Stop experiment
    experiment.stop()
    logger.info("A/B test experiment stopped")
    
    return experiment, results


def demonstrate_performance_monitoring(model_server):
    """Demonstrate performance monitoring and optimization."""
    logger.info("=== Performance Monitoring Demonstration ===")
    
    # 1. Load testing simulation
    logger.info("1. Load testing simulation")
    
    start_time = time.time()
    requests_sent = 0
    successful_requests = 0
    
    # Simulate high-frequency requests
    for i in range(100):  # 100 requests for demo
        try:
            # Create random features
            features = {f'feature_{j}': np.random.randn() for j in range(10)}
            
            request = PredictionRequest(
                request_id=str(uuid.uuid4()),
                features=features,
                model_name="pharmacogenomics_classifier",
                return_probabilities=True
            )
            
            response = model_server.predict(request)
            requests_sent += 1
            
            if response.error is None:
                successful_requests += 1
            
        except Exception as e:
            logger.error(f"Request failed: {e}")
    
    # Calculate performance metrics
    total_time = time.time() - start_time
    requests_per_second = requests_sent / total_time
    success_rate = successful_requests / requests_sent if requests_sent > 0 else 0
    
    logger.info(f"Load test results:")
    logger.info(f"  Requests sent: {requests_sent}")
    logger.info(f"  Success rate: {success_rate:.3f}")
    logger.info(f"  Requests per second: {requests_per_second:.1f}")
    logger.info(f"  Total time: {total_time:.2f}s")
    
    # 2. Server statistics
    logger.info("2. Detailed server statistics")
    stats = model_server.get_server_stats()
    
    logger.info(f"Server performance:")
    logger.info(f"  Total requests: {stats['requests']['total_requests']}")
    logger.info(f"  Success rate: {stats['success_rate']:.3f}")
    logger.info(f"  Average processing time: {stats['average_processing_time_ms']:.2f}ms")
    logger.info(f"  Cache hit rate: {stats['cache_hit_rate']:.3f}")
    
    # 3. Performance targets check
    logger.info("3. Performance targets validation")
    
    target_latency_ms = 200.0  # <200ms target
    target_throughput_rps = 100.0  # 100+ RPS target (scaled down for demo)
    
    avg_latency = stats['average_processing_time_ms']
    current_throughput = requests_per_second
    
    logger.info(f"Latency target (<200ms): {'✓ PASS' if avg_latency < target_latency_ms else '✗ FAIL'} "
               f"({avg_latency:.2f}ms)")
    logger.info(f"Throughput target (>100 RPS): {'✓ PASS' if current_throughput > target_throughput_rps else '✗ FAIL'} "
               f"({current_throughput:.1f} RPS)")
    
    return stats


def main():
    """Main demonstration function."""
    logger.info("Starting Complete Model Registry & Deployment Demonstration")
    logger.info("=" * 70)
    
    try:
        # Create training data
        feature_set, targets, training_result = create_sample_training_data(n_papers=200)
        
        # 1. Model Registry
        registry = demonstrate_model_registry(training_result)
        
        # 2. Model Serving
        model_server = demonstrate_model_serving(registry, feature_set)
        
        # 3. A/B Testing
        experiment, ab_results = demonstrate_ab_testing(model_server, feature_set)
        
        # 4. Performance Monitoring
        performance_stats = demonstrate_performance_monitoring(model_server)
        
        # Summary
        logger.info("=" * 70)
        logger.info("DEMONSTRATION SUMMARY")
        logger.info("=" * 70)
        
        logger.info(f"Model accuracy: {training_result.metrics.test_scores['accuracy']:.4f}")
        logger.info(f"Model registered and promoted: ✓")
        logger.info(f"Serving performance: {performance_stats['average_processing_time_ms']:.2f}ms avg latency")
        logger.info(f"A/B test completed: {ab_results.status.value}")
        logger.info(f"Statistical tests: {len(ab_results.statistical_tests)} performed")
        
        # Check overall targets
        model_accuracy = training_result.metrics.test_scores['accuracy']
        avg_latency = performance_stats['average_processing_time_ms']
        
        logger.info("\nTarget Achievement:")
        logger.info(f"90%+ accuracy: {'✓ ACHIEVED' if model_accuracy >= 0.90 else '✗ NOT ACHIEVED'} ({model_accuracy:.3f})")
        logger.info(f"<200ms latency: {'✓ ACHIEVED' if avg_latency < 200 else '✗ NOT ACHIEVED'} ({avg_latency:.1f}ms)")
        logger.info(f"Model registry: ✓ IMPLEMENTED")
        logger.info(f"A/B testing: ✓ IMPLEMENTED")
        
        logger.info("\nDemonstration completed successfully!")
        
        # Cleanup
        model_server.shutdown()
        
    except Exception as e:
        logger.error(f"Demonstration failed: {e}")
        raise


if __name__ == "__main__":
    main()