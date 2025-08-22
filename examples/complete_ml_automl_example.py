"""Complete ML Pipeline & AutoML example demonstrating feature engineering, model selection, and training."""

import numpy as np
import pandas as pd
from datetime import datetime
import logging
from pathlib import Path

# Import ML pipeline components
from src.ml.feature_extraction import FeatureExtractor, FeatureSet
from src.ml.text_features import AdvancedTextFeatureExtractor
from src.ml.pgx_features import PharmacogenomicsFeatureExtractor
from src.ml.feature_store import FeatureStore, FeatureDefinition, FeatureGroup
from src.ml.feature_pipeline_optimizer import OptimizedFeaturePipeline, OptimizationConfig
from src.ml.automl_engine import AutoMLEngine, AutoMLConfig
from src.ml.model_selector import (
    ModelSelector, RandomForestWrapper, XGBoostWrapper, 
    LightGBMWrapper, EnsembleWrapper
)
from src.ml.hyperopt import HyperparameterOptimizer, OptimizationObjective
from src.ml.training_pipeline import TrainingPipeline, TrainingConfig

# Import data components
from src.data_ingestion.data_models import PaperMetadata
from src.quality.quality_scorer import QualityScoreComponents
from src.config import get_logger

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = get_logger(__name__)


def create_sample_data(n_papers: int = 100) -> tuple:
    """Create sample pharmacogenomics papers and quality scores for demonstration.
    
    Args:
        n_papers: Number of sample papers to create
        
    Returns:
        Tuple of (papers, quality_scores, targets)
    """
    logger.info(f"Creating {n_papers} sample papers for demonstration")
    
    papers = []
    quality_scores = []
    targets = []
    
    # Sample pharmacogenomics terms
    drugs = ['warfarin', 'clopidogrel', 'codeine', 'tamoxifen', 'simvastatin', 'omeprazole']
    genes = ['CYP2D6', 'CYP2C19', 'CYP3A4', 'VKORC1', 'DPYD', 'SLCO1B1']
    populations = ['Caucasian', 'Asian', 'African American', 'Hispanic']
    
    for i in range(n_papers):
        # Create realistic pharmacogenomics abstracts
        drug = np.random.choice(drugs)
        gene = np.random.choice(genes)
        population = np.random.choice(populations)
        
        title = f"Pharmacogenomics of {drug} in {population} patients: role of {gene}"
        
        abstract = f"""
        Background: {drug} is a widely prescribed medication with significant inter-individual 
        variability in response. The {gene} gene polymorphisms have been associated with 
        altered drug metabolism in {population} populations.
        
        Methods: We conducted a retrospective analysis of {np.random.randint(100, 1000)} patients 
        treated with {drug}. Genotyping was performed for {gene} variants. Clinical outcomes 
        were assessed including efficacy and adverse events.
        
        Results: Patients with {gene} variants showed {np.random.choice(['increased', 'decreased'])} 
        drug clearance compared to wild-type. The frequency of adverse events was 
        {np.random.choice(['higher', 'lower'])} in variant carriers.
        
        Conclusions: {gene} genotyping may guide {drug} dosing in {population} patients 
        to optimize therapeutic outcomes and minimize adverse effects.
        """
        
        # Create paper metadata
        paper = PaperMetadata(
            pmid=f"PMID{i+1:06d}",
            title=title,
            abstract=abstract,
            authors=[f"Author{j}" for j in range(np.random.randint(2, 8))],
            journal=np.random.choice([
                'Pharmacogenomics', 'Clinical Pharmacology & Therapeutics',
                'Pharmacogenetics and Genomics', 'Journal of Clinical Pharmacology'
            ]),
            publication_date=datetime(
                year=np.random.randint(2015, 2024),
                month=np.random.randint(1, 13),
                day=np.random.randint(1, 29)
            ),
            doi=f"10.1000/example.{i+1}",
            mesh_terms=[drug, gene, 'pharmacogenomics', 'polymorphism'],
            keywords=[drug, gene, population.lower(), 'pharmacokinetics']
        )
        papers.append(paper)
        
        # Create quality scores
        quality = QualityScoreComponents(
            journal_score=np.random.uniform(0.6, 1.0),
            clinical_relevance_score=np.random.uniform(0.7, 1.0),
            ml_methodology_score=np.random.uniform(0.5, 0.9),
            recency_score=np.random.uniform(0.6, 1.0),
            total_score=0.0,  # Will be calculated
            confidence_level=np.random.uniform(0.8, 1.0)
        )
        quality.total_score = (
            quality.journal_score + quality.clinical_relevance_score + 
            quality.ml_methodology_score + quality.recency_score
        ) / 4
        quality_scores.append(quality)
        
        # Create binary target (high vs low clinical relevance)
        target = 1 if quality.clinical_relevance_score > 0.8 else 0
        targets.append(target)
    
    logger.info(f"Created {len(papers)} papers with {np.sum(targets)} high-relevance papers")
    return papers, quality_scores, np.array(targets)


def demonstrate_feature_extraction(papers, quality_scores):
    """Demonstrate comprehensive feature extraction."""
    logger.info("=== Feature Extraction Demonstration ===")
    
    # 1. Basic feature extraction
    logger.info("1. Basic feature extraction")
    feature_extractor = FeatureExtractor()
    basic_features = feature_extractor.extract_features(
        papers=papers,
        quality_scores=quality_scores,
        feature_types=['text', 'numerical', 'temporal', 'metadata', 'quality']
    )
    
    logger.info(f"Basic features extracted: {len(basic_features.features)} features")
    logger.info(f"Feature names: {basic_features.feature_names[:10]}...")  # First 10
    
    # 2. Advanced text features
    logger.info("2. Advanced text features")
    text_extractor = AdvancedTextFeatureExtractor()
    text_features = text_extractor.extract_features(papers)
    
    logger.info(f"Text features - TF-IDF shape: {text_features.tfidf_features.shape}")
    logger.info(f"Text features - Embeddings shape: {text_features.embeddings.shape}")
    logger.info(f"Text features - Entities per paper: {[len(entities) for entities in text_features.biomedical_entities[:5]]}")
    
    # 3. Pharmacogenomics-specific features
    logger.info("3. Pharmacogenomics-specific features")
    pgx_extractor = PharmacogenomicsFeatureExtractor()
    pgx_features = pgx_extractor.extract_features(papers)
    
    logger.info(f"PGx features extracted: {len(pgx_features)} features")
    logger.info(f"Sample PGx features: {list(pgx_features.keys())[:10]}")
    
    # 4. Optimized feature pipeline
    logger.info("4. Optimized feature pipeline")
    config = OptimizationConfig(
        max_workers=2,
        batch_size=50,
        enable_caching=True,
        enable_parallel_processing=True
    )
    
    optimized_pipeline = OptimizedFeaturePipeline(config)
    optimized_features, metrics = optimized_pipeline.extract_features_optimized(
        papers=papers,
        quality_scores=quality_scores,
        feature_types=['text', 'numerical', 'quality', 'pgx']
    )
    
    logger.info(f"Optimized extraction - Papers/sec: {metrics.papers_per_second:.1f}")
    logger.info(f"Optimized extraction - Cache hit rate: {metrics.cache_hit_rate:.2f}")
    logger.info(f"Optimized extraction - Memory usage: {metrics.memory_usage_mb:.1f}MB")
    
    return optimized_features


def demonstrate_feature_store(feature_set):
    """Demonstrate feature store capabilities."""
    logger.info("=== Feature Store Demonstration ===")
    
    # Initialize feature store
    feature_store = FeatureStore({
        'sqlite_path': 'demo_feature_store.db',
        'redis': {'enabled': False}  # Disable Redis for demo
    })
    
    # Register feature definitions
    for feature_name in feature_set.feature_names[:5]:  # First 5 features
        feature_def = FeatureDefinition(
            name=feature_name,
            feature_type='numerical',
            description=f"Feature {feature_name} for pharmacogenomics classification",
            source='ml_pipeline',
            owner='demo_user'
        )
        feature_store.register_feature_definition(feature_def)
    
    # Register feature group
    feature_group = FeatureGroup(
        name='pharmacogenomics_features',
        features=feature_set.feature_names[:5],
        description='Core pharmacogenomics features for ML',
        source='feature_extraction_pipeline',
        refresh_interval=pd.Timedelta(hours=24)
    )
    feature_store.register_feature_group(feature_group)
    
    # Store features offline
    success = feature_store.store_features_offline('pharmacogenomics_features', feature_set)
    logger.info(f"Features stored offline: {success}")
    
    # Retrieve features
    retrieved_features = feature_store.get_features_offline(
        'pharmacogenomics_features',
        feature_set.feature_names[:5],
        feature_set.sample_ids[:10]
    )
    logger.info(f"Retrieved features shape: {[v.shape for v in retrieved_features.values()]}")
    
    # Monitor feature drift (simplified demo)
    drift_metrics = feature_store.monitor_feature_drift(
        'pharmacogenomics_features',
        feature_set.feature_names[:3],
        reference_window_days=30
    )
    logger.info(f"Drift monitoring completed: {len(drift_metrics)} metrics")
    
    return feature_store


def demonstrate_model_selection(feature_set, targets):
    """Demonstrate automated model selection."""
    logger.info("=== Model Selection Demonstration ===")
    
    # Initialize model selector
    model_selector = ModelSelector()
    
    # Add models to compare
    model_selector.add_default_models()
    
    # Add custom ensemble
    base_models = [
        RandomForestWrapper('rf_ensemble', n_estimators=50, random_state=42),
        XGBoostWrapper('xgb_ensemble', n_estimators=50, random_state=42) if 'xgboost' in str(type(XGBoostWrapper)) else None
    ]
    base_models = [m for m in base_models if m is not None]
    
    if len(base_models) > 1:
        ensemble = EnsembleWrapper(
            'custom_ensemble',
            base_models=base_models,
            ensemble_type='voting'
        )
        model_selector.add_model(ensemble)
    
    # Evaluate all models
    results = model_selector.evaluate_models(feature_set, targets)
    
    # Display results
    logger.info(f"Model evaluation completed: {len(results)} models")
    
    for model_name, performance in results.items():
        logger.info(f"{model_name}: accuracy={performance.accuracy:.4f}, "
                   f"f1={performance.f1_score:.4f}, "
                   f"training_time={performance.training_time:.2f}s")
    
    # Get ranking
    ranking = model_selector.get_model_ranking('accuracy')
    logger.info(f"Best model: {ranking[0][0]} (accuracy: {ranking[0][1]:.4f})")
    
    # Performance summary
    summary_df = model_selector.get_performance_summary()
    logger.info(f"Performance summary shape: {summary_df.shape}")
    
    return model_selector, results


def demonstrate_hyperparameter_optimization(feature_set, targets):
    """Demonstrate hyperparameter optimization."""
    logger.info("=== Hyperparameter Optimization Demonstration ===")
    
    # Initialize optimizer
    hyperopt = HyperparameterOptimizer(random_state=42)
    
    # Create model to optimize
    model = RandomForestWrapper('rf_hyperopt', random_state=42)
    
    # Define optimization objectives
    accuracy_objective = OptimizationObjective(
        name='accuracy',
        direction='maximize'
    )
    
    speed_objective = OptimizationObjective(
        name='training_speed',
        direction='maximize'
    )
    
    # Single-objective optimization
    logger.info("Running single-objective optimization (accuracy)...")
    single_result = hyperopt.optimize_single_objective(
        model_wrapper=model,
        feature_set=feature_set,
        target=targets,
        objective=accuracy_objective,
        n_trials=20,  # Reduced for demo
        study_name='demo_single_objective'
    )
    
    logger.info(f"Best parameters: {single_result.best_parameters}")
    logger.info(f"Best accuracy: {single_result.best_values['accuracy']:.4f}")
    
    # Multi-objective optimization
    logger.info("Running multi-objective optimization (accuracy vs speed)...")
    multi_result = hyperopt.optimize_multi_objective(
        model_wrapper=model,
        feature_set=feature_set,
        target=targets,
        objectives=[accuracy_objective, speed_objective],
        n_trials=15,  # Reduced for demo
        study_name='demo_multi_objective'
    )
    
    logger.info(f"Pareto front size: {len(multi_result.pareto_front) if multi_result.pareto_front else 0}")
    if multi_result.pareto_front:
        best_pareto = multi_result.pareto_front[0]
        logger.info(f"Best Pareto solution: {best_pareto.objectives}")
    
    return hyperopt, single_result, multi_result


def demonstrate_automl_engine(feature_set, targets):
    """Demonstrate AutoML engine."""
    logger.info("=== AutoML Engine Demonstration ===")
    
    # Configure AutoML
    config = AutoMLConfig(
        n_trials=20,  # Reduced for demo
        timeout=300,  # 5 minutes
        cv_folds=3,   # Reduced for demo
        max_concurrent_trials=2,
        enable_early_stopping=True
    )
    
    # Initialize AutoML engine
    automl = AutoMLEngine(config)
    
    # Run optimization
    logger.info("Running AutoML optimization...")
    results = automl.optimize(
        feature_set=feature_set,
        target=targets,
        algorithms=['random_forest', 'xgboost'] if 'xgboost' in globals() else ['random_forest'],
        study_name='demo_automl'
    )
    
    logger.info(f"AutoML completed: {results['total_trials']} trials")
    logger.info(f"Best algorithm: {results['best_algorithm']}")
    logger.info(f"Best score: {results['best_score']:.4f}")
    logger.info(f"Best parameters: {results['best_params']}")
    
    # Save results
    automl.save_results(results, 'demo_automl_results.json')
    
    return automl, results


def demonstrate_training_pipeline(feature_set, targets):
    """Demonstrate end-to-end training pipeline."""
    logger.info("=== Training Pipeline Demonstration ===")
    
    # Configure training
    config = TrainingConfig(
        test_size=0.2,
        cv_folds=3,  # Reduced for demo
        enable_scaling=True,
        enable_feature_selection=True,
        feature_selection_k=50,  # Reduced for demo
        enable_hyperopt=True,
        hyperopt_trials=10,  # Reduced for demo
        save_models=True,
        model_dir='demo_models'
    )
    
    # Initialize training pipeline
    training_pipeline = TrainingPipeline(config)
    
    # Create models to train
    models = [
        RandomForestWrapper('rf_pipeline', n_estimators=50, random_state=42),
        RandomForestWrapper('rf_pipeline_deep', n_estimators=100, max_depth=10, random_state=42)
    ]
    
    # Train multiple models
    logger.info("Training multiple models...")
    results = training_pipeline.train_multiple_models(
        models=models,
        feature_set=feature_set,
        target=targets
    )
    
    # Display results
    for result in results:
        logger.info(f"Model: {result.model.name}")
        logger.info(f"  Test accuracy: {result.metrics.test_scores['accuracy']:.4f}")
        logger.info(f"  CV accuracy: {np.mean(result.metrics.cv_scores['accuracy']):.4f} ± {np.std(result.metrics.cv_scores['accuracy']):.4f}")
        logger.info(f"  Training time: {result.metrics.training_time:.2f}s")
        logger.info(f"  Features used: {len(result.feature_names)}")
    
    # Get training summary
    summary_df = training_pipeline.get_training_summary()
    logger.info(f"Training summary shape: {summary_df.shape}")
    
    # Get best model
    best_model = training_pipeline.get_best_model()
    if best_model:
        logger.info(f"Best model: {best_model.model.name} "
                   f"(accuracy: {best_model.metrics.test_scores['accuracy']:.4f})")
    
    return training_pipeline, results


def main():
    """Main demonstration function."""
    logger.info("Starting Complete ML Pipeline & AutoML Demonstration")
    logger.info("=" * 60)
    
    try:
        # Create sample data
        papers, quality_scores, targets = create_sample_data(n_papers=200)
        
        # 1. Feature Extraction
        feature_set = demonstrate_feature_extraction(papers, quality_scores)
        
        # 2. Feature Store
        feature_store = demonstrate_feature_store(feature_set)
        
        # 3. Model Selection
        model_selector, selection_results = demonstrate_model_selection(feature_set, targets)
        
        # 4. Hyperparameter Optimization
        hyperopt, single_result, multi_result = demonstrate_hyperparameter_optimization(
            feature_set, targets
        )
        
        # 5. AutoML Engine
        automl, automl_results = demonstrate_automl_engine(feature_set, targets)
        
        # 6. Training Pipeline
        training_pipeline, training_results = demonstrate_training_pipeline(
            feature_set, targets
        )
        
        # Summary
        logger.info("=" * 60)
        logger.info("DEMONSTRATION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Papers processed: {len(papers)}")
        logger.info(f"Features extracted: {len(feature_set.features)}")
        logger.info(f"Models evaluated: {len(selection_results)}")
        logger.info(f"AutoML best score: {automl_results['best_score']:.4f}")
        logger.info(f"Training pipeline best accuracy: {training_results[0].metrics.test_scores['accuracy']:.4f}")
        
        # Performance targets check
        best_accuracy = max(
            max(perf.accuracy for perf in selection_results.values()),
            automl_results['best_score'],
            training_results[0].metrics.test_scores['accuracy']
        )
        
        target_accuracy = 0.90  # 90% target from spec
        logger.info(f"Best accuracy achieved: {best_accuracy:.4f}")
        logger.info(f"Target accuracy (90%): {'✓ ACHIEVED' if best_accuracy >= target_accuracy else '✗ NOT ACHIEVED'}")
        
        logger.info("Demonstration completed successfully!")
        
    except Exception as e:
        logger.error(f"Demonstration failed: {e}")
        raise


if __name__ == "__main__":
    main()