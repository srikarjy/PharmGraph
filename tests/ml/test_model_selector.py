"""Tests for model selector and comparison framework."""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime
from unittest.mock import Mock, patch

from src.ml.model_selector import (
    ModelSelector, ModelConfig, ModelPerformance,
    RandomForestWrapper, XGBoostWrapper, LightGBMWrapper,
    BERTWrapper, EnsembleWrapper, ModelInterpretability
)
from src.ml.feature_extraction import FeatureSet, FeatureMetadata


@pytest.fixture
def sample_feature_set():
    """Create sample feature set for testing."""
    n_samples = 100
    n_features = 10
    
    features = {}
    feature_names = []
    
    for i in range(n_features):
        feature_name = f'feature_{i}'
        features[feature_name] = np.random.randn(n_samples)
        feature_names.append(feature_name)
    
    metadata = {
        name: FeatureMetadata(
            feature_name=name,
            feature_type='numerical',
            extraction_method='test',
            version='v1',
            created_at=datetime.now()
        )
        for name in feature_names
    }
    
    return FeatureSet(
        features=features,
        metadata=metadata,
        feature_names=feature_names,
        sample_ids=[f'sample_{i}' for i in range(n_samples)],
        extraction_timestamp=datetime.now(),
        version_hash='test_hash'
    )


@pytest.fixture
def sample_target():
    """Create sample binary target for testing."""
    return np.random.randint(0, 2, size=100)


@pytest.fixture
def sample_multiclass_target():
    """Create sample multiclass target for testing."""
    return np.random.randint(0, 3, size=100)


class TestModelConfig:
    """Test model configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = ModelConfig()
        
        assert config.test_size == 0.2
        assert config.cv_folds == 5
        assert config.random_state == 42
        assert config.enable_scaling is True
        assert config.enable_interpretability is True
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = ModelConfig(
            test_size=0.3,
            cv_folds=3,
            random_state=123,
            enable_scaling=False
        )
        
        assert config.test_size == 0.3
        assert config.cv_folds == 3
        assert config.random_state == 123
        assert config.enable_scaling is False


class TestRandomForestWrapper:
    """Test Random Forest model wrapper."""
    
    def test_init(self):
        """Test initialization."""
        wrapper = RandomForestWrapper('test_rf', n_estimators=50, random_state=42)
        
        assert wrapper.name == 'test_rf'
        assert wrapper.params['n_estimators'] == 50
        assert wrapper.params['random_state'] == 42
        assert wrapper.is_fitted is False
    
    def test_create_model(self):
        """Test model creation."""
        wrapper = RandomForestWrapper('test_rf', n_estimators=50)
        model = wrapper.create_model()
        
        assert hasattr(model, 'fit')
        assert hasattr(model, 'predict')
        assert hasattr(model, 'predict_proba')
    
    def test_fit_predict(self, sample_feature_set, sample_target):
        """Test fitting and prediction."""
        wrapper = RandomForestWrapper('test_rf', n_estimators=10, random_state=42)
        
        # Prepare features
        X = np.hstack([
            sample_feature_set.features[name].reshape(-1, 1)
            for name in sample_feature_set.feature_names
        ])
        y = sample_target
        
        # Fit model
        wrapper.fit(X, y)
        assert wrapper.is_fitted is True
        
        # Make predictions
        predictions = wrapper.predict(X)
        probabilities = wrapper.predict_proba(X)
        
        assert predictions.shape == (100,)
        assert probabilities.shape == (100, 2)
        assert np.all((predictions >= 0) & (predictions <= 1))
        assert np.allclose(probabilities.sum(axis=1), 1.0)
    
    def test_feature_importance(self, sample_feature_set, sample_target):
        """Test feature importance extraction."""
        wrapper = RandomForestWrapper('test_rf', n_estimators=10, random_state=42)
        wrapper.feature_names = sample_feature_set.feature_names
        
        # Prepare features
        X = np.hstack([
            sample_feature_set.features[name].reshape(-1, 1)
            for name in sample_feature_set.feature_names
        ])
        
        # Fit model
        wrapper.fit(X, sample_target)
        
        # Get feature importance
        importance = wrapper.get_feature_importance()
        
        assert isinstance(importance, dict)
        assert len(importance) == len(sample_feature_set.feature_names)
        assert all(isinstance(v, (int, float)) for v in importance.values())
    
    def test_model_size(self, sample_feature_set, sample_target):
        """Test model size calculation."""
        wrapper = RandomForestWrapper('test_rf', n_estimators=5, random_state=42)
        
        # Before fitting
        size_before = wrapper.get_model_size()
        assert size_before == 0.0
        
        # After fitting
        X = np.hstack([
            sample_feature_set.features[name].reshape(-1, 1)
            for name in sample_feature_set.feature_names
        ])
        wrapper.fit(X, sample_target)
        
        size_after = wrapper.get_model_size()
        assert size_after > 0.0


class TestXGBoostWrapper:
    """Test XGBoost model wrapper."""
    
    @patch('src.ml.model_selector.XGBOOST_AVAILABLE', True)
    @patch('src.ml.model_selector.xgb')
    def test_init_with_xgboost(self, mock_xgb):
        """Test initialization when XGBoost is available."""
        wrapper = XGBoostWrapper('test_xgb', n_estimators=50)
        
        assert wrapper.name == 'test_xgb'
        assert wrapper.params['n_estimators'] == 50
    
    @patch('src.ml.model_selector.XGBOOST_AVAILABLE', False)
    def test_init_without_xgboost(self):
        """Test initialization when XGBoost is not available."""
        wrapper = XGBoostWrapper('test_xgb')
        
        with pytest.raises(ImportError):
            wrapper.create_model()


class TestLightGBMWrapper:
    """Test LightGBM model wrapper."""
    
    @patch('src.ml.model_selector.LIGHTGBM_AVAILABLE', True)
    @patch('src.ml.model_selector.lgb')
    def test_init_with_lightgbm(self, mock_lgb):
        """Test initialization when LightGBM is available."""
        wrapper = LightGBMWrapper('test_lgb', n_estimators=50)
        
        assert wrapper.name == 'test_lgb'
        assert wrapper.params['n_estimators'] == 50
    
    @patch('src.ml.model_selector.LIGHTGBM_AVAILABLE', False)
    def test_init_without_lightgbm(self):
        """Test initialization when LightGBM is not available."""
        wrapper = LightGBMWrapper('test_lgb')
        
        with pytest.raises(ImportError):
            wrapper.create_model()


class TestBERTWrapper:
    """Test BERT model wrapper."""
    
    def test_init(self):
        """Test initialization."""
        wrapper = BERTWrapper('test_bert', model_name='bert-base-uncased')
        
        assert wrapper.name == 'test_bert'
        assert wrapper.model_name == 'bert-base-uncased'
    
    @patch('src.ml.model_selector.TRANSFORMERS_AVAILABLE', False)
    def test_init_without_transformers(self):
        """Test initialization when transformers is not available."""
        wrapper = BERTWrapper('test_bert')
        
        with pytest.raises(ImportError):
            wrapper.create_model()
    
    @patch('src.ml.model_selector.TRANSFORMERS_AVAILABLE', True)
    def test_fit_without_text_data(self, sample_feature_set, sample_target):
        """Test fitting without text data."""
        wrapper = BERTWrapper('test_bert')
        
        X = np.random.randn(100, 10)
        
        with pytest.raises(ValueError):
            wrapper.fit(X, sample_target)


class TestEnsembleWrapper:
    """Test ensemble model wrapper."""
    
    def test_init_voting(self):
        """Test voting ensemble initialization."""
        base_models = [
            RandomForestWrapper('rf_base', n_estimators=10),
            RandomForestWrapper('rf_base2', n_estimators=20)
        ]
        
        ensemble = EnsembleWrapper(
            'test_ensemble',
            base_models=base_models,
            ensemble_type='voting'
        )
        
        assert ensemble.name == 'test_ensemble'
        assert len(ensemble.base_models) == 2
        assert ensemble.ensemble_type == 'voting'
    
    def test_init_stacking(self):
        """Test stacking ensemble initialization."""
        base_models = [
            RandomForestWrapper('rf_base', n_estimators=10),
            RandomForestWrapper('rf_base2', n_estimators=20)
        ]
        
        ensemble = EnsembleWrapper(
            'test_ensemble',
            base_models=base_models,
            ensemble_type='stacking'
        )
        
        assert ensemble.ensemble_type == 'stacking'
    
    def test_unknown_ensemble_type(self):
        """Test unknown ensemble type."""
        base_models = [RandomForestWrapper('rf_base')]
        
        ensemble = EnsembleWrapper(
            'test_ensemble',
            base_models=base_models,
            ensemble_type='unknown'
        )
        
        with pytest.raises(ValueError):
            ensemble.create_model()
    
    def test_fit_predict(self, sample_feature_set, sample_target):
        """Test ensemble fitting and prediction."""
        base_models = [
            RandomForestWrapper('rf_base1', n_estimators=5, random_state=42),
            RandomForestWrapper('rf_base2', n_estimators=5, random_state=43)
        ]
        
        ensemble = EnsembleWrapper(
            'test_ensemble',
            base_models=base_models,
            ensemble_type='voting'
        )
        
        # Prepare features
        X = np.hstack([
            sample_feature_set.features[name].reshape(-1, 1)
            for name in sample_feature_set.feature_names
        ])
        
        # Fit and predict
        ensemble.fit(X, sample_target)
        predictions = ensemble.predict(X)
        probabilities = ensemble.predict_proba(X)
        
        assert predictions.shape == (100,)
        assert probabilities.shape == (100, 2)


class TestModelInterpretability:
    """Test model interpretability analysis."""
    
    def test_init(self):
        """Test initialization."""
        interp = ModelInterpretability()
        
        assert interp.explainer is None
        assert interp.shap_values is None
    
    @patch('src.ml.model_selector.SHAP_AVAILABLE', False)
    def test_analyze_without_shap(self, sample_feature_set, sample_target):
        """Test analysis without SHAP."""
        interp = ModelInterpretability()
        
        # Create and fit model
        wrapper = RandomForestWrapper('test_rf', n_estimators=5, random_state=42)
        X = np.random.randn(50, 10)
        wrapper.fit(X, sample_target[:50])
        
        # Analyze
        results = interp.analyze_model(wrapper, X, sample_feature_set.feature_names)
        
        assert results == {}
    
    @patch('src.ml.model_selector.SHAP_AVAILABLE', True)
    @patch('src.ml.model_selector.shap')
    def test_analyze_with_shap(self, mock_shap, sample_feature_set, sample_target):
        """Test analysis with SHAP."""
        # Mock SHAP components
        mock_explainer = Mock()
        mock_shap_values = Mock()
        mock_shap_values.values = np.random.randn(50, 10)
        
        mock_explainer.return_value = mock_shap_values
        mock_shap.Explainer.return_value = mock_explainer
        
        interp = ModelInterpretability()
        
        # Create and fit model
        wrapper = RandomForestWrapper('test_rf', n_estimators=5, random_state=42)
        X = np.random.randn(50, 10)
        wrapper.fit(X, sample_target[:50])
        
        # Analyze
        results = interp.analyze_model(wrapper, X, sample_feature_set.feature_names)
        
        assert 'feature_importance' in results
        assert 'shap_values_available' in results
        assert 'top_features' in results


class TestModelSelector:
    """Test model selector."""
    
    def test_init(self):
        """Test initialization."""
        selector = ModelSelector()
        
        assert len(selector.models) == 0
        assert len(selector.results) == 0
        assert selector.best_model is None
    
    def test_add_model(self):
        """Test adding models."""
        selector = ModelSelector()
        
        model = RandomForestWrapper('test_rf', n_estimators=10)
        selector.add_model(model)
        
        assert len(selector.models) == 1
        assert 'test_rf' in selector.models
    
    @patch('src.ml.model_selector.XGBOOST_AVAILABLE', True)
    @patch('src.ml.model_selector.LIGHTGBM_AVAILABLE', True)
    def test_add_default_models(self):
        """Test adding default models."""
        selector = ModelSelector()
        selector.add_default_models()
        
        assert len(selector.models) >= 1  # At least Random Forest
        assert 'random_forest' in selector.models
    
    def test_prepare_features(self, sample_feature_set):
        """Test feature preparation."""
        selector = ModelSelector()
        
        X = selector._prepare_features(sample_feature_set)
        
        assert X.shape == (100, 10)
        assert isinstance(X, np.ndarray)
    
    def test_evaluate_models(self, sample_feature_set, sample_target):
        """Test model evaluation."""
        config = ModelConfig(
            test_size=0.2,
            cv_folds=3,
            enable_scaling=False,
            enable_interpretability=False
        )
        
        selector = ModelSelector(config)
        
        # Add simple models
        selector.add_model(RandomForestWrapper('rf1', n_estimators=5, random_state=42))
        selector.add_model(RandomForestWrapper('rf2', n_estimators=10, random_state=42))
        
        # Evaluate
        results = selector.evaluate_models(sample_feature_set, sample_target)
        
        assert len(results) == 2
        assert 'rf1' in results
        assert 'rf2' in results
        
        # Check result structure
        for model_name, performance in results.items():
            assert isinstance(performance, ModelPerformance)
            assert 0 <= performance.accuracy <= 1
            assert 0 <= performance.precision <= 1
            assert 0 <= performance.recall <= 1
            assert 0 <= performance.f1_score <= 1
            assert performance.training_time > 0
            assert len(performance.cv_scores) == config.cv_folds
    
    def test_get_model_ranking(self, sample_feature_set, sample_target):
        """Test model ranking."""
        selector = ModelSelector()
        
        # Add models and evaluate
        selector.add_model(RandomForestWrapper('rf1', n_estimators=5, random_state=42))
        selector.add_model(RandomForestWrapper('rf2', n_estimators=10, random_state=42))
        
        results = selector.evaluate_models(sample_feature_set, sample_target)
        
        # Get ranking
        ranking = selector.get_model_ranking('accuracy')
        
        assert len(ranking) == 2
        assert all(isinstance(item, tuple) for item in ranking)
        assert all(len(item) == 2 for item in ranking)
        
        # Check ordering (should be descending)
        accuracies = [item[1] for item in ranking]
        assert accuracies == sorted(accuracies, reverse=True)
    
    def test_get_performance_summary(self, sample_feature_set, sample_target):
        """Test performance summary."""
        selector = ModelSelector()
        
        # Add models and evaluate
        selector.add_model(RandomForestWrapper('rf1', n_estimators=5, random_state=42))
        
        results = selector.evaluate_models(sample_feature_set, sample_target)
        
        # Get summary
        summary_df = selector.get_performance_summary()
        
        assert isinstance(summary_df, pd.DataFrame)
        assert len(summary_df) == 1
        assert 'model' in summary_df.columns
        assert 'accuracy' in summary_df.columns
        assert 'training_time' in summary_df.columns
    
    def test_save_load_results(self, sample_feature_set, sample_target, tmp_path):
        """Test saving and loading results."""
        selector = ModelSelector()
        
        # Add model and evaluate
        selector.add_model(RandomForestWrapper('rf1', n_estimators=5, random_state=42))
        results = selector.evaluate_models(sample_feature_set, sample_target)
        
        # Save results
        filepath = tmp_path / 'test_results.json'
        selector.save_results(str(filepath))
        
        assert filepath.exists()
        
        # Check file content
        import json
        with open(filepath, 'r') as f:
            saved_data = json.load(f)
        
        assert 'rf1' in saved_data
        assert 'accuracy' in saved_data['rf1']
    
    def test_get_best_model(self, sample_feature_set, sample_target):
        """Test getting best model."""
        selector = ModelSelector()
        
        # Before evaluation
        assert selector.get_best_model() is None
        
        # Add models and evaluate
        selector.add_model(RandomForestWrapper('rf1', n_estimators=5, random_state=42))
        selector.add_model(RandomForestWrapper('rf2', n_estimators=10, random_state=42))
        
        results = selector.evaluate_models(sample_feature_set, sample_target)
        
        # Get best model
        best_model = selector.get_best_model()
        
        assert best_model is not None
        assert best_model in ['rf1', 'rf2']
    
    def test_get_feature_importance_summary(self, sample_feature_set, sample_target):
        """Test feature importance summary."""
        config = ModelConfig(enable_interpretability=False)  # Disable SHAP for simplicity
        selector = ModelSelector(config)
        
        # Add model and evaluate
        selector.add_model(RandomForestWrapper('rf1', n_estimators=5, random_state=42))
        results = selector.evaluate_models(sample_feature_set, sample_target)
        
        # Get feature importance summary
        importance_summary = selector.get_feature_importance_summary()
        
        assert isinstance(importance_summary, dict)
        if importance_summary:  # May be empty if feature importance not available
            assert 'rf1' in importance_summary
            assert isinstance(importance_summary['rf1'], dict)


class TestModelPerformance:
    """Test model performance data structure."""
    
    def test_model_performance_creation(self):
        """Test model performance creation."""
        performance = ModelPerformance(
            model_name='test_model',
            algorithm='RandomForest',
            accuracy=0.85,
            precision=0.83,
            recall=0.87,
            f1_score=0.85,
            roc_auc=0.89,
            cv_scores=[0.82, 0.84, 0.86, 0.85, 0.83],
            cv_mean=0.84,
            cv_std=0.015,
            training_time=12.5,
            prediction_time=0.1,
            model_size_mb=2.3
        )
        
        assert performance.model_name == 'test_model'
        assert performance.algorithm == 'RandomForest'
        assert performance.accuracy == 0.85
        assert performance.cv_mean == 0.84
        assert len(performance.cv_scores) == 5


if __name__ == '__main__':
    pytest.main([__file__])