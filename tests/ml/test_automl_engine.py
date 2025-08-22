"""Tests for AutoML engine with Ray and Optuna integration."""

import pytest
import numpy as np
from datetime import datetime
from unittest.mock import Mock, patch

from src.ml.automl_engine import (
    AutoMLEngine, AutoMLConfig, HyperparameterSpace,
    RayTrialRunner, OptunaTuner, TrialResult
)
from src.ml.feature_extraction import FeatureSet, FeatureMetadata
from src.ml.model_selector import RandomForestWrapper


class TestHyperparameterSpace:
    """Test hyperparameter space definitions."""
    
    def test_random_forest_space(self):
        """Test Random Forest hyperparameter space."""
        space = HyperparameterSpace.get_random_forest_space()
        
        assert 'n_estimators' in space
        assert 'max_depth' in space
        assert 'min_samples_split' in space
        assert space['n_estimators'][0] == 'int'
        assert space['max_features'][0] == 'categorical'
    
    def test_xgboost_space(self):
        """Test XGBoost hyperparameter space."""
        space = HyperparameterSpace.get_xgboost_space()
        
        assert 'n_estimators' in space
        assert 'learning_rate' in space
        assert 'max_depth' in space
        assert space['learning_rate'][0] == 'float'
    
    def test_lightgbm_space(self):
        """Test LightGBM hyperparameter space."""
        space = HyperparameterSpace.get_lightgbm_space()
        
        assert 'n_estimators' in space
        assert 'num_leaves' in space
        assert 'learning_rate' in space
    
    def test_bert_space(self):
        """Test BERT hyperparameter space."""
        space = HyperparameterSpace.get_bert_space()
        
        assert 'learning_rate' in space
        assert 'batch_size' in space
        assert 'num_epochs' in space


class TestAutoMLConfig:
    """Test AutoML configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = AutoMLConfig()
        
        assert config.n_trials == 100
        assert config.cv_folds == 5
        assert config.random_state == 42
        assert config.enable_early_stopping is True
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = AutoMLConfig(
            n_trials=50,
            timeout=1800,
            cv_folds=3,
            random_state=123
        )
        
        assert config.n_trials == 50
        assert config.timeout == 1800
        assert config.cv_folds == 3
        assert config.random_state == 123


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
    """Create sample target for testing."""
    return np.random.randint(0, 2, size=100)


class TestRayTrialRunner:
    """Test Ray trial runner (without actual Ray)."""
    
    def test_init(self):
        """Test initialization."""
        config = AutoMLConfig()
        
        # Mock Ray availability
        with patch('src.ml.automl_engine.RAY_AVAILABLE', False):
            with pytest.raises(ImportError):
                RayTrialRunner(config)
    
    def test_prepare_features(self, sample_feature_set):
        """Test feature preparation."""
        config = AutoMLConfig()
        
        with patch('src.ml.automl_engine.RAY_AVAILABLE', True):
            runner = RayTrialRunner(config)
            X = runner._prepare_features(sample_feature_set)
            
            assert X.shape[0] == 100  # n_samples
            assert X.shape[1] == 10   # n_features
    
    def test_create_model(self):
        """Test model creation."""
        config = AutoMLConfig()
        
        with patch('src.ml.automl_engine.RAY_AVAILABLE', True):
            runner = RayTrialRunner(config)
            
            # Test Random Forest
            params = {'n_estimators': 100, 'max_depth': 10}
            model = runner._create_model('random_forest', params)
            
            assert hasattr(model, 'fit')
            assert hasattr(model, 'predict')


class TestOptunaTuner:
    """Test Optuna tuner (without actual Optuna)."""
    
    def test_init(self):
        """Test initialization."""
        config = AutoMLConfig()
        
        with patch('src.ml.automl_engine.OPTUNA_AVAILABLE', False):
            with pytest.raises(ImportError):
                OptunaTuner(config)
    
    @patch('src.ml.automl_engine.OPTUNA_AVAILABLE', True)
    @patch('src.ml.automl_engine.optuna')
    def test_create_study(self, mock_optuna):
        """Test study creation."""
        config = AutoMLConfig()
        tuner = OptunaTuner(config)
        
        # Mock optuna components
        mock_study = Mock()
        mock_optuna.create_study.return_value = mock_study
        mock_optuna.samplers.TPESampler.return_value = Mock()
        mock_optuna.pruners.MedianPruner.return_value = Mock()
        
        study = tuner.create_study('test_study')
        
        assert study == mock_study
        mock_optuna.create_study.assert_called_once()


class TestAutoMLEngine:
    """Test AutoML engine."""
    
    def test_init_without_dependencies(self):
        """Test initialization without Ray/Optuna."""
        with patch('src.ml.automl_engine.RAY_AVAILABLE', False):
            with patch('src.ml.automl_engine.OPTUNA_AVAILABLE', False):
                with pytest.raises(ImportError):
                    AutoMLEngine()
    
    @patch('src.ml.automl_engine.RAY_AVAILABLE', False)
    @patch('src.ml.automl_engine.OPTUNA_AVAILABLE', True)
    @patch('src.ml.automl_engine.optuna')
    def test_init_with_optuna_only(self, mock_optuna):
        """Test initialization with Optuna only."""
        config = AutoMLConfig()
        engine = AutoMLEngine(config)
        
        assert engine.ray_runner is None
        assert engine.optuna_tuner is not None
    
    @patch('src.ml.automl_engine.RAY_AVAILABLE', False)
    @patch('src.ml.automl_engine.OPTUNA_AVAILABLE', True)
    @patch('src.ml.automl_engine.optuna')
    def test_get_parameter_space(self, mock_optuna):
        """Test parameter space retrieval."""
        config = AutoMLConfig()
        engine = AutoMLEngine(config)
        
        # Test Random Forest
        space = engine._get_parameter_space('random_forest')
        assert 'n_estimators' in space
        
        # Test unknown algorithm
        with pytest.raises(ValueError):
            engine._get_parameter_space('unknown_algorithm')
    
    @patch('src.ml.automl_engine.RAY_AVAILABLE', False)
    @patch('src.ml.automl_engine.OPTUNA_AVAILABLE', True)
    @patch('src.ml.automl_engine.optuna')
    def test_prepare_features(self, mock_optuna, sample_feature_set):
        """Test feature preparation."""
        config = AutoMLConfig()
        engine = AutoMLEngine(config)
        
        X = engine._prepare_features(sample_feature_set)
        
        assert X.shape[0] == 100
        assert X.shape[1] == 10
    
    @patch('src.ml.automl_engine.RAY_AVAILABLE', False)
    @patch('src.ml.automl_engine.OPTUNA_AVAILABLE', True)
    @patch('src.ml.automl_engine.optuna')
    def test_run_trial_sequential(self, mock_optuna, sample_feature_set, sample_target):
        """Test sequential trial execution."""
        config = AutoMLConfig()
        engine = AutoMLEngine(config)
        
        params = {'n_estimators': 50, 'max_depth': 5}
        
        metrics = engine._run_trial_sequential(
            params, sample_feature_set, sample_target, 'random_forest'
        )
        
        assert 'accuracy' in metrics
        assert isinstance(metrics['accuracy'], float)
        assert 0 <= metrics['accuracy'] <= 1
    
    @patch('src.ml.automl_engine.RAY_AVAILABLE', False)
    @patch('src.ml.automl_engine.OPTUNA_AVAILABLE', True)
    @patch('src.ml.automl_engine.optuna')
    def test_save_load_results(self, mock_optuna, tmp_path):
        """Test saving and loading results."""
        config = AutoMLConfig()
        engine = AutoMLEngine(config)
        
        # Create mock results
        results = {
            'best_algorithm': 'random_forest',
            'best_params': {'n_estimators': 100},
            'best_score': 0.85,
            'algorithm_results': {
                'random_forest': {
                    'best_params': {'n_estimators': 100},
                    'best_value': 0.85,
                    'n_trials': 10
                }
            },
            'total_trials': 10
        }
        
        # Save results
        filepath = tmp_path / 'test_results.json'
        engine.save_results(results, str(filepath))
        
        assert filepath.exists()
        
        # Load results
        loaded_results = engine.load_results(str(filepath))
        
        assert loaded_results['best_algorithm'] == 'random_forest'
        assert loaded_results['best_score'] == 0.85


class TestTrialResult:
    """Test trial result data structure."""
    
    def test_trial_result_creation(self):
        """Test trial result creation."""
        result = TrialResult(
            trial_id='test_trial_1',
            parameters={'n_estimators': 100, 'max_depth': 10},
            metrics={'accuracy': 0.85, 'f1_score': 0.82},
            status='completed',
            duration=45.5,
            model_name='random_forest'
        )
        
        assert result.trial_id == 'test_trial_1'
        assert result.parameters['n_estimators'] == 100
        assert result.metrics['accuracy'] == 0.85
        assert result.status == 'completed'
        assert result.duration == 45.5
        assert result.model_name == 'random_forest'
        assert isinstance(result.created_at, datetime)


class TestIntegration:
    """Integration tests for AutoML components."""
    
    @patch('src.ml.automl_engine.RAY_AVAILABLE', False)
    @patch('src.ml.automl_engine.OPTUNA_AVAILABLE', True)
    @patch('src.ml.automl_engine.optuna')
    def test_end_to_end_workflow(self, mock_optuna, sample_feature_set, sample_target):
        """Test end-to-end AutoML workflow."""
        # Mock Optuna components
        mock_trial = Mock()
        mock_trial.params = {'n_estimators': 100, 'max_depth': 10}
        mock_trial.value = 0.85
        mock_trial.number = 1
        mock_trial.state = Mock()
        mock_trial.state.name = 'COMPLETE'
        
        mock_study = Mock()
        mock_study.best_trial = mock_trial
        mock_study.trials = [mock_trial]
        
        mock_optuna.create_study.return_value = mock_study
        mock_optuna.trial.TrialState.COMPLETE = 'COMPLETE'
        mock_optuna.samplers.TPESampler.return_value = Mock()
        mock_optuna.pruners.MedianPruner.return_value = Mock()
        
        # Create AutoML engine
        config = AutoMLConfig(n_trials=1, timeout=10)
        engine = AutoMLEngine(config)
        
        # This would normally run optimization, but we'll mock it
        # to avoid actual Optuna dependency in tests
        
        # Test parameter space retrieval
        space = engine._get_parameter_space('random_forest')
        assert isinstance(space, dict)
        
        # Test feature preparation
        X = engine._prepare_features(sample_feature_set)
        assert X.shape == (100, 10)
        
        # Test sequential trial execution
        metrics = engine._run_trial_sequential(
            {'n_estimators': 50}, sample_feature_set, sample_target, 'random_forest'
        )
        assert 'accuracy' in metrics


if __name__ == '__main__':
    pytest.main([__file__])