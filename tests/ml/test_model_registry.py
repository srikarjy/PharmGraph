"""Tests for MLflow model registry integration."""

import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from src.ml.model_registry import (
    MLflowModelRegistry, ModelVersion, ModelPromotionWorkflow,
    ModelLifecycleManager, PromotionCriteria
)
from src.ml.model_selector import RandomForestWrapper
from src.ml.training_pipeline import TrainingResult, TrainingMetrics, TrainingConfig


@pytest.fixture
def mock_training_result():
    """Create mock training result for testing."""
    model = RandomForestWrapper('test_model', n_estimators=10, random_state=42)
    
    # Create mock model
    from sklearn.ensemble import RandomForestClassifier
    model.model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.is_fitted = True
    
    metrics = TrainingMetrics(
        model_name='test_model',
        training_time=30.5,
        validation_scores={'accuracy': 0.85, 'precision': 0.83, 'recall': 0.87, 'f1_score': 0.85},
        test_scores={'accuracy': 0.82, 'precision': 0.80, 'recall': 0.84, 'f1_score': 0.82},
        cv_scores={'accuracy': [0.81, 0.83, 0.82, 0.84, 0.80]}
    )
    
    return TrainingResult(
        pipeline_id='test_pipeline',
        config=TrainingConfig(),
        model=model,
        metrics=metrics,
        preprocessors={},
        feature_names=['feature_1', 'feature_2', 'feature_3'],
        created_at=datetime.now()
    )


class TestModelVersion:
    """Test ModelVersion data structure."""
    
    def test_model_version_creation(self):
        """Test model version creation."""
        version = ModelVersion(
            name='test_model',
            version='1',
            stage='Production',
            run_id='run_123',
            model_uri='models:/test_model/1',
            creation_timestamp=datetime.now(),
            last_updated_timestamp=datetime.now(),
            description='Test model version',
            tags={'env': 'test'},
            metrics={'accuracy': 0.85},
            params={'n_estimators': 100}
        )
        
        assert version.name == 'test_model'
        assert version.version == '1'
        assert version.stage == 'Production'
        assert version.metrics['accuracy'] == 0.85
        assert version.tags['env'] == 'test'


class TestMLflowModelRegistry:
    """Test MLflow model registry."""
    
    def test_init_without_mlflow(self):
        """Test initialization without MLflow."""
        with patch('src.ml.model_registry.MLFLOW_AVAILABLE', False):
            with pytest.raises(ImportError):
                MLflowModelRegistry()
    
    @patch('src.ml.model_registry.MLFLOW_AVAILABLE', True)
    @patch('src.ml.model_registry.mlflow')
    @patch('src.ml.model_registry.MlflowClient')
    def test_init_with_mlflow(self, mock_client, mock_mlflow):
        """Test initialization with MLflow."""
        registry = MLflowModelRegistry()
        
        assert registry.client is not None
        mock_mlflow.set_tracking_uri.assert_not_called()  # No URI provided
        mock_mlflow.set_registry_uri.assert_not_called()   # No URI provided
    
    @patch('src.ml.model_registry.MLFLOW_AVAILABLE', True)
    @patch('src.ml.model_registry.mlflow')
    @patch('src.ml.model_registry.MlflowClient')
    def test_init_with_uris(self, mock_client, mock_mlflow):
        """Test initialization with custom URIs."""
        registry = MLflowModelRegistry(
            tracking_uri='http://localhost:5000',
            registry_uri='http://localhost:5001'
        )
        
        mock_mlflow.set_tracking_uri.assert_called_once_with('http://localhost:5000')
        mock_mlflow.set_registry_uri.assert_called_once_with('http://localhost:5001')
    
    @patch('src.ml.model_registry.MLFLOW_AVAILABLE', True)
    @patch('src.ml.model_registry.mlflow')
    @patch('src.ml.model_registry.MlflowClient')
    def test_register_model(self, mock_client, mock_mlflow, mock_training_result):
        """Test model registration."""
        # Mock MLflow components
        mock_run = Mock()
        mock_run.info.run_id = 'test_run_id'
        
        mock_model_version = Mock()
        mock_model_version.version = '1'
        mock_model_version.current_stage = 'None'
        mock_model_version.source = 'runs:/test_run_id/model'
        mock_model_version.creation_timestamp = int(datetime.now().timestamp() * 1000)
        mock_model_version.last_updated_timestamp = int(datetime.now().timestamp() * 1000)
        
        mock_mlflow.start_run.return_value.__enter__.return_value = mock_run
        mock_mlflow.register_model.return_value = mock_model_version
        
        registry = MLflowModelRegistry()
        
        # Register model
        result = registry.register_model(
            model=mock_training_result.model,
            model_name='test_model',
            training_result=mock_training_result,
            description='Test model'
        )
        
        assert isinstance(result, ModelVersion)
        assert result.name == 'test_model'
        assert result.version == '1'
        assert result.stage == 'None'
    
    @patch('src.ml.model_registry.MLFLOW_AVAILABLE', True)
    @patch('src.ml.model_registry.mlflow')
    @patch('src.ml.model_registry.MlflowClient')
    def test_get_model_versions(self, mock_client_class, mock_mlflow):
        """Test getting model versions."""
        # Mock client instance
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        # Mock model version
        mock_version = Mock()
        mock_version.version = '1'
        mock_version.current_stage = 'Production'
        mock_version.run_id = 'run_123'
        mock_version.source = 'models:/test_model/1'
        mock_version.creation_timestamp = int(datetime.now().timestamp() * 1000)
        mock_version.last_updated_timestamp = int(datetime.now().timestamp() * 1000)
        mock_version.description = 'Test version'
        mock_version.tags = {'env': 'test'}
        
        mock_client.search_model_versions.return_value = [mock_version]
        
        # Mock run for metrics
        mock_run = Mock()
        mock_run.data.metrics = {'accuracy': 0.85}
        mock_run.data.params = {'n_estimators': 100}
        mock_client.get_run.return_value = mock_run
        
        registry = MLflowModelRegistry()
        versions = registry.get_model_versions('test_model')
        
        assert len(versions) == 1
        assert versions[0].name == 'test_model'
        assert versions[0].version == '1'
        assert versions[0].metrics['accuracy'] == 0.85
    
    @patch('src.ml.model_registry.MLFLOW_AVAILABLE', True)
    @patch('src.ml.model_registry.mlflow')
    @patch('src.ml.model_registry.MlflowClient')
    def test_promote_model(self, mock_client_class, mock_mlflow):
        """Test model promotion."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        registry = MLflowModelRegistry()
        
        # Test successful promotion
        result = registry.promote_model('test_model', '1', 'Production')
        
        assert result is True
        mock_client.transition_model_version_stage.assert_called_with(
            name='test_model',
            version='1',
            stage='Production'
        )
    
    @patch('src.ml.model_registry.MLFLOW_AVAILABLE', True)
    @patch('src.ml.model_registry.mlflow')
    @patch('src.ml.model_registry.MlflowClient')
    def test_load_model(self, mock_client_class, mock_mlflow):
        """Test model loading."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        # Mock sklearn model loading
        mock_model = Mock()
        mock_mlflow.sklearn.load_model.return_value = mock_model
        
        registry = MLflowModelRegistry()
        
        # Test loading by version
        loaded_model = registry.load_model('test_model', version='1')
        
        assert loaded_model == mock_model
        mock_mlflow.sklearn.load_model.assert_called_with('models:/test_model/1')
    
    @patch('src.ml.model_registry.MLFLOW_AVAILABLE', True)
    @patch('src.ml.model_registry.mlflow')
    @patch('src.ml.model_registry.MlflowClient')
    def test_compare_models(self, mock_client_class, mock_mlflow):
        """Test model comparison."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        registry = MLflowModelRegistry()
        
        # Mock get_model_versions to return test versions
        registry.get_model_versions = Mock(return_value=[
            ModelVersion(
                name='test_model', version='1', stage='None', run_id='run1',
                model_uri='uri1', creation_timestamp=datetime.now(),
                last_updated_timestamp=datetime.now(),
                metrics={'test_accuracy': 0.80}
            ),
            ModelVersion(
                name='test_model', version='2', stage='None', run_id='run2',
                model_uri='uri2', creation_timestamp=datetime.now(),
                last_updated_timestamp=datetime.now(),
                metrics={'test_accuracy': 0.85}
            )
        ])
        
        comparison = registry.compare_models('test_model')
        
        assert len(comparison.model_versions) == 2
        assert comparison.best_model.version == '2'  # Higher accuracy
        assert comparison.best_model.metrics['test_accuracy'] == 0.85


class TestPromotionCriteria:
    """Test promotion criteria."""
    
    def test_default_criteria(self):
        """Test default promotion criteria."""
        criteria = PromotionCriteria()
        
        assert criteria.min_accuracy == 0.8
        assert criteria.min_precision == 0.75
        assert criteria.require_approval is True
    
    def test_custom_criteria(self):
        """Test custom promotion criteria."""
        criteria = PromotionCriteria(
            min_accuracy=0.9,
            min_precision=0.85,
            require_approval=False
        )
        
        assert criteria.min_accuracy == 0.9
        assert criteria.min_precision == 0.85
        assert criteria.require_approval is False


class TestModelPromotionWorkflow:
    """Test model promotion workflow."""
    
    @patch('src.ml.model_registry.MLFLOW_AVAILABLE', True)
    @patch('src.ml.model_registry.mlflow')
    @patch('src.ml.model_registry.MlflowClient')
    def test_evaluate_for_promotion(self, mock_client_class, mock_mlflow):
        """Test promotion evaluation."""
        registry = MLflowModelRegistry()
        criteria = PromotionCriteria(min_accuracy=0.8, min_precision=0.75)
        workflow = ModelPromotionWorkflow(registry, criteria)
        
        # Mock model version with good metrics
        good_version = ModelVersion(
            name='test_model', version='1', stage='None', run_id='run1',
            model_uri='uri1', creation_timestamp=datetime.now(),
            last_updated_timestamp=datetime.now(),
            metrics={
                'test_accuracy': 0.85,
                'test_precision': 0.80,
                'test_recall': 0.82,
                'test_f1_score': 0.81
            }
        )
        
        registry.get_model_versions = Mock(return_value=[good_version])
        
        evaluation = workflow.evaluate_for_promotion('test_model', '1')
        
        assert evaluation['eligible'] is True
        assert evaluation['checks']['accuracy'] is True
        assert evaluation['checks']['precision'] is True
    
    @patch('src.ml.model_registry.MLFLOW_AVAILABLE', True)
    @patch('src.ml.model_registry.mlflow')
    @patch('src.ml.model_registry.MlflowClient')
    def test_auto_promote_eligible(self, mock_client_class, mock_mlflow):
        """Test auto-promotion of eligible model."""
        registry = MLflowModelRegistry()
        registry.promote_model = Mock(return_value=True)
        
        criteria = PromotionCriteria(min_accuracy=0.8, require_approval=False)
        workflow = ModelPromotionWorkflow(registry, criteria)
        
        # Mock evaluation to return eligible
        workflow.evaluate_for_promotion = Mock(return_value={'eligible': True})
        
        result = workflow.auto_promote('test_model', '1', 'Staging')
        
        assert result is True
        registry.promote_model.assert_called_once_with(
            model_name='test_model',
            version='1',
            stage='Staging',
            archive_existing=True
        )
    
    @patch('src.ml.model_registry.MLFLOW_AVAILABLE', True)
    @patch('src.ml.model_registry.mlflow')
    @patch('src.ml.model_registry.MlflowClient')
    def test_auto_promote_not_eligible(self, mock_client_class, mock_mlflow):
        """Test auto-promotion of non-eligible model."""
        registry = MLflowModelRegistry()
        workflow = ModelPromotionWorkflow(registry)
        
        # Mock evaluation to return not eligible
        workflow.evaluate_for_promotion = Mock(return_value={
            'eligible': False,
            'reason': 'Low accuracy'
        })
        
        result = workflow.auto_promote('test_model', '1', 'Staging')
        
        assert result is False


class TestModelLifecycleManager:
    """Test model lifecycle manager."""
    
    @patch('src.ml.model_registry.MLFLOW_AVAILABLE', True)
    @patch('src.ml.model_registry.mlflow')
    @patch('src.ml.model_registry.MlflowClient')
    def test_deploy_model_pipeline(self, mock_client_class, mock_mlflow, mock_training_result):
        """Test complete model pipeline deployment."""
        registry = MLflowModelRegistry()
        manager = ModelLifecycleManager(registry)
        
        # Mock registry methods
        mock_version = ModelVersion(
            name='test_model', version='1', stage='None', run_id='run1',
            model_uri='uri1', creation_timestamp=datetime.now(),
            last_updated_timestamp=datetime.now(),
            metrics={'test_accuracy': 0.85}
        )
        
        registry.register_model = Mock(return_value=mock_version)
        manager.promotion_workflow.auto_promote = Mock(return_value=True)
        
        result = manager.deploy_model_pipeline(
            training_result=mock_training_result,
            model_name='test_model',
            auto_promote=True
        )
        
        assert result['status'] == 'success'
        assert result['registration']['version'] == '1'
        assert result['promotion']['successful'] is True
    
    @patch('src.ml.model_registry.MLFLOW_AVAILABLE', True)
    @patch('src.ml.model_registry.mlflow')
    @patch('src.ml.model_registry.MlflowClient')
    def test_get_model_health_report(self, mock_client_class, mock_mlflow):
        """Test model health report generation."""
        registry = MLflowModelRegistry()
        manager = ModelLifecycleManager(registry)
        
        # Mock model versions
        versions = [
            ModelVersion(
                name='test_model', version='2', stage='Production', run_id='run2',
                model_uri='uri2', creation_timestamp=datetime.now(),
                last_updated_timestamp=datetime.now(),
                metrics={'test_accuracy': 0.85}
            ),
            ModelVersion(
                name='test_model', version='1', stage='Staging', run_id='run1',
                model_uri='uri1', creation_timestamp=datetime.now(),
                last_updated_timestamp=datetime.now(),
                metrics={'test_accuracy': 0.80}
            )
        ]
        
        registry.get_model_versions = Mock(return_value=versions)
        
        report = manager.get_model_health_report('test_model')
        
        assert report['model_name'] == 'test_model'
        assert report['total_versions'] == 2
        assert report['production_version'] == '2'
        assert report['staging_version'] == '1'
        assert report['health_status'] == 'healthy'
    
    @patch('src.ml.model_registry.MLFLOW_AVAILABLE', True)
    @patch('src.ml.model_registry.mlflow')
    @patch('src.ml.model_registry.MlflowClient')
    def test_cleanup_old_versions(self, mock_client_class, mock_mlflow):
        """Test cleanup of old model versions."""
        registry = MLflowModelRegistry()
        registry.delete_model_version = Mock(return_value=True)
        
        manager = ModelLifecycleManager(registry)
        
        # Mock old versions (keeping production/staging)
        old_time = datetime.now() - timedelta(days=30)
        versions = [
            ModelVersion(
                name='test_model', version='5', stage='Production', run_id='run5',
                model_uri='uri5', creation_timestamp=datetime.now(),
                last_updated_timestamp=datetime.now()
            ),
            ModelVersion(
                name='test_model', version='4', stage='None', run_id='run4',
                model_uri='uri4', creation_timestamp=old_time,
                last_updated_timestamp=old_time
            ),
            ModelVersion(
                name='test_model', version='3', stage='None', run_id='run3',
                model_uri='uri3', creation_timestamp=old_time,
                last_updated_timestamp=old_time
            )
        ]
        
        registry.get_model_versions = Mock(return_value=versions)
        
        deleted_count = manager.cleanup_old_versions('test_model', keep_versions=1)
        
        assert deleted_count == 1  # Should delete 1 old version (keeping 1 + production)


if __name__ == '__main__':
    pytest.main([__file__])