"""MLflow model registry integration for model versioning and lifecycle management."""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
import json
import pickle
import os
from pathlib import Path
import logging
import tempfile
import shutil

# MLflow imports
try:
    import mlflow
    import mlflow.sklearn
    import mlflow.pytorch
    from mlflow.tracking import MlflowClient
    from mlflow.entities import ViewType
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# Model imports
from sklearn.base import BaseEstimator
import joblib

from ..config import get_logger
from .model_selector import BaseModelWrapper
from .training_pipeline import TrainingResult


logger = get_logger(__name__)


@dataclass
class ModelVersion:
    """Model version information."""
    name: str
    version: str
    stage: str  # 'None', 'Staging', 'Production', 'Archived'
    run_id: str
    model_uri: str
    creation_timestamp: datetime
    last_updated_timestamp: datetime
    description: Optional[str] = None
    tags: Dict[str, str] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelComparison:
    """Model comparison results."""
    model_versions: List[ModelVersion]
    comparison_metrics: Dict[str, Dict[str, float]]
    best_model: Optional[ModelVersion] = None
    recommendation: Optional[str] = None


@dataclass
class PromotionCriteria:
    """Criteria for model promotion."""
    min_accuracy: float = 0.8
    min_precision: float = 0.75
    min_recall: float = 0.75
    min_f1_score: float = 0.75
    max_latency_ms: float = 200.0
    min_throughput_rps: float = 100.0
    require_approval: bool = True
    staging_duration_hours: int = 24


class MLflowModelRegistry:
    """MLflow-based model registry for versioning and lifecycle management."""
    
    def __init__(self, tracking_uri: Optional[str] = None, 
                 registry_uri: Optional[str] = None):
        """Initialize MLflow model registry.
        
        Args:
            tracking_uri: MLflow tracking server URI
            registry_uri: MLflow model registry URI
        """
        if not MLFLOW_AVAILABLE:
            raise ImportError("MLflow is required for model registry functionality")
        
        # Set MLflow URIs
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        if registry_uri:
            mlflow.set_registry_uri(registry_uri)
        
        # Initialize MLflow client
        self.client = MlflowClient()
        
        # Model registry state
        self.registered_models = {}
        
        logger.info("MLflow model registry initialized")
    
    def register_model(self, model: BaseModelWrapper,
                      model_name: str,
                      training_result: TrainingResult,
                      description: Optional[str] = None,
                      tags: Optional[Dict[str, str]] = None) -> ModelVersion:
        """Register a trained model in MLflow.
        
        Args:
            model: Trained model wrapper
            model_name: Name for the registered model
            training_result: Training result containing metrics
            description: Model description
            tags: Model tags
            
        Returns:
            ModelVersion object
        """
        logger.info(f"Registering model: {model_name}")
        
        # Start MLflow run
        with mlflow.start_run() as run:
            # Log parameters
            if hasattr(model, 'params'):
                mlflow.log_params(model.params)
            
            # Log metrics
            if training_result.metrics:
                metrics_dict = {
                    'test_accuracy': training_result.metrics.test_scores.get('accuracy', 0.0),
                    'test_precision': training_result.metrics.test_scores.get('precision', 0.0),
                    'test_recall': training_result.metrics.test_scores.get('recall', 0.0),
                    'test_f1_score': training_result.metrics.test_scores.get('f1_score', 0.0),
                    'cv_accuracy_mean': np.mean(training_result.metrics.cv_scores.get('accuracy', [0.0])),
                    'cv_accuracy_std': np.std(training_result.metrics.cv_scores.get('accuracy', [0.0])),
                    'training_time': training_result.metrics.training_time
                }
                mlflow.log_metrics(metrics_dict)
            
            # Log model artifacts
            if hasattr(model.model, 'fit'):  # Sklearn-like model
                mlflow.sklearn.log_model(
                    sk_model=model.model,
                    artifact_path="model",
                    registered_model_name=model_name
                )
            else:
                # Fallback: save with joblib
                with tempfile.TemporaryDirectory() as temp_dir:
                    model_path = os.path.join(temp_dir, "model.pkl")
                    joblib.dump(model.model, model_path)
                    mlflow.log_artifact(model_path, "model")
            
            # Log additional artifacts
            if training_result.feature_names:
                feature_info = {
                    'feature_names': training_result.feature_names,
                    'n_features': len(training_result.feature_names)
                }
                
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                    json.dump(feature_info, f, indent=2)
                    mlflow.log_artifact(f.name, "feature_info")
                    os.unlink(f.name)
            
            # Register model version
            try:
                model_version = mlflow.register_model(
                    model_uri=f"runs:/{run.info.run_id}/model",
                    name=model_name,
                    description=description,
                    tags=tags
                )
                
                # Create ModelVersion object
                version_info = ModelVersion(
                    name=model_name,
                    version=model_version.version,
                    stage=model_version.current_stage,
                    run_id=run.info.run_id,
                    model_uri=model_version.source,
                    creation_timestamp=datetime.fromtimestamp(model_version.creation_timestamp / 1000),
                    last_updated_timestamp=datetime.fromtimestamp(model_version.last_updated_timestamp / 1000),
                    description=description,
                    tags=tags or {},
                    metrics=metrics_dict if 'metrics_dict' in locals() else {},
                    params=model.params if hasattr(model, 'params') else {}
                )
                
                self.registered_models[model_name] = version_info
                
                logger.info(f"Model registered: {model_name} v{model_version.version}")
                return version_info
                
            except Exception as e:
                logger.error(f"Failed to register model: {e}")
                raise
    
    def get_model_versions(self, model_name: str) -> List[ModelVersion]:
        """Get all versions of a registered model.
        
        Args:
            model_name: Name of the registered model
            
        Returns:
            List of model versions
        """
        try:
            versions = self.client.search_model_versions(f"name='{model_name}'")
            
            model_versions = []
            for version in versions:
                model_version = ModelVersion(
                    name=model_name,
                    version=version.version,
                    stage=version.current_stage,
                    run_id=version.run_id,
                    model_uri=version.source,
                    creation_timestamp=datetime.fromtimestamp(version.creation_timestamp / 1000),
                    last_updated_timestamp=datetime.fromtimestamp(version.last_updated_timestamp / 1000),
                    description=version.description,
                    tags=dict(version.tags) if version.tags else {}
                )
                
                # Get metrics from run
                try:
                    run = self.client.get_run(version.run_id)
                    model_version.metrics = dict(run.data.metrics)
                    model_version.params = dict(run.data.params)
                except:
                    pass
                
                model_versions.append(model_version)
            
            return sorted(model_versions, key=lambda x: x.version, reverse=True)
            
        except Exception as e:
            logger.error(f"Failed to get model versions: {e}")
            return []
    
    def get_latest_model_version(self, model_name: str, 
                                stage: Optional[str] = None) -> Optional[ModelVersion]:
        """Get the latest version of a model.
        
        Args:
            model_name: Name of the registered model
            stage: Optional stage filter ('Production', 'Staging', etc.)
            
        Returns:
            Latest model version or None
        """
        versions = self.get_model_versions(model_name)
        
        if stage:
            versions = [v for v in versions if v.stage == stage]
        
        return versions[0] if versions else None
    
    def compare_models(self, model_name: str, 
                      versions: Optional[List[str]] = None,
                      metrics: Optional[List[str]] = None) -> ModelComparison:
        """Compare different versions of a model.
        
        Args:
            model_name: Name of the registered model
            versions: Specific versions to compare (default: all)
            metrics: Metrics to compare (default: common metrics)
            
        Returns:
            ModelComparison object
        """
        if metrics is None:
            metrics = ['test_accuracy', 'test_precision', 'test_recall', 'test_f1_score']
        
        # Get model versions
        all_versions = self.get_model_versions(model_name)
        
        if versions:
            model_versions = [v for v in all_versions if v.version in versions]
        else:
            model_versions = all_versions
        
        # Compare metrics
        comparison_metrics = {}
        for metric in metrics:
            comparison_metrics[metric] = {}
            for version in model_versions:
                comparison_metrics[metric][version.version] = version.metrics.get(metric, 0.0)
        
        # Find best model (by accuracy)
        best_model = None
        best_accuracy = 0.0
        
        for version in model_versions:
            accuracy = version.metrics.get('test_accuracy', 0.0)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = version
        
        # Generate recommendation
        recommendation = None
        if best_model:
            if best_model.stage != 'Production':
                recommendation = f"Consider promoting version {best_model.version} to Production"
            else:
                recommendation = f"Version {best_model.version} is already in Production"
        
        return ModelComparison(
            model_versions=model_versions,
            comparison_metrics=comparison_metrics,
            best_model=best_model,
            recommendation=recommendation
        )
    
    def promote_model(self, model_name: str, version: str, 
                     stage: str, archive_existing: bool = True) -> bool:
        """Promote a model version to a new stage.
        
        Args:
            model_name: Name of the registered model
            version: Version to promote
            stage: Target stage ('Staging', 'Production', 'Archived')
            archive_existing: Whether to archive existing models in target stage
            
        Returns:
            True if successful
        """
        try:
            # Archive existing models in target stage if requested
            if archive_existing and stage in ['Staging', 'Production']:
                existing_versions = self.get_model_versions(model_name)
                for existing_version in existing_versions:
                    if existing_version.stage == stage:
                        self.client.transition_model_version_stage(
                            name=model_name,
                            version=existing_version.version,
                            stage='Archived'
                        )
                        logger.info(f"Archived {model_name} v{existing_version.version}")
            
            # Promote new version
            self.client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=stage
            )
            
            logger.info(f"Promoted {model_name} v{version} to {stage}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to promote model: {e}")
            return False
    
    def load_model(self, model_name: str, version: Optional[str] = None,
                  stage: Optional[str] = None):
        """Load a model from the registry.
        
        Args:
            model_name: Name of the registered model
            version: Specific version to load
            stage: Stage to load from ('Production', 'Staging')
            
        Returns:
            Loaded model object
        """
        try:
            if version:
                model_uri = f"models:/{model_name}/{version}"
            elif stage:
                model_uri = f"models:/{model_name}/{stage}"
            else:
                # Load latest version
                latest_version = self.get_latest_model_version(model_name)
                if not latest_version:
                    raise ValueError(f"No versions found for model {model_name}")
                model_uri = f"models:/{model_name}/{latest_version.version}"
            
            # Try to load as sklearn model first
            try:
                model = mlflow.sklearn.load_model(model_uri)
                logger.info(f"Loaded sklearn model: {model_uri}")
                return model
            except:
                # Fallback to generic model loading
                model = mlflow.pyfunc.load_model(model_uri)
                logger.info(f"Loaded generic model: {model_uri}")
                return model
                
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def delete_model_version(self, model_name: str, version: str) -> bool:
        """Delete a specific model version.
        
        Args:
            model_name: Name of the registered model
            version: Version to delete
            
        Returns:
            True if successful
        """
        try:
            self.client.delete_model_version(
                name=model_name,
                version=version
            )
            
            logger.info(f"Deleted {model_name} v{version}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete model version: {e}")
            return False
    
    def search_models(self, filter_string: Optional[str] = None,
                     max_results: int = 100) -> List[str]:
        """Search for registered models.
        
        Args:
            filter_string: Filter string for search
            max_results: Maximum number of results
            
        Returns:
            List of model names
        """
        try:
            models = self.client.search_registered_models(
                filter_string=filter_string,
                max_results=max_results
            )
            
            return [model.name for model in models]
            
        except Exception as e:
            logger.error(f"Failed to search models: {e}")
            return []
    
    def get_model_metadata(self, model_name: str, version: str) -> Dict[str, Any]:
        """Get comprehensive metadata for a model version.
        
        Args:
            model_name: Name of the registered model
            version: Model version
            
        Returns:
            Model metadata dictionary
        """
        try:
            # Get model version info
            model_version = self.client.get_model_version(model_name, version)
            
            # Get run info
            run = self.client.get_run(model_version.run_id)
            
            # Get artifacts
            artifacts = self.client.list_artifacts(model_version.run_id)
            
            metadata = {
                'model_info': {
                    'name': model_name,
                    'version': version,
                    'stage': model_version.current_stage,
                    'description': model_version.description,
                    'tags': dict(model_version.tags) if model_version.tags else {},
                    'creation_timestamp': model_version.creation_timestamp,
                    'last_updated_timestamp': model_version.last_updated_timestamp
                },
                'run_info': {
                    'run_id': run.info.run_id,
                    'experiment_id': run.info.experiment_id,
                    'status': run.info.status,
                    'start_time': run.info.start_time,
                    'end_time': run.info.end_time
                },
                'metrics': dict(run.data.metrics),
                'params': dict(run.data.params),
                'tags': dict(run.data.tags),
                'artifacts': [artifact.path for artifact in artifacts]
            }
            
            return metadata
            
        except Exception as e:
            logger.error(f"Failed to get model metadata: {e}")
            return {}


class ModelPromotionWorkflow:
    """Automated model promotion workflow."""
    
    def __init__(self, registry: MLflowModelRegistry,
                 criteria: Optional[PromotionCriteria] = None):
        """Initialize promotion workflow.
        
        Args:
            registry: MLflow model registry
            criteria: Promotion criteria
        """
        self.registry = registry
        self.criteria = criteria or PromotionCriteria()
        
        logger.info("Model promotion workflow initialized")
    
    def evaluate_for_promotion(self, model_name: str, version: str) -> Dict[str, Any]:
        """Evaluate a model version for promotion.
        
        Args:
            model_name: Name of the registered model
            version: Version to evaluate
            
        Returns:
            Evaluation results
        """
        try:
            # Get model version
            versions = self.registry.get_model_versions(model_name)
            target_version = next((v for v in versions if v.version == version), None)
            
            if not target_version:
                return {'eligible': False, 'reason': 'Version not found'}
            
            # Check metrics criteria
            metrics = target_version.metrics
            
            checks = {
                'accuracy': metrics.get('test_accuracy', 0.0) >= self.criteria.min_accuracy,
                'precision': metrics.get('test_precision', 0.0) >= self.criteria.min_precision,
                'recall': metrics.get('test_recall', 0.0) >= self.criteria.min_recall,
                'f1_score': metrics.get('test_f1_score', 0.0) >= self.criteria.min_f1_score
            }
            
            # Overall eligibility
            eligible = all(checks.values())
            
            # Generate recommendation
            if eligible:
                recommendation = f"Model {model_name} v{version} meets all criteria for promotion"
            else:
                failed_checks = [k for k, v in checks.items() if not v]
                recommendation = f"Model fails criteria: {', '.join(failed_checks)}"
            
            return {
                'eligible': eligible,
                'checks': checks,
                'metrics': metrics,
                'criteria': {
                    'min_accuracy': self.criteria.min_accuracy,
                    'min_precision': self.criteria.min_precision,
                    'min_recall': self.criteria.min_recall,
                    'min_f1_score': self.criteria.min_f1_score
                },
                'recommendation': recommendation
            }
            
        except Exception as e:
            logger.error(f"Failed to evaluate model for promotion: {e}")
            return {'eligible': False, 'reason': str(e)}
    
    def auto_promote(self, model_name: str, version: str,
                    target_stage: str = 'Staging') -> bool:
        """Automatically promote model if criteria are met.
        
        Args:
            model_name: Name of the registered model
            version: Version to promote
            target_stage: Target stage for promotion
            
        Returns:
            True if promoted successfully
        """
        # Evaluate for promotion
        evaluation = self.evaluate_for_promotion(model_name, version)
        
        if not evaluation['eligible']:
            logger.info(f"Model {model_name} v{version} not eligible for promotion: "
                       f"{evaluation.get('reason', 'Criteria not met')}")
            return False
        
        # Check if approval is required
        if self.criteria.require_approval and target_stage == 'Production':
            logger.info(f"Manual approval required for production promotion of "
                       f"{model_name} v{version}")
            return False
        
        # Promote model
        success = self.registry.promote_model(
            model_name=model_name,
            version=version,
            stage=target_stage,
            archive_existing=True
        )
        
        if success:
            logger.info(f"Successfully auto-promoted {model_name} v{version} to {target_stage}")
        
        return success
    
    def get_promotion_candidates(self, model_name: str) -> List[Dict[str, Any]]:
        """Get models that are candidates for promotion.
        
        Args:
            model_name: Name of the registered model
            
        Returns:
            List of promotion candidates
        """
        versions = self.registry.get_model_versions(model_name)
        candidates = []
        
        for version in versions:
            if version.stage in ['None', 'Staging']:
                evaluation = self.evaluate_for_promotion(model_name, version.version)
                if evaluation['eligible']:
                    candidates.append({
                        'version': version.version,
                        'current_stage': version.stage,
                        'metrics': version.metrics,
                        'evaluation': evaluation
                    })
        
        return sorted(candidates, key=lambda x: x['metrics'].get('test_accuracy', 0.0), reverse=True)


class ModelLifecycleManager:
    """Manages the complete model lifecycle."""
    
    def __init__(self, registry: MLflowModelRegistry):
        """Initialize lifecycle manager.
        
        Args:
            registry: MLflow model registry
        """
        self.registry = registry
        self.promotion_workflow = ModelPromotionWorkflow(registry)
        
        logger.info("Model lifecycle manager initialized")
    
    def deploy_model_pipeline(self, training_result: TrainingResult,
                            model_name: str,
                            auto_promote: bool = True) -> Dict[str, Any]:
        """Deploy a complete model pipeline from training to production.
        
        Args:
            training_result: Training result
            model_name: Name for the registered model
            auto_promote: Whether to auto-promote if criteria are met
            
        Returns:
            Deployment results
        """
        logger.info(f"Deploying model pipeline: {model_name}")
        
        results = {
            'model_name': model_name,
            'registration': None,
            'promotion': None,
            'status': 'failed'
        }
        
        try:
            # Register model
            model_version = self.registry.register_model(
                model=training_result.model,
                model_name=model_name,
                training_result=training_result,
                description=f"Model trained on {training_result.created_at.isoformat()}"
            )
            
            results['registration'] = {
                'version': model_version.version,
                'stage': model_version.stage,
                'metrics': model_version.metrics
            }
            
            # Auto-promote if enabled
            if auto_promote:
                promotion_success = self.promotion_workflow.auto_promote(
                    model_name=model_name,
                    version=model_version.version,
                    target_stage='Staging'
                )
                
                results['promotion'] = {
                    'attempted': True,
                    'successful': promotion_success,
                    'target_stage': 'Staging'
                }
                
                # If staging promotion successful, try production
                if promotion_success:
                    prod_promotion = self.promotion_workflow.auto_promote(
                        model_name=model_name,
                        version=model_version.version,
                        target_stage='Production'
                    )
                    
                    if prod_promotion:
                        results['promotion']['target_stage'] = 'Production'
            
            results['status'] = 'success'
            logger.info(f"Model pipeline deployment completed: {model_name}")
            
        except Exception as e:
            logger.error(f"Model pipeline deployment failed: {e}")
            results['error'] = str(e)
        
        return results
    
    def cleanup_old_versions(self, model_name: str, keep_versions: int = 5) -> int:
        """Clean up old model versions.
        
        Args:
            model_name: Name of the registered model
            keep_versions: Number of versions to keep
            
        Returns:
            Number of versions deleted
        """
        versions = self.registry.get_model_versions(model_name)
        
        # Keep production and staging models
        protected_versions = [v for v in versions if v.stage in ['Production', 'Staging']]
        
        # Get versions to potentially delete (excluding protected ones)
        deletable_versions = [v for v in versions if v.stage not in ['Production', 'Staging']]
        
        # Sort by creation time (newest first) and keep only the specified number
        deletable_versions.sort(key=lambda x: x.creation_timestamp, reverse=True)
        
        versions_to_delete = deletable_versions[keep_versions:]
        
        deleted_count = 0
        for version in versions_to_delete:
            if self.registry.delete_model_version(model_name, version.version):
                deleted_count += 1
        
        logger.info(f"Cleaned up {deleted_count} old versions of {model_name}")
        return deleted_count
    
    def get_model_health_report(self, model_name: str) -> Dict[str, Any]:
        """Generate a health report for a model.
        
        Args:
            model_name: Name of the registered model
            
        Returns:
            Model health report
        """
        versions = self.registry.get_model_versions(model_name)
        
        if not versions:
            return {'status': 'no_versions', 'message': 'No versions found'}
        
        # Get production model
        prod_version = next((v for v in versions if v.stage == 'Production'), None)
        staging_version = next((v for v in versions if v.stage == 'Staging'), None)
        
        # Calculate health metrics
        total_versions = len(versions)
        active_versions = len([v for v in versions if v.stage in ['Production', 'Staging']])
        
        # Get latest metrics
        latest_version = versions[0]  # Sorted by version desc
        latest_metrics = latest_version.metrics
        
        report = {
            'model_name': model_name,
            'total_versions': total_versions,
            'active_versions': active_versions,
            'production_version': prod_version.version if prod_version else None,
            'staging_version': staging_version.version if staging_version else None,
            'latest_version': latest_version.version,
            'latest_metrics': latest_metrics,
            'health_status': 'healthy' if prod_version else 'no_production_model',
            'recommendations': []
        }
        
        # Generate recommendations
        if not prod_version:
            report['recommendations'].append("No production model deployed")
        
        if not staging_version:
            report['recommendations'].append("No staging model for testing")
        
        if total_versions > 10:
            report['recommendations'].append("Consider cleaning up old model versions")
        
        return report