"""Automated model selection and comparison for pharmacogenomics classification."""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
import json
import pickle
from pathlib import Path
from abc import ABC, abstractmethod

# ML imports
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, classification_report, confusion_matrix
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline

# Model interpretability
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# Advanced models
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

# Transformer models
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from ..config import get_logger
from .feature_extraction import FeatureSet


logger = get_logger(__name__)


@dataclass
class ModelPerformance:
    """Model performance metrics."""
    model_name: str
    algorithm: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    cv_scores: List[float]
    cv_mean: float
    cv_std: float
    training_time: float
    prediction_time: float
    model_size_mb: float
    feature_importance: Optional[Dict[str, float]] = None
    confusion_matrix: Optional[np.ndarray] = None
    classification_report: Optional[str] = None


@dataclass
class ModelConfig:
    """Configuration for model training."""
    test_size: float = 0.2
    cv_folds: int = 5
    random_state: int = 42
    scoring: str = 'accuracy'
    enable_feature_selection: bool = True
    feature_selection_k: int = 100
    enable_scaling: bool = True
    enable_interpretability: bool = True
    save_models: bool = True
    model_dir: str = 'models/selected'


class BaseModelWrapper(ABC):
    """Abstract base class for model wrappers."""
    
    def __init__(self, name: str, **kwargs):
        """Initialize model wrapper.
        
        Args:
            name: Model name
            **kwargs: Model parameters
        """
        self.name = name
        self.params = kwargs
        self.model = None
        self.is_fitted = False
        self.feature_names = None
    
    @abstractmethod
    def create_model(self) -> Any:
        """Create model instance."""
        pass
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit model to data."""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        pass
    
    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        pass
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance if available."""
        return None
    
    def get_model_size(self) -> float:
        """Get model size in MB."""
        if self.model is None:
            return 0.0
        
        try:
            import sys
            return sys.getsizeof(pickle.dumps(self.model)) / (1024 * 1024)
        except:
            return 0.0


class RandomForestWrapper(BaseModelWrapper):
    """Random Forest model wrapper."""
    
    def create_model(self) -> RandomForestClassifier:
        """Create Random Forest model."""
        default_params = {
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'random_state': 42
        }
        default_params.update(self.params)
        return RandomForestClassifier(**default_params)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit Random Forest model."""
        self.model = self.create_model()
        self.model.fit(X, y)
        self.is_fitted = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities."""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        return self.model.predict_proba(X)
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance."""
        if not self.is_fitted or self.feature_names is None:
            return None
        
        importance = self.model.feature_importances_
        return dict(zip(self.feature_names, importance))


class XGBoostWrapper(BaseModelWrapper):
    """XGBoost model wrapper."""
    
    def create_model(self):
        """Create XGBoost model."""
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not available")
        
        default_params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'random_state': 42
        }
        default_params.update(self.params)
        return xgb.XGBClassifier(**default_params)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit XGBoost model."""
        self.model = self.create_model()
        self.model.fit(X, y)
        self.is_fitted = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities."""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        return self.model.predict_proba(X)
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance."""
        if not self.is_fitted or self.feature_names is None:
            return None
        
        importance = self.model.feature_importances_
        return dict(zip(self.feature_names, importance))


class LightGBMWrapper(BaseModelWrapper):
    """LightGBM model wrapper."""
    
    def create_model(self):
        """Create LightGBM model."""
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM not available")
        
        default_params = {
            'n_estimators': 100,
            'max_depth': -1,
            'learning_rate': 0.1,
            'random_state': 42
        }
        default_params.update(self.params)
        return lgb.LGBMClassifier(**default_params)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit LightGBM model."""
        self.model = self.create_model()
        self.model.fit(X, y)
        self.is_fitted = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities."""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        return self.model.predict_proba(X)
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance."""
        if not self.is_fitted or self.feature_names is None:
            return None
        
        importance = self.model.feature_importances_
        return dict(zip(self.feature_names, importance))


class BERTWrapper(BaseModelWrapper):
    """BERT model wrapper for text classification."""
    
    def __init__(self, name: str, model_name: str = "allenai/scibert-scivocab-uncased", **kwargs):
        """Initialize BERT wrapper.
        
        Args:
            name: Model name
            model_name: Pretrained model name
            **kwargs: Additional parameters
        """
        super().__init__(name, **kwargs)
        self.model_name = model_name
        self.tokenizer = None
        self.text_data = None
    
    def create_model(self):
        """Create BERT model."""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers not available")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=2  # Binary classification
        )
        return model
    
    def fit(self, X: np.ndarray, y: np.ndarray, text_data: List[str] = None) -> None:
        """Fit BERT model."""
        if text_data is None:
            raise ValueError("Text data required for BERT training")
        
        self.text_data = text_data
        self.model = self.create_model()
        
        # This is a simplified implementation
        # In practice, you would use Trainer with proper dataset handling
        logger.warning("BERT training not fully implemented - using placeholder")
        self.is_fitted = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        # Placeholder implementation
        return np.random.randint(0, 2, size=X.shape[0])
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities."""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        # Placeholder implementation
        probs = np.random.rand(X.shape[0], 2)
        return probs / probs.sum(axis=1, keepdims=True)


class EnsembleWrapper(BaseModelWrapper):
    """Ensemble model wrapper."""
    
    def __init__(self, name: str, base_models: List[BaseModelWrapper], 
                 ensemble_type: str = 'voting', **kwargs):
        """Initialize ensemble wrapper.
        
        Args:
            name: Model name
            base_models: List of base models
            ensemble_type: Type of ensemble ('voting', 'stacking')
            **kwargs: Additional parameters
        """
        super().__init__(name, **kwargs)
        self.base_models = base_models
        self.ensemble_type = ensemble_type
    
    def create_model(self):
        """Create ensemble model."""
        if self.ensemble_type == 'voting':
            estimators = [(model.name, model.create_model()) for model in self.base_models]
            return VotingClassifier(estimators=estimators, voting='soft')
        
        elif self.ensemble_type == 'stacking':
            estimators = [(model.name, model.create_model()) for model in self.base_models]
            return StackingClassifier(
                estimators=estimators,
                final_estimator=RandomForestClassifier(n_estimators=50, random_state=42)
            )
        
        else:
            raise ValueError(f"Unknown ensemble type: {self.ensemble_type}")
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit ensemble model."""
        self.model = self.create_model()
        self.model.fit(X, y)
        self.is_fitted = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities."""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        return self.model.predict_proba(X)


class ModelInterpretability:
    """Model interpretability analysis using SHAP."""
    
    def __init__(self):
        """Initialize interpretability analyzer."""
        self.explainer = None
        self.shap_values = None
    
    def analyze_model(self, model: BaseModelWrapper, X: np.ndarray, 
                     feature_names: List[str]) -> Dict[str, Any]:
        """Analyze model interpretability.
        
        Args:
            model: Trained model
            X: Feature matrix
            feature_names: Feature names
            
        Returns:
            Dict containing interpretability results
        """
        if not SHAP_AVAILABLE:
            logger.warning("SHAP not available for interpretability analysis")
            return {}
        
        try:
            # Create SHAP explainer
            if hasattr(model.model, 'predict_proba'):
                self.explainer = shap.Explainer(model.model.predict_proba, X)
            else:
                self.explainer = shap.Explainer(model.model.predict, X)
            
            # Calculate SHAP values
            self.shap_values = self.explainer(X[:100])  # Limit for performance
            
            # Feature importance
            feature_importance = np.abs(self.shap_values.values).mean(axis=0)
            
            if len(feature_importance.shape) > 1:
                feature_importance = feature_importance.mean(axis=1)
            
            importance_dict = dict(zip(feature_names, feature_importance))
            
            return {
                'feature_importance': importance_dict,
                'shap_values_available': True,
                'top_features': sorted(importance_dict.items(), 
                                     key=lambda x: x[1], reverse=True)[:10]
            }
            
        except Exception as e:
            logger.error(f"SHAP analysis failed: {e}")
            return {'error': str(e)}


class ModelSelector:
    """Automated model selection and comparison."""
    
    def __init__(self, config: Optional[ModelConfig] = None):
        """Initialize model selector.
        
        Args:
            config: Model configuration
        """
        self.config = config or ModelConfig()
        self.models = {}
        self.results = {}
        self.best_model = None
        self.interpretability = ModelInterpretability()
        
        # Create model directory
        Path(self.config.model_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info("ModelSelector initialized")
    
    def add_model(self, model: BaseModelWrapper) -> None:
        """Add model to selection pool.
        
        Args:
            model: Model wrapper to add
        """
        self.models[model.name] = model
        logger.info(f"Added model: {model.name}")
    
    def add_default_models(self) -> None:
        """Add default set of models."""
        # Random Forest
        self.add_model(RandomForestWrapper(
            'random_forest',
            n_estimators=100,
            max_depth=10,
            random_state=self.config.random_state
        ))
        
        # XGBoost if available
        if XGBOOST_AVAILABLE:
            self.add_model(XGBoostWrapper(
                'xgboost',
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=self.config.random_state
            ))
        
        # LightGBM if available
        if LIGHTGBM_AVAILABLE:
            self.add_model(LightGBMWrapper(
                'lightgbm',
                n_estimators=100,
                max_depth=-1,
                learning_rate=0.1,
                random_state=self.config.random_state
            ))
        
        # Ensemble
        base_models = [
            RandomForestWrapper('rf_base', n_estimators=50, random_state=self.config.random_state)
        ]
        
        if XGBOOST_AVAILABLE:
            base_models.append(XGBoostWrapper('xgb_base', n_estimators=50, random_state=self.config.random_state))
        
        if len(base_models) > 1:
            self.add_model(EnsembleWrapper(
                'ensemble_voting',
                base_models=base_models,
                ensemble_type='voting'
            ))
    
    def evaluate_models(self, feature_set: FeatureSet, 
                       target: np.ndarray,
                       text_data: Optional[List[str]] = None) -> Dict[str, ModelPerformance]:
        """Evaluate all models.
        
        Args:
            feature_set: Feature set for training
            target: Target variable
            text_data: Text data for BERT models
            
        Returns:
            Dict of model performances
        """
        logger.info(f"Evaluating {len(self.models)} models")
        
        # Prepare data
        X = self._prepare_features(feature_set)
        y = target
        
        # Feature names
        feature_names = feature_set.feature_names
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config.test_size, 
            random_state=self.config.random_state,
            stratify=y
        )
        
        # Scale features if enabled
        if self.config.enable_scaling:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
        
        results = {}
        
        for model_name, model in self.models.items():
            logger.info(f"Evaluating {model_name}...")
            
            try:
                performance = self._evaluate_single_model(
                    model, X_train, X_test, y_train, y_test, 
                    feature_names, text_data
                )
                results[model_name] = performance
                
            except Exception as e:
                logger.error(f"Failed to evaluate {model_name}: {e}")
                continue
        
        self.results = results
        
        # Find best model
        if results:
            best_model_name = max(results.keys(), 
                                key=lambda k: results[k].accuracy)
            self.best_model = best_model_name
            
            logger.info(f"Best model: {best_model_name} "
                       f"(accuracy: {results[best_model_name].accuracy:.4f})")
        
        return results
    
    def _evaluate_single_model(self, model: BaseModelWrapper,
                             X_train: np.ndarray, X_test: np.ndarray,
                             y_train: np.ndarray, y_test: np.ndarray,
                             feature_names: List[str],
                             text_data: Optional[List[str]] = None) -> ModelPerformance:
        """Evaluate a single model."""
        import time
        
        # Set feature names
        model.feature_names = feature_names
        
        # Training
        start_time = time.time()
        
        if isinstance(model, BERTWrapper) and text_data:
            model.fit(X_train, y_train, text_data)
        else:
            model.fit(X_train, y_train)
        
        training_time = time.time() - start_time
        
        # Prediction
        start_time = time.time()
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        prediction_time = time.time() - start_time
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # ROC AUC for binary classification
        roc_auc = 0.0
        if len(np.unique(y_test)) == 2 and y_pred_proba.shape[1] == 2:
            try:
                roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
            except:
                pass
        
        # Cross-validation
        cv = StratifiedKFold(n_splits=self.config.cv_folds, shuffle=True, 
                           random_state=self.config.random_state)
        
        # Create fresh model for CV
        cv_model = model.create_model()
        cv_scores = cross_val_score(cv_model, X_train, y_train, cv=cv, 
                                  scoring=self.config.scoring)
        
        # Feature importance
        feature_importance = model.get_feature_importance()
        
        # Model interpretability
        if self.config.enable_interpretability:
            try:
                interp_results = self.interpretability.analyze_model(
                    model, X_train[:100], feature_names
                )
                if 'feature_importance' in interp_results:
                    feature_importance = interp_results['feature_importance']
            except Exception as e:
                logger.warning(f"Interpretability analysis failed for {model.name}: {e}")
        
        # Model size
        model_size = model.get_model_size()
        
        # Classification report
        class_report = classification_report(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        return ModelPerformance(
            model_name=model.name,
            algorithm=model.__class__.__name__,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            roc_auc=roc_auc,
            cv_scores=cv_scores.tolist(),
            cv_mean=float(np.mean(cv_scores)),
            cv_std=float(np.std(cv_scores)),
            training_time=training_time,
            prediction_time=prediction_time,
            model_size_mb=model_size,
            feature_importance=feature_importance,
            confusion_matrix=conf_matrix,
            classification_report=class_report
        )
    
    def _prepare_features(self, feature_set: FeatureSet) -> np.ndarray:
        """Prepare features for training."""
        feature_matrix = []
        
        for feature_name in feature_set.feature_names:
            if feature_name in feature_set.features:
                feature_values = feature_set.features[feature_name]
                if feature_values.ndim == 1:
                    feature_matrix.append(feature_values.reshape(-1, 1))
                else:
                    feature_matrix.append(feature_values)
        
        if not feature_matrix:
            raise ValueError("No features available for training")
        
        return np.hstack(feature_matrix)
    
    def get_model_ranking(self, metric: str = 'accuracy') -> List[Tuple[str, float]]:
        """Get model ranking by metric.
        
        Args:
            metric: Metric to rank by
            
        Returns:
            List of (model_name, metric_value) tuples
        """
        if not self.results:
            return []
        
        ranking = []
        for model_name, performance in self.results.items():
            if hasattr(performance, metric):
                value = getattr(performance, metric)
                ranking.append((model_name, value))
        
        return sorted(ranking, key=lambda x: x[1], reverse=True)
    
    def get_performance_summary(self) -> pd.DataFrame:
        """Get performance summary as DataFrame."""
        if not self.results:
            return pd.DataFrame()
        
        data = []
        for model_name, performance in self.results.items():
            data.append({
                'model': model_name,
                'algorithm': performance.algorithm,
                'accuracy': performance.accuracy,
                'precision': performance.precision,
                'recall': performance.recall,
                'f1_score': performance.f1_score,
                'roc_auc': performance.roc_auc,
                'cv_mean': performance.cv_mean,
                'cv_std': performance.cv_std,
                'training_time': performance.training_time,
                'prediction_time': performance.prediction_time,
                'model_size_mb': performance.model_size_mb
            })
        
        return pd.DataFrame(data)
    
    def save_results(self, filepath: str) -> None:
        """Save evaluation results.
        
        Args:
            filepath: Path to save results
        """
        # Convert results to serializable format
        serializable_results = {}
        
        for model_name, performance in self.results.items():
            serializable_results[model_name] = {
                'model_name': performance.model_name,
                'algorithm': performance.algorithm,
                'accuracy': performance.accuracy,
                'precision': performance.precision,
                'recall': performance.recall,
                'f1_score': performance.f1_score,
                'roc_auc': performance.roc_auc,
                'cv_scores': performance.cv_scores,
                'cv_mean': performance.cv_mean,
                'cv_std': performance.cv_std,
                'training_time': performance.training_time,
                'prediction_time': performance.prediction_time,
                'model_size_mb': performance.model_size_mb,
                'feature_importance': performance.feature_importance,
                'classification_report': performance.classification_report
            }
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {filepath}")
    
    def get_best_model(self) -> Optional[str]:
        """Get name of best performing model."""
        return self.best_model
    
    def get_feature_importance_summary(self) -> Dict[str, Dict[str, float]]:
        """Get feature importance summary across all models."""
        importance_summary = {}
        
        for model_name, performance in self.results.items():
            if performance.feature_importance:
                importance_summary[model_name] = performance.feature_importance
        
        return importance_summary