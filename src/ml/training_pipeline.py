"""End-to-end training pipeline orchestration with cross-validation and metrics logging."""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime
import json
import pickle
import os
from pathlib import Path
import logging
import time

# ML imports
from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold, TimeSeriesSplit
)
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, classification_report, confusion_matrix
)
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif

# Data augmentation
try:
    from imblearn.over_sampling import SMOTE, ADASYN
    from imblearn.under_sampling import RandomUnderSampler
    IMBALANCED_LEARN_AVAILABLE = True
except ImportError:
    IMBALANCED_LEARN_AVAILABLE = False

from ..config import get_logger
from .feature_extraction import FeatureSet
from .model_selector import BaseModelWrapper, ModelSelector
from .hyperopt import HyperparameterOptimizer, OptimizationObjective


logger = get_logger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for training pipeline."""
    # Data splitting
    test_size: float = 0.2
    validation_size: float = 0.2
    random_state: int = 42
    
    # Cross-validation
    cv_strategy: str = 'stratified'  # 'stratified', 'time_series', 'group'
    cv_folds: int = 5
    
    # Preprocessing
    enable_scaling: bool = True
    scaler_type: str = 'standard'  # 'standard', 'robust', 'minmax'
    enable_feature_selection: bool = True
    feature_selection_k: int = 100
    feature_selection_method: str = 'f_classif'  # 'f_classif', 'mutual_info'
    
    # Data augmentation
    enable_augmentation: bool = True
    augmentation_method: str = 'smote'  # 'smote', 'adasyn', 'undersample'
    augmentation_ratio: float = 1.0
    
    # Training
    enable_hyperopt: bool = True
    hyperopt_trials: int = 50
    hyperopt_timeout: Optional[int] = 1800  # 30 minutes
    
    # Validation
    enable_early_stopping: bool = True
    patience: int = 10
    min_improvement: float = 0.001
    
    # Logging and persistence
    save_models: bool = True
    save_predictions: bool = True
    model_dir: str = 'models/training'
    log_level: str = 'INFO'


@dataclass
class TrainingMetrics:
    """Training metrics and results."""
    model_name: str
    training_time: float
    validation_scores: Dict[str, float]
    test_scores: Dict[str, float]
    cv_scores: Dict[str, List[float]]
    feature_importance: Optional[Dict[str, float]] = None
    confusion_matrix: Optional[np.ndarray] = None
    classification_report: Optional[str] = None
    hyperopt_results: Optional[Dict[str, Any]] = None
    data_stats: Optional[Dict[str, Any]] = None


@dataclass
class TrainingResult:
    """Complete training result."""
    pipeline_id: str
    config: TrainingConfig
    model: BaseModelWrapper
    metrics: TrainingMetrics
    preprocessors: Dict[str, Any]
    feature_names: List[str]
    target_encoder: Optional[LabelEncoder] = None
    created_at: datetime = field(default_factory=datetime.now)


class DataPreprocessor:
    """Data preprocessing pipeline."""
    
    def __init__(self, config: TrainingConfig):
        """Initialize data preprocessor.
        
        Args:
            config: Training configuration
        """
        self.config = config
        self.scaler = None
        self.feature_selector = None
        self.target_encoder = None
        self.augmenter = None
        
        self._setup_preprocessors()
    
    def _setup_preprocessors(self):
        """Setup preprocessing components."""
        # Scaler
        if self.config.enable_scaling:
            if self.config.scaler_type == 'standard':
                self.scaler = StandardScaler()
            elif self.config.scaler_type == 'robust':
                self.scaler = RobustScaler()
            else:
                self.scaler = StandardScaler()
        
        # Feature selector
        if self.config.enable_feature_selection:
            if self.config.feature_selection_method == 'f_classif':
                score_func = f_classif
            elif self.config.feature_selection_method == 'mutual_info':
                score_func = mutual_info_classif
            else:
                score_func = f_classif
            
            self.feature_selector = SelectKBest(
                score_func=score_func,
                k=min(self.config.feature_selection_k, 1000)  # Reasonable limit
            )
        
        # Target encoder
        self.target_encoder = LabelEncoder()
        
        # Data augmenter
        if self.config.enable_augmentation and IMBALANCED_LEARN_AVAILABLE:
            if self.config.augmentation_method == 'smote':
                self.augmenter = SMOTE(
                    sampling_strategy=self.config.augmentation_ratio,
                    random_state=self.config.random_state
                )
            elif self.config.augmentation_method == 'adasyn':
                self.augmenter = ADASYN(
                    sampling_strategy=self.config.augmentation_ratio,
                    random_state=self.config.random_state
                )
            elif self.config.augmentation_method == 'undersample':
                self.augmenter = RandomUnderSampler(
                    sampling_strategy=self.config.augmentation_ratio,
                    random_state=self.config.random_state
                )
    
    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Fit preprocessors and transform data.
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            Transformed features and targets
        """
        logger.info("Fitting and transforming training data")
        
        # Encode targets
        y_encoded = self.target_encoder.fit_transform(y)
        
        # Scale features
        if self.scaler:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = X.copy()
        
        # Feature selection
        if self.feature_selector:
            # Adjust k if necessary
            n_features = X_scaled.shape[1]
            k = min(self.config.feature_selection_k, n_features)
            self.feature_selector.k = k
            
            X_selected = self.feature_selector.fit_transform(X_scaled, y_encoded)
        else:
            X_selected = X_scaled
        
        # Data augmentation
        if self.augmenter:
            try:
                X_augmented, y_augmented = self.augmenter.fit_resample(X_selected, y_encoded)
                logger.info(f"Data augmentation: {X_selected.shape[0]} -> {X_augmented.shape[0]} samples")
            except Exception as e:
                logger.warning(f"Data augmentation failed: {e}")
                X_augmented, y_augmented = X_selected, y_encoded
        else:
            X_augmented, y_augmented = X_selected, y_encoded
        
        return X_augmented, y_augmented
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform new data using fitted preprocessors.
        
        Args:
            X: Feature matrix
            
        Returns:
            Transformed features
        """
        # Scale features
        if self.scaler:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X.copy()
        
        # Feature selection
        if self.feature_selector:
            X_selected = self.feature_selector.transform(X_scaled)
        else:
            X_selected = X_scaled
        
        return X_selected
    
    def get_selected_features(self, feature_names: List[str]) -> List[str]:
        """Get names of selected features.
        
        Args:
            feature_names: Original feature names
            
        Returns:
            Selected feature names
        """
        if not self.feature_selector:
            return feature_names
        
        selected_indices = self.feature_selector.get_support(indices=True)
        return [feature_names[i] for i in selected_indices if i < len(feature_names)]


class CrossValidator:
    """Cross-validation with different strategies."""
    
    def __init__(self, config: TrainingConfig):
        """Initialize cross-validator.
        
        Args:
            config: Training configuration
        """
        self.config = config
    
    def create_cv_splitter(self, X: np.ndarray, y: np.ndarray):
        """Create cross-validation splitter.
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            CV splitter object
        """
        if self.config.cv_strategy == 'stratified':
            return StratifiedKFold(
                n_splits=self.config.cv_folds,
                shuffle=True,
                random_state=self.config.random_state
            )
        
        elif self.config.cv_strategy == 'time_series':
            return TimeSeriesSplit(n_splits=self.config.cv_folds)
        
        else:
            # Default to stratified
            return StratifiedKFold(
                n_splits=self.config.cv_folds,
                shuffle=True,
                random_state=self.config.random_state
            )
    
    def cross_validate_model(self, model: BaseModelWrapper, 
                           X: np.ndarray, y: np.ndarray) -> Dict[str, List[float]]:
        """Perform cross-validation.
        
        Args:
            model: Model to validate
            X: Feature matrix
            y: Target vector
            
        Returns:
            Dict of CV scores for different metrics
        """
        cv_splitter = self.create_cv_splitter(X, y)
        
        # Create fresh model instance for CV
        cv_model = model.create_model()
        
        # Calculate different metrics
        cv_scores = {}
        
        # Accuracy
        scores = cross_val_score(cv_model, X, y, cv=cv_splitter, scoring='accuracy')
        cv_scores['accuracy'] = scores.tolist()
        
        # Precision, Recall, F1 (weighted for multi-class)
        for metric in ['precision_weighted', 'recall_weighted', 'f1_weighted']:
            scores = cross_val_score(cv_model, X, y, cv=cv_splitter, scoring=metric)
            cv_scores[metric] = scores.tolist()
        
        # ROC AUC for binary classification
        if len(np.unique(y)) == 2:
            try:
                scores = cross_val_score(cv_model, X, y, cv=cv_splitter, scoring='roc_auc')
                cv_scores['roc_auc'] = scores.tolist()
            except:
                cv_scores['roc_auc'] = [0.0] * self.config.cv_folds
        
        return cv_scores


class TrainingPipeline:
    """End-to-end training pipeline orchestrator."""
    
    def __init__(self, config: Optional[TrainingConfig] = None):
        """Initialize training pipeline.
        
        Args:
            config: Training configuration
        """
        self.config = config or TrainingConfig()
        
        # Initialize components
        self.preprocessor = DataPreprocessor(self.config)
        self.cross_validator = CrossValidator(self.config)
        
        if self.config.enable_hyperopt:
            self.hyperopt = HyperparameterOptimizer(
                random_state=self.config.random_state
            )
        else:
            self.hyperopt = None
        
        # Create model directory
        Path(self.config.model_dir).mkdir(parents=True, exist_ok=True)
        
        # Training history
        self.training_history = []
        
        logger.info("TrainingPipeline initialized")
    
    def train_model(self, model: BaseModelWrapper,
                   feature_set: FeatureSet,
                   target: np.ndarray,
                   pipeline_id: Optional[str] = None) -> TrainingResult:
        """Train a single model.
        
        Args:
            model: Model to train
            feature_set: Feature set
            target: Target variable
            pipeline_id: Optional pipeline ID
            
        Returns:
            Training result
        """
        if pipeline_id is None:
            pipeline_id = f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"Starting training pipeline: {pipeline_id}")
        start_time = time.time()
        
        # Prepare data
        X = self._prepare_features(feature_set)
        y = target
        
        # Data statistics
        data_stats = self._calculate_data_stats(X, y, feature_set.feature_names)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y
        )
        
        # Further split training data for validation
        X_train_final, X_val, y_train_final, y_val = train_test_split(
            X_train, y_train,
            test_size=self.config.validation_size,
            random_state=self.config.random_state,
            stratify=y_train
        )
        
        # Preprocess data
        X_train_processed, y_train_processed = self.preprocessor.fit_transform(
            X_train_final, y_train_final
        )
        X_val_processed = self.preprocessor.transform(X_val)
        X_test_processed = self.preprocessor.transform(X_test)
        
        # Get selected feature names
        selected_features = self.preprocessor.get_selected_features(feature_set.feature_names)
        model.feature_names = selected_features
        
        # Hyperparameter optimization
        hyperopt_results = None
        if self.config.enable_hyperopt and self.hyperopt:
            logger.info("Running hyperparameter optimization...")
            
            # Create temporary feature set for hyperopt
            temp_feature_set = FeatureSet(
                features={name: X_train_processed[:, i].reshape(-1) 
                         for i, name in enumerate(selected_features)},
                metadata={},
                feature_names=selected_features,
                sample_ids=[f"sample_{i}" for i in range(len(X_train_processed))],
                extraction_timestamp=datetime.now(),
                version_hash=""
            )
            
            # Define optimization objective
            objective = OptimizationObjective(
                name='accuracy',
                direction='maximize'
            )
            
            # Run optimization
            hyperopt_result = self.hyperopt.optimize_single_objective(
                model_wrapper=model,
                feature_set=temp_feature_set,
                target=y_train_processed,
                objective=objective,
                n_trials=self.config.hyperopt_trials,
                timeout=self.config.hyperopt_timeout,
                study_name=f"{pipeline_id}_hyperopt"
            )
            
            # Update model parameters
            model.params.update(hyperopt_result.best_parameters)
            hyperopt_results = {
                'best_parameters': hyperopt_result.best_parameters,
                'best_score': hyperopt_result.best_values,
                'n_trials': hyperopt_result.n_trials,
                'optimization_time': hyperopt_result.optimization_time
            }
            
            logger.info(f"Hyperparameter optimization completed: "
                       f"best accuracy = {hyperopt_result.best_values['accuracy']:.4f}")
        
        # Train final model
        logger.info("Training final model...")
        model.fit(X_train_processed, y_train_processed)
        
        # Cross-validation
        logger.info("Performing cross-validation...")
        cv_scores = self.cross_validator.cross_validate_model(
            model, X_train_processed, y_train_processed
        )
        
        # Validation predictions
        y_val_pred = model.predict(X_val_processed)
        y_val_pred_proba = model.predict_proba(X_val_processed)
        
        # Test predictions
        y_test_pred = model.predict(X_test_processed)
        y_test_pred_proba = model.predict_proba(X_test_processed)
        
        # Calculate metrics
        validation_scores = self._calculate_metrics(y_val, y_val_pred, y_val_pred_proba)
        test_scores = self._calculate_metrics(y_test, y_test_pred, y_test_pred_proba)
        
        # Feature importance
        feature_importance = model.get_feature_importance()
        
        # Classification report and confusion matrix
        class_report = classification_report(y_test, y_test_pred)
        conf_matrix = confusion_matrix(y_test, y_test_pred)
        
        # Training time
        training_time = time.time() - start_time
        
        # Create metrics object
        metrics = TrainingMetrics(
            model_name=model.name,
            training_time=training_time,
            validation_scores=validation_scores,
            test_scores=test_scores,
            cv_scores=cv_scores,
            feature_importance=feature_importance,
            confusion_matrix=conf_matrix,
            classification_report=class_report,
            hyperopt_results=hyperopt_results,
            data_stats=data_stats
        )
        
        # Create result
        result = TrainingResult(
            pipeline_id=pipeline_id,
            config=self.config,
            model=model,
            metrics=metrics,
            preprocessors={
                'scaler': self.preprocessor.scaler,
                'feature_selector': self.preprocessor.feature_selector,
                'augmenter': self.preprocessor.augmenter
            },
            feature_names=selected_features,
            target_encoder=self.preprocessor.target_encoder
        )
        
        # Save results
        if self.config.save_models:
            self._save_training_result(result)
        
        # Add to history
        self.training_history.append(result)
        
        logger.info(f"Training completed: {model.name}, "
                   f"test accuracy = {test_scores['accuracy']:.4f}, "
                   f"time = {training_time:.1f}s")
        
        return result
    
    def train_multiple_models(self, models: List[BaseModelWrapper],
                            feature_set: FeatureSet,
                            target: np.ndarray) -> List[TrainingResult]:
        """Train multiple models and compare results.
        
        Args:
            models: List of models to train
            feature_set: Feature set
            target: Target variable
            
        Returns:
            List of training results
        """
        logger.info(f"Training {len(models)} models")
        
        results = []
        for i, model in enumerate(models):
            logger.info(f"Training model {i+1}/{len(models)}: {model.name}")
            
            try:
                result = self.train_model(
                    model=model,
                    feature_set=feature_set,
                    target=target,
                    pipeline_id=f"multi_model_{i}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                )
                results.append(result)
                
            except Exception as e:
                logger.error(f"Failed to train {model.name}: {e}")
                continue
        
        # Sort by test accuracy
        results.sort(key=lambda r: r.metrics.test_scores['accuracy'], reverse=True)
        
        logger.info(f"Multi-model training completed. Best model: "
                   f"{results[0].model.name} (accuracy: {results[0].metrics.test_scores['accuracy']:.4f})")
        
        return results
    
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
    
    def _calculate_metrics(self, y_true: np.ndarray, 
                          y_pred: np.ndarray,
                          y_pred_proba: np.ndarray) -> Dict[str, float]:
        """Calculate evaluation metrics."""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
        
        # ROC AUC for binary classification
        if len(np.unique(y_true)) == 2 and y_pred_proba.shape[1] == 2:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
            except:
                metrics['roc_auc'] = 0.0
        else:
            metrics['roc_auc'] = 0.0
        
        return metrics
    
    def _calculate_data_stats(self, X: np.ndarray, y: np.ndarray, 
                            feature_names: List[str]) -> Dict[str, Any]:
        """Calculate data statistics."""
        return {
            'n_samples': X.shape[0],
            'n_features': X.shape[1],
            'n_classes': len(np.unique(y)),
            'class_distribution': dict(zip(*np.unique(y, return_counts=True))),
            'feature_stats': {
                'mean': np.mean(X, axis=0).tolist()[:10],  # First 10 features
                'std': np.std(X, axis=0).tolist()[:10],
                'min': np.min(X, axis=0).tolist()[:10],
                'max': np.max(X, axis=0).tolist()[:10]
            }
        }
    
    def _save_training_result(self, result: TrainingResult) -> None:
        """Save training result to disk."""
        # Create directory for this pipeline
        pipeline_dir = Path(self.config.model_dir) / result.pipeline_id
        pipeline_dir.mkdir(exist_ok=True)
        
        # Save model
        model_path = pipeline_dir / 'model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(result.model, f)
        
        # Save preprocessors
        preprocessor_path = pipeline_dir / 'preprocessors.pkl'
        with open(preprocessor_path, 'wb') as f:
            pickle.dump(result.preprocessors, f)
        
        # Save metadata
        metadata = {
            'pipeline_id': result.pipeline_id,
            'model_name': result.model.name,
            'feature_names': result.feature_names,
            'config': result.config.__dict__,
            'metrics': {
                'validation_scores': result.metrics.validation_scores,
                'test_scores': result.metrics.test_scores,
                'cv_scores': result.metrics.cv_scores,
                'training_time': result.metrics.training_time
            },
            'created_at': result.created_at.isoformat()
        }
        
        metadata_path = pipeline_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Training result saved to {pipeline_dir}")
    
    def load_training_result(self, pipeline_id: str) -> TrainingResult:
        """Load training result from disk.
        
        Args:
            pipeline_id: Pipeline ID to load
            
        Returns:
            Loaded training result
        """
        pipeline_dir = Path(self.config.model_dir) / pipeline_id
        
        if not pipeline_dir.exists():
            raise FileNotFoundError(f"Pipeline {pipeline_id} not found")
        
        # Load model
        model_path = pipeline_dir / 'model.pkl'
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Load preprocessors
        preprocessor_path = pipeline_dir / 'preprocessors.pkl'
        with open(preprocessor_path, 'rb') as f:
            preprocessors = pickle.load(f)
        
        # Load metadata
        metadata_path = pipeline_dir / 'metadata.json'
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Reconstruct result (simplified)
        result = TrainingResult(
            pipeline_id=pipeline_id,
            config=self.config,  # Use current config
            model=model,
            metrics=None,  # Would need to reconstruct
            preprocessors=preprocessors,
            feature_names=metadata['feature_names'],
            created_at=datetime.fromisoformat(metadata['created_at'])
        )
        
        logger.info(f"Training result loaded: {pipeline_id}")
        return result
    
    def get_training_summary(self) -> pd.DataFrame:
        """Get summary of all training runs."""
        if not self.training_history:
            return pd.DataFrame()
        
        data = []
        for result in self.training_history:
            data.append({
                'pipeline_id': result.pipeline_id,
                'model_name': result.model.name,
                'test_accuracy': result.metrics.test_scores['accuracy'],
                'test_f1': result.metrics.test_scores['f1_score'],
                'cv_accuracy_mean': np.mean(result.metrics.cv_scores['accuracy']),
                'cv_accuracy_std': np.std(result.metrics.cv_scores['accuracy']),
                'training_time': result.metrics.training_time,
                'n_features': len(result.feature_names),
                'created_at': result.created_at
            })
        
        return pd.DataFrame(data)
    
    def get_best_model(self) -> Optional[TrainingResult]:
        """Get best performing model from training history."""
        if not self.training_history:
            return None
        
        return max(self.training_history, 
                  key=lambda r: r.metrics.test_scores['accuracy'])