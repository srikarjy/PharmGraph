"""AutoML engine with Ray and Optuna integration for distributed hyperparameter optimization."""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Callable, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
import json
import pickle
import os
from pathlib import Path
import logging

# Ray imports
try:
    import ray
    from ray import tune
    from ray.tune import CLIReporter
    from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
    from ray.tune.suggest.optuna import OptunaSearch
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

# Optuna imports
try:
    import optuna
    from optuna.samplers import TPESampler, CmaEsSampler
    from optuna.pruners import MedianPruner, HyperbandPruner
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

# ML imports
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from ..config import get_logger
from .feature_extraction import FeatureSet


logger = get_logger(__name__)


@dataclass
class TrialResult:
    """Result of a single hyperparameter optimization trial."""
    trial_id: str
    parameters: Dict[str, Any]
    metrics: Dict[str, float]
    status: str  # 'completed', 'failed', 'pruned'
    duration: float
    model_path: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class AutoMLConfig:
    """Configuration for AutoML training."""
    # Ray configuration
    ray_address: Optional[str] = None
    num_cpus: int = 4
    num_gpus: int = 0
    
    # Optuna configuration
    n_trials: int = 100
    timeout: Optional[int] = 3600  # 1 hour
    sampler: str = 'tpe'  # 'tpe', 'cmaes', 'random'
    pruner: str = 'median'  # 'median', 'hyperband'
    
    # Training configuration
    cv_folds: int = 5
    test_size: float = 0.2
    random_state: int = 42
    
    # Resource allocation
    max_concurrent_trials: int = 4
    resources_per_trial: Dict[str, float] = field(default_factory=lambda: {'cpu': 1, 'gpu': 0})
    
    # Early stopping
    enable_early_stopping: bool = True
    patience: int = 10
    min_improvement: float = 0.001
    
    # Model persistence
    save_models: bool = True
    model_dir: str = 'models/automl'


class HyperparameterSpace:
    """Defines hyperparameter search spaces for different algorithms."""
    
    @staticmethod
    def get_random_forest_space() -> Dict[str, Any]:
        """Get Random Forest hyperparameter space."""
        return {
            'n_estimators': ('int', 10, 500),
            'max_depth': ('int', 3, 20),
            'min_samples_split': ('int', 2, 20),
            'min_samples_leaf': ('int', 1, 10),
            'max_features': ('categorical', ['sqrt', 'log2', None]),
            'bootstrap': ('categorical', [True, False]),
            'class_weight': ('categorical', ['balanced', None])
        }
    
    @staticmethod
    def get_xgboost_space() -> Dict[str, Any]:
        """Get XGBoost hyperparameter space."""
        return {
            'n_estimators': ('int', 50, 500),
            'max_depth': ('int', 3, 10),
            'learning_rate': ('float', 0.01, 0.3),
            'subsample': ('float', 0.6, 1.0),
            'colsample_bytree': ('float', 0.6, 1.0),
            'reg_alpha': ('float', 0.0, 1.0),
            'reg_lambda': ('float', 0.0, 1.0),
            'min_child_weight': ('int', 1, 10)
        }
    
    @staticmethod
    def get_lightgbm_space() -> Dict[str, Any]:
        """Get LightGBM hyperparameter space."""
        return {
            'n_estimators': ('int', 50, 500),
            'max_depth': ('int', 3, 15),
            'learning_rate': ('float', 0.01, 0.3),
            'num_leaves': ('int', 10, 300),
            'subsample': ('float', 0.6, 1.0),
            'colsample_bytree': ('float', 0.6, 1.0),
            'reg_alpha': ('float', 0.0, 1.0),
            'reg_lambda': ('float', 0.0, 1.0),
            'min_child_samples': ('int', 5, 100)
        }
    
    @staticmethod
    def get_bert_space() -> Dict[str, Any]:
        """Get BERT fine-tuning hyperparameter space."""
        return {
            'learning_rate': ('float', 1e-5, 5e-4),
            'batch_size': ('categorical', [8, 16, 32]),
            'num_epochs': ('int', 2, 5),
            'warmup_steps': ('int', 0, 1000),
            'weight_decay': ('float', 0.0, 0.1),
            'dropout': ('float', 0.1, 0.5)
        }


class RayTrialRunner:
    """Runs individual trials using Ray."""
    
    def __init__(self, config: AutoMLConfig):
        """Initialize Ray trial runner.
        
        Args:
            config: AutoML configuration
        """
        self.config = config
        
        if not RAY_AVAILABLE:
            raise ImportError("Ray is required for distributed training")
    
    def run_trial(self, trial_config: Dict[str, Any], 
                  feature_set: FeatureSet, 
                  target: np.ndarray,
                  algorithm: str) -> Dict[str, float]:
        """Run a single trial.
        
        Args:
            trial_config: Trial configuration from Optuna
            feature_set: Feature set for training
            target: Target variable
            algorithm: Algorithm to use
            
        Returns:
            Dict of metrics
        """
        try:
            # Create model based on algorithm
            model = self._create_model(algorithm, trial_config)
            
            # Prepare data
            X = self._prepare_features(feature_set)
            y = target
            
            # Cross-validation
            cv = StratifiedKFold(
                n_splits=self.config.cv_folds,
                shuffle=True,
                random_state=self.config.random_state
            )
            
            # Calculate metrics
            cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
            
            metrics = {
                'accuracy': float(np.mean(cv_scores)),
                'accuracy_std': float(np.std(cv_scores)),
                'cv_scores': cv_scores.tolist()
            }
            
            # Additional metrics if needed
            if len(np.unique(y)) == 2:  # Binary classification
                precision_scores = cross_val_score(model, X, y, cv=cv, scoring='precision')
                recall_scores = cross_val_score(model, X, y, cv=cv, scoring='recall')
                f1_scores = cross_val_score(model, X, y, cv=cv, scoring='f1')
                
                metrics.update({
                    'precision': float(np.mean(precision_scores)),
                    'recall': float(np.mean(recall_scores)),
                    'f1': float(np.mean(f1_scores))
                })
            
            return metrics
            
        except Exception as e:
            logger.error(f"Trial failed: {e}")
            return {'accuracy': 0.0, 'error': str(e)}
    
    def _create_model(self, algorithm: str, params: Dict[str, Any]):
        """Create model instance with parameters."""
        if algorithm == 'random_forest':
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(**params, random_state=self.config.random_state)
        
        elif algorithm == 'xgboost':
            try:
                import xgboost as xgb
                return xgb.XGBClassifier(**params, random_state=self.config.random_state)
            except ImportError:
                logger.warning("XGBoost not available, falling back to Random Forest")
                return RandomForestClassifier(random_state=self.config.random_state)
        
        elif algorithm == 'lightgbm':
            try:
                import lightgbm as lgb
                return lgb.LGBMClassifier(**params, random_state=self.config.random_state)
            except ImportError:
                logger.warning("LightGBM not available, falling back to Random Forest")
                return RandomForestClassifier(random_state=self.config.random_state)
        
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
    
    def _prepare_features(self, feature_set: FeatureSet) -> np.ndarray:
        """Prepare features for training."""
        # Combine all features into a single matrix
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


class OptunaTuner:
    """Optuna-based hyperparameter tuner."""
    
    def __init__(self, config: AutoMLConfig):
        """Initialize Optuna tuner.
        
        Args:
            config: AutoML configuration
        """
        self.config = config
        
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is required for hyperparameter optimization")
        
        # Set up logging
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    def create_study(self, study_name: str, direction: str = 'maximize') -> optuna.Study:
        """Create Optuna study.
        
        Args:
            study_name: Name of the study
            direction: Optimization direction
            
        Returns:
            Optuna study object
        """
        # Create sampler
        if self.config.sampler == 'tpe':
            sampler = TPESampler(seed=self.config.random_state)
        elif self.config.sampler == 'cmaes':
            sampler = CmaEsSampler(seed=self.config.random_state)
        else:
            sampler = optuna.samplers.RandomSampler(seed=self.config.random_state)
        
        # Create pruner
        if self.config.pruner == 'median':
            pruner = MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=10,
                interval_steps=1
            )
        elif self.config.pruner == 'hyperband':
            pruner = HyperbandPruner(
                min_resource=1,
                max_resource=self.config.cv_folds,
                reduction_factor=3
            )
        else:
            pruner = optuna.pruners.NopPruner()
        
        # Create study
        study = optuna.create_study(
            study_name=study_name,
            direction=direction,
            sampler=sampler,
            pruner=pruner
        )
        
        return study
    
    def suggest_parameters(self, trial: optuna.Trial, 
                          param_space: Dict[str, Any]) -> Dict[str, Any]:
        """Suggest parameters for a trial.
        
        Args:
            trial: Optuna trial object
            param_space: Parameter space definition
            
        Returns:
            Dict of suggested parameters
        """
        params = {}
        
        for param_name, param_config in param_space.items():
            param_type = param_config[0]
            
            if param_type == 'int':
                low, high = param_config[1], param_config[2]
                params[param_name] = trial.suggest_int(param_name, low, high)
            
            elif param_type == 'float':
                low, high = param_config[1], param_config[2]
                log = len(param_config) > 3 and param_config[3]
                params[param_name] = trial.suggest_float(param_name, low, high, log=log)
            
            elif param_type == 'categorical':
                choices = param_config[1]
                params[param_name] = trial.suggest_categorical(param_name, choices)
            
            else:
                logger.warning(f"Unknown parameter type: {param_type}")
        
        return params


class AutoMLEngine:
    """Main AutoML engine coordinating Ray and Optuna."""
    
    def __init__(self, config: Optional[AutoMLConfig] = None):
        """Initialize AutoML engine.
        
        Args:
            config: AutoML configuration
        """
        self.config = config or AutoMLConfig()
        
        # Initialize Ray if available
        if RAY_AVAILABLE:
            if not ray.is_initialized():
                ray.init(
                    address=self.config.ray_address,
                    num_cpus=self.config.num_cpus,
                    num_gpus=self.config.num_gpus,
                    ignore_reinit_error=True
                )
            self.ray_runner = RayTrialRunner(self.config)
        else:
            logger.warning("Ray not available, using sequential processing")
            self.ray_runner = None
        
        # Initialize Optuna tuner
        if OPTUNA_AVAILABLE:
            self.optuna_tuner = OptunaTuner(self.config)
        else:
            raise ImportError("Optuna is required for AutoML")
        
        # Create model directory
        os.makedirs(self.config.model_dir, exist_ok=True)
        
        # Trial results storage
        self.trial_results = []
        
        logger.info("AutoMLEngine initialized")
    
    def optimize(self, feature_set: FeatureSet, 
                target: np.ndarray,
                algorithms: List[str] = None,
                study_name: str = None) -> Dict[str, Any]:
        """Run AutoML optimization.
        
        Args:
            feature_set: Feature set for training
            target: Target variable
            algorithms: List of algorithms to try
            study_name: Name for the optimization study
            
        Returns:
            Dict containing optimization results
        """
        if algorithms is None:
            algorithms = ['random_forest', 'xgboost', 'lightgbm']
        
        if study_name is None:
            study_name = f"automl_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"Starting AutoML optimization: {len(algorithms)} algorithms, "
                   f"{self.config.n_trials} trials per algorithm")
        
        best_results = {}
        all_trials = []
        
        for algorithm in algorithms:
            logger.info(f"Optimizing {algorithm}...")
            
            # Get parameter space
            param_space = self._get_parameter_space(algorithm)
            
            # Create study
            study = self.optuna_tuner.create_study(
                study_name=f"{study_name}_{algorithm}",
                direction='maximize'
            )
            
            # Define objective function
            def objective(trial):
                return self._objective_function(
                    trial, feature_set, target, algorithm, param_space
                )
            
            # Run optimization
            study.optimize(
                objective,
                n_trials=self.config.n_trials,
                timeout=self.config.timeout,
                n_jobs=1  # Ray handles parallelization
            )
            
            # Store results
            best_trial = study.best_trial
            best_results[algorithm] = {
                'best_params': best_trial.params,
                'best_value': best_trial.value,
                'n_trials': len(study.trials),
                'study': study
            }
            
            # Collect all trials
            for trial in study.trials:
                trial_result = TrialResult(
                    trial_id=f"{algorithm}_{trial.number}",
                    parameters=trial.params,
                    metrics={'accuracy': trial.value} if trial.value else {},
                    status='completed' if trial.state == optuna.trial.TrialState.COMPLETE else 'failed',
                    duration=0.0  # Would need to track this
                )
                all_trials.append(trial_result)
        
        # Find overall best algorithm
        best_algorithm = max(best_results.keys(), 
                           key=lambda k: best_results[k]['best_value'])
        
        results = {
            'best_algorithm': best_algorithm,
            'best_params': best_results[best_algorithm]['best_params'],
            'best_score': best_results[best_algorithm]['best_value'],
            'algorithm_results': best_results,
            'all_trials': all_trials,
            'total_trials': sum(len(result['study'].trials) for result in best_results.values()),
            'optimization_time': 0.0,  # Would need to track this
            'study_name': study_name
        }
        
        logger.info(f"AutoML optimization completed. Best: {best_algorithm} "
                   f"(score: {results['best_score']:.4f})")
        
        return results
    
    def _objective_function(self, trial: optuna.Trial,
                          feature_set: FeatureSet,
                          target: np.ndarray,
                          algorithm: str,
                          param_space: Dict[str, Any]) -> float:
        """Objective function for Optuna optimization.
        
        Args:
            trial: Optuna trial
            feature_set: Feature set
            target: Target variable
            algorithm: Algorithm name
            param_space: Parameter space
            
        Returns:
            Objective value (accuracy)
        """
        # Suggest parameters
        params = self.optuna_tuner.suggest_parameters(trial, param_space)
        
        # Run trial
        if self.ray_runner:
            metrics = self.ray_runner.run_trial(params, feature_set, target, algorithm)
        else:
            # Fallback to sequential execution
            metrics = self._run_trial_sequential(params, feature_set, target, algorithm)
        
        # Handle errors
        if 'error' in metrics:
            raise optuna.TrialPruned()
        
        return metrics.get('accuracy', 0.0)
    
    def _run_trial_sequential(self, params: Dict[str, Any],
                            feature_set: FeatureSet,
                            target: np.ndarray,
                            algorithm: str) -> Dict[str, float]:
        """Run trial sequentially (fallback when Ray is not available)."""
        try:
            # Create model
            if algorithm == 'random_forest':
                from sklearn.ensemble import RandomForestClassifier
                model = RandomForestClassifier(**params, random_state=self.config.random_state)
            else:
                # Fallback to Random Forest
                model = RandomForestClassifier(random_state=self.config.random_state)
            
            # Prepare features
            X = self._prepare_features(feature_set)
            y = target
            
            # Cross-validation
            cv = StratifiedKFold(
                n_splits=self.config.cv_folds,
                shuffle=True,
                random_state=self.config.random_state
            )
            
            cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
            
            return {
                'accuracy': float(np.mean(cv_scores)),
                'accuracy_std': float(np.std(cv_scores))
            }
            
        except Exception as e:
            return {'accuracy': 0.0, 'error': str(e)}
    
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
    
    def _get_parameter_space(self, algorithm: str) -> Dict[str, Any]:
        """Get parameter space for algorithm."""
        if algorithm == 'random_forest':
            return HyperparameterSpace.get_random_forest_space()
        elif algorithm == 'xgboost':
            return HyperparameterSpace.get_xgboost_space()
        elif algorithm == 'lightgbm':
            return HyperparameterSpace.get_lightgbm_space()
        elif algorithm == 'bert':
            return HyperparameterSpace.get_bert_space()
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
    
    def save_results(self, results: Dict[str, Any], filepath: str) -> None:
        """Save optimization results.
        
        Args:
            results: Optimization results
            filepath: Path to save results
        """
        # Convert Optuna studies to serializable format
        serializable_results = results.copy()
        
        for algorithm, result in serializable_results['algorithm_results'].items():
            if 'study' in result:
                study = result['study']
                result['study_summary'] = {
                    'n_trials': len(study.trials),
                    'best_trial_number': study.best_trial.number,
                    'best_value': study.best_value,
                    'best_params': study.best_params
                }
                del result['study']  # Remove non-serializable study object
        
        # Save to JSON
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {filepath}")
    
    def load_results(self, filepath: str) -> Dict[str, Any]:
        """Load optimization results.
        
        Args:
            filepath: Path to results file
            
        Returns:
            Loaded results
        """
        with open(filepath, 'r') as f:
            results = json.load(f)
        
        logger.info(f"Results loaded from {filepath}")
        return results
    
    def get_trial_history(self) -> List[TrialResult]:
        """Get history of all trials."""
        return self.trial_results
    
    def cleanup(self) -> None:
        """Cleanup resources."""
        if RAY_AVAILABLE and ray.is_initialized():
            ray.shutdown()
        
        logger.info("AutoML engine cleanup completed")