"""Advanced hyperparameter optimization with Bayesian optimization and multi-objective support."""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime
import json
import time
from abc import ABC, abstractmethod

# Optuna imports
try:
    import optuna
    from optuna.samplers import TPESampler, CmaEsSampler, NSGAIISampler
    from optuna.pruners import MedianPruner, HyperbandPruner, SuccessiveHalvingPruner
    from optuna.importance import get_param_importances
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

# Scikit-learn imports
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from ..config import get_logger
from .feature_extraction import FeatureSet
from .model_selector import BaseModelWrapper


logger = get_logger(__name__)


@dataclass
class OptimizationObjective:
    """Optimization objective definition."""
    name: str
    direction: str  # 'maximize' or 'minimize'
    weight: float = 1.0
    metric_function: Optional[Callable] = None


@dataclass
class HyperparameterTrial:
    """Single hyperparameter optimization trial."""
    trial_id: int
    parameters: Dict[str, Any]
    objectives: Dict[str, float]
    status: str  # 'completed', 'failed', 'pruned'
    duration: float
    created_at: datetime = field(default_factory=datetime.now)
    model_name: str = ""
    cv_scores: Optional[List[float]] = None


@dataclass
class OptimizationResult:
    """Result of hyperparameter optimization."""
    study_name: str
    best_trial: HyperparameterTrial
    best_parameters: Dict[str, Any]
    best_values: Dict[str, float]
    n_trials: int
    optimization_time: float
    parameter_importance: Dict[str, float]
    pareto_front: Optional[List[HyperparameterTrial]] = None
    convergence_history: List[float] = field(default_factory=list)


class ObjectiveFunction:
    """Objective function for hyperparameter optimization."""
    
    def __init__(self, model_wrapper: BaseModelWrapper,
                 feature_set: FeatureSet,
                 target: np.ndarray,
                 objectives: List[OptimizationObjective],
                 cv_folds: int = 5,
                 random_state: int = 42):
        """Initialize objective function.
        
        Args:
            model_wrapper: Model wrapper to optimize
            feature_set: Feature set for training
            target: Target variable
            objectives: List of optimization objectives
            cv_folds: Number of CV folds
            random_state: Random state
        """
        self.model_wrapper = model_wrapper
        self.feature_set = feature_set
        self.target = target
        self.objectives = objectives
        self.cv_folds = cv_folds
        self.random_state = random_state
        
        # Prepare data
        self.X = self._prepare_features()
        self.y = target
        
        # Cross-validation setup
        self.cv = StratifiedKFold(
            n_splits=cv_folds,
            shuffle=True,
            random_state=random_state
        )
    
    def _prepare_features(self) -> np.ndarray:
        """Prepare features for training."""
        feature_matrix = []
        
        for feature_name in self.feature_set.feature_names:
            if feature_name in self.feature_set.features:
                feature_values = self.feature_set.features[feature_name]
                if feature_values.ndim == 1:
                    feature_matrix.append(feature_values.reshape(-1, 1))
                else:
                    feature_matrix.append(feature_values)
        
        if not feature_matrix:
            raise ValueError("No features available for training")
        
        return np.hstack(feature_matrix)
    
    def __call__(self, trial: optuna.Trial) -> Union[float, List[float]]:
        """Evaluate objective function.
        
        Args:
            trial: Optuna trial
            
        Returns:
            Objective value(s)
        """
        try:
            # Get parameters from trial
            params = self._suggest_parameters(trial)
            
            # Create model with suggested parameters
            model = self._create_model_with_params(params)
            
            # Evaluate model
            objective_values = self._evaluate_model(model, trial)
            
            # Return single value for single objective, list for multi-objective
            if len(self.objectives) == 1:
                return objective_values[0]
            else:
                return objective_values
                
        except Exception as e:
            logger.error(f"Trial evaluation failed: {e}")
            # Return worst possible values
            if len(self.objectives) == 1:
                return -np.inf if self.objectives[0].direction == 'maximize' else np.inf
            else:
                return [
                    -np.inf if obj.direction == 'maximize' else np.inf
                    for obj in self.objectives
                ]
    
    def _suggest_parameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest parameters for trial."""
        # This should be implemented by specific model optimizers
        return {}
    
    def _create_model_with_params(self, params: Dict[str, Any]):
        """Create model with suggested parameters."""
        # Update model wrapper parameters
        self.model_wrapper.params.update(params)
        return self.model_wrapper.create_model()
    
    def _evaluate_model(self, model, trial: optuna.Trial) -> List[float]:
        """Evaluate model and return objective values."""
        objective_values = []
        
        for i, objective in enumerate(self.objectives):
            if objective.name == 'accuracy':
                scores = cross_val_score(model, self.X, self.y, cv=self.cv, scoring='accuracy')
                value = np.mean(scores)
                
                # Report intermediate value for pruning
                trial.report(value, i)
                
                # Check if trial should be pruned
                if trial.should_prune():
                    raise optuna.TrialPruned()
                
                objective_values.append(value)
                
            elif objective.name == 'f1_score':
                scores = cross_val_score(model, self.X, self.y, cv=self.cv, scoring='f1_weighted')
                value = np.mean(scores)
                objective_values.append(value)
                
            elif objective.name == 'roc_auc':
                if len(np.unique(self.y)) == 2:  # Binary classification
                    scores = cross_val_score(model, self.X, self.y, cv=self.cv, scoring='roc_auc')
                    value = np.mean(scores)
                else:
                    value = 0.0  # Multi-class AUC not implemented
                objective_values.append(value)
                
            elif objective.name == 'training_speed':
                # Measure training time
                start_time = time.time()
                model.fit(self.X, self.y)
                training_time = time.time() - start_time
                
                # Convert to speed (inverse of time)
                speed = 1.0 / (training_time + 1e-6)
                objective_values.append(speed)
                
            elif objective.name == 'model_size':
                # Estimate model size (simplified)
                import sys
                try:
                    model.fit(self.X, self.y)
                    size_mb = sys.getsizeof(model) / (1024 * 1024)
                    # Convert to inverse for minimization
                    size_score = 1.0 / (size_mb + 1e-6)
                    objective_values.append(size_score)
                except:
                    objective_values.append(0.0)
                    
            else:
                # Custom objective function
                if objective.metric_function:
                    value = objective.metric_function(model, self.X, self.y, self.cv)
                    objective_values.append(value)
                else:
                    objective_values.append(0.0)
        
        return objective_values


class RandomForestOptimizer(ObjectiveFunction):
    """Random Forest hyperparameter optimizer."""
    
    def _suggest_parameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest Random Forest parameters."""
        return {
            'n_estimators': trial.suggest_int('n_estimators', 10, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
            'class_weight': trial.suggest_categorical('class_weight', ['balanced', None])
        }


class XGBoostOptimizer(ObjectiveFunction):
    """XGBoost hyperparameter optimizer."""
    
    def _suggest_parameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest XGBoost parameters."""
        return {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10)
        }


class LightGBMOptimizer(ObjectiveFunction):
    """LightGBM hyperparameter optimizer."""
    
    def _suggest_parameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest LightGBM parameters."""
        return {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 10, 300),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100)
        }


class MultiObjectiveOptimizer:
    """Multi-objective hyperparameter optimizer."""
    
    def __init__(self, objectives: List[OptimizationObjective],
                 sampler: str = 'nsga2',
                 pruner: str = 'median',
                 random_state: int = 42):
        """Initialize multi-objective optimizer.
        
        Args:
            objectives: List of optimization objectives
            sampler: Sampler type ('nsga2', 'tpe')
            pruner: Pruner type ('median', 'hyperband', 'successive_halving')
            random_state: Random state
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is required for hyperparameter optimization")
        
        self.objectives = objectives
        self.sampler_type = sampler
        self.pruner_type = pruner
        self.random_state = random_state
    
    def create_study(self, study_name: str) -> optuna.Study:
        """Create Optuna study for multi-objective optimization.
        
        Args:
            study_name: Name of the study
            
        Returns:
            Optuna study object
        """
        # Create sampler
        if self.sampler_type == 'nsga2':
            sampler = NSGAIISampler(seed=self.random_state)
        elif self.sampler_type == 'tpe':
            sampler = TPESampler(seed=self.random_state)
        else:
            sampler = optuna.samplers.RandomSampler(seed=self.random_state)
        
        # Create pruner
        if self.pruner_type == 'median':
            pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        elif self.pruner_type == 'hyperband':
            pruner = HyperbandPruner(min_resource=1, max_resource=10, reduction_factor=3)
        elif self.pruner_type == 'successive_halving':
            pruner = SuccessiveHalvingPruner()
        else:
            pruner = optuna.pruners.NopPruner()
        
        # Create directions
        directions = [obj.direction for obj in self.objectives]
        
        # Create study
        study = optuna.create_study(
            study_name=study_name,
            directions=directions,
            sampler=sampler,
            pruner=pruner
        )
        
        return study
    
    def optimize_pareto_front(self, study: optuna.Study) -> List[HyperparameterTrial]:
        """Extract Pareto front from multi-objective optimization.
        
        Args:
            study: Completed Optuna study
            
        Returns:
            List of Pareto optimal trials
        """
        pareto_trials = []
        
        # Get completed trials
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        
        if not completed_trials:
            return pareto_trials
        
        # Extract objective values
        objective_values = []
        for trial in completed_trials:
            if trial.values:
                objective_values.append(trial.values)
            else:
                continue
        
        if not objective_values:
            return pareto_trials
        
        objective_values = np.array(objective_values)
        
        # Find Pareto front
        pareto_indices = self._find_pareto_front(objective_values)
        
        # Convert to HyperparameterTrial objects
        for idx in pareto_indices:
            trial = completed_trials[idx]
            
            objectives_dict = {}
            for i, obj in enumerate(self.objectives):
                if i < len(trial.values):
                    objectives_dict[obj.name] = trial.values[i]
            
            pareto_trial = HyperparameterTrial(
                trial_id=trial.number,
                parameters=trial.params,
                objectives=objectives_dict,
                status='completed',
                duration=0.0,  # Would need to track this
                model_name=""
            )
            pareto_trials.append(pareto_trial)
        
        return pareto_trials
    
    def _find_pareto_front(self, objective_values: np.ndarray) -> List[int]:
        """Find Pareto front indices.
        
        Args:
            objective_values: Array of objective values
            
        Returns:
            List of Pareto optimal indices
        """
        # Convert minimization objectives to maximization
        adjusted_values = objective_values.copy()
        for i, obj in enumerate(self.objectives):
            if obj.direction == 'minimize':
                adjusted_values[:, i] = -adjusted_values[:, i]
        
        # Find Pareto front
        pareto_indices = []
        n_points = len(adjusted_values)
        
        for i in range(n_points):
            is_pareto = True
            
            for j in range(n_points):
                if i != j:
                    # Check if point j dominates point i
                    if np.all(adjusted_values[j] >= adjusted_values[i]) and \
                       np.any(adjusted_values[j] > adjusted_values[i]):
                        is_pareto = False
                        break
            
            if is_pareto:
                pareto_indices.append(i)
        
        return pareto_indices


class HyperparameterOptimizer:
    """Main hyperparameter optimization engine."""
    
    def __init__(self, random_state: int = 42):
        """Initialize hyperparameter optimizer.
        
        Args:
            random_state: Random state
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is required for hyperparameter optimization")
        
        self.random_state = random_state
        self.studies = {}
        self.results = {}
        
        logger.info("HyperparameterOptimizer initialized")
    
    def optimize_single_objective(self, model_wrapper: BaseModelWrapper,
                                 feature_set: FeatureSet,
                                 target: np.ndarray,
                                 objective: OptimizationObjective,
                                 n_trials: int = 100,
                                 timeout: Optional[int] = None,
                                 study_name: Optional[str] = None) -> OptimizationResult:
        """Optimize single objective.
        
        Args:
            model_wrapper: Model wrapper to optimize
            feature_set: Feature set for training
            target: Target variable
            objective: Optimization objective
            n_trials: Number of trials
            timeout: Timeout in seconds
            study_name: Name of the study
            
        Returns:
            Optimization result
        """
        if study_name is None:
            study_name = f"single_obj_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"Starting single-objective optimization: {objective.name}")
        
        # Create objective function
        obj_func = self._create_objective_function(
            model_wrapper, feature_set, target, [objective]
        )
        
        # Create study
        study = optuna.create_study(
            study_name=study_name,
            direction=objective.direction,
            sampler=TPESampler(seed=self.random_state),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        )
        
        # Optimize
        start_time = time.time()
        study.optimize(obj_func, n_trials=n_trials, timeout=timeout)
        optimization_time = time.time() - start_time
        
        # Get parameter importance
        try:
            param_importance = get_param_importances(study)
        except:
            param_importance = {}
        
        # Create result
        best_trial = study.best_trial
        
        best_trial_obj = HyperparameterTrial(
            trial_id=best_trial.number,
            parameters=best_trial.params,
            objectives={objective.name: best_trial.value},
            status='completed',
            duration=0.0,
            model_name=model_wrapper.name
        )
        
        result = OptimizationResult(
            study_name=study_name,
            best_trial=best_trial_obj,
            best_parameters=best_trial.params,
            best_values={objective.name: best_trial.value},
            n_trials=len(study.trials),
            optimization_time=optimization_time,
            parameter_importance=param_importance,
            convergence_history=[t.value for t in study.trials if t.value is not None]
        )
        
        self.studies[study_name] = study
        self.results[study_name] = result
        
        logger.info(f"Single-objective optimization completed: "
                   f"best {objective.name} = {best_trial.value:.4f}")
        
        return result
    
    def optimize_multi_objective(self, model_wrapper: BaseModelWrapper,
                                feature_set: FeatureSet,
                                target: np.ndarray,
                                objectives: List[OptimizationObjective],
                                n_trials: int = 100,
                                timeout: Optional[int] = None,
                                study_name: Optional[str] = None) -> OptimizationResult:
        """Optimize multiple objectives.
        
        Args:
            model_wrapper: Model wrapper to optimize
            feature_set: Feature set for training
            target: Target variable
            objectives: List of optimization objectives
            n_trials: Number of trials
            timeout: Timeout in seconds
            study_name: Name of the study
            
        Returns:
            Optimization result
        """
        if study_name is None:
            study_name = f"multi_obj_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"Starting multi-objective optimization: "
                   f"{[obj.name for obj in objectives]}")
        
        # Create multi-objective optimizer
        multi_opt = MultiObjectiveOptimizer(objectives, random_state=self.random_state)
        
        # Create objective function
        obj_func = self._create_objective_function(
            model_wrapper, feature_set, target, objectives
        )
        
        # Create study
        study = multi_opt.create_study(study_name)
        
        # Optimize
        start_time = time.time()
        study.optimize(obj_func, n_trials=n_trials, timeout=timeout)
        optimization_time = time.time() - start_time
        
        # Get Pareto front
        pareto_front = multi_opt.optimize_pareto_front(study)
        
        # Get best trial (first Pareto optimal solution)
        if pareto_front:
            best_trial_obj = pareto_front[0]
        else:
            # Fallback to best trial by first objective
            best_trial = study.best_trials[0] if study.best_trials else None
            if best_trial:
                objectives_dict = {}
                for i, obj in enumerate(objectives):
                    if i < len(best_trial.values):
                        objectives_dict[obj.name] = best_trial.values[i]
                
                best_trial_obj = HyperparameterTrial(
                    trial_id=best_trial.number,
                    parameters=best_trial.params,
                    objectives=objectives_dict,
                    status='completed',
                    duration=0.0,
                    model_name=model_wrapper.name
                )
            else:
                best_trial_obj = None
        
        # Create result
        result = OptimizationResult(
            study_name=study_name,
            best_trial=best_trial_obj,
            best_parameters=best_trial_obj.parameters if best_trial_obj else {},
            best_values=best_trial_obj.objectives if best_trial_obj else {},
            n_trials=len(study.trials),
            optimization_time=optimization_time,
            parameter_importance={},  # Not available for multi-objective
            pareto_front=pareto_front
        )
        
        self.studies[study_name] = study
        self.results[study_name] = result
        
        logger.info(f"Multi-objective optimization completed: "
                   f"{len(pareto_front)} Pareto optimal solutions")
        
        return result
    
    def _create_objective_function(self, model_wrapper: BaseModelWrapper,
                                  feature_set: FeatureSet,
                                  target: np.ndarray,
                                  objectives: List[OptimizationObjective]) -> ObjectiveFunction:
        """Create objective function based on model type."""
        model_name = model_wrapper.__class__.__name__.lower()
        
        if 'randomforest' in model_name:
            return RandomForestOptimizer(
                model_wrapper, feature_set, target, objectives,
                random_state=self.random_state
            )
        elif 'xgboost' in model_name:
            return XGBoostOptimizer(
                model_wrapper, feature_set, target, objectives,
                random_state=self.random_state
            )
        elif 'lightgbm' in model_name:
            return LightGBMOptimizer(
                model_wrapper, feature_set, target, objectives,
                random_state=self.random_state
            )
        else:
            # Generic objective function
            return ObjectiveFunction(
                model_wrapper, feature_set, target, objectives,
                random_state=self.random_state
            )
    
    def get_optimization_history(self, study_name: str) -> List[Dict[str, Any]]:
        """Get optimization history for a study.
        
        Args:
            study_name: Name of the study
            
        Returns:
            List of trial information
        """
        if study_name not in self.studies:
            return []
        
        study = self.studies[study_name]
        history = []
        
        for trial in study.trials:
            trial_info = {
                'trial_id': trial.number,
                'parameters': trial.params,
                'values': trial.values,
                'state': trial.state.name,
                'datetime_start': trial.datetime_start,
                'datetime_complete': trial.datetime_complete
            }
            history.append(trial_info)
        
        return history
    
    def save_results(self, study_name: str, filepath: str) -> None:
        """Save optimization results.
        
        Args:
            study_name: Name of the study
            filepath: Path to save results
        """
        if study_name not in self.results:
            logger.error(f"Study {study_name} not found")
            return
        
        result = self.results[study_name]
        
        # Convert to serializable format
        serializable_result = {
            'study_name': result.study_name,
            'best_parameters': result.best_parameters,
            'best_values': result.best_values,
            'n_trials': result.n_trials,
            'optimization_time': result.optimization_time,
            'parameter_importance': result.parameter_importance,
            'convergence_history': result.convergence_history
        }
        
        if result.pareto_front:
            serializable_result['pareto_front'] = [
                {
                    'trial_id': trial.trial_id,
                    'parameters': trial.parameters,
                    'objectives': trial.objectives
                }
                for trial in result.pareto_front
            ]
        
        with open(filepath, 'w') as f:
            json.dump(serializable_result, f, indent=2, default=str)
        
        logger.info(f"Results saved to {filepath}")
    
    def get_parameter_importance_analysis(self, study_name: str) -> Dict[str, Any]:
        """Get parameter importance analysis.
        
        Args:
            study_name: Name of the study
            
        Returns:
            Parameter importance analysis
        """
        if study_name not in self.studies:
            return {}
        
        study = self.studies[study_name]
        
        try:
            # Get parameter importance
            importance = get_param_importances(study)
            
            # Sort by importance
            sorted_importance = sorted(importance.items(), 
                                     key=lambda x: x[1], reverse=True)
            
            return {
                'parameter_importance': importance,
                'sorted_importance': sorted_importance,
                'most_important': sorted_importance[0] if sorted_importance else None,
                'least_important': sorted_importance[-1] if sorted_importance else None
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze parameter importance: {e}")
            return {'error': str(e)}