"""Real-time model serving infrastructure with caching and batch prediction capabilities."""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import pickle
import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from pathlib import Path
import hashlib

# Caching
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

# Model serving
from sklearn.base import BaseEstimator
import joblib

from ..config import get_logger
from .model_registry import MLflowModelRegistry, ModelVersion
from .feature_extraction import FeatureSet


logger = get_logger(__name__)


@dataclass
class PredictionRequest:
    """Single prediction request."""
    request_id: str
    features: Dict[str, Any]
    model_name: str
    model_version: Optional[str] = None
    model_stage: Optional[str] = None
    return_probabilities: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class PredictionResponse:
    """Single prediction response."""
    request_id: str
    prediction: Union[int, float, str, List]
    probabilities: Optional[Dict[str, float]] = None
    confidence: Optional[float] = None
    model_name: str = ""
    model_version: str = ""
    processing_time_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    error: Optional[str] = None


@dataclass
class BatchPredictionRequest:
    """Batch prediction request."""
    batch_id: str
    requests: List[PredictionRequest]
    model_name: str
    model_version: Optional[str] = None
    model_stage: Optional[str] = None
    parallel_processing: bool = True
    max_workers: int = 4


@dataclass
class BatchPredictionResponse:
    """Batch prediction response."""
    batch_id: str
    responses: List[PredictionResponse]
    total_requests: int
    successful_predictions: int
    failed_predictions: int
    total_processing_time_ms: float
    average_processing_time_ms: float


@dataclass
class ModelCacheEntry:
    """Model cache entry."""
    model: Any
    model_name: str
    model_version: str
    load_time: datetime
    last_accessed: datetime
    access_count: int = 0
    size_mb: float = 0.0


class ModelCache:
    """Thread-safe model cache with LRU eviction."""
    
    def __init__(self, max_size: int = 5, ttl_hours: int = 24):
        """Initialize model cache.
        
        Args:
            max_size: Maximum number of models to cache
            ttl_hours: Time-to-live for cached models in hours
        """
        self.max_size = max_size
        self.ttl = timedelta(hours=ttl_hours)
        self.cache = {}
        self.lock = threading.RLock()
        
        logger.info(f"Model cache initialized: max_size={max_size}, ttl={ttl_hours}h")
    
    def _generate_key(self, model_name: str, model_version: str) -> str:
        """Generate cache key."""
        return f"{model_name}:{model_version}"
    
    def get(self, model_name: str, model_version: str) -> Optional[Any]:
        """Get model from cache.
        
        Args:
            model_name: Name of the model
            model_version: Version of the model
            
        Returns:
            Cached model or None
        """
        key = self._generate_key(model_name, model_version)
        
        with self.lock:
            if key not in self.cache:
                return None
            
            entry = self.cache[key]
            
            # Check TTL
            if datetime.now() - entry.load_time > self.ttl:
                del self.cache[key]
                logger.debug(f"Model cache entry expired: {key}")
                return None
            
            # Update access info
            entry.last_accessed = datetime.now()
            entry.access_count += 1
            
            logger.debug(f"Model cache hit: {key}")
            return entry.model
    
    def put(self, model_name: str, model_version: str, model: Any) -> None:
        """Put model in cache.
        
        Args:
            model_name: Name of the model
            model_version: Version of the model
            model: Model object to cache
        """
        key = self._generate_key(model_name, model_version)
        
        with self.lock:
            # Calculate model size
            try:
                import sys
                size_mb = sys.getsizeof(pickle.dumps(model)) / (1024 * 1024)
            except:
                size_mb = 0.0
            
            # Create cache entry
            entry = ModelCacheEntry(
                model=model,
                model_name=model_name,
                model_version=model_version,
                load_time=datetime.now(),
                last_accessed=datetime.now(),
                size_mb=size_mb
            )
            
            # Evict if cache is full
            if len(self.cache) >= self.max_size:
                self._evict_lru()
            
            self.cache[key] = entry
            logger.debug(f"Model cached: {key} ({size_mb:.2f}MB)")
    
    def _evict_lru(self) -> None:
        """Evict least recently used model."""
        if not self.cache:
            return
        
        # Find LRU entry
        lru_key = min(self.cache.keys(), 
                     key=lambda k: self.cache[k].last_accessed)
        
        del self.cache[lru_key]
        logger.debug(f"Evicted LRU model: {lru_key}")
    
    def clear(self) -> None:
        """Clear all cached models."""
        with self.lock:
            self.cache.clear()
            logger.info("Model cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            total_size_mb = sum(entry.size_mb for entry in self.cache.values())
            total_accesses = sum(entry.access_count for entry in self.cache.values())
            
            return {
                'cached_models': len(self.cache),
                'max_size': self.max_size,
                'total_size_mb': total_size_mb,
                'total_accesses': total_accesses,
                'models': [
                    {
                        'name': entry.model_name,
                        'version': entry.model_version,
                        'size_mb': entry.size_mb,
                        'access_count': entry.access_count,
                        'last_accessed': entry.last_accessed.isoformat()
                    }
                    for entry in self.cache.values()
                ]
            }


class PredictionCache:
    """Cache for prediction results."""
    
    def __init__(self, redis_client=None, ttl_seconds: int = 3600):
        """Initialize prediction cache.
        
        Args:
            redis_client: Redis client for distributed caching
            ttl_seconds: Time-to-live for cached predictions
        """
        self.redis_client = redis_client
        self.ttl_seconds = ttl_seconds
        self.local_cache = {}
        self.lock = threading.RLock()
        
        logger.info(f"Prediction cache initialized: ttl={ttl_seconds}s")
    
    def _generate_key(self, features: Dict[str, Any], model_name: str, 
                     model_version: str) -> str:
        """Generate cache key for prediction."""
        # Create deterministic hash of features
        feature_str = json.dumps(features, sort_keys=True)
        feature_hash = hashlib.md5(feature_str.encode()).hexdigest()
        
        return f"pred:{model_name}:{model_version}:{feature_hash}"
    
    def get(self, features: Dict[str, Any], model_name: str, 
           model_version: str) -> Optional[PredictionResponse]:
        """Get cached prediction.
        
        Args:
            features: Input features
            model_name: Model name
            model_version: Model version
            
        Returns:
            Cached prediction response or None
        """
        key = self._generate_key(features, model_name, model_version)
        
        # Try Redis first
        if self.redis_client:
            try:
                cached_data = self.redis_client.get(key)
                if cached_data:
                    response_dict = json.loads(cached_data)
                    response = PredictionResponse(**response_dict)
                    logger.debug(f"Prediction cache hit (Redis): {key}")
                    return response
            except Exception as e:
                logger.warning(f"Redis cache error: {e}")
        
        # Fallback to local cache
        with self.lock:
            if key in self.local_cache:
                entry_time, response = self.local_cache[key]
                
                # Check TTL
                if time.time() - entry_time < self.ttl_seconds:
                    logger.debug(f"Prediction cache hit (local): {key}")
                    return response
                else:
                    del self.local_cache[key]
        
        return None
    
    def put(self, features: Dict[str, Any], model_name: str, 
           model_version: str, response: PredictionResponse) -> None:
        """Cache prediction result.
        
        Args:
            features: Input features
            model_name: Model name
            model_version: Model version
            response: Prediction response to cache
        """
        key = self._generate_key(features, model_name, model_version)
        
        # Cache in Redis
        if self.redis_client:
            try:
                response_dict = {
                    'request_id': response.request_id,
                    'prediction': response.prediction,
                    'probabilities': response.probabilities,
                    'confidence': response.confidence,
                    'model_name': response.model_name,
                    'model_version': response.model_version,
                    'processing_time_ms': response.processing_time_ms,
                    'timestamp': response.timestamp.isoformat()
                }
                
                self.redis_client.setex(
                    key, 
                    self.ttl_seconds, 
                    json.dumps(response_dict, default=str)
                )
                logger.debug(f"Prediction cached (Redis): {key}")
                return
            except Exception as e:
                logger.warning(f"Redis cache error: {e}")
        
        # Fallback to local cache
        with self.lock:
            self.local_cache[key] = (time.time(), response)
            logger.debug(f"Prediction cached (local): {key}")


class ModelServer:
    """High-performance model serving infrastructure."""
    
    def __init__(self, registry: MLflowModelRegistry,
                 model_cache_size: int = 5,
                 enable_prediction_cache: bool = True,
                 redis_config: Optional[Dict[str, Any]] = None):
        """Initialize model server.
        
        Args:
            registry: MLflow model registry
            model_cache_size: Maximum number of models to cache
            enable_prediction_cache: Whether to enable prediction caching
            redis_config: Redis configuration for distributed caching
        """
        self.registry = registry
        
        # Initialize model cache
        self.model_cache = ModelCache(max_size=model_cache_size)
        
        # Initialize prediction cache
        redis_client = None
        if enable_prediction_cache and redis_config and REDIS_AVAILABLE:
            try:
                redis_client = redis.Redis(**redis_config)
                redis_client.ping()  # Test connection
                logger.info("Connected to Redis for prediction caching")
            except Exception as e:
                logger.warning(f"Failed to connect to Redis: {e}")
        
        self.prediction_cache = PredictionCache(redis_client) if enable_prediction_cache else None
        
        # Thread pool for batch processing
        self.executor = ThreadPoolExecutor(max_workers=8)
        
        # Metrics
        self.metrics = {
            'total_requests': 0,
            'successful_predictions': 0,
            'failed_predictions': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'total_processing_time_ms': 0.0
        }
        self.metrics_lock = threading.Lock()
        
        logger.info("Model server initialized")
    
    def load_model(self, model_name: str, model_version: Optional[str] = None,
                  model_stage: Optional[str] = None) -> Any:
        """Load model with caching.
        
        Args:
            model_name: Name of the model
            model_version: Specific version to load
            model_stage: Stage to load from
            
        Returns:
            Loaded model
        """
        # Determine version to load
        if not model_version:
            if model_stage:
                version_info = self.registry.get_latest_model_version(model_name, model_stage)
                if not version_info:
                    raise ValueError(f"No model found in stage {model_stage}")
                model_version = version_info.version
            else:
                version_info = self.registry.get_latest_model_version(model_name)
                if not version_info:
                    raise ValueError(f"No versions found for model {model_name}")
                model_version = version_info.version
        
        # Check cache first
        cached_model = self.model_cache.get(model_name, model_version)
        if cached_model:
            return cached_model
        
        # Load from registry
        logger.info(f"Loading model: {model_name} v{model_version}")
        model = self.registry.load_model(model_name, model_version)
        
        # Cache the model
        self.model_cache.put(model_name, model_version, model)
        
        return model
    
    def predict(self, request: PredictionRequest) -> PredictionResponse:
        """Make single prediction.
        
        Args:
            request: Prediction request
            
        Returns:
            Prediction response
        """
        start_time = time.time()
        
        try:
            # Update metrics
            with self.metrics_lock:
                self.metrics['total_requests'] += 1
            
            # Check prediction cache
            if self.prediction_cache:
                cached_response = self.prediction_cache.get(
                    request.features, request.model_name, 
                    request.model_version or 'latest'
                )
                
                if cached_response:
                    with self.metrics_lock:
                        self.metrics['cache_hits'] += 1
                    
                    # Update request ID and timestamp
                    cached_response.request_id = request.request_id
                    cached_response.timestamp = datetime.now()
                    return cached_response
                else:
                    with self.metrics_lock:
                        self.metrics['cache_misses'] += 1
            
            # Load model
            model = self.load_model(
                request.model_name, 
                request.model_version, 
                request.model_stage
            )
            
            # Prepare features
            feature_array = self._prepare_features(request.features)
            
            # Make prediction
            prediction = model.predict(feature_array)[0]
            
            # Get probabilities if requested
            probabilities = None
            confidence = None
            
            if request.return_probabilities and hasattr(model, 'predict_proba'):
                proba = model.predict_proba(feature_array)[0]
                
                # Convert to dict if binary classification
                if len(proba) == 2:
                    probabilities = {'class_0': float(proba[0]), 'class_1': float(proba[1])}
                    confidence = float(max(proba))
                else:
                    probabilities = {f'class_{i}': float(p) for i, p in enumerate(proba)}
                    confidence = float(max(proba))
            
            # Create response
            processing_time_ms = (time.time() - start_time) * 1000
            
            response = PredictionResponse(
                request_id=request.request_id,
                prediction=prediction,
                probabilities=probabilities,
                confidence=confidence,
                model_name=request.model_name,
                model_version=request.model_version or 'latest',
                processing_time_ms=processing_time_ms
            )
            
            # Cache the response
            if self.prediction_cache:
                self.prediction_cache.put(
                    request.features, request.model_name,
                    request.model_version or 'latest', response
                )
            
            # Update metrics
            with self.metrics_lock:
                self.metrics['successful_predictions'] += 1
                self.metrics['total_processing_time_ms'] += processing_time_ms
            
            return response
            
        except Exception as e:
            # Update metrics
            with self.metrics_lock:
                self.metrics['failed_predictions'] += 1
            
            processing_time_ms = (time.time() - start_time) * 1000
            
            logger.error(f"Prediction failed: {e}")
            
            return PredictionResponse(
                request_id=request.request_id,
                prediction=None,
                model_name=request.model_name,
                model_version=request.model_version or 'latest',
                processing_time_ms=processing_time_ms,
                error=str(e)
            )
    
    def predict_batch(self, batch_request: BatchPredictionRequest) -> BatchPredictionResponse:
        """Make batch predictions.
        
        Args:
            batch_request: Batch prediction request
            
        Returns:
            Batch prediction response
        """
        start_time = time.time()
        
        logger.info(f"Processing batch: {batch_request.batch_id} "
                   f"({len(batch_request.requests)} requests)")
        
        responses = []
        
        if batch_request.parallel_processing and len(batch_request.requests) > 1:
            # Parallel processing
            futures = []
            
            for request in batch_request.requests:
                future = self.executor.submit(self.predict, request)
                futures.append(future)
            
            # Collect results
            for future in as_completed(futures):
                try:
                    response = future.result()
                    responses.append(response)
                except Exception as e:
                    logger.error(f"Batch prediction failed: {e}")
        else:
            # Sequential processing
            for request in batch_request.requests:
                response = self.predict(request)
                responses.append(response)
        
        # Calculate batch metrics
        total_processing_time_ms = (time.time() - start_time) * 1000
        successful_predictions = len([r for r in responses if r.error is None])
        failed_predictions = len([r for r in responses if r.error is not None])
        average_processing_time_ms = (
            total_processing_time_ms / len(responses) if responses else 0.0
        )
        
        batch_response = BatchPredictionResponse(
            batch_id=batch_request.batch_id,
            responses=responses,
            total_requests=len(batch_request.requests),
            successful_predictions=successful_predictions,
            failed_predictions=failed_predictions,
            total_processing_time_ms=total_processing_time_ms,
            average_processing_time_ms=average_processing_time_ms
        )
        
        logger.info(f"Batch completed: {batch_request.batch_id} "
                   f"({successful_predictions}/{len(batch_request.requests)} successful)")
        
        return batch_response
    
    def _prepare_features(self, features: Dict[str, Any]) -> np.ndarray:
        """Prepare features for prediction.
        
        Args:
            features: Feature dictionary
            
        Returns:
            Feature array
        """
        # Convert features to array
        # This is a simplified implementation - in practice, you'd need
        # to match the exact feature format used during training
        
        if isinstance(features, dict):
            # Assume features are in the correct order
            feature_values = list(features.values())
            return np.array(feature_values).reshape(1, -1)
        elif isinstance(features, (list, np.ndarray)):
            return np.array(features).reshape(1, -1)
        else:
            raise ValueError(f"Unsupported feature format: {type(features)}")
    
    def get_model_info(self, model_name: str, model_version: Optional[str] = None) -> Dict[str, Any]:
        """Get information about a loaded model.
        
        Args:
            model_name: Name of the model
            model_version: Version of the model
            
        Returns:
            Model information
        """
        try:
            # Get model versions
            versions = self.registry.get_model_versions(model_name)
            
            if model_version:
                version_info = next((v for v in versions if v.version == model_version), None)
            else:
                version_info = versions[0] if versions else None
            
            if not version_info:
                return {'error': 'Model not found'}
            
            # Check if model is cached
            is_cached = self.model_cache.get(model_name, version_info.version) is not None
            
            return {
                'model_name': model_name,
                'version': version_info.version,
                'stage': version_info.stage,
                'creation_timestamp': version_info.creation_timestamp.isoformat(),
                'metrics': version_info.metrics,
                'is_cached': is_cached,
                'description': version_info.description
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def get_server_stats(self) -> Dict[str, Any]:
        """Get server statistics.
        
        Returns:
            Server statistics
        """
        with self.metrics_lock:
            metrics = self.metrics.copy()
        
        # Calculate derived metrics
        if metrics['total_requests'] > 0:
            success_rate = metrics['successful_predictions'] / metrics['total_requests']
            avg_processing_time = (
                metrics['total_processing_time_ms'] / metrics['successful_predictions']
                if metrics['successful_predictions'] > 0 else 0.0
            )
        else:
            success_rate = 0.0
            avg_processing_time = 0.0
        
        cache_hit_rate = (
            metrics['cache_hits'] / (metrics['cache_hits'] + metrics['cache_misses'])
            if (metrics['cache_hits'] + metrics['cache_misses']) > 0 else 0.0
        )
        
        return {
            'requests': metrics,
            'success_rate': success_rate,
            'average_processing_time_ms': avg_processing_time,
            'cache_hit_rate': cache_hit_rate,
            'model_cache': self.model_cache.get_stats(),
            'uptime_seconds': time.time() - getattr(self, '_start_time', time.time())
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check.
        
        Returns:
            Health status
        """
        try:
            # Check registry connection
            models = self.registry.search_models(max_results=1)
            registry_healthy = True
        except Exception as e:
            registry_healthy = False
            logger.error(f"Registry health check failed: {e}")
        
        # Check cache
        cache_healthy = True
        try:
            self.model_cache.get_stats()
        except Exception as e:
            cache_healthy = False
            logger.error(f"Cache health check failed: {e}")
        
        overall_healthy = registry_healthy and cache_healthy
        
        return {
            'status': 'healthy' if overall_healthy else 'unhealthy',
            'registry': 'healthy' if registry_healthy else 'unhealthy',
            'cache': 'healthy' if cache_healthy else 'unhealthy',
            'timestamp': datetime.now().isoformat()
        }
    
    def shutdown(self) -> None:
        """Shutdown the model server."""
        logger.info("Shutting down model server...")
        
        # Shutdown thread pool
        self.executor.shutdown(wait=True)
        
        # Clear caches
        self.model_cache.clear()
        
        logger.info("Model server shutdown complete")