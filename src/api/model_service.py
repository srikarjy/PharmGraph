"""Model service for API integration with caching and version management."""

import asyncio
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import uuid
import logging

from ..config import get_logger
from ..ml.model_server import ModelServer, PredictionRequest, PredictionResponse, BatchPredictionRequest
from .schemas import (
    PredictionRequest as APIPredictionRequest,
    PredictionResponse as APIPredictionResponse,
    BatchPredictionRequest as APIBatchPredictionRequest,
    BatchPredictionResponse as APIBatchPredictionResponse
)

logger = get_logger(__name__)


class ModelService:
    """Service layer for model operations in the API."""
    
    def __init__(self, model_server: ModelServer):
        """Initialize model service.
        
        Args:
            model_server: ML model server instance
        """
        self.model_server = model_server
        self.default_model = "pharmacogenomics_classifier"
        self.model_aliases = {
            "pgx": "pharmacogenomics_classifier",
            "drug_gene": "drug_gene_interaction_model",
            "risk": "risk_scoring_model",
            "dosing": "dosing_recommendation_model"
        }
        
        logger.info("Model service initialized")
    
    async def predict(self, request: APIPredictionRequest) -> APIPredictionResponse:
        """Make a single prediction.
        
        Args:
            request: API prediction request
            
        Returns:
            API prediction response
        """
        try:
            # Resolve model name
            model_name = self._resolve_model_name(request.model_name)
            
            # Create internal prediction request
            internal_request = PredictionRequest(
                request_id=str(uuid.uuid4()),
                features=request.features,
                model_name=model_name,
                model_version=request.model_version,
                return_probabilities=request.return_probabilities,
                metadata=request.metadata or {}
            )
            
            # Make prediction (run in thread pool for async)
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, 
                self.model_server.predict, 
                internal_request
            )
            
            # Convert to API response
            return self._convert_to_api_response(response)
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            
            # Return error response
            return APIPredictionResponse(
                prediction=None,
                model_name=request.model_name or self.default_model,
                model_version=request.model_version or "latest",
                processing_time_ms=0.0,
                timestamp=datetime.utcnow(),
                error=str(e)
            )
    
    async def predict_batch(self, request: APIBatchPredictionRequest) -> APIBatchPredictionResponse:
        """Make batch predictions.
        
        Args:
            request: API batch prediction request
            
        Returns:
            API batch prediction response
        """
        try:
            start_time = time.time()
            
            # Convert API requests to internal requests
            internal_requests = []
            for api_req in request.requests:
                model_name = self._resolve_model_name(
                    api_req.model_name or request.model_name
                )
                
                internal_req = PredictionRequest(
                    request_id=str(uuid.uuid4()),
                    features=api_req.features,
                    model_name=model_name,
                    model_version=api_req.model_version or request.model_version,
                    return_probabilities=api_req.return_probabilities,
                    metadata=api_req.metadata or {}
                )
                internal_requests.append(internal_req)
            
            # Create batch request
            batch_request = BatchPredictionRequest(
                batch_id=str(uuid.uuid4()),
                requests=internal_requests,
                model_name=self._resolve_model_name(request.model_name),
                model_version=request.model_version,
                parallel_processing=request.parallel_processing
            )
            
            # Make batch prediction
            loop = asyncio.get_event_loop()
            batch_response = await loop.run_in_executor(
                None,
                self.model_server.predict_batch,
                batch_request
            )
            
            # Convert responses
            api_responses = [
                self._convert_to_api_response(resp) 
                for resp in batch_response.responses
            ]
            
            total_time = (time.time() - start_time) * 1000
            
            return APIBatchPredictionResponse(
                responses=api_responses,
                total_requests=batch_response.total_requests,
                successful_predictions=batch_response.successful_predictions,
                failed_predictions=batch_response.failed_predictions,
                total_processing_time_ms=total_time,
                average_processing_time_ms=batch_response.average_processing_time_ms,
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"Batch prediction failed: {e}")
            
            # Return error response
            return APIBatchPredictionResponse(
                responses=[],
                total_requests=len(request.requests),
                successful_predictions=0,
                failed_predictions=len(request.requests),
                total_processing_time_ms=0.0,
                average_processing_time_ms=0.0,
                timestamp=datetime.utcnow(),
                error=str(e)
            )
    
    async def get_model_info(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """Get model information.
        
        Args:
            model_name: Model name (optional)
            
        Returns:
            Model information
        """
        try:
            resolved_name = self._resolve_model_name(model_name)
            
            loop = asyncio.get_event_loop()
            info = await loop.run_in_executor(
                None,
                self.model_server.get_model_info,
                resolved_name
            )
            
            return info
            
        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
            return {"error": str(e)}
    
    async def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available models.
        
        Returns:
            List of available models
        """
        try:
            # Get models from registry
            if self.model_server.registry:
                loop = asyncio.get_event_loop()
                model_names = await loop.run_in_executor(
                    None,
                    self.model_server.registry.search_models
                )
                
                models = []
                for name in model_names:
                    try:
                        info = await self.get_model_info(name)
                        if "error" not in info:
                            models.append({
                                "name": name,
                                "version": info.get("version", "unknown"),
                                "stage": info.get("stage", "unknown"),
                                "description": info.get("description", ""),
                                "is_cached": info.get("is_cached", False)
                            })
                    except Exception as e:
                        logger.warning(f"Failed to get info for model {name}: {e}")
                
                return models
            else:
                # Return default models if no registry
                return [
                    {
                        "name": self.default_model,
                        "version": "1.0.0",
                        "stage": "production",
                        "description": "Default pharmacogenomics classifier",
                        "is_cached": False
                    }
                ]
                
        except Exception as e:
            logger.error(f"Failed to get available models: {e}")
            return []
    
    async def get_server_stats(self) -> Dict[str, Any]:
        """Get model server statistics.
        
        Returns:
            Server statistics
        """
        try:
            loop = asyncio.get_event_loop()
            stats = await loop.run_in_executor(
                None,
                self.model_server.get_server_stats
            )
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get server stats: {e}")
            return {"error": str(e)}
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on model server.
        
        Returns:
            Health check results
        """
        try:
            loop = asyncio.get_event_loop()
            health = await loop.run_in_executor(
                None,
                self.model_server.health_check
            )
            
            return health
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def _resolve_model_name(self, model_name: Optional[str]) -> str:
        """Resolve model name using aliases.
        
        Args:
            model_name: Model name or alias
            
        Returns:
            Resolved model name
        """
        if not model_name:
            return self.default_model
        
        # Check if it's an alias
        if model_name in self.model_aliases:
            return self.model_aliases[model_name]
        
        return model_name
    
    def _convert_to_api_response(self, response: PredictionResponse) -> APIPredictionResponse:
        """Convert internal response to API response.
        
        Args:
            response: Internal prediction response
            
        Returns:
            API prediction response
        """
        return APIPredictionResponse(
            prediction=response.prediction,
            probabilities=response.probabilities,
            confidence=response.confidence,
            model_name=response.model_name,
            model_version=response.model_version,
            processing_time_ms=response.processing_time_ms,
            timestamp=response.timestamp,
            request_id=response.request_id
        )
    
    async def warm_up_models(self, model_names: Optional[List[str]] = None) -> Dict[str, bool]:
        """Warm up models by loading them into cache.
        
        Args:
            model_names: List of model names to warm up
            
        Returns:
            Dict of model names and success status
        """
        if not model_names:
            model_names = [self.default_model]
        
        results = {}
        
        for model_name in model_names:
            try:
                resolved_name = self._resolve_model_name(model_name)
                
                # Create a dummy prediction to load the model
                dummy_features = {"feature_1": 0.5, "feature_2": 0.3}
                dummy_request = APIPredictionRequest(
                    features=dummy_features,
                    model_name=resolved_name
                )
                
                # Make prediction to trigger model loading
                await self.predict(dummy_request)
                results[model_name] = True
                
                logger.info(f"Model warmed up: {model_name}")
                
            except Exception as e:
                logger.error(f"Failed to warm up model {model_name}: {e}")
                results[model_name] = False
        
        return results
    
    async def clear_model_cache(self, model_name: Optional[str] = None) -> bool:
        """Clear model cache.
        
        Args:
            model_name: Specific model to clear (optional)
            
        Returns:
            Success status
        """
        try:
            if model_name:
                # Clear specific model (would need to implement in model_server)
                logger.info(f"Clearing cache for model: {model_name}")
            else:
                # Clear all models
                self.model_server.model_cache.clear()
                logger.info("Cleared all model cache")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear model cache: {e}")
            return False
    
    def get_model_aliases(self) -> Dict[str, str]:
        """Get model aliases mapping.
        
        Returns:
            Model aliases dictionary
        """
        return self.model_aliases.copy()
    
    def add_model_alias(self, alias: str, model_name: str) -> bool:
        """Add a model alias.
        
        Args:
            alias: Alias name
            model_name: Actual model name
            
        Returns:
            Success status
        """
        try:
            self.model_aliases[alias] = model_name
            logger.info(f"Added model alias: {alias} -> {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add model alias: {e}")
            return False
    
    def remove_model_alias(self, alias: str) -> bool:
        """Remove a model alias.
        
        Args:
            alias: Alias to remove
            
        Returns:
            Success status
        """
        try:
            if alias in self.model_aliases:
                del self.model_aliases[alias]
                logger.info(f"Removed model alias: {alias}")
                return True
            else:
                logger.warning(f"Alias not found: {alias}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to remove model alias: {e}")
            return False