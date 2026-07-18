"""FastAPI application for pharmacogenomics ML platform with async request handling and middleware."""

import asyncio
import os
import time
from datetime import datetime
from typing import Dict, Any, Optional
import logging

# FastAPI imports
from fastapi import FastAPI, Request, Response, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
import uvicorn

# Monitoring and metrics
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
import structlog

# Internal imports
from ..config import get_logger, get_config
from .auth import AuthManager, get_current_user
from .rate_limiter import RateLimiter
from .monitoring import APIMonitor
from .schemas import HealthResponse, ErrorResponse

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = get_logger(__name__)

# Metrics
REQUEST_COUNT = Counter('api_requests_total', 'Total API requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('api_request_duration_seconds', 'Request duration', ['method', 'endpoint'])
PREDICTION_COUNT = Counter('predictions_total', 'Total predictions made', ['model', 'status'])


class PharmacogenomicsAPI:
    """Main FastAPI application for pharmacogenomics platform."""
    
    def __init__(self):
        """Initialize the FastAPI application."""
        self.app = FastAPI(
            title="Pharmacogenomics ML Platform API",
            description="High-performance API for pharmacogenomics machine learning and clinical decision support",
            version="1.0.0",
            docs_url="/docs",
            redoc_url="/redoc",
            openapi_url="/openapi.json"
        )
        
        # Initialize components
        self.config = get_config()
        self.auth_manager = AuthManager()
        self.rate_limiter = RateLimiter()
        self.monitor = APIMonitor()
        
        # Initialize ML components. These pull in heavy native ML dependencies
        # (torch/MLflow) that are not needed by the graph explorer or the core
        # API, and whose import can crash on some platforms. They are therefore
        # opt-in: set PHARMGRAPH_ENABLE_ML=1 to load the model-serving stack.
        # The import is done lazily here so the module imports cleanly without it.
        self.model_registry = None
        self.model_server = None
        self.model_service = None
        if os.getenv("PHARMGRAPH_ENABLE_ML", "").lower() in ("1", "true", "yes"):
            try:
                from ..ml.model_server import ModelServer
                from ..ml.model_registry import MLflowModelRegistry
                from .model_service import ModelService

                self.model_registry = MLflowModelRegistry()
                self.model_server = ModelServer(self.model_registry)
                self.model_service = ModelService(self.model_server)
                logger.info("ML components initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize ML components: {e}")
                self.model_registry = None
                self.model_server = None
                self.model_service = None
        else:
            logger.info("ML components disabled (set PHARMGRAPH_ENABLE_ML=1 to enable)")
        
        # Setup middleware and routes
        self._setup_middleware()
        self._setup_routes()
        self._setup_exception_handlers()
        
        logger.info("Pharmacogenomics API initialized")
    
    def _setup_middleware(self):
        """Setup FastAPI middleware."""
        
        # CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=self.config.api.cors_origins,
            allow_credentials=True,
            allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            allow_headers=["*"],
        )

        # Trusted host middleware
        allowed_hosts = self.config.api.allowed_hosts
        if allowed_hosts != ["*"]:
            self.app.add_middleware(
                TrustedHostMiddleware,
                allowed_hosts=allowed_hosts
            )
        
        # Custom middleware for logging and metrics
        @self.app.middleware("http")
        async def logging_middleware(request: Request, call_next):
            """Log requests and collect metrics."""
            start_time = time.time()
            
            # Extract request info
            method = request.method
            url = str(request.url)
            client_ip = request.client.host if request.client else "unknown"
            user_agent = request.headers.get("user-agent", "unknown")
            
            # Create request ID
            request_id = f"{int(start_time * 1000)}-{hash(f'{client_ip}{url}') % 10000:04d}"
            
            # Add to request state
            request.state.request_id = request_id
            request.state.start_time = start_time
            
            # Log request
            logger.info(
                "Request started",
                extra={
                    "request_id": request_id,
                    "method": method,
                    "url": url,
                    "client_ip": client_ip,
                    "user_agent": user_agent,
                }
            )
            
            try:
                # Process request
                response = await call_next(request)
                
                # Calculate duration
                duration = time.time() - start_time
                
                # Update metrics
                endpoint = request.url.path
                REQUEST_COUNT.labels(method=method, endpoint=endpoint, status=response.status_code).inc()
                REQUEST_DURATION.labels(method=method, endpoint=endpoint).observe(duration)
                
                # Log response
                logger.info(
                    "Request completed",
                    extra={
                        "request_id": request_id,
                        "status_code": response.status_code,
                        "duration_ms": round(duration * 1000, 2),
                    }
                )
                
                # Add response headers
                response.headers["X-Request-ID"] = request_id
                response.headers["X-Response-Time"] = f"{duration:.3f}s"
                
                return response
                
            except Exception as e:
                duration = time.time() - start_time
                
                # Update error metrics
                endpoint = request.url.path
                REQUEST_COUNT.labels(method=method, endpoint=endpoint, status=500).inc()
                REQUEST_DURATION.labels(method=method, endpoint=endpoint).observe(duration)
                
                # Log error
                logger.error(
                    "Request failed",
                    extra={
                        "request_id": request_id,
                        "error": str(e),
                        "duration_ms": round(duration * 1000, 2),
                    },
                    exc_info=True
                )
                
                raise
        
        # Security headers middleware
        @self.app.middleware("http")
        async def security_headers_middleware(request: Request, call_next):
            """Add security headers to responses."""
            response = await call_next(request)
            
            # Security headers
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["X-Frame-Options"] = "DENY"
            response.headers["X-XSS-Protection"] = "1; mode=block"
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
            response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
            
            return response
    
    def _setup_routes(self):
        """Setup API routes."""
        
        @self.app.get("/", tags=["Root"])
        async def root():
            """Root endpoint."""
            return {
                "message": "Pharmacogenomics ML Platform API",
                "version": "1.0.0",
                "status": "operational",
                "timestamp": datetime.utcnow().isoformat()
            }
        
        @self.app.get("/health", response_model=HealthResponse, tags=["Health"])
        async def health_check():
            """Health check endpoint."""
            try:
                # Check ML components
                ml_status = "healthy"
                if self.model_server:
                    health = self.model_server.health_check()
                    ml_status = health.get("status", "unhealthy")
                
                # Check database (if available)
                db_status = "healthy"  # Would check actual database
                
                # Overall status
                overall_status = "healthy" if all([
                    ml_status == "healthy",
                    db_status == "healthy"
                ]) else "unhealthy"
                
                return HealthResponse(
                    status=overall_status,
                    timestamp=datetime.utcnow(),
                    components={
                        "ml_server": ml_status,
                        "database": db_status,
                        "api": "healthy"
                    },
                    version="1.0.0"
                )
                
            except Exception as e:
                logger.error(f"Health check failed: {e}")
                return HealthResponse(
                    status="unhealthy",
                    timestamp=datetime.utcnow(),
                    components={
                        "ml_server": "unhealthy",
                        "database": "unknown",
                        "api": "degraded"
                    },
                    version="1.0.0",
                    error=str(e)
                )
        
        @self.app.get("/status", tags=["Health"])
        async def status():
            """Detailed status endpoint."""
            try:
                status_info = {
                    "api": {
                        "status": "operational",
                        "uptime_seconds": time.time() - getattr(self, '_start_time', time.time()),
                        "version": "1.0.0"
                    },
                    "ml_server": {},
                    "metrics": {
                        "total_requests": sum([
                            metric.samples[0].value for metric in REQUEST_COUNT.collect()[0].samples
                        ]) if REQUEST_COUNT.collect() else 0
                    }
                }
                
                # Add ML server status
                if self.model_server:
                    ml_stats = self.model_server.get_server_stats()
                    status_info["ml_server"] = {
                        "status": "operational",
                        "cached_models": ml_stats.get("model_cache", {}).get("cached_models", 0),
                        "success_rate": ml_stats.get("success_rate", 0.0),
                        "average_latency_ms": ml_stats.get("average_processing_time_ms", 0.0)
                    }
                else:
                    status_info["ml_server"] = {"status": "not_available"}
                
                return status_info
                
            except Exception as e:
                logger.error(f"Status check failed: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Status check failed"
                )
        
        @self.app.get("/metrics", tags=["Monitoring"])
        async def metrics():
            """Prometheus metrics endpoint."""
            return Response(
                content=generate_latest(),
                media_type=CONTENT_TYPE_LATEST
            )
        
        # Include API routers
        self._include_api_routers()
    
    def _include_api_routers(self):
        """Include API routers for different endpoints."""
        
        # Model serving endpoints
        if self.model_service:
            from .endpoints import predictions
            self.app.include_router(
                predictions.router,
                prefix="/api/v1/predictions",
                tags=["Predictions"],
                dependencies=[Depends(self.rate_limiter.check_rate_limit)]
            )
        
        # Clinical decision support endpoints
        try:
            from .endpoints import drug_gene, risk_scoring, dosing, clinical_support
            
            self.app.include_router(
                drug_gene.router,
                prefix="/api/v1/drug-gene",
                tags=["Drug-Gene Interactions"],
                dependencies=[Depends(self.rate_limiter.check_rate_limit)]
            )
            
            self.app.include_router(
                risk_scoring.router,
                prefix="/api/v1/risk-scoring",
                tags=["Risk Scoring"],
                dependencies=[Depends(self.rate_limiter.check_rate_limit)]
            )
            
            self.app.include_router(
                dosing.router,
                prefix="/api/v1/dosing",
                tags=["Personalized Dosing"],
                dependencies=[Depends(self.rate_limiter.check_rate_limit)]
            )
            
            self.app.include_router(
                clinical_support.router,
                prefix="/api/v1/clinical-support",
                tags=["Clinical Decision Support"],
                dependencies=[Depends(self.rate_limiter.check_rate_limit)]
            )
            
        except ImportError as e:
            logger.warning(f"Some clinical endpoints not available: {e}")

        # Gene/protein/drug interaction graph explorer
        from .endpoints import graph as graph_endpoints
        self.app.include_router(
            graph_endpoints.router,
            prefix="/api/v1/graph",
            tags=["Graph Explorer"],
            dependencies=[Depends(self.rate_limiter.check_rate_limit)]
        )

        # Admin endpoints (protected)
        try:
            from .endpoints import admin
            self.app.include_router(
                admin.router,
                prefix="/api/v1/admin",
                tags=["Administration"],
                dependencies=[Depends(get_current_user)]
            )
        except ImportError:
            logger.warning("Admin endpoints not available")
    
    def _setup_exception_handlers(self):
        """Setup custom exception handlers."""
        
        @self.app.exception_handler(HTTPException)
        async def http_exception_handler(request: Request, exc: HTTPException):
            """Handle HTTP exceptions."""
            request_id = getattr(request.state, 'request_id', 'unknown')
            
            logger.warning(
                "HTTP exception",
                extra={
                    "request_id": request_id,
                    "status_code": exc.status_code,
                    "detail": exc.detail,
                }
            )
            
            return JSONResponse(
                status_code=exc.status_code,
                content=ErrorResponse(
                    error="HTTP_ERROR",
                    message=exc.detail,
                    status_code=exc.status_code,
                    request_id=request_id,
                    timestamp=datetime.utcnow()
                ).dict()
            )
        
        @self.app.exception_handler(ValueError)
        async def value_error_handler(request: Request, exc: ValueError):
            """Handle value errors."""
            request_id = getattr(request.state, 'request_id', 'unknown')
            
            logger.error(
                "Value error",
                extra={"request_id": request_id, "error": str(exc)},
                exc_info=True
            )
            
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content=ErrorResponse(
                    error="VALIDATION_ERROR",
                    message=str(exc),
                    status_code=400,
                    request_id=request_id,
                    timestamp=datetime.utcnow()
                ).dict()
            )
        
        @self.app.exception_handler(Exception)
        async def general_exception_handler(request: Request, exc: Exception):
            """Handle general exceptions."""
            request_id = getattr(request.state, 'request_id', 'unknown')
            
            logger.error(
                "Unhandled exception",
                extra={"request_id": request_id, "error": str(exc)},
                exc_info=True
            )
            
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content=ErrorResponse(
                    error="INTERNAL_ERROR",
                    message="An internal error occurred",
                    status_code=500,
                    request_id=request_id,
                    timestamp=datetime.utcnow()
                ).dict()
            )
    
    def get_app(self) -> FastAPI:
        """Get the FastAPI application instance."""
        return self.app


# Global application instance
api = PharmacogenomicsAPI()
app = api.get_app()


# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Application startup event."""
    api._start_time = time.time()
    logger.info("Pharmacogenomics API started")
    
    # Initialize background tasks if needed
    # asyncio.create_task(background_task())


@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown event."""
    logger.info("Pharmacogenomics API shutting down")
    
    # Cleanup resources
    if api.model_server:
        api.model_server.shutdown()


# Development server
def run_dev_server():
    """Run development server."""
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
        access_log=True
    )


if __name__ == "__main__":
    run_dev_server()