"""Async HTTP client for NCBI API integration with connection pooling and session management."""

import asyncio
import ssl
from typing import Optional, Dict, Any, Union
from urllib.parse import urljoin

import aiohttp
from aiohttp import ClientTimeout, TCPConnector, ClientSession, ClientResponse

from ..config import get_config, get_logger


logger = get_logger(__name__)


class AsyncAPIClient:
    """Async HTTP client with connection pooling and NCBI-specific optimizations."""
    
    def __init__(self, 
                 base_url: str = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/",
                 timeout: float = 30.0,
                 max_connections: int = 100,
                 max_connections_per_host: int = 10,
                 user_agent: str = None,
                 email: str = None,
                 api_key: str = None,
                 tool: str = "genomics-pipeline"):
        """Initialize the async API client.
        
        Args:
            base_url: Base URL for API requests
            timeout: Request timeout in seconds
            max_connections: Maximum total connections in pool
            max_connections_per_host: Maximum connections per host
            user_agent: Custom User-Agent header
            email: Email for NCBI identification (required)
            api_key: NCBI API key for higher rate limits
            tool: Tool identifier for NCBI
        """
        self.base_url = base_url.rstrip('/') + '/'
        self.timeout = timeout
        self.max_connections = max_connections
        self.max_connections_per_host = max_connections_per_host
        self.email = email
        self.api_key = api_key
        self.tool = tool
        
        # Build User-Agent string
        if user_agent is None:
            self.user_agent = f"{tool}/1.0 (Python/aiohttp; {email or 'unknown'})"
        else:
            self.user_agent = user_agent
        
        # Session will be created in __aenter__
        self._session: Optional[ClientSession] = None
        self._connector: Optional[TCPConnector] = None
        self._closed = False
        
        logger.info(f"Initialized AsyncAPIClient for {self.base_url}")
    
    @classmethod
    def from_config(cls, config=None) -> 'AsyncAPIClient':
        """Create client from configuration.
        
        Args:
            config: Configuration object. If None, loads from global config.
            
        Returns:
            AsyncAPIClient: Configured client instance
        """
        if config is None:
            app_config = get_config()
            config = app_config.ncbi
        
        return cls(
            base_url=getattr(config, 'base_url', "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"),
            timeout=getattr(config, 'timeout', 30.0),
            max_connections=getattr(config, 'max_connections', 100),
            max_connections_per_host=getattr(config, 'max_connections_per_host', 10),
            email=getattr(config, 'email', None),
            api_key=getattr(config, 'api_key', None),
            tool=getattr(config, 'tool', 'genomics-pipeline')
        )
    
    async def __aenter__(self) -> 'AsyncAPIClient':
        """Async context manager entry."""
        if self._session is not None:
            return self
        
        # Create SSL context for secure connections
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = True
        ssl_context.verify_mode = ssl.CERT_REQUIRED
        
        # Create TCP connector with connection pooling
        self._connector = TCPConnector(
            limit=self.max_connections,
            limit_per_host=self.max_connections_per_host,
            ssl=ssl_context,
            enable_cleanup_closed=True,
            keepalive_timeout=30,
            ttl_dns_cache=300,  # DNS cache TTL
            use_dns_cache=True
        )
        
        # Create timeout configuration
        timeout = ClientTimeout(total=self.timeout)
        
        # Default headers
        headers = {
            'User-Agent': self.user_agent,
            'Accept': 'application/json, application/xml, text/xml',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive'
        }
        
        # Create session
        self._session = ClientSession(
            connector=self._connector,
            timeout=timeout,
            headers=headers,
            raise_for_status=False  # We'll handle status codes manually
        )
        
        self._closed = False
        logger.info("AsyncAPIClient session initialized")
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with proper cleanup."""
        await self.close()
    
    async def close(self):
        """Close the client session and cleanup resources."""
        if self._closed:
            return
        
        if self._session:
            await self._session.close()
            self._session = None
        
        if self._connector:
            await self._connector.close()
            self._connector = None
        
        self._closed = True
        logger.info("AsyncAPIClient session closed")
    
    def _ensure_session(self):
        """Ensure session is available."""
        if self._session is None or self._closed:
            raise RuntimeError("AsyncAPIClient not initialized. Use 'async with' context manager.")
    
    def _build_url(self, endpoint: str) -> str:
        """Build full URL from endpoint.
        
        Args:
            endpoint: API endpoint path
            
        Returns:
            str: Full URL
        """
        if endpoint.startswith('http'):
            return endpoint
        return urljoin(self.base_url, endpoint.lstrip('/'))
    
    def _add_ncbi_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Add NCBI-required parameters to request.
        
        Args:
            params: Existing parameters
            
        Returns:
            Dict[str, Any]: Parameters with NCBI requirements added
        """
        if params is None:
            params = {}
        else:
            params = params.copy()
        
        # Add required NCBI parameters
        if self.email:
            params['email'] = self.email
        
        if self.api_key:
            params['api_key'] = self.api_key
        
        if self.tool:
            params['tool'] = self.tool
        
        return params
    
    async def get(self, 
                  url: str, 
                  params: Optional[Dict[str, Any]] = None,
                  **kwargs) -> ClientResponse:
        """Perform GET request.
        
        Args:
            url: URL or endpoint path
            params: Query parameters
            **kwargs: Additional aiohttp parameters
            
        Returns:
            ClientResponse: HTTP response
        """
        self._ensure_session()
        
        full_url = self._build_url(url)
        params = self._add_ncbi_params(params)
        
        logger.debug(f"GET request to {full_url} with params: {params}")
        
        try:
            response = await self._session.get(full_url, params=params, **kwargs)
            logger.debug(f"GET response: {response.status} from {full_url}")
            return response
        except Exception as e:
            logger.error(f"GET request failed for {full_url}: {str(e)}")
            raise
    
    async def post(self, 
                   url: str, 
                   data: Optional[Union[Dict[str, Any], str]] = None,
                   params: Optional[Dict[str, Any]] = None,
                   **kwargs) -> ClientResponse:
        """Perform POST request.
        
        Args:
            url: URL or endpoint path
            data: Request body data
            params: Query parameters
            **kwargs: Additional aiohttp parameters
            
        Returns:
            ClientResponse: HTTP response
        """
        self._ensure_session()
        
        full_url = self._build_url(url)
        params = self._add_ncbi_params(params)
        
        logger.debug(f"POST request to {full_url} with params: {params}")
        
        try:
            response = await self._session.post(full_url, data=data, params=params, **kwargs)
            logger.debug(f"POST response: {response.status} from {full_url}")
            return response
        except Exception as e:
            logger.error(f"POST request failed for {full_url}: {str(e)}")
            raise
    
    def get_session_info(self) -> Dict[str, Any]:
        """Get information about the current session.
        
        Returns:
            Dict[str, Any]: Session information and statistics
        """
        if self._session is None or self._closed:
            return {
                'status': 'closed',
                'base_url': self.base_url,
                'timeout': self.timeout
            }
        
        connector_info = {}
        if self._connector:
            connector_info = {
                'total_connections': len(self._connector._conns),
                'available_connections': len([c for c in self._connector._conns.values() if c.available]),
                'max_connections': self.max_connections,
                'max_connections_per_host': self.max_connections_per_host
            }
        
        return {
            'status': 'active',
            'base_url': self.base_url,
            'timeout': self.timeout,
            'user_agent': self.user_agent,
            'email': self.email,
            'has_api_key': bool(self.api_key),
            'tool': self.tool,
            'connector': connector_info,
            'session_closed': self._session.closed if self._session else True
        }
    
    @property
    def is_closed(self) -> bool:
        """Check if the client is closed.
        
        Returns:
            bool: True if client is closed
        """
        return self._closed or (self._session is not None and self._session.closed)