"""NCBI API client with rate limiting, error handling, and retry logic."""

import asyncio
import json
import xml.etree.ElementTree as ET
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
from enum import Enum

import aiohttp

from ..config import get_config, get_logger
from .async_api_client import AsyncAPIClient
from .rate_limiter import AdaptiveRateLimiter


logger = get_logger(__name__)


class NCBIDatabase(Enum):
    """NCBI database identifiers."""
    PUBMED = "pubmed"
    PMC = "pmc"
    CLINVAR = "clinvar"
    DBSNP = "snp"
    GENE = "gene"
    PROTEIN = "protein"


class ResponseFormat(Enum):
    """Response format options."""
    JSON = "json"
    XML = "xml"


@dataclass
class NCBIResponse:
    """Structured response from NCBI API."""
    success: bool
    data: Optional[Union[Dict[str, Any], str]]
    error_message: Optional[str] = None
    status_code: Optional[int] = None
    response_format: Optional[ResponseFormat] = None
    retry_count: int = 0


class NCBIAPIError(Exception):
    """Custom exception for NCBI API errors."""
    
    def __init__(self, message: str, status_code: Optional[int] = None, response_data: Optional[str] = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_data = response_data


class NCBIClient:
    """High-level NCBI API client with rate limiting and error handling."""
    
    def __init__(self,
                 email: str,
                 api_key: Optional[str] = None,
                 tool: str = "genomics-pipeline",
                 max_retries: int = 3,
                 timeout: float = 30.0):
        """Initialize NCBI client.
        
        Args:
            email: Email address for NCBI identification (required)
            api_key: NCBI API key for higher rate limits
            tool: Tool identifier
            max_retries: Maximum retry attempts
            timeout: Request timeout in seconds
        """
        if not email:
            raise ValueError("Email is required for NCBI API access")
        
        self.email = email
        self.api_key = api_key
        self.tool = tool
        self.max_retries = max_retries
        self.timeout = timeout
        
        # Initialize HTTP client
        self.http_client = AsyncAPIClient(
            email=email,
            api_key=api_key,
            tool=tool,
            timeout=timeout
        )
        
        # Initialize rate limiter
        self.rate_limiter = AdaptiveRateLimiter(
            has_api_key=bool(api_key),
            default_rate=3.0,
            api_key_rate=10.0
        )
        
        self._session_active = False
        
        logger.info(f"Initialized NCBIClient for {email} with {'API key' if api_key else 'no API key'}")
    
    @classmethod
    def from_config(cls, config=None) -> 'NCBIClient':
        """Create client from configuration.
        
        Args:
            config: Configuration object. If None, loads from global config.
            
        Returns:
            NCBIClient: Configured client instance
        """
        if config is None:
            app_config = get_config()
            config = app_config.ncbi
        
        return cls(
            email=getattr(config, 'email', None),
            api_key=getattr(config, 'api_key', None),
            tool=getattr(config, 'tool', 'genomics-pipeline'),
            max_retries=getattr(config, 'max_retries', 3),
            timeout=getattr(config, 'timeout', 30.0)
        )
    
    async def __aenter__(self) -> 'NCBIClient':
        """Async context manager entry."""
        await self.http_client.__aenter__()
        self._session_active = True
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.http_client.__aexit__(exc_type, exc_val, exc_tb)
        self._session_active = False
    
    def _ensure_session(self):
        """Ensure session is active."""
        if not self._session_active:
            raise RuntimeError("NCBIClient not initialized. Use 'async with' context manager.")
    
    async def _make_request_with_retry(self,
                                     method: str,
                                     endpoint: str,
                                     params: Optional[Dict[str, Any]] = None,
                                     data: Optional[Union[Dict[str, Any], str]] = None) -> NCBIResponse:
        """Make HTTP request with retry logic and error handling.
        
        Args:
            method: HTTP method (GET or POST)
            endpoint: API endpoint
            params: Query parameters
            data: Request body data (for POST)
            
        Returns:
            NCBIResponse: Structured response
        """
        self._ensure_session()
        
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                # Apply rate limiting
                await self.rate_limiter.acquire()
                
                # Make the request
                if method.upper() == 'GET':
                    response = await self.http_client.get(endpoint, params=params)
                elif method.upper() == 'POST':
                    response = await self.http_client.post(endpoint, data=data, params=params)
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")
                
                # Handle response
                return await self._handle_response(response, attempt)
                
            except aiohttp.ClientError as e:
                last_exception = e
                logger.warning(f"Request attempt {attempt + 1} failed: {str(e)}")
                
                # Record error for rate limiting
                self.rate_limiter.record_error(0)  # Network error
                
                if attempt < self.max_retries:
                    # Exponential backoff with jitter
                    wait_time = (2 ** attempt) + (0.1 * attempt)
                    await asyncio.sleep(wait_time)
                else:
                    break
            
            except Exception as e:
                last_exception = e
                logger.error(f"Unexpected error in request attempt {attempt + 1}: {str(e)}")
                break
        
        # All retries failed
        error_msg = f"Request failed after {self.max_retries + 1} attempts"
        if last_exception:
            error_msg += f": {str(last_exception)}"
        
        return NCBIResponse(
            success=False,
            data=None,
            error_message=error_msg,
            retry_count=self.max_retries + 1
        )
    
    async def _handle_response(self, response: aiohttp.ClientResponse, attempt: int) -> NCBIResponse:
        """Handle HTTP response with proper error handling.
        
        Args:
            response: HTTP response
            attempt: Current attempt number
            
        Returns:
            NCBIResponse: Structured response
        """
        status_code = response.status
        
        # Handle rate limiting
        if status_code == 429:
            self.rate_limiter.record_error(status_code)
            
            # Check for Retry-After header
            retry_after = response.headers.get('Retry-After')
            if retry_after:
                try:
                    await self.rate_limiter.wait_for_reset(int(retry_after))
                except ValueError:
                    # Retry-After is not a number, use default backoff
                    await asyncio.sleep(2 ** attempt)
            
            raise aiohttp.ClientResponseError(
                request_info=response.request_info,
                history=response.history,
                status=status_code,
                message="Rate limited"
            )
        
        # Handle client errors (4xx)
        if 400 <= status_code < 500:
            error_text = await response.text()
            self.rate_limiter.record_error(status_code)
            
            return NCBIResponse(
                success=False,
                data=None,
                error_message=f"Client error {status_code}: {error_text}",
                status_code=status_code,
                retry_count=attempt
            )
        
        # Handle server errors (5xx) - these should be retried
        if status_code >= 500:
            error_text = await response.text()
            self.rate_limiter.record_error(status_code)
            
            raise aiohttp.ClientResponseError(
                request_info=response.request_info,
                history=response.history,
                status=status_code,
                message=f"Server error: {error_text}"
            )
        
        # Success response
        if 200 <= status_code < 300:
            self.rate_limiter.record_success()
            
            # Determine response format
            content_type = response.headers.get('Content-Type', '').lower()
            
            try:
                if 'json' in content_type:
                    data = await response.json()
                    response_format = ResponseFormat.JSON
                elif 'xml' in content_type or 'text' in content_type:
                    data = await response.text()
                    response_format = ResponseFormat.XML
                else:
                    # Default to text
                    data = await response.text()
                    response_format = ResponseFormat.XML
                
                return NCBIResponse(
                    success=True,
                    data=data,
                    status_code=status_code,
                    response_format=response_format,
                    retry_count=attempt
                )
                
            except Exception as e:
                logger.error(f"Failed to parse response: {str(e)}")
                return NCBIResponse(
                    success=False,
                    data=None,
                    error_message=f"Failed to parse response: {str(e)}",
                    status_code=status_code,
                    retry_count=attempt
                )
        
        # Unexpected status code
        error_text = await response.text()
        return NCBIResponse(
            success=False,
            data=None,
            error_message=f"Unexpected status code {status_code}: {error_text}",
            status_code=status_code,
            retry_count=attempt
        )
    
    async def search(self,
                    database: NCBIDatabase,
                    query: str,
                    max_results: int = 100,
                    start: int = 0) -> NCBIResponse:
        """Search NCBI database using esearch.
        
        Args:
            database: NCBI database to search
            query: Search query
            max_results: Maximum number of results
            start: Starting position for results
            
        Returns:
            NCBIResponse: Search results
        """
        params = {
            'db': database.value,
            'term': query,
            'retmax': str(max_results),
            'retstart': str(start),
            'retmode': 'json'
        }
        
        logger.info(f"Searching {database.value} for: {query} (max={max_results}, start={start})")
        
        return await self._make_request_with_retry('GET', 'esearch.fcgi', params=params)
    
    async def fetch(self,
                   database: NCBIDatabase,
                   ids: List[str],
                   return_type: str = 'abstract',
                   return_mode: str = 'xml') -> NCBIResponse:
        """Fetch records from NCBI database using efetch.
        
        Args:
            database: NCBI database
            ids: List of record IDs to fetch
            return_type: Type of data to return
            return_mode: Format of returned data
            
        Returns:
            NCBIResponse: Fetched records
        """
        if not ids:
            return NCBIResponse(
                success=False,
                data=None,
                error_message="No IDs provided for fetch"
            )
        
        params = {
            'db': database.value,
            'id': ','.join(ids),
            'rettype': return_type,
            'retmode': return_mode
        }
        
        logger.info(f"Fetching {len(ids)} records from {database.value}")
        
        return await self._make_request_with_retry('GET', 'efetch.fcgi', params=params)
    
    async def get_database_info(self, database: NCBIDatabase) -> NCBIResponse:
        """Get information about an NCBI database using einfo.
        
        Args:
            database: NCBI database
            
        Returns:
            NCBIResponse: Database information
        """
        params = {
            'db': database.value,
            'retmode': 'json'
        }
        
        logger.info(f"Getting info for database: {database.value}")
        
        return await self._make_request_with_retry('GET', 'einfo.fcgi', params=params)
    
    def get_client_status(self) -> Dict[str, Any]:
        """Get comprehensive client status and metrics.
        
        Returns:
            Dict[str, Any]: Client status information
        """
        return {
            'email': self.email,
            'has_api_key': bool(self.api_key),
            'tool': self.tool,
            'max_retries': self.max_retries,
            'timeout': self.timeout,
            'session_active': self._session_active,
            'http_client': self.http_client.get_session_info(),
            'rate_limiter': self.rate_limiter.get_metrics()
        }