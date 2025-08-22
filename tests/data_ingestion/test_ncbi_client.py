"""Tests for NCBIClient."""

import pytest
import asyncio
import json
from unittest.mock import Mock, patch, AsyncMock
from aiohttp import ClientResponseError

from src.data_ingestion.ncbi_client import (
    NCBIClient, NCBIDatabase, ResponseFormat, NCBIResponse, NCBIAPIError
)


class TestNCBIResponse:
    """Test cases for NCBIResponse dataclass."""
    
    def test_init_success(self):
        """Test successful response initialization."""
        response = NCBIResponse(
            success=True,
            data={"test": "data"},
            status_code=200,
            response_format=ResponseFormat.JSON
        )
        
        assert response.success is True
        assert response.data == {"test": "data"}
        assert response.error_message is None
        assert response.status_code == 200
        assert response.response_format == ResponseFormat.JSON
        assert response.retry_count == 0
    
    def test_init_error(self):
        """Test error response initialization."""
        response = NCBIResponse(
            success=False,
            data=None,
            error_message="Test error",
            status_code=400,
            retry_count=2
        )
        
        assert response.success is False
        assert response.data is None
        assert response.error_message == "Test error"
        assert response.status_code == 400
        assert response.response_format is None
        assert response.retry_count == 2


class TestNCBIAPIError:
    """Test cases for NCBIAPIError exception."""
    
    def test_init_basic(self):
        """Test basic error initialization."""
        error = NCBIAPIError("Test error message")
        
        assert str(error) == "Test error message"
        assert error.status_code is None
        assert error.response_data is None
    
    def test_init_with_details(self):
        """Test error initialization with details."""
        error = NCBIAPIError(
            "Test error message",
            status_code=400,
            response_data="Error response"
        )
        
        assert str(error) == "Test error message"
        assert error.status_code == 400
        assert error.response_data == "Error response"


class TestNCBIClient:
    """Test cases for NCBIClient."""
    
    def test_init_required_email(self):
        """Test that email is required."""
        with pytest.raises(ValueError, match="Email is required"):
            NCBIClient(email="")
        
        with pytest.raises(ValueError, match="Email is required"):
            NCBIClient(email=None)
    
    def test_init_default_values(self):
        """Test initialization with default values."""
        client = NCBIClient(email="test@example.com")
        
        assert client.email == "test@example.com"
        assert client.api_key is None
        assert client.tool == "genomics-pipeline"
        assert client.max_retries == 3
        assert client.timeout == 30.0
        assert client._session_active is False
        assert client.rate_limiter.has_api_key is False
        assert client.rate_limiter.current_rate == 3.0  # Default rate
    
    def test_init_with_api_key(self):
        """Test initialization with API key."""
        client = NCBIClient(
            email="test@example.com",
            api_key="test-key",
            tool="custom-tool",
            max_retries=5,
            timeout=60.0
        )
        
        assert client.email == "test@example.com"
        assert client.api_key == "test-key"
        assert client.tool == "custom-tool"
        assert client.max_retries == 5
        assert client.timeout == 60.0
        assert client.rate_limiter.has_api_key is True
        assert client.rate_limiter.current_rate == 10.0  # API key rate
    
    @patch('src.data_ingestion.ncbi_client.get_config')
    def test_from_config(self, mock_get_config):
        """Test creating client from configuration."""
        mock_config = Mock()
        mock_config.ncbi.email = "config@example.com"
        mock_config.ncbi.api_key = "config-key"
        mock_config.ncbi.tool = "config-tool"
        mock_config.ncbi.max_retries = 4
        mock_config.ncbi.timeout = 45.0
        mock_get_config.return_value = mock_config
        
        client = NCBIClient.from_config()
        
        assert client.email == "config@example.com"
        assert client.api_key == "config-key"
        assert client.tool == "config-tool"
        assert client.max_retries == 4
        assert client.timeout == 45.0
    
    @pytest.mark.asyncio
    async def test_context_manager_lifecycle(self):
        """Test async context manager lifecycle."""
        client = NCBIClient(email="test@example.com")
        
        assert client._session_active is False
        
        async with client as ctx_client:
            assert ctx_client is client
            assert client._session_active is True
        
        assert client._session_active is False
    
    def test_ensure_session_not_active(self):
        """Test _ensure_session raises error when session not active."""
        client = NCBIClient(email="test@example.com")
        
        with pytest.raises(RuntimeError, match="NCBIClient not initialized"):
            client._ensure_session()
    
    @pytest.mark.asyncio
    async def test_make_request_success(self):
        """Test successful request with retry logic."""
        client = NCBIClient(email="test@example.com")
        
        # Mock HTTP client response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.headers = {'Content-Type': 'application/json'}
        mock_response.json.return_value = {"test": "data"}
        
        with patch.object(client.http_client, 'get', return_value=mock_response):
            with patch.object(client.rate_limiter, 'acquire'):
                with patch.object(client.rate_limiter, 'record_success'):
                    client._session_active = True
                    
                    response = await client._make_request_with_retry(
                        'GET', 'test-endpoint', params={'q': 'test'}
                    )
                    
                    assert response.success is True
                    assert response.data == {"test": "data"}
                    assert response.status_code == 200
                    assert response.response_format == ResponseFormat.JSON
    
    @pytest.mark.asyncio
    async def test_make_request_xml_response(self):
        """Test request with XML response."""
        client = NCBIClient(email="test@example.com")
        
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.headers = {'Content-Type': 'application/xml'}
        mock_response.text.return_value = "<xml>test</xml>"
        
        with patch.object(client.http_client, 'get', return_value=mock_response):
            with patch.object(client.rate_limiter, 'acquire'):
                with patch.object(client.rate_limiter, 'record_success'):
                    client._session_active = True
                    
                    response = await client._make_request_with_retry(
                        'GET', 'test-endpoint'
                    )
                    
                    assert response.success is True
                    assert response.data == "<xml>test</xml>"
                    assert response.response_format == ResponseFormat.XML
    
    @pytest.mark.asyncio
    async def test_make_request_rate_limited(self):
        """Test request handling rate limiting (429)."""
        client = NCBIClient(email="test@example.com", max_retries=1)
        
        # First response: rate limited
        mock_response_429 = AsyncMock()
        mock_response_429.status = 429
        mock_response_429.headers = {'Retry-After': '2'}
        mock_response_429.request_info = Mock()
        mock_response_429.history = []
        
        # Second response: success
        mock_response_200 = AsyncMock()
        mock_response_200.status = 200
        mock_response_200.headers = {'Content-Type': 'application/json'}
        mock_response_200.json.return_value = {"success": True}
        
        with patch.object(client.http_client, 'get', side_effect=[mock_response_429, mock_response_200]):
            with patch.object(client.rate_limiter, 'acquire'):
                with patch.object(client.rate_limiter, 'record_error'):
                    with patch.object(client.rate_limiter, 'wait_for_reset') as mock_wait:
                        with patch.object(client.rate_limiter, 'record_success'):
                            client._session_active = True
                            
                            response = await client._make_request_with_retry(
                                'GET', 'test-endpoint'
                            )
                            
                            assert response.success is True
                            assert response.data == {"success": True}
                            mock_wait.assert_called_once_with(2)
    
    @pytest.mark.asyncio
    async def test_make_request_client_error(self):
        """Test request handling client errors (4xx)."""
        client = NCBIClient(email="test@example.com")
        
        mock_response = AsyncMock()
        mock_response.status = 400
        mock_response.text.return_value = "Bad Request"
        
        with patch.object(client.http_client, 'get', return_value=mock_response):
            with patch.object(client.rate_limiter, 'acquire'):
                with patch.object(client.rate_limiter, 'record_error'):
                    client._session_active = True
                    
                    response = await client._make_request_with_retry(
                        'GET', 'test-endpoint'
                    )
                    
                    assert response.success is False
                    assert response.status_code == 400
                    assert "Bad Request" in response.error_message
    
    @pytest.mark.asyncio
    async def test_make_request_server_error_retry(self):
        """Test request retrying on server errors (5xx)."""
        client = NCBIClient(email="test@example.com", max_retries=2)
        
        # All responses are server errors
        mock_response = AsyncMock()
        mock_response.status = 500
        mock_response.text.return_value = "Internal Server Error"
        mock_response.request_info = Mock()
        mock_response.history = []
        
        with patch.object(client.http_client, 'get', return_value=mock_response):
            with patch.object(client.rate_limiter, 'acquire'):
                with patch.object(client.rate_limiter, 'record_error'):
                    with patch('asyncio.sleep'):  # Speed up test
                        client._session_active = True
                        
                        response = await client._make_request_with_retry(
                            'GET', 'test-endpoint'
                        )
                        
                        assert response.success is False
                        assert "Request failed after 3 attempts" in response.error_message
                        assert response.retry_count == 3
    
    @pytest.mark.asyncio
    async def test_make_request_network_error(self):
        """Test request handling network errors."""
        client = NCBIClient(email="test@example.com", max_retries=1)
        
        with patch.object(client.http_client, 'get', side_effect=Exception("Network error")):
            with patch.object(client.rate_limiter, 'acquire'):
                with patch.object(client.rate_limiter, 'record_error'):
                    with patch('asyncio.sleep'):  # Speed up test
                        client._session_active = True
                        
                        response = await client._make_request_with_retry(
                            'GET', 'test-endpoint'
                        )
                        
                        assert response.success is False
                        assert "Network error" in response.error_message
    
    @pytest.mark.asyncio
    async def test_search_pubmed(self):
        """Test PubMed search functionality."""
        client = NCBIClient(email="test@example.com")
        
        expected_response = NCBIResponse(
            success=True,
            data={"esearchresult": {"idlist": ["12345", "67890"]}},
            status_code=200,
            response_format=ResponseFormat.JSON
        )
        
        with patch.object(client, '_make_request_with_retry', return_value=expected_response):
            client._session_active = True
            
            response = await client.search(
                database=NCBIDatabase.PUBMED,
                query="cancer genomics",
                max_results=50,
                start=10
            )
            
            assert response.success is True
            assert response.data["esearchresult"]["idlist"] == ["12345", "67890"]
    
    @pytest.mark.asyncio
    async def test_fetch_papers(self):
        """Test fetching paper details."""
        client = NCBIClient(email="test@example.com")
        
        expected_response = NCBIResponse(
            success=True,
            data="<PubmedArticleSet>...</PubmedArticleSet>",
            status_code=200,
            response_format=ResponseFormat.XML
        )
        
        with patch.object(client, '_make_request_with_retry', return_value=expected_response):
            client._session_active = True
            
            response = await client.fetch(
                database=NCBIDatabase.PUBMED,
                ids=["12345", "67890"],
                return_type="abstract",
                return_mode="xml"
            )
            
            assert response.success is True
            assert "<PubmedArticleSet>" in response.data
    
    @pytest.mark.asyncio
    async def test_fetch_no_ids(self):
        """Test fetch with no IDs provided."""
        client = NCBIClient(email="test@example.com")
        client._session_active = True
        
        response = await client.fetch(
            database=NCBIDatabase.PUBMED,
            ids=[]
        )
        
        assert response.success is False
        assert "No IDs provided" in response.error_message
    
    @pytest.mark.asyncio
    async def test_get_database_info(self):
        """Test getting database information."""
        client = NCBIClient(email="test@example.com")
        
        expected_response = NCBIResponse(
            success=True,
            data={"einforesult": {"dbname": "pubmed"}},
            status_code=200,
            response_format=ResponseFormat.JSON
        )
        
        with patch.object(client, '_make_request_with_retry', return_value=expected_response):
            client._session_active = True
            
            response = await client.get_database_info(NCBIDatabase.PUBMED)
            
            assert response.success is True
            assert response.data["einforesult"]["dbname"] == "pubmed"
    
    def test_get_client_status(self):
        """Test getting client status."""
        client = NCBIClient(
            email="test@example.com",
            api_key="test-key",
            tool="test-tool",
            max_retries=5,
            timeout=45.0
        )
        
        status = client.get_client_status()
        
        assert status['email'] == "test@example.com"
        assert status['has_api_key'] is True
        assert status['tool'] == "test-tool"
        assert status['max_retries'] == 5
        assert status['timeout'] == 45.0
        assert status['session_active'] is False
        assert 'http_client' in status
        assert 'rate_limiter' in status
    
    @pytest.mark.asyncio
    async def test_post_request(self):
        """Test POST request functionality."""
        client = NCBIClient(email="test@example.com")
        
        expected_response = NCBIResponse(
            success=True,
            data={"result": "success"},
            status_code=200,
            response_format=ResponseFormat.JSON
        )
        
        with patch.object(client, '_make_request_with_retry', return_value=expected_response) as mock_request:
            client._session_active = True
            
            response = await client._make_request_with_retry(
                'POST', 'test-endpoint', data={'key': 'value'}
            )
            
            assert response.success is True
            mock_request.assert_called_once_with(
                'POST', 'test-endpoint', data={'key': 'value'}
            )
    
    @pytest.mark.asyncio
    async def test_unsupported_http_method(self):
        """Test unsupported HTTP method raises error."""
        client = NCBIClient(email="test@example.com")
        client._session_active = True
        
        with patch.object(client.rate_limiter, 'acquire'):
            response = await client._make_request_with_retry(
                'DELETE', 'test-endpoint'
            )
            
            assert response.success is False
            assert "Unsupported HTTP method" in response.error_message
    
    @pytest.mark.asyncio
    async def test_response_parsing_error(self):
        """Test handling response parsing errors."""
        client = NCBIClient(email="test@example.com")
        
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.headers = {'Content-Type': 'application/json'}
        mock_response.json.side_effect = Exception("JSON parsing error")
        
        with patch.object(client.http_client, 'get', return_value=mock_response):
            with patch.object(client.rate_limiter, 'acquire'):
                with patch.object(client.rate_limiter, 'record_success'):
                    client._session_active = True
                    
                    response = await client._make_request_with_retry(
                        'GET', 'test-endpoint'
                    )
                    
                    assert response.success is False
                    assert "Failed to parse response" in response.error_message