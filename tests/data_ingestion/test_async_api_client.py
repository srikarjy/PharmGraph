"""Tests for AsyncAPIClient."""

import pytest
import asyncio
import ssl
from unittest.mock import Mock, patch, AsyncMock
from aiohttp import ClientSession, ClientTimeout, TCPConnector

from src.data_ingestion.async_api_client import AsyncAPIClient


class TestAsyncAPIClient:
    """Test cases for AsyncAPIClient."""
    
    def test_init_default_values(self):
        """Test initialization with default values."""
        client = AsyncAPIClient(email="test@example.com")
        
        assert client.base_url == "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        assert client.timeout == 30.0
        assert client.max_connections == 100
        assert client.max_connections_per_host == 10
        assert client.email == "test@example.com"
        assert client.api_key is None
        assert client.tool == "genomics-pipeline"
        assert "test@example.com" in client.user_agent
        assert client._session is None
        assert client._closed is False
    
    def test_init_custom_values(self):
        """Test initialization with custom values."""
        client = AsyncAPIClient(
            base_url="https://custom.api.com/",
            timeout=60.0,
            max_connections=50,
            max_connections_per_host=5,
            user_agent="CustomAgent/1.0",
            email="custom@example.com",
            api_key="test-key",
            tool="custom-tool"
        )
        
        assert client.base_url == "https://custom.api.com/"
        assert client.timeout == 60.0
        assert client.max_connections == 50
        assert client.max_connections_per_host == 5
        assert client.user_agent == "CustomAgent/1.0"
        assert client.email == "custom@example.com"
        assert client.api_key == "test-key"
        assert client.tool == "custom-tool"
    
    @patch('src.data_ingestion.async_api_client.get_config')
    def test_from_config(self, mock_get_config):
        """Test creating client from configuration."""
        mock_config = Mock()
        mock_config.ncbi.base_url = "https://test.api.com/"
        mock_config.ncbi.timeout = 45.0
        mock_config.ncbi.max_connections = 75
        mock_config.ncbi.max_connections_per_host = 8
        mock_config.ncbi.email = "config@example.com"
        mock_config.ncbi.api_key = "config-key"
        mock_config.ncbi.tool = "config-tool"
        mock_get_config.return_value = mock_config
        
        client = AsyncAPIClient.from_config()
        
        assert client.base_url == "https://test.api.com/"
        assert client.timeout == 45.0
        assert client.max_connections == 75
        assert client.max_connections_per_host == 8
        assert client.email == "config@example.com"
        assert client.api_key == "config-key"
        assert client.tool == "config-tool"
    
    @pytest.mark.asyncio
    async def test_context_manager_lifecycle(self):
        """Test async context manager lifecycle."""
        client = AsyncAPIClient(email="test@example.com")
        
        # Initially not initialized
        assert client._session is None
        assert client._connector is None
        assert client._closed is False
        
        # Enter context
        async with client as ctx_client:
            assert ctx_client is client
            assert client._session is not None
            assert client._connector is not None
            assert client._closed is False
            assert isinstance(client._session, ClientSession)
            assert isinstance(client._connector, TCPConnector)
        
        # After exit
        assert client._session is None
        assert client._connector is None
        assert client._closed is True
    
    @pytest.mark.asyncio
    async def test_session_configuration(self):
        """Test session is configured correctly."""
        client = AsyncAPIClient(
            email="test@example.com",
            timeout=15.0,
            max_connections=25,
            max_connections_per_host=3
        )
        
        async with client:
            # Check connector configuration
            connector = client._connector
            assert connector.limit == 25
            assert connector.limit_per_host == 3
            assert connector.enable_cleanup_closed is True
            assert connector.keepalive_timeout == 30
            assert connector.ttl_dns_cache == 300
            assert connector.use_dns_cache is True
            
            # Check session configuration
            session = client._session
            assert session.timeout.total == 15.0
            assert "User-Agent" in session.headers
            assert "test@example.com" in session.headers["User-Agent"]
            assert session.headers["Accept"] == 'application/json, application/xml, text/xml'
    
    def test_build_url(self):
        """Test URL building logic."""
        client = AsyncAPIClient(
            base_url="https://api.example.com/v1/",
            email="test@example.com"
        )
        
        # Relative endpoint
        assert client._build_url("search") == "https://api.example.com/v1/search"
        assert client._build_url("/search") == "https://api.example.com/v1/search"
        
        # Absolute URL
        assert client._build_url("https://other.api.com/endpoint") == "https://other.api.com/endpoint"
    
    def test_add_ncbi_params(self):
        """Test NCBI parameter addition."""
        client = AsyncAPIClient(
            email="test@example.com",
            api_key="test-key",
            tool="test-tool"
        )
        
        # Empty params
        result = client._add_ncbi_params(None)
        expected = {
            'email': 'test@example.com',
            'api_key': 'test-key',
            'tool': 'test-tool'
        }
        assert result == expected
        
        # Existing params
        existing = {'query': 'test', 'limit': 10}
        result = client._add_ncbi_params(existing)
        expected = {
            'query': 'test',
            'limit': 10,
            'email': 'test@example.com',
            'api_key': 'test-key',
            'tool': 'test-tool'
        }
        assert result == expected
        
        # Without API key
        client_no_key = AsyncAPIClient(email="test@example.com", tool="test-tool")
        result = client_no_key._add_ncbi_params({'query': 'test'})
        expected = {
            'query': 'test',
            'email': 'test@example.com',
            'tool': 'test-tool'
        }
        assert result == expected
    
    @pytest.mark.asyncio
    async def test_get_request(self):
        """Test GET request functionality."""
        client = AsyncAPIClient(email="test@example.com")
        
        with patch.object(client, '_session') as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_session.get.return_value = mock_response
            
            client._closed = False  # Simulate active session
            
            response = await client.get("test-endpoint", params={'q': 'test'})
            
            assert response is mock_response
            mock_session.get.assert_called_once()
            call_args = mock_session.get.call_args
            assert "test-endpoint" in call_args[0][0]  # URL contains endpoint
            assert call_args[1]['params']['q'] == 'test'
            assert call_args[1]['params']['email'] == 'test@example.com'
    
    @pytest.mark.asyncio
    async def test_post_request(self):
        """Test POST request functionality."""
        client = AsyncAPIClient(email="test@example.com")
        
        with patch.object(client, '_session') as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_session.post.return_value = mock_response
            
            client._closed = False  # Simulate active session
            
            response = await client.post("test-endpoint", data={'key': 'value'})
            
            assert response is mock_response
            mock_session.post.assert_called_once()
            call_args = mock_session.post.call_args
            assert "test-endpoint" in call_args[0][0]  # URL contains endpoint
            assert call_args[1]['data']['key'] == 'value'
            assert call_args[1]['params']['email'] == 'test@example.com'
    
    def test_ensure_session_not_initialized(self):
        """Test _ensure_session raises error when not initialized."""
        client = AsyncAPIClient(email="test@example.com")
        
        with pytest.raises(RuntimeError, match="AsyncAPIClient not initialized"):
            client._ensure_session()
    
    def test_ensure_session_closed(self):
        """Test _ensure_session raises error when closed."""
        client = AsyncAPIClient(email="test@example.com")
        client._closed = True
        
        with pytest.raises(RuntimeError, match="AsyncAPIClient not initialized"):
            client._ensure_session()
    
    def test_get_session_info_not_initialized(self):
        """Test get_session_info when not initialized."""
        client = AsyncAPIClient(
            base_url="https://test.api.com/",
            timeout=25.0,
            email="test@example.com"
        )
        
        info = client.get_session_info()
        
        assert info['status'] == 'closed'
        assert info['base_url'] == 'https://test.api.com/'
        assert info['timeout'] == 25.0
    
    @pytest.mark.asyncio
    async def test_get_session_info_active(self):
        """Test get_session_info when active."""
        client = AsyncAPIClient(
            email="test@example.com",
            api_key="test-key",
            tool="test-tool"
        )
        
        async with client:
            info = client.get_session_info()
            
            assert info['status'] == 'active'
            assert info['email'] == 'test@example.com'
            assert info['has_api_key'] is True
            assert info['tool'] == 'test-tool'
            assert 'connector' in info
            assert info['session_closed'] is False
    
    def test_is_closed_property(self):
        """Test is_closed property."""
        client = AsyncAPIClient(email="test@example.com")
        
        # Initially not closed (but not initialized)
        assert client.is_closed is False
        
        # After closing
        client._closed = True
        assert client.is_closed is True
    
    @pytest.mark.asyncio
    async def test_close_method(self):
        """Test explicit close method."""
        client = AsyncAPIClient(email="test@example.com")
        
        async with client:
            assert not client.is_closed
        
        # Should already be closed by context manager
        assert client.is_closed
        
        # Calling close again should be safe
        await client.close()
        assert client.is_closed
    
    @pytest.mark.asyncio
    async def test_request_error_handling(self):
        """Test error handling in requests."""
        client = AsyncAPIClient(email="test@example.com")
        
        with patch.object(client, '_session') as mock_session:
            mock_session.get.side_effect = Exception("Network error")
            client._closed = False
            
            with pytest.raises(Exception, match="Network error"):
                await client.get("test-endpoint")
    
    @pytest.mark.asyncio
    async def test_multiple_context_entries(self):
        """Test multiple context manager entries."""
        client = AsyncAPIClient(email="test@example.com")
        
        # First entry
        async with client:
            session1 = client._session
            assert session1 is not None
            
            # Second entry should return same instance
            async with client:
                session2 = client._session
                assert session2 is session1