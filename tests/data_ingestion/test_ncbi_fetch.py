"""Tests for NCBIFetchClient."""

import pytest
from unittest.mock import Mock, patch, AsyncMock

from src.data_ingestion.ncbi_fetch import NCBIFetchClient, FetchStats
from src.data_ingestion.ncbi_client import NCBIResponse, ResponseFormat


class TestNCBIFetchClient:
    """Test cases for NCBIFetchClient."""
    
    def test_init_default_values(self):
        """Test initialization with default values."""
        client = NCBIFetchClient(email="test@example.com")
        
        assert client.ncbi_client.email == "test@example.com"
        assert client.ncbi_client.api_key is None
        assert client.ncbi_client.tool == "genomics-pipeline-fetch"
        assert client.default_batch_size == 200
        assert client.max_pmids_per_request == 200
        assert client.max_url_length == 8000
    
    def test_init_custom_values(self):
        """Test initialization with custom values."""
        client = NCBIFetchClient(
            email="custom@example.com",
            api_key="test-key",
            tool="custom-tool",
            max_retries=5,
            timeout=120.0,
            default_batch_size=100
        )
        
        assert client.ncbi_client.email == "custom@example.com"
        assert client.ncbi_client.api_key == "test-key"
        assert client.ncbi_client.tool == "custom-tool"
        assert client.ncbi_client.max_retries == 5
        assert client.ncbi_client.timeout == 120.0
        assert client.default_batch_size == 100
    
    def test_split_pmids_for_batching_empty(self):
        """Test splitting empty PMID list."""
        client = NCBIFetchClient(email="test@example.com")
        
        batches = client.split_pmids_for_batching([])
        assert batches == []
    
    def test_split_pmids_for_batching_small_list(self):
        """Test splitting small PMID list."""
        client = NCBIFetchClient(email="test@example.com", default_batch_size=100)
        
        pmids = ["12345", "67890", "11111"]
        batches = client.split_pmids_for_batching(pmids)
        
        assert len(batches) == 1
        assert batches[0] == pmids
    
    def test_split_pmids_for_batching_large_list(self):
        """Test splitting large PMID list."""
        client = NCBIFetchClient(email="test@example.com", default_batch_size=3)
        
        pmids = [str(i) for i in range(10)]  # 10 PMIDs
        batches = client.split_pmids_for_batching(pmids)
        
        assert len(batches) == 4  # 3, 3, 3, 1
        assert len(batches[0]) == 3
        assert len(batches[1]) == 3
        assert len(batches[2]) == 3
        assert len(batches[3]) == 1
    
    def test_split_pmids_for_batching_custom_size(self):
        """Test splitting with custom batch size."""
        client = NCBIFetchClient(email="test@example.com")
        
        pmids = [str(i) for i in range(10)]
        batches = client.split_pmids_for_batching(pmids, max_batch_size=4)
        
        assert len(batches) == 3  # 4, 4, 2
        assert len(batches[0]) == 4
        assert len(batches[1]) == 4
        assert len(batches[2]) == 2
    
    def test_split_pmids_respects_limits(self):
        """Test that splitting respects NCBI limits."""
        client = NCBIFetchClient(email="test@example.com", default_batch_size=300)
        
        pmids = [str(i) for i in range(250)]
        batches = client.split_pmids_for_batching(pmids)
        
        # Should be limited by max_pmids_per_request (200)
        assert len(batches) == 2  # 200, 50
        assert len(batches[0]) == 200
        assert len(batches[1]) == 50
    
    @pytest.mark.asyncio
    async def test_fetch_single_paper_success(self):
        """Test fetching single paper successfully."""
        client = NCBIFetchClient(email="test@example.com")
        
        mock_response = NCBIResponse(
            success=True,
            data="<PubmedArticle>...</PubmedArticle>",
            status_code=200,
            response_format=ResponseFormat.XML
        )
        
        with patch.object(client.ncbi_client, 'fetch', return_value=mock_response):
            response = await client.fetch_single_paper("12345")
            
            assert response.success is True
            assert response.data == "<PubmedArticle>...</PubmedArticle>"
    
    @pytest.mark.asyncio
    async def test_fetch_single_paper_invalid_database(self):
        """Test fetching with invalid database."""
        client = NCBIFetchClient(email="test@example.com")
        
        response = await client.fetch_single_paper("12345", database="invalid_db")
        
        assert response.success is False
        assert "Unsupported database" in response.error_message
    
    @pytest.mark.asyncio
    async def test_fetch_batch_success(self):
        """Test fetching batch successfully."""
        client = NCBIFetchClient(email="test@example.com")
        
        mock_response = NCBIResponse(
            success=True,
            data="<PubmedArticleSet>...</PubmedArticleSet>",
            status_code=200,
            response_format=ResponseFormat.XML
        )
        
        with patch.object(client.ncbi_client, 'fetch', return_value=mock_response):
            response = await client.fetch_batch(["12345", "67890"])
            
            assert response.success is True
            assert response.data == "<PubmedArticleSet>...</PubmedArticleSet>"
    
    @pytest.mark.asyncio
    async def test_fetch_batch_empty_pmids(self):
        """Test fetching batch with empty PMIDs."""
        client = NCBIFetchClient(email="test@example.com")
        
        response = await client.fetch_batch([])
        
        assert response.success is False
        assert "No PMIDs provided" in response.error_message
    
    @pytest.mark.asyncio
    async def test_fetch_batch_size_limit(self):
        """Test batch size limiting."""
        client = NCBIFetchClient(email="test@example.com", default_batch_size=3)
        
        mock_response = NCBIResponse(success=True, data="<xml>test</xml>")
        
        with patch.object(client.ncbi_client, 'fetch', return_value=mock_response) as mock_fetch:
            # Provide 5 PMIDs but batch size is 3
            await client.fetch_batch(["1", "2", "3", "4", "5"])
            
            # Should only fetch first 3
            call_args = mock_fetch.call_args
            assert len(call_args[1]['ids']) == 3
    
    @pytest.mark.asyncio
    async def test_fetch_paper_details_empty(self):
        """Test fetching details with empty PMID list."""
        client = NCBIFetchClient(email="test@example.com")
        
        responses, stats = await client.fetch_paper_details([])
        
        assert responses == []
        assert stats.total_pmids == 0
        assert stats.successful_fetches == 0
        assert stats.failed_fetches == 0
    
    @pytest.mark.asyncio
    async def test_fetch_paper_details_success(self):
        """Test fetching paper details successfully."""
        client = NCBIFetchClient(email="test@example.com", default_batch_size=2)
        
        mock_response = NCBIResponse(
            success=True,
            data="<PubmedArticleSet>...</PubmedArticleSet>"
        )
        
        with patch.object(client, 'fetch_batch', return_value=mock_response):
            with patch('asyncio.sleep'):  # Speed up test
                responses, stats = await client.fetch_paper_details(["1", "2", "3"])
                
                assert len(responses) == 2  # 2 batches
                assert stats.total_pmids == 3
                assert stats.successful_fetches == 3
                assert stats.failed_fetches == 0
                assert stats.total_batches == 2
    
    @pytest.mark.asyncio
    async def test_fetch_paper_details_with_failures(self):
        """Test fetching with some failures."""
        client = NCBIFetchClient(email="test@example.com", default_batch_size=2)
        
        # First batch succeeds, second fails
        responses = [
            NCBIResponse(success=True, data="<xml>batch1</xml>"),
            NCBIResponse(success=False, error_message="Batch failed")
        ]
        
        with patch.object(client, 'fetch_batch', side_effect=responses):
            with patch('asyncio.sleep'):
                result_responses, stats = await client.fetch_paper_details(["1", "2", "3", "4"])
                
                assert len(result_responses) == 1  # Only successful batch
                assert stats.successful_fetches == 2  # First batch
                assert stats.failed_fetches == 2   # Second batch
    
    @pytest.mark.asyncio
    async def test_get_paper_formats_exists(self):
        """Test getting formats for existing paper."""
        client = NCBIFetchClient(email="test@example.com")
        
        mock_response = NCBIResponse(success=True, data="<xml>paper</xml>")
        
        with patch.object(client, 'fetch_single_paper', return_value=mock_response):
            formats = await client.get_paper_formats("12345")
            
            assert "abstract" in formats
            assert "full" in formats
            assert "medline" in formats
            assert "bibtex" in formats
    
    @pytest.mark.asyncio
    async def test_get_paper_formats_not_found(self):
        """Test getting formats for non-existent paper."""
        client = NCBIFetchClient(email="test@example.com")
        
        mock_response = NCBIResponse(success=False, error_message="Not found")
        
        with patch.object(client, 'fetch_single_paper', return_value=mock_response):
            formats = await client.get_paper_formats("99999")
            
            assert formats == []
    
    @pytest.mark.asyncio
    async def test_fetch_with_retry_on_batch_failure(self):
        """Test retry logic for failed batches."""
        client = NCBIFetchClient(email="test@example.com")
        
        # Mock fetch_paper_details to simulate failures then success
        call_count = 0
        
        async def mock_fetch_details(pmids, **kwargs):
            nonlocal call_count
            call_count += 1
            
            if call_count == 1:
                # First call: some failures
                return ["<xml>success</xml>"], FetchStats(
                    total_pmids=len(pmids),
                    successful_fetches=len(pmids) - 2,
                    failed_fetches=2,
                    total_batches=1,
                    total_time=1.0,
                    average_batch_time=1.0,
                    papers_per_second=1.0
                )
            else:
                # Retry call: success
                return ["<xml>retry_success</xml>"], FetchStats(
                    total_pmids=len(pmids),
                    successful_fetches=len(pmids),
                    failed_fetches=0,
                    total_batches=1,
                    total_time=1.0,
                    average_batch_time=1.0,
                    papers_per_second=1.0
                )
        
        with patch.object(client, 'fetch_paper_details', side_effect=mock_fetch_details):
            responses, stats = await client.fetch_with_retry_on_batch_failure(
                ["1", "2", "3", "4"], max_retries=1
            )
            
            assert len(responses) == 2  # Original + retry
            assert call_count == 2  # Should have retried
    
    def test_get_client_status(self):
        """Test getting client status."""
        client = NCBIFetchClient(email="test@example.com", default_batch_size=150)
        
        with patch.object(client.ncbi_client, 'get_client_status', return_value={"status": "active"}):
            status = client.get_client_status()
            
            assert status["status"] == "active"
            assert status["fetch_client"]["default_batch_size"] == 150
            assert status["fetch_client"]["max_pmids_per_request"] == 200
            assert status["fetch_client"]["max_url_length"] == 8000


class TestFetchStats:
    """Test cases for FetchStats dataclass."""
    
    def test_init(self):
        """Test FetchStats initialization."""
        stats = FetchStats(
            total_pmids=100,
            successful_fetches=95,
            failed_fetches=5,
            total_batches=10,
            total_time=30.0,
            average_batch_time=3.0,
            papers_per_second=3.17
        )
        
        assert stats.total_pmids == 100
        assert stats.successful_fetches == 95
        assert stats.failed_fetches == 5
        assert stats.total_batches == 10
        assert stats.total_time == 30.0
        assert stats.average_batch_time == 3.0
        assert stats.papers_per_second == 3.17