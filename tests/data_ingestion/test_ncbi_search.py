"""Tests for NCBISearchClient."""

import pytest
from unittest.mock import Mock, patch, AsyncMock

from src.data_ingestion.ncbi_search import (
    NCBISearchClient, SearchDatabase, SearchResult, SearchResponse
)
from src.data_ingestion.ncbi_client import NCBIResponse, ResponseFormat


class TestNCBISearchClient:
    """Test cases for NCBISearchClient."""
    
    def test_init_default_values(self):
        """Test initialization with default values."""
        client = NCBISearchClient(email="test@example.com")
        
        assert client.ncbi_client.email == "test@example.com"
        assert client.ncbi_client.api_key is None
        assert client.ncbi_client.tool == "genomics-pipeline-search"
        assert client.ncbi_client.max_retries == 3
        assert client.ncbi_client.timeout == 30.0
    
    def test_init_custom_values(self):
        """Test initialization with custom values."""
        client = NCBISearchClient(
            email="custom@example.com",
            api_key="test-key",
            tool="custom-tool",
            max_retries=5,
            timeout=60.0
        )
        
        assert client.ncbi_client.email == "custom@example.com"
        assert client.ncbi_client.api_key == "test-key"
        assert client.ncbi_client.tool == "custom-tool"
        assert client.ncbi_client.max_retries == 5
        assert client.ncbi_client.timeout == 60.0
    
    def test_build_search_query_single_term(self):
        """Test building query with single term."""
        client = NCBISearchClient(email="test@example.com")
        
        query = client.build_search_query(["pharmacogenomics"])
        assert query == '"pharmacogenomics"'
        
        # With field tag
        query = client.build_search_query(["pharmacogenomics"], field_tags=["[MeSH]"])
        assert query == '"pharmacogenomics"[MeSH]'
    
    def test_build_search_query_multiple_terms(self):
        """Test building query with multiple terms."""
        client = NCBISearchClient(email="test@example.com")
        
        # Default AND operator
        query = client.build_search_query(["pharmacogenomics", "machine learning"])
        assert query == '"pharmacogenomics" AND "machine learning"'
        
        # Custom operators
        query = client.build_search_query(
            ["pharmacogenomics", "machine learning", "AI"],
            operators=["OR", "AND"]
        )
        assert query == '"pharmacogenomics" OR "machine learning" AND "AI"'
        
        # With field tags
        query = client.build_search_query(
            ["pharmacogenomics", "machine learning"],
            field_tags=["[MeSH]", "[Title/Abstract]"]
        )
        assert query == '"pharmacogenomics"[MeSH] AND "machine learning"[Title/Abstract]'
    
    def test_build_search_query_empty_terms(self):
        """Test building query with empty terms."""
        client = NCBISearchClient(email="test@example.com")
        
        query = client.build_search_query([])
        assert query == ""
    
    @pytest.mark.asyncio
    async def test_get_search_count_success(self):
        """Test getting search count successfully."""
        client = NCBISearchClient(email="test@example.com")
        
        mock_response = NCBIResponse(
            success=True,
            data={
                "esearchresult": {
                    "count": "1234",
                    "idlist": []
                }
            }
        )
        
        with patch.object(client.ncbi_client, 'search', return_value=mock_response):
            count = await client.get_search_count("test query", "pubmed")
            assert count == 1234
    
    @pytest.mark.asyncio
    async def test_get_search_count_failure(self):
        """Test getting search count with failure."""
        client = NCBISearchClient(email="test@example.com")
        
        mock_response = NCBIResponse(
            success=False,
            data=None,
            error_message="Search failed"
        )
        
        with patch.object(client.ncbi_client, 'search', return_value=mock_response):
            count = await client.get_search_count("test query", "pubmed")
            assert count == 0
    
    @pytest.mark.asyncio
    async def test_get_search_count_invalid_database(self):
        """Test getting search count with invalid database."""
        client = NCBISearchClient(email="test@example.com")
        
        count = await client.get_search_count("test query", "invalid_db")
        assert count == 0
    
    @pytest.mark.asyncio
    async def test_search_papers_success(self):
        """Test searching papers successfully."""
        client = NCBISearchClient(email="test@example.com")
        
        mock_response = NCBIResponse(
            success=True,
            data={
                "esearchresult": {
                    "idlist": ["12345", "67890", "11111"],
                    "count": "3"
                }
            }
        )
        
        with patch.object(client.ncbi_client, 'search', return_value=mock_response):
            pmids = await client.search_papers("test query", max_results=10)
            assert pmids == ["12345", "67890", "11111"]
    
    @pytest.mark.asyncio
    async def test_search_papers_failure(self):
        """Test searching papers with failure."""
        client = NCBISearchClient(email="test@example.com")
        
        mock_response = NCBIResponse(
            success=False,
            data=None,
            error_message="Search failed"
        )
        
        with patch.object(client.ncbi_client, 'search', return_value=mock_response):
            pmids = await client.search_papers("test query")
            assert pmids == []
    
    @pytest.mark.asyncio
    async def test_search_with_filters(self):
        """Test searching with filters."""
        client = NCBISearchClient(email="test@example.com")
        
        mock_response = NCBIResponse(
            success=True,
            data={
                "esearchresult": {
                    "idlist": ["12345", "67890"],
                    "count": "2"
                }
            }
        )
        
        with patch.object(client.ncbi_client, 'search', return_value=mock_response) as mock_search:
            pmids = await client.search_with_filters(
                base_query="pharmacogenomics",
                date_range=("2020/01/01", "2023/12/31"),
                journal_filter="Nature",
                study_type="clinical trial",
                language="eng"
            )
            
            assert pmids == ["12345", "67890"]
            
            # Check that the query was properly constructed
            call_args = mock_search.call_args
            query_used = call_args[1]['query']
            assert "pharmacogenomics" in query_used
            assert "2020/01/01" in query_used
            assert "2023/12/31" in query_used
            assert "Nature" in query_used
            assert "clinical trial" in query_used
            assert "eng" in query_used
    
    @pytest.mark.asyncio
    async def test_search_batched(self):
        """Test batched searching."""
        client = NCBISearchClient(email="test@example.com")
        
        # Mock get_search_count to return 25 total results
        with patch.object(client, 'get_search_count', return_value=25):
            # Mock search_papers to return batches
            batch_responses = [
                ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],  # First batch
                ["11", "12", "13", "14", "15", "16", "17", "18", "19", "20"],  # Second batch
                ["21", "22", "23", "24", "25"]  # Final batch
            ]
            
            with patch.object(client, 'search_papers', side_effect=batch_responses):
                with patch('asyncio.sleep'):  # Speed up test
                    pmids = await client.search_batched("test query", batch_size=10)
                    
                    assert len(pmids) == 25
                    assert pmids == [str(i) for i in range(1, 26)]
    
    @pytest.mark.asyncio
    async def test_search_batched_with_max_total(self):
        """Test batched searching with max_total limit."""
        client = NCBISearchClient(email="test@example.com")
        
        # Mock get_search_count to return 1000 total results
        with patch.object(client, 'get_search_count', return_value=1000):
            # Mock search_papers to return a batch
            with patch.object(client, 'search_papers', return_value=["1", "2", "3", "4", "5"]):
                with patch('asyncio.sleep'):  # Speed up test
                    pmids = await client.search_batched("test query", batch_size=10, max_total=5)
                    
                    assert len(pmids) == 5
                    assert pmids == ["1", "2", "3", "4", "5"]
    
    @pytest.mark.asyncio
    async def test_search_detailed(self):
        """Test detailed search with metadata."""
        client = NCBISearchClient(email="test@example.com")
        
        # Mock search_papers and get_search_count
        with patch.object(client, 'search_papers', return_value=["12345", "67890"]):
            with patch.object(client, 'get_search_count', return_value=100):
                response = await client.search_detailed("test query", max_results=10)
                
                assert isinstance(response, SearchResponse)
                assert len(response.results) == 2
                assert response.total_count == 100
                assert response.query_used == "test query"
                assert response.database == "pubmed"
                assert response.start_position == 0
                assert response.max_results == 10
                assert response.has_more is True
                assert response.search_time > 0
                
                # Check individual results
                assert response.results[0].pmid == "12345"
                assert response.results[1].pmid == "67890"
                assert response.results[0].relevance_score > response.results[1].relevance_score
    
    def test_get_client_status(self):
        """Test getting client status."""
        client = NCBISearchClient(email="test@example.com")
        
        with patch.object(client.ncbi_client, 'get_client_status', return_value={"status": "active"}):
            status = client.get_client_status()
            assert status == {"status": "active"}


class TestSearchResult:
    """Test cases for SearchResult dataclass."""
    
    def test_init_minimal(self):
        """Test initialization with minimal data."""
        result = SearchResult(pmid="12345")
        
        assert result.pmid == "12345"
        assert result.title is None
        assert result.authors is None
        assert result.journal is None
        assert result.pub_date is None
        assert result.abstract is None
        assert result.doi is None
        assert result.relevance_score == 0.0
        assert result.database == "pubmed"
    
    def test_init_complete(self):
        """Test initialization with complete data."""
        result = SearchResult(
            pmid="12345",
            title="Test Paper",
            authors=["Author 1", "Author 2"],
            journal="Test Journal",
            pub_date="2023",
            abstract="Test abstract",
            doi="10.1000/test",
            relevance_score=0.95,
            database="pmc"
        )
        
        assert result.pmid == "12345"
        assert result.title == "Test Paper"
        assert result.authors == ["Author 1", "Author 2"]
        assert result.journal == "Test Journal"
        assert result.pub_date == "2023"
        assert result.abstract == "Test abstract"
        assert result.doi == "10.1000/test"
        assert result.relevance_score == 0.95
        assert result.database == "pmc"


class TestSearchResponse:
    """Test cases for SearchResponse dataclass."""
    
    def test_init_complete(self):
        """Test initialization with complete data."""
        results = [
            SearchResult(pmid="12345"),
            SearchResult(pmid="67890")
        ]
        
        response = SearchResponse(
            results=results,
            total_count=100,
            query_used="test query",
            database="pubmed",
            start_position=0,
            max_results=10,
            has_more=True,
            search_time=1.5
        )
        
        assert len(response.results) == 2
        assert response.total_count == 100
        assert response.query_used == "test query"
        assert response.database == "pubmed"
        assert response.start_position == 0
        assert response.max_results == 10
        assert response.has_more is True
        assert response.search_time == 1.5