"""NCBI search client with specialized esearch.fcgi endpoint wrapper."""

import asyncio
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum

from ..config import get_logger
from .ncbi_client import NCBIClient, NCBIDatabase, NCBIResponse


logger = get_logger(__name__)


class SearchDatabase(Enum):
    """Supported NCBI databases for searching."""
    PUBMED = "pubmed"
    PMC = "pmc"
    CLINVAR = "clinvar"
    DBSNP = "snp"
    GENE = "gene"
    PROTEIN = "protein"


@dataclass
class SearchResult:
    """Structured search result from NCBI."""
    pmid: str
    title: Optional[str] = None
    authors: Optional[List[str]] = None
    journal: Optional[str] = None
    pub_date: Optional[str] = None
    abstract: Optional[str] = None
    doi: Optional[str] = None
    relevance_score: float = 0.0
    database: str = "pubmed"


@dataclass
class SearchResponse:
    """Complete search response with metadata."""
    results: List[SearchResult]
    total_count: int
    query_used: str
    database: str
    start_position: int
    max_results: int
    has_more: bool
    search_time: float = 0.0


class NCBISearchClient:
    """Specialized NCBI search client with esearch.fcgi endpoint wrapper."""
    
    def __init__(self, 
                 email: str,
                 api_key: Optional[str] = None,
                 tool: str = "genomics-pipeline-search",
                 max_retries: int = 3,
                 timeout: float = 30.0):
        """Initialize NCBI search client.
        
        Args:
            email: Email address for NCBI identification (required)
            api_key: NCBI API key for higher rate limits
            tool: Tool identifier
            max_retries: Maximum retry attempts
            timeout: Request timeout in seconds
        """
        self.ncbi_client = NCBIClient(
            email=email,
            api_key=api_key,
            tool=tool,
            max_retries=max_retries,
            timeout=timeout
        )
        
        logger.info(f"Initialized NCBISearchClient for {email}")
    
    @classmethod
    def from_config(cls, config=None) -> 'NCBISearchClient':
        """Create search client from configuration.
        
        Args:
            config: Configuration object. If None, loads from global config.
            
        Returns:
            NCBISearchClient: Configured search client instance
        """
        ncbi_client = NCBIClient.from_config(config)
        return cls(
            email=ncbi_client.email,
            api_key=ncbi_client.api_key,
            tool=ncbi_client.tool,
            max_retries=ncbi_client.max_retries,
            timeout=ncbi_client.timeout
        )
    
    async def __aenter__(self) -> 'NCBISearchClient':
        """Async context manager entry."""
        await self.ncbi_client.__aenter__()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.ncbi_client.__aexit__(exc_type, exc_val, exc_tb)
    
    def build_search_query(self, 
                          terms: List[str], 
                          operators: Optional[List[str]] = None,
                          field_tags: Optional[List[str]] = None) -> str:
        """Build a search query from terms and operators.
        
        Args:
            terms: List of search terms
            operators: List of operators (AND, OR, NOT) between terms
            field_tags: List of field tags for each term (e.g., [MeSH], [Title/Abstract])
            
        Returns:
            str: Formatted search query
        """
        if not terms:
            return ""
        
        if len(terms) == 1:
            term = terms[0]
            if field_tags and len(field_tags) > 0:
                return f'"{term}"{field_tags[0]}'
            return f'"{term}"'
        
        # Default to AND if no operators provided
        if not operators:
            operators = ["AND"] * (len(terms) - 1)
        
        # Ensure we have the right number of operators
        if len(operators) != len(terms) - 1:
            operators = (operators * len(terms))[:len(terms) - 1]
        
        query_parts = []
        for i, term in enumerate(terms):
            # Add field tag if provided
            if field_tags and i < len(field_tags) and field_tags[i]:
                formatted_term = f'"{term}"{field_tags[i]}'
            else:
                formatted_term = f'"{term}"'
            
            query_parts.append(formatted_term)
            
            # Add operator if not the last term
            if i < len(terms) - 1:
                query_parts.append(operators[i])
        
        return " ".join(query_parts)
    
    async def get_search_count(self, 
                              query: str, 
                              database: str = "pubmed") -> int:
        """Get the total count of search results without retrieving them.
        
        Args:
            query: Search query
            database: NCBI database to search
            
        Returns:
            int: Total number of matching results
        """
        logger.debug(f"Getting search count for query: {query}")
        
        # Convert string database to enum if needed
        if isinstance(database, str):
            try:
                db_enum = NCBIDatabase(database.lower())
            except ValueError:
                logger.error(f"Unsupported database: {database}")
                return 0
        else:
            db_enum = database
        
        response = await self.ncbi_client.search(
            database=db_enum,
            query=query,
            max_results=0,  # We only want the count
            start=0
        )
        
        if response.success and response.data:
            search_result = response.data.get('esearchresult', {})
            count = search_result.get('count', '0')
            try:
                return int(count)
            except (ValueError, TypeError):
                logger.warning(f"Invalid count value: {count}")
                return 0
        
        logger.error(f"Failed to get search count: {response.error_message}")
        return 0
    
    async def search_papers(self, 
                           query: str, 
                           max_results: int = 100,
                           database: str = "pubmed",
                           start: int = 0) -> List[str]:
        """Search for papers and return PMIDs.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            database: NCBI database to search
            start: Starting position for pagination
            
        Returns:
            List[str]: List of PMIDs
        """
        logger.info(f"Searching {database} for: {query} (max={max_results}, start={start})")
        
        # Convert string database to enum if needed
        if isinstance(database, str):
            try:
                db_enum = NCBIDatabase(database.lower())
            except ValueError:
                logger.error(f"Unsupported database: {database}")
                return []
        else:
            db_enum = database
        
        response = await self.ncbi_client.search(
            database=db_enum,
            query=query,
            max_results=max_results,
            start=start
        )
        
        if response.success and response.data:
            search_result = response.data.get('esearchresult', {})
            id_list = search_result.get('idlist', [])
            
            logger.info(f"Found {len(id_list)} papers")
            return id_list
        
        logger.error(f"Search failed: {response.error_message}")
        return []
    
    async def search_with_filters(self,
                                 base_query: str,
                                 date_range: Optional[Tuple[str, str]] = None,
                                 journal_filter: Optional[str] = None,
                                 study_type: Optional[str] = None,
                                 language: str = "eng",
                                 max_results: int = 100,
                                 database: str = "pubmed") -> List[str]:
        """Search with additional filters applied.
        
        Args:
            base_query: Base search query
            date_range: Tuple of (start_date, end_date) in YYYY/MM/DD format
            journal_filter: Journal name filter
            study_type: Study type filter (e.g., "clinical trial", "meta-analysis")
            language: Language filter (default: "eng")
            max_results: Maximum number of results
            database: NCBI database to search
            
        Returns:
            List[str]: List of PMIDs
        """
        # Build filtered query
        query_parts = [base_query]
        
        # Add date range filter
        if date_range:
            start_date, end_date = date_range
            date_filter = f'("{start_date}"[Date - Publication] : "{end_date}"[Date - Publication])'
            query_parts.append(date_filter)
        
        # Add journal filter
        if journal_filter:
            journal_filter_formatted = f'"{journal_filter}"[Journal]'
            query_parts.append(journal_filter_formatted)
        
        # Add study type filter
        if study_type:
            study_type_filter = f'"{study_type}"[Publication Type]'
            query_parts.append(study_type_filter)
        
        # Add language filter
        if language:
            language_filter = f'"{language}"[Language]'
            query_parts.append(language_filter)
        
        # Combine all filters with AND
        filtered_query = " AND ".join(query_parts)
        
        logger.info(f"Searching with filters: {filtered_query}")
        
        return await self.search_papers(
            query=filtered_query,
            max_results=max_results,
            database=database
        )
    
    async def search_batched(self, 
                            query: str, 
                            batch_size: int = 10000,
                            max_total: Optional[int] = None,
                            database: str = "pubmed") -> List[str]:
        """Search in batches to handle large result sets.
        
        Args:
            query: Search query
            batch_size: Size of each batch
            max_total: Maximum total results to retrieve (None for all)
            database: NCBI database to search
            
        Returns:
            List[str]: List of all PMIDs
        """
        logger.info(f"Starting batched search: {query}")
        
        # First, get the total count
        total_count = await self.get_search_count(query, database)
        logger.info(f"Total results available: {total_count}")
        
        if total_count == 0:
            return []
        
        # Determine how many results to actually retrieve
        if max_total:
            results_to_get = min(total_count, max_total)
        else:
            results_to_get = total_count
        
        all_pmids = []
        start = 0
        
        while start < results_to_get:
            # Calculate batch size for this iteration
            current_batch_size = min(batch_size, results_to_get - start)
            
            logger.debug(f"Fetching batch: start={start}, size={current_batch_size}")
            
            # Get this batch
            batch_pmids = await self.search_papers(
                query=query,
                max_results=current_batch_size,
                database=database,
                start=start
            )
            
            if not batch_pmids:
                logger.warning(f"No results in batch starting at {start}")
                break
            
            all_pmids.extend(batch_pmids)
            start += len(batch_pmids)
            
            # If we got fewer results than expected, we've reached the end
            if len(batch_pmids) < current_batch_size:
                break
            
            # Add a small delay between batches to be respectful
            await asyncio.sleep(0.1)
        
        logger.info(f"Batched search complete: retrieved {len(all_pmids)} PMIDs")
        return all_pmids
    
    async def search_detailed(self,
                             query: str,
                             max_results: int = 100,
                             database: str = "pubmed",
                             start: int = 0) -> SearchResponse:
        """Search and return detailed response with metadata.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            database: NCBI database to search
            start: Starting position for pagination
            
        Returns:
            SearchResponse: Detailed search response
        """
        import time
        
        start_time = time.time()
        
        # Get search results
        pmids = await self.search_papers(
            query=query,
            max_results=max_results,
            database=database,
            start=start
        )
        
        # Get total count
        total_count = await self.get_search_count(query, database)
        
        # Create SearchResult objects (basic info only from search)
        results = [
            SearchResult(
                pmid=pmid,
                database=database,
                relevance_score=1.0 - (i * 0.01)  # Simple relevance based on order
            )
            for i, pmid in enumerate(pmids)
        ]
        
        search_time = time.time() - start_time
        has_more = (start + len(pmids)) < total_count
        
        return SearchResponse(
            results=results,
            total_count=total_count,
            query_used=query,
            database=database,
            start_position=start,
            max_results=max_results,
            has_more=has_more,
            search_time=search_time
        )
    
    def get_client_status(self) -> Dict[str, Any]:
        """Get comprehensive client status.
        
        Returns:
            Dict[str, Any]: Client status information
        """
        return self.ncbi_client.get_client_status()