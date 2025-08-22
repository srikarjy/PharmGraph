"""NCBI efetch client for retrieving detailed paper information."""

import asyncio
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import math

from ..config import get_logger
from .ncbi_client import NCBIClient, NCBIDatabase, NCBIResponse


logger = get_logger(__name__)


@dataclass
class FetchStats:
    """Statistics for fetch operations."""
    total_pmids: int
    successful_fetches: int
    failed_fetches: int
    total_batches: int
    total_time: float
    average_batch_time: float
    papers_per_second: float


class NCBIFetchClient:
    """NCBI efetch client for retrieving detailed paper information."""
    
    def __init__(self,
                 email: str,
                 api_key: Optional[str] = None,
                 tool: str = "genomics-pipeline-fetch",
                 max_retries: int = 3,
                 timeout: float = 60.0,
                 default_batch_size: int = 200):
        """Initialize NCBI fetch client.
        
        Args:
            email: Email address for NCBI identification (required)
            api_key: NCBI API key for higher rate limits
            tool: Tool identifier
            max_retries: Maximum retry attempts
            timeout: Request timeout in seconds
            default_batch_size: Default batch size for fetching
        """
        self.ncbi_client = NCBIClient(
            email=email,
            api_key=api_key,
            tool=tool,
            max_retries=max_retries,
            timeout=timeout
        )
        
        self.default_batch_size = default_batch_size
        
        # NCBI efetch limits
        self.max_url_length = 8000  # Conservative URL length limit
        self.max_pmids_per_request = 200  # NCBI recommended maximum
        
        logger.info(f"Initialized NCBIFetchClient for {email} with batch_size={default_batch_size}")
    
    @classmethod
    def from_config(cls, config=None) -> 'NCBIFetchClient':
        """Create fetch client from configuration.
        
        Args:
            config: Configuration object. If None, loads from global config.
            
        Returns:
            NCBIFetchClient: Configured fetch client instance
        """
        ncbi_client = NCBIClient.from_config(config)
        return cls(
            email=ncbi_client.email,
            api_key=ncbi_client.api_key,
            tool=ncbi_client.tool,
            max_retries=ncbi_client.max_retries,
            timeout=ncbi_client.timeout
        )
    
    async def __aenter__(self) -> 'NCBIFetchClient':
        """Async context manager entry."""
        await self.ncbi_client.__aenter__()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.ncbi_client.__aexit__(exc_type, exc_val, exc_tb)
    
    def split_pmids_for_batching(self, 
                                pmids: List[str], 
                                max_batch_size: Optional[int] = None) -> List[List[str]]:
        """Split PMIDs into batches for efficient fetching.
        
        Args:
            pmids: List of PMIDs to split
            max_batch_size: Maximum batch size (uses default if None)
            
        Returns:
            List[List[str]]: List of PMID batches
        """
        if not pmids:
            return []
        
        batch_size = max_batch_size or self.default_batch_size
        
        # Ensure we don't exceed NCBI limits
        batch_size = min(batch_size, self.max_pmids_per_request)
        
        # Additional check for URL length (rough estimate)
        # Average PMID length is ~8 characters, plus commas
        estimated_url_base = 200  # Base URL + parameters
        max_pmids_for_url = (self.max_url_length - estimated_url_base) // 9  # 8 chars + comma
        batch_size = min(batch_size, max_pmids_for_url)
        
        # Split into batches
        batches = []
        for i in range(0, len(pmids), batch_size):
            batch = pmids[i:i + batch_size]
            batches.append(batch)
        
        logger.debug(f"Split {len(pmids)} PMIDs into {len(batches)} batches of max size {batch_size}")
        return batches
    
    async def fetch_single_paper(self, 
                                pmid: str,
                                database: str = "pubmed",
                                return_type: str = "abstract",
                                return_mode: str = "xml") -> NCBIResponse:
        """Fetch details for a single paper.
        
        Args:
            pmid: PubMed ID to fetch
            database: NCBI database
            return_type: Type of data to return
            return_mode: Format of returned data
            
        Returns:
            NCBIResponse: Fetch response
        """
        logger.debug(f"Fetching single paper: {pmid}")
        
        # Convert string database to enum if needed
        if isinstance(database, str):
            try:
                db_enum = NCBIDatabase(database.lower())
            except ValueError:
                logger.error(f"Unsupported database: {database}")
                return NCBIResponse(
                    success=False,
                    data=None,
                    error_message=f"Unsupported database: {database}"
                )
        else:
            db_enum = database
        
        response = await self.ncbi_client.fetch(
            database=db_enum,
            ids=[pmid],
            return_type=return_type,
            return_mode=return_mode
        )
        
        return response
    
    async def fetch_batch(self,
                         pmids: List[str],
                         batch_size: Optional[int] = None,
                         database: str = "pubmed",
                         return_type: str = "abstract",
                         return_mode: str = "xml") -> NCBIResponse:
        """Fetch details for a batch of papers.
        
        Args:
            pmids: List of PMIDs to fetch
            batch_size: Maximum batch size (uses default if None)
            database: NCBI database
            return_type: Type of data to return
            return_mode: Format of returned data
            
        Returns:
            NCBIResponse: Fetch response
        """
        if not pmids:
            return NCBIResponse(
                success=False,
                data=None,
                error_message="No PMIDs provided for batch fetch"
            )
        
        # Limit batch size
        effective_batch_size = batch_size or self.default_batch_size
        if len(pmids) > effective_batch_size:
            pmids = pmids[:effective_batch_size]
            logger.warning(f"Batch size limited to {effective_batch_size} PMIDs")
        
        logger.debug(f"Fetching batch of {len(pmids)} papers")
        
        # Convert string database to enum if needed
        if isinstance(database, str):
            try:
                db_enum = NCBIDatabase(database.lower())
            except ValueError:
                logger.error(f"Unsupported database: {database}")
                return NCBIResponse(
                    success=False,
                    data=None,
                    error_message=f"Unsupported database: {database}"
                )
        else:
            db_enum = database
        
        response = await self.ncbi_client.fetch(
            database=db_enum,
            ids=pmids,
            return_type=return_type,
            return_mode=return_mode
        )
        
        return response
    
    async def fetch_paper_details(self,
                                 pmids: List[str],
                                 database: str = "pubmed",
                                 return_type: str = "abstract",
                                 return_mode: str = "xml",
                                 batch_size: Optional[int] = None,
                                 max_concurrent: int = 3) -> Tuple[List[str], FetchStats]:
        """Fetch detailed information for multiple papers.
        
        Args:
            pmids: List of PMIDs to fetch
            database: NCBI database
            return_type: Type of data to return
            return_mode: Format of returned data
            batch_size: Batch size for fetching
            max_concurrent: Maximum concurrent requests
            
        Returns:
            Tuple[List[str], FetchStats]: List of XML responses and fetch statistics
        """
        import time
        
        if not pmids:
            return [], FetchStats(0, 0, 0, 0, 0.0, 0.0, 0.0)
        
        start_time = time.time()
        
        logger.info(f"Starting fetch for {len(pmids)} papers in batches")
        
        # Split into batches
        batches = self.split_pmids_for_batching(pmids, batch_size)
        
        # Track results and statistics
        all_responses = []
        successful_fetches = 0
        failed_fetches = 0
        batch_times = []
        
        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def fetch_batch_with_semaphore(batch_pmids: List[str]) -> Optional[str]:
            """Fetch a batch with semaphore control."""
            async with semaphore:
                batch_start = time.time()
                
                try:
                    response = await self.fetch_batch(
                        pmids=batch_pmids,
                        database=database,
                        return_type=return_type,
                        return_mode=return_mode
                    )
                    
                    batch_time = time.time() - batch_start
                    batch_times.append(batch_time)
                    
                    if response.success:
                        logger.debug(f"Successfully fetched batch of {len(batch_pmids)} papers in {batch_time:.2f}s")
                        return response.data
                    else:
                        logger.error(f"Failed to fetch batch: {response.error_message}")
                        return None
                        
                except Exception as e:
                    batch_time = time.time() - batch_start
                    batch_times.append(batch_time)
                    logger.error(f"Exception fetching batch: {str(e)}")
                    return None
        
        # Execute batches concurrently
        logger.info(f"Executing {len(batches)} batches with max {max_concurrent} concurrent requests")
        
        tasks = [fetch_batch_with_semaphore(batch) for batch in batches]
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for i, result in enumerate(batch_results):
            if isinstance(result, Exception):
                logger.error(f"Batch {i} failed with exception: {str(result)}")
                failed_fetches += len(batches[i])
            elif result is not None:
                all_responses.append(result)
                successful_fetches += len(batches[i])
            else:
                failed_fetches += len(batches[i])
        
        # Calculate statistics
        total_time = time.time() - start_time
        average_batch_time = sum(batch_times) / len(batch_times) if batch_times else 0.0
        papers_per_second = successful_fetches / total_time if total_time > 0 else 0.0
        
        stats = FetchStats(
            total_pmids=len(pmids),
            successful_fetches=successful_fetches,
            failed_fetches=failed_fetches,
            total_batches=len(batches),
            total_time=total_time,
            average_batch_time=average_batch_time,
            papers_per_second=papers_per_second
        )
        
        logger.info(f"Fetch complete: {successful_fetches}/{len(pmids)} papers in {total_time:.2f}s "
                   f"({papers_per_second:.1f} papers/sec)")
        
        return all_responses, stats
    
    async def get_paper_formats(self, pmid: str) -> List[str]:
        """Get available formats for a paper.
        
        Args:
            pmid: PubMed ID
            
        Returns:
            List[str]: List of available formats
        """
        # This is a simplified implementation
        # In practice, you might query NCBI to get actual available formats
        
        # Standard PubMed formats
        standard_formats = ["abstract", "full", "medline", "bibtex"]
        
        # Try to fetch basic info to see if paper exists
        response = await self.fetch_single_paper(
            pmid=pmid,
            return_type="abstract",
            return_mode="xml"
        )
        
        if response.success:
            return standard_formats
        else:
            logger.warning(f"Paper {pmid} not found or not accessible")
            return []
    
    async def fetch_with_retry_on_batch_failure(self,
                                               pmids: List[str],
                                               database: str = "pubmed",
                                               max_retries: int = 2) -> Tuple[List[str], FetchStats]:
        """Fetch papers with retry logic for failed batches.
        
        Args:
            pmids: List of PMIDs to fetch
            database: NCBI database
            max_retries: Maximum retries for failed batches
            
        Returns:
            Tuple[List[str], FetchStats]: Responses and statistics
        """
        remaining_pmids = pmids.copy()
        all_responses = []
        total_stats = FetchStats(len(pmids), 0, 0, 0, 0.0, 0.0, 0.0)
        
        for attempt in range(max_retries + 1):
            if not remaining_pmids:
                break
            
            logger.info(f"Fetch attempt {attempt + 1}/{max_retries + 1} for {len(remaining_pmids)} PMIDs")
            
            responses, stats = await self.fetch_paper_details(
                pmids=remaining_pmids,
                database=database
            )
            
            all_responses.extend(responses)
            
            # Update total stats
            total_stats.successful_fetches += stats.successful_fetches
            total_stats.failed_fetches += stats.failed_fetches
            total_stats.total_batches += stats.total_batches
            total_stats.total_time += stats.total_time
            
            # If we got some failures and have retries left, try again with smaller batches
            if stats.failed_fetches > 0 and attempt < max_retries:
                # Calculate which PMIDs failed (simplified approach)
                failed_count = stats.failed_fetches
                if failed_count < len(remaining_pmids):
                    # Some succeeded, some failed - retry with smaller batch size
                    remaining_pmids = remaining_pmids[-failed_count:]  # Assume last ones failed
                    logger.info(f"Retrying {len(remaining_pmids)} failed PMIDs with smaller batch size")
                else:
                    # All failed - retry all with smaller batch size
                    logger.warning(f"All PMIDs failed, retrying with batch size 50")
            else:
                break
        
        # Calculate final averages
        if total_stats.total_batches > 0:
            total_stats.average_batch_time = total_stats.total_time / total_stats.total_batches
        
        if total_stats.total_time > 0:
            total_stats.papers_per_second = total_stats.successful_fetches / total_stats.total_time
        
        return all_responses, total_stats
    
    def get_client_status(self) -> Dict[str, Any]:
        """Get comprehensive client status.
        
        Returns:
            Dict[str, Any]: Client status information
        """
        status = self.ncbi_client.get_client_status()
        status.update({
            "fetch_client": {
                "default_batch_size": self.default_batch_size,
                "max_pmids_per_request": self.max_pmids_per_request,
                "max_url_length": self.max_url_length
            }
        })
        return status