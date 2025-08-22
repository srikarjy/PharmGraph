"""Search result processor with caching, deduplication, and ranking."""

import asyncio
import hashlib
import json
import time
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict
import re

from ..config import get_logger
from .ncbi_search import SearchResult, SearchResponse


logger = get_logger(__name__)


@dataclass
class ProcessedSearchResult(SearchResult):
    """Extended search result with processing metadata."""
    processing_time: float = 0.0
    duplicate_count: int = 0
    relevance_factors: Dict[str, float] = None
    keywords_matched: List[str] = None
    
    def __post_init__(self):
        if self.relevance_factors is None:
            self.relevance_factors = {}
        if self.keywords_matched is None:
            self.keywords_matched = []


@dataclass
class ProcessingStats:
    """Statistics from search result processing."""
    total_results: int
    unique_results: int
    duplicates_removed: int
    processing_time: float
    cache_hits: int
    cache_misses: int
    ranking_time: float


class SearchResultCache:
    """Simple in-memory cache for search results."""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        """Initialize cache.
        
        Args:
            max_size: Maximum number of cached queries
            ttl_seconds: Time-to-live for cached results in seconds
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: Dict[str, Tuple[List[SearchResult], float]] = {}
        self._access_times: Dict[str, float] = {}
        
        logger.info(f"Initialized SearchResultCache: max_size={max_size}, ttl={ttl_seconds}s")
    
    def _generate_cache_key(self, query: str, database: str, max_results: int) -> str:
        """Generate cache key for query parameters.
        
        Args:
            query: Search query
            database: Database name
            max_results: Maximum results
            
        Returns:
            str: Cache key
        """
        key_data = f"{query}|{database}|{max_results}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _cleanup_expired(self):
        """Remove expired entries from cache."""
        current_time = time.time()
        expired_keys = []
        
        for key, (_, timestamp) in self._cache.items():
            if current_time - timestamp > self.ttl_seconds:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self._cache[key]
            if key in self._access_times:
                del self._access_times[key]
        
        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
    
    def _evict_lru(self):
        """Evict least recently used entries if cache is full."""
        if len(self._cache) < self.max_size:
            return
        
        # Find least recently used key
        lru_key = min(self._access_times.keys(), key=lambda k: self._access_times[k])
        
        del self._cache[lru_key]
        del self._access_times[lru_key]
        
        logger.debug(f"Evicted LRU cache entry: {lru_key}")
    
    async def get(self, query: str, database: str, max_results: int) -> Optional[List[SearchResult]]:
        """Get cached results.
        
        Args:
            query: Search query
            database: Database name
            max_results: Maximum results
            
        Returns:
            Optional[List[SearchResult]]: Cached results if available
        """
        self._cleanup_expired()
        
        cache_key = self._generate_cache_key(query, database, max_results)
        
        if cache_key in self._cache:
            results, _ = self._cache[cache_key]
            self._access_times[cache_key] = time.time()
            logger.debug(f"Cache hit for query: {query[:50]}...")
            return results
        
        logger.debug(f"Cache miss for query: {query[:50]}...")
        return None
    
    async def set(self, query: str, database: str, max_results: int, results: List[SearchResult]):
        """Cache search results.
        
        Args:
            query: Search query
            database: Database name
            max_results: Maximum results
            results: Search results to cache
        """
        self._cleanup_expired()
        self._evict_lru()
        
        cache_key = self._generate_cache_key(query, database, max_results)
        current_time = time.time()
        
        self._cache[cache_key] = (results, current_time)
        self._access_times[cache_key] = current_time
        
        logger.debug(f"Cached {len(results)} results for query: {query[:50]}...")
    
    def clear(self):
        """Clear all cached results."""
        self._cache.clear()
        self._access_times.clear()
        logger.info("Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dict[str, Any]: Cache statistics
        """
        current_time = time.time()
        expired_count = sum(
            1 for _, timestamp in self._cache.values()
            if current_time - timestamp > self.ttl_seconds
        )
        
        return {
            "total_entries": len(self._cache),
            "expired_entries": expired_count,
            "max_size": self.max_size,
            "ttl_seconds": self.ttl_seconds
        }


class SearchResultProcessor:
    """Processor for NCBI search results with deduplication, ranking, and caching."""
    
    def __init__(self, 
                 cache_size: int = 1000,
                 cache_ttl: int = 3600,
                 enable_caching: bool = True):
        """Initialize search result processor.
        
        Args:
            cache_size: Maximum cache size
            cache_ttl: Cache time-to-live in seconds
            enable_caching: Whether to enable caching
        """
        self.enable_caching = enable_caching
        self.cache = SearchResultCache(cache_size, cache_ttl) if enable_caching else None
        
        # Relevance scoring weights
        self.relevance_weights = {
            "title_match": 3.0,
            "abstract_match": 2.0,
            "author_match": 1.5,
            "journal_impact": 1.0,
            "recency": 0.5,
            "citation_count": 0.3  # Placeholder for future implementation
        }
        
        logger.info(f"Initialized SearchResultProcessor: caching={enable_caching}")
    
    def _extract_keywords(self, text: str) -> Set[str]:
        """Extract keywords from text.
        
        Args:
            text: Input text
            
        Returns:
            Set[str]: Set of keywords
        """
        if not text:
            return set()
        
        # Simple keyword extraction (can be enhanced with NLP)
        # Remove punctuation and convert to lowercase
        cleaned_text = re.sub(r'[^\w\s]', ' ', text.lower())
        words = cleaned_text.split()
        
        # Filter out common stop words and short words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
            'may', 'might', 'can', 'this', 'that', 'these', 'those', 'from', 'as'
        }
        
        keywords = {word for word in words if len(word) > 2 and word not in stop_words}
        return keywords
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            float: Similarity score (0-1)
        """
        if not text1 or not text2:
            return 0.0
        
        keywords1 = self._extract_keywords(text1)
        keywords2 = self._extract_keywords(text2)
        
        if not keywords1 or not keywords2:
            return 0.0
        
        # Jaccard similarity
        intersection = keywords1.intersection(keywords2)
        union = keywords1.union(keywords2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _is_duplicate(self, result1: SearchResult, result2: SearchResult, 
                     threshold: float = 0.8) -> bool:
        """Check if two results are duplicates.
        
        Args:
            result1: First result
            result2: Second result
            threshold: Similarity threshold for duplicates
            
        Returns:
            bool: True if results are duplicates
        """
        # Same PMID is definitely a duplicate
        if result1.pmid == result2.pmid:
            return True
        
        # Check title similarity
        if result1.title and result2.title:
            title_similarity = self._calculate_text_similarity(result1.title, result2.title)
            if title_similarity >= threshold:
                return True
        
        # Check DOI if available
        if result1.doi and result2.doi and result1.doi == result2.doi:
            return True
        
        return False
    
    def deduplicate_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """Remove duplicate results.
        
        Args:
            results: List of search results
            
        Returns:
            List[SearchResult]: Deduplicated results
        """
        if not results:
            return results
        
        unique_results = []
        seen_pmids = set()
        
        for result in results:
            # Quick check by PMID first
            if result.pmid in seen_pmids:
                continue
            
            # Check for duplicates against existing unique results
            is_duplicate = False
            for unique_result in unique_results:
                if self._is_duplicate(result, unique_result):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_results.append(result)
                seen_pmids.add(result.pmid)
        
        duplicates_removed = len(results) - len(unique_results)
        if duplicates_removed > 0:
            logger.info(f"Removed {duplicates_removed} duplicate results")
        
        return unique_results
    
    def _calculate_relevance_score(self, result: SearchResult, query: str) -> float:
        """Calculate relevance score for a result.
        
        Args:
            result: Search result
            query: Original search query
            
        Returns:
            float: Relevance score
        """
        score = 0.0
        query_keywords = self._extract_keywords(query)
        
        # Title matching
        if result.title:
            title_keywords = self._extract_keywords(result.title)
            title_matches = query_keywords.intersection(title_keywords)
            title_score = len(title_matches) / len(query_keywords) if query_keywords else 0
            score += title_score * self.relevance_weights["title_match"]
        
        # Abstract matching
        if result.abstract:
            abstract_keywords = self._extract_keywords(result.abstract)
            abstract_matches = query_keywords.intersection(abstract_keywords)
            abstract_score = len(abstract_matches) / len(query_keywords) if query_keywords else 0
            score += abstract_score * self.relevance_weights["abstract_match"]
        
        # Journal impact (simplified - could be enhanced with actual impact factors)
        if result.journal:
            high_impact_journals = [
                "nature", "science", "cell", "new england journal of medicine",
                "lancet", "jama", "nature genetics", "nature medicine"
            ]
            if any(journal.lower() in result.journal.lower() for journal in high_impact_journals):
                score += self.relevance_weights["journal_impact"]
        
        # Recency (simplified - could be enhanced with actual date parsing)
        if result.pub_date:
            try:
                # Simple year extraction
                year_match = re.search(r'\b(20\d{2})\b', result.pub_date)
                if year_match:
                    year = int(year_match.group(1))
                    current_year = 2024  # Could use datetime.now().year
                    recency_score = max(0, 1 - (current_year - year) / 10)  # Decay over 10 years
                    score += recency_score * self.relevance_weights["recency"]
            except (ValueError, AttributeError):
                pass
        
        return score
    
    def rank_by_relevance(self, results: List[SearchResult], query: str) -> List[SearchResult]:
        """Rank results by relevance to query.
        
        Args:
            results: List of search results
            query: Original search query
            
        Returns:
            List[SearchResult]: Results ranked by relevance
        """
        if not results:
            return results
        
        start_time = time.time()
        
        # Calculate relevance scores
        for result in results:
            result.relevance_score = self._calculate_relevance_score(result, query)
        
        # Sort by relevance score (descending)
        ranked_results = sorted(results, key=lambda r: r.relevance_score, reverse=True)
        
        ranking_time = time.time() - start_time
        logger.debug(f"Ranked {len(results)} results in {ranking_time:.3f}s")
        
        return ranked_results
    
    async def process_search_results(self, 
                                   raw_results: Dict[str, Any],
                                   query: str = "",
                                   database: str = "pubmed") -> List[ProcessedSearchResult]:
        """Process raw NCBI search results.
        
        Args:
            raw_results: Raw search results from NCBI API
            query: Original search query
            database: Database searched
            
        Returns:
            List[ProcessedSearchResult]: Processed search results
        """
        start_time = time.time()
        
        # Extract basic search results
        search_result = raw_results.get('esearchresult', {})
        id_list = search_result.get('idlist', [])
        
        if not id_list:
            logger.warning("No results found in raw search data")
            return []
        
        # Create SearchResult objects
        results = []
        for i, pmid in enumerate(id_list):
            result = ProcessedSearchResult(
                pmid=pmid,
                database=database,
                relevance_score=1.0 - (i * 0.01)  # Simple initial scoring
            )
            results.append(result)
        
        processing_time = time.time() - start_time
        
        # Set processing metadata
        for result in results:
            result.processing_time = processing_time
        
        logger.info(f"Processed {len(results)} search results in {processing_time:.3f}s")
        return results
    
    async def cache_search_results(self, 
                                 query: str, 
                                 database: str,
                                 max_results: int,
                                 results: List[SearchResult]):
        """Cache search results.
        
        Args:
            query: Search query
            database: Database name
            max_results: Maximum results
            results: Results to cache
        """
        if self.cache:
            await self.cache.set(query, database, max_results, results)
    
    async def get_cached_results(self, 
                               query: str, 
                               database: str,
                               max_results: int) -> Optional[List[SearchResult]]:
        """Get cached search results.
        
        Args:
            query: Search query
            database: Database name
            max_results: Maximum results
            
        Returns:
            Optional[List[SearchResult]]: Cached results if available
        """
        if self.cache:
            return await self.cache.get(query, database, max_results)
        return None
    
    async def process_complete_search(self,
                                    raw_results: Dict[str, Any],
                                    query: str,
                                    database: str = "pubmed",
                                    deduplicate: bool = True,
                                    rank_results: bool = True,
                                    cache_results: bool = True) -> Tuple[List[ProcessedSearchResult], ProcessingStats]:
        """Complete processing pipeline for search results.
        
        Args:
            raw_results: Raw search results from NCBI API
            query: Original search query
            database: Database searched
            deduplicate: Whether to remove duplicates
            rank_results: Whether to rank by relevance
            cache_results: Whether to cache results
            
        Returns:
            Tuple[List[ProcessedSearchResult], ProcessingStats]: Processed results and stats
        """
        start_time = time.time()
        cache_hits = 0
        cache_misses = 1  # This call is a cache miss
        
        # Check cache first
        max_results = len(raw_results.get('esearchresult', {}).get('idlist', []))
        cached_results = await self.get_cached_results(query, database, max_results)
        
        if cached_results and cache_results:
            cache_hits = 1
            cache_misses = 0
            logger.info(f"Using cached results for query: {query[:50]}...")
            
            stats = ProcessingStats(
                total_results=len(cached_results),
                unique_results=len(cached_results),
                duplicates_removed=0,
                processing_time=time.time() - start_time,
                cache_hits=cache_hits,
                cache_misses=cache_misses,
                ranking_time=0.0
            )
            
            return cached_results, stats
        
        # Process results
        results = await self.process_search_results(raw_results, query, database)
        total_results = len(results)
        
        # Deduplicate if requested
        if deduplicate:
            results = self.deduplicate_results(results)
        
        unique_results = len(results)
        duplicates_removed = total_results - unique_results
        
        # Rank by relevance if requested
        ranking_start = time.time()
        if rank_results:
            results = self.rank_by_relevance(results, query)
        ranking_time = time.time() - ranking_start
        
        # Cache results if requested
        if cache_results:
            await self.cache_search_results(query, database, max_results, results)
        
        processing_time = time.time() - start_time
        
        stats = ProcessingStats(
            total_results=total_results,
            unique_results=unique_results,
            duplicates_removed=duplicates_removed,
            processing_time=processing_time,
            cache_hits=cache_hits,
            cache_misses=cache_misses,
            ranking_time=ranking_time
        )
        
        logger.info(f"Complete processing finished: {unique_results} unique results from {total_results} total")
        
        return results, stats
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics.
        
        Returns:
            Dict[str, Any]: Processing statistics
        """
        stats = {
            "caching_enabled": self.enable_caching,
            "relevance_weights": self.relevance_weights
        }
        
        if self.cache:
            stats["cache_stats"] = self.cache.get_stats()
        
        return stats
    
    def clear_cache(self):
        """Clear the result cache."""
        if self.cache:
            self.cache.clear()
            logger.info("Search result cache cleared")