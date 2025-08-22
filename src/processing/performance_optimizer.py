"""Performance optimization layer for pharmacogenomics pipeline."""

import asyncio
import time
import gc
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict
import json
import hashlib
from concurrent.futures import ThreadPoolExecutor
import threading

from ..config import get_logger, get_config
from ..data_ingestion.data_models import PaperMetadata
from ..quality.quality_scorer import QualityScoreComponents
from ..storage.database import get_database_manager, db_session_scope
from ..storage.models import PaperModel, QualityMetrics


logger = get_logger(__name__)


@dataclass
class OptimizationMetrics:
    """Performance optimization metrics."""
    cache_hit_rate: float = 0.0
    cache_size: int = 0
    memory_usage_mb: float = 0.0
    concurrent_operations: int = 0
    batch_efficiency: float = 0.0
    db_connection_pool_usage: float = 0.0
    
    # Timing metrics
    avg_api_call_time: float = 0.0
    avg_db_insert_time: float = 0.0
    avg_scoring_time: float = 0.0
    
    # Throughput metrics
    papers_per_second: float = 0.0
    api_calls_per_second: float = 0.0
    db_operations_per_second: float = 0.0


class MemoryOptimizer:
    """Memory management and optimization."""
    
    def __init__(self, max_memory_mb: int = 2048):
        """Initialize memory optimizer.
        
        Args:
            max_memory_mb: Maximum memory usage in MB
        """
        self.max_memory_mb = max_memory_mb
        self.memory_threshold_mb = max_memory_mb * 0.8  # 80% threshold
        self.gc_frequency = 100  # Run GC every N operations
        self.operation_count = 0
        
    def check_memory_usage(self) -> float:
        """Check current memory usage."""
        import psutil
        process = psutil.Process()
        memory_mb = process.memory_info().rss / (1024 * 1024)
        return memory_mb
    
    def optimize_memory(self, force: bool = False):
        """Optimize memory usage."""
        self.operation_count += 1
        
        current_memory = self.check_memory_usage()
        
        if force or current_memory > self.memory_threshold_mb or self.operation_count % self.gc_frequency == 0:
            # Force garbage collection
            collected = gc.collect()
            
            new_memory = self.check_memory_usage()
            freed_mb = current_memory - new_memory
            
            if freed_mb > 10:  # Only log if significant memory was freed
                logger.debug(f"Memory optimization: freed {freed_mb:.1f}MB, "
                           f"collected {collected} objects")
            
            if new_memory > self.max_memory_mb:
                logger.warning(f"Memory usage {new_memory:.1f}MB exceeds limit {self.max_memory_mb}MB")
    
    def stream_process_papers(self, papers: List[PaperMetadata], 
                            batch_size: int = 50) -> List[List[PaperMetadata]]:
        """Stream process papers in memory-efficient batches."""
        batches = []
        for i in range(0, len(papers), batch_size):
            batch = papers[i:i + batch_size]
            batches.append(batch)
            
            # Optimize memory after each batch
            if i % (batch_size * 5) == 0:  # Every 5 batches
                self.optimize_memory()
        
        return batches


class CacheManager:
    """Intelligent caching for pipeline components."""
    
    def __init__(self, max_cache_size: int = 10000):
        """Initialize cache manager.
        
        Args:
            max_cache_size: Maximum number of cached items
        """
        self.max_cache_size = max_cache_size
        self.journal_scores_cache: Dict[str, float] = {}
        self.mesh_term_cache: Dict[str, Dict[str, Any]] = {}
        self.quality_score_cache: Dict[str, QualityScoreComponents] = {}
        self.paper_metadata_cache: Dict[str, PaperMetadata] = {}
        
        # Cache statistics
        self.cache_hits = defaultdict(int)
        self.cache_misses = defaultdict(int)
        self.cache_lock = threading.RLock()
    
    def _generate_cache_key(self, *args) -> str:
        """Generate cache key from arguments."""
        key_string = "|".join(str(arg) for arg in args)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def get_journal_score(self, journal_name: str) -> Optional[float]:
        """Get cached journal score."""
        with self.cache_lock:
            if journal_name in self.journal_scores_cache:
                self.cache_hits['journal_scores'] += 1
                return self.journal_scores_cache[journal_name]
            
            self.cache_misses['journal_scores'] += 1
            return None
    
    def cache_journal_score(self, journal_name: str, score: float):
        """Cache journal score."""
        with self.cache_lock:
            if len(self.journal_scores_cache) >= self.max_cache_size:
                # Remove oldest entries (simple LRU approximation)
                keys_to_remove = list(self.journal_scores_cache.keys())[:100]
                for key in keys_to_remove:
                    del self.journal_scores_cache[key]
            
            self.journal_scores_cache[journal_name] = score
    
    def get_mesh_term_info(self, mesh_term: str) -> Optional[Dict[str, Any]]:
        """Get cached MeSH term information."""
        with self.cache_lock:
            if mesh_term in self.mesh_term_cache:
                self.cache_hits['mesh_terms'] += 1
                return self.mesh_term_cache[mesh_term]
            
            self.cache_misses['mesh_terms'] += 1
            return None
    
    def cache_mesh_term_info(self, mesh_term: str, info: Dict[str, Any]):
        """Cache MeSH term information."""
        with self.cache_lock:
            if len(self.mesh_term_cache) >= self.max_cache_size:
                keys_to_remove = list(self.mesh_term_cache.keys())[:100]
                for key in keys_to_remove:
                    del self.mesh_term_cache[key]
            
            self.mesh_term_cache[mesh_term] = info
    
    def get_quality_score(self, paper_key: str) -> Optional[QualityScoreComponents]:
        """Get cached quality score."""
        with self.cache_lock:
            if paper_key in self.quality_score_cache:
                self.cache_hits['quality_scores'] += 1
                return self.quality_score_cache[paper_key]
            
            self.cache_misses['quality_scores'] += 1
            return None
    
    def cache_quality_score(self, paper_key: str, score: QualityScoreComponents):
        """Cache quality score."""
        with self.cache_lock:
            if len(self.quality_score_cache) >= self.max_cache_size:
                keys_to_remove = list(self.quality_score_cache.keys())[:100]
                for key in keys_to_remove:
                    del self.quality_score_cache[key]
            
            self.quality_score_cache[paper_key] = score
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.cache_lock:
            stats = {}
            for cache_type in ['journal_scores', 'mesh_terms', 'quality_scores']:
                hits = self.cache_hits[cache_type]
                misses = self.cache_misses[cache_type]
                total = hits + misses
                hit_rate = (hits / total * 100) if total > 0 else 0
                
                stats[cache_type] = {
                    'hits': hits,
                    'misses': misses,
                    'hit_rate': hit_rate
                }
            
            stats['cache_sizes'] = {
                'journal_scores': len(self.journal_scores_cache),
                'mesh_terms': len(self.mesh_term_cache),
                'quality_scores': len(self.quality_score_cache)
            }
            
            return stats
    
    def clear_cache(self, cache_type: str = None):
        """Clear cache."""
        with self.cache_lock:
            if cache_type == 'journal_scores' or cache_type is None:
                self.journal_scores_cache.clear()
            if cache_type == 'mesh_terms' or cache_type is None:
                self.mesh_term_cache.clear()
            if cache_type == 'quality_scores' or cache_type is None:
                self.quality_score_cache.clear()
            
            logger.info(f"Cleared cache: {cache_type or 'all'}")


class ConcurrencyManager:
    """Manage concurrent operations and resource allocation."""
    
    def __init__(self, max_concurrent_api: int = 5, max_concurrent_db: int = 10):
        """Initialize concurrency manager.
        
        Args:
            max_concurrent_api: Maximum concurrent API calls
            max_concurrent_db: Maximum concurrent database operations
        """
        self.max_concurrent_api = max_concurrent_api
        self.max_concurrent_db = max_concurrent_db
        
        # Semaphores for resource management
        self.api_semaphore = asyncio.Semaphore(max_concurrent_api)
        self.db_semaphore = asyncio.Semaphore(max_concurrent_db)
        
        # Thread pool for CPU-intensive tasks
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        
        # Active operation tracking
        self.active_api_calls = 0
        self.active_db_operations = 0
        self.operation_lock = threading.Lock()
    
    async def execute_api_call(self, api_func, *args, **kwargs):
        """Execute API call with concurrency control."""
        async with self.api_semaphore:
            with self.operation_lock:
                self.active_api_calls += 1
            
            try:
                start_time = time.time()
                result = await api_func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                logger.debug(f"API call completed in {execution_time:.2f}s")
                return result, execution_time
                
            finally:
                with self.operation_lock:
                    self.active_api_calls -= 1
    
    async def execute_db_operation(self, db_func, *args, **kwargs):
        """Execute database operation with concurrency control."""
        async with self.db_semaphore:
            with self.operation_lock:
                self.active_db_operations += 1
            
            try:
                start_time = time.time()
                result = await db_func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                logger.debug(f"DB operation completed in {execution_time:.2f}s")
                return result, execution_time
                
            finally:
                with self.operation_lock:
                    self.active_db_operations -= 1
    
    def execute_cpu_task(self, cpu_func, *args, **kwargs):
        """Execute CPU-intensive task in thread pool."""
        loop = asyncio.get_event_loop()
        return loop.run_in_executor(self.thread_pool, cpu_func, *args, **kwargs)
    
    def get_resource_usage(self) -> Dict[str, Any]:
        """Get current resource usage."""
        with self.operation_lock:
            return {
                'active_api_calls': self.active_api_calls,
                'active_db_operations': self.active_db_operations,
                'api_utilization': (self.active_api_calls / self.max_concurrent_api) * 100,
                'db_utilization': (self.active_db_operations / self.max_concurrent_db) * 100
            }


class BatchOptimizer:
    """Optimize batch processing for maximum efficiency."""
    
    def __init__(self, target_batch_size: int = 200, min_batch_size: int = 50):
        """Initialize batch optimizer.
        
        Args:
            target_batch_size: Target batch size for optimal performance
            min_batch_size: Minimum batch size
        """
        self.target_batch_size = target_batch_size
        self.min_batch_size = min_batch_size
        self.performance_history = []
        self.optimal_batch_size = target_batch_size
    
    def optimize_batch_size(self, papers: List[Any]) -> List[List[Any]]:
        """Optimize batch size based on performance history."""
        if len(papers) <= self.min_batch_size:
            return [papers]
        
        # Use adaptive batch sizing based on performance
        current_batch_size = self._calculate_optimal_batch_size(len(papers))
        
        batches = []
        for i in range(0, len(papers), current_batch_size):
            batch = papers[i:i + current_batch_size]
            batches.append(batch)
        
        logger.debug(f"Created {len(batches)} batches with size ~{current_batch_size}")
        return batches
    
    def _calculate_optimal_batch_size(self, total_items: int) -> int:
        """Calculate optimal batch size based on performance history."""
        if not self.performance_history:
            return self.target_batch_size
        
        # Analyze recent performance
        recent_performance = self.performance_history[-10:]  # Last 10 batches
        
        if recent_performance:
            # Find batch size with best throughput
            best_throughput = 0
            best_batch_size = self.target_batch_size
            
            for perf in recent_performance:
                throughput = perf['items_processed'] / perf['processing_time']
                if throughput > best_throughput:
                    best_throughput = throughput
                    best_batch_size = perf['batch_size']
            
            # Adjust optimal batch size gradually
            self.optimal_batch_size = int(0.8 * self.optimal_batch_size + 0.2 * best_batch_size)
        
        # Ensure batch size is within bounds
        return max(self.min_batch_size, min(self.optimal_batch_size, total_items))
    
    def record_batch_performance(self, batch_size: int, processing_time: float, 
                               items_processed: int, success_rate: float):
        """Record batch performance for optimization."""
        performance_record = {
            'batch_size': batch_size,
            'processing_time': processing_time,
            'items_processed': items_processed,
            'success_rate': success_rate,
            'throughput': items_processed / processing_time if processing_time > 0 else 0,
            'timestamp': time.time()
        }
        
        self.performance_history.append(performance_record)
        
        # Keep only recent history
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-50:]
    
    def get_batch_recommendations(self) -> Dict[str, Any]:
        """Get batch size recommendations."""
        if not self.performance_history:
            return {
                'recommended_batch_size': self.target_batch_size,
                'confidence': 0.0,
                'reason': 'No performance history available'
            }
        
        recent_performance = self.performance_history[-20:]
        
        # Calculate average throughput by batch size
        throughput_by_size = defaultdict(list)
        for perf in recent_performance:
            throughput_by_size[perf['batch_size']].append(perf['throughput'])
        
        # Find best performing batch size
        best_size = self.target_batch_size
        best_avg_throughput = 0
        
        for batch_size, throughputs in throughput_by_size.items():
            avg_throughput = sum(throughputs) / len(throughputs)
            if avg_throughput > best_avg_throughput:
                best_avg_throughput = avg_throughput
                best_size = batch_size
        
        confidence = len(recent_performance) / 20.0  # Confidence based on data points
        
        return {
            'recommended_batch_size': best_size,
            'confidence': confidence,
            'avg_throughput': best_avg_throughput,
            'reason': f'Based on {len(recent_performance)} recent batches'
        }


class DatabaseOptimizer:
    """Database operation optimization."""
    
    def __init__(self, bulk_insert_size: int = 100):
        """Initialize database optimizer.
        
        Args:
            bulk_insert_size: Size for bulk insert operations
        """
        self.bulk_insert_size = bulk_insert_size
        self.pending_inserts = []
        self.insert_lock = threading.Lock()
    
    async def bulk_insert_papers(self, papers_and_scores: List[Tuple[PaperMetadata, QualityScoreComponents]]) -> int:
        """Perform bulk insert of papers with optimization."""
        if not papers_and_scores:
            return 0
        
        start_time = time.time()
        stored_count = 0
        
        try:
            # Initialize database
            db_manager = get_database_manager()
            db_manager.initialize()
            
            # Process in optimized batches
            batch_size = min(self.bulk_insert_size, len(papers_and_scores))
            
            with db_session_scope() as session:
                # Prepare bulk insert data
                paper_models = []
                quality_models = []
                
                for paper, scores in papers_and_scores:
                    # Check for duplicates in batch
                    existing = session.query(PaperModel).filter_by(pmid=paper.pmid).first()
                    if existing:
                        logger.debug(f"Paper {paper.pmid} already exists, skipping")
                        continue
                    
                    # Convert to database models
                    db_paper = self._convert_to_paper_model(paper)
                    paper_models.append(db_paper)
                    
                    quality_metrics = QualityMetrics(
                        pmid=paper.pmid,
                        journal_score=scores.journal_score,
                        clinical_relevance_score=scores.clinical_relevance_score,
                        ml_methodology_score=scores.ml_methodology_score,
                        recency_score=scores.recency_score,
                        total_score=scores.total_score,
                        confidence_level=scores.confidence_level,
                        scoring_version=scores.scoring_version
                    )
                    quality_models.append(quality_metrics)
                
                # Bulk insert papers
                if paper_models:
                    session.bulk_save_objects(paper_models)
                    session.bulk_save_objects(quality_models)
                    session.commit()
                    stored_count = len(paper_models)
            
            execution_time = time.time() - start_time
            
            if stored_count > 0:
                papers_per_second = stored_count / execution_time
                logger.info(f"Bulk inserted {stored_count} papers in {execution_time:.2f}s "
                          f"({papers_per_second:.1f} papers/sec)")
            
            return stored_count
            
        except Exception as e:
            logger.error(f"Bulk insert failed: {str(e)}")
            raise
    
    def _convert_to_paper_model(self, paper: PaperMetadata) -> PaperModel:
        """Convert PaperMetadata to database model."""
        # Convert authors to JSON
        authors_json = [
            {
                "last_name": author.last_name,
                "first_name": author.first_name,
                "initials": author.initials,
                "affiliation": author.affiliation,
                "orcid": author.orcid
            }
            for author in paper.authors
        ]
        
        # Convert MeSH terms to JSON
        mesh_terms_json = [
            {
                "descriptor_name": term.descriptor_name,
                "qualifier_name": term.qualifier_name,
                "major_topic": term.major_topic
            }
            for term in paper.mesh_terms
        ]
        
        return PaperModel(
            pmid=paper.pmid,
            title=paper.title,
            abstract=paper.abstract,
            authors_json=authors_json,
            journal=paper.journal.title if paper.journal else None,
            publication_date=paper.publication_date.date() if paper.publication_date else None,
            doi=paper.doi,
            mesh_terms_json=mesh_terms_json,
            keywords_json=paper.keywords
        )


class PerformanceOptimizer:
    """Main performance optimization coordinator."""
    
    def __init__(self, 
                 max_memory_mb: int = 2048,
                 max_concurrent_api: int = 5,
                 max_concurrent_db: int = 10,
                 target_batch_size: int = 200,
                 cache_size: int = 10000):
        """Initialize performance optimizer.
        
        Args:
            max_memory_mb: Maximum memory usage
            max_concurrent_api: Maximum concurrent API calls
            max_concurrent_db: Maximum concurrent DB operations
            target_batch_size: Target batch size
            cache_size: Cache size
        """
        self.memory_optimizer = MemoryOptimizer(max_memory_mb)
        self.cache_manager = CacheManager(cache_size)
        self.concurrency_manager = ConcurrencyManager(max_concurrent_api, max_concurrent_db)
        self.batch_optimizer = BatchOptimizer(target_batch_size)
        self.db_optimizer = DatabaseOptimizer()
        
        # Performance tracking
        self.start_time = time.time()
        self.total_papers_processed = 0
        self.total_api_calls = 0
        self.total_db_operations = 0
        
        logger.info("PerformanceOptimizer initialized")
    
    async def optimize_paper_processing(self, papers: List[PaperMetadata], 
                                      processing_func) -> List[Any]:
        """Optimize paper processing with all optimizations."""
        start_time = time.time()
        
        # Memory optimization
        self.memory_optimizer.optimize_memory()
        
        # Batch optimization
        batches = self.batch_optimizer.optimize_batch_size(papers)
        
        # Process batches concurrently
        results = []
        for batch in batches:
            batch_start = time.time()
            
            # Process batch with concurrency control
            batch_result = await self.concurrency_manager.execute_api_call(
                processing_func, batch
            )
            
            results.extend(batch_result[0] if isinstance(batch_result, tuple) else batch_result)
            
            # Record batch performance
            batch_time = time.time() - batch_start
            self.batch_optimizer.record_batch_performance(
                len(batch), batch_time, len(batch), 1.0  # Assume 100% success for now
            )
            
            # Memory optimization after each batch
            self.memory_optimizer.optimize_memory()
        
        # Update global metrics
        processing_time = time.time() - start_time
        self.total_papers_processed += len(papers)
        
        logger.info(f"Optimized processing of {len(papers)} papers in {processing_time:.2f}s")
        
        return results
    
    async def optimize_database_storage(self, papers_and_scores: List[Tuple[PaperMetadata, QualityScoreComponents]]) -> int:
        """Optimize database storage operations."""
        return await self.db_optimizer.bulk_insert_papers(papers_and_scores)
    
    def get_optimization_metrics(self) -> OptimizationMetrics:
        """Get current optimization metrics."""
        cache_stats = self.cache_manager.get_cache_stats()
        resource_usage = self.concurrency_manager.get_resource_usage()
        memory_usage = self.memory_optimizer.check_memory_usage()
        
        # Calculate overall cache hit rate
        total_hits = sum(stats['hits'] for stats in cache_stats.values() if isinstance(stats, dict))
        total_requests = sum(stats['hits'] + stats['misses'] for stats in cache_stats.values() if isinstance(stats, dict))
        overall_hit_rate = (total_hits / total_requests * 100) if total_requests > 0 else 0
        
        # Calculate throughput
        elapsed_time = time.time() - self.start_time
        papers_per_second = self.total_papers_processed / elapsed_time if elapsed_time > 0 else 0
        
        return OptimizationMetrics(
            cache_hit_rate=overall_hit_rate,
            cache_size=sum(cache_stats['cache_sizes'].values()),
            memory_usage_mb=memory_usage,
            concurrent_operations=resource_usage['active_api_calls'] + resource_usage['active_db_operations'],
            batch_efficiency=self.batch_optimizer.optimal_batch_size / self.batch_optimizer.target_batch_size * 100,
            papers_per_second=papers_per_second
        )
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        metrics = self.get_optimization_metrics()
        cache_stats = self.cache_manager.get_cache_stats()
        resource_usage = self.concurrency_manager.get_resource_usage()
        batch_recommendations = self.batch_optimizer.get_batch_recommendations()
        
        elapsed_time = time.time() - self.start_time
        
        return {
            'optimization_metrics': {
                'cache_hit_rate': metrics.cache_hit_rate,
                'memory_usage_mb': metrics.memory_usage_mb,
                'papers_per_second': metrics.papers_per_second,
                'batch_efficiency': metrics.batch_efficiency
            },
            'cache_performance': cache_stats,
            'resource_utilization': resource_usage,
            'batch_optimization': batch_recommendations,
            'overall_stats': {
                'total_papers_processed': self.total_papers_processed,
                'total_runtime_minutes': elapsed_time / 60,
                'avg_papers_per_minute': (self.total_papers_processed / elapsed_time * 60) if elapsed_time > 0 else 0
            }
        }
    
    def reset_metrics(self):
        """Reset performance metrics."""
        self.start_time = time.time()
        self.total_papers_processed = 0
        self.total_api_calls = 0
        self.total_db_operations = 0
        self.cache_manager.clear_cache()
        
        logger.info("Performance metrics reset")