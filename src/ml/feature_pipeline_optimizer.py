"""Feature pipeline optimization for large-scale processing."""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import threading
import time
import psutil
import hashlib
import pickle
from functools import lru_cache, wraps
import gc

from ..config import get_logger
from .feature_extraction import FeatureExtractor, FeatureSet
from .text_features import AdvancedTextFeatureExtractor
from .pgx_features import PharmacogenomicsFeatureExtractor
from ..data_ingestion.data_models import PaperMetadata
from ..quality.quality_scorer import QualityScoreComponents


logger = get_logger(__name__)


@dataclass
class ProcessingMetrics:
    """Metrics for feature processing performance."""
    total_papers: int
    processing_time: float
    papers_per_second: float
    memory_usage_mb: float
    cpu_usage_percent: float
    cache_hit_rate: float
    parallel_efficiency: float
    feature_extraction_time: float
    feature_validation_time: float


@dataclass
class OptimizationConfig:
    """Configuration for feature pipeline optimization."""
    max_workers: int = mp.cpu_count()
    batch_size: int = 100
    enable_caching: bool = True
    cache_size: int = 1000
    memory_limit_mb: int = 4096
    enable_parallel_processing: bool = True
    chunk_size: int = 50
    gc_frequency: int = 100
    enable_profiling: bool = False


class FeatureCache:
    """Thread-safe feature caching system."""
    
    def __init__(self, max_size: int = 1000):
        """Initialize feature cache.
        
        Args:
            max_size: Maximum number of cached items
        """
        self.max_size = max_size
        self.cache = {}
        self.access_times = {}
        self.lock = threading.RLock()
        self.hit_count = 0
        self.miss_count = 0
    
    def _generate_key(self, papers: List[PaperMetadata], 
                     quality_scores: List[QualityScoreComponents],
                     feature_types: List[str]) -> str:
        """Generate cache key for papers and parameters."""
        # Create hash from paper PMIDs and feature types
        pmids = [paper.pmid for paper in papers]
        key_data = {
            'pmids': sorted(pmids),
            'feature_types': sorted(feature_types),
            'quality_count': len(quality_scores)
        }
        
        key_string = str(key_data)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def get(self, papers: List[PaperMetadata], 
            quality_scores: List[QualityScoreComponents],
            feature_types: List[str]) -> Optional[FeatureSet]:
        """Get cached features if available.
        
        Args:
            papers: List of papers
            quality_scores: List of quality scores
            feature_types: List of feature types
            
        Returns:
            Cached FeatureSet or None
        """
        key = self._generate_key(papers, quality_scores, feature_types)
        
        with self.lock:
            if key in self.cache:
                self.hit_count += 1
                self.access_times[key] = time.time()
                return self.cache[key]
            else:
                self.miss_count += 1
                return None
    
    def put(self, papers: List[PaperMetadata], 
            quality_scores: List[QualityScoreComponents],
            feature_types: List[str], features: FeatureSet) -> None:
        """Cache features.
        
        Args:
            papers: List of papers
            quality_scores: List of quality scores
            feature_types: List of feature types
            features: Features to cache
        """
        key = self._generate_key(papers, quality_scores, feature_types)
        
        with self.lock:
            # Implement LRU eviction if cache is full
            if len(self.cache) >= self.max_size:
                # Remove least recently used item
                lru_key = min(self.access_times.keys(), 
                             key=lambda k: self.access_times[k])
                del self.cache[lru_key]
                del self.access_times[lru_key]
            
            self.cache[key] = features
            self.access_times[key] = time.time()
    
    def get_hit_rate(self) -> float:
        """Get cache hit rate."""
        total = self.hit_count + self.miss_count
        return self.hit_count / total if total > 0 else 0.0
    
    def clear(self) -> None:
        """Clear cache."""
        with self.lock:
            self.cache.clear()
            self.access_times.clear()
            self.hit_count = 0
            self.miss_count = 0


class MemoryManager:
    """Manages memory usage during feature extraction."""
    
    def __init__(self, memory_limit_mb: int = 4096):
        """Initialize memory manager.
        
        Args:
            memory_limit_mb: Memory limit in MB
        """
        self.memory_limit_mb = memory_limit_mb
        self.memory_limit_bytes = memory_limit_mb * 1024 * 1024
        self.gc_frequency = 100
        self.operation_count = 0
    
    def check_memory_usage(self) -> float:
        """Check current memory usage in MB."""
        process = psutil.Process()
        memory_mb = process.memory_info().rss / (1024 * 1024)
        return memory_mb
    
    def optimize_memory(self, force: bool = False) -> None:
        """Optimize memory usage."""
        self.operation_count += 1
        
        current_memory = self.check_memory_usage()
        
        if force or current_memory > self.memory_limit_mb * 0.8 or \
           self.operation_count % self.gc_frequency == 0:
            
            # Force garbage collection
            collected = gc.collect()
            
            new_memory = self.check_memory_usage()
            freed_mb = current_memory - new_memory
            
            if freed_mb > 10:  # Only log if significant memory was freed
                logger.debug(f"Memory optimization: freed {freed_mb:.1f}MB, "
                           f"collected {collected} objects")
            
            if new_memory > self.memory_limit_mb:
                logger.warning(f"Memory usage {new_memory:.1f}MB exceeds limit "
                             f"{self.memory_limit_mb}MB")
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Get memory statistics."""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            'rss_mb': memory_info.rss / (1024 * 1024),
            'vms_mb': memory_info.vms / (1024 * 1024),
            'percent': process.memory_percent(),
            'available_mb': psutil.virtual_memory().available / (1024 * 1024)
        }


class ParallelFeatureProcessor:
    """Parallel processing for feature extraction."""
    
    def __init__(self, max_workers: int = None, chunk_size: int = 50):
        """Initialize parallel processor.
        
        Args:
            max_workers: Maximum number of worker processes
            chunk_size: Size of processing chunks
        """
        self.max_workers = max_workers or mp.cpu_count()
        self.chunk_size = chunk_size
    
    def process_papers_parallel(self, papers: List[PaperMetadata],
                               quality_scores: List[QualityScoreComponents],
                               feature_extractor: FeatureExtractor,
                               feature_types: List[str]) -> FeatureSet:
        """Process papers in parallel.
        
        Args:
            papers: List of papers to process
            quality_scores: List of quality scores
            feature_extractor: Feature extractor instance
            feature_types: Types of features to extract
            
        Returns:
            Combined FeatureSet
        """
        if len(papers) < self.chunk_size:
            # Not worth parallelizing for small datasets
            return feature_extractor.extract_features(papers, quality_scores, feature_types)
        
        # Split data into chunks
        chunks = self._create_chunks(papers, quality_scores)
        
        # Process chunks in parallel
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            
            for paper_chunk, quality_chunk in chunks:
                future = executor.submit(
                    self._process_chunk,
                    paper_chunk,
                    quality_chunk,
                    feature_types
                )
                futures.append(future)
            
            # Collect results
            feature_sets = []
            for future in as_completed(futures):
                try:
                    feature_set = future.result()
                    feature_sets.append(feature_set)
                except Exception as e:
                    logger.error(f"Chunk processing failed: {e}")
        
        # Combine feature sets
        return self._combine_feature_sets(feature_sets)
    
    def _create_chunks(self, papers: List[PaperMetadata],
                      quality_scores: List[QualityScoreComponents]) -> List[Tuple[List[PaperMetadata], List[QualityScoreComponents]]]:
        """Create processing chunks."""
        chunks = []
        
        for i in range(0, len(papers), self.chunk_size):
            end_idx = min(i + self.chunk_size, len(papers))
            paper_chunk = papers[i:end_idx]
            quality_chunk = quality_scores[i:end_idx] if quality_scores else []
            chunks.append((paper_chunk, quality_chunk))
        
        return chunks
    
    def _process_chunk(self, papers: List[PaperMetadata],
                      quality_scores: List[QualityScoreComponents],
                      feature_types: List[str]) -> FeatureSet:
        """Process a single chunk."""
        # Create new extractor instance for this process
        extractor = FeatureExtractor()
        return extractor.extract_features(papers, quality_scores, feature_types)
    
    def _combine_feature_sets(self, feature_sets: List[FeatureSet]) -> FeatureSet:
        """Combine multiple feature sets."""
        if not feature_sets:
            raise ValueError("No feature sets to combine")
        
        if len(feature_sets) == 1:
            return feature_sets[0]
        
        # Combine features
        combined_features = {}
        combined_metadata = {}
        combined_sample_ids = []
        combined_feature_names = []
        
        for feature_set in feature_sets:
            # Combine sample IDs
            combined_sample_ids.extend(feature_set.sample_ids)
            
            # Combine features
            for name, values in feature_set.features.items():
                if name not in combined_features:
                    combined_features[name] = []
                    combined_metadata[name] = feature_set.metadata.get(name)
                    if name not in combined_feature_names:
                        combined_feature_names.append(name)
                
                combined_features[name].extend(values)
        
        # Convert to numpy arrays
        for name in combined_features:
            combined_features[name] = np.array(combined_features[name])
        
        return FeatureSet(
            features=combined_features,
            metadata=combined_metadata,
            feature_names=combined_feature_names,
            sample_ids=combined_sample_ids,
            extraction_timestamp=datetime.now(),
            version_hash=""
        )


class PerformanceProfiler:
    """Profiles feature extraction performance."""
    
    def __init__(self):
        """Initialize performance profiler."""
        self.metrics = {}
        self.start_times = {}
    
    def start_timer(self, operation: str) -> None:
        """Start timing an operation."""
        self.start_times[operation] = time.time()
    
    def end_timer(self, operation: str) -> float:
        """End timing an operation and return duration."""
        if operation not in self.start_times:
            return 0.0
        
        duration = time.time() - self.start_times[operation]
        
        if operation not in self.metrics:
            self.metrics[operation] = []
        
        self.metrics[operation].append(duration)
        return duration
    
    def get_metrics(self) -> Dict[str, Dict[str, float]]:
        """Get performance metrics."""
        result = {}
        
        for operation, times in self.metrics.items():
            result[operation] = {
                'count': len(times),
                'total_time': sum(times),
                'avg_time': np.mean(times),
                'min_time': min(times),
                'max_time': max(times),
                'std_time': np.std(times)
            }
        
        return result


def memoize_features(cache_size: int = 128):
    """Decorator for memoizing feature extraction results."""
    def decorator(func):
        @lru_cache(maxsize=cache_size)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        
        wrapper.cache_info = wrapper.cache_info
        wrapper.cache_clear = wrapper.cache_clear
        return wrapper
    
    return decorator


class OptimizedFeaturePipeline:
    """Optimized feature extraction pipeline."""
    
    def __init__(self, config: Optional[OptimizationConfig] = None):
        """Initialize optimized feature pipeline.
        
        Args:
            config: Optimization configuration
        """
        self.config = config or OptimizationConfig()
        
        # Initialize components
        self.feature_extractor = FeatureExtractor()
        self.text_extractor = AdvancedTextFeatureExtractor()
        self.pgx_extractor = PharmacogenomicsFeatureExtractor()
        
        # Initialize optimization components
        if self.config.enable_caching:
            self.cache = FeatureCache(max_size=self.config.cache_size)
        else:
            self.cache = None
        
        self.memory_manager = MemoryManager(
            memory_limit_mb=self.config.memory_limit_mb
        )
        
        if self.config.enable_parallel_processing:
            self.parallel_processor = ParallelFeatureProcessor(
                max_workers=self.config.max_workers,
                chunk_size=self.config.chunk_size
            )
        else:
            self.parallel_processor = None
        
        if self.config.enable_profiling:
            self.profiler = PerformanceProfiler()
        else:
            self.profiler = None
        
        logger.info("OptimizedFeaturePipeline initialized")
    
    def extract_features_optimized(self, papers: List[PaperMetadata],
                                 quality_scores: List[QualityScoreComponents],
                                 feature_types: Optional[List[str]] = None) -> Tuple[FeatureSet, ProcessingMetrics]:
        """Extract features with optimization.
        
        Args:
            papers: List of papers
            quality_scores: List of quality scores
            feature_types: Types of features to extract
            
        Returns:
            Tuple of (FeatureSet, ProcessingMetrics)
        """
        if feature_types is None:
            feature_types = ['text', 'numerical', 'temporal', 'metadata', 'quality', 'pgx']
        
        start_time = time.time()
        start_memory = self.memory_manager.check_memory_usage()
        
        if self.profiler:
            self.profiler.start_timer('total_extraction')
        
        # Check cache first
        cached_features = None
        if self.cache:
            cached_features = self.cache.get(papers, quality_scores, feature_types)
            if cached_features:
                logger.debug(f"Cache hit for {len(papers)} papers")
                
                # Calculate metrics for cached result
                end_time = time.time()
                processing_time = end_time - start_time
                
                metrics = ProcessingMetrics(
                    total_papers=len(papers),
                    processing_time=processing_time,
                    papers_per_second=len(papers) / processing_time,
                    memory_usage_mb=self.memory_manager.check_memory_usage(),
                    cpu_usage_percent=psutil.cpu_percent(),
                    cache_hit_rate=self.cache.get_hit_rate(),
                    parallel_efficiency=1.0,
                    feature_extraction_time=0.0,
                    feature_validation_time=0.0
                )
                
                return cached_features, metrics
        
        # Extract features
        if self.profiler:
            self.profiler.start_timer('feature_extraction')
        
        # Use parallel processing for large datasets
        if (self.parallel_processor and 
            len(papers) >= self.config.chunk_size and 
            self.config.enable_parallel_processing):
            
            logger.debug(f"Using parallel processing for {len(papers)} papers")
            feature_set = self.parallel_processor.process_papers_parallel(
                papers, quality_scores, self.feature_extractor, feature_types
            )
        else:
            # Sequential processing
            feature_set = self._extract_features_sequential(
                papers, quality_scores, feature_types
            )
        
        extraction_time = 0.0
        if self.profiler:
            extraction_time = self.profiler.end_timer('feature_extraction')
        
        # Validate features
        if self.profiler:
            self.profiler.start_timer('feature_validation')
        
        validation_results = self.feature_extractor.validator.validate_features(feature_set)
        
        validation_time = 0.0
        if self.profiler:
            validation_time = self.profiler.end_timer('feature_validation')
        
        # Cache results
        if self.cache and validation_results['is_valid']:
            self.cache.put(papers, quality_scores, feature_types, feature_set)
        
        # Memory optimization
        self.memory_manager.optimize_memory()
        
        # Calculate metrics
        end_time = time.time()
        processing_time = end_time - start_time
        end_memory = self.memory_manager.check_memory_usage()
        
        metrics = ProcessingMetrics(
            total_papers=len(papers),
            processing_time=processing_time,
            papers_per_second=len(papers) / processing_time if processing_time > 0 else 0,
            memory_usage_mb=end_memory,
            cpu_usage_percent=psutil.cpu_percent(),
            cache_hit_rate=self.cache.get_hit_rate() if self.cache else 0.0,
            parallel_efficiency=self._calculate_parallel_efficiency(),
            feature_extraction_time=extraction_time,
            feature_validation_time=validation_time
        )
        
        if self.profiler:
            self.profiler.end_timer('total_extraction')
        
        logger.info(f"Feature extraction completed: {len(papers)} papers, "
                   f"{processing_time:.2f}s, {metrics.papers_per_second:.1f} papers/sec")
        
        return feature_set, metrics
    
    def _extract_features_sequential(self, papers: List[PaperMetadata],
                                   quality_scores: List[QualityScoreComponents],
                                   feature_types: List[str]) -> FeatureSet:
        """Extract features sequentially with batching."""
        if len(papers) <= self.config.batch_size:
            return self.feature_extractor.extract_features(papers, quality_scores, feature_types)
        
        # Process in batches
        feature_sets = []
        
        for i in range(0, len(papers), self.config.batch_size):
            end_idx = min(i + self.config.batch_size, len(papers))
            paper_batch = papers[i:end_idx]
            quality_batch = quality_scores[i:end_idx] if quality_scores else []
            
            batch_features = self.feature_extractor.extract_features(
                paper_batch, quality_batch, feature_types
            )
            feature_sets.append(batch_features)
            
            # Memory optimization after each batch
            if i % (self.config.batch_size * 5) == 0:  # Every 5 batches
                self.memory_manager.optimize_memory()
        
        # Combine batches
        return self.parallel_processor._combine_feature_sets(feature_sets)
    
    def _calculate_parallel_efficiency(self) -> float:
        """Calculate parallel processing efficiency."""
        if not self.parallel_processor:
            return 1.0
        
        # Simplified efficiency calculation
        # In practice, this would compare parallel vs sequential timing
        return min(1.0, self.config.max_workers / mp.cpu_count())
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        report = {
            'configuration': {
                'max_workers': self.config.max_workers,
                'batch_size': self.config.batch_size,
                'cache_enabled': self.config.enable_caching,
                'parallel_enabled': self.config.enable_parallel_processing,
                'memory_limit_mb': self.config.memory_limit_mb
            },
            'memory_stats': self.memory_manager.get_memory_stats(),
            'cache_stats': {
                'hit_rate': self.cache.get_hit_rate() if self.cache else 0.0,
                'cache_size': len(self.cache.cache) if self.cache else 0
            }
        }
        
        if self.profiler:
            report['profiling_metrics'] = self.profiler.get_metrics()
        
        return report
    
    def optimize_for_dataset_size(self, dataset_size: int) -> None:
        """Optimize configuration for specific dataset size.
        
        Args:
            dataset_size: Number of papers to process
        """
        if dataset_size < 100:
            # Small dataset - disable parallelization
            self.config.enable_parallel_processing = False
            self.config.batch_size = dataset_size
        elif dataset_size < 1000:
            # Medium dataset - moderate parallelization
            self.config.max_workers = min(4, mp.cpu_count())
            self.config.batch_size = 50
        else:
            # Large dataset - full parallelization
            self.config.max_workers = mp.cpu_count()
            self.config.batch_size = 100
            self.config.enable_caching = True
        
        logger.info(f"Optimized configuration for {dataset_size} papers: "
                   f"parallel={self.config.enable_parallel_processing}, "
                   f"workers={self.config.max_workers}, "
                   f"batch_size={self.config.batch_size}")
    
    def clear_cache(self) -> None:
        """Clear feature cache."""
        if self.cache:
            self.cache.clear()
            logger.info("Feature cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if not self.cache:
            return {'enabled': False}
        
        return {
            'enabled': True,
            'hit_rate': self.cache.get_hit_rate(),
            'cache_size': len(self.cache.cache),
            'max_size': self.cache.max_size,
            'hit_count': self.cache.hit_count,
            'miss_count': self.cache.miss_count
        }