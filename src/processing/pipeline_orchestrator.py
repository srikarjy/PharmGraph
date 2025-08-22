"""Complete pharmacogenomics research pipeline orchestrator."""

import asyncio
import time
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json

from ..config import get_logger, get_config
from ..data_ingestion.ncbi_search import NCBISearchClient
from ..data_ingestion.ncbi_fetch import NCBIFetchClient
from ..data_ingestion.pubmed_parser import PubMedXMLParser
from ..data_ingestion.pgx_queries import PharmacogenomicsQueryBuilder
from ..data_ingestion.data_models import PaperMetadata
from ..quality.quality_scorer import PharmacogenomicsQualityScorer, QualityScoreComponents
from ..storage.database import get_database_manager, db_session_scope
from ..storage.models import PaperModel, QualityMetrics


logger = get_logger(__name__)


class PipelineStage(Enum):
    """Pipeline execution stages."""
    INITIALIZING = "initializing"
    SEARCHING = "searching"
    FETCHING = "fetching"
    PARSING = "parsing"
    SCORING = "scoring"
    VALIDATING = "validating"
    STORING = "storing"
    REPORTING = "reporting"
    COMPLETED = "completed"
    FAILED = "failed"


class PipelineStatus(Enum):
    """Pipeline execution status."""
    NOT_STARTED = "not_started"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class PipelineMetrics:
    """Pipeline execution metrics."""
    papers_searched: int = 0
    papers_fetched: int = 0
    papers_parsed: int = 0
    papers_scored: int = 0
    papers_stored: int = 0
    
    search_time: float = 0.0
    fetch_time: float = 0.0
    parse_time: float = 0.0
    score_time: float = 0.0
    store_time: float = 0.0
    
    total_time: float = 0.0
    
    errors_search: int = 0
    errors_fetch: int = 0
    errors_parse: int = 0
    errors_score: int = 0
    errors_store: int = 0
    
    avg_quality_score: float = 0.0
    quality_distribution: Dict[str, int] = field(default_factory=dict)
    
    @property
    def success_rate(self) -> float:
        """Calculate overall success rate."""
        if self.papers_searched == 0:
            return 0.0
        return (self.papers_stored / self.papers_searched) * 100
    
    @property
    def papers_per_minute(self) -> float:
        """Calculate processing rate."""
        if self.total_time == 0:
            return 0.0
        return (self.papers_stored / self.total_time) * 60


@dataclass
class PipelineState:
    """Pipeline execution state for resumption."""
    pipeline_id: str
    status: PipelineStatus
    current_stage: PipelineStage
    search_queries: List[str]
    max_papers: int
    
    # Progress tracking
    collected_pmids: List[str] = field(default_factory=list)
    processed_pmids: List[str] = field(default_factory=list)
    failed_pmids: List[str] = field(default_factory=list)
    
    # Metrics
    metrics: PipelineMetrics = field(default_factory=PipelineMetrics)
    
    # Timestamps
    started_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    
    # Configuration snapshot
    config_snapshot: Dict[str, Any] = field(default_factory=dict)
    
    def update_progress(self):
        """Update progress timestamps."""
        self.updated_at = datetime.utcnow()


@dataclass
class PipelineReport:
    """Comprehensive pipeline execution report."""
    pipeline_id: str
    status: PipelineStatus
    
    # Summary metrics
    papers_collected: int
    papers_processed: int
    papers_stored: int
    success_rate: float
    
    # Performance metrics
    total_time_minutes: float
    papers_per_minute: float
    avg_quality_score: float
    
    # Quality distribution
    high_quality_papers: int  # Score >= 75
    medium_quality_papers: int  # Score 50-74
    low_quality_papers: int  # Score < 50
    
    # Error summary
    total_errors: int
    error_breakdown: Dict[str, int]
    
    # Top papers
    top_papers: List[Dict[str, Any]]
    
    # Timestamps
    started_at: datetime
    completed_at: Optional[datetime]
    
    # Configuration
    search_queries: List[str]
    max_papers: int


class PharmacogenomicsPipeline:
    """Complete end-to-end pharmacogenomics research pipeline."""
    
    def __init__(self, 
                 email: str = None,
                 api_key: str = None,
                 batch_size: int = 200,
                 max_concurrent: int = 5,
                 enable_resume: bool = True):
        """Initialize the pipeline.
        
        Args:
            email: NCBI email (required)
            api_key: NCBI API key (optional)
            batch_size: Batch size for fetching
            max_concurrent: Maximum concurrent operations
            enable_resume: Enable pipeline resumption
        """
        # Load configuration
        config = get_config()
        
        self.email = email or getattr(config.ncbi, 'email', None)
        self.api_key = api_key or getattr(config.ncbi, 'api_key', None)
        self.batch_size = batch_size
        self.max_concurrent = max_concurrent
        self.enable_resume = enable_resume
        
        if not self.email:
            raise ValueError("Email is required for NCBI API access")
        
        # Initialize components
        self.search_client = NCBISearchClient(self.email, self.api_key)
        self.fetch_client = NCBIFetchClient(self.email, self.api_key, default_batch_size=batch_size)
        self.parser = PubMedXMLParser()
        self.query_builder = PharmacogenomicsQueryBuilder()
        self.quality_scorer = PharmacogenomicsQualityScorer()
        
        # Pipeline state
        self.current_pipeline: Optional[PipelineState] = None
        self.pipeline_history: Dict[str, PipelineState] = {}
        
        logger.info(f"Initialized PharmacogenomicsPipeline for {self.email}")
    
    async def run_full_pipeline(self, 
                               search_queries: List[str],
                               max_papers: int = 1000,
                               resume_pipeline_id: str = None) -> PipelineReport:
        """Execute the complete pipeline.
        
        Args:
            search_queries: List of search queries to execute
            max_papers: Maximum papers to process
            resume_pipeline_id: Pipeline ID to resume (optional)
            
        Returns:
            PipelineReport: Comprehensive execution report
        """
        # Initialize or resume pipeline
        if resume_pipeline_id and resume_pipeline_id in self.pipeline_history:
            pipeline_state = self.pipeline_history[resume_pipeline_id]
            logger.info(f"Resuming pipeline {resume_pipeline_id}")
        else:
            pipeline_id = str(uuid.uuid4())
            pipeline_state = PipelineState(
                pipeline_id=pipeline_id,
                status=PipelineStatus.NOT_STARTED,
                current_stage=PipelineStage.INITIALIZING,
                search_queries=search_queries,
                max_papers=max_papers,
                config_snapshot=self._capture_config_snapshot()
            )
            self.pipeline_history[pipeline_id] = pipeline_state
        
        self.current_pipeline = pipeline_state
        
        try:
            # Execute pipeline stages
            async with self.search_client, self.fetch_client:
                await self._execute_pipeline_stages(pipeline_state)
            
            # Generate final report
            report = self._generate_pipeline_report(pipeline_state)
            
            pipeline_state.status = PipelineStatus.COMPLETED
            pipeline_state.completed_at = datetime.utcnow()
            pipeline_state.update_progress()
            
            logger.info(f"Pipeline {pipeline_state.pipeline_id} completed successfully")
            return report
            
        except Exception as e:
            pipeline_state.status = PipelineStatus.FAILED
            pipeline_state.current_stage = PipelineStage.FAILED
            pipeline_state.update_progress()
            
            logger.error(f"Pipeline {pipeline_state.pipeline_id} failed: {str(e)}")
            
            # Generate failure report
            report = self._generate_pipeline_report(pipeline_state)
            return report
    
    async def _execute_pipeline_stages(self, pipeline_state: PipelineState):
        """Execute all pipeline stages."""
        start_time = time.time()
        
        try:
            # Stage 1: Search Strategy
            if pipeline_state.current_stage in [PipelineStage.INITIALIZING, PipelineStage.SEARCHING]:
                await self._stage_search_strategy(pipeline_state)
            
            # Stage 2: Data Fetching
            if pipeline_state.current_stage == PipelineStage.FETCHING:
                await self._stage_data_fetching(pipeline_state)
            
            # Stage 3: Quality Scoring
            if pipeline_state.current_stage == PipelineStage.SCORING:
                await self._stage_quality_scoring(pipeline_state)
            
            # Stage 4: Validation
            if pipeline_state.current_stage == PipelineStage.VALIDATING:
                await self._stage_validation(pipeline_state)
            
            # Stage 5: Storage
            if pipeline_state.current_stage == PipelineStage.STORING:
                await self._stage_storage(pipeline_state)
            
            # Stage 6: Reporting
            pipeline_state.current_stage = PipelineStage.REPORTING
            pipeline_state.update_progress()
            
        finally:
            pipeline_state.metrics.total_time = time.time() - start_time
    
    async def _stage_search_strategy(self, pipeline_state: PipelineState):
        """Execute search strategy stage."""
        logger.info("Starting search strategy stage")
        pipeline_state.status = PipelineStatus.RUNNING
        pipeline_state.current_stage = PipelineStage.SEARCHING
        pipeline_state.update_progress()
        
        start_time = time.time()
        all_pmids = []
        
        try:
            for query in pipeline_state.search_queries:
                logger.info(f"Executing search query: {query}")
                
                # Get search count first
                count = await self.search_client.get_search_count(query)
                logger.info(f"Query '{query}' found {count} papers")
                
                if count > 0:
                    # Calculate how many papers to get from this query
                    remaining_papers = pipeline_state.max_papers - len(all_pmids)
                    papers_to_get = min(count, remaining_papers)
                    
                    if papers_to_get > 0:
                        # Use batched search for large result sets
                        pmids = await self.search_client.search_batched(
                            query=query,
                            max_total=papers_to_get
                        )
                        
                        all_pmids.extend(pmids)
                        logger.info(f"Collected {len(pmids)} PMIDs from query")
                        
                        # Stop if we've reached the limit
                        if len(all_pmids) >= pipeline_state.max_papers:
                            break
            
            # Remove duplicates while preserving order
            unique_pmids = []
            seen = set()
            for pmid in all_pmids:
                if pmid not in seen:
                    unique_pmids.append(pmid)
                    seen.add(pmid)
            
            pipeline_state.collected_pmids = unique_pmids[:pipeline_state.max_papers]
            pipeline_state.metrics.papers_searched = len(pipeline_state.collected_pmids)
            
            logger.info(f"Search completed: {len(pipeline_state.collected_pmids)} unique PMIDs collected")
            
            # Move to next stage
            pipeline_state.current_stage = PipelineStage.FETCHING
            
        except Exception as e:
            pipeline_state.metrics.errors_search += 1
            logger.error(f"Search stage failed: {str(e)}")
            raise
        
        finally:
            pipeline_state.metrics.search_time = time.time() - start_time
            pipeline_state.update_progress()
    
    async def _stage_data_fetching(self, pipeline_state: PipelineState):
        """Execute data fetching stage."""
        logger.info("Starting data fetching stage")
        pipeline_state.current_stage = PipelineStage.FETCHING
        pipeline_state.update_progress()
        
        start_time = time.time()
        
        try:
            if not pipeline_state.collected_pmids:
                logger.warning("No PMIDs to fetch")
                pipeline_state.current_stage = PipelineStage.COMPLETED
                return
            
            logger.info(f"Fetching details for {len(pipeline_state.collected_pmids)} papers")
            
            # Fetch paper details in batches
            xml_responses, fetch_stats = await self.fetch_client.fetch_paper_details(
                pmids=pipeline_state.collected_pmids,
                batch_size=self.batch_size,
                max_concurrent=self.max_concurrent
            )
            
            pipeline_state.metrics.papers_fetched = fetch_stats.successful_fetches
            pipeline_state.metrics.errors_fetch = fetch_stats.failed_fetches
            
            logger.info(f"Fetching completed: {fetch_stats.successful_fetches} papers fetched")
            
            # Parse XML responses
            logger.info("Parsing XML responses")
            parse_start = time.time()
            
            all_papers = []
            for xml_response in xml_responses:
                try:
                    papers = self.parser.parse_pubmed_xml(xml_response)
                    all_papers.extend(papers)
                except Exception as e:
                    pipeline_state.metrics.errors_parse += 1
                    logger.error(f"XML parsing error: {str(e)}")
            
            pipeline_state.metrics.papers_parsed = len(all_papers)
            pipeline_state.metrics.parse_time = time.time() - parse_start
            
            # Store parsed papers for next stage
            pipeline_state._parsed_papers = all_papers
            
            logger.info(f"Parsing completed: {len(all_papers)} papers parsed")
            
            # Move to next stage
            pipeline_state.current_stage = PipelineStage.SCORING
            
        except Exception as e:
            pipeline_state.metrics.errors_fetch += 1
            logger.error(f"Fetching stage failed: {str(e)}")
            raise
        
        finally:
            pipeline_state.metrics.fetch_time = time.time() - start_time
            pipeline_state.update_progress()
    
    async def _stage_quality_scoring(self, pipeline_state: PipelineState):
        """Execute quality scoring stage."""
        logger.info("Starting quality scoring stage")
        pipeline_state.current_stage = PipelineStage.SCORING
        pipeline_state.update_progress()
        
        start_time = time.time()
        
        try:
            papers = getattr(pipeline_state, '_parsed_papers', [])
            if not papers:
                logger.warning("No papers to score")
                pipeline_state.current_stage = PipelineStage.COMPLETED
                return
            
            logger.info(f"Scoring {len(papers)} papers")
            
            # Score papers in batch
            quality_scores = self.quality_scorer.score_papers_batch(papers)
            
            pipeline_state.metrics.papers_scored = len(quality_scores)
            
            # Calculate quality distribution
            high_quality = sum(1 for score in quality_scores if score.total_score >= 75)
            medium_quality = sum(1 for score in quality_scores if 50 <= score.total_score < 75)
            low_quality = sum(1 for score in quality_scores if score.total_score < 50)
            
            pipeline_state.metrics.quality_distribution = {
                'high': high_quality,
                'medium': medium_quality,
                'low': low_quality
            }
            
            # Calculate average quality score
            if quality_scores:
                pipeline_state.metrics.avg_quality_score = sum(
                    score.total_score for score in quality_scores
                ) / len(quality_scores)
            
            # Store scored papers for next stage
            pipeline_state._scored_papers = list(zip(papers, quality_scores))
            
            logger.info(f"Scoring completed: avg score {pipeline_state.metrics.avg_quality_score:.1f}")
            
            # Move to next stage
            pipeline_state.current_stage = PipelineStage.VALIDATING
            
        except Exception as e:
            pipeline_state.metrics.errors_score += 1
            logger.error(f"Scoring stage failed: {str(e)}")
            raise
        
        finally:
            pipeline_state.metrics.score_time = time.time() - start_time
            pipeline_state.update_progress()
    
    async def _stage_validation(self, pipeline_state: PipelineState):
        """Execute validation stage."""
        logger.info("Starting validation stage")
        pipeline_state.current_stage = PipelineStage.VALIDATING
        pipeline_state.update_progress()
        
        try:
            scored_papers = getattr(pipeline_state, '_scored_papers', [])
            if not scored_papers:
                logger.warning("No papers to validate")
                pipeline_state.current_stage = PipelineStage.COMPLETED
                return
            
            # Validate papers
            valid_papers = []
            for paper, scores in scored_papers:
                if self._validate_paper_data(paper, scores):
                    valid_papers.append((paper, scores))
                else:
                    logger.warning(f"Paper {paper.pmid} failed validation")
            
            pipeline_state._validated_papers = valid_papers
            
            logger.info(f"Validation completed: {len(valid_papers)} valid papers")
            
            # Move to next stage
            pipeline_state.current_stage = PipelineStage.STORING
            
        except Exception as e:
            logger.error(f"Validation stage failed: {str(e)}")
            raise
        
        finally:
            pipeline_state.update_progress()
    
    async def _stage_storage(self, pipeline_state: PipelineState):
        """Execute storage stage."""
        logger.info("Starting storage stage")
        pipeline_state.current_stage = PipelineStage.STORING
        pipeline_state.update_progress()
        
        start_time = time.time()
        
        try:
            validated_papers = getattr(pipeline_state, '_validated_papers', [])
            if not validated_papers:
                logger.warning("No papers to store")
                pipeline_state.current_stage = PipelineStage.COMPLETED
                return
            
            logger.info(f"Storing {len(validated_papers)} papers")
            
            # Store papers in database
            stored_count = await self._store_papers_batch(validated_papers)
            
            pipeline_state.metrics.papers_stored = stored_count
            
            # Track processed PMIDs
            for paper, _ in validated_papers:
                pipeline_state.processed_pmids.append(paper.pmid)
            
            logger.info(f"Storage completed: {stored_count} papers stored")
            
            # Move to final stage
            pipeline_state.current_stage = PipelineStage.COMPLETED
            
        except Exception as e:
            pipeline_state.metrics.errors_store += 1
            logger.error(f"Storage stage failed: {str(e)}")
            raise
        
        finally:
            pipeline_state.metrics.store_time = time.time() - start_time
            pipeline_state.update_progress()
    
    def _validate_paper_data(self, paper: PaperMetadata, scores: QualityScoreComponents) -> bool:
        """Validate paper data integrity."""
        # Required fields
        if not paper.pmid or not paper.title:
            return False
        
        # Quality score validation
        if not (0 <= scores.total_score <= 100):
            return False
        
        # Confidence validation
        if not (0 <= scores.confidence_level <= 1):
            return False
        
        return True
    
    async def _store_papers_batch(self, papers_and_scores: List[Tuple[PaperMetadata, QualityScoreComponents]]) -> int:
        """Store papers and quality scores in database."""
        stored_count = 0
        
        try:
            # Initialize database
            db_manager = get_database_manager()
            db_manager.initialize()
            
            with db_session_scope() as session:
                for paper, scores in papers_and_scores:
                    try:
                        # Check if paper already exists
                        existing = session.query(PaperModel).filter_by(pmid=paper.pmid).first()
                        if existing:
                            logger.debug(f"Paper {paper.pmid} already exists, skipping")
                            continue
                        
                        # Create paper model
                        db_paper = self._convert_to_paper_model(paper)
                        session.add(db_paper)
                        
                        # Create quality metrics
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
                        session.add(quality_metrics)
                        
                        stored_count += 1
                        
                    except Exception as e:
                        logger.error(f"Error storing paper {paper.pmid}: {str(e)}")
                        continue
            
            logger.info(f"Successfully stored {stored_count} papers")
            return stored_count
            
        except Exception as e:
            logger.error(f"Database storage failed: {str(e)}")
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
    
    def _capture_config_snapshot(self) -> Dict[str, Any]:
        """Capture configuration snapshot for reproducibility."""
        return {
            'email': self.email,
            'has_api_key': bool(self.api_key),
            'batch_size': self.batch_size,
            'max_concurrent': self.max_concurrent,
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def _generate_pipeline_report(self, pipeline_state: PipelineState) -> PipelineReport:
        """Generate comprehensive pipeline report."""
        metrics = pipeline_state.metrics
        
        # Calculate error breakdown
        error_breakdown = {
            'search': metrics.errors_search,
            'fetch': metrics.errors_fetch,
            'parse': metrics.errors_parse,
            'score': metrics.errors_score,
            'store': metrics.errors_store
        }
        
        total_errors = sum(error_breakdown.values())
        
        # Get top papers
        top_papers = []
        if hasattr(pipeline_state, '_scored_papers'):
            scored_papers = pipeline_state._scored_papers
            # Sort by quality score and take top 10
            sorted_papers = sorted(scored_papers, key=lambda x: x[1].total_score, reverse=True)
            
            for paper, scores in sorted_papers[:10]:
                top_papers.append({
                    'pmid': paper.pmid,
                    'title': paper.title[:100] + '...' if len(paper.title) > 100 else paper.title,
                    'journal': paper.journal.title if paper.journal else 'Unknown',
                    'quality_score': round(scores.total_score, 1),
                    'confidence': round(scores.confidence_level, 2)
                })
        
        return PipelineReport(
            pipeline_id=pipeline_state.pipeline_id,
            status=pipeline_state.status,
            papers_collected=metrics.papers_searched,
            papers_processed=metrics.papers_scored,
            papers_stored=metrics.papers_stored,
            success_rate=metrics.success_rate,
            total_time_minutes=metrics.total_time / 60,
            papers_per_minute=metrics.papers_per_minute,
            avg_quality_score=metrics.avg_quality_score,
            high_quality_papers=metrics.quality_distribution.get('high', 0),
            medium_quality_papers=metrics.quality_distribution.get('medium', 0),
            low_quality_papers=metrics.quality_distribution.get('low', 0),
            total_errors=total_errors,
            error_breakdown=error_breakdown,
            top_papers=top_papers,
            started_at=pipeline_state.started_at,
            completed_at=pipeline_state.completed_at,
            search_queries=pipeline_state.search_queries,
            max_papers=pipeline_state.max_papers
        )
    
    def get_pipeline_status(self, pipeline_id: str = None) -> Optional[PipelineState]:
        """Get current pipeline status.
        
        Args:
            pipeline_id: Specific pipeline ID, or current if None
            
        Returns:
            Optional[PipelineState]: Pipeline state if found
        """
        if pipeline_id:
            return self.pipeline_history.get(pipeline_id)
        return self.current_pipeline
    
    async def resume_failed_pipeline(self, pipeline_id: str) -> PipelineReport:
        """Resume a failed pipeline from its last successful stage.
        
        Args:
            pipeline_id: Pipeline ID to resume
            
        Returns:
            PipelineReport: Execution report
        """
        if pipeline_id not in self.pipeline_history:
            raise ValueError(f"Pipeline {pipeline_id} not found")
        
        pipeline_state = self.pipeline_history[pipeline_id]
        
        if pipeline_state.status != PipelineStatus.FAILED:
            raise ValueError(f"Pipeline {pipeline_id} is not in failed state")
        
        logger.info(f"Resuming failed pipeline {pipeline_id} from stage {pipeline_state.current_stage}")
        
        # Reset status and resume
        pipeline_state.status = PipelineStatus.RUNNING
        
        return await self.run_full_pipeline(
            search_queries=pipeline_state.search_queries,
            max_papers=pipeline_state.max_papers,
            resume_pipeline_id=pipeline_id
        )
    
    async def process_paper_batch(self, pmids: List[str]) -> Dict[str, Any]:
        """Process a batch of papers through the complete pipeline.
        
        Args:
            pmids: List of PMIDs to process
            
        Returns:
            Dict[str, Any]: Processing results and metrics
        """
        if not pmids:
            return {"processed": 0, "errors": 0, "results": []}
        
        logger.info(f"Processing batch of {len(pmids)} papers")
        batch_start_time = time.time()
        
        try:
            # Initialize clients if not already done
            if not hasattr(self, '_clients_initialized'):
                await self.search_client.__aenter__()
                await self.fetch_client.__aenter__()
                self._clients_initialized = True
            
            # Fetch paper details with progress tracking
            logger.debug(f"Fetching details for {len(pmids)} papers")
            xml_responses, fetch_stats = await self.fetch_client.fetch_paper_details(
                pmids=pmids,
                batch_size=self.batch_size,
                max_concurrent=self.max_concurrent
            )
            
            # Parse XML responses with error tracking
            all_papers = []
            parse_errors = 0
            
            for xml_response in xml_responses:
                try:
                    papers = self.parser.parse_pubmed_xml(xml_response)
                    all_papers.extend(papers)
                except Exception as e:
                    parse_errors += 1
                    logger.error(f"XML parsing error: {str(e)}")
            
            # Score papers with timing
            scoring_start = time.time()
            quality_scores = self.quality_scorer.score_papers_batch(all_papers)
            scoring_time = time.time() - scoring_start
            
            # Validate papers before storage
            valid_papers = []
            validation_errors = 0
            
            for paper, score in zip(all_papers, quality_scores):
                if self._validate_paper_data(paper, score):
                    valid_papers.append((paper, score))
                else:
                    validation_errors += 1
                    logger.warning(f"Paper {paper.pmid} failed validation")
            
            # Store papers with transaction management
            storage_start = time.time()
            stored_count = 0
            if valid_papers:
                stored_count = await self._store_papers_batch(valid_papers)
            storage_time = time.time() - storage_start
            
            batch_time = time.time() - batch_start_time
            
            # Calculate performance metrics
            papers_per_second = len(all_papers) / batch_time if batch_time > 0 else 0
            avg_scoring_time = (scoring_time / len(all_papers) * 1000) if all_papers else 0  # ms per paper
            
            return {
                "processed": len(all_papers),
                "stored": stored_count,
                "validated": len(valid_papers),
                "fetch_errors": fetch_stats.failed_fetches,
                "parse_errors": parse_errors,
                "validation_errors": validation_errors,
                "batch_time": batch_time,
                "papers_per_second": papers_per_second,
                "avg_scoring_time_ms": avg_scoring_time,
                "storage_time": storage_time,
                "results": [
                    {
                        "pmid": paper.pmid,
                        "title": paper.title[:100] + "..." if len(paper.title) > 100 else paper.title,
                        "quality_score": score.total_score,
                        "confidence": score.confidence_level
                    }
                    for paper, score in valid_papers[:10]  # Top 10 for summary
                ]
            }
            
        except Exception as e:
            batch_time = time.time() - batch_start_time
            logger.error(f"Batch processing failed after {batch_time:.2f}s: {str(e)}")
            return {
                "processed": 0, 
                "stored": 0, 
                "errors": len(pmids), 
                "error": str(e),
                "batch_time": batch_time
            }

    def generate_pipeline_report(self, pipeline_id: str = None) -> Optional[PipelineReport]:
        """Generate report for specified or current pipeline.
        
        Args:
            pipeline_id: Pipeline ID, or current if None
            
        Returns:
            Optional[PipelineReport]: Pipeline report if found
        """
        pipeline_state = self.get_pipeline_status(pipeline_id)
        if pipeline_state:
            return self._generate_pipeline_report(pipeline_state)
        return None
    
    def list_pipelines(self) -> List[Dict[str, Any]]:
        """List all pipeline executions.
        
        Returns:
            List[Dict[str, Any]]: List of pipeline summaries
        """
        summaries = []
        for pipeline_id, state in self.pipeline_history.items():
            summaries.append({
                'pipeline_id': pipeline_id,
                'status': state.status.value,
                'stage': state.current_stage.value,
                'papers_collected': state.metrics.papers_searched,
                'papers_stored': state.metrics.papers_stored,
                'success_rate': state.metrics.success_rate,
                'started_at': state.started_at.isoformat(),
                'completed_at': state.completed_at.isoformat() if state.completed_at else None,
                'queries': state.search_queries
            })
        
        return summaries