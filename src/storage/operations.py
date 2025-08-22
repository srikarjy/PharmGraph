"""Database operations for papers, quality metrics, and collection runs."""

from datetime import datetime, date
from typing import List, Optional, Dict, Any
from sqlalchemy import and_, or_, desc, asc, func
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from ..config import get_logger
from .database import db_session_scope, get_database_manager
from .models import PaperModel, QualityMetrics, CollectionRun, CollectionStatus

logger = get_logger(__name__)


class PaperOperations:
    """Operations for managing research papers."""
    
    @staticmethod
    def save_paper(paper_data: Dict[str, Any]) -> str:
        """Save a paper to the database.
        
        Args:
            paper_data: Dictionary containing paper information
            
        Returns:
            str: PMID of the saved paper
            
        Raises:
            ValueError: If required fields are missing
            IntegrityError: If paper already exists
        """
        required_fields = ['pmid', 'title']
        for field in required_fields:
            if field not in paper_data:
                raise ValueError(f"Required field '{field}' is missing")
        
        try:
            with db_session_scope() as session:
                paper = PaperModel(**paper_data)
                session.add(paper)
                session.flush()  # Get any database-generated values
                logger.info(f"Saved paper {paper.pmid}: {paper.title[:50]}...")
                return paper.pmid
        except IntegrityError as e:
            logger.error(f"Paper {paper_data.get('pmid')} already exists: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Failed to save paper {paper_data.get('pmid')}: {str(e)}")
            raise
    
    @staticmethod
    def get_paper_by_pmid(pmid: str) -> Optional[PaperModel]:
        """Get a paper by its PMID.
        
        Args:
            pmid: PubMed ID
            
        Returns:
            PaperModel: Paper instance or None if not found
        """
        try:
            with db_session_scope() as session:
                paper = session.query(PaperModel).filter(PaperModel.pmid == pmid).first()
                if paper:
                    # Detach from session to avoid lazy loading issues
                    session.expunge(paper)
                return paper
        except Exception as e:
            logger.error(f"Failed to get paper {pmid}: {str(e)}")
            return None
    
    @staticmethod
    def search_papers(filters: Dict[str, Any], limit: int = 100, offset: int = 0) -> List[PaperModel]:
        """Search papers with various filters.
        
        Args:
            filters: Dictionary of search filters
            limit: Maximum number of results
            offset: Number of results to skip
            
        Returns:
            List[PaperModel]: List of matching papers
        """
        try:
            with db_session_scope() as session:
                query = session.query(PaperModel)
                
                # Apply filters
                if 'title_contains' in filters:
                    query = query.filter(PaperModel.title.contains(filters['title_contains']))
                
                if 'journal' in filters:
                    query = query.filter(PaperModel.journal == filters['journal'])
                
                if 'author_contains' in filters:
                    query = query.filter(PaperModel.authors_json.contains(filters['author_contains']))
                
                if 'date_from' in filters:
                    query = query.filter(PaperModel.publication_date >= filters['date_from'])
                
                if 'date_to' in filters:
                    query = query.filter(PaperModel.publication_date <= filters['date_to'])
                
                if 'mesh_term' in filters:
                    query = query.filter(PaperModel.mesh_terms_json.contains(filters['mesh_term']))
                
                # Apply ordering
                order_by = filters.get('order_by', 'publication_date')
                order_dir = filters.get('order_dir', 'desc')
                
                if hasattr(PaperModel, order_by):
                    column = getattr(PaperModel, order_by)
                    if order_dir.lower() == 'desc':
                        query = query.order_by(desc(column))
                    else:
                        query = query.order_by(asc(column))
                
                # Apply pagination
                papers = query.offset(offset).limit(limit).all()
                
                # Detach from session
                for paper in papers:
                    session.expunge(paper)
                
                logger.info(f"Found {len(papers)} papers matching filters")
                return papers
                
        except Exception as e:
            logger.error(f"Failed to search papers: {str(e)}")
            return []
    
    @staticmethod
    def bulk_insert_papers(papers: List[Dict[str, Any]]) -> int:
        """Bulk insert papers for better performance.
        
        Args:
            papers: List of paper dictionaries
            
        Returns:
            int: Number of papers inserted
        """
        if not papers:
            return 0
        
        try:
            db_manager = get_database_manager()
            count = db_manager.execute_bulk_insert(PaperModel, papers)
            logger.info(f"Bulk inserted {count} papers")
            return count
        except Exception as e:
            logger.error(f"Failed to bulk insert papers: {str(e)}")
            raise
    
    @staticmethod
    def update_paper(pmid: str, updates: Dict[str, Any]) -> bool:
        """Update a paper's information.
        
        Args:
            pmid: PubMed ID
            updates: Dictionary of fields to update
            
        Returns:
            bool: True if update successful, False otherwise
        """
        try:
            with db_session_scope() as session:
                paper = session.query(PaperModel).filter(PaperModel.pmid == pmid).first()
                if not paper:
                    logger.warning(f"Paper {pmid} not found for update")
                    return False
                
                # Update fields
                for field, value in updates.items():
                    if hasattr(paper, field):
                        setattr(paper, field, value)
                
                # Update timestamp
                paper.updated_at = datetime.utcnow()
                
                logger.info(f"Updated paper {pmid}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to update paper {pmid}: {str(e)}")
            return False
    
    @staticmethod
    def delete_paper(pmid: str) -> bool:
        """Delete a paper from the database.
        
        Args:
            pmid: PubMed ID
            
        Returns:
            bool: True if deletion successful, False otherwise
        """
        try:
            with db_session_scope() as session:
                paper = session.query(PaperModel).filter(PaperModel.pmid == pmid).first()
                if not paper:
                    logger.warning(f"Paper {pmid} not found for deletion")
                    return False
                
                session.delete(paper)
                logger.info(f"Deleted paper {pmid}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to delete paper {pmid}: {str(e)}")
            return False
    
    @staticmethod
    def get_paper_count() -> int:
        """Get total number of papers in the database.
        
        Returns:
            int: Total paper count
        """
        try:
            with db_session_scope() as session:
                count = session.query(PaperModel).count()
                return count
        except Exception as e:
            logger.error(f"Failed to get paper count: {str(e)}")
            return 0


class QualityOperations:
    """Operations for managing quality metrics."""
    
    @staticmethod
    def save_quality_metrics(pmid: str, metrics: Dict[str, Any]) -> int:
        """Save quality metrics for a paper.
        
        Args:
            pmid: PubMed ID
            metrics: Dictionary containing quality metrics
            
        Returns:
            int: ID of the saved quality metrics record
        """
        try:
            with db_session_scope() as session:
                # Check if paper exists
                paper = session.query(PaperModel).filter(PaperModel.pmid == pmid).first()
                if not paper:
                    raise ValueError(f"Paper {pmid} not found")
                
                # Create quality metrics
                quality = QualityMetrics(pmid=pmid, **metrics)
                
                # Calculate total score if not provided
                if 'total_score' not in metrics:
                    quality.calculate_total_score()
                
                session.add(quality)
                session.flush()
                
                logger.info(f"Saved quality metrics for paper {pmid}: {quality.total_score}")
                return quality.id
                
        except Exception as e:
            logger.error(f"Failed to save quality metrics for {pmid}: {str(e)}")
            raise
    
    @staticmethod
    def get_quality_by_pmid(pmid: str) -> Optional[QualityMetrics]:
        """Get quality metrics for a paper.
        
        Args:
            pmid: PubMed ID
            
        Returns:
            QualityMetrics: Quality metrics or None if not found
        """
        try:
            with db_session_scope() as session:
                quality = session.query(QualityMetrics).filter(QualityMetrics.pmid == pmid).first()
                if quality:
                    session.expunge(quality)
                return quality
        except Exception as e:
            logger.error(f"Failed to get quality metrics for {pmid}: {str(e)}")
            return None
    
    @staticmethod
    def get_papers_by_quality_range(min_score: float, max_score: float, limit: int = 100) -> List[Dict[str, Any]]:
        """Get papers within a quality score range.
        
        Args:
            min_score: Minimum quality score
            max_score: Maximum quality score
            limit: Maximum number of results
            
        Returns:
            List[Dict]: List of papers with their quality scores
        """
        try:
            with db_session_scope() as session:
                results = session.query(PaperModel, QualityMetrics).join(
                    QualityMetrics, PaperModel.pmid == QualityMetrics.pmid
                ).filter(
                    and_(
                        QualityMetrics.total_score >= min_score,
                        QualityMetrics.total_score <= max_score
                    )
                ).order_by(desc(QualityMetrics.total_score)).limit(limit).all()
                
                papers = []
                for paper, quality in results:
                    session.expunge(paper)
                    session.expunge(quality)
                    papers.append({
                        'paper': paper,
                        'quality': quality
                    })
                
                logger.info(f"Found {len(papers)} papers with quality score {min_score}-{max_score}")
                return papers
                
        except Exception as e:
            logger.error(f"Failed to get papers by quality range: {str(e)}")
            return []
    
    @staticmethod
    def update_quality_metrics(pmid: str, metrics: Dict[str, Any]) -> bool:
        """Update quality metrics for a paper.
        
        Args:
            pmid: PubMed ID
            metrics: Dictionary of metrics to update
            
        Returns:
            bool: True if update successful, False otherwise
        """
        try:
            with db_session_scope() as session:
                quality = session.query(QualityMetrics).filter(QualityMetrics.pmid == pmid).first()
                if not quality:
                    logger.warning(f"Quality metrics for paper {pmid} not found")
                    return False
                
                # Update fields
                for field, value in metrics.items():
                    if hasattr(quality, field):
                        setattr(quality, field, value)
                
                # Recalculate total score if component scores were updated
                if any(field.endswith('_score') for field in metrics.keys()):
                    quality.calculate_total_score()
                
                logger.info(f"Updated quality metrics for paper {pmid}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to update quality metrics for {pmid}: {str(e)}")
            return False
    
    @staticmethod
    def get_quality_distribution() -> Dict[str, Any]:
        """Get distribution statistics for quality scores.
        
        Returns:
            Dict: Quality score distribution statistics
        """
        try:
            with db_session_scope() as session:
                stats = session.query(
                    func.count(QualityMetrics.id).label('total_papers'),
                    func.avg(QualityMetrics.total_score).label('avg_score'),
                    func.min(QualityMetrics.total_score).label('min_score'),
                    func.max(QualityMetrics.total_score).label('max_score'),
                    func.avg(QualityMetrics.journal_score).label('avg_journal_score'),
                    func.avg(QualityMetrics.clinical_relevance_score).label('avg_clinical_score'),
                    func.avg(QualityMetrics.ml_methodology_score).label('avg_ml_score'),
                    func.avg(QualityMetrics.recency_score).label('avg_recency_score')
                ).first()
                
                # Count papers in different quality ranges
                high_quality = session.query(QualityMetrics).filter(QualityMetrics.total_score >= 80).count()
                medium_quality = session.query(QualityMetrics).filter(
                    and_(QualityMetrics.total_score >= 60, QualityMetrics.total_score < 80)
                ).count()
                low_quality = session.query(QualityMetrics).filter(QualityMetrics.total_score < 60).count()
                
                distribution = {
                    'total_papers': stats.total_papers or 0,
                    'avg_score': float(stats.avg_score) if stats.avg_score else 0.0,
                    'min_score': float(stats.min_score) if stats.min_score else 0.0,
                    'max_score': float(stats.max_score) if stats.max_score else 0.0,
                    'avg_component_scores': {
                        'journal': float(stats.avg_journal_score) if stats.avg_journal_score else 0.0,
                        'clinical': float(stats.avg_clinical_score) if stats.avg_clinical_score else 0.0,
                        'ml_methodology': float(stats.avg_ml_score) if stats.avg_ml_score else 0.0,
                        'recency': float(stats.avg_recency_score) if stats.avg_recency_score else 0.0
                    },
                    'quality_ranges': {
                        'high_quality': high_quality,  # 80-100
                        'medium_quality': medium_quality,  # 60-79
                        'low_quality': low_quality  # 0-59
                    }
                }
                
                return distribution
                
        except Exception as e:
            logger.error(f"Failed to get quality distribution: {str(e)}")
            return {}


class CollectionOperations:
    """Operations for managing collection runs."""
    
    @staticmethod
    def start_collection_run(query: str, config_snapshot: Dict[str, Any] = None) -> int:
        """Start a new collection run.
        
        Args:
            query: Search query used for collection
            config_snapshot: Configuration snapshot for this run
            
        Returns:
            int: Collection run ID
        """
        try:
            with db_session_scope() as session:
                run = CollectionRun(
                    query_used=query,
                    config_snapshot=config_snapshot,
                    status=CollectionStatus.RUNNING
                )
                session.add(run)
                session.flush()
                
                logger.info(f"Started collection run {run.id} with query: {query}")
                return run.id
                
        except Exception as e:
            logger.error(f"Failed to start collection run: {str(e)}")
            raise
    
    @staticmethod
    def update_collection_progress(run_id: int, progress: Dict[str, Any]) -> bool:
        """Update collection run progress.
        
        Args:
            run_id: Collection run ID
            progress: Progress information
            
        Returns:
            bool: True if update successful, False otherwise
        """
        try:
            with db_session_scope() as session:
                run = session.query(CollectionRun).filter(CollectionRun.id == run_id).first()
                if not run:
                    logger.warning(f"Collection run {run_id} not found")
                    return False
                
                # Update progress fields
                if 'papers_collected' in progress:
                    run.papers_collected = progress['papers_collected']
                if 'papers_processed' in progress:
                    run.papers_processed = progress['papers_processed']
                if 'avg_quality_score' in progress:
                    run.avg_quality_score = progress['avg_quality_score']
                if 'error_message' in progress:
                    run.error_message = progress['error_message']
                if 'status' in progress:
                    run.status = CollectionStatus(progress['status'])
                
                logger.debug(f"Updated collection run {run_id} progress")
                return True
                
        except Exception as e:
            logger.error(f"Failed to update collection run {run_id}: {str(e)}")
            return False
    
    @staticmethod
    def complete_collection_run(run_id: int, summary: Dict[str, Any]) -> bool:
        """Complete a collection run.
        
        Args:
            run_id: Collection run ID
            summary: Final summary information
            
        Returns:
            bool: True if completion successful, False otherwise
        """
        try:
            with db_session_scope() as session:
                run = session.query(CollectionRun).filter(CollectionRun.id == run_id).first()
                if not run:
                    logger.warning(f"Collection run {run_id} not found")
                    return False
                
                # Update final status
                run.status = CollectionStatus.COMPLETED
                run.collection_end = datetime.utcnow()
                
                # Update summary fields
                if 'papers_collected' in summary:
                    run.papers_collected = summary['papers_collected']
                if 'papers_processed' in summary:
                    run.papers_processed = summary['papers_processed']
                if 'avg_quality_score' in summary:
                    run.avg_quality_score = summary['avg_quality_score']
                
                logger.info(f"Completed collection run {run_id}: {run.papers_collected} papers collected")
                return True
                
        except Exception as e:
            logger.error(f"Failed to complete collection run {run_id}: {str(e)}")
            return False
    
    @staticmethod
    def fail_collection_run(run_id: int, error_message: str) -> bool:
        """Mark a collection run as failed.
        
        Args:
            run_id: Collection run ID
            error_message: Error description
            
        Returns:
            bool: True if update successful, False otherwise
        """
        try:
            with db_session_scope() as session:
                run = session.query(CollectionRun).filter(CollectionRun.id == run_id).first()
                if not run:
                    logger.warning(f"Collection run {run_id} not found")
                    return False
                
                run.status = CollectionStatus.FAILED
                run.collection_end = datetime.utcnow()
                run.error_message = error_message
                
                logger.error(f"Failed collection run {run_id}: {error_message}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to mark collection run {run_id} as failed: {str(e)}")
            return False
    
    @staticmethod
    def get_collection_history(limit: int = 50) -> List[CollectionRun]:
        """Get collection run history.
        
        Args:
            limit: Maximum number of runs to return
            
        Returns:
            List[CollectionRun]: List of collection runs
        """
        try:
            with db_session_scope() as session:
                runs = session.query(CollectionRun).order_by(
                    desc(CollectionRun.collection_start)
                ).limit(limit).all()
                
                # Detach from session
                for run in runs:
                    session.expunge(run)
                
                return runs
                
        except Exception as e:
            logger.error(f"Failed to get collection history: {str(e)}")
            return []
    
    @staticmethod
    def get_collection_stats() -> Dict[str, Any]:
        """Get collection statistics.
        
        Returns:
            Dict: Collection statistics
        """
        try:
            with db_session_scope() as session:
                total_runs = session.query(CollectionRun).count()
                completed_runs = session.query(CollectionRun).filter(
                    CollectionRun.status == CollectionStatus.COMPLETED
                ).count()
                failed_runs = session.query(CollectionRun).filter(
                    CollectionRun.status == CollectionStatus.FAILED
                ).count()
                running_runs = session.query(CollectionRun).filter(
                    CollectionRun.status == CollectionStatus.RUNNING
                ).count()
                
                # Get total papers collected
                total_papers = session.query(
                    func.sum(CollectionRun.papers_collected)
                ).filter(
                    CollectionRun.status == CollectionStatus.COMPLETED
                ).scalar() or 0
                
                # Get average quality score across all runs
                avg_quality = session.query(
                    func.avg(CollectionRun.avg_quality_score)
                ).filter(
                    CollectionRun.status == CollectionStatus.COMPLETED
                ).scalar() or 0.0
                
                stats = {
                    'total_runs': total_runs,
                    'completed_runs': completed_runs,
                    'failed_runs': failed_runs,
                    'running_runs': running_runs,
                    'success_rate': (completed_runs / total_runs * 100) if total_runs > 0 else 0.0,
                    'total_papers_collected': int(total_papers),
                    'avg_quality_score': float(avg_quality)
                }
                
                return stats
                
        except Exception as e:
            logger.error(f"Failed to get collection stats: {str(e)}")
            return {}