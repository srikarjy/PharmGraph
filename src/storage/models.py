"""SQLAlchemy ORM models for genomics paper collection and quality scoring."""

from datetime import datetime, date
from typing import Dict, Any, Optional, List
import json
import enum

from sqlalchemy import (
    Column, Integer, String, Float, DateTime, Date, Text, Boolean,
    ForeignKey, Index, UniqueConstraint, CheckConstraint, Enum
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.sqlite import JSON
from sqlalchemy.types import TypeDecorator, VARCHAR


class JSONType(TypeDecorator):
    """Platform-independent JSON type."""
    
    impl = VARCHAR
    cache_ok = True
    
    def load_dialect_impl(self, dialect):
        if dialect.name == 'postgresql':
            return dialect.type_descriptor(JSONB())
        else:
            return dialect.type_descriptor(JSON())
    
    def process_bind_param(self, value, dialect):
        if value is not None:
            return json.dumps(value)
        return value
    
    def process_result_value(self, value, dialect):
        if value is not None:
            return json.loads(value)
        return value


class CollectionStatus(enum.Enum):
    """Enumeration for collection run status."""
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


Base = declarative_base()


class PaperModel(Base):
    """Model for storing research paper information from NCBI and other sources."""
    
    __tablename__ = 'papers'
    
    # Primary key
    pmid = Column(String(20), primary_key=True)  # PubMed ID
    
    # Paper metadata
    title = Column(Text, nullable=False, index=True)
    abstract = Column(Text, nullable=True)
    authors_json = Column(JSONType)  # JSON serialized list of authors
    journal = Column(String(255), nullable=True, index=True)
    publication_date = Column(Date, nullable=True, index=True)
    doi = Column(String(100), unique=True, nullable=True)
    
    # Classification and keywords
    mesh_terms_json = Column(JSONType)  # JSON serialized list of MeSH terms
    keywords_json = Column(JSONType)  # JSON serialized list of keywords
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    quality_metrics = relationship("QualityMetrics", back_populates="paper", cascade="all, delete-orphan")
    
    # Constraints and indexes
    __table_args__ = (
        Index('idx_paper_title', 'title'),
        Index('idx_paper_journal_date', 'journal', 'publication_date'),
        Index('idx_paper_created', 'created_at'),
    )
    
    def __repr__(self):
        return f"<PaperModel(pmid='{self.pmid}', title='{self.title[:50]}...')>"
    
    @property
    def authors(self) -> List[str]:
        """Get authors as a list."""
        return self.authors_json or []
    
    @authors.setter
    def authors(self, value: List[str]):
        """Set authors from a list."""
        self.authors_json = value
    
    @property
    def mesh_terms(self) -> List[str]:
        """Get MeSH terms as a list."""
        return self.mesh_terms_json or []
    
    @mesh_terms.setter
    def mesh_terms(self, value: List[str]):
        """Set MeSH terms from a list."""
        self.mesh_terms_json = value
    
    @property
    def keywords(self) -> List[str]:
        """Get keywords as a list."""
        return self.keywords_json or []
    
    @keywords.setter
    def keywords(self, value: List[str]):
        """Set keywords from a list."""
        self.keywords_json = value


class QualityMetrics(Base):
    """Model for storing quality assessment scores for papers."""
    
    __tablename__ = 'quality_metrics'
    
    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Foreign key to paper
    pmid = Column(String(20), ForeignKey('papers.pmid'), nullable=False, index=True)
    
    # Quality score components (each 0-25 points)
    journal_score = Column(Float, nullable=False, default=0.0)  # Journal impact and reputation
    clinical_relevance_score = Column(Float, nullable=False, default=0.0)  # Clinical applicability
    ml_methodology_score = Column(Float, nullable=False, default=0.0)  # ML/AI methodology quality
    recency_score = Column(Float, nullable=False, default=0.0)  # Publication recency
    
    # Total score (0-100)
    total_score = Column(Float, nullable=False, default=0.0, index=True)
    
    # Confidence and versioning
    confidence_level = Column(Float, nullable=False, default=0.0)  # 0-1 confidence in scoring
    scoring_version = Column(String(10), nullable=False, default="1.0")
    
    # Timestamp
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    paper = relationship("PaperModel", back_populates="quality_metrics")
    
    # Constraints
    __table_args__ = (
        Index('idx_quality_total_score', 'total_score'),
        Index('idx_quality_pmid_score', 'pmid', 'total_score'),
        CheckConstraint('journal_score >= 0 AND journal_score <= 25', name='check_journal_score'),
        CheckConstraint('clinical_relevance_score >= 0 AND clinical_relevance_score <= 25', name='check_clinical_score'),
        CheckConstraint('ml_methodology_score >= 0 AND ml_methodology_score <= 25', name='check_ml_score'),
        CheckConstraint('recency_score >= 0 AND recency_score <= 25', name='check_recency_score'),
        CheckConstraint('total_score >= 0 AND total_score <= 100', name='check_total_score'),
        CheckConstraint('confidence_level >= 0 AND confidence_level <= 1', name='check_confidence_level'),
    )
    
    def __repr__(self):
        return f"<QualityMetrics(id={self.id}, pmid='{self.pmid}', total_score={self.total_score})>"
    
    def calculate_total_score(self):
        """Calculate and update the total score from component scores."""
        self.total_score = (
            self.journal_score + 
            self.clinical_relevance_score + 
            self.ml_methodology_score + 
            self.recency_score
        )
        return self.total_score


class CollectionRun(Base):
    """Model for tracking paper collection runs and their progress."""
    
    __tablename__ = 'collection_runs'
    
    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Collection parameters
    query_used = Column(String(500), nullable=False)
    
    # Progress tracking
    papers_collected = Column(Integer, nullable=False, default=0)
    papers_processed = Column(Integer, nullable=False, default=0)
    avg_quality_score = Column(Float, nullable=True)
    
    # Timing
    collection_start = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    collection_end = Column(DateTime, nullable=True)
    
    # Status tracking
    status = Column(Enum(CollectionStatus), nullable=False, default=CollectionStatus.RUNNING, index=True)
    error_message = Column(Text, nullable=True)
    
    # Configuration snapshot
    config_snapshot = Column(JSONType)  # JSON of config used for this run
    
    # Constraints and indexes
    __table_args__ = (
        Index('idx_collection_status', 'status'),
        Index('idx_collection_start', 'collection_start'),
        Index('idx_collection_status_start', 'status', 'collection_start'),
    )
    
    def __repr__(self):
        return f"<CollectionRun(id={self.id}, status='{self.status.value}', papers={self.papers_collected})>"
    
    @property
    def duration_seconds(self) -> Optional[float]:
        """Calculate duration of the collection run."""
        if self.collection_end and self.collection_start:
            return (self.collection_end - self.collection_start).total_seconds()
        return None
    
    @property
    def is_running(self) -> bool:
        """Check if the collection run is currently running."""
        return self.status == CollectionStatus.RUNNING
    
    @property
    def is_completed(self) -> bool:
        """Check if the collection run completed successfully."""
        return self.status == CollectionStatus.COMPLETED


