"""Tests for database models."""

import pytest
from datetime import datetime, date
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.storage.models import (
    Base, PaperModel, QualityMetrics, CollectionRun, CollectionStatus
)


class TestDatabaseModels:
    """Test cases for database models."""
    
    def setup_method(self):
        """Set up test database."""
        # Use in-memory SQLite for testing
        self.engine = create_engine('sqlite:///:memory:', echo=False)
        Base.metadata.create_all(self.engine)
        
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
    
    def teardown_method(self):
        """Clean up test database."""
        self.session.close()
        Base.metadata.drop_all(self.engine)
    
    def test_paper_model_creation(self):
        """Test creating a paper model."""
        paper = PaperModel(
            pmid='12345678',
            title='Machine Learning Applications in Pharmacogenomics: A Comprehensive Review',
            abstract='This paper reviews the current state of machine learning applications in pharmacogenomics...',
            authors_json=['Smith, J.', 'Johnson, A.', 'Williams, B.'],
            journal='Nature Genetics',
            publication_date=date(2024, 1, 15),
            doi='10.1038/ng.2024.001',
            mesh_terms_json=['Pharmacogenomics', 'Machine Learning', 'Drug Response'],
            keywords_json=['AI', 'precision medicine', 'drug discovery']
        )
        
        self.session.add(paper)
        self.session.commit()
        
        # Verify the paper was saved
        saved_paper = self.session.query(PaperModel).filter_by(pmid='12345678').first()
        assert saved_paper is not None
        assert saved_paper.title.startswith('Machine Learning Applications')
        assert saved_paper.journal == 'Nature Genetics'
        assert saved_paper.doi == '10.1038/ng.2024.001'
        assert saved_paper.created_at is not None
        assert saved_paper.updated_at is not None
    
    def test_paper_model_constraints(self):
        """Test paper model constraints."""
        # Test unique DOI constraint
        paper1 = PaperModel(
            pmid='11111111',
            title='First Paper',
            doi='10.1000/test.001'
        )
        
        paper2 = PaperModel(
            pmid='22222222',
            title='Second Paper',
            doi='10.1000/test.001'  # Same DOI - should fail
        )
        
        self.session.add(paper1)
        self.session.commit()
        
        self.session.add(paper2)
        with pytest.raises(Exception):  # Should raise unique constraint violation
            self.session.commit()
    
    def test_quality_metrics_creation(self):
        """Test creating quality metrics."""
        # First create a paper
        paper = PaperModel(
            pmid='33333333',
            title='Test Paper for Quality Metrics'
        )
        self.session.add(paper)
        self.session.commit()
        
        # Create quality metrics for the paper
        quality = QualityMetrics(
            pmid='33333333',
            journal_score=20.5,
            clinical_relevance_score=18.0,
            ml_methodology_score=22.5,
            recency_score=15.0,
            confidence_level=0.85,
            scoring_version='1.0'
        )
        
        # Calculate total score
        quality.calculate_total_score()
        
        self.session.add(quality)
        self.session.commit()
        
        # Verify the quality metrics were saved
        saved_quality = self.session.query(QualityMetrics).filter_by(pmid='33333333').first()
        assert saved_quality is not None
        assert saved_quality.journal_score == 20.5
        assert saved_quality.clinical_relevance_score == 18.0
        assert saved_quality.ml_methodology_score == 22.5
        assert saved_quality.recency_score == 15.0
        assert saved_quality.total_score == 76.0  # Sum of component scores
        assert saved_quality.confidence_level == 0.85
    
    def test_quality_metrics_constraints(self):
        """Test quality metrics constraints."""
        # First create a paper
        paper = PaperModel(
            pmid='44444444',
            title='Test Paper for Constraint Testing'
        )
        self.session.add(paper)
        self.session.commit()
        
        # Test invalid score (> 25)
        quality = QualityMetrics(
            pmid='44444444',
            journal_score=30.0,  # Invalid: > 25
            clinical_relevance_score=20.0,
            ml_methodology_score=20.0,
            recency_score=20.0
        )
        
        self.session.add(quality)
        
        with pytest.raises(Exception):  # Should raise constraint violation
            self.session.commit()
    
    def test_collection_run_creation(self):
        """Test creating collection runs."""
        # Create a collection run
        run = CollectionRun(
            query_used='pharmacogenomics AND machine learning',
            papers_collected=150,
            papers_processed=120,
            avg_quality_score=75.5,
            status=CollectionStatus.RUNNING,
            config_snapshot={
                'max_results': 1000,
                'quality_threshold': 60.0
            }
        )
        
        self.session.add(run)
        self.session.commit()
        
        # Verify the collection run was saved
        saved_run = self.session.query(CollectionRun).filter_by(query_used='pharmacogenomics AND machine learning').first()
        assert saved_run is not None
        assert saved_run.papers_collected == 150
        assert saved_run.papers_processed == 120
        assert saved_run.avg_quality_score == 75.5
        assert saved_run.status == CollectionStatus.RUNNING
        assert saved_run.config_snapshot['max_results'] == 1000
    
    def test_processing_log_creation(self):
        """Test creating processing logs."""
        log = ProcessingLog(
            job_id='job_20240822_001',
            pipeline_name='variant_processing',
            step_name='quality_assessment',
            status='completed',
            started_at=datetime.utcnow(),
            completed_at=datetime.utcnow(),
            duration_seconds=45.2,
            input_count=1000,
            output_count=950,
            error_count=50,
            parameters={'threshold': 0.5, 'method': 'composite'},
            metrics={'accuracy': 0.95, 'recall': 0.88},
            memory_usage_mb=256.5,
            cpu_usage_percent=75.2
        )
        
        self.session.add(log)
        self.session.commit()
        
        # Verify the log was saved
        saved_log = self.session.query(ProcessingLog).filter_by(job_id='job_20240822_001').first()
        assert saved_log is not None
        assert saved_log.pipeline_name == 'variant_processing'
        assert saved_log.status == 'completed'
        assert saved_log.input_count == 1000
        assert saved_log.parameters['threshold'] == 0.5
        assert saved_log.metrics['accuracy'] == 0.95
    
    def test_analysis_report_creation(self):
        """Test creating analysis reports."""
        report = AnalysisReport(
            report_id='report_001',
            report_name='Variant Quality Analysis',
            report_type='quality_assessment',
            analysis_method='statistical_summary',
            analysis_version='2.1',
            data_source='ClinVar_2024',
            variant_count=5000,
            association_count=1200,
            gene_count=500,
            drug_count=300,
            summary_statistics={
                'mean_quality_score': 0.75,
                'median_quality_score': 0.78,
                'high_quality_variants': 3500
            },
            key_findings=[
                'High quality variants represent 70% of dataset',
                'Most variants are in cancer-related genes'
            ],
            quality_metrics={'completeness': 0.95, 'accuracy': 0.92},
            status='completed',
            created_by='analysis_system'
        )
        
        self.session.add(report)
        self.session.commit()
        
        # Verify the report was saved
        saved_report = self.session.query(AnalysisReport).filter_by(report_id='report_001').first()
        assert saved_report is not None
        assert saved_report.report_name == 'Variant Quality Analysis'
        assert saved_report.variant_count == 5000
        assert saved_report.summary_statistics['mean_quality_score'] == 0.75
        assert 'High quality variants' in saved_report.key_findings[0]
    
    def test_relationships(self):
        """Test model relationships."""
        # Create a variant
        variant = GenomicVariant(
            variant_id='rs111222',
            chromosome='3',
            position=98765,
            reference_allele='G',
            alternate_allele='A',
            gene_symbol='APOE',
            source_database='ClinVar'
        )
        self.session.add(variant)
        self.session.flush()
        
        # Create a drug-gene association linked to the variant
        association = DrugGeneAssociation(
            association_id='assoc_003',
            drug_name='statins',
            gene_symbol='APOE',
            variant_id=variant.id,
            association_type='efficacy',
            evidence_level='2A',
            source_database='PharmGKB'
        )
        self.session.add(association)
        
        # Create quality scores for both
        variant_quality = QualityScore(
            variant_id=variant.id,
            overall_score=0.82,
            scoring_method='variant_v1',
            calculated_by='system'
        )
        
        association_quality = QualityScore(
            association_id=association.id,
            overall_score=0.75,
            scoring_method='association_v1',
            calculated_by='system'
        )
        
        self.session.add_all([variant_quality, association_quality])
        self.session.commit()
        
        # Test relationships
        # Variant should have quality scores and drug associations
        assert len(variant.quality_scores) == 1
        assert len(variant.drug_associations) == 1
        assert variant.quality_scores[0].overall_score == 0.82
        assert variant.drug_associations[0].drug_name == 'statins'
        
        # Association should have quality scores and link back to variant
        assert len(association.quality_scores) == 1
        assert association.variant == variant
        assert association.quality_scores[0].overall_score == 0.75
    
    def test_json_fields(self):
        """Test JSON field storage and retrieval."""
        variant = GenomicVariant(
            variant_id='rs333444',
            chromosome='4',
            position=55555,
            reference_allele='T',
            alternate_allele='C',
            population_frequency={
                'EUR': 0.05,
                'AFR': 0.12,
                'ASN': 0.03,
                'AMR': 0.08
            },
            annotations={
                'consequence': 'missense_variant',
                'impact': 'moderate',
                'sift_score': 0.02,
                'polyphen_score': 0.95,
                'cadd_score': 25.3
            },
            source_database='gnomAD'
        )
        
        self.session.add(variant)
        self.session.commit()
        
        # Verify JSON fields are properly stored and retrieved
        saved_variant = self.session.query(GenomicVariant).filter_by(variant_id='rs333444').first()
        assert saved_variant.population_frequency['EUR'] == 0.05
        assert saved_variant.population_frequency['AFR'] == 0.12
        assert saved_variant.annotations['consequence'] == 'missense_variant'
        assert saved_variant.annotations['sift_score'] == 0.02
        assert saved_variant.annotations['cadd_score'] == 25.3