"""Tests for PharmacogenomicsQualityScorer."""

import pytest
from datetime import datetime, date
from src.quality.quality_scorer import PharmacogenomicsQualityScorer, QualityScoreComponents
from src.data_ingestion.data_models import PaperMetadata, Author, MeshTerm, Journal


class TestQualityScoreComponents:
    """Test cases for QualityScoreComponents."""
    
    def test_init_default(self):
        """Test default initialization."""
        components = QualityScoreComponents()
        
        assert components.journal_score == 0.0
        assert components.clinical_relevance_score == 0.0
        assert components.ml_methodology_score == 0.0
        assert components.recency_score == 0.0
        assert components.total_score == 0.0
        assert components.confidence_level == 0.0
        assert components.scoring_version == "1.0"
        assert components.scored_at is not None
    
    def test_init_with_values(self):
        """Test initialization with values."""
        components = QualityScoreComponents(
            journal_score=20.0,
            clinical_relevance_score=18.0,
            ml_methodology_score=22.0,
            recency_score=15.0,
            total_score=75.0,
            confidence_level=0.85
        )
        
        assert components.journal_score == 20.0
        assert components.clinical_relevance_score == 18.0
        assert components.ml_methodology_score == 22.0
        assert components.recency_score == 15.0
        assert components.total_score == 75.0
        assert components.confidence_level == 0.85


class TestPharmacogenomicsQualityScorer:
    """Test cases for PharmacogenomicsQualityScorer."""
    
    def test_init(self):
        """Test scorer initialization."""
        scorer = PharmacogenomicsQualityScorer()
        
        assert scorer.journal_scorer is not None
        assert len(scorer.weights) == 4
        assert sum(scorer.weights.values()) == 100.0
        assert len(scorer.pgx_core_terms) > 0
        assert len(scorer.pharmacogenes) > 0
        assert len(scorer.clinical_drugs) > 0
        assert len(scorer.ml_methodology_terms) > 0
    
    def create_sample_paper(self, **kwargs) -> PaperMetadata:
        """Create a sample paper for testing."""
        defaults = {
            'pmid': '12345',
            'title': 'Test Paper',
            'abstract': 'This is a test abstract.',
            'authors': [Author(last_name='Smith', first_name='John')],
            'journal': Journal(title='Test Journal'),
            'publication_date': datetime(2023, 1, 1),
            'mesh_terms': [],
            'keywords': [],
            'publication_types': ['Journal Article']
        }
        defaults.update(kwargs)
        return PaperMetadata(**defaults)
    
    def test_score_paper_basic(self):
        """Test basic paper scoring."""
        scorer = PharmacogenomicsQualityScorer()
        
        paper = self.create_sample_paper()
        scores = scorer.score_paper(paper)
        
        assert isinstance(scores, QualityScoreComponents)
        assert scores.total_score >= 0
        assert scores.total_score <= 100
        assert scores.confidence_level >= 0
        assert scores.confidence_level <= 1
        assert scores.scored_at is not None
    
    def test_score_journal_quality_high_impact(self):
        """Test journal quality scoring for high-impact journals."""
        scorer = PharmacogenomicsQualityScorer()
        
        # Test with Nature
        paper = self.create_sample_paper(
            journal=Journal(title='Nature')
        )
        scores = scorer.score_paper(paper)
        
        assert scores.journal_score > 20.0  # Should be high for Nature
        
        # Test with pharmacogenomics journal
        pgx_paper = self.create_sample_paper(
            journal=Journal(title='Pharmacogenomics'),
            title='CYP2D6 pharmacogenomics study',
            abstract='This study investigates CYP2D6 variants and drug metabolism.'
        )
        pgx_scores = scorer.score_paper(pgx_paper)
        
        # Should get domain bonus
        assert pgx_scores.journal_score > 15.0
    
    def test_score_clinical_relevance_high(self):
        """Test clinical relevance scoring for highly relevant papers."""
        scorer = PharmacogenomicsQualityScorer()
        
        paper = self.create_sample_paper(
            title='Pharmacogenomics of CYP2D6 and warfarin dosing',
            abstract='This randomized controlled trial investigates CYP2D6 variants and warfarin dosing in precision medicine.',
            mesh_terms=[
                MeshTerm(descriptor_name='Pharmacogenetics', major_topic=True),
                MeshTerm(descriptor_name='Warfarin', major_topic=True)
            ],
            publication_types=['Journal Article', 'Randomized Controlled Trial']
        )
        
        scores = scorer.score_paper(paper)
        
        # Should score reasonably on clinical relevance
        assert scores.clinical_relevance_score > 8.0
    
    def test_score_ml_methodology_high(self):
        """Test ML methodology scoring for ML-heavy papers."""
        scorer = PharmacogenomicsQualityScorer()
        
        paper = self.create_sample_paper(
            title='Deep learning for pharmacogenomics prediction',
            abstract='We developed a deep neural network with cross-validation and feature selection for predicting drug response using machine learning and artificial intelligence methods.'
        )
        
        scores = scorer.score_paper(paper)
        
        # Should score reasonably on ML methodology
        assert scores.ml_methodology_score > 8.0
    
    def test_score_recency_recent_paper(self):
        """Test recency scoring for recent papers."""
        scorer = PharmacogenomicsQualityScorer()
        
        # Very recent paper
        recent_paper = self.create_sample_paper(
            publication_date=datetime(2024, 1, 1)
        )
        recent_scores = scorer.score_paper(recent_paper)
        
        # Old paper
        old_paper = self.create_sample_paper(
            publication_date=datetime(2010, 1, 1)
        )
        old_scores = scorer.score_paper(old_paper)
        
        # Recent paper should score higher
        assert recent_scores.recency_score > old_scores.recency_score
        assert recent_scores.recency_score > 20.0
        assert old_scores.recency_score < 10.0
    
    def test_score_recency_no_date(self):
        """Test recency scoring when no publication date is available."""
        scorer = PharmacogenomicsQualityScorer()
        
        paper = self.create_sample_paper(publication_date=None)
        scores = scorer.score_paper(paper)
        
        # Should get default middle score
        assert scores.recency_score == 10.0
    
    def test_calculate_confidence_high(self):
        """Test confidence calculation for high-quality data."""
        scorer = PharmacogenomicsQualityScorer()
        
        paper = self.create_sample_paper(
            journal=Journal(title='Nature'),  # High-tier journal
            abstract='Detailed abstract with lots of information.',
            mesh_terms=[MeshTerm(descriptor_name='Pharmacogenetics')],
            authors=[Author(last_name='Smith', first_name='John')],
            publication_date=datetime(2023, 1, 1)
        )
        
        scores = scorer.score_paper(paper)
        
        # Should have high confidence
        assert scores.confidence_level > 0.7
    
    def test_calculate_confidence_low(self):
        """Test confidence calculation for low-quality data."""
        scorer = PharmacogenomicsQualityScorer()
        
        paper = self.create_sample_paper(
            journal=None,  # No journal
            abstract='',   # No abstract
            mesh_terms=[], # No MeSH terms
            authors=[],    # No authors
            publication_date=None  # No date
        )
        
        scores = scorer.score_paper(paper)
        
        # Should have low confidence
        assert scores.confidence_level < 0.5
    
    def test_score_papers_batch(self):
        """Test batch scoring of multiple papers."""
        scorer = PharmacogenomicsQualityScorer()
        
        papers = [
            self.create_sample_paper(pmid='1', title='Paper 1'),
            self.create_sample_paper(pmid='2', title='Paper 2'),
            self.create_sample_paper(pmid='3', title='Paper 3')
        ]
        
        scores = scorer.score_papers_batch(papers)
        
        assert len(scores) == 3
        assert all(isinstance(score, QualityScoreComponents) for score in scores)
        assert all(score.total_score >= 0 for score in scores)
    
    def test_score_papers_batch_with_error(self):
        """Test batch scoring handles errors gracefully."""
        scorer = PharmacogenomicsQualityScorer()
        
        # Create a paper that might cause errors
        problematic_paper = PaperMetadata(pmid='error')  # Minimal paper
        
        papers = [
            self.create_sample_paper(pmid='1'),
            problematic_paper,
            self.create_sample_paper(pmid='3')
        ]
        
        scores = scorer.score_papers_batch(papers)
        
        # Should still return scores for all papers
        assert len(scores) == 3
        assert all(isinstance(score, QualityScoreComponents) for score in scores)
    
    def test_get_quality_distribution(self):
        """Test quality distribution analysis."""
        scorer = PharmacogenomicsQualityScorer()
        
        # Create papers with different quality levels
        high_quality_paper = self.create_sample_paper(
            journal=Journal(title='Nature'),
            title='Pharmacogenomics machine learning study',
            abstract='Deep learning for CYP2D6 prediction with cross-validation',
            publication_date=datetime(2024, 1, 1)
        )
        
        low_quality_paper = self.create_sample_paper(
            journal=Journal(title='Unknown Journal'),
            title='Basic study',
            abstract='Simple study',
            publication_date=datetime(2010, 1, 1)
        )
        
        scores = [
            scorer.score_paper(high_quality_paper),
            scorer.score_paper(low_quality_paper)
        ]
        
        distribution = scorer.get_quality_distribution(scores)
        
        assert 'total_papers' in distribution
        assert 'total_score_stats' in distribution
        assert 'component_stats' in distribution
        assert 'quality_distribution' in distribution
        assert 'average_confidence' in distribution
        
        assert distribution['total_papers'] == 2
        assert 'mean' in distribution['total_score_stats']
        assert 'journal' in distribution['component_stats']
    
    def test_get_quality_distribution_empty(self):
        """Test quality distribution with empty list."""
        scorer = PharmacogenomicsQualityScorer()
        
        distribution = scorer.get_quality_distribution([])
        
        assert distribution == {}
    
    def test_get_top_papers(self):
        """Test getting top-scoring papers."""
        scorer = PharmacogenomicsQualityScorer()
        
        # Create papers with different scores
        papers_and_scores = []
        
        for i in range(5):
            paper = self.create_sample_paper(
                pmid=str(i),
                title=f'Paper {i}',
                journal=Journal(title='Nature' if i < 2 else 'Test Journal')
            )
            score = scorer.score_paper(paper)
            papers_and_scores.append((paper, score))
        
        top_papers = scorer.get_top_papers(papers_and_scores, limit=3)
        
        assert len(top_papers) == 3
        
        # Should be sorted by score (descending)
        scores = [score.total_score for _, score in top_papers]
        assert scores == sorted(scores, reverse=True)
    
    def test_comprehensive_scoring(self):
        """Test comprehensive scoring with all components."""
        scorer = PharmacogenomicsQualityScorer()
        
        # Create a high-quality pharmacogenomics + ML paper
        paper = self.create_sample_paper(
            title='Machine learning for CYP2D6 pharmacogenomics in warfarin dosing',
            abstract='''This randomized controlled trial investigates the use of deep learning 
                       and neural networks for predicting warfarin dosing based on CYP2D6 
                       pharmacogenomics variants. We used cross-validation and feature selection 
                       in our machine learning approach for precision medicine.''',
            journal=Journal(title='Clinical Pharmacology & Therapeutics'),
            publication_date=datetime(2024, 1, 1),
            mesh_terms=[
                MeshTerm(descriptor_name='Pharmacogenetics', major_topic=True),
                MeshTerm(descriptor_name='Machine Learning', major_topic=True),
                MeshTerm(descriptor_name='Warfarin', major_topic=False)
            ],
            keywords=['pharmacogenomics', 'machine learning', 'CYP2D6', 'warfarin'],
            publication_types=['Journal Article', 'Randomized Controlled Trial'],
            authors=[
                Author(last_name='Smith', first_name='John', affiliation='University of Medicine'),
                Author(last_name='Doe', first_name='Jane', affiliation='Research Institute')
            ]
        )
        
        scores = scorer.score_paper(paper)
        
        # Should score well in all components
        assert scores.journal_score > 15.0  # Good pharmacogenomics journal
        assert scores.clinical_relevance_score > 8.0  # Good PGx relevance
        assert scores.ml_methodology_score > 5.0  # Some ML content
        assert scores.recency_score > 20.0  # Recent publication
        assert scores.total_score > 50.0  # Good overall score
        assert scores.confidence_level > 0.7  # Good confidence
    
    def test_scoring_weights_sum_to_100(self):
        """Test that scoring weights sum to 100."""
        scorer = PharmacogenomicsQualityScorer()
        
        total_weight = sum(scorer.weights.values())
        assert abs(total_weight - 100.0) < 0.01  # Allow for floating point precision
    
    def test_score_components_within_bounds(self):
        """Test that individual score components stay within expected bounds."""
        scorer = PharmacogenomicsQualityScorer()
        
        # Test with various papers
        papers = [
            self.create_sample_paper(journal=Journal(title='Nature')),
            self.create_sample_paper(journal=Journal(title='PLoS ONE')),
            self.create_sample_paper(journal=None),
        ]
        
        for paper in papers:
            scores = scorer.score_paper(paper)
            
            # Each component should be within its weight limit
            assert 0 <= scores.journal_score <= scorer.weights['journal']
            assert 0 <= scores.clinical_relevance_score <= scorer.weights['clinical_relevance']
            assert 0 <= scores.ml_methodology_score <= scorer.weights['ml_methodology']
            assert 0 <= scores.recency_score <= scorer.weights['recency']
            
            # Total should not exceed 100
            assert scores.total_score <= 100.0