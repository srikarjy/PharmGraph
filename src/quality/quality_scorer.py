"""Main quality scoring system for pharmacogenomics research papers."""

import re
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from ..config import get_logger
from ..data_ingestion.data_models import PaperMetadata
from .journal_scorer import JournalImpactScorer


logger = get_logger(__name__)


@dataclass
class QualityScoreComponents:
    """Individual components of quality score."""
    journal_score: float = 0.0
    clinical_relevance_score: float = 0.0
    ml_methodology_score: float = 0.0
    recency_score: float = 0.0
    total_score: float = 0.0
    
    # Detailed breakdowns
    journal_base_score: float = 0.0
    journal_impact_bonus: float = 0.0
    journal_domain_bonus: float = 0.0
    
    # Confidence and metadata
    confidence_level: float = 0.0
    scoring_version: str = "1.0"
    scored_at: datetime = None
    
    def __post_init__(self):
        if self.scored_at is None:
            self.scored_at = datetime.utcnow()


class PharmacogenomicsQualityScorer:
    """Comprehensive quality scorer for pharmacogenomics research papers."""
    
    def __init__(self):
        """Initialize the quality scorer."""
        self.journal_scorer = JournalImpactScorer()
        
        # Scoring weights (total = 100 points)
        self.weights = {
            'journal': 25.0,           # Journal impact and prestige
            'clinical_relevance': 25.0, # Clinical applicability
            'ml_methodology': 25.0,     # ML/AI methodology quality
            'recency': 25.0            # Publication recency
        }
        
        # Domain-specific terms for scoring
        self._initialize_scoring_terms()
        
        logger.info("Initialized PharmacogenomicsQualityScorer")
    
    def _initialize_scoring_terms(self):
        """Initialize domain-specific terms for quality scoring."""
        
        # High-value pharmacogenomics terms
        self.pgx_core_terms = [
            'pharmacogenomics', 'pharmacogenetics', 'precision medicine',
            'personalized medicine', 'individualized therapy', 'drug metabolism'
        ]
        
        # Pharmacogenes (high clinical relevance)
        self.pharmacogenes = [
            'cyp2d6', 'cyp2c19', 'cyp2c9', 'cyp3a4', 'cyp3a5', 'cyp1a2',
            'dpyd', 'tpmt', 'ugt1a1', 'slco1b1', 'abcb1', 'vkorc1',
            'cftr', 'g6pd', 'hla-b', 'hla-a', 'nudt15', 'ryr1'
        ]
        
        # High-impact drugs
        self.clinical_drugs = [
            'warfarin', 'clopidogrel', 'tamoxifen', 'codeine', 'tramadol',
            'simvastatin', 'atorvastatin', 'metoprolol', 'propranolol',
            '5-fluorouracil', 'irinotecan', 'tacrolimus', 'azathioprine'
        ]
        
        # ML/AI methodology terms
        self.ml_methodology_terms = [
            'machine learning', 'artificial intelligence', 'deep learning',
            'neural network', 'random forest', 'support vector machine',
            'gradient boosting', 'ensemble methods', 'cross-validation',
            'feature selection', 'model validation', 'predictive modeling',
            'classification', 'regression', 'clustering', 'dimensionality reduction'
        ]
        
        # Advanced ML terms (higher weight)
        self.advanced_ml_terms = [
            'deep neural network', 'convolutional neural network', 'lstm',
            'transformer', 'attention mechanism', 'transfer learning',
            'federated learning', 'reinforcement learning', 'gan',
            'autoencoder', 'bert', 'gpt', 'explainable ai', 'interpretable ml'
        ]
        
        # Clinical relevance indicators
        self.clinical_terms = [
            'clinical trial', 'randomized controlled trial', 'cohort study',
            'case-control study', 'meta-analysis', 'systematic review',
            'clinical implementation', 'clinical decision support',
            'drug dosing', 'adverse drug reaction', 'therapeutic outcome',
            'biomarker', 'diagnostic', 'prognostic', 'treatment response'
        ]
        
        # High-impact study types
        self.high_impact_studies = [
            'randomized controlled trial', 'meta-analysis', 'systematic review',
            'genome-wide association study', 'clinical implementation study',
            'prospective cohort', 'multicenter study'
        ]
    
    def score_paper(self, paper: PaperMetadata) -> QualityScoreComponents:
        """Score a paper comprehensively.
        
        Args:
            paper: Paper metadata to score
            
        Returns:
            QualityScoreComponents: Detailed quality scores
        """
        logger.debug(f"Scoring paper: {paper.pmid}")
        
        # Initialize score components
        scores = QualityScoreComponents()
        
        # Score each component
        scores.journal_score = self._score_journal_quality(paper, scores)
        scores.clinical_relevance_score = self._score_clinical_relevance(paper)
        scores.ml_methodology_score = self._score_ml_methodology(paper)
        scores.recency_score = self._score_recency(paper)
        
        # Calculate total score
        scores.total_score = (
            scores.journal_score + 
            scores.clinical_relevance_score + 
            scores.ml_methodology_score + 
            scores.recency_score
        )
        
        # Calculate confidence level
        scores.confidence_level = self._calculate_confidence(paper, scores)
        
        logger.debug(f"Paper {paper.pmid} scored {scores.total_score:.1f} points")
        return scores
    
    def _score_journal_quality(self, paper: PaperMetadata, scores: QualityScoreComponents) -> float:
        """Score journal quality component.
        
        Args:
            paper: Paper metadata
            scores: Score components to update with details
            
        Returns:
            float: Journal quality score (0-25 points)
        """
        if not paper.journal:
            return 0.0
        
        # Get paper content for domain relevance
        content = f"{paper.title} {paper.abstract}"
        
        # Get detailed journal scoring
        raw_score = self.journal_scorer.get_journal_score_with_domain_bonus(
            paper.journal.title, content
        )
        
        # Get base score without bonuses for breakdown
        base_score = self.journal_scorer.get_journal_score(paper.journal.title)
        
        # Calculate bonuses
        impact_bonus = 0.0
        if paper.journal:
            impact_factor = self.journal_scorer.get_impact_factor(paper.journal.title)
            if impact_factor:
                import math
                impact_bonus = min(math.log10(impact_factor) * 10, 20.0)
        
        domain_bonus = raw_score - base_score - impact_bonus
        
        # Store detailed breakdown
        scores.journal_base_score = min(base_score, 25.0)
        scores.journal_impact_bonus = min(impact_bonus * 0.2, 5.0)  # Scale to fit in 25 points
        scores.journal_domain_bonus = min(domain_bonus * 0.2, 5.0)
        
        # Scale to 25 points maximum
        final_score = min(raw_score * (self.weights['journal'] / 100.0), self.weights['journal'])
        
        return final_score
    
    def _score_clinical_relevance(self, paper: PaperMetadata) -> float:
        """Score clinical relevance component.
        
        Args:
            paper: Paper metadata
            
        Returns:
            float: Clinical relevance score (0-25 points)
        """
        score = 0.0
        content = f"{paper.title} {paper.abstract}".lower()
        
        # Base pharmacogenomics relevance (0-10 points)
        pgx_matches = sum(1 for term in self.pgx_core_terms if term in content)
        score += min(pgx_matches * 2.0, 10.0)
        
        # Pharmacogene mentions (0-8 points)
        gene_matches = sum(1 for gene in self.pharmacogenes if gene in content)
        score += min(gene_matches * 1.0, 8.0)
        
        # Clinical drug mentions (0-5 points)
        drug_matches = sum(1 for drug in self.clinical_drugs if drug in content)
        score += min(drug_matches * 0.5, 5.0)
        
        # Clinical study type bonus (0-2 points)
        study_matches = sum(1 for study in self.high_impact_studies if study in content)
        score += min(study_matches * 1.0, 2.0)
        
        # MeSH term relevance bonus
        if paper.mesh_terms:
            relevant_mesh = sum(1 for term in paper.mesh_terms 
                              if any(pgx_term in term.descriptor_name.lower() 
                                   for pgx_term in self.pgx_core_terms))
            score += min(relevant_mesh * 0.5, 2.0)
        
        # Clinical trial bonus
        if paper.is_clinical_trial:
            score += 3.0
        
        return min(score, self.weights['clinical_relevance'])
    
    def _score_ml_methodology(self, paper: PaperMetadata) -> float:
        """Score ML/AI methodology component.
        
        Args:
            paper: Paper metadata
            
        Returns:
            float: ML methodology score (0-25 points)
        """
        score = 0.0
        content = f"{paper.title} {paper.abstract}".lower()
        
        # Basic ML terms (0-10 points)
        ml_matches = sum(1 for term in self.ml_methodology_terms if term in content)
        score += min(ml_matches * 1.0, 10.0)
        
        # Advanced ML terms bonus (0-8 points)
        advanced_matches = sum(1 for term in self.advanced_ml_terms if term in content)
        score += min(advanced_matches * 2.0, 8.0)
        
        # Methodology quality indicators (0-5 points)
        quality_terms = [
            'validation', 'cross-validation', 'external validation',
            'performance evaluation', 'model comparison', 'feature importance',
            'hyperparameter optimization', 'model interpretation'
        ]
        quality_matches = sum(1 for term in quality_terms if term in content)
        score += min(quality_matches * 0.5, 5.0)
        
        # Statistical rigor indicators (0-2 points)
        stats_terms = [
            'statistical significance', 'confidence interval', 'p-value',
            'effect size', 'power analysis', 'multiple testing correction'
        ]
        stats_matches = sum(1 for term in stats_terms if term in content)
        score += min(stats_matches * 0.3, 2.0)
        
        return min(score, self.weights['ml_methodology'])
    
    def _score_recency(self, paper: PaperMetadata) -> float:
        """Score publication recency component.
        
        Args:
            paper: Paper metadata
            
        Returns:
            float: Recency score (0-25 points)
        """
        if not paper.publication_date:
            return 10.0  # Default middle score for unknown dates
        
        current_year = datetime.now().year
        pub_year = paper.publication_date.year
        
        years_old = current_year - pub_year
        
        # Scoring based on age
        if years_old <= 1:
            return self.weights['recency']  # 25 points for very recent
        elif years_old <= 2:
            return self.weights['recency'] * 0.9  # 22.5 points
        elif years_old <= 3:
            return self.weights['recency'] * 0.8  # 20 points
        elif years_old <= 5:
            return self.weights['recency'] * 0.6  # 15 points
        elif years_old <= 7:
            return self.weights['recency'] * 0.4  # 10 points
        elif years_old <= 10:
            return self.weights['recency'] * 0.2  # 5 points
        else:
            return 0.0  # 0 points for very old papers
    
    def _calculate_confidence(self, paper: PaperMetadata, scores: QualityScoreComponents) -> float:
        """Calculate confidence level in the scoring.
        
        Args:
            paper: Paper metadata
            scores: Score components
            
        Returns:
            float: Confidence level (0-1)
        """
        confidence_factors = []
        
        # Journal confidence
        if paper.journal:
            journal_tier = self.journal_scorer.get_journal_tier(paper.journal.title)
            if journal_tier <= 3:
                confidence_factors.append(0.9)
            elif journal_tier <= 5:
                confidence_factors.append(0.7)
            else:
                confidence_factors.append(0.5)
        else:
            confidence_factors.append(0.3)
        
        # Abstract availability
        if paper.has_abstract:
            confidence_factors.append(0.9)
        else:
            confidence_factors.append(0.4)
        
        # MeSH terms availability
        if paper.mesh_terms:
            confidence_factors.append(0.8)
        else:
            confidence_factors.append(0.5)
        
        # Publication date availability
        if paper.publication_date:
            confidence_factors.append(0.9)
        else:
            confidence_factors.append(0.6)
        
        # Author information
        if paper.authors:
            confidence_factors.append(0.8)
        else:
            confidence_factors.append(0.4)
        
        # Calculate average confidence
        return sum(confidence_factors) / len(confidence_factors)
    
    def score_papers_batch(self, papers: List[PaperMetadata]) -> List[QualityScoreComponents]:
        """Score multiple papers efficiently.
        
        Args:
            papers: List of papers to score
            
        Returns:
            List[QualityScoreComponents]: List of quality scores
        """
        logger.info(f"Scoring batch of {len(papers)} papers")
        
        scores = []
        for paper in papers:
            try:
                score = self.score_paper(paper)
                scores.append(score)
            except Exception as e:
                logger.error(f"Error scoring paper {paper.pmid}: {str(e)}")
                # Create default score for failed papers
                default_score = QualityScoreComponents()
                scores.append(default_score)
        
        logger.info(f"Successfully scored {len(scores)} papers")
        return scores
    
    def get_quality_distribution(self, scores: List[QualityScoreComponents]) -> Dict[str, any]:
        """Analyze quality score distribution.
        
        Args:
            scores: List of quality scores
            
        Returns:
            Dict[str, any]: Distribution statistics
        """
        if not scores:
            return {}
        
        total_scores = [s.total_score for s in scores]
        journal_scores = [s.journal_score for s in scores]
        clinical_scores = [s.clinical_relevance_score for s in scores]
        ml_scores = [s.ml_methodology_score for s in scores]
        recency_scores = [s.recency_score for s in scores]
        
        def calc_stats(values):
            return {
                'mean': sum(values) / len(values),
                'min': min(values),
                'max': max(values),
                'median': sorted(values)[len(values) // 2]
            }
        
        # Quality tiers
        high_quality = sum(1 for s in total_scores if s >= 75)
        medium_quality = sum(1 for s in total_scores if 50 <= s < 75)
        low_quality = sum(1 for s in total_scores if s < 50)
        
        return {
            'total_papers': len(scores),
            'total_score_stats': calc_stats(total_scores),
            'component_stats': {
                'journal': calc_stats(journal_scores),
                'clinical_relevance': calc_stats(clinical_scores),
                'ml_methodology': calc_stats(ml_scores),
                'recency': calc_stats(recency_scores)
            },
            'quality_distribution': {
                'high_quality': high_quality,
                'medium_quality': medium_quality,
                'low_quality': low_quality,
                'high_quality_percentage': (high_quality / len(scores)) * 100
            },
            'average_confidence': sum(s.confidence_level for s in scores) / len(scores)
        }
    
    def get_top_papers(self, scores: List[Tuple[PaperMetadata, QualityScoreComponents]], 
                      limit: int = 10) -> List[Tuple[PaperMetadata, QualityScoreComponents]]:
        """Get top-scoring papers.
        
        Args:
            scores: List of (paper, score) tuples
            limit: Number of top papers to return
            
        Returns:
            List[Tuple[PaperMetadata, QualityScoreComponents]]: Top papers
        """
        # Sort by total score (descending)
        sorted_papers = sorted(scores, key=lambda x: x[1].total_score, reverse=True)
        return sorted_papers[:limit]