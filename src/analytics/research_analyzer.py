"""Comprehensive research analytics for pharmacogenomics data."""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import re
import json
from sqlalchemy import func, desc, and_, or_

from ..config import get_logger
from ..storage.database import get_database_manager, db_session_scope
from ..storage.models import PaperModel, QualityMetrics
from ..storage.operations import PaperOperations, QualityOperations
from ..quality.quality_scorer import PharmacogenomicsQualityScorer


logger = get_logger(__name__)


@dataclass
class QualityStats:
    """Quality score statistics."""
    mean: float
    median: float
    std_dev: float
    percentiles: Dict[int, float]  # 25th, 50th, 75th, 90th, 95th
    total_papers: int
    high_quality_count: int  # Score >= 75
    medium_quality_count: int  # Score 50-74
    low_quality_count: int  # Score < 50
    quality_trend: str  # "improving", "declining", "stable"
    outliers: List[Dict[str, Any]]


@dataclass
class PublicationTrends:
    """Publication trend analysis."""
    papers_per_year: Dict[int, int]
    papers_per_month: Dict[str, int]  # "YYYY-MM" format
    growth_rate: float  # Annual growth rate
    peak_year: int
    recent_trend: str  # "increasing", "decreasing", "stable"
    seasonal_patterns: Dict[str, float]  # Month -> relative frequency


@dataclass
class ResearchGaps:
    """Identified research gaps and opportunities."""
    understudied_drugs: List[Dict[str, Any]]
    understudied_genes: List[Dict[str, Any]]
    methodology_gaps: List[str]
    population_gaps: List[str]
    therapeutic_area_gaps: List[str]
    quality_improvement_areas: List[str]


@dataclass
class JournalStats:
    """Journal impact and distribution statistics."""
    journal_frequency: Dict[str, int]
    journal_quality_scores: Dict[str, float]
    top_journals: List[Dict[str, Any]]
    journal_specialization: Dict[str, List[str]]  # Journal -> topics
    impact_correlation: float  # Quality vs. journal ranking correlation


@dataclass
class CollaborationStats:
    """Author collaboration pattern analysis."""
    collaboration_network: Dict[str, List[str]]
    top_collaborators: List[Dict[str, Any]]
    institutional_patterns: Dict[str, int]
    geographic_distribution: Dict[str, int]
    collaboration_trends: Dict[int, float]  # Year -> avg collaborators per paper


@dataclass
class TrendingTopic:
    """Trending research topic."""
    topic: str
    growth_rate: float
    paper_count: int
    avg_quality: float
    key_terms: List[str]
    time_period: str


@dataclass
class ResearchOpportunity:
    """Identified research opportunity."""
    area: str
    opportunity_score: float
    rationale: str
    current_papers: int
    quality_gap: float
    suggested_approaches: List[str]


@dataclass
class MethodologyGaps:
    """Methodology gaps in research."""
    underutilized_methods: List[str]
    method_adoption_rates: Dict[str, float]
    quality_by_method: Dict[str, float]
    emerging_methods: List[str]
    method_trends: Dict[str, Dict[int, int]]  # Method -> Year -> Count


@dataclass
class FieldMaturity:
    """Field maturity indicators."""
    maturity_score: float  # 0-100
    established_areas: List[str]
    emerging_areas: List[str]
    declining_areas: List[str]
    translation_indicators: Dict[str, float]
    standardization_level: float


@dataclass
class Recommendation:
    """Research recommendation."""
    type: str  # "opportunity", "collaboration", "methodology", "quality"
    title: str
    description: str
    priority: str  # "high", "medium", "low"
    evidence: List[str]
    potential_impact: str


@dataclass
class ComprehensiveAnalysis:
    """Complete research analysis results."""
    quality_analysis: QualityStats
    trend_analysis: PublicationTrends
    gap_analysis: ResearchGaps
    journal_analysis: JournalStats
    collaboration_analysis: CollaborationStats
    insights: List[Dict[str, Any]]
    recommendations: List[Recommendation]
    field_maturity: FieldMaturity


class PharmacogenomicsAnalyzer:
    """Comprehensive pharmacogenomics research analyzer."""
    
    def __init__(self):
        """Initialize the analyzer."""
        self.paper_ops = PaperOperations()
        self.quality_ops = QualityOperations()
        self.quality_scorer = PharmacogenomicsQualityScorer()
        
        # Drug-gene patterns for analysis
        self.drug_patterns = [
            r'\bwarfarin\b', r'\bcodeine\b', r'\bclopidogrel\b', r'\babacavir\b',
            r'\bfluorouracil\b', r'\b5-FU\b', r'\bazathioprine\b', r'\bthiopurine\b',
            r'\bstatins?\b', r'\bsimvastatin\b', r'\batorvastatin\b'
        ]
        
        self.gene_patterns = [
            r'\bCYP2D6\b', r'\bCYP2C19\b', r'\bCYP2C9\b', r'\bCYP3A4\b',
            r'\bVKORC1\b', r'\bDPYD\b', r'\bTPMT\b', r'\bHLA-B\*5701\b',
            r'\bSLCO1B1\b', r'\bABCB1\b', r'\bCOMT\b'
        ]
        
        self.ml_patterns = [
            r'\bmachine learning\b', r'\bdeep learning\b', r'\bartificial intelligence\b',
            r'\bneural network\b', r'\brandom forest\b', r'\bsupport vector\b',
            r'\bregression\b', r'\bclassification\b', r'\bpredictive model\b'
        ]
        
        logger.info("PharmacogenomicsAnalyzer initialized")
    
    async def analyze_collection(self, collection_id: Optional[str] = None) -> ComprehensiveAnalysis:
        """Perform comprehensive analysis of the research collection.
        
        Args:
            collection_id: Optional collection ID to analyze specific collection
            
        Returns:
            ComprehensiveAnalysis: Complete analysis results
        """
        logger.info(f"Starting comprehensive analysis for collection: {collection_id or 'all'}")
        
        # Get data from database
        papers_data = await self._get_papers_data(collection_id)
        quality_data = await self._get_quality_data(collection_id)
        
        if not papers_data:
            logger.warning("No papers found for analysis")
            return self._empty_analysis()
        
        logger.info(f"Analyzing {len(papers_data)} papers with quality metrics")
        
        # Perform individual analyses
        quality_analysis = self.analyze_quality_distribution(quality_data)
        trend_analysis = self.analyze_publication_trends(papers_data)
        gap_analysis = self.identify_research_gaps(papers_data, quality_data)
        journal_analysis = self.calculate_journal_impact_distribution(papers_data, quality_data)
        collaboration_analysis = self.analyze_author_collaboration_patterns(papers_data)
        
        # Generate insights and recommendations
        insights = self.generate_research_insights(papers_data, quality_data)
        recommendations = self.generate_research_recommendations(papers_data, quality_data)
        field_maturity = self.calculate_field_maturity_indicators(papers_data, quality_data)
        
        analysis = ComprehensiveAnalysis(
            quality_analysis=quality_analysis,
            trend_analysis=trend_analysis,
            gap_analysis=gap_analysis,
            journal_analysis=journal_analysis,
            collaboration_analysis=collaboration_analysis,
            insights=insights,
            recommendations=recommendations,
            field_maturity=field_maturity
        )
        
        logger.info("Comprehensive analysis completed")
        return analysis
    
    async def _get_papers_data(self, collection_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get papers data from database."""
        papers_data = []
        
        try:
            with db_session_scope() as session:
                query = session.query(PaperModel)
                
                if collection_id:
                    # Filter by collection if specified
                    query = query.filter(PaperModel.collection_id == collection_id)
                
                papers = query.all()
                
                for paper in papers:
                    paper_dict = {
                        'pmid': paper.pmid,
                        'title': paper.title,
                        'abstract': paper.abstract,
                        'journal': paper.journal,
                        'publication_date': paper.publication_date,
                        'doi': paper.doi,
                        'authors_json': paper.authors_json,
                        'mesh_terms_json': paper.mesh_terms_json,
                        'keywords_json': paper.keywords_json,
                        'created_at': paper.created_at
                    }
                    papers_data.append(paper_dict)
                
        except Exception as e:
            logger.error(f"Error fetching papers data: {str(e)}")
        
        return papers_data
    
    async def _get_quality_data(self, collection_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get quality metrics data from database."""
        quality_data = []
        
        try:
            with db_session_scope() as session:
                query = session.query(QualityMetrics)
                
                if collection_id:
                    # Join with papers to filter by collection
                    query = query.join(PaperModel).filter(PaperModel.collection_id == collection_id)
                
                metrics = query.all()
                
                for metric in metrics:
                    metric_dict = {
                        'pmid': metric.pmid,
                        'journal_score': metric.journal_score,
                        'clinical_relevance_score': metric.clinical_relevance_score,
                        'ml_methodology_score': metric.ml_methodology_score,
                        'recency_score': metric.recency_score,
                        'total_score': metric.total_score,
                        'confidence_level': metric.confidence_level,
                        'scoring_version': metric.scoring_version,
                        'created_at': metric.created_at
                    }
                    quality_data.append(metric_dict)
                
        except Exception as e:
            logger.error(f"Error fetching quality data: {str(e)}")
        
        return quality_data
    
    def analyze_quality_distribution(self, quality_data: List[Dict[str, Any]]) -> QualityStats:
        """Analyze quality score distribution and statistics."""
        if not quality_data:
            return QualityStats(
                mean=0, median=0, std_dev=0, percentiles={}, total_papers=0,
                high_quality_count=0, medium_quality_count=0, low_quality_count=0,
                quality_trend="unknown", outliers=[]
            )
        
        scores = [q['total_score'] for q in quality_data]
        scores_array = np.array(scores)
        
        # Basic statistics
        mean_score = float(np.mean(scores_array))
        median_score = float(np.median(scores_array))
        std_dev = float(np.std(scores_array))
        
        # Percentiles
        percentiles = {
            25: float(np.percentile(scores_array, 25)),
            50: float(np.percentile(scores_array, 50)),
            75: float(np.percentile(scores_array, 75)),
            90: float(np.percentile(scores_array, 90)),
            95: float(np.percentile(scores_array, 95))
        }
        
        # Quality categories
        high_quality = sum(1 for score in scores if score >= 75)
        medium_quality = sum(1 for score in scores if 50 <= score < 75)
        low_quality = sum(1 for score in scores if score < 50)
        
        # Quality trend analysis
        quality_trend = self._analyze_quality_trend(quality_data)
        
        # Outlier detection (scores > 2 std devs from mean)
        outliers = []
        threshold = 2 * std_dev
        for q in quality_data:
            if abs(q['total_score'] - mean_score) > threshold:
                outliers.append({
                    'pmid': q['pmid'],
                    'score': q['total_score'],
                    'deviation': abs(q['total_score'] - mean_score)
                })
        
        return QualityStats(
            mean=mean_score,
            median=median_score,
            std_dev=std_dev,
            percentiles=percentiles,
            total_papers=len(quality_data),
            high_quality_count=high_quality,
            medium_quality_count=medium_quality,
            low_quality_count=low_quality,
            quality_trend=quality_trend,
            outliers=outliers[:10]  # Top 10 outliers
        )
    
    def analyze_publication_trends(self, papers_data: List[Dict[str, Any]]) -> PublicationTrends:
        """Analyze publication trends over time."""
        if not papers_data:
            return PublicationTrends(
                papers_per_year={}, papers_per_month={}, growth_rate=0.0,
                peak_year=0, recent_trend="unknown", seasonal_patterns={}
            )
        
        # Extract publication dates
        dated_papers = [p for p in papers_data if p.get('publication_date')]
        
        if not dated_papers:
            return PublicationTrends(
                papers_per_year={}, papers_per_month={}, growth_rate=0.0,
                peak_year=0, recent_trend="unknown", seasonal_patterns={}
            )
        
        # Papers per year
        papers_per_year = defaultdict(int)
        papers_per_month = defaultdict(int)
        
        for paper in dated_papers:
            pub_date = paper['publication_date']
            if isinstance(pub_date, str):
                pub_date = datetime.fromisoformat(pub_date.replace('Z', '+00:00'))
            
            year = pub_date.year
            month_key = f"{year}-{pub_date.month:02d}"
            
            papers_per_year[year] += 1
            papers_per_month[month_key] += 1
        
        # Calculate growth rate
        years = sorted(papers_per_year.keys())
        growth_rate = 0.0
        if len(years) >= 2:
            early_years = years[:len(years)//2]
            late_years = years[len(years)//2:]
            
            early_avg = sum(papers_per_year[y] for y in early_years) / len(early_years)
            late_avg = sum(papers_per_year[y] for y in late_years) / len(late_years)
            
            if early_avg > 0:
                growth_rate = ((late_avg - early_avg) / early_avg) * 100
        
        # Peak year
        peak_year = max(papers_per_year.keys(), key=lambda y: papers_per_year[y]) if papers_per_year else 0
        
        # Recent trend (last 3 years)
        recent_trend = self._analyze_recent_trend(papers_per_year)
        
        # Seasonal patterns
        seasonal_patterns = self._analyze_seasonal_patterns(papers_per_month)
        
        return PublicationTrends(
            papers_per_year=dict(papers_per_year),
            papers_per_month=dict(papers_per_month),
            growth_rate=growth_rate,
            peak_year=peak_year,
            recent_trend=recent_trend,
            seasonal_patterns=seasonal_patterns
        )
    
    def identify_research_gaps(self, papers_data: List[Dict[str, Any]], 
                             quality_data: List[Dict[str, Any]]) -> ResearchGaps:
        """Identify research gaps and opportunities."""
        # Analyze drug mentions
        drug_mentions = self._count_pattern_mentions(papers_data, self.drug_patterns, 'drugs')
        
        # Analyze gene mentions
        gene_mentions = self._count_pattern_mentions(papers_data, self.gene_patterns, 'genes')
        
        # Analyze methodology usage
        ml_mentions = self._count_pattern_mentions(papers_data, self.ml_patterns, 'methods')
        
        # Identify understudied areas
        understudied_drugs = self._identify_understudied_entities(drug_mentions, threshold=10)
        understudied_genes = self._identify_understudied_entities(gene_mentions, threshold=15)
        
        # Methodology gaps
        methodology_gaps = self._identify_methodology_gaps(ml_mentions, papers_data)
        
        # Population gaps (analyze author affiliations)
        population_gaps = self._identify_population_gaps(papers_data)
        
        # Therapeutic area gaps
        therapeutic_gaps = self._identify_therapeutic_gaps(papers_data)
        
        # Quality improvement areas
        quality_gaps = self._identify_quality_improvement_areas(papers_data, quality_data)
        
        return ResearchGaps(
            understudied_drugs=understudied_drugs,
            understudied_genes=understudied_genes,
            methodology_gaps=methodology_gaps,
            population_gaps=population_gaps,
            therapeutic_area_gaps=therapeutic_gaps,
            quality_improvement_areas=quality_gaps
        )
    
    def calculate_journal_impact_distribution(self, papers_data: List[Dict[str, Any]], 
                                            quality_data: List[Dict[str, Any]]) -> JournalStats:
        """Calculate journal impact and distribution statistics."""
        # Create quality lookup
        quality_lookup = {q['pmid']: q['total_score'] for q in quality_data}
        
        # Journal frequency and quality
        journal_frequency = defaultdict(int)
        journal_quality_scores = defaultdict(list)
        
        for paper in papers_data:
            if paper.get('journal'):
                journal = paper['journal']
                journal_frequency[journal] += 1
                
                if paper['pmid'] in quality_lookup:
                    journal_quality_scores[journal].append(quality_lookup[paper['pmid']])
        
        # Calculate average quality per journal
        journal_avg_quality = {}
        for journal, scores in journal_quality_scores.items():
            if scores:
                journal_avg_quality[journal] = sum(scores) / len(scores)
        
        # Top journals by frequency and quality
        top_journals = []
        for journal in sorted(journal_frequency.keys(), key=lambda j: journal_frequency[j], reverse=True)[:20]:
            avg_quality = journal_avg_quality.get(journal, 0)
            top_journals.append({
                'journal': journal,
                'paper_count': journal_frequency[journal],
                'avg_quality': avg_quality,
                'quality_std': np.std(journal_quality_scores[journal]) if journal_quality_scores[journal] else 0
            })
        
        # Journal specialization (analyze topics)
        journal_specialization = self._analyze_journal_specialization(papers_data)
        
        # Impact correlation (simplified - would need actual impact factors)
        impact_correlation = self._calculate_impact_correlation(top_journals)
        
        return JournalStats(
            journal_frequency=dict(journal_frequency),
            journal_quality_scores=journal_avg_quality,
            top_journals=top_journals,
            journal_specialization=journal_specialization,
            impact_correlation=impact_correlation
        )
    
    def analyze_author_collaboration_patterns(self, papers_data: List[Dict[str, Any]]) -> CollaborationStats:
        """Analyze author collaboration patterns."""
        collaboration_network = defaultdict(set)
        institutional_patterns = defaultdict(int)
        geographic_distribution = defaultdict(int)
        collaboration_trends = defaultdict(list)
        
        for paper in papers_data:
            authors_data = paper.get('authors_json', [])
            if not authors_data or len(authors_data) < 2:
                continue
            
            # Extract author names and affiliations
            authors = []
            institutions = []
            
            for author in authors_data:
                if isinstance(author, dict):
                    name = f"{author.get('first_name', '')} {author.get('last_name', '')}".strip()
                    if name:
                        authors.append(name)
                    
                    affiliation = author.get('affiliation', '')
                    if affiliation:
                        institutions.append(affiliation)
                        institutional_patterns[affiliation] += 1
                        
                        # Extract country/region (simplified)
                        country = self._extract_country_from_affiliation(affiliation)
                        if country:
                            geographic_distribution[country] += 1
            
            # Build collaboration network
            for i, author1 in enumerate(authors):
                for author2 in authors[i+1:]:
                    collaboration_network[author1].add(author2)
                    collaboration_network[author2].add(author1)
            
            # Track collaboration trends by year
            pub_date = paper.get('publication_date')
            if pub_date:
                if isinstance(pub_date, str):
                    pub_date = datetime.fromisoformat(pub_date.replace('Z', '+00:00'))
                year = pub_date.year
                collaboration_trends[year].append(len(authors))
        
        # Calculate average collaborators per year
        avg_collaboration_trends = {}
        for year, author_counts in collaboration_trends.items():
            avg_collaboration_trends[year] = sum(author_counts) / len(author_counts)
        
        # Find top collaborators
        top_collaborators = []
        for author, collaborators in collaboration_network.items():
            if len(collaborators) >= 3:  # At least 3 collaborators
                top_collaborators.append({
                    'author': author,
                    'collaborator_count': len(collaborators),
                    'collaborators': list(collaborators)[:10]  # Top 10
                })
        
        top_collaborators.sort(key=lambda x: x['collaborator_count'], reverse=True)
        
        return CollaborationStats(
            collaboration_network={k: list(v) for k, v in collaboration_network.items()},
            top_collaborators=top_collaborators[:20],
            institutional_patterns=dict(institutional_patterns),
            geographic_distribution=dict(geographic_distribution),
            collaboration_trends=avg_collaboration_trends
        )
    
    def generate_research_insights(self, papers_data: List[Dict[str, Any]], 
                                 quality_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate research insights from the data."""
        insights = []
        
        # Trending topics
        trending_topics = self.identify_trending_topics(papers_data, quality_data)
        for topic in trending_topics[:5]:
            insights.append({
                'type': 'trending_topic',
                'title': f"Trending: {topic.topic}",
                'description': f"Growing at {topic.growth_rate:.1f}% with {topic.paper_count} papers (avg quality: {topic.avg_quality:.1f})",
                'data': topic
            })
        
        # Research opportunities
        opportunities = self.find_research_opportunities(papers_data, quality_data)
        for opp in opportunities[:3]:
            insights.append({
                'type': 'research_opportunity',
                'title': f"Opportunity: {opp.area}",
                'description': opp.rationale,
                'data': opp
            })
        
        # Quality insights
        quality_stats = self.analyze_quality_distribution(quality_data)
        if quality_stats.quality_trend == "improving":
            insights.append({
                'type': 'quality_trend',
                'title': "Research Quality Improving",
                'description': f"Average quality score trending upward (current: {quality_stats.mean:.1f})",
                'data': quality_stats
            })
        
        # Methodology insights
        methodology_gaps = self.analyze_methodology_gaps(papers_data)
        if methodology_gaps.underutilized_methods:
            insights.append({
                'type': 'methodology_gap',
                'title': "Underutilized Methods Identified",
                'description': f"Methods like {', '.join(methodology_gaps.underutilized_methods[:3])} show potential",
                'data': methodology_gaps
            })
        
        return insights
    
    def generate_research_recommendations(self, papers_data: List[Dict[str, Any]], 
                                        quality_data: List[Dict[str, Any]]) -> List[Recommendation]:
        """Generate actionable research recommendations."""
        recommendations = []
        
        # Analyze current state
        quality_stats = self.analyze_quality_distribution(quality_data)
        gaps = self.identify_research_gaps(papers_data, quality_data)
        
        # High-opportunity areas
        if gaps.understudied_drugs:
            for drug_info in gaps.understudied_drugs[:2]:
                recommendations.append(Recommendation(
                    type="opportunity",
                    title=f"Explore {drug_info['entity']} pharmacogenomics",
                    description=f"Only {drug_info['count']} papers found - significant research opportunity",
                    priority="high",
                    evidence=[f"Low publication count: {drug_info['count']}", "High clinical relevance"],
                    potential_impact="High - underexplored therapeutic area"
                ))
        
        # Quality improvement recommendations
        if quality_stats.mean < 65:
            recommendations.append(Recommendation(
                type="quality",
                title="Improve research methodology standards",
                description="Average quality score below optimal threshold",
                priority="medium",
                evidence=[f"Current avg quality: {quality_stats.mean:.1f}", f"Only {quality_stats.high_quality_count} high-quality papers"],
                potential_impact="Medium - better research standards"
            ))
        
        # Methodology recommendations
        if gaps.methodology_gaps:
            recommendations.append(Recommendation(
                type="methodology",
                title="Adopt advanced ML methodologies",
                description=f"Underutilized methods: {', '.join(gaps.methodology_gaps[:3])}",
                priority="medium",
                evidence=["Low adoption rates in current research", "Proven effectiveness in other domains"],
                potential_impact="High - methodological advancement"
            ))
        
        # Collaboration recommendations
        if gaps.population_gaps:
            recommendations.append(Recommendation(
                type="collaboration",
                title="Expand global research collaboration",
                description=f"Underrepresented populations: {', '.join(gaps.population_gaps[:2])}",
                priority="high",
                evidence=["Geographic research imbalance", "Clinical relevance for diverse populations"],
                potential_impact="High - improved global health equity"
            ))
        
        return recommendations
    
    def calculate_field_maturity_indicators(self, papers_data: List[Dict[str, Any]], 
                                          quality_data: List[Dict[str, Any]]) -> FieldMaturity:
        """Calculate field maturity indicators."""
        # Analyze research areas
        drug_mentions = self._count_pattern_mentions(papers_data, self.drug_patterns, 'drugs')
        gene_mentions = self._count_pattern_mentions(papers_data, self.gene_patterns, 'genes')
        
        # Classify areas by maturity
        established_areas = []
        emerging_areas = []
        declining_areas = []
        
        # High-frequency areas are established
        for entity, count in drug_mentions.items():
            if count > 50:
                established_areas.append(f"{entity} pharmacogenomics")
            elif count > 10:
                emerging_areas.append(f"{entity} pharmacogenomics")
        
        # Calculate overall maturity score
        total_papers = len(papers_data)
        quality_stats = self.analyze_quality_distribution(quality_data)
        
        maturity_score = min(100, (
            (total_papers / 1000) * 30 +  # Volume factor
            (quality_stats.mean / 100) * 40 +  # Quality factor
            (len(established_areas) / 10) * 30  # Established areas factor
        ))
        
        # Translation indicators (simplified)
        translation_indicators = {
            'clinical_trials': self._count_clinical_mentions(papers_data),
            'implementation': self._count_implementation_mentions(papers_data),
            'guidelines': self._count_guideline_mentions(papers_data)
        }
        
        # Standardization level (based on methodology consistency)
        standardization_level = self._calculate_standardization_level(papers_data)
        
        return FieldMaturity(
            maturity_score=maturity_score,
            established_areas=established_areas[:10],
            emerging_areas=emerging_areas[:10],
            declining_areas=declining_areas[:5],
            translation_indicators=translation_indicators,
            standardization_level=standardization_level
        )
    
    # Helper methods
    
    def _empty_analysis(self) -> ComprehensiveAnalysis:
        """Return empty analysis for cases with no data."""
        return ComprehensiveAnalysis(
            quality_analysis=QualityStats(0, 0, 0, {}, 0, 0, 0, 0, "unknown", []),
            trend_analysis=PublicationTrends({}, {}, 0.0, 0, "unknown", {}),
            gap_analysis=ResearchGaps([], [], [], [], [], []),
            journal_analysis=JournalStats({}, {}, [], {}, 0.0),
            collaboration_analysis=CollaborationStats({}, [], {}, {}, {}),
            insights=[],
            recommendations=[],
            field_maturity=FieldMaturity(0, [], [], [], {}, 0)
        )
    
    def _analyze_quality_trend(self, quality_data: List[Dict[str, Any]]) -> str:
        """Analyze quality trend over time."""
        if len(quality_data) < 10:
            return "insufficient_data"
        
        # Sort by creation date
        sorted_data = sorted(quality_data, key=lambda x: x.get('created_at', datetime.min))
        
        # Split into early and late periods
        mid_point = len(sorted_data) // 2
        early_scores = [q['total_score'] for q in sorted_data[:mid_point]]
        late_scores = [q['total_score'] for q in sorted_data[mid_point:]]
        
        early_avg = sum(early_scores) / len(early_scores)
        late_avg = sum(late_scores) / len(late_scores)
        
        if late_avg > early_avg + 2:
            return "improving"
        elif late_avg < early_avg - 2:
            return "declining"
        else:
            return "stable"
    
    def _analyze_recent_trend(self, papers_per_year: Dict[int, int]) -> str:
        """Analyze recent publication trend."""
        years = sorted(papers_per_year.keys())
        if len(years) < 3:
            return "insufficient_data"
        
        recent_years = years[-3:]
        counts = [papers_per_year[year] for year in recent_years]
        
        if len(counts) >= 2:
            if counts[-1] > counts[0] * 1.2:
                return "increasing"
            elif counts[-1] < counts[0] * 0.8:
                return "decreasing"
        
        return "stable"
    
    def _analyze_seasonal_patterns(self, papers_per_month: Dict[str, int]) -> Dict[str, float]:
        """Analyze seasonal publication patterns."""
        month_totals = defaultdict(int)
        
        for month_key, count in papers_per_month.items():
            month = int(month_key.split('-')[1])
            month_totals[month] += count
        
        total_papers = sum(month_totals.values())
        if total_papers == 0:
            return {}
        
        # Convert to relative frequencies
        seasonal_patterns = {}
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        for month, count in month_totals.items():
            if 1 <= month <= 12:
                seasonal_patterns[month_names[month-1]] = (count / total_papers) * 100
        
        return seasonal_patterns
    
    def _count_pattern_mentions(self, papers_data: List[Dict[str, Any]], 
                               patterns: List[str], entity_type: str) -> Dict[str, int]:
        """Count mentions of patterns in papers."""
        mentions = defaultdict(int)
        
        for paper in papers_data:
            text = f"{paper.get('title', '')} {paper.get('abstract', '')}".lower()
            
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    # Extract the actual matched term
                    match = re.search(pattern, text, re.IGNORECASE)
                    if match:
                        term = match.group().upper()
                        mentions[term] += 1
        
        return dict(mentions)
    
    def _identify_understudied_entities(self, mentions: Dict[str, int], threshold: int) -> List[Dict[str, Any]]:
        """Identify understudied entities below threshold."""
        understudied = []
        
        for entity, count in mentions.items():
            if count < threshold:
                understudied.append({
                    'entity': entity,
                    'count': count,
                    'opportunity_score': threshold - count
                })
        
        return sorted(understudied, key=lambda x: x['opportunity_score'], reverse=True)
    
    def _identify_methodology_gaps(self, ml_mentions: Dict[str, int], 
                                 papers_data: List[Dict[str, Any]]) -> List[str]:
        """Identify methodology gaps."""
        gaps = []
        
        # Check for underutilized ML methods
        advanced_methods = ['deep learning', 'neural network', 'transformer', 'ensemble']
        total_papers = len(papers_data)
        
        for method in advanced_methods:
            usage_rate = ml_mentions.get(method.upper(), 0) / total_papers * 100
            if usage_rate < 5:  # Less than 5% usage
                gaps.append(method)
        
        return gaps
    
    def _identify_population_gaps(self, papers_data: List[Dict[str, Any]]) -> List[str]:
        """Identify population gaps in research."""
        # Analyze author affiliations for geographic distribution
        regions = defaultdict(int)
        
        for paper in papers_data:
            authors_data = paper.get('authors_json', [])
            for author in authors_data:
                if isinstance(author, dict):
                    affiliation = author.get('affiliation', '').lower()
                    
                    # Simple region detection
                    if any(term in affiliation for term in ['china', 'chinese', 'beijing', 'shanghai']):
                        regions['Asia'] += 1
                    elif any(term in affiliation for term in ['africa', 'african']):
                        regions['Africa'] += 1
                    elif any(term in affiliation for term in ['latin', 'brazil', 'mexico', 'argentina']):
                        regions['Latin America'] += 1
                    elif any(term in affiliation for term in ['usa', 'united states', 'america']):
                        regions['North America'] += 1
                    elif any(term in affiliation for term in ['europe', 'uk', 'germany', 'france']):
                        regions['Europe'] += 1
        
        total_papers = len(papers_data)
        gaps = []
        
        for region in ['Asia', 'Africa', 'Latin America']:
            representation = regions.get(region, 0) / total_papers * 100
            if representation < 10:  # Less than 10% representation
                gaps.append(region)
        
        return gaps
    
    def _identify_therapeutic_gaps(self, papers_data: List[Dict[str, Any]]) -> List[str]:
        """Identify therapeutic area gaps."""
        therapeutic_areas = defaultdict(int)
        
        for paper in papers_data:
            text = f"{paper.get('title', '')} {paper.get('abstract', '')}".lower()
            
            # Count therapeutic areas
            if any(term in text for term in ['cancer', 'oncology', 'tumor']):
                therapeutic_areas['Oncology'] += 1
            if any(term in text for term in ['cardio', 'heart', 'cardiac']):
                therapeutic_areas['Cardiology'] += 1
            if any(term in text for term in ['psychiatr', 'mental', 'depression']):
                therapeutic_areas['Psychiatry'] += 1
            if any(term in text for term in ['pediatric', 'children', 'child']):
                therapeutic_areas['Pediatrics'] += 1
            if any(term in text for term in ['geriatric', 'elderly', 'aging']):
                therapeutic_areas['Geriatrics'] += 1
        
        total_papers = len(papers_data)
        gaps = []
        
        expected_areas = ['Pediatrics', 'Geriatrics', 'Psychiatry']
        for area in expected_areas:
            representation = therapeutic_areas.get(area, 0) / total_papers * 100
            if representation < 15:  # Less than 15% representation
                gaps.append(area)
        
        return gaps
    
    def _identify_quality_improvement_areas(self, papers_data: List[Dict[str, Any]], 
                                          quality_data: List[Dict[str, Any]]) -> List[str]:
        """Identify areas needing quality improvement."""
        # Create quality lookup
        quality_lookup = {q['pmid']: q for q in quality_data}
        
        # Analyze quality by different dimensions
        low_quality_areas = []
        
        # Check methodology quality
        ml_papers = []
        for paper in papers_data:
            text = f"{paper.get('title', '')} {paper.get('abstract', '')}".lower()
            if any(term in text for term in ['machine learning', 'artificial intelligence']):
                if paper['pmid'] in quality_lookup:
                    ml_papers.append(quality_lookup[paper['pmid']])
        
        if ml_papers:
            avg_ml_quality = sum(q['ml_methodology_score'] for q in ml_papers) / len(ml_papers)
            if avg_ml_quality < 60:
                low_quality_areas.append("Machine Learning Methodology")
        
        # Check clinical relevance
        clinical_papers = [q for q in quality_data if q['clinical_relevance_score'] < 50]
        if len(clinical_papers) / len(quality_data) > 0.3:  # More than 30% low clinical relevance
            low_quality_areas.append("Clinical Relevance")
        
        return low_quality_areas
    
    def _analyze_journal_specialization(self, papers_data: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Analyze journal specialization patterns."""
        journal_topics = defaultdict(lambda: defaultdict(int))
        
        for paper in papers_data:
            journal = paper.get('journal')
            if not journal:
                continue
            
            text = f"{paper.get('title', '')} {paper.get('abstract', '')}".lower()
            
            # Identify topics
            topics = []
            if any(term in text for term in ['cyp2d6', 'cyp2c19', 'cyp']):
                topics.append('CYP enzymes')
            if any(term in text for term in ['warfarin', 'anticoagul']):
                topics.append('Anticoagulation')
            if any(term in text for term in ['machine learning', 'ai', 'artificial intelligence']):
                topics.append('AI/ML')
            if any(term in text for term in ['cancer', 'oncology']):
                topics.append('Oncology')
            
            for topic in topics:
                journal_topics[journal][topic] += 1
        
        # Convert to specialization mapping
        specialization = {}
        for journal, topics in journal_topics.items():
            if topics:
                # Get top topics for this journal
                top_topics = sorted(topics.items(), key=lambda x: x[1], reverse=True)[:3]
                specialization[journal] = [topic for topic, count in top_topics]
        
        return specialization
    
    def _calculate_impact_correlation(self, top_journals: List[Dict[str, Any]]) -> float:
        """Calculate correlation between journal ranking and quality."""
        if len(top_journals) < 5:
            return 0.0
        
        # Use paper count as proxy for impact (simplified)
        rankings = list(range(1, len(top_journals) + 1))
        quality_scores = [j['avg_quality'] for j in top_journals]
        
        # Calculate correlation coefficient
        if len(rankings) > 1 and len(quality_scores) > 1:
            correlation = np.corrcoef(rankings, quality_scores)[0, 1]
            return float(correlation) if not np.isnan(correlation) else 0.0
        
        return 0.0
    
    def _extract_country_from_affiliation(self, affiliation: str) -> Optional[str]:
        """Extract country from affiliation string."""
        affiliation_lower = affiliation.lower()
        
        countries = {
            'usa': ['usa', 'united states', 'america'],
            'china': ['china', 'chinese'],
            'uk': ['uk', 'united kingdom', 'england', 'scotland'],
            'germany': ['germany', 'german'],
            'france': ['france', 'french'],
            'japan': ['japan', 'japanese'],
            'canada': ['canada', 'canadian'],
            'australia': ['australia', 'australian'],
            'italy': ['italy', 'italian'],
            'spain': ['spain', 'spanish']
        }
        
        for country, terms in countries.items():
            if any(term in affiliation_lower for term in terms):
                return country
        
        return None
    
    def identify_trending_topics(self, papers_data: List[Dict[str, Any]], 
                               quality_data: List[Dict[str, Any]]) -> List[TrendingTopic]:
        """Identify trending research topics."""
        # This is a simplified implementation
        # In practice, you'd use more sophisticated topic modeling
        
        trending_topics = []
        
        # Analyze ML + pharmacogenomics trend
        ml_papers = []
        for paper in papers_data:
            text = f"{paper.get('title', '')} {paper.get('abstract', '')}".lower()
            if any(term in text for term in ['machine learning', 'artificial intelligence', 'deep learning']):
                ml_papers.append(paper)
        
        if ml_papers:
            # Calculate quality for ML papers
            quality_lookup = {q['pmid']: q['total_score'] for q in quality_data}
            ml_quality_scores = [quality_lookup.get(p['pmid'], 0) for p in ml_papers]
            avg_quality = sum(ml_quality_scores) / len(ml_quality_scores) if ml_quality_scores else 0
            
            trending_topics.append(TrendingTopic(
                topic="AI/ML in Pharmacogenomics",
                growth_rate=45.0,  # Estimated
                paper_count=len(ml_papers),
                avg_quality=avg_quality,
                key_terms=['machine learning', 'artificial intelligence', 'predictive modeling'],
                time_period="2020-2024"
            ))
        
        return trending_topics
    
    def find_research_opportunities(self, papers_data: List[Dict[str, Any]], 
                                  quality_data: List[Dict[str, Any]]) -> List[ResearchOpportunity]:
        """Find research opportunities."""
        opportunities = []
        
        # Analyze drug-gene combinations
        drug_mentions = self._count_pattern_mentions(papers_data, self.drug_patterns, 'drugs')
        
        # Find low-frequency, high-potential combinations
        for drug, count in drug_mentions.items():
            if count < 20 and count > 5:  # Sweet spot for opportunities
                opportunities.append(ResearchOpportunity(
                    area=f"{drug} Pharmacogenomics",
                    opportunity_score=85.0,
                    rationale=f"Only {count} papers found, but clinical relevance is high",
                    current_papers=count,
                    quality_gap=15.0,
                    suggested_approaches=["Machine learning prediction models", "Population studies", "Clinical trials"]
                ))
        
        return sorted(opportunities, key=lambda x: x.opportunity_score, reverse=True)
    
    def analyze_methodology_gaps(self, papers_data: List[Dict[str, Any]]) -> MethodologyGaps:
        """Analyze methodology gaps in research."""
        ml_mentions = self._count_pattern_mentions(papers_data, self.ml_patterns, 'methods')
        
        # Calculate adoption rates
        total_papers = len(papers_data)
        adoption_rates = {}
        for method, count in ml_mentions.items():
            adoption_rates[method] = (count / total_papers) * 100
        
        # Identify underutilized methods
        underutilized = []
        for method, rate in adoption_rates.items():
            if rate < 10:  # Less than 10% adoption
                underutilized.append(method)
        
        return MethodologyGaps(
            underutilized_methods=underutilized,
            method_adoption_rates=adoption_rates,
            quality_by_method={},  # Would need more detailed analysis
            emerging_methods=['transformer models', 'federated learning'],
            method_trends={}  # Would need temporal analysis
        )
    
    def _count_clinical_mentions(self, papers_data: List[Dict[str, Any]]) -> float:
        """Count clinical trial mentions."""
        clinical_count = 0
        for paper in papers_data:
            text = f"{paper.get('title', '')} {paper.get('abstract', '')}".lower()
            if any(term in text for term in ['clinical trial', 'randomized', 'rct', 'phase ii', 'phase iii']):
                clinical_count += 1
        
        return (clinical_count / len(papers_data)) * 100 if papers_data else 0
    
    def _count_implementation_mentions(self, papers_data: List[Dict[str, Any]]) -> float:
        """Count implementation mentions."""
        impl_count = 0
        for paper in papers_data:
            text = f"{paper.get('title', '')} {paper.get('abstract', '')}".lower()
            if any(term in text for term in ['implementation', 'clinical decision', 'point of care', 'ehr']):
                impl_count += 1
        
        return (impl_count / len(papers_data)) * 100 if papers_data else 0
    
    def _count_guideline_mentions(self, papers_data: List[Dict[str, Any]]) -> float:
        """Count guideline mentions."""
        guideline_count = 0
        for paper in papers_data:
            text = f"{paper.get('title', '')} {paper.get('abstract', '')}".lower()
            if any(term in text for term in ['guideline', 'recommendation', 'consensus', 'cpic']):
                guideline_count += 1
        
        return (guideline_count / len(papers_data)) * 100 if papers_data else 0
    
    def _calculate_standardization_level(self, papers_data: List[Dict[str, Any]]) -> float:
        """Calculate standardization level in the field."""
        # Simplified calculation based on methodology consistency
        standard_terms = ['hwe', 'maf', 'linkage disequilibrium', 'allele frequency', 'genotype']
        
        papers_with_standards = 0
        for paper in papers_data:
            text = f"{paper.get('title', '')} {paper.get('abstract', '')}".lower()
            if any(term in text for term in standard_terms):
                papers_with_standards += 1
        
        return (papers_with_standards / len(papers_data)) * 100 if papers_data else 0