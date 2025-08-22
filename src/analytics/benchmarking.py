"""Benchmarking and comparison system for pharmacogenomics research."""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import json

from ..config import get_logger
from ..storage.database import get_database_manager, db_session_scope
from ..storage.models import PaperModel, QualityMetrics
from .research_analyzer import PharmacogenomicsAnalyzer, QualityStats


logger = get_logger(__name__)


@dataclass
class BenchmarkResults:
    """Benchmark comparison results."""
    collection_performance: Dict[str, float]
    field_benchmarks: Dict[str, float]
    performance_percentile: Dict[str, float]
    strengths: List[str]
    improvement_areas: List[str]
    overall_score: float
    comparison_summary: str


@dataclass
class JournalComparison:
    """Journal performance comparison."""
    journal_rankings: List[Dict[str, Any]]
    quality_benchmarks: Dict[str, float]
    publication_volume: Dict[str, int]
    specialization_scores: Dict[str, float]
    emerging_journals: List[str]
    declining_journals: List[str]


@dataclass
class ProductivityAnalysis:
    """Research group productivity analysis."""
    top_productive_groups: List[Dict[str, Any]]
    productivity_metrics: Dict[str, float]
    collaboration_efficiency: Dict[str, float]
    geographic_productivity: Dict[str, Dict[str, float]]
    temporal_productivity: Dict[int, float]


@dataclass
class ImpactMetrics:
    """Research impact metrics."""
    citation_indicators: Dict[str, float]
    quality_impact_correlation: float
    temporal_impact_trends: Dict[int, float]
    high_impact_characteristics: List[str]
    impact_prediction_factors: Dict[str, float]


@dataclass
class FieldSummary:
    """Comprehensive field summary."""
    field_overview: Dict[str, Any]
    maturity_assessment: Dict[str, float]
    growth_indicators: Dict[str, float]
    quality_trends: Dict[str, float]
    research_landscape: Dict[str, Any]
    future_directions: List[str]


class PharmacogenomicsBenchmarking:
    """Benchmarking and comparison system for pharmacogenomics research."""
    
    def __init__(self):
        """Initialize the benchmarking system."""
        self.analyzer = PharmacogenomicsAnalyzer()
        
        # Field-wide benchmarks (these would be updated from external sources)
        self.field_benchmarks = {
            'avg_quality_score': 68.5,
            'high_quality_percentage': 35.0,
            'papers_per_year_growth': 12.5,
            'ml_adoption_rate': 25.0,
            'clinical_translation_rate': 18.0,
            'international_collaboration_rate': 45.0,
            'avg_authors_per_paper': 6.2,
            'avg_citations_per_paper': 15.8,
            'journal_diversity_index': 0.75
        }
        
        # Journal impact rankings (simplified - would come from external sources)
        self.journal_impact_rankings = {
            'Nature Genetics': 95.0,
            'Clinical Pharmacology & Therapeutics': 88.0,
            'Pharmacogenomics': 82.0,
            'Genome Medicine': 85.0,
            'The Pharmacogenomics Journal': 78.0,
            'Pharmacogenetics and Genomics': 75.0,
            'Human Genetics': 80.0,
            'PLOS Genetics': 77.0,
            'Frontiers in Pharmacology': 70.0,
            'Journal of Personalized Medicine': 68.0
        }
        
        logger.info("PharmacogenomicsBenchmarking initialized")
    
    async def benchmark_against_field_standards(self, collection_id: Optional[str] = None) -> BenchmarkResults:
        """Benchmark collection against field-wide standards.
        
        Args:
            collection_id: Optional collection ID to benchmark specific collection
            
        Returns:
            BenchmarkResults: Comprehensive benchmark comparison
        """
        logger.info(f"Benchmarking collection {collection_id or 'all'} against field standards")
        
        # Get collection analysis
        analysis = await self.analyzer.analyze_collection(collection_id)
        
        # Calculate collection performance metrics
        collection_performance = {
            'avg_quality_score': analysis.quality_analysis.mean,
            'high_quality_percentage': (analysis.quality_analysis.high_quality_count / 
                                      max(1, analysis.quality_analysis.total_papers)) * 100,
            'papers_per_year_growth': self._calculate_growth_rate(analysis.trend_analysis.papers_per_year),
            'journal_diversity': self._calculate_journal_diversity(analysis.journal_analysis.journal_frequency),
            'collaboration_rate': self._calculate_collaboration_rate(analysis.collaboration_analysis),
            'methodology_adoption': self._calculate_methodology_adoption(analysis.gap_analysis)
        }
        
        # Calculate percentiles against field benchmarks
        performance_percentile = {}
        for metric, value in collection_performance.items():
            if metric in self.field_benchmarks:
                percentile = self._calculate_percentile(value, self.field_benchmarks[metric])
                performance_percentile[metric] = percentile
        
        # Identify strengths and improvement areas
        strengths = []
        improvement_areas = []
        
        for metric, percentile in performance_percentile.items():
            if percentile >= 75:
                strengths.append(f"{metric.replace('_', ' ').title()}: {percentile:.1f}th percentile")
            elif percentile <= 25:
                improvement_areas.append(f"{metric.replace('_', ' ').title()}: {percentile:.1f}th percentile")
        
        # Calculate overall score
        overall_score = sum(performance_percentile.values()) / len(performance_percentile)
        
        # Generate comparison summary
        comparison_summary = self._generate_benchmark_summary(
            collection_performance, self.field_benchmarks, overall_score
        )
        
        return BenchmarkResults(
            collection_performance=collection_performance,
            field_benchmarks=self.field_benchmarks,
            performance_percentile=performance_percentile,
            strengths=strengths,
            improvement_areas=improvement_areas,
            overall_score=overall_score,
            comparison_summary=comparison_summary
        )
    
    async def compare_journal_performance(self, collection_id: Optional[str] = None) -> JournalComparison:
        """Compare journal performance within pharmacogenomics domain.
        
        Args:
            collection_id: Optional collection ID for specific comparison
            
        Returns:
            JournalComparison: Journal performance comparison
        """
        logger.info("Comparing journal performance in pharmacogenomics domain")
        
        # Get collection analysis
        analysis = await self.analyzer.analyze_collection(collection_id)
        
        # Create journal rankings
        journal_rankings = []
        for journal_info in analysis.journal_analysis.top_journals:
            journal = journal_info['journal']
            impact_score = self.journal_impact_rankings.get(journal, 50.0)  # Default score
            
            ranking_data = {
                'journal': journal,
                'paper_count': journal_info['paper_count'],
                'avg_quality': journal_info['avg_quality'],
                'impact_score': impact_score,
                'combined_score': (journal_info['avg_quality'] * 0.6 + impact_score * 0.4),
                'specialization': analysis.journal_analysis.journal_specialization.get(journal, [])
            }
            journal_rankings.append(ranking_data)
        
        # Sort by combined score
        journal_rankings.sort(key=lambda x: x['combined_score'], reverse=True)
        
        # Calculate quality benchmarks
        quality_benchmarks = {
            'top_tier': np.percentile([j['avg_quality'] for j in journal_rankings], 90),
            'high_tier': np.percentile([j['avg_quality'] for j in journal_rankings], 75),
            'mid_tier': np.percentile([j['avg_quality'] for j in journal_rankings], 50),
            'field_average': np.mean([j['avg_quality'] for j in journal_rankings])
        }
        
        # Publication volume analysis
        publication_volume = {j['journal']: j['paper_count'] for j in journal_rankings}
        
        # Specialization scores
        specialization_scores = self._calculate_specialization_scores(journal_rankings)
        
        # Identify emerging and declining journals
        emerging_journals = self._identify_emerging_journals(journal_rankings)
        declining_journals = self._identify_declining_journals(journal_rankings)
        
        return JournalComparison(
            journal_rankings=journal_rankings[:20],  # Top 20
            quality_benchmarks=quality_benchmarks,
            publication_volume=publication_volume,
            specialization_scores=specialization_scores,
            emerging_journals=emerging_journals,
            declining_journals=declining_journals
        )
    
    async def analyze_research_group_productivity(self, collection_id: Optional[str] = None) -> ProductivityAnalysis:
        """Analyze research group productivity patterns.
        
        Args:
            collection_id: Optional collection ID for specific analysis
            
        Returns:
            ProductivityAnalysis: Research group productivity analysis
        """
        logger.info("Analyzing research group productivity patterns")
        
        # Get collection analysis
        analysis = await self.analyzer.analyze_collection(collection_id)
        
        # Analyze top productive groups (based on collaboration patterns)
        top_productive_groups = []
        for collab_info in analysis.collaboration_analysis.top_collaborators[:10]:
            productivity_score = self._calculate_productivity_score(collab_info)
            
            group_data = {
                'lead_author': collab_info['author'],
                'collaborator_count': collab_info['collaborator_count'],
                'productivity_score': productivity_score,
                'collaboration_network_size': len(collab_info['collaborators']),
                'estimated_papers': collab_info['collaborator_count'] * 2  # Simplified estimate
            }
            top_productive_groups.append(group_data)
        
        # Calculate productivity metrics
        productivity_metrics = {
            'avg_collaborators_per_paper': np.mean(list(analysis.collaboration_analysis.collaboration_trends.values())),
            'collaboration_network_density': len(analysis.collaboration_analysis.collaboration_network) / 1000,
            'institutional_diversity': len(analysis.collaboration_analysis.institutional_patterns),
            'geographic_diversity': len(analysis.collaboration_analysis.geographic_distribution)
        }
        
        # Collaboration efficiency
        collaboration_efficiency = self._calculate_collaboration_efficiency(analysis.collaboration_analysis)
        
        # Geographic productivity
        geographic_productivity = self._analyze_geographic_productivity(analysis.collaboration_analysis)
        
        # Temporal productivity trends
        temporal_productivity = analysis.collaboration_analysis.collaboration_trends
        
        return ProductivityAnalysis(
            top_productive_groups=top_productive_groups,
            productivity_metrics=productivity_metrics,
            collaboration_efficiency=collaboration_efficiency,
            geographic_productivity=geographic_productivity,
            temporal_productivity=temporal_productivity
        )
    
    async def calculate_research_impact_metrics(self, collection_id: Optional[str] = None) -> ImpactMetrics:
        """Calculate research impact metrics.
        
        Args:
            collection_id: Optional collection ID for specific analysis
            
        Returns:
            ImpactMetrics: Research impact analysis
        """
        logger.info("Calculating research impact metrics")
        
        # Get collection analysis
        analysis = await self.analyzer.analyze_collection(collection_id)
        
        # Citation indicators (simplified - would need actual citation data)
        citation_indicators = {
            'estimated_avg_citations': self._estimate_citations_from_quality(analysis.quality_analysis),
            'high_impact_paper_percentage': (analysis.quality_analysis.high_quality_count / 
                                           max(1, analysis.quality_analysis.total_papers)) * 100,
            'journal_impact_weighted_score': self._calculate_journal_weighted_impact(analysis.journal_analysis),
            'recency_impact_factor': self._calculate_recency_impact(analysis.trend_analysis)
        }
        
        # Quality-impact correlation
        quality_impact_correlation = self._calculate_quality_impact_correlation(analysis)
        
        # Temporal impact trends
        temporal_impact_trends = self._calculate_temporal_impact_trends(analysis.trend_analysis)
        
        # High-impact characteristics
        high_impact_characteristics = self._identify_high_impact_characteristics(analysis)
        
        # Impact prediction factors
        impact_prediction_factors = {
            'quality_score_weight': 0.35,
            'journal_impact_weight': 0.25,
            'collaboration_weight': 0.20,
            'methodology_weight': 0.15,
            'recency_weight': 0.05
        }
        
        return ImpactMetrics(
            citation_indicators=citation_indicators,
            quality_impact_correlation=quality_impact_correlation,
            temporal_impact_trends=temporal_impact_trends,
            high_impact_characteristics=high_impact_characteristics,
            impact_prediction_factors=impact_prediction_factors
        )
    
    async def generate_field_summary_report(self, collection_id: Optional[str] = None) -> FieldSummary:
        """Generate comprehensive field summary report.
        
        Args:
            collection_id: Optional collection ID for specific analysis
            
        Returns:
            FieldSummary: Comprehensive field summary
        """
        logger.info("Generating comprehensive field summary report")
        
        # Get all analyses
        analysis = await self.analyzer.analyze_collection(collection_id)
        benchmark_results = await self.benchmark_against_field_standards(collection_id)
        journal_comparison = await self.compare_journal_performance(collection_id)
        productivity_analysis = await self.analyze_research_group_productivity(collection_id)
        impact_metrics = await self.calculate_research_impact_metrics(collection_id)
        
        # Field overview
        field_overview = {
            'total_papers_analyzed': analysis.quality_analysis.total_papers,
            'analysis_period': self._get_analysis_period(analysis.trend_analysis),
            'journal_coverage': len(analysis.journal_analysis.journal_frequency),
            'geographic_coverage': len(analysis.collaboration_analysis.geographic_distribution),
            'research_areas_covered': len(analysis.gap_analysis.understudied_drugs) + len(analysis.gap_analysis.understudied_genes)
        }
        
        # Maturity assessment
        maturity_assessment = {
            'field_maturity_score': analysis.field_maturity.maturity_score,
            'established_areas_count': len(analysis.field_maturity.established_areas),
            'emerging_areas_count': len(analysis.field_maturity.emerging_areas),
            'translation_readiness': analysis.field_maturity.translation_indicators.get('clinical_trials', 0),
            'standardization_level': analysis.field_maturity.standardization_level
        }
        
        # Growth indicators
        growth_indicators = {
            'annual_growth_rate': analysis.trend_analysis.growth_rate,
            'quality_improvement_trend': 1.0 if analysis.quality_analysis.quality_trend == "improving" else 0.0,
            'methodology_adoption_rate': benchmark_results.collection_performance.get('methodology_adoption', 0),
            'collaboration_growth': self._calculate_collaboration_growth(analysis.collaboration_analysis)
        }
        
        # Quality trends
        quality_trends = {
            'current_avg_quality': analysis.quality_analysis.mean,
            'quality_consistency': 100 - analysis.quality_analysis.std_dev,
            'high_quality_percentage': (analysis.quality_analysis.high_quality_count / 
                                      max(1, analysis.quality_analysis.total_papers)) * 100,
            'quality_trend_direction': 1.0 if analysis.quality_analysis.quality_trend == "improving" else 
                                     -1.0 if analysis.quality_analysis.quality_trend == "declining" else 0.0
        }
        
        # Research landscape
        research_landscape = {
            'dominant_research_areas': analysis.field_maturity.established_areas[:5],
            'emerging_opportunities': [gap['entity'] for gap in analysis.gap_analysis.understudied_drugs[:3]],
            'methodology_gaps': analysis.gap_analysis.methodology_gaps[:3],
            'top_journals': [j['journal'] for j in journal_comparison.journal_rankings[:5]],
            'leading_research_groups': [g['lead_author'] for g in productivity_analysis.top_productive_groups[:5]]
        }
        
        # Future directions
        future_directions = self._identify_future_directions(analysis, benchmark_results)
        
        return FieldSummary(
            field_overview=field_overview,
            maturity_assessment=maturity_assessment,
            growth_indicators=growth_indicators,
            quality_trends=quality_trends,
            research_landscape=research_landscape,
            future_directions=future_directions
        )
    
    # Helper methods
    
    def _calculate_growth_rate(self, papers_per_year: Dict[int, int]) -> float:
        """Calculate annual growth rate."""
        years = sorted(papers_per_year.keys())
        if len(years) < 2:
            return 0.0
        
        # Calculate compound annual growth rate
        start_count = papers_per_year[years[0]]
        end_count = papers_per_year[years[-1]]
        years_span = years[-1] - years[0]
        
        if start_count > 0 and years_span > 0:
            growth_rate = ((end_count / start_count) ** (1 / years_span) - 1) * 100
            return growth_rate
        
        return 0.0
    
    def _calculate_journal_diversity(self, journal_frequency: Dict[str, int]) -> float:
        """Calculate journal diversity index."""
        if not journal_frequency:
            return 0.0
        
        total_papers = sum(journal_frequency.values())
        if total_papers == 0:
            return 0.0
        
        # Calculate Shannon diversity index
        diversity = 0.0
        for count in journal_frequency.values():
            if count > 0:
                proportion = count / total_papers
                diversity -= proportion * np.log(proportion)
        
        # Normalize to 0-1 scale
        max_diversity = np.log(len(journal_frequency))
        return diversity / max_diversity if max_diversity > 0 else 0.0
    
    def _calculate_collaboration_rate(self, collaboration_stats) -> float:
        """Calculate collaboration rate."""
        if not collaboration_stats.collaboration_trends:
            return 0.0
        
        avg_collaborators = np.mean(list(collaboration_stats.collaboration_trends.values()))
        # Convert to percentage (assuming single author = 0% collaboration)
        return min(100, (avg_collaborators - 1) * 20)  # Simplified calculation
    
    def _calculate_methodology_adoption(self, gap_analysis) -> float:
        """Calculate methodology adoption rate."""
        total_gaps = len(gap_analysis.methodology_gaps)
        if total_gaps == 0:
            return 100.0  # No gaps means high adoption
        
        # Simplified calculation - fewer gaps means higher adoption
        return max(0, 100 - (total_gaps * 10))
    
    def _calculate_percentile(self, value: float, benchmark: float) -> float:
        """Calculate percentile against benchmark."""
        # Simplified percentile calculation
        # In practice, you'd use actual distribution data
        ratio = value / benchmark if benchmark > 0 else 1.0
        
        if ratio >= 1.5:
            return 95.0
        elif ratio >= 1.25:
            return 85.0
        elif ratio >= 1.1:
            return 75.0
        elif ratio >= 0.9:
            return 50.0
        elif ratio >= 0.75:
            return 25.0
        else:
            return 10.0
    
    def _generate_benchmark_summary(self, collection_perf: Dict[str, float], 
                                  field_benchmarks: Dict[str, float], overall_score: float) -> str:
        """Generate benchmark comparison summary."""
        if overall_score >= 75:
            performance_level = "excellent"
        elif overall_score >= 60:
            performance_level = "good"
        elif overall_score >= 40:
            performance_level = "average"
        else:
            performance_level = "below average"
        
        summary = f"Collection shows {performance_level} performance (overall score: {overall_score:.1f}). "
        
        # Highlight key comparisons
        quality_ratio = collection_perf.get('avg_quality_score', 0) / field_benchmarks.get('avg_quality_score', 1)
        if quality_ratio > 1.1:
            summary += f"Quality scores are {((quality_ratio - 1) * 100):.1f}% above field average. "
        elif quality_ratio < 0.9:
            summary += f"Quality scores are {((1 - quality_ratio) * 100):.1f}% below field average. "
        
        return summary
    
    def _calculate_specialization_scores(self, journal_rankings: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate journal specialization scores."""
        specialization_scores = {}
        
        for journal_info in journal_rankings:
            journal = journal_info['journal']
            specializations = journal_info.get('specialization', [])
            
            # Calculate specialization score based on focus areas
            if len(specializations) == 1:
                specialization_scores[journal] = 100.0  # Highly specialized
            elif len(specializations) <= 3:
                specialization_scores[journal] = 75.0   # Moderately specialized
            else:
                specialization_scores[journal] = 50.0   # General
        
        return specialization_scores
    
    def _identify_emerging_journals(self, journal_rankings: List[Dict[str, Any]]) -> List[str]:
        """Identify emerging journals."""
        emerging = []
        
        for journal_info in journal_rankings:
            journal = journal_info['journal']
            # Simplified criteria: high quality but not in traditional top tier
            if (journal_info['avg_quality'] > 75 and 
                journal not in ['Nature Genetics', 'Clinical Pharmacology & Therapeutics']):
                emerging.append(journal)
        
        return emerging[:5]  # Top 5 emerging
    
    def _identify_declining_journals(self, journal_rankings: List[Dict[str, Any]]) -> List[str]:
        """Identify declining journals."""
        declining = []
        
        for journal_info in journal_rankings:
            journal = journal_info['journal']
            # Simplified criteria: low quality despite being established
            if (journal_info['avg_quality'] < 60 and 
                journal_info['paper_count'] < 10):  # Low activity
                declining.append(journal)
        
        return declining[:3]  # Top 3 declining
    
    def _calculate_productivity_score(self, collab_info: Dict[str, Any]) -> float:
        """Calculate productivity score for research group."""
        collaborator_count = collab_info['collaborator_count']
        
        # Simplified productivity score
        base_score = min(100, collaborator_count * 5)  # 5 points per collaborator
        
        # Bonus for high collaboration
        if collaborator_count > 20:
            base_score += 20
        elif collaborator_count > 10:
            base_score += 10
        
        return min(100, base_score)
    
    def _calculate_collaboration_efficiency(self, collaboration_stats) -> Dict[str, float]:
        """Calculate collaboration efficiency metrics."""
        return {
            'network_density': len(collaboration_stats.collaboration_network) / 1000,
            'avg_collaborators': np.mean(list(collaboration_stats.collaboration_trends.values())) if collaboration_stats.collaboration_trends else 0,
            'institutional_diversity': len(collaboration_stats.institutional_patterns) / 100,
            'geographic_reach': len(collaboration_stats.geographic_distribution) / 20
        }
    
    def _analyze_geographic_productivity(self, collaboration_stats) -> Dict[str, Dict[str, float]]:
        """Analyze geographic productivity patterns."""
        geographic_productivity = {}
        
        total_papers = sum(collaboration_stats.geographic_distribution.values())
        
        for region, count in collaboration_stats.geographic_distribution.items():
            productivity_metrics = {
                'paper_count': count,
                'percentage_share': (count / total_papers) * 100 if total_papers > 0 else 0,
                'productivity_index': min(100, count * 2)  # Simplified index
            }
            geographic_productivity[region] = productivity_metrics
        
        return geographic_productivity
    
    def _estimate_citations_from_quality(self, quality_stats: QualityStats) -> float:
        """Estimate citations based on quality scores."""
        # Simplified estimation model
        base_citations = quality_stats.mean * 0.3  # Base correlation
        
        # Bonus for high-quality papers
        high_quality_bonus = (quality_stats.high_quality_count / max(1, quality_stats.total_papers)) * 10
        
        return base_citations + high_quality_bonus
    
    def _calculate_journal_weighted_impact(self, journal_analysis) -> float:
        """Calculate journal impact-weighted score."""
        weighted_score = 0.0
        total_papers = 0
        
        for journal_info in journal_analysis.top_journals:
            journal = journal_info['journal']
            paper_count = journal_info['paper_count']
            impact_score = self.journal_impact_rankings.get(journal, 50.0)
            
            weighted_score += impact_score * paper_count
            total_papers += paper_count
        
        return weighted_score / total_papers if total_papers > 0 else 0.0
    
    def _calculate_recency_impact(self, trend_analysis) -> float:
        """Calculate recency impact factor."""
        # Recent papers have higher potential impact
        current_year = datetime.now().year
        recent_years = [current_year - i for i in range(3)]  # Last 3 years
        
        recent_papers = sum(trend_analysis.papers_per_year.get(year, 0) for year in recent_years)
        total_papers = sum(trend_analysis.papers_per_year.values())
        
        return (recent_papers / total_papers) * 100 if total_papers > 0 else 0.0
    
    def _calculate_quality_impact_correlation(self, analysis) -> float:
        """Calculate correlation between quality and estimated impact."""
        # Simplified correlation calculation
        # In practice, you'd use actual citation data
        
        quality_scores = [analysis.quality_analysis.mean]
        impact_scores = [self._estimate_citations_from_quality(analysis.quality_analysis)]
        
        if len(quality_scores) > 1:
            correlation = np.corrcoef(quality_scores, impact_scores)[0, 1]
            return float(correlation) if not np.isnan(correlation) else 0.0
        
        return 0.75  # Assumed positive correlation
    
    def _calculate_temporal_impact_trends(self, trend_analysis) -> Dict[int, float]:
        """Calculate temporal impact trends."""
        temporal_trends = {}
        
        for year, paper_count in trend_analysis.papers_per_year.items():
            # Simplified impact calculation based on recency and volume
            recency_factor = max(0, 1 - (datetime.now().year - year) * 0.1)
            volume_factor = min(1, paper_count / 50)  # Normalize by expected volume
            
            temporal_trends[year] = (recency_factor + volume_factor) * 50
        
        return temporal_trends
    
    def _identify_high_impact_characteristics(self, analysis) -> List[str]:
        """Identify characteristics of high-impact research."""
        characteristics = []
        
        # High-quality research
        if analysis.quality_analysis.mean > 75:
            characteristics.append("High average quality scores (>75)")
        
        # Strong collaboration
        avg_collaborators = np.mean(list(analysis.collaboration_analysis.collaboration_trends.values())) if analysis.collaboration_analysis.collaboration_trends else 0
        if avg_collaborators > 6:
            characteristics.append("Strong international collaboration")
        
        # Top-tier journals
        top_journals = [j for j in analysis.journal_analysis.top_journals if j['avg_quality'] > 80]
        if len(top_journals) > 3:
            characteristics.append("Publication in high-impact journals")
        
        # Methodology innovation
        if len(analysis.gap_analysis.methodology_gaps) < 3:
            characteristics.append("Advanced methodology adoption")
        
        return characteristics
    
    def _get_analysis_period(self, trend_analysis) -> str:
        """Get analysis time period."""
        years = sorted(trend_analysis.papers_per_year.keys())
        if years:
            return f"{years[0]}-{years[-1]}"
        return "Unknown"
    
    def _calculate_collaboration_growth(self, collaboration_stats) -> float:
        """Calculate collaboration growth rate."""
        trends = collaboration_stats.collaboration_trends
        if len(trends) < 2:
            return 0.0
        
        years = sorted(trends.keys())
        early_avg = np.mean([trends[year] for year in years[:len(years)//2]])
        late_avg = np.mean([trends[year] for year in years[len(years)//2:]])
        
        return ((late_avg - early_avg) / early_avg) * 100 if early_avg > 0 else 0.0
    
    def _identify_future_directions(self, analysis, benchmark_results) -> List[str]:
        """Identify future research directions."""
        directions = []
        
        # Based on gaps
        if analysis.gap_analysis.understudied_drugs:
            directions.append(f"Expand research on {analysis.gap_analysis.understudied_drugs[0]['entity']} pharmacogenomics")
        
        # Based on methodology gaps
        if analysis.gap_analysis.methodology_gaps:
            directions.append(f"Adopt {analysis.gap_analysis.methodology_gaps[0]} methodologies")
        
        # Based on quality trends
        if analysis.quality_analysis.quality_trend == "improving":
            directions.append("Continue quality improvement initiatives")
        else:
            directions.append("Implement quality enhancement programs")
        
        # Based on collaboration patterns
        if benchmark_results.collection_performance.get('collaboration_rate', 0) < 50:
            directions.append("Strengthen international research collaborations")
        
        # Based on field maturity
        if analysis.field_maturity.maturity_score < 70:
            directions.append("Focus on clinical translation and implementation")
        
        return directions[:5]  # Top 5 directions