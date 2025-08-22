"""Tests for the benchmarking module."""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock

from src.analytics.benchmarking import (
    PharmacogenomicsBenchmarking, BenchmarkResults, JournalComparison,
    ProductivityAnalysis, ImpactMetrics, FieldSummary
)
from src.analytics.research_analyzer import (
    ComprehensiveAnalysis, QualityStats, PublicationTrends, ResearchGaps,
    JournalStats, CollaborationStats, FieldMaturity
)


class TestPharmacogenomicsBenchmarking:
    """Test cases for PharmacogenomicsBenchmarking."""
    
    @pytest.fixture
    def benchmarking(self):
        """Create benchmarking instance."""
        return PharmacogenomicsBenchmarking()
    
    @pytest.fixture
    def sample_analysis(self):
        """Sample comprehensive analysis for testing."""
        return ComprehensiveAnalysis(
            quality_analysis=QualityStats(
                mean=75.5,
                median=76.0,
                std_dev=12.3,
                percentiles={25: 65.0, 50: 76.0, 75: 85.0, 90: 92.0, 95: 95.0},
                total_papers=100,
                high_quality_count=40,
                medium_quality_count=45,
                low_quality_count=15,
                quality_trend="improving",
                outliers=[]
            ),
            trend_analysis=PublicationTrends(
                papers_per_year={2020: 20, 2021: 25, 2022: 30, 2023: 35},
                papers_per_month={"2023-01": 3, "2023-02": 4, "2023-03": 2},
                growth_rate=15.2,
                peak_year=2023,
                recent_trend="increasing",
                seasonal_patterns={"Jan": 8.5, "Feb": 9.2, "Mar": 7.8}
            ),
            gap_analysis=ResearchGaps(
                understudied_drugs=[
                    {"entity": "ABACAVIR", "count": 8, "opportunity_score": 12}
                ],
                understudied_genes=[
                    {"entity": "DPYD", "count": 12, "opportunity_score": 8}
                ],
                methodology_gaps=["deep learning", "ensemble methods"],
                population_gaps=["Asian populations", "African populations"],
                therapeutic_area_gaps=["Pediatrics", "Geriatrics"],
                quality_improvement_areas=["Clinical Relevance"]
            ),
            journal_analysis=JournalStats(
                journal_frequency={"Clinical Pharmacology & Therapeutics": 25, "Pharmacogenomics": 20},
                journal_quality_scores={"Clinical Pharmacology & Therapeutics": 85.0, "Pharmacogenomics": 78.0},
                top_journals=[
                    {"journal": "Clinical Pharmacology & Therapeutics", "paper_count": 25, "avg_quality": 85.0, "quality_std": 8.5},
                    {"journal": "Pharmacogenomics", "paper_count": 20, "avg_quality": 78.0, "quality_std": 12.2}
                ],
                journal_specialization={"Clinical Pharmacology & Therapeutics": ["CYP enzymes", "Warfarin"]},
                impact_correlation=0.73
            ),
            collaboration_analysis=CollaborationStats(
                collaboration_network={"John Smith": ["Jane Doe", "Bob Wilson"]},
                top_collaborators=[
                    {"author": "John Smith", "collaborator_count": 15, "collaborators": ["Jane Doe", "Bob Wilson"]}
                ],
                institutional_patterns={"Harvard Medical School": 10, "Mayo Clinic": 8},
                geographic_distribution={"USA": 45, "UK": 25, "Canada": 15},
                collaboration_trends={2020: 4.2, 2021: 4.8, 2022: 5.1, 2023: 5.5}
            ),
            insights=[
                {"type": "trending_topic", "title": "AI in Pharmacogenomics", "description": "Growing trend"}
            ],
            recommendations=[],
            field_maturity=FieldMaturity(
                maturity_score=72.5,
                established_areas=["CYP2D6 pharmacogenomics", "Warfarin dosing"],
                emerging_areas=["AI applications", "Rare variants"],
                declining_areas=[],
                translation_indicators={"clinical_trials": 25.0, "implementation": 15.0, "guidelines": 10.0},
                standardization_level=68.5
            )
        )
    
    def test_benchmarking_initialization(self, benchmarking):
        """Test benchmarking initialization."""
        assert benchmarking.analyzer is not None
        assert len(benchmarking.field_benchmarks) > 0
        assert len(benchmarking.journal_impact_rankings) > 0
        
        # Check key benchmarks exist
        assert 'avg_quality_score' in benchmarking.field_benchmarks
        assert 'high_quality_percentage' in benchmarking.field_benchmarks
        assert 'papers_per_year_growth' in benchmarking.field_benchmarks
    
    @pytest.mark.asyncio
    async def test_benchmark_against_field_standards(self, benchmarking, sample_analysis):
        """Test benchmarking against field standards."""
        with patch.object(benchmarking.analyzer, 'analyze_collection', new_callable=AsyncMock) as mock_analyze:
            mock_analyze.return_value = sample_analysis
            
            benchmark_results = await benchmarking.benchmark_against_field_standards()
            
            assert isinstance(benchmark_results, BenchmarkResults)
            assert len(benchmark_results.collection_performance) > 0
            assert len(benchmark_results.field_benchmarks) > 0
            assert len(benchmark_results.performance_percentile) > 0
            assert isinstance(benchmark_results.strengths, list)
            assert isinstance(benchmark_results.improvement_areas, list)
            assert 0 <= benchmark_results.overall_score <= 100
            assert isinstance(benchmark_results.comparison_summary, str)
    
    @pytest.mark.asyncio
    async def test_compare_journal_performance(self, benchmarking, sample_analysis):
        """Test journal performance comparison."""
        with patch.object(benchmarking.analyzer, 'analyze_collection', new_callable=AsyncMock) as mock_analyze:
            mock_analyze.return_value = sample_analysis
            
            journal_comparison = await benchmarking.compare_journal_performance()
            
            assert isinstance(journal_comparison, JournalComparison)
            assert len(journal_comparison.journal_rankings) > 0
            assert len(journal_comparison.quality_benchmarks) > 0
            assert len(journal_comparison.publication_volume) > 0
            assert isinstance(journal_comparison.specialization_scores, dict)
            assert isinstance(journal_comparison.emerging_journals, list)
            assert isinstance(journal_comparison.declining_journals, list)
            
            # Check journal ranking structure
            for journal in journal_comparison.journal_rankings:
                assert 'journal' in journal
                assert 'paper_count' in journal
                assert 'avg_quality' in journal
                assert 'impact_score' in journal
                assert 'combined_score' in journal
    
    @pytest.mark.asyncio
    async def test_analyze_research_group_productivity(self, benchmarking, sample_analysis):
        """Test research group productivity analysis."""
        with patch.object(benchmarking.analyzer, 'analyze_collection', new_callable=AsyncMock) as mock_analyze:
            mock_analyze.return_value = sample_analysis
            
            productivity_analysis = await benchmarking.analyze_research_group_productivity()
            
            assert isinstance(productivity_analysis, ProductivityAnalysis)
            assert isinstance(productivity_analysis.top_productive_groups, list)
            assert isinstance(productivity_analysis.productivity_metrics, dict)
            assert isinstance(productivity_analysis.collaboration_efficiency, dict)
            assert isinstance(productivity_analysis.geographic_productivity, dict)
            assert isinstance(productivity_analysis.temporal_productivity, dict)
            
            # Check productivity metrics structure
            for group in productivity_analysis.top_productive_groups:
                assert 'lead_author' in group
                assert 'collaborator_count' in group
                assert 'productivity_score' in group
                assert 0 <= group['productivity_score'] <= 100
    
    @pytest.mark.asyncio
    async def test_calculate_research_impact_metrics(self, benchmarking, sample_analysis):
        """Test research impact metrics calculation."""
        with patch.object(benchmarking.analyzer, 'analyze_collection', new_callable=AsyncMock) as mock_analyze:
            mock_analyze.return_value = sample_analysis
            
            impact_metrics = await benchmarking.calculate_research_impact_metrics()
            
            assert isinstance(impact_metrics, ImpactMetrics)
            assert isinstance(impact_metrics.citation_indicators, dict)
            assert isinstance(impact_metrics.quality_impact_correlation, float)
            assert isinstance(impact_metrics.temporal_impact_trends, dict)
            assert isinstance(impact_metrics.high_impact_characteristics, list)
            assert isinstance(impact_metrics.impact_prediction_factors, dict)
            
            # Check citation indicators
            assert 'estimated_avg_citations' in impact_metrics.citation_indicators
            assert 'high_impact_paper_percentage' in impact_metrics.citation_indicators
            
            # Check correlation is valid
            assert -1 <= impact_metrics.quality_impact_correlation <= 1
    
    @pytest.mark.asyncio
    async def test_generate_field_summary_report(self, benchmarking, sample_analysis):
        """Test field summary report generation."""
        with patch.object(benchmarking.analyzer, 'analyze_collection', new_callable=AsyncMock) as mock_analyze:
            with patch.object(benchmarking, 'benchmark_against_field_standards', new_callable=AsyncMock) as mock_benchmark:
                with patch.object(benchmarking, 'compare_journal_performance', new_callable=AsyncMock) as mock_journal:
                    with patch.object(benchmarking, 'analyze_research_group_productivity', new_callable=AsyncMock) as mock_productivity:
                        with patch.object(benchmarking, 'calculate_research_impact_metrics', new_callable=AsyncMock) as mock_impact:
                            
                            # Setup mocks
                            mock_analyze.return_value = sample_analysis
                            mock_benchmark.return_value = BenchmarkResults(
                                collection_performance={'avg_quality_score': 75.5},
                                field_benchmarks={'avg_quality_score': 68.5},
                                performance_percentile={'avg_quality_score': 75.0},
                                strengths=["High quality scores"],
                                improvement_areas=["Collaboration rate"],
                                overall_score=72.5,
                                comparison_summary="Good performance"
                            )
                            mock_journal.return_value = JournalComparison(
                                journal_rankings=[{"journal": "Test Journal", "combined_score": 80.0}],
                                quality_benchmarks={'top_tier': 90.0},
                                publication_volume={'Test Journal': 25},
                                specialization_scores={'Test Journal': 85.0},
                                emerging_journals=["Emerging Journal"],
                                declining_journals=[]
                            )
                            mock_productivity.return_value = ProductivityAnalysis(
                                top_productive_groups=[{"lead_author": "John Smith", "productivity_score": 85.0}],
                                productivity_metrics={'avg_collaborators_per_paper': 5.2},
                                collaboration_efficiency={'network_density': 0.15},
                                geographic_productivity={'USA': {'paper_count': 45}},
                                temporal_productivity={2023: 5.5}
                            )
                            mock_impact.return_value = ImpactMetrics(
                                citation_indicators={'estimated_avg_citations': 18.5},
                                quality_impact_correlation=0.73,
                                temporal_impact_trends={2023: 85.0},
                                high_impact_characteristics=["High quality"],
                                impact_prediction_factors={'quality_score_weight': 0.35}
                            )
                            
                            field_summary = await benchmarking.generate_field_summary_report()
                            
                            assert isinstance(field_summary, FieldSummary)
                            assert isinstance(field_summary.field_overview, dict)
                            assert isinstance(field_summary.maturity_assessment, dict)
                            assert isinstance(field_summary.growth_indicators, dict)
                            assert isinstance(field_summary.quality_trends, dict)
                            assert isinstance(field_summary.research_landscape, dict)
                            assert isinstance(field_summary.future_directions, list)
                            
                            # Check field overview structure
                            assert 'total_papers_analyzed' in field_summary.field_overview
                            assert 'journal_coverage' in field_summary.field_overview
                            
                            # Check maturity assessment
                            assert 'field_maturity_score' in field_summary.maturity_assessment
                            assert 'standardization_level' in field_summary.maturity_assessment
    
    def test_calculate_growth_rate(self, benchmarking):
        """Test growth rate calculation."""
        papers_per_year = {2020: 10, 2021: 12, 2022: 15, 2023: 18}
        growth_rate = benchmarking._calculate_growth_rate(papers_per_year)
        
        assert isinstance(growth_rate, float)
        assert growth_rate > 0  # Should show positive growth
    
    def test_calculate_growth_rate_empty(self, benchmarking):
        """Test growth rate calculation with empty data."""
        growth_rate = benchmarking._calculate_growth_rate({})
        assert growth_rate == 0.0
    
    def test_calculate_journal_diversity(self, benchmarking):
        """Test journal diversity calculation."""
        journal_frequency = {"Journal A": 20, "Journal B": 15, "Journal C": 10, "Journal D": 5}
        diversity = benchmarking._calculate_journal_diversity(journal_frequency)
        
        assert isinstance(diversity, float)
        assert 0 <= diversity <= 1
    
    def test_calculate_journal_diversity_empty(self, benchmarking):
        """Test journal diversity calculation with empty data."""
        diversity = benchmarking._calculate_journal_diversity({})
        assert diversity == 0.0
    
    def test_calculate_collaboration_rate(self, benchmarking):
        """Test collaboration rate calculation."""
        collaboration_stats = Mock()
        collaboration_stats.collaboration_trends = {2020: 4.2, 2021: 4.8, 2022: 5.1}
        
        collab_rate = benchmarking._calculate_collaboration_rate(collaboration_stats)
        
        assert isinstance(collab_rate, float)
        assert 0 <= collab_rate <= 100
    
    def test_calculate_methodology_adoption(self, benchmarking):
        """Test methodology adoption calculation."""
        gap_analysis = Mock()
        gap_analysis.methodology_gaps = ["deep learning", "ensemble methods"]
        
        adoption_rate = benchmarking._calculate_methodology_adoption(gap_analysis)
        
        assert isinstance(adoption_rate, float)
        assert 0 <= adoption_rate <= 100
    
    def test_calculate_percentile(self, benchmarking):
        """Test percentile calculation."""
        # Test various ratios
        assert benchmarking._calculate_percentile(75.0, 50.0) == 95.0  # 1.5x benchmark
        assert benchmarking._calculate_percentile(62.5, 50.0) == 85.0  # 1.25x benchmark
        assert benchmarking._calculate_percentile(55.0, 50.0) == 75.0  # 1.1x benchmark
        assert benchmarking._calculate_percentile(50.0, 50.0) == 50.0  # Equal to benchmark
        assert benchmarking._calculate_percentile(37.5, 50.0) == 25.0  # 0.75x benchmark
        assert benchmarking._calculate_percentile(25.0, 50.0) == 10.0  # 0.5x benchmark
    
    def test_generate_benchmark_summary(self, benchmarking):
        """Test benchmark summary generation."""
        collection_perf = {'avg_quality_score': 75.0}
        field_benchmarks = {'avg_quality_score': 68.5}
        overall_score = 78.5
        
        summary = benchmarking._generate_benchmark_summary(collection_perf, field_benchmarks, overall_score)
        
        assert isinstance(summary, str)
        assert len(summary) > 0
        assert "good" in summary.lower() or "excellent" in summary.lower()
    
    def test_calculate_specialization_scores(self, benchmarking):
        """Test specialization scores calculation."""
        journal_rankings = [
            {"journal": "Specialized Journal", "specialization": ["CYP enzymes"]},
            {"journal": "General Journal", "specialization": ["CYP enzymes", "Warfarin", "AI", "Clinical"]},
            {"journal": "Moderate Journal", "specialization": ["CYP enzymes", "Warfarin"]}
        ]
        
        spec_scores = benchmarking._calculate_specialization_scores(journal_rankings)
        
        assert isinstance(spec_scores, dict)
        assert spec_scores["Specialized Journal"] == 100.0  # Highly specialized
        assert spec_scores["General Journal"] == 50.0      # General
        assert spec_scores["Moderate Journal"] == 75.0     # Moderately specialized
    
    def test_identify_emerging_journals(self, benchmarking):
        """Test emerging journals identification."""
        journal_rankings = [
            {"journal": "Nature Genetics", "avg_quality": 95.0},  # Traditional top tier
            {"journal": "Emerging Journal", "avg_quality": 85.0}, # High quality, not traditional
            {"journal": "Low Quality Journal", "avg_quality": 60.0}
        ]
        
        emerging = benchmarking._identify_emerging_journals(journal_rankings)
        
        assert isinstance(emerging, list)
        assert "Emerging Journal" in emerging
        assert "Nature Genetics" not in emerging  # Traditional journal
    
    def test_identify_declining_journals(self, benchmarking):
        """Test declining journals identification."""
        journal_rankings = [
            {"journal": "High Quality Journal", "avg_quality": 85.0, "paper_count": 25},
            {"journal": "Declining Journal", "avg_quality": 55.0, "paper_count": 5}  # Low quality, low activity
        ]
        
        declining = benchmarking._identify_declining_journals(journal_rankings)
        
        assert isinstance(declining, list)
        assert "Declining Journal" in declining
        assert "High Quality Journal" not in declining
    
    def test_calculate_productivity_score(self, benchmarking):
        """Test productivity score calculation."""
        collab_info = {"collaborator_count": 15}
        
        score = benchmarking._calculate_productivity_score(collab_info)
        
        assert isinstance(score, float)
        assert 0 <= score <= 100
        
        # Test high collaboration bonus
        high_collab_info = {"collaborator_count": 25}
        high_score = benchmarking._calculate_productivity_score(high_collab_info)
        assert high_score >= score  # Should be higher
    
    def test_calculate_collaboration_efficiency(self, benchmarking):
        """Test collaboration efficiency calculation."""
        collaboration_stats = Mock()
        collaboration_stats.collaboration_network = {"author1": ["author2"], "author2": ["author1"]}
        collaboration_stats.collaboration_trends = {2020: 4.2, 2021: 4.8}
        collaboration_stats.institutional_patterns = {"Institution1": 10, "Institution2": 8}
        collaboration_stats.geographic_distribution = {"USA": 45, "UK": 25}
        
        efficiency = benchmarking._calculate_collaboration_efficiency(collaboration_stats)
        
        assert isinstance(efficiency, dict)
        assert 'network_density' in efficiency
        assert 'avg_collaborators' in efficiency
        assert 'institutional_diversity' in efficiency
        assert 'geographic_reach' in efficiency
        
        for value in efficiency.values():
            assert isinstance(value, float)
            assert value >= 0
    
    def test_analyze_geographic_productivity(self, benchmarking):
        """Test geographic productivity analysis."""
        collaboration_stats = Mock()
        collaboration_stats.geographic_distribution = {"USA": 45, "UK": 25, "Canada": 15}
        
        geo_productivity = benchmarking._analyze_geographic_productivity(collaboration_stats)
        
        assert isinstance(geo_productivity, dict)
        
        for region, metrics in geo_productivity.items():
            assert 'paper_count' in metrics
            assert 'percentage_share' in metrics
            assert 'productivity_index' in metrics
            assert metrics['paper_count'] > 0
            assert 0 <= metrics['percentage_share'] <= 100
    
    def test_estimate_citations_from_quality(self, benchmarking):
        """Test citation estimation from quality scores."""
        quality_stats = Mock()
        quality_stats.mean = 75.0
        quality_stats.high_quality_count = 25
        quality_stats.total_papers = 100
        
        estimated_citations = benchmarking._estimate_citations_from_quality(quality_stats)
        
        assert isinstance(estimated_citations, float)
        assert estimated_citations > 0
    
    def test_calculate_journal_weighted_impact(self, benchmarking):
        """Test journal weighted impact calculation."""
        journal_analysis = Mock()
        journal_analysis.top_journals = [
            {"journal": "Clinical Pharmacology & Therapeutics", "paper_count": 25},
            {"journal": "Unknown Journal", "paper_count": 10}
        ]
        
        weighted_impact = benchmarking._calculate_journal_weighted_impact(journal_analysis)
        
        assert isinstance(weighted_impact, float)
        assert weighted_impact > 0
    
    def test_calculate_recency_impact(self, benchmarking):
        """Test recency impact calculation."""
        trend_analysis = Mock()
        current_year = 2024
        trend_analysis.papers_per_year = {
            2021: 10,
            2022: 15,
            2023: 20,
            2024: 25
        }
        
        recency_impact = benchmarking._calculate_recency_impact(trend_analysis)
        
        assert isinstance(recency_impact, float)
        assert 0 <= recency_impact <= 100
    
    def test_identify_future_directions(self, benchmarking):
        """Test future directions identification."""
        analysis = Mock()
        analysis.gap_analysis.understudied_drugs = [{"entity": "ABACAVIR", "count": 8}]
        analysis.gap_analysis.methodology_gaps = ["deep learning"]
        analysis.quality_analysis.quality_trend = "improving"
        analysis.field_maturity.maturity_score = 65.0
        
        benchmark_results = Mock()
        benchmark_results.collection_performance = {"collaboration_rate": 40.0}
        
        directions = benchmarking._identify_future_directions(analysis, benchmark_results)
        
        assert isinstance(directions, list)
        assert len(directions) <= 5  # Should limit to top 5
        
        for direction in directions:
            assert isinstance(direction, str)
            assert len(direction) > 0


class TestBenchmarkingIntegration:
    """Integration tests for benchmarking components."""
    
    @pytest.mark.asyncio
    async def test_full_benchmarking_pipeline(self):
        """Test complete benchmarking pipeline."""
        benchmarking = PharmacogenomicsBenchmarking()
        
        # Create comprehensive mock analysis
        mock_analysis = ComprehensiveAnalysis(
            quality_analysis=QualityStats(75.0, 76.0, 12.0, {}, 100, 40, 45, 15, "improving", []),
            trend_analysis=PublicationTrends({2023: 35}, {}, 15.0, 2023, "increasing", {}),
            gap_analysis=ResearchGaps([], [], [], [], [], []),
            journal_analysis=JournalStats({}, {}, [], {}, 0.7),
            collaboration_analysis=CollaborationStats({}, [], {}, {}, {}),
            insights=[],
            recommendations=[],
            field_maturity=FieldMaturity(70.0, [], [], [], {}, 65.0)
        )
        
        with patch.object(benchmarking.analyzer, 'analyze_collection', new_callable=AsyncMock) as mock_analyze:
            mock_analyze.return_value = mock_analysis
            
            # Test all major benchmarking functions
            benchmark_results = await benchmarking.benchmark_against_field_standards()
            journal_comparison = await benchmarking.compare_journal_performance()
            productivity_analysis = await benchmarking.analyze_research_group_productivity()
            impact_metrics = await benchmarking.calculate_research_impact_metrics()
            field_summary = await benchmarking.generate_field_summary_report()
            
            # Verify all results are valid
            assert isinstance(benchmark_results, BenchmarkResults)
            assert isinstance(journal_comparison, JournalComparison)
            assert isinstance(productivity_analysis, ProductivityAnalysis)
            assert isinstance(impact_metrics, ImpactMetrics)
            assert isinstance(field_summary, FieldSummary)
    
    def test_error_handling_in_benchmarking(self):
        """Test error handling in benchmarking."""
        benchmarking = PharmacogenomicsBenchmarking()
        
        # Test with edge cases
        empty_papers_per_year = {}
        growth_rate = benchmarking._calculate_growth_rate(empty_papers_per_year)
        assert growth_rate == 0.0
        
        empty_journal_freq = {}
        diversity = benchmarking._calculate_journal_diversity(empty_journal_freq)
        assert diversity == 0.0
        
        # Test percentile calculation with zero benchmark
        percentile = benchmarking._calculate_percentile(50.0, 0.0)
        assert isinstance(percentile, float)
    
    def test_benchmarking_consistency(self):
        """Test consistency of benchmarking calculations."""
        benchmarking = PharmacogenomicsBenchmarking()
        
        # Test that same inputs produce same outputs
        papers_per_year = {2020: 10, 2021: 15, 2022: 20}
        
        growth_rate_1 = benchmarking._calculate_growth_rate(papers_per_year)
        growth_rate_2 = benchmarking._calculate_growth_rate(papers_per_year)
        
        assert growth_rate_1 == growth_rate_2
        
        # Test percentile consistency
        percentile_1 = benchmarking._calculate_percentile(75.0, 50.0)
        percentile_2 = benchmarking._calculate_percentile(75.0, 50.0)
        
        assert percentile_1 == percentile_2