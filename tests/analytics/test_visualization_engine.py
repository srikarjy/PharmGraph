"""Tests for the visualization engine module."""

import pytest
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from unittest.mock import Mock, patch

from src.analytics.visualization_engine import VisualizationEngine, VisualizationConfig
from src.analytics.research_analyzer import (
    ComprehensiveAnalysis, QualityStats, PublicationTrends, 
    ResearchGaps, JournalStats, CollaborationStats, FieldMaturity
)
from src.analytics.benchmarking import BenchmarkResults


class TestVisualizationEngine:
    """Test cases for VisualizationEngine."""
    
    @pytest.fixture
    def viz_config(self):
        """Create visualization configuration."""
        return VisualizationConfig(
            interactive=True,
            figure_size=(10, 6),
            color_palette="viridis"
        )
    
    @pytest.fixture
    def viz_engine(self, viz_config):
        """Create visualization engine instance."""
        return VisualizationEngine(viz_config)
    
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
                population_gaps=["Asian populations"],
                therapeutic_area_gaps=["Pediatrics"],
                quality_improvement_areas=["Clinical Relevance"]
            ),
            journal_analysis=JournalStats(
                journal_frequency={"Clinical Pharmacology & Therapeutics": 25, "Pharmacogenomics": 20},
                journal_quality_scores={"Clinical Pharmacology & Therapeutics": 85.0, "Pharmacogenomics": 78.0},
                top_journals=[
                    {"journal": "Clinical Pharmacology & Therapeutics", "paper_count": 25, "avg_quality": 85.0, "quality_std": 8.5},
                    {"journal": "Pharmacogenomics", "paper_count": 20, "avg_quality": 78.0, "quality_std": 12.2}
                ],
                journal_specialization={"Clinical Pharmacology & Therapeutics": ["CYP enzymes"]},
                impact_correlation=0.73
            ),
            collaboration_analysis=CollaborationStats(
                collaboration_network={"John Smith": ["Jane Doe"]},
                top_collaborators=[
                    {"author": "John Smith", "collaborator_count": 15, "collaborators": ["Jane Doe"]}
                ],
                institutional_patterns={"Harvard Medical School": 10},
                geographic_distribution={"USA": 45, "UK": 25},
                collaboration_trends={2020: 4.2, 2021: 4.8, 2022: 5.1, 2023: 5.5}
            ),
            insights=[
                {"type": "trending_topic", "title": "AI in Pharmacogenomics", "description": "Growing trend"}
            ],
            recommendations=[],
            field_maturity=FieldMaturity(
                maturity_score=72.5,
                established_areas=["CYP2D6 pharmacogenomics"],
                emerging_areas=["AI applications"],
                declining_areas=[],
                translation_indicators={"clinical_trials": 25.0},
                standardization_level=68.5
            )
        )
    
    @pytest.fixture
    def sample_benchmark_results(self):
        """Sample benchmark results for testing."""
        return BenchmarkResults(
            collection_performance={'avg_quality_score': 75.5},
            field_benchmarks={'avg_quality_score': 68.5},
            performance_percentile={'avg_quality_score': 75.0},
            strengths=["High quality scores"],
            improvement_areas=["Collaboration rate"],
            overall_score=72.5,
            comparison_summary="Good performance"
        )
    
    def test_visualization_engine_initialization(self, viz_engine):
        """Test visualization engine initialization."""
        assert viz_engine.config is not None
        assert viz_engine.config.interactive is True
        assert viz_engine.config.figure_size == (10, 6)
        assert len(viz_engine.quality_colors) == 3
        assert len(viz_engine.trend_colors) == 3
    
    def test_create_quality_distribution_dashboard(self, viz_engine, sample_analysis):
        """Test quality distribution dashboard creation."""
        dashboard = viz_engine.create_quality_distribution_dashboard(sample_analysis)
        
        assert isinstance(dashboard, dict)
        assert 'histogram' in dashboard
        assert 'pie_chart' in dashboard
        assert 'box_plot' in dashboard
        assert 'trend_chart' in dashboard
        assert 'scatter_plot' in dashboard
    
    def test_create_publication_trends_dashboard(self, viz_engine, sample_analysis):
        """Test publication trends dashboard creation."""
        dashboard = viz_engine.create_publication_trends_dashboard(sample_analysis)
        
        assert isinstance(dashboard, dict)
        assert 'yearly_trends' in dashboard
        assert 'monthly_heatmap' in dashboard
        assert 'growth_analysis' in dashboard
        assert 'seasonal_radar' in dashboard
        assert 'velocity_chart' in dashboard
    
    def test_create_journal_analysis_dashboard(self, viz_engine, sample_analysis):
        """Test journal analysis dashboard creation."""
        dashboard = viz_engine.create_journal_analysis_dashboard(sample_analysis)
        
        assert isinstance(dashboard, dict)
        assert 'top_journals' in dashboard
        assert 'quality_volume_bubble' in dashboard
        assert 'specialization_network' in dashboard
        assert 'impact_distribution' in dashboard
        assert 'quality_comparison' in dashboard
    
    def test_create_research_gaps_dashboard(self, viz_engine, sample_analysis):
        """Test research gaps dashboard creation."""
        dashboard = viz_engine.create_research_gaps_dashboard(sample_analysis)
        
        assert isinstance(dashboard, dict)
        assert 'drug_opportunities' in dashboard
        assert 'gene_opportunities' in dashboard
        assert 'methodology_radar' in dashboard
        assert 'population_map' in dashboard
        assert 'opportunity_matrix' in dashboard
    
    def test_create_collaboration_network_dashboard(self, viz_engine, sample_analysis):
        """Test collaboration network dashboard creation."""
        dashboard = viz_engine.create_collaboration_network_dashboard(sample_analysis)
        
        assert isinstance(dashboard, dict)
        assert 'author_network' in dashboard
        assert 'geographic_map' in dashboard
        assert 'institutional_chart' in dashboard
        assert 'collaboration_trends' in dashboard
        assert 'network_metrics' in dashboard
    
    def test_create_research_topics_wordcloud(self, viz_engine, sample_analysis):
        """Test research topics word cloud creation."""
        dashboard = viz_engine.create_research_topics_wordcloud(sample_analysis)
        
        assert isinstance(dashboard, dict)
        assert 'overall_topics' in dashboard
        assert 'drug_gene_cloud' in dashboard
        assert 'methodology_cloud' in dashboard
        assert 'trending_cloud' in dashboard
    
    def test_create_benchmarking_dashboard(self, viz_engine, sample_benchmark_results):
        """Test benchmarking dashboard creation."""
        dashboard = viz_engine.create_benchmarking_dashboard(sample_benchmark_results)
        
        assert isinstance(dashboard, dict)
        assert 'performance_radar' in dashboard
        assert 'percentile_chart' in dashboard
        assert 'strengths_improvements' in dashboard
        assert 'score_gauge' in dashboard
    
    def test_create_interactive_dashboard(self, viz_engine, sample_analysis, sample_benchmark_results):
        """Test interactive dashboard creation."""
        html_dashboard = viz_engine.create_interactive_dashboard(sample_analysis, sample_benchmark_results)
        
        assert isinstance(html_dashboard, str)
        assert 'html' in html_dashboard.lower()
        assert 'plotly' in html_dashboard.lower()
    
    def test_create_quality_histogram_interactive(self, viz_engine, sample_analysis):
        """Test quality histogram creation (interactive)."""
        viz_engine.config.interactive = True
        histogram = viz_engine._create_quality_histogram(sample_analysis.quality_analysis)
        
        assert isinstance(histogram, go.Figure)
    
    def test_create_quality_histogram_static(self, viz_engine, sample_analysis):
        """Test quality histogram creation (static)."""
        viz_engine.config.interactive = False
        histogram = viz_engine._create_quality_histogram(sample_analysis.quality_analysis)
        
        assert isinstance(histogram, str)  # Base64 encoded image
        assert histogram.startswith('data:image/')
    
    def test_create_quality_pie_chart(self, viz_engine, sample_analysis):
        """Test quality pie chart creation."""
        pie_chart = viz_engine._create_quality_pie_chart(sample_analysis.quality_analysis)
        
        if viz_engine.config.interactive:
            assert isinstance(pie_chart, go.Figure)
        else:
            assert isinstance(pie_chart, str)
    
    def test_create_yearly_trends_chart(self, viz_engine, sample_analysis):
        """Test yearly trends chart creation."""
        trends_chart = viz_engine._create_yearly_trends_chart(sample_analysis.trend_analysis)
        
        if viz_engine.config.interactive:
            assert isinstance(trends_chart, go.Figure)
        else:
            assert isinstance(trends_chart, str)
    
    def test_create_monthly_heatmap(self, viz_engine, sample_analysis):
        """Test monthly heatmap creation."""
        heatmap = viz_engine._create_monthly_heatmap(sample_analysis.trend_analysis)
        
        if viz_engine.config.interactive:
            assert isinstance(heatmap, go.Figure)
        else:
            assert isinstance(heatmap, str)
    
    def test_create_top_journals_chart(self, viz_engine, sample_analysis):
        """Test top journals chart creation."""
        journals_chart = viz_engine._create_top_journals_chart(sample_analysis.journal_analysis)
        
        if viz_engine.config.interactive:
            assert isinstance(journals_chart, go.Figure)
        else:
            assert isinstance(journals_chart, str)
    
    def test_create_opportunity_chart(self, viz_engine):
        """Test opportunity chart creation."""
        opportunities = [
            {"entity": "DRUG_A", "count": 5, "opportunity_score": 85},
            {"entity": "DRUG_B", "count": 8, "opportunity_score": 78}
        ]
        
        chart = viz_engine._create_opportunity_chart(opportunities, "Test Opportunities")
        
        if viz_engine.config.interactive:
            assert isinstance(chart, go.Figure)
        else:
            assert isinstance(chart, str)
    
    def test_create_empty_chart(self, viz_engine):
        """Test empty chart creation."""
        empty_chart = viz_engine._create_empty_chart("No data available")
        
        if viz_engine.config.interactive:
            assert isinstance(empty_chart, go.Figure)
        else:
            assert isinstance(empty_chart, str)
    
    def test_fig_to_base64(self, viz_engine):
        """Test matplotlib figure to base64 conversion."""
        # Create a simple matplotlib figure
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 2])
        ax.set_title("Test Figure")
        
        base64_str = viz_engine._fig_to_base64(fig)
        
        assert isinstance(base64_str, str)
        assert base64_str.startswith('data:image/')
        assert 'base64,' in base64_str
    
    def test_save_dashboard_html(self, viz_engine, tmp_path):
        """Test dashboard HTML saving."""
        dashboard_components = {
            'quality_analysis': {
                'histogram': viz_engine._create_empty_chart("Test histogram"),
                'pie_chart': viz_engine._create_empty_chart("Test pie chart")
            },
            'trends': {
                'yearly': viz_engine._create_empty_chart("Test trends")
            }
        }
        
        output_path = tmp_path / "test_dashboard.html"
        
        viz_engine.save_dashboard_html(
            dashboard_components, 
            str(output_path), 
            "Test Dashboard"
        )
        
        assert output_path.exists()
        
        # Check HTML content
        html_content = output_path.read_text(encoding='utf-8')
        assert "Test Dashboard" in html_content
        assert "quality_analysis" in html_content.lower()
        assert "trends" in html_content.lower()
    
    def test_visualization_config_defaults(self):
        """Test visualization configuration defaults."""
        config = VisualizationConfig()
        
        assert config.style == "seaborn-v0_8"
        assert config.color_palette == "viridis"
        assert config.figure_size == (12, 8)
        assert config.dpi == 300
        assert config.interactive is True
        assert config.save_format == "png"
    
    def test_visualization_config_custom(self):
        """Test custom visualization configuration."""
        config = VisualizationConfig(
            style="ggplot",
            color_palette="plasma",
            figure_size=(8, 6),
            dpi=150,
            interactive=False,
            save_format="jpg"
        )
        
        assert config.style == "ggplot"
        assert config.color_palette == "plasma"
        assert config.figure_size == (8, 6)
        assert config.dpi == 150
        assert config.interactive is False
        assert config.save_format == "jpg"
    
    def test_empty_data_handling(self, viz_engine):
        """Test handling of empty data."""
        # Empty quality stats
        empty_quality = QualityStats(0, 0, 0, {}, 0, 0, 0, 0, "unknown", [])
        histogram = viz_engine._create_quality_histogram(empty_quality)
        
        # Should handle gracefully
        assert histogram is not None
        
        # Empty trends
        empty_trends = PublicationTrends({}, {}, 0.0, 0, "unknown", {})
        trends_chart = viz_engine._create_yearly_trends_chart(empty_trends)
        
        # Should return empty chart
        assert trends_chart is not None
    
    def test_color_schemes(self, viz_engine):
        """Test color scheme usage."""
        assert 'high' in viz_engine.quality_colors
        assert 'medium' in viz_engine.quality_colors
        assert 'low' in viz_engine.quality_colors
        
        assert 'increasing' in viz_engine.trend_colors
        assert 'stable' in viz_engine.trend_colors
        assert 'decreasing' in viz_engine.trend_colors
        
        # Colors should be valid hex codes or color names
        for color in viz_engine.quality_colors.values():
            assert isinstance(color, str)
            assert len(color) > 0
    
    def test_large_dataset_handling(self, viz_engine):
        """Test handling of large datasets."""
        # Create large quality stats
        large_quality = QualityStats(
            mean=75.0,
            median=76.0,
            std_dev=15.0,
            percentiles={25: 60.0, 50: 76.0, 75: 90.0, 90: 95.0, 95: 98.0},
            total_papers=10000,  # Large dataset
            high_quality_count=3000,
            medium_quality_count=5000,
            low_quality_count=2000,
            quality_trend="improving",
            outliers=[]
        )
        
        # Should handle large datasets without issues
        histogram = viz_engine._create_quality_histogram(large_quality)
        pie_chart = viz_engine._create_quality_pie_chart(large_quality)
        
        assert histogram is not None
        assert pie_chart is not None


class TestVisualizationIntegration:
    """Integration tests for visualization components."""
    
    def test_complete_dashboard_generation(self):
        """Test complete dashboard generation pipeline."""
        viz_engine = VisualizationEngine()
        
        # Create minimal analysis data
        analysis = ComprehensiveAnalysis(
            quality_analysis=QualityStats(75.0, 76.0, 12.0, {50: 76.0}, 100, 40, 45, 15, "improving", []),
            trend_analysis=PublicationTrends({2023: 35}, {}, 15.0, 2023, "increasing", {}),
            gap_analysis=ResearchGaps([], [], [], [], [], []),
            journal_analysis=JournalStats({}, {}, [], {}, 0.7),
            collaboration_analysis=CollaborationStats({}, [], {}, {}, {}),
            insights=[],
            recommendations=[],
            field_maturity=FieldMaturity(70.0, [], [], [], {}, 65.0)
        )
        
        # Generate all dashboards
        quality_dashboard = viz_engine.create_quality_distribution_dashboard(analysis)
        trends_dashboard = viz_engine.create_publication_trends_dashboard(analysis)
        journal_dashboard = viz_engine.create_journal_analysis_dashboard(analysis)
        
        # Verify all dashboards are created
        assert isinstance(quality_dashboard, dict)
        assert isinstance(trends_dashboard, dict)
        assert isinstance(journal_dashboard, dict)
        
        # Verify dashboard components
        assert len(quality_dashboard) > 0
        assert len(trends_dashboard) > 0
        assert len(journal_dashboard) > 0
    
    def test_visualization_error_handling(self):
        """Test error handling in visualization generation."""
        viz_engine = VisualizationEngine()
        
        # Test with None data
        try:
            result = viz_engine._create_empty_chart("Test message")
            assert result is not None
        except Exception as e:
            pytest.fail(f"Empty chart creation should not raise exception: {e}")
        
        # Test with invalid data
        try:
            invalid_quality = QualityStats(-1, -1, -1, {}, -1, -1, -1, -1, "invalid", [])
            result = viz_engine._create_quality_histogram(invalid_quality)
            assert result is not None
        except Exception as e:
            pytest.fail(f"Invalid data should be handled gracefully: {e}")
    
    def test_performance_with_realistic_data(self):
        """Test performance with realistic data sizes."""
        viz_engine = VisualizationEngine()
        
        # Create realistic-sized dataset
        realistic_analysis = ComprehensiveAnalysis(
            quality_analysis=QualityStats(
                mean=72.5, median=74.0, std_dev=18.2,
                percentiles={25: 58.0, 50: 74.0, 75: 87.0, 90: 94.0, 95: 97.0},
                total_papers=1500, high_quality_count=450, medium_quality_count=750, low_quality_count=300,
                quality_trend="improving", outliers=[]
            ),
            trend_analysis=PublicationTrends(
                papers_per_year={2019: 200, 2020: 250, 2021: 300, 2022: 400, 2023: 350},
                papers_per_month={f"2023-{i:02d}": 25 + i for i in range(1, 13)},
                growth_rate=18.5, peak_year=2022, recent_trend="stable",
                seasonal_patterns={f"Month{i}": 8.0 + i*0.5 for i in range(1, 13)}
            ),
            gap_analysis=ResearchGaps(
                understudied_drugs=[{"entity": f"DRUG_{i}", "count": 10-i, "opportunity_score": 80+i} for i in range(5)],
                understudied_genes=[{"entity": f"GENE_{i}", "count": 15-i, "opportunity_score": 75+i} for i in range(3)],
                methodology_gaps=["deep learning", "ensemble methods", "transfer learning"],
                population_gaps=["Asian", "African", "Hispanic"],
                therapeutic_area_gaps=["Pediatrics", "Geriatrics"],
                quality_improvement_areas=["Clinical Relevance", "Methodology"]
            ),
            journal_analysis=JournalStats(
                journal_frequency={f"Journal {i}": 50-i*5 for i in range(10)},
                journal_quality_scores={f"Journal {i}": 85-i*2 for i in range(10)},
                top_journals=[
                    {"journal": f"Journal {i}", "paper_count": 50-i*5, "avg_quality": 85-i*2, "quality_std": 10+i}
                    for i in range(10)
                ],
                journal_specialization={f"Journal {i}": [f"Topic {j}" for j in range(2)] for i in range(5)},
                impact_correlation=0.78
            ),
            collaboration_analysis=CollaborationStats(
                collaboration_network={f"Author {i}": [f"Collab {j}" for j in range(3)] for i in range(20)},
                top_collaborators=[
                    {"author": f"Author {i}", "collaborator_count": 20-i, "collaborators": [f"C{j}" for j in range(5)]}
                    for i in range(10)
                ],
                institutional_patterns={f"Institution {i}": 30-i*2 for i in range(15)},
                geographic_distribution={"USA": 400, "Europe": 300, "Asia": 200, "Others": 100},
                collaboration_trends={2019+i: 4.0+i*0.3 for i in range(5)}
            ),
            insights=[{"type": "trend", "title": f"Insight {i}", "description": f"Description {i}"} for i in range(10)],
            recommendations=[],
            field_maturity=FieldMaturity(
                maturity_score=75.5,
                established_areas=[f"Area {i}" for i in range(8)],
                emerging_areas=[f"Emerging {i}" for i in range(5)],
                declining_areas=[f"Declining {i}" for i in range(2)],
                translation_indicators={"clinical": 30.0, "implementation": 20.0, "guidelines": 15.0},
                standardization_level=72.0
            )
        )
        
        # Test performance - should complete within reasonable time
        import time
        start_time = time.time()
        
        # Generate multiple dashboards
        quality_dashboard = viz_engine.create_quality_distribution_dashboard(realistic_analysis)
        trends_dashboard = viz_engine.create_publication_trends_dashboard(realistic_analysis)
        journal_dashboard = viz_engine.create_journal_analysis_dashboard(realistic_analysis)
        gaps_dashboard = viz_engine.create_research_gaps_dashboard(realistic_analysis)
        
        end_time = time.time()
        
        # Should complete within 30 seconds for realistic data
        assert end_time - start_time < 30
        
        # Verify all dashboards were created
        assert len(quality_dashboard) > 0
        assert len(trends_dashboard) > 0
        assert len(journal_dashboard) > 0
        assert len(gaps_dashboard) > 0