"""Tests for the report generator module."""

import pytest
import os
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, mock_open

from src.analytics.report_generator import ReportGenerator, ReportConfig, ReportMetadata
from src.analytics.research_analyzer import ComprehensiveAnalysis, QualityStats, PublicationTrends
from src.analytics.benchmarking import BenchmarkResults


class TestReportGenerator:
    """Test cases for ReportGenerator."""
    
    @pytest.fixture
    def report_config(self, tmp_path):
        """Create report configuration."""
        return ReportConfig(
            output_dir=str(tmp_path / "reports"),
            include_visualizations=True,
            organization_name="Test Organization"
        )
    
    @pytest.fixture
    def report_generator(self, report_config):
        """Create report generator instance."""
        return ReportGenerator(report_config)
    
    @pytest.fixture
    def sample_analysis(self):
        """Sample analysis for testing."""
        return ComprehensiveAnalysis(
            quality_analysis=QualityStats(75.0, 76.0, 12.0, {50: 76.0}, 100, 40, 45, 15, "improving", []),
            trend_analysis=PublicationTrends({2023: 35}, {}, 15.0, 2023, "increasing", {}),
            gap_analysis=Mock(),
            journal_analysis=Mock(),
            collaboration_analysis=Mock(),
            insights=[{"type": "test", "title": "Test Insight", "description": "Test description"}],
            recommendations=[],
            field_maturity=Mock()
        )
    
    def test_report_generator_initialization(self, report_generator):
        """Test report generator initialization."""
        assert report_generator.config is not None
        assert report_generator.visualization_engine is not None
        assert len(report_generator.templates) > 0
        assert 'executive' in report_generator.templates
        assert 'technical' in report_generator.templates
    
    @pytest.mark.asyncio
    async def test_generate_executive_summary(self, report_generator, sample_analysis):
        """Test executive summary generation."""
        with patch.object(report_generator, '_generate_html_report') as mock_html:
            with patch.object(report_generator, '_generate_json_report') as mock_json:
                mock_html.return_value = "test_exec.html"
                mock_json.return_value = "test_exec.json"
                
                reports = report_generator.generate_executive_summary(
                    sample_analysis, export_formats=['html', 'json']
                )
                
                assert isinstance(reports, dict)
                assert 'html' in reports
                assert 'json' in reports
                mock_html.assert_called_once()
                mock_json.assert_called_once()
    
    def test_export_data_csv(self, report_generator, sample_analysis, tmp_path):
        """Test CSV data export."""
        output_path = tmp_path / "test_export.csv"
        
        result_path = report_generator.export_data_csv(sample_analysis, str(output_path))
        
        assert result_path == str(output_path)
        assert output_path.exists()
        
        # Check CSV content
        csv_content = output_path.read_text()
        assert "Category" in csv_content
        assert "Quality Statistics" in csv_content
    
    def test_export_data_json(self, report_generator, sample_analysis, tmp_path):
        """Test JSON data export."""
        output_path = tmp_path / "test_export.json"
        
        result_path = report_generator.export_data_json(sample_analysis, str(output_path))
        
        assert result_path == str(output_path)
        assert output_path.exists()
        
        # Check JSON content
        with open(output_path, 'r') as f:
            json_data = json.load(f)
        
        assert 'metadata' in json_data
        assert 'quality_analysis' in json_data
        assert json_data['quality_analysis']['mean'] == 75.0