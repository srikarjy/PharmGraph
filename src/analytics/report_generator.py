"""Comprehensive reporting framework for pharmacogenomics research analytics."""

import os
import json
import csv
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import pandas as pd
from jinja2 import Template, Environment, FileSystemLoader
import pdfkit
from io import StringIO, BytesIO
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

from ..config import get_logger, get_config
from .research_analyzer import ComprehensiveAnalysis
from .benchmarking import BenchmarkResults, JournalComparison, ProductivityAnalysis, ImpactMetrics, FieldSummary
from .visualization_engine import VisualizationEngine


logger = get_logger(__name__)


@dataclass
class ReportConfig:
    """Configuration for report generation."""
    template_dir: str = "templates/reports"
    output_dir: str = "reports"
    include_visualizations: bool = True
    include_raw_data: bool = False
    logo_path: Optional[str] = None
    organization_name: str = "Pharmacogenomics Research Analytics"
    contact_email: str = "analytics@pharmgx.org"


@dataclass
class ReportMetadata:
    """Metadata for generated reports."""
    report_id: str
    report_type: str
    generated_at: datetime
    generated_by: str
    data_period: str
    total_papers: int
    analysis_version: str
    export_formats: List[str]


class ReportGenerator:
    """Comprehensive reporting system for pharmacogenomics analytics."""
    
    def __init__(self, config: Optional[ReportConfig] = None):
        """Initialize the report generator.
        
        Args:
            config: Report generation configuration
        """
        self.config = config or ReportConfig()
        self.visualization_engine = VisualizationEngine()
        
        # Ensure output directory exists
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Setup Jinja2 environment
        self.jinja_env = Environment(
            loader=FileSystemLoader(self.config.template_dir),
            autoescape=True
        )
        
        # Report templates
        self.templates = {
            'executive': 'executive_summary.html',
            'technical': 'technical_report.html',
            'quality': 'quality_assessment.html',
            'benchmarking': 'benchmarking_report.html',
            'email_summary': 'email_summary.html'
        }
        
        logger.info("ReportGenerator initialized")
    
    def generate_executive_summary(self, analysis: ComprehensiveAnalysis,
                                 benchmark_results: Optional[BenchmarkResults] = None,
                                 export_formats: List[str] = ['html', 'pdf']) -> Dict[str, str]:
        """Generate executive summary report.
        
        Args:
            analysis: Complete research analysis results
            benchmark_results: Optional benchmarking results
            export_formats: List of export formats ('html', 'pdf', 'json')
            
        Returns:
            Dict mapping format to file path
        """
        logger.info("Generating executive summary report")
        
        # Prepare report data
        report_data = self._prepare_executive_data(analysis, benchmark_results)
        
        # Generate report metadata
        metadata = ReportMetadata(
            report_id=f"exec_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            report_type="Executive Summary",
            generated_at=datetime.now(),
            generated_by="Analytics System",
            data_period=self._get_data_period(analysis),
            total_papers=analysis.quality_analysis.total_papers,
            analysis_version="1.0",
            export_formats=export_formats
        )
        
        # Generate visualizations if requested
        visualizations = {}
        if self.config.include_visualizations:
            visualizations = self._generate_executive_visualizations(analysis, benchmark_results)
        
        # Combine all data
        template_data = {
            'metadata': metadata,
            'analysis': report_data,
            'visualizations': visualizations,
            'config': self.config
        }
        
        # Generate reports in requested formats
        output_files = {}
        
        for format_type in export_formats:
            if format_type == 'html':
                output_files['html'] = self._generate_html_report(
                    template_data, 'executive', metadata.report_id
                )
            elif format_type == 'pdf':
                output_files['pdf'] = self._generate_pdf_report(
                    template_data, 'executive', metadata.report_id
                )
            elif format_type == 'json':
                output_files['json'] = self._generate_json_report(
                    template_data, metadata.report_id
                )
        
        logger.info(f"Executive summary generated: {list(output_files.keys())}")
        return output_files
    
    def generate_technical_report(self, analysis: ComprehensiveAnalysis,
                                benchmark_results: Optional[BenchmarkResults] = None,
                                journal_comparison: Optional[JournalComparison] = None,
                                productivity_analysis: Optional[ProductivityAnalysis] = None,
                                export_formats: List[str] = ['html', 'pdf']) -> Dict[str, str]:
        """Generate comprehensive technical report.
        
        Args:
            analysis: Complete research analysis results
            benchmark_results: Optional benchmarking results
            journal_comparison: Optional journal comparison results
            productivity_analysis: Optional productivity analysis results
            export_formats: List of export formats
            
        Returns:
            Dict mapping format to file path
        """
        logger.info("Generating technical report")
        
        # Prepare comprehensive technical data
        report_data = self._prepare_technical_data(
            analysis, benchmark_results, journal_comparison, productivity_analysis
        )
        
        # Generate metadata
        metadata = ReportMetadata(
            report_id=f"tech_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            report_type="Technical Report",
            generated_at=datetime.now(),
            generated_by="Analytics System",
            data_period=self._get_data_period(analysis),
            total_papers=analysis.quality_analysis.total_papers,
            analysis_version="1.0",
            export_formats=export_formats
        )
        
        # Generate comprehensive visualizations
        visualizations = {}
        if self.config.include_visualizations:
            visualizations = self._generate_technical_visualizations(
                analysis, benchmark_results, journal_comparison
            )
        
        # Combine all data
        template_data = {
            'metadata': metadata,
            'analysis': report_data,
            'visualizations': visualizations,
            'config': self.config
        }
        
        # Generate reports
        output_files = {}
        
        for format_type in export_formats:
            if format_type == 'html':
                output_files['html'] = self._generate_html_report(
                    template_data, 'technical', metadata.report_id
                )
            elif format_type == 'pdf':
                output_files['pdf'] = self._generate_pdf_report(
                    template_data, 'technical', metadata.report_id
                )
            elif format_type == 'json':
                output_files['json'] = self._generate_json_report(
                    template_data, metadata.report_id
                )
            elif format_type == 'excel':
                output_files['excel'] = self._generate_excel_report(
                    template_data, metadata.report_id
                )
        
        logger.info(f"Technical report generated: {list(output_files.keys())}")
        return output_files
    
    def generate_quality_assessment_report(self, analysis: ComprehensiveAnalysis,
                                         export_formats: List[str] = ['html', 'pdf']) -> Dict[str, str]:
        """Generate quality assessment report.
        
        Args:
            analysis: Complete research analysis results
            export_formats: List of export formats
            
        Returns:
            Dict mapping format to file path
        """
        logger.info("Generating quality assessment report")
        
        # Prepare quality-focused data
        report_data = self._prepare_quality_data(analysis)
        
        # Generate metadata
        metadata = ReportMetadata(
            report_id=f"qual_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            report_type="Quality Assessment",
            generated_at=datetime.now(),
            generated_by="Analytics System",
            data_period=self._get_data_period(analysis),
            total_papers=analysis.quality_analysis.total_papers,
            analysis_version="1.0",
            export_formats=export_formats
        )
        
        # Generate quality-focused visualizations
        visualizations = {}
        if self.config.include_visualizations:
            visualizations = self._generate_quality_visualizations(analysis)
        
        # Combine data
        template_data = {
            'metadata': metadata,
            'analysis': report_data,
            'visualizations': visualizations,
            'config': self.config
        }
        
        # Generate reports
        output_files = {}
        
        for format_type in export_formats:
            if format_type == 'html':
                output_files['html'] = self._generate_html_report(
                    template_data, 'quality', metadata.report_id
                )
            elif format_type == 'pdf':
                output_files['pdf'] = self._generate_pdf_report(
                    template_data, 'quality', metadata.report_id
                )
            elif format_type == 'csv':
                output_files['csv'] = self._generate_csv_report(
                    template_data, metadata.report_id
                )
        
        logger.info(f"Quality assessment report generated: {list(output_files.keys())}")
        return output_files
    
    def generate_email_summary(self, analysis: ComprehensiveAnalysis,
                             benchmark_results: Optional[BenchmarkResults] = None,
                             recipient_type: str = 'stakeholder') -> str:
        """Generate email-ready summary report.
        
        Args:
            analysis: Complete research analysis results
            benchmark_results: Optional benchmarking results
            recipient_type: Type of recipient ('stakeholder', 'researcher', 'executive')
            
        Returns:
            HTML string ready for email
        """
        logger.info(f"Generating email summary for {recipient_type}")
        
        # Prepare email-specific data
        report_data = self._prepare_email_data(analysis, benchmark_results, recipient_type)
        
        # Generate compact visualizations
        visualizations = self._generate_email_visualizations(analysis, benchmark_results)
        
        # Prepare template data
        template_data = {
            'recipient_type': recipient_type,
            'analysis': report_data,
            'visualizations': visualizations,
            'config': self.config,
            'generated_at': datetime.now()
        }
        
        # Render email template
        template = self.jinja_env.get_template(self.templates['email_summary'])
        html_content = template.render(**template_data)
        
        return html_content
    
    def send_email_report(self, html_content: str, recipients: List[str],
                         subject: str, attachments: Optional[List[str]] = None,
                         smtp_config: Optional[Dict[str, Any]] = None) -> bool:
        """Send email report to recipients.
        
        Args:
            html_content: HTML content of the email
            recipients: List of recipient email addresses
            subject: Email subject
            attachments: Optional list of file paths to attach
            smtp_config: SMTP configuration (host, port, username, password)
            
        Returns:
            True if email sent successfully
        """
        logger.info(f"Sending email report to {len(recipients)} recipients")
        
        try:
            # Use provided SMTP config or get from system config
            if not smtp_config:
                config = get_config()
                smtp_config = {
                    'host': getattr(config, 'smtp_host', 'localhost'),
                    'port': getattr(config, 'smtp_port', 587),
                    'username': getattr(config, 'smtp_username', ''),
                    'password': getattr(config, 'smtp_password', ''),
                    'use_tls': getattr(config, 'smtp_use_tls', True)
                }
            
            # Create message
            msg = MIMEMultipart('alternative')
            msg['From'] = self.config.contact_email
            msg['To'] = ', '.join(recipients)
            msg['Subject'] = subject
            
            # Add HTML content
            html_part = MIMEText(html_content, 'html')
            msg.attach(html_part)
            
            # Add attachments if provided
            if attachments:
                for file_path in attachments:
                    if os.path.exists(file_path):
                        with open(file_path, 'rb') as attachment:
                            part = MIMEBase('application', 'octet-stream')
                            part.set_payload(attachment.read())
                        
                        encoders.encode_base64(part)
                        part.add_header(
                            'Content-Disposition',
                            f'attachment; filename= {os.path.basename(file_path)}'
                        )
                        msg.attach(part)
            
            # Send email
            server = smtplib.SMTP(smtp_config['host'], smtp_config['port'])
            if smtp_config.get('use_tls'):
                server.starttls()
            
            if smtp_config.get('username'):
                server.login(smtp_config['username'], smtp_config['password'])
            
            server.sendmail(self.config.contact_email, recipients, msg.as_string())
            server.quit()
            
            logger.info("Email report sent successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email report: {str(e)}")
            return False
    
    def export_data_csv(self, analysis: ComprehensiveAnalysis, output_path: str) -> str:
        """Export analysis data to CSV format.
        
        Args:
            analysis: Complete research analysis results
            output_path: Path to save CSV file
            
        Returns:
            Path to saved CSV file
        """
        logger.info("Exporting data to CSV")
        
        # Prepare data for CSV export
        data_rows = []
        
        # Quality statistics
        quality_stats = analysis.quality_analysis
        data_rows.append({
            'Category': 'Quality Statistics',
            'Metric': 'Mean Score',
            'Value': quality_stats.mean,
            'Description': 'Average quality score across all papers'
        })
        data_rows.append({
            'Category': 'Quality Statistics',
            'Metric': 'Median Score',
            'Value': quality_stats.median,
            'Description': 'Median quality score'
        })
        data_rows.append({
            'Category': 'Quality Statistics',
            'Metric': 'Standard Deviation',
            'Value': quality_stats.std_dev,
            'Description': 'Quality score standard deviation'
        })
        data_rows.append({
            'Category': 'Quality Statistics',
            'Metric': 'High Quality Papers',
            'Value': quality_stats.high_quality_count,
            'Description': 'Papers with quality score ≥75'
        })
        
        # Publication trends
        trend_stats = analysis.trend_analysis
        data_rows.append({
            'Category': 'Publication Trends',
            'Metric': 'Growth Rate',
            'Value': trend_stats.growth_rate,
            'Description': 'Annual publication growth rate (%)'
        })
        data_rows.append({
            'Category': 'Publication Trends',
            'Metric': 'Peak Year',
            'Value': trend_stats.peak_year,
            'Description': 'Year with highest publication count'
        })
        
        # Research gaps
        gaps = analysis.gap_analysis
        data_rows.append({
            'Category': 'Research Gaps',
            'Metric': 'Understudied Drugs',
            'Value': len(gaps.understudied_drugs),
            'Description': 'Number of understudied drug entities'
        })
        data_rows.append({
            'Category': 'Research Gaps',
            'Metric': 'Understudied Genes',
            'Value': len(gaps.understudied_genes),
            'Description': 'Number of understudied gene entities'
        })
        
        # Field maturity
        maturity = analysis.field_maturity
        data_rows.append({
            'Category': 'Field Maturity',
            'Metric': 'Maturity Score',
            'Value': maturity.maturity_score,
            'Description': 'Overall field maturity score (0-100)'
        })
        data_rows.append({
            'Category': 'Field Maturity',
            'Metric': 'Standardization Level',
            'Value': maturity.standardization_level,
            'Description': 'Field standardization level (%)'
        })
        
        # Save to CSV
        df = pd.DataFrame(data_rows)
        df.to_csv(output_path, index=False)
        
        logger.info(f"Data exported to CSV: {output_path}")
        return output_path
    
    def export_data_json(self, analysis: ComprehensiveAnalysis, output_path: str) -> str:
        """Export analysis data to JSON format.
        
        Args:
            analysis: Complete research analysis results
            output_path: Path to save JSON file
            
        Returns:
            Path to saved JSON file
        """
        logger.info("Exporting data to JSON")
        
        # Convert analysis to dictionary
        data = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'analysis_version': '1.0',
                'total_papers': analysis.quality_analysis.total_papers
            },
            'quality_analysis': {
                'mean': analysis.quality_analysis.mean,
                'median': analysis.quality_analysis.median,
                'std_dev': analysis.quality_analysis.std_dev,
                'percentiles': analysis.quality_analysis.percentiles,
                'total_papers': analysis.quality_analysis.total_papers,
                'high_quality_count': analysis.quality_analysis.high_quality_count,
                'medium_quality_count': analysis.quality_analysis.medium_quality_count,
                'low_quality_count': analysis.quality_analysis.low_quality_count,
                'quality_trend': analysis.quality_analysis.quality_trend
            },
            'publication_trends': {
                'papers_per_year': analysis.trend_analysis.papers_per_year,
                'growth_rate': analysis.trend_analysis.growth_rate,
                'peak_year': analysis.trend_analysis.peak_year,
                'recent_trend': analysis.trend_analysis.recent_trend,
                'seasonal_patterns': analysis.trend_analysis.seasonal_patterns
            },
            'research_gaps': {
                'understudied_drugs': analysis.gap_analysis.understudied_drugs,
                'understudied_genes': analysis.gap_analysis.understudied_genes,
                'methodology_gaps': analysis.gap_analysis.methodology_gaps,
                'population_gaps': analysis.gap_analysis.population_gaps,
                'therapeutic_area_gaps': analysis.gap_analysis.therapeutic_area_gaps
            },
            'journal_analysis': {
                'journal_frequency': analysis.journal_analysis.journal_frequency,
                'journal_quality_scores': analysis.journal_analysis.journal_quality_scores,
                'top_journals': analysis.journal_analysis.top_journals,
                'impact_correlation': analysis.journal_analysis.impact_correlation
            },
            'field_maturity': {
                'maturity_score': analysis.field_maturity.maturity_score,
                'established_areas': analysis.field_maturity.established_areas,
                'emerging_areas': analysis.field_maturity.emerging_areas,
                'standardization_level': analysis.field_maturity.standardization_level
            },
            'insights': analysis.insights,
            'recommendations': [asdict(rec) for rec in analysis.recommendations]
        }
        
        # Save to JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Data exported to JSON: {output_path}")
        return output_path
    
    # Private helper methods
    
    def _prepare_executive_data(self, analysis: ComprehensiveAnalysis,
                              benchmark_results: Optional[BenchmarkResults]) -> Dict[str, Any]:
        """Prepare data for executive summary."""
        quality_stats = analysis.quality_analysis
        trend_stats = analysis.trend_analysis
        
        # Key metrics for executives
        key_metrics = {
            'total_papers': quality_stats.total_papers,
            'avg_quality_score': quality_stats.mean,
            'quality_trend': quality_stats.quality_trend,
            'growth_rate': trend_stats.growth_rate,
            'high_quality_percentage': (quality_stats.high_quality_count / 
                                      max(1, quality_stats.total_papers)) * 100,
            'field_maturity': analysis.field_maturity.maturity_score
        }
        
        # Top insights (limit to 5 for executive summary)
        top_insights = analysis.insights[:5]
        
        # High-priority recommendations
        high_priority_recs = [rec for rec in analysis.recommendations if rec.priority == 'high'][:3]
        
        # Benchmarking summary if available
        benchmark_summary = None
        if benchmark_results:
            benchmark_summary = {
                'overall_score': benchmark_results.overall_score,
                'strengths': benchmark_results.strengths[:3],
                'improvement_areas': benchmark_results.improvement_areas[:3]
            }
        
        return {
            'key_metrics': key_metrics,
            'insights': top_insights,
            'recommendations': high_priority_recs,
            'benchmark_summary': benchmark_summary,
            'research_opportunities': analysis.gap_analysis.understudied_drugs[:3]
        }
    
    def _prepare_technical_data(self, analysis: ComprehensiveAnalysis,
                              benchmark_results: Optional[BenchmarkResults],
                              journal_comparison: Optional[JournalComparison],
                              productivity_analysis: Optional[ProductivityAnalysis]) -> Dict[str, Any]:
        """Prepare comprehensive technical data."""
        return {
            'quality_analysis': analysis.quality_analysis,
            'trend_analysis': analysis.trend_analysis,
            'gap_analysis': analysis.gap_analysis,
            'journal_analysis': analysis.journal_analysis,
            'collaboration_analysis': analysis.collaboration_analysis,
            'field_maturity': analysis.field_maturity,
            'insights': analysis.insights,
            'recommendations': analysis.recommendations,
            'benchmark_results': benchmark_results,
            'journal_comparison': journal_comparison,
            'productivity_analysis': productivity_analysis
        }
    
    def _prepare_quality_data(self, analysis: ComprehensiveAnalysis) -> Dict[str, Any]:
        """Prepare quality-focused data."""
        return {
            'quality_statistics': analysis.quality_analysis,
            'quality_trends': analysis.trend_analysis,
            'quality_by_journal': analysis.journal_analysis.journal_quality_scores,
            'quality_improvement_areas': analysis.gap_analysis.quality_improvement_areas,
            'quality_insights': [insight for insight in analysis.insights 
                               if 'quality' in insight.get('type', '').lower()]
        }
    
    def _prepare_email_data(self, analysis: ComprehensiveAnalysis,
                          benchmark_results: Optional[BenchmarkResults],
                          recipient_type: str) -> Dict[str, Any]:
        """Prepare email-specific data based on recipient type."""
        base_data = {
            'total_papers': analysis.quality_analysis.total_papers,
            'avg_quality': analysis.quality_analysis.mean,
            'growth_rate': analysis.trend_analysis.growth_rate
        }
        
        if recipient_type == 'executive':
            return {
                **base_data,
                'key_insights': analysis.insights[:2],
                'top_opportunities': analysis.gap_analysis.understudied_drugs[:2],
                'performance_score': benchmark_results.overall_score if benchmark_results else None
            }
        elif recipient_type == 'researcher':
            return {
                **base_data,
                'methodology_gaps': analysis.gap_analysis.methodology_gaps[:3],
                'research_opportunities': analysis.gap_analysis.understudied_drugs[:3],
                'collaboration_insights': analysis.collaboration_analysis.top_collaborators[:3]
            }
        else:  # stakeholder
            return {
                **base_data,
                'field_maturity': analysis.field_maturity.maturity_score,
                'top_journals': analysis.journal_analysis.top_journals[:3],
                'recommendations': [rec for rec in analysis.recommendations if rec.priority == 'high'][:2]
            }
    
    def _get_data_period(self, analysis: ComprehensiveAnalysis) -> str:
        """Get data period string from analysis."""
        if analysis.trend_analysis.papers_per_year:
            years = sorted(analysis.trend_analysis.papers_per_year.keys())
            return f"{years[0]}-{years[-1]}"
        return "Unknown period"
    
    def _generate_executive_visualizations(self, analysis: ComprehensiveAnalysis,
                                         benchmark_results: Optional[BenchmarkResults]) -> Dict[str, Any]:
        """Generate visualizations for executive summary."""
        viz = {}
        
        # Quality distribution pie chart
        viz['quality_distribution'] = self.visualization_engine.create_quality_distribution_dashboard(analysis)['pie_chart']
        
        # Publication trends
        viz['publication_trends'] = self.visualization_engine.create_publication_trends_dashboard(analysis)['yearly_trends']
        
        # Performance radar if benchmarking available
        if benchmark_results:
            viz['performance_radar'] = self.visualization_engine.create_benchmarking_dashboard(benchmark_results)['performance_radar']
        
        return viz
    
    def _generate_technical_visualizations(self, analysis: ComprehensiveAnalysis,
                                         benchmark_results: Optional[BenchmarkResults],
                                         journal_comparison: Optional[JournalComparison]) -> Dict[str, Any]:
        """Generate comprehensive technical visualizations."""
        viz = {}
        
        # All quality visualizations
        viz.update(self.visualization_engine.create_quality_distribution_dashboard(analysis))
        
        # All trend visualizations
        viz.update(self.visualization_engine.create_publication_trends_dashboard(analysis))
        
        # Journal analysis
        viz.update(self.visualization_engine.create_journal_analysis_dashboard(analysis))
        
        # Research gaps
        viz.update(self.visualization_engine.create_research_gaps_dashboard(analysis))
        
        # Collaboration networks
        viz.update(self.visualization_engine.create_collaboration_network_dashboard(analysis))
        
        return viz
    
    def _generate_quality_visualizations(self, analysis: ComprehensiveAnalysis) -> Dict[str, Any]:
        """Generate quality-focused visualizations."""
        return self.visualization_engine.create_quality_distribution_dashboard(analysis)
    
    def _generate_email_visualizations(self, analysis: ComprehensiveAnalysis,
                                     benchmark_results: Optional[BenchmarkResults]) -> Dict[str, Any]:
        """Generate compact visualizations for email."""
        viz = {}
        
        # Single key chart for email
        viz['key_chart'] = self.visualization_engine.create_quality_distribution_dashboard(analysis)['histogram']
        
        return viz
    
    def _generate_html_report(self, template_data: Dict[str, Any], 
                            template_type: str, report_id: str) -> str:
        """Generate HTML report."""
        template = self.jinja_env.get_template(self.templates[template_type])
        html_content = template.render(**template_data)
        
        output_path = os.path.join(self.config.output_dir, f"{report_id}.html")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return output_path
    
    def _generate_pdf_report(self, template_data: Dict[str, Any], 
                           template_type: str, report_id: str) -> str:
        """Generate PDF report."""
        # First generate HTML
        html_path = self._generate_html_report(template_data, template_type, f"{report_id}_temp")
        
        # Convert to PDF
        output_path = os.path.join(self.config.output_dir, f"{report_id}.pdf")
        
        try:
            # Configure PDF options
            options = {
                'page-size': 'A4',
                'margin-top': '0.75in',
                'margin-right': '0.75in',
                'margin-bottom': '0.75in',
                'margin-left': '0.75in',
                'encoding': "UTF-8",
                'no-outline': None,
                'enable-local-file-access': None
            }
            
            pdfkit.from_file(html_path, output_path, options=options)
            
            # Clean up temporary HTML file
            os.remove(html_path)
            
        except Exception as e:
            logger.error(f"PDF generation failed: {str(e)}")
            # Return HTML path as fallback
            return html_path
        
        return output_path
    
    def _generate_json_report(self, template_data: Dict[str, Any], report_id: str) -> str:
        """Generate JSON report."""
        output_path = os.path.join(self.config.output_dir, f"{report_id}.json")
        
        # Convert template data to JSON-serializable format
        json_data = self._convert_to_json_serializable(template_data)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False, default=str)
        
        return output_path
    
    def _generate_excel_report(self, template_data: Dict[str, Any], report_id: str) -> str:
        """Generate Excel report."""
        output_path = os.path.join(self.config.output_dir, f"{report_id}.xlsx")
        
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Quality statistics sheet
            if 'analysis' in template_data and hasattr(template_data['analysis'], 'quality_analysis'):
                quality_data = self._prepare_quality_excel_data(template_data['analysis'])
                quality_df = pd.DataFrame(quality_data)
                quality_df.to_excel(writer, sheet_name='Quality Analysis', index=False)
            
            # Publication trends sheet
            if 'analysis' in template_data and hasattr(template_data['analysis'], 'trend_analysis'):
                trend_data = self._prepare_trend_excel_data(template_data['analysis'])
                trend_df = pd.DataFrame(trend_data)
                trend_df.to_excel(writer, sheet_name='Publication Trends', index=False)
            
            # Research gaps sheet
            if 'analysis' in template_data and hasattr(template_data['analysis'], 'gap_analysis'):
                gaps_data = self._prepare_gaps_excel_data(template_data['analysis'])
                gaps_df = pd.DataFrame(gaps_data)
                gaps_df.to_excel(writer, sheet_name='Research Gaps', index=False)
        
        return output_path
    
    def _generate_csv_report(self, template_data: Dict[str, Any], report_id: str) -> str:
        """Generate CSV report."""
        output_path = os.path.join(self.config.output_dir, f"{report_id}.csv")
        
        # Flatten data for CSV
        csv_data = self._flatten_data_for_csv(template_data)
        
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            if csv_data:
                fieldnames = csv_data[0].keys()
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(csv_data)
        
        return output_path
    
    def _convert_to_json_serializable(self, data: Any) -> Any:
        """Convert data to JSON-serializable format."""
        if hasattr(data, '__dict__'):
            return {k: self._convert_to_json_serializable(v) for k, v in data.__dict__.items()}
        elif isinstance(data, dict):
            return {k: self._convert_to_json_serializable(v) for k, v in data.items()}
        elif isinstance(data, (list, tuple)):
            return [self._convert_to_json_serializable(item) for item in data]
        elif isinstance(data, datetime):
            return data.isoformat()
        else:
            return data
    
    def _prepare_quality_excel_data(self, analysis) -> List[Dict[str, Any]]:
        """Prepare quality data for Excel export."""
        return [
            {'Metric': 'Mean Quality Score', 'Value': analysis.quality_analysis.mean},
            {'Metric': 'Median Quality Score', 'Value': analysis.quality_analysis.median},
            {'Metric': 'Standard Deviation', 'Value': analysis.quality_analysis.std_dev},
            {'Metric': 'Total Papers', 'Value': analysis.quality_analysis.total_papers},
            {'Metric': 'High Quality Papers', 'Value': analysis.quality_analysis.high_quality_count},
            {'Metric': 'Medium Quality Papers', 'Value': analysis.quality_analysis.medium_quality_count},
            {'Metric': 'Low Quality Papers', 'Value': analysis.quality_analysis.low_quality_count}
        ]
    
    def _prepare_trend_excel_data(self, analysis) -> List[Dict[str, Any]]:
        """Prepare trend data for Excel export."""
        data = []
        for year, count in analysis.trend_analysis.papers_per_year.items():
            data.append({'Year': year, 'Papers': count})
        return data
    
    def _prepare_gaps_excel_data(self, analysis) -> List[Dict[str, Any]]:
        """Prepare gaps data for Excel export."""
        data = []
        
        for drug in analysis.gap_analysis.understudied_drugs:
            data.append({
                'Type': 'Drug',
                'Entity': drug['entity'],
                'Current Papers': drug['count'],
                'Opportunity Score': drug['opportunity_score']
            })
        
        for gene in analysis.gap_analysis.understudied_genes:
            data.append({
                'Type': 'Gene',
                'Entity': gene['entity'],
                'Current Papers': gene['count'],
                'Opportunity Score': gene['opportunity_score']
            })
        
        return data
    
    def _flatten_data_for_csv(self, template_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Flatten complex data structure for CSV export."""
        flattened = []
        
        # Extract key metrics
        if 'analysis' in template_data:
            analysis = template_data['analysis']
            
            if hasattr(analysis, 'quality_analysis'):
                flattened.append({
                    'Category': 'Quality',
                    'Metric': 'Mean Score',
                    'Value': analysis.quality_analysis.mean
                })
                flattened.append({
                    'Category': 'Quality',
                    'Metric': 'Total Papers',
                    'Value': analysis.quality_analysis.total_papers
                })
            
            if hasattr(analysis, 'trend_analysis'):
                flattened.append({
                    'Category': 'Trends',
                    'Metric': 'Growth Rate',
                    'Value': analysis.trend_analysis.growth_rate
                })
        
        return flattened