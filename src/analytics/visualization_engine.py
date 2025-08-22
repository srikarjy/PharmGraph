"""Visualization engine for pharmacogenomics research analytics."""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import networkx as nx
from wordcloud import WordCloud
import io
import base64

from ..config import get_logger
from .research_analyzer import ComprehensiveAnalysis, QualityStats, PublicationTrends
from .benchmarking import BenchmarkResults, JournalComparison


logger = get_logger(__name__)


@dataclass
class VisualizationConfig:
    """Configuration for visualization generation."""
    style: str = "seaborn-v0_8"  # matplotlib style
    color_palette: str = "viridis"  # color scheme
    figure_size: Tuple[int, int] = (12, 8)  # default figure size
    dpi: int = 300  # resolution for saved figures
    interactive: bool = True  # use plotly for interactive charts
    save_format: str = "png"  # default save format


class VisualizationEngine:
    """Comprehensive visualization engine for pharmacogenomics analytics."""
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        """Initialize the visualization engine.
        
        Args:
            config: Visualization configuration options
        """
        self.config = config or VisualizationConfig()
        
        # Set matplotlib style
        plt.style.use(self.config.style)
        sns.set_palette(self.config.color_palette)
        
        # Color schemes for different chart types
        self.quality_colors = {
            'high': '#2E8B57',    # Sea Green
            'medium': '#FFD700',  # Gold
            'low': '#DC143C'      # Crimson
        }
        
        self.trend_colors = {
            'increasing': '#32CD32',  # Lime Green
            'stable': '#4169E1',      # Royal Blue
            'decreasing': '#FF6347'   # Tomato
        }
        
        logger.info("VisualizationEngine initialized")
    
    def create_quality_distribution_dashboard(self, analysis: ComprehensiveAnalysis) -> Dict[str, Any]:
        """Create comprehensive quality distribution dashboard.
        
        Args:
            analysis: Complete research analysis results
            
        Returns:
            Dict containing all quality visualization components
        """
        logger.info("Creating quality distribution dashboard")
        
        dashboard = {}
        quality_stats = analysis.quality_analysis
        
        # 1. Quality Score Histogram
        dashboard['histogram'] = self._create_quality_histogram(quality_stats)
        
        # 2. Quality Distribution Pie Chart
        dashboard['pie_chart'] = self._create_quality_pie_chart(quality_stats)
        
        # 3. Quality Percentile Box Plot
        dashboard['box_plot'] = self._create_quality_box_plot(quality_stats)
        
        # 4. Quality Trend Over Time
        dashboard['trend_chart'] = self._create_quality_trend_chart(quality_stats)
        
        # 5. Quality vs Journal Impact Scatter
        dashboard['scatter_plot'] = self._create_quality_journal_scatter(analysis)
        
        return dashboard
    
    def create_publication_trends_dashboard(self, analysis: ComprehensiveAnalysis) -> Dict[str, Any]:
        """Create publication trends visualization dashboard.
        
        Args:
            analysis: Complete research analysis results
            
        Returns:
            Dict containing all trend visualization components
        """
        logger.info("Creating publication trends dashboard")
        
        dashboard = {}
        trend_stats = analysis.trend_analysis
        
        # 1. Papers per Year Line Chart
        dashboard['yearly_trends'] = self._create_yearly_trends_chart(trend_stats)
        
        # 2. Monthly Publication Patterns Heatmap
        dashboard['monthly_heatmap'] = self._create_monthly_heatmap(trend_stats)
        
        # 3. Growth Rate Analysis
        dashboard['growth_analysis'] = self._create_growth_analysis_chart(trend_stats)
        
        # 4. Seasonal Patterns Radar Chart
        dashboard['seasonal_radar'] = self._create_seasonal_radar_chart(trend_stats)
        
        # 5. Publication Velocity Dashboard
        dashboard['velocity_chart'] = self._create_publication_velocity_chart(trend_stats)
        
        return dashboard
    
    def create_journal_analysis_dashboard(self, analysis: ComprehensiveAnalysis) -> Dict[str, Any]:
        """Create journal analysis visualization dashboard.
        
        Args:
            analysis: Complete research analysis results
            
        Returns:
            Dict containing all journal visualization components
        """
        logger.info("Creating journal analysis dashboard")
        
        dashboard = {}
        journal_stats = analysis.journal_analysis
        
        # 1. Top Journals Bar Chart
        dashboard['top_journals'] = self._create_top_journals_chart(journal_stats)
        
        # 2. Journal Quality vs Volume Bubble Chart
        dashboard['quality_volume_bubble'] = self._create_journal_bubble_chart(journal_stats)
        
        # 3. Journal Specialization Network
        dashboard['specialization_network'] = self._create_journal_network(journal_stats)
        
        # 4. Journal Impact Distribution
        dashboard['impact_distribution'] = self._create_journal_impact_distribution(journal_stats)
        
        # 5. Journal Quality Comparison
        dashboard['quality_comparison'] = self._create_journal_quality_comparison(journal_stats)
        
        return dashboard
    
    def create_research_gaps_dashboard(self, analysis: ComprehensiveAnalysis) -> Dict[str, Any]:
        """Create research gaps visualization dashboard.
        
        Args:
            analysis: Complete research analysis results
            
        Returns:
            Dict containing all research gaps visualization components
        """
        logger.info("Creating research gaps dashboard")
        
        dashboard = {}
        gaps = analysis.gap_analysis
        
        # 1. Understudied Drugs Opportunity Chart
        dashboard['drug_opportunities'] = self._create_opportunity_chart(
            gaps.understudied_drugs, "Understudied Drugs"
        )
        
        # 2. Understudied Genes Chart
        dashboard['gene_opportunities'] = self._create_opportunity_chart(
            gaps.understudied_genes, "Understudied Genes"
        )
        
        # 3. Methodology Gaps Radar
        dashboard['methodology_radar'] = self._create_methodology_gaps_radar(gaps)
        
        # 4. Population Gaps Geographic Map
        dashboard['population_map'] = self._create_population_gaps_map(gaps)
        
        # 5. Research Opportunity Matrix
        dashboard['opportunity_matrix'] = self._create_opportunity_matrix(gaps)
        
        return dashboard
    
    def create_collaboration_network_dashboard(self, analysis: ComprehensiveAnalysis) -> Dict[str, Any]:
        """Create collaboration network visualization dashboard.
        
        Args:
            analysis: Complete research analysis results
            
        Returns:
            Dict containing all collaboration visualization components
        """
        logger.info("Creating collaboration network dashboard")
        
        dashboard = {}
        collab_stats = analysis.collaboration_analysis
        
        # 1. Author Collaboration Network
        dashboard['author_network'] = self._create_author_network(collab_stats)
        
        # 2. Geographic Collaboration Map
        dashboard['geographic_map'] = self._create_geographic_collaboration_map(collab_stats)
        
        # 3. Institutional Collaboration Chart
        dashboard['institutional_chart'] = self._create_institutional_collaboration_chart(collab_stats)
        
        # 4. Collaboration Trends Over Time
        dashboard['collaboration_trends'] = self._create_collaboration_trends_chart(collab_stats)
        
        # 5. Network Metrics Dashboard
        dashboard['network_metrics'] = self._create_network_metrics_dashboard(collab_stats)
        
        return dashboard
    
    def create_research_topics_wordcloud(self, analysis: ComprehensiveAnalysis) -> Dict[str, Any]:
        """Create research topics word cloud visualizations.
        
        Args:
            analysis: Complete research analysis results
            
        Returns:
            Dict containing word cloud visualizations
        """
        logger.info("Creating research topics word clouds")
        
        dashboard = {}
        
        # 1. Overall Research Topics Word Cloud
        dashboard['overall_topics'] = self._create_topics_wordcloud(analysis)
        
        # 2. Drug-Gene Interaction Word Cloud
        dashboard['drug_gene_cloud'] = self._create_drug_gene_wordcloud(analysis)
        
        # 3. Methodology Word Cloud
        dashboard['methodology_cloud'] = self._create_methodology_wordcloud(analysis)
        
        # 4. Trending Topics Cloud
        dashboard['trending_cloud'] = self._create_trending_topics_wordcloud(analysis)
        
        return dashboard
    
    def create_benchmarking_dashboard(self, benchmark_results: BenchmarkResults) -> Dict[str, Any]:
        """Create benchmarking visualization dashboard.
        
        Args:
            benchmark_results: Benchmarking analysis results
            
        Returns:
            Dict containing all benchmarking visualization components
        """
        logger.info("Creating benchmarking dashboard")
        
        dashboard = {}
        
        # 1. Performance Radar Chart
        dashboard['performance_radar'] = self._create_performance_radar(benchmark_results)
        
        # 2. Percentile Rankings Chart
        dashboard['percentile_chart'] = self._create_percentile_chart(benchmark_results)
        
        # 3. Strengths vs Improvements
        dashboard['strengths_improvements'] = self._create_strengths_improvements_chart(benchmark_results)
        
        # 4. Overall Score Gauge
        dashboard['score_gauge'] = self._create_score_gauge(benchmark_results)
        
        return dashboard
    
    def create_interactive_dashboard(self, analysis: ComprehensiveAnalysis, 
                                   benchmark_results: Optional[BenchmarkResults] = None) -> str:
        """Create comprehensive interactive dashboard.
        
        Args:
            analysis: Complete research analysis results
            benchmark_results: Optional benchmarking results
            
        Returns:
            HTML string containing the interactive dashboard
        """
        logger.info("Creating comprehensive interactive dashboard")
        
        # Create subplot structure
        fig = make_subplots(
            rows=4, cols=3,
            subplot_titles=[
                'Quality Distribution', 'Publication Trends', 'Top Journals',
                'Research Gaps', 'Collaboration Network', 'Geographic Distribution',
                'Quality vs Impact', 'Monthly Patterns', 'Field Maturity',
                'Performance Radar', 'Trending Topics', 'Future Opportunities'
            ],
            specs=[
                [{"type": "histogram"}, {"type": "scatter"}, {"type": "bar"}],
                [{"type": "bar"}, {"type": "scatter"}, {"type": "scattergeo"}],
                [{"type": "scatter"}, {"type": "heatmap"}, {"type": "indicator"}],
                [{"type": "scatterpolar"}, {"type": "treemap"}, {"type": "bar"}]
            ]
        )
        
        # Add individual charts
        self._add_quality_histogram_to_subplot(fig, analysis.quality_analysis, 1, 1)
        self._add_publication_trends_to_subplot(fig, analysis.trend_analysis, 1, 2)
        self._add_top_journals_to_subplot(fig, analysis.journal_analysis, 1, 3)
        
        # Configure layout
        fig.update_layout(
            height=1200,
            title_text="Pharmacogenomics Research Analytics Dashboard",
            title_x=0.5,
            showlegend=True
        )
        
        return fig.to_html(include_plotlyjs=True)
    
    # Private helper methods for creating individual visualizations
    
    def _create_quality_histogram(self, quality_stats: QualityStats) -> go.Figure:
        """Create quality score histogram."""
        if not self.config.interactive:
            # Matplotlib version
            fig, ax = plt.subplots(figsize=self.config.figure_size)
            
            # Create sample data for histogram (in real implementation, you'd have actual scores)
            scores = np.random.normal(quality_stats.mean, quality_stats.std_dev, quality_stats.total_papers)
            scores = np.clip(scores, 0, 100)  # Ensure scores are in valid range
            
            ax.hist(scores, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            ax.axvline(quality_stats.mean, color='red', linestyle='--', label=f'Mean: {quality_stats.mean:.1f}')
            ax.axvline(quality_stats.median, color='green', linestyle='--', label=f'Median: {quality_stats.median:.1f}')
            
            ax.set_xlabel('Quality Score')
            ax.set_ylabel('Number of Papers')
            ax.set_title('Quality Score Distribution')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            return self._fig_to_base64(fig)
        
        else:
            # Plotly version
            scores = np.random.normal(quality_stats.mean, quality_stats.std_dev, quality_stats.total_papers)
            scores = np.clip(scores, 0, 100)
            
            fig = go.Figure()
            
            fig.add_trace(go.Histogram(
                x=scores,
                nbinsx=20,
                name='Quality Scores',
                marker_color='skyblue',
                opacity=0.7
            ))
            
            fig.add_vline(
                x=quality_stats.mean,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Mean: {quality_stats.mean:.1f}"
            )
            
            fig.add_vline(
                x=quality_stats.median,
                line_dash="dash",
                line_color="green",
                annotation_text=f"Median: {quality_stats.median:.1f}"
            )
            
            fig.update_layout(
                title='Quality Score Distribution',
                xaxis_title='Quality Score',
                yaxis_title='Number of Papers',
                showlegend=True
            )
            
            return fig
    
    def _create_quality_pie_chart(self, quality_stats: QualityStats) -> go.Figure:
        """Create quality distribution pie chart."""
        labels = ['High Quality (≥75)', 'Medium Quality (50-74)', 'Low Quality (<50)']
        values = [quality_stats.high_quality_count, quality_stats.medium_quality_count, quality_stats.low_quality_count]
        colors = [self.quality_colors['high'], self.quality_colors['medium'], self.quality_colors['low']]
        
        if self.config.interactive:
            fig = go.Figure(data=[go.Pie(
                labels=labels,
                values=values,
                marker_colors=colors,
                textinfo='label+percent+value',
                hovertemplate='<b>%{label}</b><br>Papers: %{value}<br>Percentage: %{percent}<extra></extra>'
            )])
            
            fig.update_layout(
                title='Quality Distribution',
                showlegend=True
            )
            
            return fig
        
        else:
            fig, ax = plt.subplots(figsize=self.config.figure_size)
            wedges, texts, autotexts = ax.pie(values, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            ax.set_title('Quality Distribution')
            return self._fig_to_base64(fig)
    
    def _create_quality_box_plot(self, quality_stats: QualityStats) -> go.Figure:
        """Create quality percentile box plot."""
        if self.config.interactive:
            fig = go.Figure()
            
            # Create box plot using percentiles
            fig.add_trace(go.Box(
                q1=[quality_stats.percentiles.get(25, 0)],
                median=[quality_stats.percentiles.get(50, 0)],
                q3=[quality_stats.percentiles.get(75, 0)],
                lowerfence=[0],
                upperfence=[100],
                mean=[quality_stats.mean],
                name='Quality Scores',
                boxpoints='outliers'
            ))
            
            fig.update_layout(
                title='Quality Score Distribution (Box Plot)',
                yaxis_title='Quality Score',
                showlegend=False
            )
            
            return fig
        
        else:
            fig, ax = plt.subplots(figsize=self.config.figure_size)
            
            # Create box plot data
            box_data = [
                quality_stats.percentiles.get(25, 0),
                quality_stats.percentiles.get(50, 0),
                quality_stats.percentiles.get(75, 0)
            ]
            
            ax.boxplot([box_data], labels=['Quality Scores'])
            ax.set_ylabel('Quality Score')
            ax.set_title('Quality Score Distribution (Box Plot)')
            ax.grid(True, alpha=0.3)
            
            return self._fig_to_base64(fig)
    
    def _create_quality_trend_chart(self, quality_stats: QualityStats) -> go.Figure:
        """Create quality trend over time chart."""
        # Generate sample trend data
        months = pd.date_range(start='2020-01-01', end='2024-01-01', freq='M')
        
        if quality_stats.quality_trend == "improving":
            trend_values = np.linspace(quality_stats.mean - 10, quality_stats.mean + 5, len(months))
        elif quality_stats.quality_trend == "declining":
            trend_values = np.linspace(quality_stats.mean + 5, quality_stats.mean - 10, len(months))
        else:
            trend_values = np.full(len(months), quality_stats.mean) + np.random.normal(0, 2, len(months))
        
        if self.config.interactive:
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=months,
                y=trend_values,
                mode='lines+markers',
                name='Quality Trend',
                line=dict(color=self.trend_colors.get(quality_stats.quality_trend, 'blue'))
            ))
            
            fig.update_layout(
                title=f'Quality Trend Over Time ({quality_stats.quality_trend.title()})',
                xaxis_title='Date',
                yaxis_title='Average Quality Score',
                showlegend=True
            )
            
            return fig
        
        else:
            fig, ax = plt.subplots(figsize=self.config.figure_size)
            
            ax.plot(months, trend_values, marker='o', 
                   color=self.trend_colors.get(quality_stats.quality_trend, 'blue'))
            ax.set_xlabel('Date')
            ax.set_ylabel('Average Quality Score')
            ax.set_title(f'Quality Trend Over Time ({quality_stats.quality_trend.title()})')
            ax.grid(True, alpha=0.3)
            
            # Format x-axis
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
            plt.xticks(rotation=45)
            
            return self._fig_to_base64(fig)
    
    def _create_quality_journal_scatter(self, analysis: ComprehensiveAnalysis) -> go.Figure:
        """Create quality vs journal impact scatter plot."""
        journal_stats = analysis.journal_analysis
        
        if not journal_stats.top_journals:
            return self._create_empty_chart("No journal data available")
        
        journals = [j['journal'] for j in journal_stats.top_journals[:15]]
        quality_scores = [j['avg_quality'] for j in journal_stats.top_journals[:15]]
        paper_counts = [j['paper_count'] for j in journal_stats.top_journals[:15]]
        
        if self.config.interactive:
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=quality_scores,
                y=paper_counts,
                mode='markers+text',
                text=journals,
                textposition='top center',
                marker=dict(
                    size=[min(50, max(10, count/2)) for count in paper_counts],
                    color=quality_scores,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Quality Score")
                ),
                hovertemplate='<b>%{text}</b><br>Quality: %{x:.1f}<br>Papers: %{y}<extra></extra>'
            ))
            
            fig.update_layout(
                title='Journal Quality vs Publication Volume',
                xaxis_title='Average Quality Score',
                yaxis_title='Number of Papers',
                showlegend=False
            )
            
            return fig
        
        else:
            fig, ax = plt.subplots(figsize=self.config.figure_size)
            
            scatter = ax.scatter(quality_scores, paper_counts, 
                               s=[count*2 for count in paper_counts],
                               c=quality_scores, cmap='viridis', alpha=0.7)
            
            # Add journal labels
            for i, journal in enumerate(journals):
                ax.annotate(journal[:20], (quality_scores[i], paper_counts[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
            
            ax.set_xlabel('Average Quality Score')
            ax.set_ylabel('Number of Papers')
            ax.set_title('Journal Quality vs Publication Volume')
            
            plt.colorbar(scatter, label='Quality Score')
            
            return self._fig_to_base64(fig)
    
    def _create_yearly_trends_chart(self, trend_stats: PublicationTrends) -> go.Figure:
        """Create yearly publication trends chart."""
        if not trend_stats.papers_per_year:
            return self._create_empty_chart("No publication trend data available")
        
        years = sorted(trend_stats.papers_per_year.keys())
        counts = [trend_stats.papers_per_year[year] for year in years]
        
        if self.config.interactive:
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=years,
                y=counts,
                mode='lines+markers',
                name='Publications per Year',
                line=dict(color='blue', width=3),
                marker=dict(size=8)
            ))
            
            # Add trend line
            if len(years) > 1:
                z = np.polyfit(years, counts, 1)
                p = np.poly1d(z)
                fig.add_trace(go.Scatter(
                    x=years,
                    y=p(years),
                    mode='lines',
                    name=f'Trend (Growth: {trend_stats.growth_rate:.1f}%)',
                    line=dict(color='red', dash='dash')
                ))
            
            fig.update_layout(
                title=f'Publication Trends ({trend_stats.recent_trend.title()})',
                xaxis_title='Year',
                yaxis_title='Number of Papers',
                showlegend=True
            )
            
            return fig
        
        else:
            fig, ax = plt.subplots(figsize=self.config.figure_size)
            
            ax.plot(years, counts, marker='o', linewidth=3, markersize=8, label='Publications')
            
            # Add trend line
            if len(years) > 1:
                z = np.polyfit(years, counts, 1)
                p = np.poly1d(z)
                ax.plot(years, p(years), '--', color='red', 
                       label=f'Trend (Growth: {trend_stats.growth_rate:.1f}%)')
            
            ax.set_xlabel('Year')
            ax.set_ylabel('Number of Papers')
            ax.set_title(f'Publication Trends ({trend_stats.recent_trend.title()})')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            return self._fig_to_base64(fig)
    
    def _create_monthly_heatmap(self, trend_stats: PublicationTrends) -> go.Figure:
        """Create monthly publication patterns heatmap."""
        if not trend_stats.papers_per_month:
            return self._create_empty_chart("No monthly data available")
        
        # Parse monthly data
        monthly_data = {}
        for month_key, count in trend_stats.papers_per_month.items():
            try:
                year, month = month_key.split('-')
                year, month = int(year), int(month)
                if year not in monthly_data:
                    monthly_data[year] = {}
                monthly_data[year][month] = count
            except ValueError:
                continue
        
        if not monthly_data:
            return self._create_empty_chart("No valid monthly data")
        
        # Create matrix for heatmap
        years = sorted(monthly_data.keys())
        months = list(range(1, 13))
        
        matrix = []
        for year in years:
            row = []
            for month in months:
                row.append(monthly_data[year].get(month, 0))
            matrix.append(row)
        
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        if self.config.interactive:
            fig = go.Figure(data=go.Heatmap(
                z=matrix,
                x=month_names,
                y=years,
                colorscale='Viridis',
                hovertemplate='Year: %{y}<br>Month: %{x}<br>Papers: %{z}<extra></extra>'
            ))
            
            fig.update_layout(
                title='Monthly Publication Patterns',
                xaxis_title='Month',
                yaxis_title='Year'
            )
            
            return fig
        
        else:
            fig, ax = plt.subplots(figsize=self.config.figure_size)
            
            im = ax.imshow(matrix, cmap='viridis', aspect='auto')
            
            ax.set_xticks(range(len(month_names)))
            ax.set_xticklabels(month_names)
            ax.set_yticks(range(len(years)))
            ax.set_yticklabels(years)
            
            ax.set_xlabel('Month')
            ax.set_ylabel('Year')
            ax.set_title('Monthly Publication Patterns')
            
            plt.colorbar(im, label='Number of Papers')
            
            return self._fig_to_base64(fig)
    
    def _create_top_journals_chart(self, journal_stats) -> go.Figure:
        """Create top journals bar chart."""
        if not journal_stats.top_journals:
            return self._create_empty_chart("No journal data available")
        
        top_journals = journal_stats.top_journals[:10]  # Top 10
        journals = [j['journal'][:30] + '...' if len(j['journal']) > 30 else j['journal'] for j in top_journals]
        counts = [j['paper_count'] for j in top_journals]
        qualities = [j['avg_quality'] for j in top_journals]
        
        if self.config.interactive:
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                y=journals,
                x=counts,
                orientation='h',
                marker=dict(
                    color=qualities,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Avg Quality")
                ),
                hovertemplate='<b>%{y}</b><br>Papers: %{x}<br>Quality: %{marker.color:.1f}<extra></extra>'
            ))
            
            fig.update_layout(
                title='Top Journals by Publication Count',
                xaxis_title='Number of Papers',
                yaxis_title='Journal',
                height=600
            )
            
            return fig
        
        else:
            fig, ax = plt.subplots(figsize=self.config.figure_size)
            
            bars = ax.barh(journals, counts)
            
            # Color bars by quality
            norm = plt.Normalize(min(qualities), max(qualities))
            colors = plt.cm.viridis(norm(qualities))
            for bar, color in zip(bars, colors):
                bar.set_color(color)
            
            ax.set_xlabel('Number of Papers')
            ax.set_ylabel('Journal')
            ax.set_title('Top Journals by Publication Count')
            
            # Add colorbar
            sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=norm)
            sm.set_array([])
            plt.colorbar(sm, label='Average Quality Score')
            
            return self._fig_to_base64(fig)
    
    def _create_opportunity_chart(self, opportunities: List[Dict[str, Any]], title: str) -> go.Figure:
        """Create research opportunities chart."""
        if not opportunities:
            return self._create_empty_chart(f"No {title.lower()} data available")
        
        entities = [opp['entity'] for opp in opportunities[:10]]
        counts = [opp['count'] for opp in opportunities[:10]]
        scores = [opp['opportunity_score'] for opp in opportunities[:10]]
        
        if self.config.interactive:
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=counts,
                y=scores,
                mode='markers+text',
                text=entities,
                textposition='top center',
                marker=dict(
                    size=[max(10, min(50, score)) for score in scores],
                    color=scores,
                    colorscale='RdYlGn',
                    showscale=True,
                    colorbar=dict(title="Opportunity Score")
                ),
                hovertemplate='<b>%{text}</b><br>Current Papers: %{x}<br>Opportunity Score: %{y}<extra></extra>'
            ))
            
            fig.update_layout(
                title=f'{title} - Research Opportunities',
                xaxis_title='Current Paper Count',
                yaxis_title='Opportunity Score',
                showlegend=False
            )
            
            return fig
        
        else:
            fig, ax = plt.subplots(figsize=self.config.figure_size)
            
            scatter = ax.scatter(counts, scores, s=[score*2 for score in scores],
                               c=scores, cmap='RdYlGn', alpha=0.7)
            
            for i, entity in enumerate(entities):
                ax.annotate(entity, (counts[i], scores[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
            
            ax.set_xlabel('Current Paper Count')
            ax.set_ylabel('Opportunity Score')
            ax.set_title(f'{title} - Research Opportunities')
            
            plt.colorbar(scatter, label='Opportunity Score')
            
            return self._fig_to_base64(fig)
    
    def _create_empty_chart(self, message: str) -> go.Figure:
        """Create empty chart with message."""
        if self.config.interactive:
            fig = go.Figure()
            fig.add_annotation(
                text=message,
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=16)
            )
            fig.update_layout(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False)
            )
            return fig
        else:
            fig, ax = plt.subplots(figsize=self.config.figure_size)
            ax.text(0.5, 0.5, message, ha='center', va='center', fontsize=16)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            return self._fig_to_base64(fig)
    
    def _fig_to_base64(self, fig) -> str:
        """Convert matplotlib figure to base64 string."""
        buffer = io.BytesIO()
        fig.savefig(buffer, format=self.config.save_format, dpi=self.config.dpi, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        buffer.close()
        plt.close(fig)
        return f"data:image/{self.config.save_format};base64,{image_base64}"
    
    # Additional helper methods would be implemented here for:
    # - _create_journal_bubble_chart
    # - _create_journal_network
    # - _create_author_network
    # - _create_geographic_collaboration_map
    # - _create_topics_wordcloud
    # - _create_performance_radar
    # - etc.
    
    def save_dashboard_html(self, dashboard_components: Dict[str, Any], 
                           output_path: str, title: str = "Research Analytics Dashboard"):
        """Save complete dashboard as HTML file.
        
        Args:
            dashboard_components: Dictionary of visualization components
            output_path: Path to save HTML file
            title: Dashboard title
        """
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{title}</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .dashboard-section {{ margin-bottom: 40px; }}
                .chart-container {{ margin-bottom: 20px; }}
                h1, h2 {{ color: #333; }}
            </style>
        </head>
        <body>
            <h1>{title}</h1>
        """
        
        for section_name, charts in dashboard_components.items():
            html_content += f"<div class='dashboard-section'><h2>{section_name.replace('_', ' ').title()}</h2>"
            
            if isinstance(charts, dict):
                for chart_name, chart in charts.items():
                    html_content += f"<div class='chart-container'>"
                    if hasattr(chart, 'to_html'):
                        html_content += chart.to_html(include_plotlyjs=False, div_id=f"{section_name}_{chart_name}")
                    else:
                        html_content += f"<img src='{chart}' alt='{chart_name}' style='max-width: 100%;'>"
                    html_content += "</div>"
            
            html_content += "</div>"
        
        html_content += """
        </body>
        </html>
        """
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Dashboard saved to {output_path}")
    
    # Placeholder methods for additional visualizations
    def _create_growth_analysis_chart(self, trend_stats): pass
    def _create_seasonal_radar_chart(self, trend_stats): pass
    def _create_publication_velocity_chart(self, trend_stats): pass
    def _create_journal_bubble_chart(self, journal_stats): pass
    def _create_journal_network(self, journal_stats): pass
    def _create_journal_impact_distribution(self, journal_stats): pass
    def _create_journal_quality_comparison(self, journal_stats): pass
    def _create_methodology_gaps_radar(self, gaps): pass
    def _create_population_gaps_map(self, gaps): pass
    def _create_opportunity_matrix(self, gaps): pass
    def _create_author_network(self, collab_stats): pass
    def _create_geographic_collaboration_map(self, collab_stats): pass
    def _create_institutional_collaboration_chart(self, collab_stats): pass
    def _create_collaboration_trends_chart(self, collab_stats): pass
    def _create_network_metrics_dashboard(self, collab_stats): pass
    def _create_topics_wordcloud(self, analysis): pass
    def _create_drug_gene_wordcloud(self, analysis): pass
    def _create_methodology_wordcloud(self, analysis): pass
    def _create_trending_topics_wordcloud(self, analysis): pass
    def _create_performance_radar(self, benchmark_results): pass
    def _create_percentile_chart(self, benchmark_results): pass
    def _create_strengths_improvements_chart(self, benchmark_results): pass
    def _create_score_gauge(self, benchmark_results): pass
    def _add_quality_histogram_to_subplot(self, fig, quality_stats, row, col): pass
    def _add_publication_trends_to_subplot(self, fig, trend_stats, row, col): pass
    def _add_top_journals_to_subplot(self, fig, journal_stats, row, col): pass