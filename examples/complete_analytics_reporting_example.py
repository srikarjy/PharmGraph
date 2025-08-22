"""Complete analytics and reporting system example."""

import asyncio
import os
from datetime import datetime
from pathlib import Path

from src.analytics.research_analyzer import PharmacogenomicsAnalyzer
from src.analytics.benchmarking import PharmacogenomicsBenchmarking
from src.analytics.visualization_engine import VisualizationEngine, VisualizationConfig
from src.analytics.report_generator import ReportGenerator, ReportConfig
from src.config.logging_config import setup_logging


# Setup logging
setup_logging()


async def run_complete_analytics_and_reporting():
    """Run complete analytics and reporting pipeline."""
    
    print("🔬 COMPLETE PHARMACOGENOMICS ANALYTICS & REPORTING SYSTEM")
    print("=" * 70)
    
    # Initialize all components
    print("\n📊 Initializing Analytics Components...")
    
    analyzer = PharmacogenomicsAnalyzer()
    benchmarking = PharmacogenomicsBenchmarking()
    
    # Configure visualization engine
    viz_config = VisualizationConfig(
        interactive=True,
        figure_size=(12, 8),
        color_palette="viridis"
    )
    visualization_engine = VisualizationEngine(viz_config)
    
    # Configure report generator
    report_config = ReportConfig(
        output_dir="reports/pharmacogenomics",
        include_visualizations=True,
        organization_name="Pharmacogenomics Research Institute",
        contact_email="analytics@pharmgx-institute.org"
    )
    report_generator = ReportGenerator(report_config)
    
    print("✅ All components initialized successfully")
    
    try:
        # Phase 1: Comprehensive Analytics
        print("\n📈 Phase 1: Running Comprehensive Analytics...")
        
        # Run complete analysis
        analysis = await analyzer.analyze_collection()
        print(f"✅ Analysis completed: {analysis.quality_analysis.total_papers} papers analyzed")
        
        # Run benchmarking
        benchmark_results = await benchmarking.benchmark_against_field_standards()
        print(f"✅ Benchmarking completed: Overall score {benchmark_results.overall_score:.1f}/100")
        
        # Run additional analyses
        journal_comparison = await benchmarking.compare_journal_performance()
        productivity_analysis = await benchmarking.analyze_research_group_productivity()
        impact_metrics = await benchmarking.calculate_research_impact_metrics()
        field_summary = await benchmarking.generate_field_summary_report()
        
        print("✅ All analytics completed successfully")
        
        # Phase 2: Visualization Generation
        print("\n🎨 Phase 2: Generating Visualizations...")
        
        # Create comprehensive dashboards
        quality_dashboard = visualization_engine.create_quality_distribution_dashboard(analysis)
        trends_dashboard = visualization_engine.create_publication_trends_dashboard(analysis)
        journal_dashboard = visualization_engine.create_journal_analysis_dashboard(analysis)
        gaps_dashboard = visualization_engine.create_research_gaps_dashboard(analysis)
        collaboration_dashboard = visualization_engine.create_collaboration_network_dashboard(analysis)
        wordcloud_dashboard = visualization_engine.create_research_topics_wordcloud(analysis)
        benchmarking_dashboard = visualization_engine.create_benchmarking_dashboard(benchmark_results)
        
        print("✅ All visualizations generated successfully")
        
        # Create interactive dashboard
        interactive_dashboard = visualization_engine.create_interactive_dashboard(
            analysis, benchmark_results
        )
        
        # Save interactive dashboard
        dashboard_path = "reports/pharmacogenomics/interactive_dashboard.html"
        Path(dashboard_path).parent.mkdir(parents=True, exist_ok=True)
        with open(dashboard_path, 'w', encoding='utf-8') as f:
            f.write(interactive_dashboard)
        
        print(f"✅ Interactive dashboard saved: {dashboard_path}")
        
        # Phase 3: Report Generation
        print("\n📋 Phase 3: Generating Reports...")
        
        # Generate Executive Summary
        print("  📊 Generating Executive Summary...")
        exec_reports = report_generator.generate_executive_summary(
            analysis, benchmark_results, 
            export_formats=['html', 'pdf', 'json']
        )
        print(f"     ✅ Executive Summary: {list(exec_reports.keys())}")
        
        # Generate Technical Report
        print("  🔬 Generating Technical Report...")
        tech_reports = report_generator.generate_technical_report(
            analysis, benchmark_results, journal_comparison, productivity_analysis,
            export_formats=['html', 'pdf', 'excel', 'json']
        )
        print(f"     ✅ Technical Report: {list(tech_reports.keys())}")
        
        # Generate Quality Assessment Report
        print("  🎯 Generating Quality Assessment Report...")
        quality_reports = report_generator.generate_quality_assessment_report(
            analysis, export_formats=['html', 'pdf', 'csv']
        )
        print(f"     ✅ Quality Assessment: {list(quality_reports.keys())}")
        
        # Generate Email Summaries
        print("  📧 Generating Email Summaries...")
        
        executive_email = report_generator.generate_email_summary(
            analysis, benchmark_results, recipient_type='executive'
        )
        researcher_email = report_generator.generate_email_summary(
            analysis, benchmark_results, recipient_type='researcher'
        )
        stakeholder_email = report_generator.generate_email_summary(
            analysis, benchmark_results, recipient_type='stakeholder'
        )
        
        # Save email summaries
        email_dir = Path("reports/pharmacogenomics/email_summaries")
        email_dir.mkdir(parents=True, exist_ok=True)
        
        with open(email_dir / "executive_summary.html", 'w', encoding='utf-8') as f:
            f.write(executive_email)
        with open(email_dir / "researcher_summary.html", 'w', encoding='utf-8') as f:
            f.write(researcher_email)
        with open(email_dir / "stakeholder_summary.html", 'w', encoding='utf-8') as f:
            f.write(stakeholder_email)
        
        print("     ✅ Email summaries generated and saved")
        
        # Phase 4: Data Export
        print("\n💾 Phase 4: Exporting Data...")
        
        # Export to CSV
        csv_path = "reports/pharmacogenomics/analytics_data.csv"
        report_generator.export_data_csv(analysis, csv_path)
        print(f"     ✅ CSV export: {csv_path}")
        
        # Export to JSON
        json_path = "reports/pharmacogenomics/analytics_data.json"
        report_generator.export_data_json(analysis, json_path)
        print(f"     ✅ JSON export: {json_path}")
        
        # Phase 5: Results Summary
        print("\n" + "=" * 70)
        print("📈 ANALYTICS RESULTS SUMMARY")
        print("=" * 70)
        
        # Key Findings
        print(f"\n🎯 Key Findings:")
        print(f"• Total Papers Analyzed: {analysis.quality_analysis.total_papers:,}")
        print(f"• Average Quality Score: {analysis.quality_analysis.mean:.1f}/100")
        print(f"• Quality Trend: {analysis.quality_analysis.quality_trend.title()}")
        print(f"• Annual Growth Rate: {analysis.trend_analysis.growth_rate:.1f}%")
        print(f"• Field Maturity Score: {analysis.field_maturity.maturity_score:.1f}/100")
        print(f"• Performance Percentile: {benchmark_results.overall_score:.1f}th")
        
        # Quality Distribution
        print(f"\n📊 Quality Distribution:")
        total_papers = analysis.quality_analysis.total_papers
        high_pct = (analysis.quality_analysis.high_quality_count / max(1, total_papers)) * 100
        medium_pct = (analysis.quality_analysis.medium_quality_count / max(1, total_papers)) * 100
        low_pct = (analysis.quality_analysis.low_quality_count / max(1, total_papers)) * 100
        
        print(f"• High Quality (≥75): {analysis.quality_analysis.high_quality_count:,} ({high_pct:.1f}%)")
        print(f"• Medium Quality (50-74): {analysis.quality_analysis.medium_quality_count:,} ({medium_pct:.1f}%)")
        print(f"• Low Quality (<50): {analysis.quality_analysis.low_quality_count:,} ({low_pct:.1f}%)")
        
        # Top Research Opportunities
        print(f"\n🔍 Top Research Opportunities:")
        for i, opp in enumerate(analysis.gap_analysis.understudied_drugs[:5], 1):
            print(f"  {i}. {opp['entity']}: {opp['count']} papers (Opportunity Score: {opp['opportunity_score']})")
        
        # Benchmarking Results
        print(f"\n🏆 Benchmarking Results:")
        print(f"• Overall Performance: {benchmark_results.overall_score:.1f}/100")
        if benchmark_results.strengths:
            print(f"• Key Strengths: {', '.join(benchmark_results.strengths[:2])}")
        if benchmark_results.improvement_areas:
            print(f"• Improvement Areas: {', '.join(benchmark_results.improvement_areas[:2])}")
        
        # Top Journals
        print(f"\n📚 Top Journals:")
        for i, journal in enumerate(journal_comparison.journal_rankings[:5], 1):
            print(f"  {i}. {journal['journal']}: {journal['paper_count']} papers (Quality: {journal['avg_quality']:.1f})")
        
        # Research Insights
        print(f"\n💡 Key Research Insights:")
        for i, insight in enumerate(analysis.insights[:3], 1):
            print(f"  {i}. {insight['title']}: {insight['description']}")
        
        # High-Priority Recommendations
        high_priority_recs = [rec for rec in analysis.recommendations if rec.priority == 'high']
        if high_priority_recs:
            print(f"\n🎯 High-Priority Recommendations:")
            for i, rec in enumerate(high_priority_recs[:3], 1):
                print(f"  {i}. {rec.title}")
                print(f"     {rec.description}")
        
        # Generated Files Summary
        print(f"\n📁 Generated Files:")
        print(f"• Interactive Dashboard: {dashboard_path}")
        print(f"• Executive Summary: {exec_reports.get('html', 'N/A')}")
        print(f"• Technical Report: {tech_reports.get('html', 'N/A')}")
        print(f"• Quality Assessment: {quality_reports.get('html', 'N/A')}")
        print(f"• Data Exports: {csv_path}, {json_path}")
        print(f"• Email Summaries: {email_dir}")
        
        # Performance Metrics
        print(f"\n⚡ System Performance:")
        estimated_citations = impact_metrics.citation_indicators.get('estimated_avg_citations', 0)
        print(f"• Estimated Avg Citations: {estimated_citations:.1f} per paper")
        print(f"• Quality-Impact Correlation: {impact_metrics.quality_impact_correlation:.2f}")
        print(f"• Field Standardization: {analysis.field_maturity.standardization_level:.1f}%")
        
        # Future Directions
        if field_summary.future_directions:
            print(f"\n🔮 Future Research Directions:")
            for i, direction in enumerate(field_summary.future_directions[:3], 1):
                print(f"  {i}. {direction}")
        
        print(f"\n" + "=" * 70)
        print(f"✅ Complete analytics and reporting pipeline executed successfully!")
        print(f"📊 Generated comprehensive insights for {analysis.quality_analysis.total_papers:,} papers")
        print(f"🎯 Identified {len(analysis.gap_analysis.understudied_drugs)} research opportunities")
        print(f"📈 Performance ranking: {benchmark_results.overall_score:.1f}th percentile")
        print(f"⏱️  Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"=" * 70)
        
        return {
            'analysis': analysis,
            'benchmark_results': benchmark_results,
            'journal_comparison': journal_comparison,
            'productivity_analysis': productivity_analysis,
            'impact_metrics': impact_metrics,
            'field_summary': field_summary,
            'reports': {
                'executive': exec_reports,
                'technical': tech_reports,
                'quality': quality_reports
            },
            'visualizations': {
                'quality': quality_dashboard,
                'trends': trends_dashboard,
                'journals': journal_dashboard,
                'gaps': gaps_dashboard,
                'collaboration': collaboration_dashboard,
                'wordclouds': wordcloud_dashboard,
                'benchmarking': benchmarking_dashboard
            },
            'exports': {
                'csv': csv_path,
                'json': json_path,
                'dashboard': dashboard_path
            }
        }
        
    except Exception as e:
        print(f"\n❌ Analytics pipeline failed: {str(e)}")
        raise


async def run_targeted_reporting_example():
    """Run targeted reporting for specific use cases."""
    
    print("\n🎯 TARGETED REPORTING EXAMPLES")
    print("=" * 50)
    
    analyzer = PharmacogenomicsAnalyzer()
    report_generator = ReportGenerator()
    
    # Run basic analysis
    analysis = await analyzer.analyze_collection()
    
    # Example 1: Quality-focused report for QA team
    print("\n📊 Quality Assessment Report for QA Team...")
    quality_reports = report_generator.generate_quality_assessment_report(
        analysis, export_formats=['html', 'csv']
    )
    print(f"✅ Quality reports generated: {list(quality_reports.keys())}")
    
    # Example 2: Executive briefing
    print("\n👔 Executive Briefing...")
    benchmarking = PharmacogenomicsBenchmarking()
    benchmark_results = await benchmarking.benchmark_against_field_standards()
    
    exec_email = report_generator.generate_email_summary(
        analysis, benchmark_results, recipient_type='executive'
    )
    
    # Save executive briefing
    exec_path = "reports/executive_briefing.html"
    Path(exec_path).parent.mkdir(parents=True, exist_ok=True)
    with open(exec_path, 'w', encoding='utf-8') as f:
        f.write(exec_email)
    
    print(f"✅ Executive briefing saved: {exec_path}")
    
    # Example 3: Researcher collaboration report
    print("\n🤝 Researcher Collaboration Report...")
    researcher_email = report_generator.generate_email_summary(
        analysis, benchmark_results, recipient_type='researcher'
    )
    
    researcher_path = "reports/researcher_collaboration.html"
    with open(researcher_path, 'w', encoding='utf-8') as f:
        f.write(researcher_email)
    
    print(f"✅ Researcher report saved: {researcher_path}")
    
    # Example 4: Data export for external analysis
    print("\n💾 Data Export for External Analysis...")
    
    csv_export = report_generator.export_data_csv(analysis, "reports/external_analysis.csv")
    json_export = report_generator.export_data_json(analysis, "reports/external_analysis.json")
    
    print(f"✅ Data exported: CSV and JSON formats")
    
    print("\n✅ Targeted reporting examples completed!")


def demonstrate_email_integration():
    """Demonstrate email integration capabilities."""
    
    print("\n📧 EMAIL INTEGRATION DEMONSTRATION")
    print("=" * 40)
    
    # Note: This is a demonstration - actual email sending would require SMTP configuration
    
    report_generator = ReportGenerator()
    
    # Example email configurations
    email_configs = {
        'executives': {
            'recipients': ['ceo@pharmgx.org', 'cto@pharmgx.org'],
            'subject': 'Weekly Pharmacogenomics Research Analytics Summary',
            'type': 'executive'
        },
        'researchers': {
            'recipients': ['research-team@pharmgx.org'],
            'subject': 'Research Opportunities and Collaboration Insights',
            'type': 'researcher'
        },
        'stakeholders': {
            'recipients': ['board@pharmgx.org', 'investors@pharmgx.org'],
            'subject': 'Pharmacogenomics Field Analysis and Performance Report',
            'type': 'stakeholder'
        }
    }
    
    print("📧 Email Integration Configured:")
    for group, config in email_configs.items():
        print(f"  • {group.title()}: {len(config['recipients'])} recipients")
        print(f"    Subject: {config['subject']}")
        print(f"    Type: {config['type']}")
    
    print("\n💡 Email features available:")
    print("  • HTML-formatted reports with embedded visualizations")
    print("  • Recipient-specific content (executive, researcher, stakeholder)")
    print("  • Attachment support for detailed reports")
    print("  • Automated scheduling capabilities")
    print("  • SMTP integration with authentication")
    
    print("\n✅ Email integration ready for deployment!")


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Complete Analytics and Reporting System")
    parser.add_argument("--mode", choices=["full", "targeted", "email-demo"], default="full",
                       help="Execution mode")
    
    args = parser.parse_args()
    
    try:
        if args.mode == "full":
            results = asyncio.run(run_complete_analytics_and_reporting())
            print(f"\n🎉 Complete system executed successfully!")
            print(f"📊 Results available in: {results['exports']}")
            
        elif args.mode == "targeted":
            asyncio.run(run_targeted_reporting_example())
            
        elif args.mode == "email-demo":
            demonstrate_email_integration()
            
    except KeyboardInterrupt:
        print("\n⏹️  Execution interrupted by user")
    except Exception as e:
        print(f"\n❌ Execution failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()