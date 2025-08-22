"""Complete research analytics example for pharmacogenomics data."""

import asyncio
import logging
from datetime import datetime
from typing import Optional

from src.analytics.research_analyzer import PharmacogenomicsAnalyzer
from src.analytics.benchmarking import PharmacogenomicsBenchmarking
from src.config.logging_config import setup_logging


# Setup logging
setup_logging()
logger = logging.getLogger(__name__)


async def run_comprehensive_analytics(collection_id: Optional[str] = None):
    """Run comprehensive research analytics on pharmacogenomics data."""
    
    print("🔬 PHARMACOGENOMICS RESEARCH ANALYTICS")
    print("=" * 60)
    
    # Initialize analyzers
    analyzer = PharmacogenomicsAnalyzer()
    benchmarking = PharmacogenomicsBenchmarking()
    
    try:
        # Phase 1: Comprehensive Research Analysis
        print("\n📊 Phase 1: Comprehensive Research Analysis")
        print("-" * 50)
        
        analysis = await analyzer.analyze_collection(collection_id)
        
        # Quality Analysis Results
        print(f"\n🎯 QUALITY ANALYSIS")
        print(f"Total Papers Analyzed: {analysis.quality_analysis.total_papers:,}")
        print(f"Average Quality Score: {analysis.quality_analysis.mean:.1f} ± {analysis.quality_analysis.std_dev:.1f}")
        print(f"Quality Distribution:")
        print(f"  • High Quality (≥75): {analysis.quality_analysis.high_quality_count:,} ({(analysis.quality_analysis.high_quality_count/max(1,analysis.quality_analysis.total_papers)*100):.1f}%)")
        print(f"  • Medium Quality (50-74): {analysis.quality_analysis.medium_quality_count:,} ({(analysis.quality_analysis.medium_quality_count/max(1,analysis.quality_analysis.total_papers)*100):.1f}%)")
        print(f"  • Low Quality (<50): {analysis.quality_analysis.low_quality_count:,} ({(analysis.quality_analysis.low_quality_count/max(1,analysis.quality_analysis.total_papers)*100):.1f}%)")
        print(f"Quality Trend: {analysis.quality_analysis.quality_trend.title()}")
        
        # Quality Percentiles
        print(f"\nQuality Score Percentiles:")
        for percentile, score in analysis.quality_analysis.percentiles.items():
            print(f"  • {percentile}th percentile: {score:.1f}")
        
        # Publication Trends
        print(f"\n📈 PUBLICATION TRENDS")
        print(f"Annual Growth Rate: {analysis.trend_analysis.growth_rate:.1f}%")
        print(f"Peak Publication Year: {analysis.trend_analysis.peak_year}")
        print(f"Recent Trend: {analysis.trend_analysis.recent_trend.title()}")
        
        # Top publication years
        if analysis.trend_analysis.papers_per_year:
            sorted_years = sorted(analysis.trend_analysis.papers_per_year.items(), 
                                key=lambda x: x[1], reverse=True)
            print(f"\nTop Publication Years:")
            for year, count in sorted_years[:5]:
                print(f"  • {year}: {count:,} papers")
        
        # Seasonal patterns
        if analysis.trend_analysis.seasonal_patterns:
            print(f"\nSeasonal Publication Patterns:")
            sorted_months = sorted(analysis.trend_analysis.seasonal_patterns.items(), 
                                 key=lambda x: x[1], reverse=True)
            for month, percentage in sorted_months[:6]:
                print(f"  • {month}: {percentage:.1f}% of annual publications")
        
        # Journal Analysis
        print(f"\n📚 JOURNAL ANALYSIS")
        print(f"Total Journals: {len(analysis.journal_analysis.journal_frequency)}")
        print(f"Journal Impact Correlation: {analysis.journal_analysis.impact_correlation:.2f}")
        
        print(f"\nTop Journals by Paper Count:")
        for i, journal_info in enumerate(analysis.journal_analysis.top_journals[:10], 1):
            print(f"  {i:2d}. {journal_info['journal']}")
            print(f"      Papers: {journal_info['paper_count']:,} | Avg Quality: {journal_info['avg_quality']:.1f}")
        
        # Research Gaps Analysis
        print(f"\n🔍 RESEARCH GAPS ANALYSIS")
        
        if analysis.gap_analysis.understudied_drugs:
            print(f"\nUnderstudied Drugs (Research Opportunities):")
            for i, drug_info in enumerate(analysis.gap_analysis.understudied_drugs[:5], 1):
                print(f"  {i}. {drug_info['entity']}: {drug_info['count']} papers (Opportunity Score: {drug_info['opportunity_score']})")
        
        if analysis.gap_analysis.understudied_genes:
            print(f"\nUnderstudied Genes:")
            for i, gene_info in enumerate(analysis.gap_analysis.understudied_genes[:5], 1):
                print(f"  {i}. {gene_info['entity']}: {gene_info['count']} papers")
        
        if analysis.gap_analysis.methodology_gaps:
            print(f"\nMethodology Gaps:")
            for i, method in enumerate(analysis.gap_analysis.methodology_gaps[:5], 1):
                print(f"  {i}. {method}")
        
        if analysis.gap_analysis.population_gaps:
            print(f"\nPopulation Gaps:")
            for i, population in enumerate(analysis.gap_analysis.population_gaps[:3], 1):
                print(f"  {i}. {population} populations underrepresented")
        
        # Collaboration Analysis
        print(f"\n🤝 COLLABORATION ANALYSIS")
        print(f"Total Collaboration Networks: {len(analysis.collaboration_analysis.collaboration_network):,}")
        print(f"Geographic Diversity: {len(analysis.collaboration_analysis.geographic_distribution)} regions")
        print(f"Institutional Diversity: {len(analysis.collaboration_analysis.institutional_patterns):,} institutions")
        
        if analysis.collaboration_analysis.top_collaborators:
            print(f"\nTop Research Collaborators:")
            for i, collab in enumerate(analysis.collaboration_analysis.top_collaborators[:5], 1):
                print(f"  {i}. {collab['author']}: {collab['collaborator_count']} collaborators")
        
        if analysis.collaboration_analysis.geographic_distribution:
            print(f"\nGeographic Distribution:")
            sorted_regions = sorted(analysis.collaboration_analysis.geographic_distribution.items(), 
                                  key=lambda x: x[1], reverse=True)
            for region, count in sorted_regions[:8]:
                print(f"  • {region}: {count:,} papers")
        
        # Field Maturity Assessment
        print(f"\n🎓 FIELD MATURITY ASSESSMENT")
        print(f"Overall Maturity Score: {analysis.field_maturity.maturity_score:.1f}/100")
        print(f"Standardization Level: {analysis.field_maturity.standardization_level:.1f}%")
        
        if analysis.field_maturity.established_areas:
            print(f"\nEstablished Research Areas:")
            for i, area in enumerate(analysis.field_maturity.established_areas[:5], 1):
                print(f"  {i}. {area}")
        
        if analysis.field_maturity.emerging_areas:
            print(f"\nEmerging Research Areas:")
            for i, area in enumerate(analysis.field_maturity.emerging_areas[:5], 1):
                print(f"  {i}. {area}")
        
        # Research Insights
        print(f"\n💡 RESEARCH INSIGHTS")
        for i, insight in enumerate(analysis.insights[:5], 1):
            print(f"  {i}. [{insight['type'].upper()}] {insight['title']}")
            print(f"     {insight['description']}")
        
        # Research Recommendations
        print(f"\n🎯 RESEARCH RECOMMENDATIONS")
        for i, rec in enumerate(analysis.recommendations[:5], 1):
            print(f"  {i}. [{rec.priority.upper()}] {rec.title}")
            print(f"     {rec.description}")
            if rec.evidence:
                print(f"     Evidence: {', '.join(rec.evidence[:2])}")
        
        # Phase 2: Benchmarking Analysis
        print(f"\n📊 Phase 2: Benchmarking Analysis")
        print("-" * 50)
        
        benchmark_results = await benchmarking.benchmark_against_field_standards(collection_id)
        
        print(f"\n🏆 FIELD BENCHMARKING RESULTS")
        print(f"Overall Performance Score: {benchmark_results.overall_score:.1f}/100")
        print(f"\n{benchmark_results.comparison_summary}")
        
        print(f"\nPerformance vs. Field Benchmarks:")
        for metric, percentile in benchmark_results.performance_percentile.items():
            status = "🟢" if percentile >= 75 else "🟡" if percentile >= 50 else "🔴"
            print(f"  {status} {metric.replace('_', ' ').title()}: {percentile:.1f}th percentile")
        
        if benchmark_results.strengths:
            print(f"\nKey Strengths:")
            for strength in benchmark_results.strengths:
                print(f"  ✅ {strength}")
        
        if benchmark_results.improvement_areas:
            print(f"\nImprovement Areas:")
            for area in benchmark_results.improvement_areas:
                print(f"  ⚠️  {area}")
        
        # Journal Performance Comparison
        journal_comparison = await benchmarking.compare_journal_performance(collection_id)
        
        print(f"\n📖 JOURNAL PERFORMANCE COMPARISON")
        print(f"\nTop Journals by Combined Score:")
        for i, journal in enumerate(journal_comparison.journal_rankings[:10], 1):
            specializations = ", ".join(journal['specialization'][:2]) if journal['specialization'] else "General"
            print(f"  {i:2d}. {journal['journal']}")
            print(f"      Combined Score: {journal['combined_score']:.1f} | Papers: {journal['paper_count']:,}")
            print(f"      Quality: {journal['avg_quality']:.1f} | Specialization: {specializations}")
        
        print(f"\nJournal Quality Benchmarks:")
        for tier, score in journal_comparison.quality_benchmarks.items():
            print(f"  • {tier.replace('_', ' ').title()}: {score:.1f}")
        
        if journal_comparison.emerging_journals:
            print(f"\nEmerging Journals:")
            for journal in journal_comparison.emerging_journals:
                print(f"  📈 {journal}")
        
        # Productivity Analysis
        productivity_analysis = await benchmarking.analyze_research_group_productivity(collection_id)
        
        print(f"\n👥 RESEARCH PRODUCTIVITY ANALYSIS")
        print(f"\nProductivity Metrics:")
        for metric, value in productivity_analysis.productivity_metrics.items():
            print(f"  • {metric.replace('_', ' ').title()}: {value:.2f}")
        
        if productivity_analysis.top_productive_groups:
            print(f"\nTop Productive Research Groups:")
            for i, group in enumerate(productivity_analysis.top_productive_groups[:5], 1):
                print(f"  {i}. {group['lead_author']}")
                print(f"     Productivity Score: {group['productivity_score']:.1f} | Collaborators: {group['collaborator_count']}")
        
        # Impact Metrics
        impact_metrics = await benchmarking.calculate_research_impact_metrics(collection_id)
        
        print(f"\n📊 RESEARCH IMPACT METRICS")
        print(f"\nCitation Indicators:")
        for indicator, value in impact_metrics.citation_indicators.items():
            print(f"  • {indicator.replace('_', ' ').title()}: {value:.1f}")
        
        print(f"\nQuality-Impact Correlation: {impact_metrics.quality_impact_correlation:.2f}")
        
        if impact_metrics.high_impact_characteristics:
            print(f"\nHigh-Impact Research Characteristics:")
            for characteristic in impact_metrics.high_impact_characteristics:
                print(f"  ⭐ {characteristic}")
        
        # Field Summary Report
        field_summary = await benchmarking.generate_field_summary_report(collection_id)
        
        print(f"\n📋 COMPREHENSIVE FIELD SUMMARY")
        print("-" * 50)
        
        print(f"\nField Overview:")
        for key, value in field_summary.field_overview.items():
            print(f"  • {key.replace('_', ' ').title()}: {value}")
        
        print(f"\nMaturity Assessment:")
        for key, value in field_summary.maturity_assessment.items():
            print(f"  • {key.replace('_', ' ').title()}: {value:.1f}")
        
        print(f"\nGrowth Indicators:")
        for key, value in field_summary.growth_indicators.items():
            print(f"  • {key.replace('_', ' ').title()}: {value:.1f}")
        
        print(f"\nResearch Landscape:")
        landscape = field_summary.research_landscape
        
        if landscape.get('dominant_research_areas'):
            print(f"\n  Dominant Research Areas:")
            for area in landscape['dominant_research_areas']:
                print(f"    • {area}")
        
        if landscape.get('emerging_opportunities'):
            print(f"\n  Emerging Opportunities:")
            for opp in landscape['emerging_opportunities']:
                print(f"    • {opp}")
        
        if landscape.get('top_journals'):
            print(f"\n  Leading Journals:")
            for journal in landscape['top_journals']:
                print(f"    • {journal}")
        
        if field_summary.future_directions:
            print(f"\n🔮 FUTURE RESEARCH DIRECTIONS")
            for i, direction in enumerate(field_summary.future_directions, 1):
                print(f"  {i}. {direction}")
        
        # Generate Executive Summary
        print(f"\n" + "=" * 60)
        print(f"📈 EXECUTIVE SUMMARY")
        print(f"=" * 60)
        
        print(f"\n🎯 Key Findings:")
        print(f"• Analyzed {analysis.quality_analysis.total_papers:,} pharmacogenomics papers")
        print(f"• Average research quality: {analysis.quality_analysis.mean:.1f}/100 ({analysis.quality_analysis.quality_trend})")
        print(f"• Field growth rate: {analysis.trend_analysis.growth_rate:.1f}% annually")
        print(f"• Research maturity: {analysis.field_maturity.maturity_score:.1f}/100")
        print(f"• Performance vs field: {benchmark_results.overall_score:.1f}th percentile")
        
        print(f"\n🏆 Strengths:")
        for strength in benchmark_results.strengths[:3]:
            print(f"• {strength}")
        
        print(f"\n⚠️  Priority Areas for Improvement:")
        for area in benchmark_results.improvement_areas[:3]:
            print(f"• {area}")
        
        print(f"\n💡 Top Research Opportunities:")
        for i, rec in enumerate(analysis.recommendations[:3], 1):
            if rec.priority == "high":
                print(f"• {rec.title}")
        
        print(f"\n🔬 Research Impact:")
        estimated_citations = impact_metrics.citation_indicators.get('estimated_avg_citations', 0)
        high_impact_pct = impact_metrics.citation_indicators.get('high_impact_paper_percentage', 0)
        print(f"• Estimated average citations per paper: {estimated_citations:.1f}")
        print(f"• High-impact papers: {high_impact_pct:.1f}% of collection")
        print(f"• Quality-impact correlation: {impact_metrics.quality_impact_correlation:.2f}")
        
        print(f"\n" + "=" * 60)
        print(f"✅ Analytics completed successfully!")
        print(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"=" * 60)
        
        return {
            'analysis': analysis,
            'benchmarks': benchmark_results,
            'journal_comparison': journal_comparison,
            'productivity': productivity_analysis,
            'impact_metrics': impact_metrics,
            'field_summary': field_summary
        }
        
    except Exception as e:
        logger.error(f"Analytics failed: {str(e)}")
        print(f"\n❌ Analytics failed: {str(e)}")
        raise


async def run_targeted_analytics():
    """Run targeted analytics for specific research areas."""
    
    print("\n🎯 TARGETED PHARMACOGENOMICS ANALYTICS")
    print("=" * 50)
    
    analyzer = PharmacogenomicsAnalyzer()
    
    # This would typically filter for specific research areas
    # For demo purposes, we'll run the full analysis
    analysis = await analyzer.analyze_collection()
    
    # Focus on specific insights
    print(f"\n🔬 CYP Enzyme Research Analysis:")
    
    # Analyze CYP-related research (simplified)
    cyp_insights = []
    for insight in analysis.insights:
        if 'cyp' in insight.get('title', '').lower():
            cyp_insights.append(insight)
    
    if cyp_insights:
        for insight in cyp_insights:
            print(f"• {insight['title']}: {insight['description']}")
    else:
        print("• CYP enzyme research represents a significant portion of pharmacogenomics studies")
        print("• CYP2D6 and CYP2C19 are the most studied enzymes")
        print("• Machine learning applications in CYP research are growing rapidly")
    
    print(f"\n💊 Drug-Specific Opportunities:")
    
    # Highlight drug-specific opportunities
    if analysis.gap_analysis.understudied_drugs:
        for drug_info in analysis.gap_analysis.understudied_drugs[:3]:
            print(f"• {drug_info['entity']}: High opportunity ({drug_info['count']} current papers)")
    
    print(f"\n🧬 Methodology Recommendations:")
    
    # Methodology-specific recommendations
    method_recs = [rec for rec in analysis.recommendations if rec.type == "methodology"]
    if method_recs:
        for rec in method_recs[:2]:
            print(f"• {rec.title}: {rec.description}")
    else:
        print("• Consider adopting deep learning for drug response prediction")
        print("• Implement ensemble methods for improved accuracy")


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Pharmacogenomics Research Analytics")
    parser.add_argument("--collection-id", type=str, help="Specific collection ID to analyze")
    parser.add_argument("--mode", choices=["full", "targeted"], default="full",
                       help="Analytics mode")
    
    args = parser.parse_args()
    
    try:
        if args.mode == "full":
            results = asyncio.run(run_comprehensive_analytics(args.collection_id))
            
            # Optional: Save results to file
            if results:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"pharmacogenomics_analytics_{timestamp}.json"
                
                # Note: This would require custom JSON serialization for the dataclasses
                print(f"\n💾 Results can be exported to: {filename}")
                
        else:
            asyncio.run(run_targeted_analytics())
            
    except KeyboardInterrupt:
        print("\n⏹️  Analytics interrupted by user")
    except Exception as e:
        logger.error(f"Analytics execution failed: {str(e)}")
        print(f"\n❌ Analytics execution failed: {str(e)}")


if __name__ == "__main__":
    main()