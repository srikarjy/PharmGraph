"""Complete production pipeline example for pharmacogenomics research."""

import asyncio
import logging
import os
from datetime import datetime
from typing import List, Dict, Any

from src.processing.pipeline_orchestrator import PharmacogenomicsPipeline
from src.processing.pipeline_monitor import PipelineMonitor, Alert, AlertLevel
from src.processing.performance_optimizer import PerformanceOptimizer
from src.config.logging_config import setup_logging


# Setup logging
setup_logging()
logger = logging.getLogger(__name__)


def alert_handler(alert: Alert):
    """Handle pipeline alerts."""
    if alert.level == AlertLevel.CRITICAL:
        logger.critical(f"CRITICAL ALERT: {alert.message} - {alert.component}.{alert.metric_name}")
        # In production, you might send notifications, emails, or trigger automated responses
    elif alert.level == AlertLevel.ERROR:
        logger.error(f"ERROR ALERT: {alert.message}")
    elif alert.level == AlertLevel.WARNING:
        logger.warning(f"WARNING: {alert.message}")
    else:
        logger.info(f"INFO: {alert.message}")


async def run_production_pipeline():
    """Run the complete production pharmacogenomics pipeline."""
    
    logger.info("Starting production pharmacogenomics research pipeline")
    
    # Configuration
    email = os.getenv("NCBI_EMAIL", "your-email@example.com")
    api_key = os.getenv("NCBI_API_KEY")  # Optional but recommended
    
    if email == "your-email@example.com":
        logger.error("Please set NCBI_EMAIL environment variable")
        return
    
    # Initialize components
    logger.info("Initializing pipeline components...")
    
    # Pipeline orchestrator with production settings
    pipeline = PharmacogenomicsPipeline(
        email=email,
        api_key=api_key,
        batch_size=200,  # Optimal batch size for NCBI API
        max_concurrent=5,  # Respect API rate limits
        enable_resume=True
    )
    
    # Monitor with production thresholds
    monitor = PipelineMonitor(
        monitoring_interval=30.0,  # Monitor every 30 seconds
        alert_thresholds={
            'error_rate_warning': 10.0,  # 10% error rate warning
            'error_rate_critical': 25.0,  # 25% error rate critical
            'success_rate_warning': 85.0,  # 85% success rate warning
            'success_rate_critical': 70.0,  # 70% success rate critical
            'papers_per_minute_warning': 15.0,  # Minimum 15 papers/min
            'papers_per_minute_critical': 8.0,  # Minimum 8 papers/min
            'quality_score_warning': 55.0,  # Average quality warning
            'quality_score_critical': 40.0,  # Average quality critical
        },
        enable_resource_monitoring=True
    )
    
    # Performance optimizer with production settings
    optimizer = PerformanceOptimizer(
        max_memory_mb=4096,  # 4GB memory limit
        max_concurrent_api=5,
        max_concurrent_db=15,
        target_batch_size=200,
        cache_size=20000  # Large cache for production
    )
    
    # Add alert handler
    monitor.add_alert_callback(alert_handler)
    
    # Define comprehensive research queries
    research_queries = [
        # Core pharmacogenomics queries
        "CYP2D6 AND (pharmacogenomics OR pharmacogenetics) AND (machine learning OR artificial intelligence)",
        "warfarin AND dosing AND (precision medicine OR personalized medicine)",
        "CYP2C19 AND (clopidogrel OR antiplatelet) AND genetic variants",
        "DPYD AND fluorouracil AND toxicity prediction",
        "TPMT AND thiopurine AND adverse drug reactions",
        
        # Advanced ML and AI in pharmacogenomics
        "pharmacogenomics AND deep learning AND drug response prediction",
        "GWAS AND drug metabolism AND machine learning algorithms",
        "pharmacokinetics AND artificial intelligence AND dose optimization",
        "biomarkers AND precision dosing AND computational methods",
        "drug-gene interactions AND predictive modeling",
        
        # Clinical decision support
        "clinical decision support AND pharmacogenomics AND implementation",
        "electronic health records AND pharmacogenetic testing",
        "point-of-care AND genetic testing AND drug prescribing",
        
        # Population pharmacogenomics
        "population pharmacokinetics AND ethnic differences",
        "pharmacogenomics AND health disparities AND ancestry",
        "rare variants AND drug response AND clinical significance"
    ]
    
    logger.info(f"Configured {len(research_queries)} research queries")
    logger.info("Starting monitoring and optimization...")
    
    # Start monitoring
    monitor.start_monitoring()
    
    try:
        # Execute the complete pipeline
        logger.info("Executing complete pharmacogenomics research pipeline...")
        
        start_time = datetime.utcnow()
        
        results = await pipeline.run_full_pipeline(
            search_queries=research_queries,
            max_papers=2000  # Production volume
        )
        
        end_time = datetime.utcnow()
        
        # Generate comprehensive report
        logger.info("Generating pipeline report...")
        
        print("\n" + "="*80)
        print("PHARMACOGENOMICS RESEARCH PIPELINE RESULTS")
        print("="*80)
        
        print(f"\n📊 EXECUTION SUMMARY")
        print(f"Pipeline ID: {results.pipeline_id}")
        print(f"Status: {results.status.value.upper()}")
        print(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print(f"Completed: {end_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print(f"Total Runtime: {results.total_time_minutes:.1f} minutes")
        
        print(f"\n📈 PERFORMANCE METRICS")
        print(f"Papers Collected: {results.papers_collected:,}")
        print(f"Papers Processed: {results.papers_processed:,}")
        print(f"Papers Stored: {results.papers_stored:,}")
        print(f"Success Rate: {results.success_rate:.1f}%")
        print(f"Processing Rate: {results.papers_per_minute:.1f} papers/minute")
        
        print(f"\n🎯 QUALITY METRICS")
        print(f"Average Quality Score: {results.avg_quality_score:.1f}/100")
        print(f"High Quality Papers (≥75): {results.high_quality_papers:,}")
        print(f"Medium Quality Papers (50-74): {results.medium_quality_papers:,}")
        print(f"Low Quality Papers (<50): {results.low_quality_papers:,}")
        
        print(f"\n⚠️  ERROR ANALYSIS")
        print(f"Total Errors: {results.total_errors:,}")
        if results.error_breakdown:
            for error_type, count in results.error_breakdown.items():
                if count > 0:
                    print(f"  {error_type.title()} Errors: {count:,}")
        
        print(f"\n🏆 TOP QUALITY PAPERS")
        if results.top_papers:
            for i, paper in enumerate(results.top_papers[:5], 1):
                print(f"{i}. [{paper['pmid']}] {paper['title']}")
                print(f"   Journal: {paper['journal']} | Quality: {paper['quality_score']}/100")
        
        # Performance optimization report
        perf_report = optimizer.get_performance_report()
        
        print(f"\n⚡ OPTIMIZATION METRICS")
        opt_metrics = perf_report['optimization_metrics']
        print(f"Cache Hit Rate: {opt_metrics['cache_hit_rate']:.1f}%")
        print(f"Memory Usage: {opt_metrics['memory_usage_mb']:.1f} MB")
        print(f"Batch Efficiency: {opt_metrics['batch_efficiency']:.1f}%")
        
        # Monitoring report
        current_metrics = monitor.get_current_metrics()
        if current_metrics:
            print(f"\n📡 MONITORING METRICS")
            print(f"CPU Usage: {current_metrics.cpu_usage:.1f}%")
            print(f"Memory Usage: {current_metrics.memory_usage:.1f}%")
            print(f"API Success Rate: {current_metrics.api_success_rate:.1f}%")
        
        # Active alerts
        active_alerts = monitor.get_active_alerts()
        if active_alerts:
            print(f"\n🚨 ACTIVE ALERTS ({len(active_alerts)})")
            for alert in active_alerts[:5]:  # Show top 5 alerts
                print(f"  [{alert.level.value.upper()}] {alert.message}")
        else:
            print(f"\n✅ NO ACTIVE ALERTS")
        
        # Research insights
        print(f"\n🔬 RESEARCH INSIGHTS")
        if results.papers_stored > 0:
            high_quality_rate = (results.high_quality_papers / results.papers_stored) * 100
            print(f"High-quality research rate: {high_quality_rate:.1f}%")
            
            if results.avg_quality_score >= 70:
                print("✅ Excellent overall research quality detected")
            elif results.avg_quality_score >= 55:
                print("✅ Good overall research quality detected")
            else:
                print("⚠️  Research quality below optimal threshold")
            
            # Estimate research coverage
            coverage_estimate = min(100, (results.papers_collected / 5000) * 100)
            print(f"Estimated pharmacogenomics coverage: {coverage_estimate:.1f}%")
        
        print(f"\n💾 DATA STORAGE")
        print(f"Database records created: {results.papers_stored:,}")
        print(f"Quality metrics stored: {results.papers_stored:,}")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS")
        
        if results.success_rate < 80:
            print("• Consider adjusting API rate limits or batch sizes")
        
        if results.avg_quality_score < 60:
            print("• Review quality scoring thresholds")
            print("• Consider additional quality filters")
        
        if results.papers_per_minute < 10:
            print("• Optimize batch processing parameters")
            print("• Consider increasing concurrent operations")
        
        if perf_report['optimization_metrics']['cache_hit_rate'] < 50:
            print("• Increase cache size for better performance")
        
        if results.papers_stored > 1000:
            print("• Consider implementing data archiving strategy")
            print("• Set up automated quality monitoring")
        
        print("\n" + "="*80)
        print("Pipeline execution completed successfully!")
        print("="*80)
        
        return results
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        
        # Try to get partial results
        pipeline_state = pipeline.get_pipeline_status()
        if pipeline_state:
            partial_report = pipeline.generate_pipeline_report()
            if partial_report:
                print(f"\n⚠️  PARTIAL RESULTS AVAILABLE")
                print(f"Papers processed before failure: {partial_report.papers_processed}")
                print(f"Papers stored before failure: {partial_report.papers_stored}")
        
        raise
        
    finally:
        # Stop monitoring
        monitor.stop_monitoring()
        
        # Final monitoring summary
        error_summary = monitor.get_error_summary(hours=2)
        if error_summary['total_errors'] > 0:
            logger.info(f"Total errors in last 2 hours: {error_summary['total_errors']}")


async def run_targeted_research_pipeline():
    """Run a targeted research pipeline for specific pharmacogenomics topics."""
    
    logger.info("Starting targeted pharmacogenomics research pipeline")
    
    # Focused research queries for specific drug-gene pairs
    targeted_queries = [
        "CYP2D6 AND codeine AND metabolism",
        "VKORC1 AND warfarin AND dose prediction",
        "HLA-B*5701 AND abacavir AND hypersensitivity",
        "DPYD AND 5-fluorouracil AND toxicity",
        "TPMT AND azathioprine AND adverse effects"
    ]
    
    # Initialize with focused settings
    pipeline = PharmacogenomicsPipeline(
        email=os.getenv("NCBI_EMAIL", "your-email@example.com"),
        api_key=os.getenv("NCBI_API_KEY"),
        batch_size=100,
        max_concurrent=3
    )
    
    monitor = PipelineMonitor(monitoring_interval=15.0)
    monitor.start_monitoring()
    
    try:
        results = await pipeline.run_full_pipeline(
            search_queries=targeted_queries,
            max_papers=500  # Smaller, focused dataset
        )
        
        print(f"\n🎯 TARGETED RESEARCH RESULTS")
        print(f"Focused on {len(targeted_queries)} specific drug-gene interactions")
        print(f"Papers collected: {results.papers_collected}")
        print(f"Average quality: {results.avg_quality_score:.1f}/100")
        print(f"High-quality papers: {results.high_quality_papers}")
        
        return results
        
    finally:
        monitor.stop_monitoring()


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Pharmacogenomics Research Pipeline")
    parser.add_argument("--mode", choices=["full", "targeted"], default="full",
                       help="Pipeline execution mode")
    parser.add_argument("--max-papers", type=int, default=1000,
                       help="Maximum papers to process")
    
    args = parser.parse_args()
    
    # Check environment
    if not os.getenv("NCBI_EMAIL"):
        print("❌ Error: NCBI_EMAIL environment variable not set")
        print("Please set your email address: export NCBI_EMAIL='your-email@example.com'")
        return
    
    if not os.getenv("NCBI_API_KEY"):
        print("⚠️  Warning: NCBI_API_KEY not set. Consider getting an API key for better performance.")
        print("Get your API key at: https://www.ncbi.nlm.nih.gov/account/settings/")
    
    # Run pipeline
    try:
        if args.mode == "full":
            asyncio.run(run_production_pipeline())
        else:
            asyncio.run(run_targeted_research_pipeline())
            
    except KeyboardInterrupt:
        print("\n⏹️  Pipeline execution interrupted by user")
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        print(f"\n❌ Pipeline execution failed: {str(e)}")


if __name__ == "__main__":
    main()