#!/usr/bin/env python3
"""Example demonstrating the database system."""

import sys
from datetime import datetime, date
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_config, configure_logging
from src.config.models import LoggingConfig, DatabaseConfig
from src.storage.database import init_database, create_tables, db_session_scope
from src.storage.operations import PaperOperations, QualityOperations, CollectionOperations
from src.storage.models import PaperModel, QualityMetrics, CollectionRun


def main():
    """Demonstrate database functionality."""
    print("=== Genomics Pipeline Database Example ===\n")
    
    # Configure logging
    logging_config = LoggingConfig(
        level="INFO",
        console_output=True,
        structured_logging=False
    )
    configure_logging(logging_config)
    
    # Initialize database
    db_config = DatabaseConfig(
        url="sqlite:///example_genomics.db",
        echo=False
    )
    
    init_database(db_config)
    create_tables()
    
    print("✅ Database initialized and tables created!")
    
    # Example 1: Save papers
    print("\n--- Example 1: Saving Papers ---")
    
    paper_data = {
        'pmid': '12345678',
        'title': 'Machine Learning Applications in Pharmacogenomics: A Comprehensive Review',
        'abstract': 'This paper reviews the current state of machine learning applications in pharmacogenomics, focusing on drug response prediction and personalized medicine approaches.',
        'authors_json': ['Smith, J.A.', 'Johnson, B.C.', 'Williams, D.E.'],
        'journal': 'Nature Genetics',
        'publication_date': date(2024, 1, 15),
        'doi': '10.1038/ng.2024.001',
        'mesh_terms_json': ['Pharmacogenomics', 'Machine Learning', 'Drug Response', 'Precision Medicine'],
        'keywords_json': ['AI', 'precision medicine', 'drug discovery', 'genomics']
    }
    
    try:
        pmid = PaperOperations.save_paper(paper_data)
        print(f"✅ Saved paper: {pmid}")
    except Exception as e:
        print(f"❌ Error saving paper: {e}")
    
    # Example 2: Save quality metrics
    print("\n--- Example 2: Quality Metrics ---")
    
    quality_data = {
        'journal_score': 22.5,  # High-impact journal
        'clinical_relevance_score': 20.0,  # Highly relevant to clinical practice
        'ml_methodology_score': 18.5,  # Good ML methodology
        'recency_score': 20.0,  # Recent publication
        'confidence_level': 0.85,
        'scoring_version': '1.0'
    }
    
    try:
        quality_id = QualityOperations.save_quality_metrics('12345678', quality_data)
        print(f"✅ Saved quality metrics: ID {quality_id}")
        
        # Get the quality metrics back
        quality = QualityOperations.get_quality_by_pmid('12345678')
        if quality:
            print(f"   Total Score: {quality.total_score}/100")
            print(f"   Confidence: {quality.confidence_level}")
    except Exception as e:
        print(f"❌ Error with quality metrics: {e}")
    
    # Example 3: Collection run tracking
    print("\n--- Example 3: Collection Run Tracking ---")
    
    try:
        # Start a collection run
        config_snapshot = {
            'query': 'pharmacogenomics AND machine learning',
            'max_results': 1000,
            'quality_threshold': 60.0
        }
        
        run_id = CollectionOperations.start_collection_run(
            query='pharmacogenomics AND machine learning',
            config_snapshot=config_snapshot
        )
        print(f"✅ Started collection run: ID {run_id}")
        
        # Update progress
        progress = {
            'papers_collected': 150,
            'papers_processed': 120,
            'avg_quality_score': 72.5
        }
        
        CollectionOperations.update_collection_progress(run_id, progress)
        print(f"✅ Updated progress: {progress['papers_collected']} papers collected")
        
        # Complete the run
        summary = {
            'papers_collected': 200,
            'papers_processed': 180,
            'avg_quality_score': 74.2
        }
        
        CollectionOperations.complete_collection_run(run_id, summary)
        print(f"✅ Completed collection run with {summary['papers_collected']} papers")
        
    except Exception as e:
        print(f"❌ Error with collection run: {e}")
    
    # Example 4: Search and query operations
    print("\n--- Example 4: Search Operations ---")
    
    try:
        # Search papers
        filters = {
            'title_contains': 'Machine Learning',
            'date_from': date(2024, 1, 1),
            'order_by': 'publication_date',
            'order_dir': 'desc'
        }
        
        papers = PaperOperations.search_papers(filters, limit=10)
        print(f"✅ Found {len(papers)} papers matching search criteria")
        
        for paper in papers:
            print(f"   - {paper.pmid}: {paper.title[:60]}...")
        
        # Get quality distribution
        distribution = QualityOperations.get_quality_distribution()
        if distribution:
            print(f"\n✅ Quality Distribution:")
            print(f"   Total papers with quality scores: {distribution['total_papers']}")
            print(f"   Average quality score: {distribution['avg_score']:.1f}")
            print(f"   High quality papers (80+): {distribution['quality_ranges']['high_quality']}")
            print(f"   Medium quality papers (60-79): {distribution['quality_ranges']['medium_quality']}")
            print(f"   Low quality papers (<60): {distribution['quality_ranges']['low_quality']}")
        
        # Get collection statistics
        stats = CollectionOperations.get_collection_stats()
        if stats:
            print(f"\n✅ Collection Statistics:")
            print(f"   Total collection runs: {stats['total_runs']}")
            print(f"   Success rate: {stats['success_rate']:.1f}%")
            print(f"   Total papers collected: {stats['total_papers_collected']}")
            print(f"   Average quality score: {stats['avg_quality_score']:.1f}")
        
    except Exception as e:
        print(f"❌ Error with search operations: {e}")
    
    # Example 5: Direct database session usage
    print("\n--- Example 5: Direct Database Access ---")
    
    try:
        with db_session_scope() as session:
            # Count total papers
            paper_count = session.query(PaperModel).count()
            print(f"✅ Total papers in database: {paper_count}")
            
            # Get papers with quality scores
            papers_with_quality = session.query(PaperModel).join(QualityMetrics).count()
            print(f"✅ Papers with quality scores: {papers_with_quality}")
            
            # Get recent collection runs
            recent_runs = session.query(CollectionRun).order_by(
                CollectionRun.collection_start.desc()
            ).limit(5).all()
            
            print(f"✅ Recent collection runs:")
            for run in recent_runs:
                status = run.status.value if hasattr(run.status, 'value') else run.status
                print(f"   - Run {run.id}: {status}, {run.papers_collected} papers")
    
    except Exception as e:
        print(f"❌ Error with direct database access: {e}")
    
    print(f"\n✅ Database example completed successfully!")
    print(f"Database file: example_genomics.db")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())