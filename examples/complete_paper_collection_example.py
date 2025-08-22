"""Complete paper collection pipeline example - from search to database storage."""

import asyncio
import json
from pathlib import Path
import sys
from typing import List

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_ingestion.ncbi_search import NCBISearchClient
from src.data_ingestion.ncbi_fetch import NCBIFetchClient
from src.data_ingestion.pubmed_parser import PubMedXMLParser
from src.data_ingestion.pgx_queries import PharmacogenomicsQueryBuilder
from src.data_ingestion.data_models import PaperMetadata
from src.storage.models import PaperModel
from src.storage.database import get_database_manager, db_session_scope
from src.config import get_logger


logger = get_logger(__name__)


async def demonstrate_complete_pipeline():
    """Demonstrate complete paper collection pipeline."""
    print("=== Complete Paper Collection Pipeline ===\n")
    
    # Configuration
    email = "your-email@example.com"  # Required by NCBI
    api_key = None  # Optional: add your NCBI API key
    
    # Initialize components
    search_client = NCBISearchClient(email=email, api_key=api_key)
    fetch_client = NCBIFetchClient(email=email, api_key=api_key)
    parser = PubMedXMLParser()
    query_builder = PharmacogenomicsQueryBuilder()
    
    print(f"Initialized pipeline components for: {email}")
    print()
    
    async with search_client, fetch_client:
        # Phase 1: Search for relevant papers
        print("Phase 1: Searching for Pharmacogenomics Papers")
        print("-" * 50)
        
        # Build sophisticated query
        query = query_builder.build_pgx_ml_query(
            drug_terms=["warfarin", "clopidogrel"],
            gene_terms=["CYP2D6", "CYP2C19"],
            ml_terms=["machine learning", "artificial intelligence"],
            date_range=("2020/01/01", "2024/01/01")
        )
        
        print(f"Search query: {query}")
        print()
        
        # Get search count
        total_count = await search_client.get_search_count(query)
        print(f"Total papers available: {total_count}")
        
        if total_count == 0:
            print("No papers found. Exiting.")
            return
        
        # Search for papers (limit for demo)
        max_papers = min(20, total_count)
        print(f"Retrieving first {max_papers} papers...")
        
        pmids = await search_client.search_papers(
            query=query,
            max_results=max_papers
        )
        
        print(f"Found {len(pmids)} PMIDs: {pmids[:5]}...")
        print()
        
        # Phase 2: Fetch detailed paper information
        print("Phase 2: Fetching Detailed Paper Information")
        print("-" * 50)
        
        print(f"Fetching details for {len(pmids)} papers...")
        
        xml_responses, fetch_stats = await fetch_client.fetch_paper_details(
            pmids=pmids,
            batch_size=10,  # Small batches for demo
            max_concurrent=2
        )
        
        print(f"Fetch Statistics:")
        print(f"  Total PMIDs: {fetch_stats.total_pmids}")
        print(f"  Successful fetches: {fetch_stats.successful_fetches}")
        print(f"  Failed fetches: {fetch_stats.failed_fetches}")
        print(f"  Total time: {fetch_stats.total_time:.2f}s")
        print(f"  Papers per second: {fetch_stats.papers_per_second:.1f}")
        print(f"  XML responses received: {len(xml_responses)}")
        print()
        
        # Phase 3: Parse XML responses
        print("Phase 3: Parsing XML Responses")
        print("-" * 50)
        
        all_papers = []
        parsing_errors = 0
        
        for i, xml_response in enumerate(xml_responses):
            try:
                papers = parser.parse_pubmed_xml(xml_response)
                all_papers.extend(papers)
                print(f"  Batch {i+1}: Parsed {len(papers)} papers")
            except Exception as e:
                parsing_errors += 1
                logger.error(f"Error parsing XML batch {i+1}: {str(e)}")
        
        print(f"\nParsing Summary:")
        print(f"  Total papers parsed: {len(all_papers)}")
        print(f"  Parsing errors: {parsing_errors}")
        print()
        
        # Phase 4: Analyze parsed papers
        print("Phase 4: Analyzing Parsed Papers")
        print("-" * 50)
        
        if all_papers:
            # Sample paper analysis
            sample_paper = all_papers[0]
            
            print(f"Sample Paper Analysis:")
            print(f"  PMID: {sample_paper.pmid}")
            print(f"  Title: {sample_paper.title[:80]}...")
            print(f"  Authors: {len(sample_paper.authors)}")
            print(f"  First Author: {sample_paper.first_author.full_name if sample_paper.first_author else 'Unknown'}")
            print(f"  Journal: {sample_paper.journal.title if sample_paper.journal else 'Unknown'}")
            print(f"  Publication Year: {sample_paper.publication_year}")
            print(f"  Has Abstract: {sample_paper.has_abstract}")
            print(f"  Abstract Length: {len(sample_paper.abstract)} characters")
            print(f"  MeSH Terms: {len(sample_paper.mesh_terms)}")
            print(f"  Keywords: {len(sample_paper.keywords)}")
            print(f"  Publication Types: {', '.join(sample_paper.publication_types)}")
            print(f"  Is Clinical Trial: {sample_paper.is_clinical_trial}")
            print(f"  Is Review: {sample_paper.is_review}")
            print()
            
            # Show some MeSH terms
            if sample_paper.mesh_terms:
                print(f"  Sample MeSH Terms:")
                for mesh_term in sample_paper.mesh_terms[:5]:
                    major = " (Major)" if mesh_term.major_topic else ""
                    print(f"    - {mesh_term.descriptor_name}{major}")
                print()
            
            # Show some keywords
            if sample_paper.keywords:
                print(f"  Keywords: {', '.join(sample_paper.keywords[:10])}")
                print()
        
        # Phase 5: Database storage simulation
        print("Phase 5: Database Storage Preparation")
        print("-" * 50)
        
        # Convert to database models (simulation)
        db_papers = []
        conversion_errors = 0
        
        for paper in all_papers:
            try:
                # Convert PaperMetadata to PaperModel
                db_paper = convert_to_db_model(paper)
                db_papers.append(db_paper)
            except Exception as e:
                conversion_errors += 1
                logger.error(f"Error converting paper {paper.pmid}: {str(e)}")
        
        print(f"Database Conversion:")
        print(f"  Papers ready for storage: {len(db_papers)}")
        print(f"  Conversion errors: {conversion_errors}")
        print()
        
        # Phase 6: Quality analysis
        print("Phase 6: Quality Analysis")
        print("-" * 50)
        
        # Analyze paper quality characteristics
        quality_stats = analyze_paper_quality(all_papers)
        
        print(f"Quality Statistics:")
        print(f"  Papers with abstracts: {quality_stats['with_abstracts']}/{len(all_papers)} ({quality_stats['abstract_percentage']:.1f}%)")
        print(f"  Papers with MeSH terms: {quality_stats['with_mesh']}/{len(all_papers)} ({quality_stats['mesh_percentage']:.1f}%)")
        print(f"  Papers with keywords: {quality_stats['with_keywords']}/{len(all_papers)} ({quality_stats['keyword_percentage']:.1f}%)")
        print(f"  Clinical trials: {quality_stats['clinical_trials']}")
        print(f"  Reviews: {quality_stats['reviews']}")
        print(f"  Average authors per paper: {quality_stats['avg_authors']:.1f}")
        print(f"  Average MeSH terms per paper: {quality_stats['avg_mesh_terms']:.1f}")
        print(f"  Publication year range: {quality_stats['year_range']}")
        print()
        
        # Phase 7: Pharmacogenomics relevance analysis
        print("Phase 7: Pharmacogenomics Relevance Analysis")
        print("-" * 50)
        
        pgx_stats = analyze_pharmacogenomics_relevance(all_papers)
        
        print(f"Pharmacogenomics Relevance:")
        print(f"  Papers with pharmacogenomics MeSH: {pgx_stats['pgx_mesh_count']}")
        print(f"  Papers with ML/AI terms: {pgx_stats['ml_terms_count']}")
        print(f"  Papers with drug names: {pgx_stats['drug_names_count']}")
        print(f"  Papers with gene names: {pgx_stats['gene_names_count']}")
        print(f"  High relevance papers: {pgx_stats['high_relevance_count']}")
        print()
        
        if pgx_stats['top_mesh_terms']:
            print("Top MeSH Terms:")
            for term, count in pgx_stats['top_mesh_terms'][:10]:
                print(f"  - {term}: {count}")
            print()
        
        # Final summary
        print("Pipeline Summary")
        print("-" * 50)
        print(f"✅ Search: Found {total_count} total papers, retrieved {len(pmids)} PMIDs")
        print(f"✅ Fetch: Retrieved {fetch_stats.successful_fetches}/{fetch_stats.total_pmids} papers ({fetch_stats.papers_per_second:.1f} papers/sec)")
        print(f"✅ Parse: Extracted {len(all_papers)} complete paper records")
        print(f"✅ Convert: Prepared {len(db_papers)} papers for database storage")
        print(f"✅ Quality: {quality_stats['abstract_percentage']:.1f}% have abstracts, {quality_stats['mesh_percentage']:.1f}% have MeSH terms")
        print(f"✅ Relevance: {pgx_stats['high_relevance_count']} papers highly relevant to pharmacogenomics")
        print()
        
        return all_papers


def convert_to_db_model(paper: PaperMetadata) -> PaperModel:
    """Convert PaperMetadata to database model.
    
    Args:
        paper: Paper metadata
        
    Returns:
        PaperModel: Database model
    """
    # Convert authors to JSON
    authors_json = [
        {
            "last_name": author.last_name,
            "first_name": author.first_name,
            "initials": author.initials,
            "affiliation": author.affiliation,
            "orcid": author.orcid
        }
        for author in paper.authors
    ]
    
    # Convert MeSH terms to JSON
    mesh_terms_json = [
        {
            "descriptor_name": term.descriptor_name,
            "qualifier_name": term.qualifier_name,
            "major_topic": term.major_topic
        }
        for term in paper.mesh_terms
    ]
    
    return PaperModel(
        pmid=paper.pmid,
        title=paper.title,
        abstract=paper.abstract,
        authors_json=authors_json,
        journal=paper.journal.title if paper.journal else None,
        publication_date=paper.publication_date.date() if paper.publication_date else None,
        doi=paper.doi,
        mesh_terms_json=mesh_terms_json,
        keywords_json=paper.keywords
    )


def analyze_paper_quality(papers: List[PaperMetadata]) -> dict:
    """Analyze quality characteristics of papers.
    
    Args:
        papers: List of paper metadata
        
    Returns:
        dict: Quality statistics
    """
    if not papers:
        return {}
    
    with_abstracts = sum(1 for p in papers if p.has_abstract)
    with_mesh = sum(1 for p in papers if p.mesh_terms)
    with_keywords = sum(1 for p in papers if p.keywords)
    clinical_trials = sum(1 for p in papers if p.is_clinical_trial)
    reviews = sum(1 for p in papers if p.is_review)
    
    total_authors = sum(len(p.authors) for p in papers)
    total_mesh_terms = sum(len(p.mesh_terms) for p in papers)
    
    years = [p.publication_year for p in papers if p.publication_year]
    year_range = f"{min(years)}-{max(years)}" if years else "Unknown"
    
    return {
        'with_abstracts': with_abstracts,
        'abstract_percentage': (with_abstracts / len(papers)) * 100,
        'with_mesh': with_mesh,
        'mesh_percentage': (with_mesh / len(papers)) * 100,
        'with_keywords': with_keywords,
        'keyword_percentage': (with_keywords / len(papers)) * 100,
        'clinical_trials': clinical_trials,
        'reviews': reviews,
        'avg_authors': total_authors / len(papers),
        'avg_mesh_terms': total_mesh_terms / len(papers),
        'year_range': year_range
    }


def analyze_pharmacogenomics_relevance(papers: List[PaperMetadata]) -> dict:
    """Analyze pharmacogenomics relevance of papers.
    
    Args:
        papers: List of paper metadata
        
    Returns:
        dict: Relevance statistics
    """
    from collections import Counter
    
    # Pharmacogenomics-related terms
    pgx_mesh_terms = [
        'pharmacogenetics', 'pharmacogenomics', 'precision medicine',
        'personalized medicine', 'drug metabolism', 'cytochrome p-450'
    ]
    
    ml_terms = [
        'machine learning', 'artificial intelligence', 'deep learning',
        'neural network', 'random forest', 'support vector'
    ]
    
    drug_names = [
        'warfarin', 'clopidogrel', 'tamoxifen', 'codeine', 'tramadol',
        'simvastatin', 'atorvastatin', 'metoprolol'
    ]
    
    gene_names = [
        'cyp2d6', 'cyp2c19', 'cyp2c9', 'cyp3a4', 'vkorc1', 'dpyd', 'tpmt'
    ]
    
    # Count papers with relevant terms
    pgx_mesh_count = 0
    ml_terms_count = 0
    drug_names_count = 0
    gene_names_count = 0
    high_relevance_count = 0
    
    all_mesh_terms = []
    
    for paper in papers:
        # Check MeSH terms
        paper_mesh = [term.descriptor_name.lower() for term in paper.mesh_terms]
        all_mesh_terms.extend(paper_mesh)
        
        has_pgx_mesh = any(term in ' '.join(paper_mesh) for term in pgx_mesh_terms)
        if has_pgx_mesh:
            pgx_mesh_count += 1
        
        # Check title and abstract for ML terms
        text_content = (paper.title + ' ' + paper.abstract).lower()
        has_ml_terms = any(term in text_content for term in ml_terms)
        if has_ml_terms:
            ml_terms_count += 1
        
        # Check for drug names
        has_drug_names = any(drug in text_content for drug in drug_names)
        if has_drug_names:
            drug_names_count += 1
        
        # Check for gene names
        has_gene_names = any(gene in text_content for gene in gene_names)
        if has_gene_names:
            gene_names_count += 1
        
        # High relevance: has PGx terms AND (ML terms OR drug names OR gene names)
        if has_pgx_mesh and (has_ml_terms or has_drug_names or has_gene_names):
            high_relevance_count += 1
    
    # Top MeSH terms
    mesh_counter = Counter(all_mesh_terms)
    top_mesh_terms = mesh_counter.most_common(20)
    
    return {
        'pgx_mesh_count': pgx_mesh_count,
        'ml_terms_count': ml_terms_count,
        'drug_names_count': drug_names_count,
        'gene_names_count': gene_names_count,
        'high_relevance_count': high_relevance_count,
        'top_mesh_terms': top_mesh_terms
    }


async def demonstrate_database_storage(papers: List[PaperMetadata]):
    """Demonstrate database storage (requires database setup).
    
    Args:
        papers: List of papers to store
    """
    print("\n=== Database Storage Demonstration ===\n")
    
    try:
        # Initialize database
        db_manager = get_database_manager()
        db_manager.initialize()
        
        print(f"Storing {len(papers)} papers in database...")
        
        stored_count = 0
        error_count = 0
        
        with db_session_scope() as session:
            for paper in papers:
                try:
                    # Convert to database model
                    db_paper = convert_to_db_model(paper)
                    
                    # Check if paper already exists
                    existing = session.query(PaperModel).filter_by(pmid=paper.pmid).first()
                    if existing:
                        print(f"  Paper {paper.pmid} already exists, skipping")
                        continue
                    
                    # Add to session
                    session.add(db_paper)
                    stored_count += 1
                    
                except Exception as e:
                    error_count += 1
                    logger.error(f"Error storing paper {paper.pmid}: {str(e)}")
        
        print(f"Database Storage Results:")
        print(f"  Papers stored: {stored_count}")
        print(f"  Errors: {error_count}")
        
    except Exception as e:
        print(f"Database storage failed: {str(e)}")
        print("Note: Database storage requires proper database setup")


async def main():
    """Main demonstration function."""
    try:
        papers = await demonstrate_complete_pipeline()
        
        if papers:
            # Optionally demonstrate database storage
            # await demonstrate_database_storage(papers)
            pass
        
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        logger.error(f"Demo failed: {str(e)}")
        print(f"\nDemo failed: {str(e)}")
        print("Make sure to:")
        print("1. Replace 'your-email@example.com' with your actual email")
        print("2. Optionally add your NCBI API key for higher rate limits")
        print("3. Ensure you have internet connectivity")


if __name__ == "__main__":
    print("Complete Paper Collection Pipeline Demo")
    print("======================================")
    print()
    print("This demo showcases the complete pipeline:")
    print("1. Search for pharmacogenomics papers using sophisticated queries")
    print("2. Fetch detailed paper information from NCBI")
    print("3. Parse XML responses into structured metadata")
    print("4. Analyze paper quality and relevance")
    print("5. Prepare data for database storage")
    print("6. Generate comprehensive statistics")
    print()
    
    asyncio.run(main())