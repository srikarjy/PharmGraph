"""Example demonstrating pharmacogenomics search functionality."""

import asyncio
import json
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_ingestion.ncbi_search import NCBISearchClient
from src.data_ingestion.pgx_queries import PharmacogenomicsQueryBuilder, QueryTemplate
from src.data_ingestion.search_processor import SearchResultProcessor
from src.config import get_logger


logger = get_logger(__name__)


async def demonstrate_pharmacogenomics_search():
    """Demonstrate pharmacogenomics search capabilities."""
    print("=== Pharmacogenomics Search Demonstration ===\n")
    
    # Initialize components
    email = "your-email@example.com"  # Required by NCBI
    api_key = None  # Optional: add your NCBI API key
    
    search_client = NCBISearchClient(
        email=email,
        api_key=api_key,
        tool="pharmacogenomics-search-demo"
    )
    
    query_builder = PharmacogenomicsQueryBuilder()
    processor = SearchResultProcessor(enable_caching=True)
    
    print(f"Initialized components for: {email}")
    print(f"Available query templates: {len(query_builder.get_available_templates())}")
    print()
    
    async with search_client:
        # Example 1: Basic pharmacogenomics + machine learning query
        print("1. Basic Pharmacogenomics + Machine Learning Search")
        print("-" * 50)
        
        pgx_ml_query = query_builder.build_pgx_ml_query(
            drug_terms=["warfarin", "clopidogrel"],
            gene_terms=["CYP2D6", "CYP2C19"],
            ml_terms=["machine learning", "neural network"],
            date_range=("2020/01/01", "2024/01/01")
        )
        
        print(f"Query: {pgx_ml_query}")
        print()
        
        # Get search count first
        count = await search_client.get_search_count(pgx_ml_query)
        print(f"Total papers found: {count}")
        
        if count > 0:
            # Search for papers
            pmids = await search_client.search_papers(
                query=pgx_ml_query,
                max_results=10
            )
            print(f"Retrieved {len(pmids)} PMIDs: {pmids[:5]}...")
        print()
        
        # Example 2: CYP enzyme + AI query
        print("2. CYP Enzyme + Artificial Intelligence Search")
        print("-" * 50)
        
        cyp_ai_query = query_builder.build_cyp_ai_query(
            cyp_enzymes=["CYP2D6", "CYP3A4"],
            ai_terms=["artificial intelligence", "deep learning"]
        )
        
        print(f"Query: {cyp_ai_query}")
        
        count = await search_client.get_search_count(cyp_ai_query)
        print(f"Total papers found: {count}")
        print()
        
        # Example 3: Drug dosing algorithm query
        print("3. Drug Dosing Algorithm Search")
        print("-" * 50)
        
        dosing_query = query_builder.build_drug_dosing_query(
            drugs=["warfarin", "tacrolimus"],
            algorithm_types=["dosing algorithm", "dose prediction"]
        )
        
        print(f"Query: {dosing_query}")
        
        count = await search_client.get_search_count(dosing_query)
        print(f"Total papers found: {count}")
        print()
        
        # Example 4: Template-based query
        print("4. Template-Based Query (Clinical Trials)")
        print("-" * 50)
        
        clinical_query = query_builder.build_template_query(
            QueryTemplate.CLINICAL_TRIALS_PGX,
            custom_params={"phase": "III"}
        )
        
        print(f"Query: {clinical_query}")
        
        count = await search_client.get_search_count(clinical_query)
        print(f"Total papers found: {count}")
        print()
        
        # Example 5: Advanced search with filters
        print("5. Advanced Search with Filters")
        print("-" * 50)
        
        base_query = query_builder.build_pgx_ml_query()
        
        # Add high-impact journal filter
        journal_filter = query_builder.get_high_impact_journal_filter(
            ["Nature", "Science", "Cell"]
        )
        
        # Add recent years filter
        recent_filter = query_builder.get_recent_years_filter(years_back=3)
        
        # Combine filters
        advanced_query = query_builder.combine_queries([
            base_query,
            journal_filter,
            recent_filter
        ], operator="AND")
        
        print(f"Advanced query (truncated): {advanced_query[:100]}...")
        
        count = await search_client.get_search_count(advanced_query)
        print(f"Total papers found: {count}")
        print()
        
        # Example 6: Batched search for large result sets
        print("6. Batched Search for Large Result Sets")
        print("-" * 50)
        
        large_query = query_builder.build_pgx_ml_query(use_mesh=True)
        
        print(f"Query: {large_query}")
        
        # Get total count
        total_count = await search_client.get_search_count(large_query)
        print(f"Total available results: {total_count}")
        
        if total_count > 0:
            # Demonstrate batched retrieval (limit to 50 for demo)
            max_to_retrieve = min(50, total_count)
            
            print(f"Retrieving first {max_to_retrieve} results in batches...")
            
            batched_pmids = await search_client.search_batched(
                query=large_query,
                batch_size=20,
                max_total=max_to_retrieve
            )
            
            print(f"Retrieved {len(batched_pmids)} PMIDs in batches")
            print(f"First 5 PMIDs: {batched_pmids[:5]}")
        print()
        
        # Example 7: Search result processing
        print("7. Search Result Processing and Ranking")
        print("-" * 50)
        
        # Get detailed search response
        detailed_response = await search_client.search_detailed(
            query=pgx_ml_query,
            max_results=20
        )
        
        print(f"Search response:")
        print(f"  Results: {len(detailed_response.results)}")
        print(f"  Total count: {detailed_response.total_count}")
        print(f"  Has more: {detailed_response.has_more}")
        print(f"  Search time: {detailed_response.search_time:.3f}s")
        
        # Process results (simulate raw NCBI response)
        raw_results = {
            "esearchresult": {
                "idlist": [result.pmid for result in detailed_response.results],
                "count": str(detailed_response.total_count)
            }
        }
        
        processed_results, stats = await processor.process_complete_search(
            raw_results=raw_results,
            query=pgx_ml_query,
            database="pubmed"
        )
        
        print(f"\nProcessing statistics:")
        print(f"  Total results: {stats.total_results}")
        print(f"  Unique results: {stats.unique_results}")
        print(f"  Duplicates removed: {stats.duplicates_removed}")
        print(f"  Processing time: {stats.processing_time:.3f}s")
        print(f"  Cache hits: {stats.cache_hits}")
        print(f"  Cache misses: {stats.cache_misses}")
        print()
        
        # Example 8: Pharmacogene category queries
        print("8. Pharmacogene Category Queries")
        print("-" * 50)
        
        # Get genes by category
        cyp_genes = query_builder.get_pharmacogenes_by_category("cyp")
        transporter_genes = query_builder.get_pharmacogenes_by_category("transporter")
        
        print(f"CYP genes: {cyp_genes}")
        print(f"Transporter genes: {transporter_genes}")
        
        # Build category-specific query
        cyp_query = query_builder.build_cyp_ai_query(
            cyp_enzymes=cyp_genes[:3],  # Use first 3 CYP genes
            include_substrates=True
        )
        
        count = await search_client.get_search_count(cyp_query)
        print(f"CYP-specific query results: {count}")
        print()
        
        # Final statistics
        print("Final Statistics")
        print("-" * 50)
        
        client_status = search_client.get_client_status()
        rate_metrics = client_status['rate_limiter']['metrics']
        
        print(f"Total API requests: {rate_metrics['total_requests']}")
        print(f"Success rate: {rate_metrics['success_rate']:.1f}%")
        print(f"Average wait time: {rate_metrics['average_wait_time']:.3f}s")
        
        processor_stats = processor.get_processing_stats()
        if processor_stats.get('cache_stats'):
            cache_stats = processor_stats['cache_stats']
            print(f"Cache entries: {cache_stats['total_entries']}")


async def demonstrate_query_combinations():
    """Demonstrate advanced query combination techniques."""
    print("\n=== Advanced Query Combination Techniques ===\n")
    
    builder = PharmacogenomicsQueryBuilder()
    
    # Example 1: Multi-drug, multi-gene query
    print("1. Multi-Drug, Multi-Gene Query")
    print("-" * 40)
    
    anticoagulant_drugs = builder.drug_categories["anticoagulants"]
    cardiovascular_drugs = builder.drug_categories["cardiovascular"]
    
    multi_drug_query = builder.build_pgx_ml_query(
        drug_terms=anticoagulant_drugs[:2] + cardiovascular_drugs[:2],
        gene_terms=["CYP2D6", "CYP2C19", "VKORC1"],
        ml_terms=["machine learning", "predictive modeling"]
    )
    
    print(f"Multi-drug query: {multi_drug_query[:100]}...")
    print()
    
    # Example 2: Adverse reaction prediction
    print("2. Adverse Reaction Prediction Query")
    print("-" * 40)
    
    adr_query = builder.build_adverse_reactions_ml_query(
        reaction_types=["hepatotoxicity", "cardiotoxicity"],
        prediction_focus=True
    )
    
    print(f"ADR prediction query: {adr_query[:100]}...")
    print()
    
    # Example 3: Precision medicine combination
    print("3. Precision Medicine Combination")
    print("-" * 40)
    
    precision_queries = [
        builder.build_precision_medicine_query(focus_area="drug_interactions"),
        builder.build_precision_medicine_query(focus_area="adverse_events"),
        builder.build_precision_medicine_query(focus_area="efficacy")
    ]
    
    combined_precision = builder.combine_queries(precision_queries, operator="OR")
    
    print(f"Combined precision medicine query: {combined_precision[:100]}...")
    print()
    
    # Example 4: Template combinations
    print("4. Template Combinations")
    print("-" * 40)
    
    template_queries = [
        builder.build_template_query(QueryTemplate.PGX_ML_BASIC),
        builder.build_template_query(QueryTemplate.CYP_AI),
        builder.build_template_query(QueryTemplate.PHARMACOKINETICS_ML)
    ]
    
    combined_templates = builder.combine_queries(template_queries, operator="OR")
    
    print(f"Combined template query: {combined_templates[:100]}...")
    print()


async def main():
    """Main demonstration function."""
    try:
        await demonstrate_pharmacogenomics_search()
        await demonstrate_query_combinations()
        
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
    print("Pharmacogenomics Search Demo")
    print("============================")
    print()
    print("This demo showcases:")
    print("- Domain-specific query building for pharmacogenomics research")
    print("- Advanced search techniques with filters and templates")
    print("- Batched searching for large result sets")
    print("- Search result processing and caching")
    print("- Multi-gene, multi-drug query combinations")
    print()
    
    asyncio.run(main())