"""Example demonstrating NCBI API client usage."""

import asyncio
import json
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_ingestion.ncbi_client import NCBIClient, NCBIDatabase
from src.config import get_logger


logger = get_logger(__name__)


async def demonstrate_ncbi_search():
    """Demonstrate NCBI PubMed search functionality."""
    print("=== NCBI API Client Demonstration ===\n")
    
    # Initialize client (replace with your email)
    email = "your-email@example.com"  # Required by NCBI
    api_key = None  # Optional: add your NCBI API key for higher rate limits
    
    client = NCBIClient(
        email=email,
        api_key=api_key,
        tool="genomics-pipeline-demo",
        max_retries=3,
        timeout=30.0
    )
    
    print(f"Initialized NCBI client for: {email}")
    print(f"Rate limit: {'10 req/s (with API key)' if api_key else '3 req/s (no API key)'}")
    print()
    
    async with client:
        # Get client status
        status = client.get_client_status()
        print("Client Status:")
        print(f"  Session active: {status['session_active']}")
        print(f"  Rate limiter: {status['rate_limiter']['current_rate']:.1f} req/s")
        print()
        
        # Example 1: Search for genomics papers
        print("1. Searching PubMed for genomics papers...")
        search_response = await client.search(
            database=NCBIDatabase.PUBMED,
            query="genomics AND machine learning AND 2023[PDAT]",
            max_results=10,
            start=0
        )
        
        if search_response.success:
            search_data = search_response.data
            if 'esearchresult' in search_data:
                id_list = search_data['esearchresult'].get('idlist', [])
                count = search_data['esearchresult'].get('count', '0')
                print(f"  Found {count} papers, showing first {len(id_list)} IDs:")
                for i, pmid in enumerate(id_list[:5], 1):
                    print(f"    {i}. PMID: {pmid}")
                print()
                
                # Example 2: Fetch details for first few papers
                if id_list:
                    print("2. Fetching paper details...")
                    fetch_response = await client.fetch(
                        database=NCBIDatabase.PUBMED,
                        ids=id_list[:3],  # Get first 3 papers
                        return_type="abstract",
                        return_mode="xml"
                    )
                    
                    if fetch_response.success:
                        print(f"  Successfully fetched {len(id_list[:3])} paper details")
                        print(f"  Response format: {fetch_response.response_format.value}")
                        print(f"  Data length: {len(fetch_response.data)} characters")
                        
                        # Show first 200 characters of XML
                        xml_preview = fetch_response.data[:200] + "..." if len(fetch_response.data) > 200 else fetch_response.data
                        print(f"  XML preview: {xml_preview}")
                    else:
                        print(f"  Error fetching papers: {fetch_response.error_message}")
                    print()
            else:
                print("  Unexpected search response format")
                print(f"  Response: {search_data}")
        else:
            print(f"  Search failed: {search_response.error_message}")
            print()
        
        # Example 3: Get database information
        print("3. Getting PubMed database information...")
        info_response = await client.get_database_info(NCBIDatabase.PUBMED)
        
        if info_response.success:
            info_data = info_response.data
            if 'einforesult' in info_data:
                db_info = info_data['einforesult']
                print(f"  Database: {db_info.get('dbname', 'Unknown')}")
                print(f"  Description: {db_info.get('description', 'No description')}")
                print(f"  Record count: {db_info.get('count', 'Unknown')}")
                print(f"  Last update: {db_info.get('lastupdate', 'Unknown')}")
            else:
                print("  Unexpected info response format")
        else:
            print(f"  Database info failed: {info_response.error_message}")
        print()
        
        # Example 4: Demonstrate rate limiting
        print("4. Demonstrating rate limiting...")
        rate_metrics = client.rate_limiter.get_metrics()
        print(f"  Current rate: {rate_metrics['current_rate']:.1f} req/s")
        print(f"  Total requests: {rate_metrics['metrics']['total_requests']}")
        print(f"  Successful requests: {rate_metrics['metrics']['successful_requests']}")
        print(f"  Success rate: {rate_metrics['metrics']['success_rate']:.1f}%")
        
        if rate_metrics['metrics']['rate_limited_requests'] > 0:
            print(f"  Rate limited requests: {rate_metrics['metrics']['rate_limited_requests']}")
        
        print()
        
        # Example 5: Test error handling with invalid query
        print("5. Testing error handling with invalid database...")
        try:
            # This should fail gracefully
            invalid_response = await client._make_request_with_retry(
                'GET', 'esearch.fcgi', 
                params={'db': 'invalid_database', 'term': 'test'}
            )
            
            if not invalid_response.success:
                print(f"  Handled error gracefully: {invalid_response.error_message}")
                print(f"  Status code: {invalid_response.status_code}")
            else:
                print("  Unexpected success with invalid database")
        except Exception as e:
            print(f"  Exception caught: {str(e)}")
        
        print()
        
        # Final status
        final_status = client.get_client_status()
        final_metrics = final_status['rate_limiter']['metrics']
        print("Final Statistics:")
        print(f"  Total requests made: {final_metrics['total_requests']}")
        print(f"  Success rate: {final_metrics['success_rate']:.1f}%")
        print(f"  Average wait time: {final_metrics['average_wait_time']:.3f}s")
        if final_metrics['adaptive_adjustments'] > 0:
            print(f"  Rate adjustments: {final_metrics['adaptive_adjustments']}")


async def demonstrate_rate_limiting():
    """Demonstrate rate limiting behavior."""
    print("\n=== Rate Limiting Demonstration ===\n")
    
    from src.data_ingestion.rate_limiter import AdaptiveRateLimiter
    
    # Create a rate limiter with low limits for demonstration
    limiter = AdaptiveRateLimiter(
        default_rate=2.0,  # 2 requests per second
        has_api_key=False,
        max_burst=5
    )
    
    print("Testing rate limiter (2 req/s limit)...")
    
    start_time = asyncio.get_event_loop().time()
    
    # Make several requests quickly
    for i in range(6):
        await limiter.acquire()
        elapsed = asyncio.get_event_loop().time() - start_time
        print(f"  Request {i+1} at {elapsed:.2f}s")
        
        # Simulate successful request
        limiter.record_success()
    
    total_time = asyncio.get_event_loop().time() - start_time
    print(f"\nCompleted 6 requests in {total_time:.2f}s")
    print(f"Expected minimum time: {6/2:.1f}s (at 2 req/s)")
    
    # Show metrics
    metrics = limiter.get_metrics()
    print(f"Rate limiter metrics:")
    print(f"  Total requests: {metrics['metrics']['total_requests']}")
    print(f"  Success rate: {metrics['metrics']['success_rate']:.1f}%")


async def main():
    """Main demonstration function."""
    try:
        await demonstrate_ncbi_search()
        await demonstrate_rate_limiting()
        
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
    print("NCBI API Client Demo")
    print("===================")
    print()
    print("Note: This demo requires:")
    print("- A valid email address (required by NCBI)")
    print("- Internet connectivity")
    print("- Optional: NCBI API key for higher rate limits")
    print()
    
    asyncio.run(main())