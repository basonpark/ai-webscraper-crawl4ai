import os
import sys
import psutil
import asyncio
import requests
import time
from xml.etree import ElementTree
from concurrent.futures import ThreadPoolExecutor
from functools import partial

__location__ = os.path.dirname(os.path.abspath(__file__))
__output__ = os.path.join(__location__, "output")

# Append parent directory to system path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from typing import List, Dict, Any, Tuple
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode

def get_optimal_concurrency() -> int:
    """Determine optimal concurrency based on system resources."""
    # Use CPU count and available memory to determine optimal concurrency
    cpu_count = os.cpu_count() or 1
    mem = psutil.virtual_memory()
    mem_gb = mem.available / (1024 * 1024 * 1024)  # Available memory in GB
    
    # Base concurrency on both CPU and memory
    # Rule of thumb: 2 concurrent tasks per CPU core, but capped by available memory
    # Each browser instance can use ~100-200MB at minimum
    cpu_based = cpu_count * 2
    mem_based = max(1, int(mem_gb * 5))  # Allow ~200MB per crawler
    
    # Use the more conservative of the two values
    concurrency = min(cpu_based, mem_based)
    print(f"System has {cpu_count} CPU cores and {mem_gb:.2f}GB available memory")
    print(f"Setting optimal concurrency to {concurrency}")
    return concurrency

async def crawl_parallel(urls: List[str], max_concurrent: int = None):
    print("\n=== Enhanced Parallel Crawling with Adaptive Resource Management ===")
    
    # Auto-determine concurrency if not specified
    if max_concurrent is None:
        max_concurrent = get_optimal_concurrency()
    
    # Track timing and resources
    start_time = time.time()
    process = psutil.Process(os.getpid())
    peak_memory = 0
    metrics = {
        "url_timings": {},
        "batch_timings": [],
        "errors": {}
    }

    def log_memory(prefix: str = ""):
        nonlocal peak_memory
        current_mem = process.memory_info().rss  # in bytes
        if current_mem > peak_memory:
            peak_memory = current_mem
        print(f"{prefix} Current Memory: {current_mem // (1024 * 1024)} MB, Peak: {peak_memory // (1024 * 1024)} MB")

    # Optimized browser config with additional performance flags
    browser_config = BrowserConfig(
        headless=True,
        verbose=False,
        # Additional performance optimizations
        extra_args=[
            "--disable-gpu", 
            "--disable-dev-shm-usage", 
            "--no-sandbox",
            "--disable-extensions",
            "--disable-translate",
            "--disable-sync",
            "--disable-background-networking",
            "--disable-default-apps",
            "--mute-audio",
            "--no-first-run",
            "--no-zygote",
            "--disable-popup-blocking",
            "--blink-settings=imagesEnabled=false"  # Disable image loading for even faster crawling
        ],
    )
    
    # Optimized crawler config with timeouts
    crawl_config = CrawlerRunConfig(
        cache_mode=CacheMode.ENABLED,  # Use caching for better performance
        page_timeout=30000,          # Timeout for page load (in milliseconds)
        wait_until='domcontentloaded', # Don't wait for full page load
        wait_for=None,               # Don't wait for specific elements
        delay_before_return_html=1.0, # Minimal wait time
        verbose=False                # Reduce console output
    )

    # Create the crawler instance
    crawler = AsyncWebCrawler(config=browser_config)
    await crawler.start()
    
    try:
        # Distribute URLs into batches optimally
        success_count = 0
        fail_count = 0
        session_pool = []  # Reuse sessions where possible
        
        # Initialize session pool
        for i in range(min(max_concurrent, 5)):  # Create a smaller pool of reusable sessions
            session_pool.append(f"session_pool_{i}")
        
        # Process in batches
        batch_num = 0
        for i in range(0, len(urls), max_concurrent):
            batch_num += 1
            batch_start = time.time()
            batch = urls[i : i + max_concurrent]
            tasks = []
            
            # Distribute URLs based on estimated complexity
            # Sort URLs by likely complexity (domain or path length can be a simple heuristic)
            sorted_batch = sorted(batch, key=lambda u: len(u))
            
            # Check memory before batch
            log_memory(prefix=f"Before batch {batch_num}: ")
            
            for j, url in enumerate(sorted_batch):
                # Reuse sessions from the pool using round-robin
                session_id = session_pool[j % len(session_pool)]
                
                # Track start time for this URL
                url_key = url[:50] + "..." if len(url) > 50 else url
                metrics["url_timings"][url_key] = {"start": time.time()}
                
                # Create task with timeout handling
                task = asyncio.create_task(
                    crawler.arun(
                        url=url, 
                        config=crawl_config,
                        session_id=session_id
                    )
                )
                tasks.append((url, task))

            # Process all URLs in this batch concurrently
            for url, task in tasks:
                try:
                    url_key = url[:50] + "..." if len(url) > 50 else url
                    result = await asyncio.wait_for(task, timeout=45)  # Overall task timeout
                    metrics["url_timings"][url_key]["end"] = time.time()
                    metrics["url_timings"][url_key]["duration"] = metrics["url_timings"][url_key]["end"] - metrics["url_timings"][url_key]["start"]
                    
                    if result.success:
                        success_count += 1
                    else:
                        fail_count += 1
                        metrics["errors"][url_key] = result.error_message
                        print(f"Failed: {url} - {result.error_message}")
                
                except asyncio.TimeoutError:
                    fail_count += 1
                    metrics["errors"][url_key] = "Task timeout"
                    print(f"Timeout: {url}")
                
                except Exception as e:
                    fail_count += 1
                    metrics["errors"][url_key] = str(e)
                    print(f"Error crawling {url}: {e}")
            
            # Clean up tasks
            for _, task in tasks:
                if not task.done():
                    task.cancel()
            
            # Check memory after batch
            batch_end = time.time()
            metrics["batch_timings"].append(batch_end - batch_start)
            log_memory(prefix=f"After batch {batch_num}: ")
            
            # Optional: Add a small delay between batches to allow for GC
            if batch_num % 3 == 0:  # Every few batches
                await asyncio.sleep(1)
                # Suggest garbage collection
                import gc
                gc.collect()

        # Calculate and log performance metrics
        total_time = time.time() - start_time
        avg_batch_time = sum(metrics["batch_timings"]) / len(metrics["batch_timings"]) if metrics["batch_timings"] else 0
        url_times = [data["duration"] for data in metrics["url_timings"].values() if "duration" in data]
        avg_url_time = sum(url_times) / len(url_times) if url_times else 0
        
        print(f"\nPerformance Summary:")
        print(f"  - Total crawl time: {total_time:.2f} seconds")
        print(f"  - Average batch time: {avg_batch_time:.2f} seconds")
        print(f"  - Average URL time: {avg_url_time:.2f} seconds")
        print(f"  - URLs per second: {(success_count + fail_count) / total_time:.2f}")
        
        print(f"\nCrawl Summary:")
        print(f"  - Successfully crawled: {success_count}")
        print(f"  - Failed: {fail_count}")
        print(f"  - Success rate: {success_count / (success_count + fail_count) * 100:.1f}%")

    finally:
        print("\nClosing crawler...")
        await crawler.close()
        # Final memory log
        log_memory(prefix="Final: ")
        print(f"\nPeak memory usage (MB): {peak_memory // (1024 * 1024)}")
        print(f"Total execution time: {time.time() - start_time:.2f} seconds")

def get_pydantic_ai_docs_urls():
    """
    Fetches all URLs from the Pydantic AI documentation.
    Uses the sitemap (https://ai.pydantic.dev/sitemap.xml) to get these URLs.
    
    Returns:
        List[str]: List of URLs
    """            
    sitemap_url = "https://ai.pydantic.dev/sitemap.xml"
    try:
        # Use a session for better performance
        session = requests.Session()
        response = session.get(sitemap_url, timeout=10)
        response.raise_for_status()
        
        # Parse the XML
        root = ElementTree.fromstring(response.content)
        
        # Extract all URLs from the sitemap
        # The namespace is usually defined in the root element
        namespace = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
        urls = [loc.text for loc in root.findall('.//ns:loc', namespace)]
        
        return urls
    except Exception as e:
        print(f"Error fetching sitemap: {e}")
        return []        

async def main():
    urls = get_pydantic_ai_docs_urls()
    if urls:
        print(f"Found {len(urls)} URLs to crawl")
        # Let the function determine optimal concurrency
        await crawl_parallel(urls)
    else:
        print("No URLs found to crawl")    

if __name__ == "__main__":
    asyncio.run(main())
