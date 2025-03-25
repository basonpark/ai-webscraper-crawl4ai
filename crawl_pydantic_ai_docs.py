import os
import sys
import json
import asyncio
import requests
import time
import argparse
import logging
from xml.etree import ElementTree
from typing import List, Dict, Any, TypeVar, Callable, Awaitable, Optional
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import urlparse
from dotenv import load_dotenv
from functools import wraps
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio
import traceback

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from openai import AsyncOpenAI
from supabase import create_client, Client

# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Crawl Pydantic AI documentation and store in Supabase")
    parser.add_argument("--chunk-size", type=int, default=5000, help="Size of text chunks (default: 5000)")
    parser.add_argument("--max-concurrent", type=int, default=5, help="Maximum concurrent crawls (default: 5)")
    parser.add_argument("--model", type=str, help="OpenAI model to use (overrides env variable)")
    parser.add_argument("--embedding-model", type=str, default="text-embedding-3-small", 
                        help="Embedding model to use (default: text-embedding-3-small)")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode with more verbose output")
    parser.add_argument("--sitemap-url", type=str, default="https://ai.pydantic.dev/sitemap.xml", 
                        help="URL of the sitemap to crawl")
    parser.add_argument("--specific-urls", type=str, nargs="+", help="Specific URLs to crawl")
    parser.add_argument("--retry-max", type=int, default=3, help="Maximum number of retries for API calls")
    parser.add_argument("--skip-existing", action="store_true", 
                        help="Skip URLs that already exist in the database")
    parser.add_argument("--log-file", type=str, help="Path to log file")
    parser.add_argument("--log-level", type=str, default="INFO", 
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Logging level")
    return parser.parse_args()

# Global args variable
args = parse_args()

# Setup logging
def setup_logging():
    log_level = getattr(logging, args.log_level)
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Add file handler if log file specified
    if args.log_file:
        file_handler = logging.FileHandler(args.log_file)
        file_handler.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(file_handler)
    
    return logging.getLogger(__name__)

# Initialize logger
logger = setup_logging()

load_dotenv()

logger.info("Starting Pydantic AI documentation crawler")

if args.debug:
    logger.debug(f"SUPABASE_URL: {os.getenv('SUPABASE_URL')}")
    
# If URL contains postgresql://, extract project ID and construct proper URL
supabase_url = os.getenv("SUPABASE_URL")
if supabase_url and "postgresql://" in supabase_url:
    project_id = supabase_url.split("@")[1].split(".")[0]
    supabase_url = f"https://{project_id}.supabase.co"
    if args.debug:
        logger.debug(f"Fixed Supabase URL: {supabase_url}")

# Initialize OpenAI and Supabase clients
logger.info("Initializing OpenAI and Supabase clients")
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
supabase: Client = create_client(
    supabase_url,
    os.getenv("SUPABASE_SERVICE_KEY")
)

# Stats for tracking progress
stats = {
    "urls_processed": 0,
    "urls_skipped": 0,
    "urls_failed": 0,
    "chunks_processed": 0,
    "chunks_stored": 0,
    "start_time": None,
    "end_time": None
}

# Define a type variable for the retry decorator
T = TypeVar('T')

# Retry decorator for async functions
async def retry_async(func: Callable[..., Awaitable[T]], max_retries: int = None, base_delay: float = 1.0) -> T:
    """Retry an async function with exponential backoff."""
    if max_retries is None:
        max_retries = args.retry_max
        
    retries = 0
    while True:
        try:
            return await func()
        except Exception as e:
            retries += 1
            if retries > max_retries:
                logger.error(f"Failed after {max_retries} retries: {str(e)}")
                logger.debug(f"Exception details: {traceback.format_exc()}")
                raise
            
            delay = base_delay * (2 ** (retries - 1))  # Exponential backoff
            logger.warning(f"Retry {retries}/{max_retries} after {delay:.2f}s due to: {str(e)}")
            await asyncio.sleep(delay)

@dataclass
class ProcessedChunk:
    url: str
    chunk_number: int
    title: str
    summary: str
    content: str
    metadata: Dict[str, Any]
    embedding: List[float]

def chunk_text(text: str, chunk_size: Optional[int] = None) -> List[str]:
    """Split text into chunks, respecting document structure like headers and code blocks."""
    if chunk_size is None:
        chunk_size = args.chunk_size
        
    chunks = []
    start = 0
    text_length = len(text)
    
    # Headers in markdown (# to ######)
    header_patterns = ['# ', '## ', '### ', '#### ', '##### ', '###### ']
    
    # Track if we're inside a code block
    in_code_block = False
    code_fence_marker = '```'
    
    while start < text_length:
        # Calculate end position
        end = start + chunk_size
        
        # If we're at the end of the text, just take what's left
        if end >= text_length:
            chunks.append(text[start:].strip())
            break
        
        # Get potential chunk
        potential_chunk = text[start:end]
        
        # Check for code blocks within this chunk
        code_blocks = []
        pos = 0
        while True:
            # Find next code fence
            fence_pos = potential_chunk.find(code_fence_marker, pos)
            if fence_pos == -1:
                break
            code_blocks.append(fence_pos)
            pos = fence_pos + len(code_fence_marker)
        
        # If we find an odd number of code fences, we're breaking inside a code block
        if len(code_blocks) % 2 == 1:
            # Find the last code fence in the entire text
            next_fence = text.find(code_fence_marker, start + code_blocks[-1] + len(code_fence_marker))
            if next_fence != -1 and next_fence < start + chunk_size * 1.5:  # Allow up to 50% more for code blocks
                end = next_fence + len(code_fence_marker)
            else:
                # If the next fence is too far, find another suitable break point
                pass
        
        # Look for headers as natural breaking points (prefer to start new chunks at headers)
        for header in header_patterns:
            # Find last header in the first 70% of the potential chunk
            last_header_pos = -1
            search_end = int(chunk_size * 0.7)
            pos = 0
            
            while pos < search_end:
                header_pos = potential_chunk.find('\n' + header, pos)
                if header_pos == -1:
                    break
                last_header_pos = header_pos
                pos = header_pos + len(header) + 1
            
            # If we found a header in a good position, break there
            if last_header_pos > chunk_size * 0.3:  # Only break if past 30% of chunk_size
                end = start + last_header_pos + 1  # Include the newline before header
                break
        
        # If no header found, try paragraph breaks
        if end == start + chunk_size:
            para_break = potential_chunk.rfind('\n\n')
            if para_break > chunk_size * 0.3:  # Only break if past 30% of chunk_size
                end = start + para_break + 2  # Include both newlines
        
        # If no paragraph break, try line breaks
        if end == start + chunk_size:
            line_break = potential_chunk.rfind('\n')
            if line_break > chunk_size * 0.7:  # Only break if well into the chunk
                end = start + line_break + 1  # Include the newline
        
        # If no line break, try sentence breaks
        if end == start + chunk_size:
            sentence_patterns = ['. ', '! ', '? ']
            for pattern in sentence_patterns:
                sentence_break = potential_chunk.rfind(pattern)
                if sentence_break > chunk_size * 0.7:
                    end = start + sentence_break + len(pattern)
                    break
        
        # Extract chunk and clean it up
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        # Move start position for next chunk
        start = end
    
    return chunks

async def get_title_and_summary(chunk: str, url: str) -> Dict[str, str]:
    """Extract title and summary using GPT-4."""
    system_prompt = """You are an AI that extracts titles and summaries from documentation chunks.
    Return a JSON object with 'title' and 'summary' keys.
    For the title: If this seems like the start of a document, extract its title. If it's a middle chunk, derive a descriptive title.
    For the summary: Create a concise summary of the main points in this chunk.
    Keep both title and summary concise but informative."""
    
    async def _get_title_summary():
        model = args.model if args.model else os.getenv("LLM_MODEL", "gpt-4o-mini")
        logger.debug(f"Getting title and summary for chunk from {url} using model {model}")
        response = await openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"URL: {url}\n\nContent:\n{chunk[:1000]}..."}  # Send first 1000 chars for context
            ],
            response_format={ "type": "json_object" }
        )
        return json.loads(response.choices[0].message.content)
    
    try:
        return await retry_async(_get_title_summary)
    except Exception as e:
        logger.error(f"Error getting title and summary for {url}: {str(e)}")
        return {"title": "Error processing title", "summary": "Error processing summary"}

async def get_embedding(text: str) -> List[float]:
    """Get embedding vector from OpenAI."""
    async def _get_embedding():
        logger.debug(f"Getting embedding using model {args.embedding_model}")
        response = await openai_client.embeddings.create(
            model=args.embedding_model,
            input=text
        )
        return response.data[0].embedding
    
    try:
        return await retry_async(_get_embedding)
    except Exception as e:
        logger.error(f"Error getting embedding: {str(e)}")
        return [0] * 1536  # Return zero vector on error

async def process_chunk(chunk: str, chunk_number: int, url: str) -> ProcessedChunk:
    """Process a single chunk of text."""
    # Get title and summary
    logger.debug(f"Processing chunk {chunk_number} for {url}")
    extracted = await get_title_and_summary(chunk, url)
    
    # Get embedding
    embedding = await get_embedding(chunk)
    
    # Create metadata
    metadata = {
        "source": "pydantic_ai_docs",
        "chunk_size": len(chunk),
        "crawled_at": datetime.now(timezone.utc).isoformat(),
        "url_path": urlparse(url).path
    }
    
    stats["chunks_processed"] += 1
    
    return ProcessedChunk(
        url=url,
        chunk_number=chunk_number,
        title=extracted['title'],
        summary=extracted['summary'],
        content=chunk,  # Store the original chunk content
        metadata=metadata,
        embedding=embedding
    )

async def insert_chunk(chunk: ProcessedChunk):
    """Insert a processed chunk into Supabase."""
    async def _insert_chunk():
        data = {
            "url": chunk.url,
            "chunk_number": chunk.chunk_number,
            "title": chunk.title,
            "summary": chunk.summary,
            "content": chunk.content,
            "metadata": chunk.metadata,
            "embedding": chunk.embedding
        }
        
        result = supabase.table("site_pages").insert(data).execute()
        logger.info(f"Inserted chunk {chunk.chunk_number} for {chunk.url}")
        stats["chunks_stored"] += 1
        return result
    
    try:
        return await retry_async(_insert_chunk, max_retries=3)
    except Exception as e:
        logger.error(f"Error inserting chunk {chunk.chunk_number} for {chunk.url}: {str(e)}")
        return None

async def process_and_store_document(url: str, markdown: str):
    """Process a document and store its chunks in parallel."""
    # Split into chunks
    chunks = chunk_text(markdown)
    logger.info(f"Split {url} into {len(chunks)} chunks")
    
    # Process chunks in parallel with progress bar
    tasks = [
        process_chunk(chunk, i, url) 
        for i, chunk in enumerate(chunks)
    ]
    processed_chunks = await tqdm_asyncio.gather(*tasks, desc=f"Processing {urlparse(url).path}")
    
    # Store chunks in parallel
    insert_tasks = [
        insert_chunk(chunk) 
        for chunk in processed_chunks
    ]
    await tqdm_asyncio.gather(*insert_tasks, desc=f"Storing {urlparse(url).path}")

async def crawl_parallel(urls: List[str], max_concurrent: Optional[int] = None):
    """Crawl multiple URLs in parallel with a concurrency limit."""
    if max_concurrent is None:
        max_concurrent = args.max_concurrent
        
    logger.info(f"Starting crawler with max concurrency: {max_concurrent}")
    
    browser_config = BrowserConfig(
        headless=True,
        verbose=args.debug,
        extra_args=["--disable-gpu", "--disable-dev-shm-usage", "--no-sandbox"],
    )
    crawl_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)

    # Create the crawler instance
    logger.info("Initializing web crawler")
    crawler = AsyncWebCrawler(config=browser_config)
    await crawler.start()

    try:
        # Create a semaphore to limit concurrency
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_url(url: str):
            async with semaphore:
                # Skip if already in database and skip_existing flag is set
                if args.skip_existing:
                    existing = supabase.table("site_pages").select("url").eq("url", url).execute()
                    if existing.data:
                        logger.info(f"Skipping already processed URL: {url}")
                        stats["urls_skipped"] += 1
                        return
                        
                logger.info(f"Crawling: {url}")
                result = await crawler.arun(
                    url=url,
                    config=crawl_config,
                    session_id="session1"
                )
                if result.success:
                    logger.info(f"Successfully crawled: {url}")
                    await process_and_store_document(url, result.markdown_v2.raw_markdown)
                    stats["urls_processed"] += 1
                else:
                    logger.error(f"Failed: {url} - Error: {result.error_message}")
                    stats["urls_failed"] += 1
        
        # Process all URLs in parallel with limited concurrency and progress bar
        with tqdm(total=len(urls), desc="URLs processed") as pbar:
            # Create task completion callback to update progress bar
            async def process_url_with_progress(url):
                try:
                    await process_url(url)
                finally:
                    pbar.update(1)
            
            # Create and gather all tasks
            tasks = [process_url_with_progress(url) for url in urls]
            await asyncio.gather(*tasks)
            
    finally:
        logger.info("Closing crawler")
        await crawler.close()

def get_pydantic_ai_docs_urls() -> List[str]:
    """Get URLs from Pydantic AI docs sitemap."""
    sitemap_url = args.sitemap_url
    logger.info(f"Fetching sitemap from {sitemap_url}")
    try:
        response = requests.get(sitemap_url)
        response.raise_for_status()
        
        # Parse the XML
        root = ElementTree.fromstring(response.content)
        
        # Extract all URLs from the sitemap
        namespace = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
        urls = [loc.text for loc in root.findall('.//ns:loc', namespace)]
        
        logger.info(f"Found {len(urls)} URLs in sitemap")
        return urls
    except Exception as e:
        logger.error(f"Error fetching sitemap: {str(e)}")
        logger.debug(traceback.format_exc())
        return []

def print_summary():
    """Print a summary of the crawl results."""
    if not stats["start_time"] or not stats["end_time"]:
        return
    
    duration = stats["end_time"] - stats["start_time"]
    duration_str = str(duration).split('.')[0]  # Remove microseconds
    
    logger.info("=" * 50)
    logger.info("CRAWL SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Duration: {duration_str}")
    logger.info(f"URLs processed: {stats['urls_processed']}")
    logger.info(f"URLs skipped: {stats['urls_skipped']}")
    logger.info(f"URLs failed: {stats['urls_failed']}")
    logger.info(f"Chunks processed: {stats['chunks_processed']}")
    logger.info(f"Chunks stored: {stats['chunks_stored']}")
    
    if stats["urls_processed"] > 0:
        logger.info(f"Average chunks per URL: {stats['chunks_processed'] / stats['urls_processed']:.2f}")
    
    if duration.total_seconds() > 0:
        logger.info(f"URLs per second: {stats['urls_processed'] / duration.total_seconds():.2f}")
        logger.info(f"Chunks per second: {stats['chunks_processed'] / duration.total_seconds():.2f}")
    
    logger.info("=" * 50)

async def main():
    stats["start_time"] = datetime.now(timezone.utc)
    
    if args.specific_urls:
        urls = args.specific_urls
        logger.info(f"Using {len(urls)} specific URLs provided via command line")
    else:
        # Get URLs from Pydantic AI docs
        urls = get_pydantic_ai_docs_urls()
        
    if not urls:
        logger.error("No URLs found to crawl")
        return
    
    logger.info(f"Found {len(urls)} URLs to crawl")
    await crawl_parallel(urls)
    
    stats["end_time"] = datetime.now(timezone.utc)
    print_summary()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Crawler stopped by user")
        stats["end_time"] = datetime.now(timezone.utc)
        print_summary()
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Unhandled exception: {str(e)}")
        logger.debug(traceback.format_exc())
        stats["end_time"] = datetime.now(timezone.utc)
        print_summary()
        sys.exit(1)
