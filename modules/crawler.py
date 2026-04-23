import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from time import mktime
from typing import List, Optional

import feedparser
import httpx
from bs4 import BeautifulSoup
from pydantic import ValidationError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Import our universal Data Contract for articles
from models.article import Article

# Initialize a logger as requested by Raspberry Pi environment standards
logger = logging.getLogger(__name__)

class NewsCrawler:
    """
    An asynchronous crawler responsible for fetching RSS feeds, extracting new
    article links, and scraping the main article content concurrently.
    Built heavily on asyncio and httpx to prevent blocking on the Raspberry Pi.
    """
    
    def __init__(self, sources_path: str = "config/sources.json"):
        """
        Initializes the crawler and loads target RSS feeds from the configuration file.
        """
        self.sources_path = Path(sources_path)
        self.urls = self._load_sources()
        
    def _load_sources(self) -> List[str]:
        """
        Loads the list of RSS feed URLs from the local JSON file.
        
        Returns:
            List[str]: A list of target URLs.
        """
        if not self.sources_path.exists():
            logger.error(f"Sources file not found at {self.sources_path}")
            return []
            
        with open(self.sources_path, 'r', encoding='utf-8') as f:
            try:
                sources = json.load(f)
                logger.info(f"Loaded {len(sources)} sources from {self.sources_path}")
                return sources
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse sources file: {e}")
                return []

    # Implement tenacity retry mechanism for network instability (e.g., bad Pi Wi-Fi)
    # Exponential backoff: starts at 2s, increases multiplier, up to 10s between retries.
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((httpx.RequestError, httpx.HTTPStatusError))
    )
    async def _fetch_url(self, client: httpx.AsyncClient, url: str) -> str:
        """
        Asynchronously fetches a URL using httpx with resilience against network hiccups.
        
        Args:
            client (httpx.AsyncClient): The session client.
            url (str): The URL to GET.
            
        Returns:
            str: The raw HTML or XML response string.
        """
        logger.debug(f"Fetching URL: {url}")
        
        # We rely entirely on the AsyncClient's global configuration
        # for realistic headers, redirects, and timeouts.
        response = await client.get(url)
        
        # Raise an exception for HTTP error statuses (triggers the tenacity retry)
        response.raise_for_status()
        return response.text

    async def _fetch_and_parse_feed(self, client: httpx.AsyncClient, url: str) -> List[dict]:
        """
        Fetches an RSS feed asynchronously via httpx, then passes the text to feedparser.
        (Note: Calling feedparser.parse(url) directly is IO-blocking and not allowed).
        """
        try:
            # 1. Non-blocking network request
            feed_content = await self._fetch_url(client, url)
            
            # 2. String parsing in-memory (fast, CPU-bound but minimal)
            parsed_feed = feedparser.parse(feed_content)
            
            entries = []
            for entry in parsed_feed.entries:
                entries.append({
                    "title": entry.get("title", "No Title"),
                    "link": entry.get("link", ""),
                    "published_parsed": entry.get("published_parsed", None)
                })
            return entries
        except Exception as e:
            logger.error(f"Failed to fetch or parse feed at {url}: {e.__class__.__name__} - {e}")
            return []

    async def _extract_article_content(self, client: httpx.AsyncClient, entry: dict) -> Optional[Article]:
        """
        For a given entry URL, fetches the HTML, cleans it using BeautifulSoup,
        extracts the text, and constructs a validated Article Pydantic model.
        """
        link = entry.get("link")
        if not link:
            return None
            
        try:
            # 1. Fetch raw HTML
            html = await self._fetch_url(client, link)
            soup = BeautifulSoup(html, "html.parser")
            
            # 2. Remove boilerplate DOM elements (headers, form tags, styling, navbars, footers)
            elements_to_ignore = ["header", "footer", "nav", "aside", "script", "style", "meta", "object", "iframe"]
            for unwanted in soup(elements_to_ignore):
                unwanted.decompose()
                
            # 3. Extract the core text. Try extracting <p> tags first, 
            # fallback to generalized stripped strings.
            paragraphs = soup.find_all("p")
            if paragraphs:
                raw_text = " ".join([p.get_text(separator=" ", strip=True) for p in paragraphs])
            else:
                raw_text = " ".join(soup.stripped_strings)
            
            # Discard articles that failed to parse meaningful content
            if not raw_text or len(raw_text) < 150:
                logger.warning(f"Skipping {link}: Extracted content too short.")
                return None
            
            # 4. Resolve the correct publication datetime
            published_parsed = entry.get("published_parsed")
            if published_parsed:
                dt = datetime.fromtimestamp(mktime(published_parsed))
            else:
                # Default to now if the RSS feed didn't provide a date
                dt = datetime.now()
                
            # 5. Build and validate our Data Contract
            article = Article(
                source_url=link,
                title=entry.get("title"),
                raw_text=raw_text,
                published_at=dt,
                summary=None,
                tags=[]
            )
            return article
            
        except ValidationError as e:
            logger.error(f"Pydantic validation failed for {link}: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to completely extract {link}: {e.__class__.__name__} - {e}")
            return None

    async def crawl(self, limit: Optional[int] = None) -> List[Article]:
        """
        Main orchestrator. Loads feeds, extracts entry links, limits concurrent
        article scraping, and finally returns a list of valid Article objects.
        """
        valid_articles = []
        
        # Configure the global client with realistic modern browser headers
        # to bypass basic bot mitigation across all requests.
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
            "Accept-Language": "en-US,en;q=0.9",
            "Upgrade-Insecure-Requests": "1"
        }
        
        # Centralizing requests into an AsyncClient leverages connection pooling
        async with httpx.AsyncClient(headers=headers, follow_redirects=True, timeout=15.0) as client:
            
            # Step 1: Concurrently grab all RSS feeds
            feed_tasks = [self._fetch_and_parse_feed(client, url) for url in self.urls]
            feeds_results = await asyncio.gather(*feed_tasks, return_exceptions=True)
            
            # Combine items across all feeds
            all_entries = []
            for result in feeds_results:
                if isinstance(result, list):
                    all_entries.extend(result)
            
            logger.info(f"Discovered {len(all_entries)} potential articles from RSS feeds.")
            if limit:
                all_entries = all_entries[:limit]
            
            # Step 2: Concurrently scrape article HTML, but tightly bounded!
            # A Semaphore prevents flooding the network or overloading Raspberry Pi's RAM
            semaphore = asyncio.Semaphore(15)
            
            async def sem_extract(entry):
                async with semaphore:
                    return await self._extract_article_content(client, entry)
                    
            extract_tasks = [sem_extract(entry) for entry in all_entries]
            articles_results = await asyncio.gather(*extract_tasks, return_exceptions=True)
            
            # Step 3: Tally results and filter None/Exception instances
            for result in articles_results:
                if isinstance(result, Article):
                    valid_articles.append(result)
                    
        logger.info(f"Successfully scraped, parsed, and validated {len(valid_articles)} articles.")
        return valid_articles

# Ad-hoc testing block
if __name__ == "__main__":
    # Local execution verification
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    crawler = NewsCrawler()
    
    # Python 3.11+ event loop runner
    scraped_articles = asyncio.run(crawler.crawl())
    
    for idx, art in enumerate(scraped_articles[:5]):
        print(f"{idx+1}. {art.title} ({len(art.raw_text)} chars)")
