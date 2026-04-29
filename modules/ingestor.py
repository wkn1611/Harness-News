"""
Tier 1 — Ingestion Layer for the Hermes AI News Agent.

Replaces the legacy httpx + BeautifulSoup scraper with the Jina Reader API,
which cleanly extracts Markdown (including images and code blocks) from any
URL — including arXiv PDFs and JS-heavy web pages.

Designed for Raspberry Pi 4: synchronous requests, conservative concurrency,
and generous timeouts to handle unstable Wi-Fi gracefully.
"""
import json
import logging
from datetime import datetime
from pathlib import Path
from time import mktime
from typing import Any, Optional

import feedparser
import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Jina Reader API Configuration
# ---------------------------------------------------------------------------
JINA_READER_PREFIX = "https://r.jina.ai/"
JINA_TIMEOUT = 30  # seconds — generous for arXiv PDF extraction
JINA_HEADERS = {
    # Jina Reader accepts a plain Accept header; text/plain returns raw markdown
    "Accept": "text/plain",
}

# Default path for the new structured sources file
DEFAULT_SOURCES_PATH = "sources.json"


# ---------------------------------------------------------------------------
# Core Extraction Function
# ---------------------------------------------------------------------------
def get_markdown_from_url(url: str) -> Optional[str]:
    """
    Calls the Jina Reader API to extract clean Markdown from a target URL.

    The Jina Reader handles JS rendering, PDF extraction, and layout cleanup
    internally — no need for BeautifulSoup or headless browsers on our side.

    Args:
        url: The original article or paper URL to extract content from.

    Returns:
        The raw Markdown string (with images and code blocks intact),
        or None if extraction fails.
    """
    jina_url = f"{JINA_READER_PREFIX}{url}"
    logger.debug(f"Requesting Jina Reader: {jina_url}")

    try:
        response = requests.get(
            jina_url,
            headers=JINA_HEADERS,
            timeout=JINA_TIMEOUT,
        )

        # Check for HTTP-level failures from the Jina proxy
        if response.status_code != 200:
            logger.warning(
                f"Jina returned HTTP {response.status_code} for {url} — skipping."
            )
            return None

        raw_markdown = response.text.strip()

        # Guard against empty or near-empty responses
        if not raw_markdown or len(raw_markdown) < 100:
            logger.warning(
                f"Jina returned empty/too-short content for {url} "
                f"({len(raw_markdown)} chars) — skipping."
            )
            return None

        logger.info(f"Extracted {len(raw_markdown)} chars from {url}")
        return raw_markdown

    except requests.Timeout:
        logger.error(f"Timeout ({JINA_TIMEOUT}s) fetching {url} via Jina — skipping.")
        return None
    except requests.ConnectionError as e:
        logger.error(f"Connection error for {url}: {e} — skipping.")
        return None
    except requests.RequestException as e:
        logger.error(f"Request failed for {url}: {e} — skipping.")
        return None


# ---------------------------------------------------------------------------
# Feed Parsing & Source Processing
# ---------------------------------------------------------------------------
def _load_sources(sources_path: str = DEFAULT_SOURCES_PATH) -> list[dict[str, str]]:
    """
    Reads the structured sources.json file from disk.

    Returns:
        A list of source dicts, each containing 'url', 'category_tag', and 'type'.
    """
    path = Path(sources_path)
    if not path.exists():
        logger.error(f"Sources file not found: {path.resolve()}")
        return []

    with open(path, "r", encoding="utf-8") as f:
        try:
            sources = json.load(f)
            logger.info(f"Loaded {len(sources)} sources from {path}")
            return sources
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {path}: {e}")
            return []


def _parse_published_date(entry: Any) -> str:
    """
    Extracts a publication date string from a feedparser entry.
    Falls back to the current UTC timestamp if the field is missing.

    Args:
        entry: A single feedparser entry object.

    Returns:
        An ISO-8601 formatted date string.
    """
    published_parsed = entry.get("published_parsed")
    if published_parsed:
        try:
            return datetime.fromtimestamp(mktime(published_parsed)).isoformat()
        except (ValueError, OverflowError):
            pass

    # Fallback: use the raw 'published' string if present
    if entry.get("published"):
        return entry["published"]

    return datetime.utcnow().isoformat()


def process_sources(
    sources_path: str = DEFAULT_SOURCES_PATH,
    limit_per_feed: Optional[int] = None,
) -> list[dict[str, Any]]:
    """
    Main ingestion entrypoint. Reads sources.json, parses each RSS feed,
    extracts article content via Jina Reader, and returns structured dicts
    ready for the downstream pipeline (filter → summarizer → database).

    Args:
        sources_path:   Path to the sources JSON file.
        limit_per_feed: Optional cap on entries per feed (useful for testing).

    Returns:
        A list of dicts with keys:
            - title:                str
            - url:                  str
            - published_date:       str (ISO-8601)
            - category_tag:         str
            - raw_markdown_content: str
    """
    sources = _load_sources(sources_path)
    if not sources:
        return []

    results: list[dict[str, Any]] = []

    for source in sources:
        feed_url = source["url"]
        category_tag = source.get("category_tag", "Uncategorized")
        source_type = source.get("type", "rss")

        # Currently only RSS feeds are supported; future types (e.g., "api")
        # can be handled here with an if/elif chain.
        if source_type != "rss":
            logger.warning(f"Unsupported source type '{source_type}' for {feed_url} — skipping.")
            continue

        logger.info(f"Parsing RSS feed: {feed_url} [{category_tag}]")

        # feedparser.parse() handles both URLs and raw XML strings
        feed = feedparser.parse(feed_url)

        if feed.bozo:
            logger.warning(
                f"feedparser reported a problem with {feed_url}: "
                f"{feed.bozo_exception}"
            )

        entries = feed.entries
        if limit_per_feed:
            entries = entries[:limit_per_feed]

        logger.info(f"Found {len(entries)} entries in [{category_tag}]")

        for entry in entries:
            title = entry.get("title", "Untitled")
            link = entry.get("link", "")
            published_date = _parse_published_date(entry)

            if not link:
                logger.debug(f"Entry '{title}' has no link — skipping.")
                continue

            # Extract full article content via Jina Reader API
            raw_markdown = get_markdown_from_url(link)
            if raw_markdown is None:
                # Jina extraction failed — already logged inside the function
                continue

            results.append({
                "title": title,
                "url": link,
                "published_date": published_date,
                "category_tag": category_tag,
                "raw_markdown_content": raw_markdown,
            })

    logger.info(
        f"Ingestion complete. {len(results)} articles extracted from "
        f"{len(sources)} sources."
    )
    return results


# ---------------------------------------------------------------------------
# Ad-hoc CLI testing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    articles = process_sources(limit_per_feed=2)

    print(f"\n{'='*60}")
    print(f"Total articles extracted: {len(articles)}")
    print(f"{'='*60}\n")

    for i, article in enumerate(articles, 1):
        print(f"[{i}] {article['title']}")
        print(f"    URL:      {article['url']}")
        print(f"    Date:     {article['published_date']}")
        print(f"    Category: {article['category_tag']}")
        print(f"    Content:  {len(article['raw_markdown_content'])} chars")
        print(f"    Preview:  {article['raw_markdown_content'][:120]}...")
        print()
