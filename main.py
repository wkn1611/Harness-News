"""
Main Pipeline for the Hermes AI News Agent.

Orchestrates the 3-Tier architecture:
- Tier 1: Ingestion (Jina Reader API)
- Tier 2: Intelligence Engine (Groq LLM)
- Tier 3: The Vault (MongoDB Atlas)
"""
import logging
from datetime import datetime
from rich.console import Console
from rich.logging import RichHandler

# Internal imports
from modules.ingestor import process_sources
from modules.summarizer import analyze_article
from modules.database import setup_database, save_article

# Configure Rich Logging
console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, console=console)]
)
logger = logging.getLogger("HermesPipeline")

def run_pipeline():
    """
    Executes the end-to-end news radar pipeline.
    """
    logger.info("[bold blue]Starting Hermes AI News Pipeline...[/bold blue]")
    
    # 1. Setup Database (Tier 3)
    logger.info("Initializing Tier 3 (The Vault) Database...")
    try:
        setup_database()
    except Exception as e:
        logger.error(f"Critical Database Setup Error: {e}. Exiting pipeline.")
        return

    # 2. Ingest Articles (Tier 1)
    logger.info("Starting Tier 1 (Ingestor)...")
    raw_articles = process_sources()
    
    if not raw_articles:
        logger.warning("No articles ingested. Exiting.")
        return
        
    logger.info(f"Ingested {len(raw_articles)} articles.")
    
    success_count = 0
    failure_count = 0

    # 3. Process & Save Each Article
    for i, article in enumerate(raw_articles, 1):
        title = article.get("title", "Untitled")
        logger.info(f"[yellow]Processing ({i}/{len(raw_articles)}):[/yellow] {title}")
        
        # Extract markdown content
        markdown_content = article.get("raw_markdown_content")
        if not markdown_content:
            logger.warning(f"No markdown content for '{title}'. Skipping.")
            failure_count += 1
            continue

        # Run Tier 2 Analysis
        logger.debug(f"Analyzing '{title}' via Tier 2 Intelligence Engine...")
        intelligence = analyze_article(markdown_content)
        
        if intelligence:
            # Attach intelligence to the article dict
            article["intelligence"] = intelligence
            
            # Convert published_date string to a datetime object for MongoDB TTL index
            date_str = article.get("published_date")
            if date_str:
                try:
                    # Clean the ISO string if it ends with 'Z'
                    if date_str.endswith('Z'):
                        date_str = date_str[:-1] + '+00:00'
                    article["published_date"] = datetime.fromisoformat(date_str)
                except Exception as e:
                    logger.warning(f"Failed to parse date '{date_str}': {e}. Using current UTC time.")
                    article["published_date"] = datetime.utcnow()
            else:
                article["published_date"] = datetime.utcnow()
            
            # Save to MongoDB
            if save_article(article):
                success_count += 1
                logger.info(f"[green]Saved successfully:[/green] {title}")
            else:
                failure_count += 1
                logger.error(f"[red]Failed to save:[/red] {title}")
        else:
            failure_count += 1
            logger.error(f"[red]Analysis failed for:[/red] {title}. Skipping save.")

    logger.info(
        f"[bold green]Pipeline Complete![/bold green] "
        f"Saved: {success_count} | Failed/Skipped: {failure_count}"
    )

if __name__ == "__main__":
    try:
        run_pipeline()
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user.")
