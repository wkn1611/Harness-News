import asyncio
import json
import logging
import time
from datetime import datetime

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn

# Internal imports
import json
from modules.crawler import NewsCrawler
from modules.summarizer import ArticleSummarizer
from modules.database import NewsDatabase
from modules.filter import should_process
from models.article import TechRadarAnalysis

# Configure Rich Logging
console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, console=console)]
)
logger = logging.getLogger("HermesAgent")

async def run_agent_cycle():
    """
    The core pipeline:
    1. Crawl RSS feeds for new articles.
    2. Pre-filter articles with signal scoring (Pre-LLM Funnel).
    3. Deduplicate against the database.
    4. Analyze surviving articles with the LLM.
    5. Save structured Tech Radar analysis to the database.
    """
    logger.info("[bold blue]Starting Hermes AI Agent Cycle...[/bold blue]")
    
    crawler = NewsCrawler()
    db = NewsDatabase()
    summarizer = ArticleSummarizer()
    
    try:
        # Step 1: Crawling
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            progress.add_task(description="Fetching RSS feeds...", total=None)
            articles = await crawler.crawl()
        
        logger.info(f"Discovered [bold]{len(articles)}[/bold] articles total.")
        
        new_articles_count = 0
        filtered_count = 0

        # Step 2-4: Process each article
        for article in articles:
            # --- STAGE 1: Pre-LLM Signal Filter ---
            # Use title + first 500 chars of raw_text as the scoring corpus
            description_snippet = article.raw_text[:500] if article.raw_text else ""
            if not should_process(article.title, description_snippet):
                filtered_count += 1
                continue

            # --- STAGE 2: Deduplication ---
            if await db.article_exists(article.source_url):
                logger.debug(f"Skipping existing article: {article.title}")
                continue

            logger.info(f"[yellow]Processing new article:[/yellow] {article.title}")

            try:
                # Tech Radar analysis — returns a validated structured dict
                analysis_dict = summarizer.process_article(article.raw_text)

                # Validate the required fields are present before persisting
                if not analysis_dict.get('tl_dr') or not analysis_dict.get('category'):
                    logger.warning(f"[yellow]Analysis incomplete or missing fields for:[/yellow] {article.title}")
                    continue

                # Hydrate the structured Pydantic model from the LLM dict
                article.analysis = TechRadarAnalysis(**analysis_dict)

                # Final storage - ONLY if analysis was successfully structured
                await db.save_article(article)
                new_articles_count += 1
                logger.info(
                    f"[green]Saved:[/green] [{article.analysis.category}] "
                    f"(score: {article.analysis.relevance_score}) {article.title}"
                )

            except json.JSONDecodeError:
                logger.error(f"[red]LLM returned invalid JSON for:[/red] '{article.title}' — skipping.")
                continue
            except Exception as e:
                logger.error(f"[red]Failed to process article '{article.title}': {e}[/red]")
                continue

            # --- STAGE 3: Rate Limit Guard ---
            # Sleep 12s between API calls to respect the Free Tier limit (~5 RPM).
            # Only executed after a successful LLM call, not on skipped/failed articles.
            logger.debug("Rate limit guard: sleeping 12s before next API call...")
            await asyncio.sleep(12)

        logger.info(
            f"[bold green]Cycle Complete.[/bold green] "
            f"Saved={new_articles_count} | Filtered={filtered_count}"
        )
        
    except Exception as e:
        logger.error(f"Cycle failed with error: {e}")
    finally:
        await db.close()

async def main():
    """
    Entry point for the Hermes AI News Agent.
    Sets up scheduling and initiates the initial test run.
    """
    # 1. Run immediately once to verify everything is working
    await run_agent_cycle()
    
    # 2. Setup APScheduler
    scheduler = AsyncIOScheduler()
    
    # Schedule: 08:00, 14:00, 20:00 Daily
    scheduler.add_job(run_agent_cycle, 'cron', hour=8, minute=0)
    scheduler.add_job(run_agent_cycle, 'cron', hour=14, minute=0)
    scheduler.add_job(run_agent_cycle, 'cron', hour=20, minute=0)
    
    scheduler.start()
    logger.info("Scheduler started. Jobs set for [bold]08:00, 14:00, 20:00[/bold] daily.")
    
    # 3. Continuous Execution
    # Using an Event to keep the script alive headless on the Raspberry Pi
    stop_event = asyncio.Event()
    try:
        await stop_event.wait()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Shutting down agent...")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
