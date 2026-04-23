import asyncio
import logging
import sys
from datetime import datetime

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn

# Internal imports
from modules.crawler import NewsCrawler
from modules.summarizer import ArticleSummarizer
from modules.database import NewsDatabase

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
    2. Check if the article already exists in the database.
    3. If new, summarize it using LLM.
    4. Save the summarized article to the database.
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
        
        # Step 2-4: Process each article
        for article in articles:
            # Check for duplication to prevent redundant LLM costs and DB clutter
            if await db.article_exists(article.source_url):
                logger.debug(f"Skipping existing article: {article.title}")
                continue
            
            logger.info(f"[yellow]Processing new article:[/yellow] {article.title}")
            
            try:
                # AI Summarization with strict verification
                summary = await summarizer.process_article(article.raw_text)
                
                # Check for empty or very short failure-state summaries
                if not summary or len(summary) < 10:
                    logger.warning(f"[yellow]Summary generation failed or empty for:[/yellow] {article.title}")
                    continue
                
                article.summary = summary
                
                # Final storage - ONLY if summarization was successful and verified
                await db.save_article(article)
                new_articles_count += 1
                
            except Exception as e:
                logger.error(f"[red]Failed to process article '{article.title}': {e}[/red]")
                continue
            
        logger.info(f"[bold green]Cycle Complete.[/bold green] Added {new_articles_count} new articles.")
        
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
