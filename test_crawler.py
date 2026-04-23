import asyncio
import logging
from rich.console import Console
from rich.panel import Panel
from modules.crawler import NewsCrawler

# Optional: Silence standard library debug logs if preferred, or leave as INFO
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

async def test_crawler():
    console = Console()
    console.print("\n[bold green]Initializing NewsCrawler for Test...[/bold green]\n")
    
    # Instantiate the crawler (uses config/sources.json by default)
    crawler = NewsCrawler()
    
    # Limit to fetching just the first 3 entries to save testing time
    console.print("[yellow]Crawling sources (Limited to 3 articles)...[/yellow]")
    articles = await crawler.crawl(limit=3)
    
    console.print(f"\n[bold green]Successfully extracted {len(articles)} articles![/bold green]\n")
    
    for i, article in enumerate(articles, 1):
        # Create a 150-character snippet of the raw_text, replace newlines with spaces for formatting
        raw_snippet = article.raw_text[:150].replace('\n', ' ')
        if len(article.raw_text) > 150:
            raw_snippet += "..."
            
        content = (
            f"[bold cyan]URL:[/bold cyan] [link={article.source_url}]{article.source_url}[/link]\n"
            f"[bold yellow]Title:[/bold yellow] {article.title}\n"
            f"[bold magenta]Snippet:[/bold magenta] [white]{raw_snippet}[/white]"
        )
        
        # Display the formatted data inside a beautiful Rich panel
        console.print(Panel(content, title=f"[bold]Article {i}[/bold]", expand=False))

if __name__ == "__main__":
    asyncio.run(test_crawler())
