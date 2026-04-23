import os
import logging
from typing import Optional
from motor.motor_asyncio import AsyncIOMotorClient
from models.article import Article
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

class NewsDatabase:
    """
    Asynchronous MongoDB client for storing and managing articles.
    """
    def __init__(self):
        mongodb_uri = os.getenv("MONGODB_URI") or os.getenv("mongodb+srv://htp_admin:kn111004@cluster0.eil3waj.mongodb.net/?appName=Cluster0")
        # In case the .env key was literally the second line of the file without a variable name
        if not mongodb_uri or "mongodb+srv" not in mongodb_uri:
            mongodb_uri = "mongodb+srv://htp_admin:kn111004@cluster0.eil3waj.mongodb.net/?appName=Cluster0"
            
        self.client = AsyncIOMotorClient(mongodb_uri)
        self.db = self.client.hermes_news
        self.collection = self.db.articles
        logger.info("MongoDB Initialized.")

    async def article_exists(self, source_url: str) -> bool:
        """Checks if an article with the given URL already exists in the database."""
        count = await self.collection.count_documents({"source_url": source_url})
        return count > 0

    async def save_article(self, article: Article):
        """Saves a validated Article object to MongoDB."""
        try:
            # Pydantic's dict() or model_dump() (v2)
            data = article.model_dump()
            await self.collection.insert_one(data)
            logger.debug(f"Saved to DB: {article.title}")
        except Exception as e:
            logger.error(f"Error saving to MongoDB: {e}")

    async def close(self):
        self.client.close()
