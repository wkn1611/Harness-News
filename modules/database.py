"""
Tier 3 — The Vault for the Hermes AI News Agent.

Handles connection to MongoDB Atlas, index initialization, and
saving documents containing both raw content and AI-generated intelligence.
"""
import os
import logging
from pymongo import MongoClient, ASCENDING, UpdateOne
from pymongo.errors import PyMongoError
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Global client to reuse connection across the synchronous pipeline
_client = None

def get_db():
    """Returns the 'news_radar' database instance, lazily connecting if needed."""
    global _client
    if _client is None:
        uri = os.environ.get("MONGODB_URI")
        if not uri:
            logger.error("MONGODB_URI not found in environment variables!")
            raise ValueError("MONGODB_URI is required.")
        _client = MongoClient(uri)
    return _client.news_radar

def setup_database():
    """
    Initializes the database by creating necessary indexes.
    1. A UNIQUE index on the `url` field to prevent saving duplicate articles.
    2. A TTL index on the `published_date` field set to 60 days (5,184,000 seconds).
    """
    try:
        db = get_db()
        collection = db.articles
        
        # 1. Unique index on 'url'
        collection.create_index([("url", ASCENDING)], unique=True)
        logger.debug("Unique index on 'url' ensured.")
        
        # 2. TTL index on 'published_date' (60 days = 5184000 seconds)
        # We catch potential errors if the index already exists with different options.
        try:
            collection.create_index([("published_date", ASCENDING)], expireAfterSeconds=5184000)
            logger.debug("TTL index on 'published_date' ensured.")
        except PyMongoError as e:
            logger.warning(f"Could not create/update TTL index (might already exist with different params): {e}")
            
        logger.info("Database indexes initialized successfully.")
    except PyMongoError as e:
        logger.error(f"Failed to setup database indexes: {e}")
        raise

def save_article(article_data: dict) -> bool:
    """
    Upserts an article into the MongoDB collection based on its URL.
    
    Args:
        article_data: A dict containing title, url, published_date, category_tag, 
                      raw_markdown_content, and intelligence.
                      
    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        db = get_db()
        collection = db.articles
        
        url = article_data.get("url")
        if not url:
            logger.error("Article data is missing 'url' field.")
            return False
            
        result = collection.update_one(
            {"url": url},
            {"$set": article_data},
            upsert=True
        )
        
        if result.upserted_id:
            logger.debug(f"Inserted new article: {url}")
        else:
            logger.debug(f"Updated existing article: {url}")
            
        return True
    except PyMongoError as e:
        logger.error(f"Database error while saving article '{article_data.get('url')}': {e}")
        return False
