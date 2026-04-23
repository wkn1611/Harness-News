import os
import logging
import google.generativeai as genai
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

class ArticleSummarizer:
    """
    Leverages Google's Gemini LLM to generate concise summaries for news articles.
    """
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            logger.warning("GEMINI_API_KEY not found in environment. Summarization will be mocked.")
            self.model = None
        else:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-1.5-flash')
            logger.info("Summarizer Initialized with Gemini.")

    async def process_article(self, raw_text: str) -> str:
        """
        Sends the article text to Gemini and returns a 3-sentence summary.
        """
        if not self.model:
            return "Summary unavailable (no API key)."

        prompt = f"Summarize the following tech/AI news article in exactly 3 concise bullet points:\n\n{raw_text[:5000]}"
        
        try:
            response = await generate_content_async(self.model, prompt)
            if not response or not response.text:
                raise ValueError("LLM returned an empty response")
            return response.text
        except Exception as e:
            logger.error(f"LLM Summarization error: {e}")
            raise e

async def generate_content_async(model, prompt):
    """Utility to run the generative AI call asynchronously."""
    # The SDK actually supports async directly in newer versions
    return await model.generate_content_async(prompt)
