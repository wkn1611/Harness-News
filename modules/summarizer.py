import os
from google import genai
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

class ArticleSummarizer:
    """
    Leverages the NEW Google GenAI SDK to generate concise summaries for news articles.
    Optimized for stability on Raspberry Pi ARM architectures by using synchronous calls.
    """
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            logger.error("GEMINI_API_KEY not found in environment!")
            self.client = None
        else:
            # Initialize the NEW google-genai client
            self.client = genai.Client(api_key=api_key)
            self.model_id = "gemini-1.5-flash"
            logger.info(f"ArticleSummarizer initialized with model: {self.model_id}")

    def process_article(self, raw_text: str) -> str:
        """
        Sends the article text to Gemini and returns a 3-bullet point summary in Vietnamese.
        This execution is SYNCHRONOUS for maximum stability on low-resource environments.
        """
        if not self.client:
            raise ValueError("Summarizer client not initialized (missing API key).")

        # Keeping the exact high-quality "Senior Tech Journalist" prompt
        prompt = (
            "You are a Senior Tech Journalist. Your task is to summarize the following tech/AI news article "
            "into exactly 3 concise and impactful bullet points in Vietnamese. "
            "\n\nSTRICT RULES:\n"
            "1. LANGUAGE: Use professional, natural, and modern Vietnamese tech-journalism tone.\n"
            "2. TECHNICAL TERMS: Do NOT translate specialized technical terms. Keep the following words "
            "EXACTLY as they are in English: LLM, API, Framework, Prompt, Agent, RAG, Open-source, "
            "Machine Learning, Deep Learning, Cloud, Hardware, Software, Backend, Frontend.\n"
            "3. ACCURACY: The meaning must perfectly match the English context without ANY hallucinations.\n"
            "4. FORMAT: Output only the 3 bullet points, nothing else.\n\n"
            f"ARTICLE TEXT:\n{raw_text[:7000]}"
        )

        try:
            # Using the SYNCHRONOUS generation method as requested
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=prompt
            )

            if not response or not response.text:
                raise ValueError("Gemini returned an empty or invalid response.")

            return response.text.strip()

        except Exception as e:
            logger.error(f"Failed to generate summary for article: {e}")
            raise e
