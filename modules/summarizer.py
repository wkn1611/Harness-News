import os
from google import genai
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

class ArticleSummarizer:
    """
    Leverages the google-genai SDK to generate concise Vietnamese summaries.
    Uses a Dynamic Model Selector to avoid 404 errors from deprecated/unavailable models.
    Optimized for stability on Raspberry Pi ARM architectures via synchronous calls.
    """
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            logger.error("GEMINI_API_KEY not found in environment!")
            self.client = None
            self.model_name = None
            return

        # Initialize the client with no hardcoded http_options
        # so the SDK resolves the optimal default API version automatically
        self.client = genai.Client(api_key=api_key)
        self.model_name = self._select_model()

    def _select_model(self) -> str:
        """
        Dynamically queries available Gemini models and selects the best one.
        Priority: first 'flash' model > first valid model with generateContent support.
        """
        try:
            available_models = list(self.client.models.list())

            # Filter only models that support content generation
            generative_models = [
                m for m in available_models
                if hasattr(m, 'supported_generation_methods')
                and 'generateContent' in m.supported_generation_methods
            ]

            if not generative_models:
                raise RuntimeError("No generative models are available from the API.")

            # Priority 1: first model with 'flash' in name (speed + cost optimized)
            flash_model = next(
                (m for m in generative_models if 'flash' in m.name.lower()),
                None
            )

            # Priority 2: fallback to first valid model
            selected = flash_model if flash_model else generative_models[0]

            logger.info(f"Dynamically selected model: {selected.name}")
            return selected.name

        except Exception as e:
            logger.error(f"Failed to fetch model list, using hardcoded fallback: {e}")
            # Last-resort hardcoded fallback
            return "models/gemini-1.5-flash"

    def process_article(self, raw_text: str) -> str:
        """
        Sends the article text to Gemini and returns a 3-bullet point summary in Vietnamese.
        Execution is SYNCHRONOUS for maximum stability on low-resource environments.
        """
        if not self.client or not self.model_name:
            raise ValueError("Summarizer client not initialized (missing API key or no available models).")

        # Preserving the exact "Senior Tech Journalist" system prompt
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
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )

            if not response or not response.text:
                raise ValueError("Gemini returned an empty or invalid response.")

            return response.text.strip()

        except Exception as e:
            logger.error(f"Failed to generate summary using model '{self.model_name}': {e}")
            raise e
