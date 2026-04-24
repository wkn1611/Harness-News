import json
import os
import re

from google import genai
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

# The exact categories for the Tech Radar classification
TECH_RADAR_CATEGORIES = [
    "Vibe Coding & Tools",
    "Agent Infrastructure",
    "Community Alert",
    "Trending Repo",
    "Local AI & Homelab",
    "Industry Titans",
    "Research & Benchmarks",
]

class ArticleSummarizer:
    """
    Expert Technical Analyst powered by the google-genai SDK.
    Analyzes articles and returns structured Tech Radar JSON intelligence
    using a Dynamic Model Selector for resilience against model deprecation.
    """
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            logger.error("GEMINI_API_KEY not found in environment!")
            self.client = None
            self.model_name = None
            return

        # No hardcoded http_options — SDK resolves the optimal API version automatically
        self.client = genai.Client(api_key=api_key)
        self.model_name = self._select_model()

    def _select_model(self) -> str:
        """
        Dynamically queries available Gemini models, preferring gemini-2.5-flash,
        then any 'flash' model, then the first valid generative model.
        """
        try:
            available_models = list(self.client.models.list())

            # Filter models with content generation capability
            generative_models = [
                m for m in available_models
                if hasattr(m, 'supported_generation_methods')
                and 'generateContent' in m.supported_generation_methods
            ]

            if not generative_models:
                raise RuntimeError("No generative models available from the API.")

            # Priority 1: gemini-2.5-flash specifically
            flagship = next(
                (m for m in generative_models if 'gemini-2.5-flash' in m.name.lower()),
                None
            )

            # Priority 2: any flash model (speed + cost optimized)
            flash = next(
                (m for m in generative_models if 'flash' in m.name.lower()),
                None
            )

            # Priority 3: first available generative model
            selected = flagship or flash or generative_models[0]

            logger.info(f"Dynamically selected model: {selected.name}")
            return selected.name

        except Exception as e:
            logger.error(f"Failed to fetch model list, using hardcoded fallback: {e}")
            return "models/gemini-2.5-flash"

    def process_article(self, raw_text: str) -> dict:
        """
        Analyzes a tech article and returns a validated Tech Radar JSON dict.
        The model is strictly instructed to return raw JSON — no markdown fences.
        Execution is SYNCHRONOUS for maximum stability on Raspberry Pi ARM.
        """
        if not self.client or not self.model_name:
            raise ValueError("Summarizer not initialized (missing API key or no available models).")

        categories_str = "\n".join(f'  - "{c}"' for c in TECH_RADAR_CATEGORIES)

        prompt = (
            "You are an Expert Technical Analyst building a Tech Radar for senior AI software engineers. "
            "Your output must be a STRICTLY VALID RAW JSON object. "
            "CRITICAL: Do NOT wrap output in markdown code blocks (no ```json). "
            "Output ONLY the JSON object and nothing else.\n\n"

            "ANALYZE the following article and return a JSON object with this EXACT schema:\n"
            "{\n"
            "  \"category\": \"MUST be exactly ONE of the following strings (in Vietnamese):\n"
            f"{categories_str}\",\n"
            "  \"relevance_score\": <integer 1-10, rating impact for an AI software engineer>,\n"
            "  \"tl_dr\": \"<1-2 sentences in Vietnamese summarizing the core event>\",\n"
            "  \"key_insights\": [\n"
            "    \"<Insight 1: the technical WHAT — specific models, tools, version numbers>\",\n"
            "    \"<Insight 2: the HOW or WHY — architecture, mechanism, methodology>\",\n"
            "    \"<Insight 3: performance data, benchmarks, comparisons, or industry drama>\",\n"
            "    \"<Insight 4 (optional): developer impact, migration notes, or ecosystem shift>\"\n"
            "  ],\n"
            "  \"tech_stack\": [\"<model/framework/library/tool name>\"],\n"
            "  \"why_it_matters\": \"<1 sentence in Vietnamese on practical value for developers>\"\n"
            "}\n\n"

            "STRICT RULES:\n"
            "1. ALL content values (tl_dr, key_insights, why_it_matters) MUST be in Vietnamese.\n"
            "2. NEVER translate technical English terms. Keep EXACTLY as-is: LLM, API, Framework, "
            "Prompt, Agent, RAG, Open-source, Machine Learning, Deep Learning, Cloud, Hardware, "
            "Software, Backend, Frontend, GPU, CUDA, fine-tuning, inference, token, embedding.\n"
            "3. Be SPECIFIC and FACTUAL. AVOID generic filler like 'this is important' or "
            "'this is a breakthrough'. Use version numbers, benchmark scores, and proper nouns.\n"
            "4. relevance_score: 10 = paradigm shift, 7-9 = major update, 4-6 = noteworthy, 1-3 = minor.\n\n"

            f"ARTICLE TEXT:\n{raw_text[:8000]}"
        )

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )

            if not response or not response.text:
                raise ValueError("Gemini returned an empty response.")

            raw_output = response.text.strip()

            # Defensive strip: remove markdown code fences if model ignores instructions
            cleaned = re.sub(r'^```(?:json)?\s*', '', raw_output)
            cleaned = re.sub(r'\s*```$', '', cleaned).strip()

            parsed = json.loads(cleaned)
            logger.success(
                f"Analysis complete | category='{parsed.get('category')}' "
                f"score={parsed.get('relevance_score')}"
            )
            return parsed

        except json.JSONDecodeError as e:
            logger.error(
                f"JSONDecodeError from model '{self.model_name}': {e}\n"
                f"Raw output was: {raw_output[:500]}"
            )
            raise e
        except Exception as e:
            logger.error(f"Failed to process article with model '{self.model_name}': {e}")
            raise e
