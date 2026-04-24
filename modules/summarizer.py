import json
import os
import re
import time

from google import genai
from google.api_core import exceptions as google_exceptions
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Model Fallback Matrix — tried in order if previous models quota-out or fail
# ---------------------------------------------------------------------------
MODEL_FALLBACK_MATRIX = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-2.5-flash-lite",
]

# Exponential backoff delays (seconds) between retries on the SAME model
BACKOFF_SCHEDULE = [15, 30]  # Retry 1 → 15s, Retry 2 → 30s
MAX_RETRIES_PER_MODEL = 3    # Attempts per model before cascading to the next

# Rate limit pacing: 12s keeps us comfortably under the 5 req/min Free Tier
RATE_LIMIT_SLEEP = 12

# The exact 7 Tech Radar categories (Vietnamese)
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
    Ironclad Expert Technical Analyst powered by the google-genai SDK.

    Resilience features:
    - Model Fallback Matrix: cascades through 3 models on total failure.
    - Exponential Backoff: 15s / 30s sleeps on 429 / 503 transient errors.
    - Strict JSON Extraction: defensive regex fence-stripping before json.loads().
    - Rate Limit Pacing: 12s sleep after every successful API call.
    """

    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            logger.error("GEMINI_API_KEY not found in environment!")
            self.client = None
            return

        # No hardcoded http_options — SDK resolves optimal API version
        self.client = genai.Client(api_key=api_key)
        logger.info(
            f"ArticleSummarizer initialized. "
            f"Model matrix: {MODEL_FALLBACK_MATRIX}"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_prompt(self, raw_text: str) -> str:
        """Builds the strict Tech Radar JSON extraction prompt."""
        categories_str = "\n".join(f'  - "{c}"' for c in TECH_RADAR_CATEGORIES)

        return (
            "You are an Expert Technical Analyst building a Tech Radar for senior AI software engineers. "
            "Your output must be a STRICTLY VALID RAW JSON object. "
            "CRITICAL: Do NOT wrap output in markdown code blocks (no ```json). "
            "Output ONLY the raw JSON object and absolutely nothing else.\n\n"

            "ANALYZE the following article and return a JSON object with this EXACT schema:\n"
            "{\n"
            '  "category": "MUST be exactly ONE of the following strings:\n'
            f"{categories_str}\",\n"
            '  "relevance_score": <integer 1-10, rating impact for an AI software engineer>,\n'
            '  "tl_dr": "<1-2 sentences in Vietnamese summarizing the core event>",\n'
            '  "key_insights": [\n'
            '    "<Insight 1: the technical WHAT — specific models, tools, version numbers>",\n'
            '    "<Insight 2: the HOW or WHY — architecture, mechanism, methodology>",\n'
            '    "<Insight 3: performance data, benchmarks, comparisons, or industry drama>",\n'
            '    "<Insight 4 (optional): developer impact, migration notes, or ecosystem shift>"\n'
            "  ],\n"
            '  "tech_stack": ["<model/framework/library/tool name>"],\n'
            '  "why_it_matters": "<1 sentence in Vietnamese on practical value for developers>"\n'
            "}\n\n"

            "STRICT RULES:\n"
            "1. ALL content values (tl_dr, key_insights, why_it_matters) MUST be in Vietnamese.\n"
            "2. NEVER translate technical English terms. Keep EXACTLY as-is: LLM, API, Framework, "
            "Prompt, Agent, RAG, Open-source, Machine Learning, Deep Learning, Cloud, Hardware, "
            "Software, Backend, Frontend, GPU, CUDA, fine-tuning, inference, token, embedding.\n"
            "3. Be SPECIFIC and FACTUAL. AVOID generic filler phrases. "
            "Always prefer version numbers, benchmark scores, and proper nouns.\n"
            "4. relevance_score rubric: 10=paradigm shift, 7-9=major update, 4-6=noteworthy, 1-3=minor.\n\n"

            f"ARTICLE TEXT:\n{raw_text[:8000]}"
        )

    @staticmethod
    def _clean_and_parse(raw_output: str) -> dict:
        """
        Strips markdown code fences defensively and parses the JSON string.
        Raises json.JSONDecodeError if the output is not valid JSON.
        """
        cleaned = re.sub(r'^```(?:json)?\s*', '', raw_output.strip())
        cleaned = re.sub(r'\s*```$', '', cleaned).strip()
        return json.loads(cleaned)

    @staticmethod
    def _is_transient_error(exc: Exception) -> bool:
        """Returns True for 429 Quota and 503 Overload errors that merit a retry."""
        error_str = str(exc).lower()
        return (
            "429" in error_str
            or "quota" in error_str
            or "resource_exhausted" in error_str
            or "503" in error_str
            or "overloaded" in error_str
            or "service_unavailable" in error_str
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def process_article(self, raw_text: str) -> dict:
        """
        Analyzes a tech article and returns a validated Tech Radar JSON dict.

        Implements a full reliability loop:
          1. Try MODEL_FALLBACK_MATRIX[0], up to MAX_RETRIES_PER_MODEL times.
          2. On 429 / 503 → Exponential Backoff (15s, 30s).
          3. On total model failure → cascade to next model in matrix.
          4. On success → sleep RATE_LIMIT_SLEEP seconds before returning.

        Raises:
            RuntimeError: if all models in the fallback matrix are exhausted.
            json.JSONDecodeError: if the final successful response is not valid JSON.
        """
        if not self.client:
            raise ValueError("Summarizer client not initialized (missing API key).")

        prompt = self._build_prompt(raw_text)

        for model_name in MODEL_FALLBACK_MATRIX:
            attempt = 0

            while attempt < MAX_RETRIES_PER_MODEL:
                attempt += 1
                logger.debug(
                    f"Trying model='{model_name}' attempt={attempt}/{MAX_RETRIES_PER_MODEL}"
                )

                try:
                    response = self.client.models.generate_content(
                        model=model_name,
                        contents=prompt
                    )

                    if not response or not response.text:
                        raise ValueError("Gemini returned an empty response.")

                    # --- Strict JSON Extraction ---
                    parsed = self._clean_and_parse(response.text)

                    logger.success(
                        f"✓ Analysis complete | model='{model_name}' "
                        f"category='{parsed.get('category')}' "
                        f"score={parsed.get('relevance_score')}"
                    )

                    # --- Rate Limit Pacing ---
                    # Sleep AFTER a successful call to respect the 5 RPM Free Tier
                    logger.debug(f"Rate pacing: sleeping {RATE_LIMIT_SLEEP}s...")
                    time.sleep(RATE_LIMIT_SLEEP)

                    return parsed

                except json.JSONDecodeError as e:
                    # JSON parse failure is NOT a transient API error — don't retry
                    logger.error(
                        f"JSONDecodeError on model='{model_name}': {e}\n"
                        f"Raw output preview: {response.text[:400] if response else 'N/A'}"
                    )
                    raise e

                except Exception as e:
                    if self._is_transient_error(e):
                        # Apply exponential backoff from the schedule
                        backoff_secs = BACKOFF_SCHEDULE[min(attempt - 1, len(BACKOFF_SCHEDULE) - 1)]
                        logger.warning(
                            f"Transient error on model='{model_name}' attempt={attempt}: {e}\n"
                            f"Backing off for {backoff_secs}s..."
                        )
                        time.sleep(backoff_secs)
                        # Continue the while loop to retry the SAME model
                        continue
                    else:
                        # Non-transient error (e.g., 404, auth failure) — don't retry this model
                        logger.error(
                            f"Non-transient error on model='{model_name}': {e} — skipping to next model."
                        )
                        break  # Break while loop, cascade to next model

            # All attempts exhausted for this model
            logger.warning(
                f"Model '{model_name}' exhausted after {MAX_RETRIES_PER_MODEL} attempts. "
                f"Cascading to next model in fallback matrix..."
            )

        # All models in the fallback matrix have been exhausted
        raise RuntimeError(
            f"All models in fallback matrix exhausted: {MODEL_FALLBACK_MATRIX}. "
            "Cannot process article. Check API quota or connectivity."
        )
