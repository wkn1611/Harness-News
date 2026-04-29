"""
Tier 2 — Intelligence Engine for the Hermes AI News Agent.

Uses a Primary-Fallback multi-provider LLM pipeline (Cerebras + OpenRouter)
to analyze raw article markdown and extract structured JSON intelligence reports.

Fallback logic:
    1. Primary:  Cerebras (llama3.1-8b)
    2. Fallback: OpenRouter (meta-llama/llama-3.3-70b-instruct:free)
"""
import json
import os
import re
import time
from typing import Optional

from openai import OpenAI
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Provider & Model Configuration
# ---------------------------------------------------------------------------
PRIMARY_MODEL = "llama3.1-8b"
FALLBACK_MODEL = "meta-llama/llama-3.3-70b-instruct:free"

MAX_CONTEXT_CHARS = 25_000

# ---------------------------------------------------------------------------
# System Prompt — defines the LLM's role and output schema
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """
You are an elite Technical Intelligence Analyst and System Architect. 
Your primary user is a Senior Developer building autonomous AI systems, Next.js web apps, and Edge computing solutions (e.g., Raspberry Pi).

YOUR TASK:
Analyze the provided technology article, news, or research paper. Extract the most high-value, actionable intelligence. 
Ignore all marketing fluff. Focus entirely on system architectures, new frameworks, performance metrics, and practical implementations.

CONSTRAINTS & RULES:
1. NO HALLUCINATION: Only extract technologies, metrics, or insights explicitly mentioned.
2. BE SPECIFIC: Quote exact numbers (e.g., "reduced latency by 40ms").
3. ACTION-ORIENTED: The 'actionable_takeaway' must be a direct command the developer can execute.

OUTPUT FORMAT:
You MUST output ONLY a valid JSON object. Do not include markdown blocks like ```json. 
Strictly adhere to this exact schema:
{
  "article_type": "Choose EXACTLY ONE: [News, Tool Release, Tutorial, Deep Dive]",
  "category": "Identify the primary domain (e.g., AI & Agents, System Architecture, Open Source Tools, Security)",
  "relevance_score": "Integer 1-10. Rate importance.",
  "tl_dr": "A punchy, 2-sentence summary.",
  "actionable_takeaway": "What should the developer DO next? Provide commands or steps.",
  "tech_stack": ["Array of max 5 specific tools mentioned."],
  "key_insights": ["Array of 3 highly technical bullet points. Include metrics/versions."],
  "deep_dive_analysis": {
    "architecture_design": "Explain system design (ONLY if Deep Dive, else empty)",
    "benchmark_metrics": ["Array of performance data (ONLY if Deep Dive)"],
    "core_limitations": "Known issues (ONLY if Deep Dive)"
  }
}
""".strip()

# ---------------------------------------------------------------------------
# OpenAI Clients Initialization
# ---------------------------------------------------------------------------
_cerebras_api_key = os.environ.get("CEREBRAS_API_KEY")
_openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")

if not _cerebras_api_key:
    logger.warning("CEREBRAS_API_KEY not found. Primary model will fail.")
if not _openrouter_api_key:
    logger.warning("OPENROUTER_API_KEY not found. Fallback model will fail.")

client_primary = OpenAI(
    api_key=_cerebras_api_key or "dummy",
    base_url="https://api.cerebras.ai/v1"
)

client_fallback = OpenAI(
    api_key=_openrouter_api_key or "dummy",
    base_url="https://openrouter.ai/api/v1"
)

logger.info(
    f"Tier 2 clients initialized.\n"
    f"Primary (Cerebras): {PRIMARY_MODEL}\n"
    f"Fallback (OpenRouter): {FALLBACK_MODEL}"
)

# ---------------------------------------------------------------------------
# Core Analysis Function
# ---------------------------------------------------------------------------
def analyze_article(markdown_text: str) -> Optional[dict]:
    """
    Sends raw article markdown to the LLM pipeline for structured intelligence
    extraction. Returns a parsed dict matching the SYSTEM_PROMPT schema,
    or None if all attempts fail.

    Fallback logic:
        1. Try client_primary (Cerebras)
        2. On any Exception -> retry with client_fallback (OpenRouter)
        3. If both fail or JSON parsing fails -> return None

    Args:
        markdown_text: Raw markdown content from the Tier 1 ingestor.

    Returns:
        A dict matching the intelligence report schema, or None on failure.
    """
    truncated_text = markdown_text[:MAX_CONTEXT_CHARS]

    # --- Attempt 1: Primary Client (Cerebras) ---
    if _cerebras_api_key:
        try:
            return _call_llm(client_primary, PRIMARY_MODEL, truncated_text)
        except Exception as e:
            logger.warning(f"Primary model ({PRIMARY_MODEL}) failed: {e}. Sleeping 10s to respect OpenRouter rate limits before fallback...")
            time.sleep(10)
    else:
        logger.warning("Skipping Primary model (Cerebras) because API key is missing. Trying Fallback...")

    # --- Attempt 2: Fallback Client (OpenRouter) ---
    if _openrouter_api_key:
        try:
            return _call_llm(client_fallback, FALLBACK_MODEL, truncated_text)
        except Exception as e:
            logger.error(f"Fallback model ({FALLBACK_MODEL}) also failed: {e}. Returning None.")
            return None
    else:
        logger.error("Skipping Fallback model (OpenRouter) because API key is missing. Returning None.")
        return None


# ---------------------------------------------------------------------------
# Internal API Call Helper
# ---------------------------------------------------------------------------
def _call_llm(client: OpenAI, model: str, text: str) -> dict:
    """
    Makes a chat completion call to the given OpenAI client and robustly parses
    the JSON response.

    Args:
        client: The OpenAI client instance to use.
        model:  The model identifier.
        text:   Truncated article markdown.

    Returns:
        Parsed dict from the model's JSON output.

    Raises:
        Exception: If the API call fails or the output is completely unparseable.
    """
    logger.debug(f"Calling LLM | model={model} | input_chars={len(text)}")

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": text},
        ],
        temperature=0.2,
        # OpenRouter and Cerebras support response_format for specific models.
        # Passing it ensures strict JSON if the API accepts it.
        response_format={"type": "json_object"},
    )

    raw_output = response.choices[0].message.content
    if not raw_output:
        raise ValueError(f"Empty response from {model}.")

    # Robust JSON extraction: strip markdown tags if the model hallucinates them
    cleaned_output = re.sub(r'^```(?:json)?\s*', '', raw_output.strip())
    cleaned_output = re.sub(r'\s*```$', '', cleaned_output).strip()

    try:
        parsed = json.loads(cleaned_output)
    except json.JSONDecodeError as e:
        logger.error(
            f"JSONDecodeError from {model}: {e}\n"
            f"Raw output preview: {raw_output[:500]}"
        )
        raise ValueError(f"Failed to parse JSON from {model}: {e}")

    logger.success(
        f"✓ Analysis complete | model={model} | "
        f"type='{parsed.get('article_type')}' | "
        f"category='{parsed.get('category')}' | "
        f"score={parsed.get('relevance_score')}"
    )
    return parsed
