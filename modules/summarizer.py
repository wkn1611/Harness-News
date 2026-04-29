"""
Tier 2 — Intelligence Engine for the Hermes AI News Agent.

Uses the Groq API (OpenAI-compatible proxy) to analyze raw article markdown
and extract structured JSON intelligence reports using LLaMA models.

Fallback logic:
    1. Primary:  llama-3.3-70b-versatile  (deepest reasoning)
    2. Fallback: llama-3.1-8b-instant     (fast, low-latency)
"""
import json
import os
from typing import Optional

from openai import OpenAI, RateLimitError
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Model Fallback Matrix
# ---------------------------------------------------------------------------
PRIMARY_MODEL = "llama-3.3-70b-versatile"
FALLBACK_MODEL = "llama-3.1-8b-instant"

# Maximum characters of markdown to send to the API.
# Prevents context window overflow on Groq's hosted LLaMA models.
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
# Groq Client Initialization
# ---------------------------------------------------------------------------
_groq_api_key = os.environ.get("GROQ_API_KEY")

if not _groq_api_key:
    logger.warning(
        "GROQ_API_KEY not found in environment. "
        "Tier 2 Intelligence Engine will be unavailable."
    )
    client: Optional[OpenAI] = None
else:
    client = OpenAI(
        api_key=_groq_api_key,
        base_url="https://api.groq.com/openai/v1",
    )
    logger.info(
        f"Groq client initialized. "
        f"Primary={PRIMARY_MODEL} | Fallback={FALLBACK_MODEL}"
    )


# ---------------------------------------------------------------------------
# Core Analysis Function
# ---------------------------------------------------------------------------
def analyze_article(markdown_text: str) -> Optional[dict]:
    """
    Sends raw article markdown to the Groq API for structured intelligence
    extraction. Returns a parsed dict matching the SYSTEM_PROMPT schema,
    or None if all attempts fail.

    Fallback logic:
        1. Try PRIMARY_MODEL (llama-3.3-70b-versatile).
        2. If RateLimitError (429) → retry with FALLBACK_MODEL.
        3. If JSON parsing fails or any other error → return None.

    Args:
        markdown_text: Raw markdown content from the Jina Reader ingestor.

    Returns:
        A dict matching the intelligence report schema, or None on failure.
    """
    if client is None:
        logger.error("Groq client not initialized — cannot analyze article.")
        return None

    # Truncate to stay within the context window
    truncated_text = markdown_text[:MAX_CONTEXT_CHARS]

    # --- Attempt 1: Primary Model ---
    try:
        result = _call_groq(truncated_text, model=PRIMARY_MODEL)
        return result

    except RateLimitError as e:
        logger.warning(
            f"RateLimitError on {PRIMARY_MODEL}: {e} — "
            f"falling back to {FALLBACK_MODEL}..."
        )

    except Exception as e:
        logger.error(
            f"Unexpected error on {PRIMARY_MODEL}: {e} — "
            f"falling back to {FALLBACK_MODEL}..."
        )

    # --- Attempt 2: Fallback Model ---
    try:
        result = _call_groq(truncated_text, model=FALLBACK_MODEL)
        return result

    except Exception as e:
        logger.error(
            f"Fallback model {FALLBACK_MODEL} also failed: {e} — "
            "returning None."
        )
        return None


# ---------------------------------------------------------------------------
# Internal API Call Helper
# ---------------------------------------------------------------------------
def _call_groq(text: str, model: str) -> Optional[dict]:
    """
    Makes a single chat completion call to the Groq API and parses the
    JSON response.

    Args:
        text:  Truncated article markdown.
        model: The Groq-hosted model identifier to use.

    Returns:
        Parsed dict from the model's JSON output.

    Raises:
        RateLimitError: Propagated so the caller can trigger fallback.
        json.JSONDecodeError: If the model returns unparseable output.
        Exception: Any other API or network error.
    """
    logger.debug(f"Calling Groq | model={model} | input_chars={len(text)}")

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": text},
        ],
        temperature=0.2,            # Low temp for factual, deterministic output
        response_format={"type": "json_object"},
    )

    raw_output = response.choices[0].message.content
    if not raw_output:
        logger.error(f"Empty response from {model}.")
        return None

    # Parse the JSON — the response_format flag should guarantee valid JSON,
    # but we still guard defensively.
    try:
        parsed = json.loads(raw_output)
    except json.JSONDecodeError as e:
        logger.error(
            f"JSONDecodeError from {model}: {e}\n"
            f"Raw output preview: {raw_output[:500]}"
        )
        return None

    logger.success(
        f"✓ Analysis complete | model={model} | "
        f"type='{parsed.get('article_type')}' | "
        f"category='{parsed.get('category')}' | "
        f"score={parsed.get('relevance_score')}"
    )
    return parsed
