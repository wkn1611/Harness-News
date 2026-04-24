"""
Pre-LLM Filtering Funnel for the Hermes AI News Agent.

Scores articles using a keyword-based signal heuristic before they are
sent to the Gemini API. This protects against rate limits (Free Tier: ~5 RPM)
and ensures only high-signal tech content reaches the costly LLM step.
"""
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

# --- Signal Keyword Definitions ---

POSITIVE_KEYWORDS: dict[str, int] = {
    "agent":        5,
    "cursor":       5,
    "copilot":      5,
    "llama":        5,
    "gguf":         5,
    "api":          5,
    "outage":       5,
    "prompt":       5,
    "raspberry":    5,
    "open-source":  5,
    "opensource":   5,  # catch common alternate spelling
    "vulnerability": 5,
    "exploit":      5,
}

NEGATIVE_KEYWORDS: dict[str, int] = {
    "crypto":       -10,
    "bitcoin":      -10,
    "politics":     -10,
    "sports":       -10,
    "celebrity":    -10,
    "fashion":      -10,
    "investment":   -10,
}

# Minimum score an article must reach to pass the filter
SCORE_THRESHOLD = 5


def calculate_article_score(title: str, description: str) -> Tuple[int, list[str]]:
    """
    Computes a signal score for an article based on keyword presence.
    Both title and description are checked (case-insensitive).

    Args:
        title:       The article title from the RSS feed entry.
        description: The article summary/description from RSS, or raw_text snippet.

    Returns:
        A tuple of (score: int, matched_keywords: List[str]) for transparent logging.
    """
    # Combine and lowercase for a single case-insensitive pass
    corpus = f"{title} {description}".lower()

    total_score = 0
    matched = []

    for keyword, points in POSITIVE_KEYWORDS.items():
        if keyword in corpus:
            total_score += points
            matched.append(f"+{points}[{keyword}]")

    for keyword, penalty in NEGATIVE_KEYWORDS.items():
        if keyword in corpus:
            total_score += penalty  # penalty is already negative
            matched.append(f"{penalty}[{keyword}]")

    return total_score, matched


def should_process(title: str, description: str) -> bool:
    """
    Public gate function. Returns True if the article passes the signal threshold.
    Logs the result for full pipeline transparency.

    Args:
        title:       Article title.
        description: Article body snippet or RSS description.

    Returns:
        bool: True if article should be sent to the LLM, False otherwise.
    """
    score, matched = calculate_article_score(title, description)
    keywords_log = ", ".join(matched) if matched else "none"

    if score >= SCORE_THRESHOLD:
        logger.debug(
            f"[Filter:PASS] score={score} ({keywords_log}) | {title[:80]}"
        )
        return True
    else:
        logger.info(
            f"[Filter:SKIP] score={score} ({keywords_log}) | {title[:80]}"
        )
        return False
