from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class DeepDiveAnalysis(BaseModel):
    """
    Extended analysis fields populated only for 'Deep Dive' article types.
    """
    architecture_design: str = Field(
        default="",
        description="System design explanation (empty if not a Deep Dive)."
    )
    benchmark_metrics: List[str] = Field(
        default_factory=list,
        description="Performance data points (empty if not a Deep Dive)."
    )
    core_limitations: str = Field(
        default="",
        description="Known issues or caveats (empty if not a Deep Dive)."
    )


class IntelligenceReport(BaseModel):
    """
    Structured intelligence report returned by the Groq LLM (Tier 2).
    Matches the exact JSON schema defined in the SYSTEM_PROMPT.
    """
    article_type: str = Field(
        ...,
        description="One of: News, Tool Release, Tutorial, Deep Dive."
    )
    category: str = Field(
        ...,
        description="Primary domain (e.g., AI & Agents, System Architecture)."
    )
    relevance_score: int = Field(
        ...,
        ge=1, le=10,
        description="Impact score for a senior developer (1-10)."
    )
    tl_dr: str = Field(
        ...,
        description="Punchy 2-sentence summary."
    )
    actionable_takeaway: str = Field(
        ...,
        description="Direct command or next step the developer should execute."
    )
    tech_stack: List[str] = Field(
        default_factory=list,
        description="Up to 5 specific tools/frameworks mentioned."
    )
    key_insights: List[str] = Field(
        default_factory=list,
        description="3 highly technical bullet points with metrics/versions."
    )
    deep_dive_analysis: Optional[DeepDiveAnalysis] = Field(
        default=None,
        description="Extended analysis for Deep Dive articles only."
    )


# --- Legacy model kept for backward compatibility with existing DB docs ---
class TechRadarAnalysis(BaseModel):
    """
    [Deprecated] Old Gemini-era analysis schema. Kept so existing MongoDB
    documents can still be deserialized without migration.
    """
    category: str = ""
    relevance_score: int = Field(default=5, ge=1, le=10)
    tl_dr: str = ""
    key_insights: List[str] = Field(default_factory=list)
    tech_stack: List[str] = Field(default_factory=list)
    why_it_matters: str = ""


class Article(BaseModel):
    """
    Universal Data Contract for an Article across all modules.

    - analysis:  New Tier 2 IntelligenceReport from Groq.
    - legacy_analysis: Old TechRadarAnalysis from Gemini (backward compat).
    - summary: Plain-text fallback (deprecated).
    """
    source_url: str = Field(..., description="The URL of the original article")
    title: str = Field(..., description="The title of the article")
    raw_text: str = Field(..., description="The full, unparsed text of the article")
    published_at: datetime = Field(..., description="The publication timestamp")
    analysis: Optional[IntelligenceReport] = Field(
        default=None,
        description="Structured intelligence report from Tier 2 (Groq LLM)"
    )
    legacy_analysis: Optional[TechRadarAnalysis] = Field(
        default=None,
        description="[Deprecated] Old Gemini-era TechRadarAnalysis"
    )
    summary: Optional[str] = Field(
        default=None,
        description="[Deprecated] Legacy plain-text summary field"
    )
    tags: List[str] = Field(default_factory=list, description="Categorization tags")
