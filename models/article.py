from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field

class Article(BaseModel):
    """
    Data Contract for an Article across all modules in the Hermes AI News Agent.
    """
    source_url: str = Field(..., description="The URL of the original article")
    title: str = Field(..., description="The title of the article")
    raw_text: str = Field(..., description="The full, unparsed text of the article")
    published_at: datetime = Field(..., description="The publication timestamp")
    summary: Optional[str] = Field(default=None, description="An AI-generated summary of the content")
    tags: List[str] = Field(default_factory=list, description="Categorization tags")
