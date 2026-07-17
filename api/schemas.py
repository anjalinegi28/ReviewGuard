from typing import List, Optional
from pydantic import BaseModel, Field


class ReviewIn(BaseModel):
    rating: int = Field(..., ge=1, le=5, description="star rating, 1-5")
    text: str = Field(..., min_length=1, max_length=5000)
    review_id: Optional[str] = None


class BatchReviewsIn(BaseModel):
    reviews: List[ReviewIn]


class TokenContributionOut(BaseModel):
    token: str
    weight: float


class AnalyzeOut(BaseModel):
    review_id: Optional[str]
    rating: int
    predicted_sentiment: str
    sentiment_confidence: float
    mismatch_score: float
    is_flagged: bool
    reason: str
    explanation: Optional[List[TokenContributionOut]] = None
    explanation_method: Optional[str] = None


class DuplicateGroupOut(BaseModel):
    cluster_id: int
    review_indices: List[int]
    size: int
    representative_text: str


class BatchAnalyzeOut(BaseModel):
    results: List[AnalyzeOut]
    duplicate_clusters: List[DuplicateGroupOut]
