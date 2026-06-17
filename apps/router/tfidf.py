"""Router: in-memory TF-IDF search baseline."""
from __future__ import annotations

from functools import lru_cache
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from apps.services.data_loader import load_anime
from apps.services.tfidf_search import TFIDFSearchService

router = APIRouter(prefix="/search/tfidf", tags=["tfidf"])


class TFIDFSearchRequest(BaseModel):
    query: str
    top_k: int = 10


class TFIDFSimilarRequest(BaseModel):
    anime_name: str
    top_k: int = 10


class TFIDFSearchResult(BaseModel):
    anime_id: Optional[int] = None
    name: str
    genre: Optional[str] = None
    type: Optional[str] = None
    members: Optional[int] = None
    rating: Optional[float] = None
    score: float


class TFIDFSearchResponse(BaseModel):
    model: str = "tfidf"
    query: str
    top_k: int
    results: list[TFIDFSearchResult]


@lru_cache(maxsize=1)
def get_tfidf_service() -> TFIDFSearchService:
    df = load_anime()
    service = TFIDFSearchService(
        max_features=20000,
        ngram_range=(1, 2),
        min_df=2,
    )
    service.fit(df, text_col="document")
    return service


@router.post("", response_model=TFIDFSearchResponse)
def search_tfidf(payload: TFIDFSearchRequest):
    try:
        service = get_tfidf_service()
        results = service.search(payload.query, top_k=payload.top_k)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return TFIDFSearchResponse(
        query=payload.query,
        top_k=payload.top_k,
        results=results,
    )


@router.post("/similar", response_model=TFIDFSearchResponse)
def search_tfidf_similar(payload: TFIDFSimilarRequest):
    try:
        service = get_tfidf_service()
        results = service.search_similar_to_anime(
            anime_name=payload.anime_name,
            top_k=payload.top_k,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return TFIDFSearchResponse(
        query=payload.anime_name,
        top_k=payload.top_k,
        results=results,
    )
