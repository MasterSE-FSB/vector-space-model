"""Router: truy vấn ngữ nghĩa / research."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from apps.controller import search_controller
from apps.router.schemas import (
    CompareRequest,
    CompareResponse,
    SearchRequest,
    SearchResponse,
    SimilarRequest,
)

router = APIRouter(prefix="/search", tags=["search"])


@router.post("", response_model=SearchResponse, summary="Truy vấn free-text")
def search(req: SearchRequest):
    try:
        results = search_controller.search(req.query, req.model, req.top_k)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return SearchResponse(model=req.model, query=req.query, results=results)


@router.post("/similar", response_model=SearchResponse, summary="Tìm anime tương tự (more-like-this)")
def search_similar(req: SimilarRequest):
    try:
        results = search_controller.search_similar(req.anime_name, req.model, req.top_k)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not results:
        raise HTTPException(status_code=404, detail=f"Không tìm thấy anime '{req.anime_name}'")
    return SearchResponse(model=req.model, query=req.anime_name, results=results)


@router.post("/compare", response_model=CompareResponse, summary="So sánh kết quả 2 model")
def compare(req: CompareRequest):
    results = search_controller.compare(req.query, req.top_k)
    return CompareResponse(query=req.query, results=results)
