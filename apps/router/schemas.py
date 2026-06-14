"""Pydantic schemas cho FastAPI."""
from __future__ import annotations

from pydantic import BaseModel, Field

from apps.config import MODEL_REGISTRY

_DEFAULT_MODEL = "e5-small-v2"
_MODEL_KEYS = list(MODEL_REGISTRY.keys())


class ImportRequest(BaseModel):
    models: list[str] | None = Field(
        default=None, description=f"Danh sách model cần import. Mặc định tất cả: {_MODEL_KEYS}"
    )
    force: bool = Field(
        default=False,
        description="False = bỏ qua nếu collection đã có dữ liệu (idempotent); True = xoá & nạp lại",
    )
    min_members: int = Field(default=0, ge=0, description="Lọc anime có ít người xem")


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Câu truy vấn tự do")
    model: str = Field(default=_DEFAULT_MODEL, description=f"Một trong {_MODEL_KEYS}")
    top_k: int = Field(default=10, ge=1, le=100)


class SimilarRequest(BaseModel):
    anime_name: str = Field(..., min_length=1)
    model: str = Field(default=_DEFAULT_MODEL)
    top_k: int = Field(default=10, ge=1, le=100)


class CompareRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=10, ge=1, le=100)


class SearchHit(BaseModel):
    anime_id: int
    name: str
    genre: str | None = None
    type: str | None = None
    rating: float | None = None
    members: int | None = None
    score: float


class SearchResponse(BaseModel):
    model: str
    query: str
    results: list[SearchHit]


class CompareResponse(BaseModel):
    query: str
    results: dict[str, list[SearchHit]]
