"""Controller: nghiệp vụ truy vấn / research."""
from __future__ import annotations

from functools import lru_cache

import pandas as pd

from apps.config import MODEL_REGISTRY
from apps.services import search_service
from apps.services.data_loader import load_anime


@lru_cache(maxsize=1)
def _get_df() -> pd.DataFrame:
    """Cache dataframe anime (dùng cho more-like-this)."""
    return load_anime()


def _validate(model: str) -> None:
    if model not in MODEL_REGISTRY:
        raise ValueError(f"Model '{model}' không hợp lệ. Hợp lệ: {list(MODEL_REGISTRY)}")


def search(query: str, model: str, top_k: int) -> list[dict]:
    _validate(model)
    return search_service.search_text(model, query, top_k=top_k)


def search_similar(anime_name: str, model: str, top_k: int) -> list[dict]:
    _validate(model)
    return search_service.search_similar_to_anime(model, anime_name, _get_df(), top_k=top_k)


def compare(query: str, top_k: int) -> dict[str, list[dict]]:
    return {
        key: search_service.search_text(key, query, top_k=top_k)
        for key in MODEL_REGISTRY
    }
