"""Điều phối nghiệp vụ cấp cao: import dữ liệu vào Milvus & truy vấn ngữ nghĩa.

Dùng chung cho cả notebook (Mục 3) và FastAPI.
"""
from __future__ import annotations

import time

import pandas as pd

from apps.config import MODEL_REGISTRY
from apps.services import milvus_store
from apps.services.data_loader import load_anime
from apps.services.embedding import get_model


def import_anime_to_milvus(
    model_key: str,
    df: pd.DataFrame | None = None,
    recreate: bool = True,
    min_members: int = 0,
) -> dict:
    """Encode toàn bộ anime bằng `model_key` và nạp vào Milvus. Chạy 1 lần.

    Returns thống kê: số bản ghi, thời gian encode/insert.
    """
    if df is None:
        df = load_anime(min_members=min_members)

    model = get_model(model_key)
    milvus_store.create_collection(model_key, recreate=recreate)

    t0 = time.perf_counter()
    vectors = model.encode_passages(df["document"].tolist())
    t_encode = time.perf_counter() - t0

    rows = df.to_dict(orient="records")
    t1 = time.perf_counter()
    inserted = milvus_store.insert_embeddings(model_key, rows, vectors)
    t_insert = time.perf_counter() - t1

    return {
        "model": model_key,
        "collection": MODEL_REGISTRY[model_key].collection,
        "inserted": inserted,
        "encode_seconds": round(t_encode, 2),
        "insert_seconds": round(t_insert, 2),
        "dim": model.dim,
    }


def ensure_imported(
    model_key: str,
    df: pd.DataFrame | None = None,
    force: bool = False,
    min_members: int = 0,
) -> dict:
    """Import idempotent: nếu collection đã có dữ liệu thì BỎ QUA (tránh trùng & tốn tài nguyên).

    - `force=True`: xoá & nạp lại từ đầu.
    - Mặc định: chỉ import khi collection rỗng.
    """
    existing = milvus_store.count(model_key)
    if existing > 0 and not force:
        return {
            "model": model_key,
            "collection": MODEL_REGISTRY[model_key].collection,
            "status": "skipped",
            "existing": existing,
        }
    stats = import_anime_to_milvus(model_key, df=df, recreate=True, min_members=min_members)
    stats["status"] = "imported"
    return stats


def search_text(model_key: str, query: str, top_k: int = 10) -> list[dict]:
    """Truy vấn ngữ nghĩa bằng câu hỏi tự do (free-text)."""
    model = get_model(model_key)
    qvec = model.encode_queries([query])
    results = milvus_store.search(model_key, qvec, top_k=top_k)
    return results[0] if results else []


def search_similar_to_anime(
    model_key: str,
    anime_name: str,
    df: pd.DataFrame,
    top_k: int = 10,
) -> list[dict]:
    """Tìm anime tương đồng với 1 anime cho trước (dựa trên document của nó)."""
    match = df[df["name"].str.lower() == anime_name.lower()]
    if match.empty:
        match = df[df["name"].str.contains(anime_name, case=False, na=False, regex=False)]
    if match.empty:
        return []
    doc = match.iloc[0]["document"]
    model = get_model(model_key)
    qvec = model.encode_queries([doc])
    results = milvus_store.search(model_key, qvec, top_k=top_k + 1)
    out = results[0] if results else []
    # loại bỏ chính nó
    return [r for r in out if str(r.get("name", "")).lower() != anime_name.lower()][:top_k]
