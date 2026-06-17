"""FastAPI app: import dữ liệu & research (truy vấn ngữ nghĩa) cho Anime VSM.

Chạy:
    uvicorn apps.main:app --reload
Docs: http://localhost:8000/docs
"""
from __future__ import annotations

from fastapi import FastAPI

from apps.config import MODEL_REGISTRY, settings
from apps.services import milvus_store
from apps.router.import_router import router as import_router
from apps.router.search_router import router as search_router
from apps.router.tfidf import router as tfidf_router

app = FastAPI(
    title="Anime Vector Space Model API",
    description=(
        "API import dữ liệu anime vào Milvus và truy vấn ngữ nghĩa "
        "bằng 2 embedding model (e5-small-v2 & all-MiniLM-L6-v2)."
    ),
    version="1.0.0",
)

app.include_router(import_router)
app.include_router(search_router)
app.include_router(tfidf_router)


@app.get("/", tags=["meta"], summary="Thông tin & trạng thái")
def root():
    try:
        counts = {k: milvus_store.count(k) for k in MODEL_REGISTRY}
        milvus_ok = True
    except Exception as exc:  # noqa: BLE001
        counts = {"error": str(exc)}
        milvus_ok = False
    return {
        "service": "Anime Vector Space Model API",
        "milvus": f"{settings.milvus_host}:{settings.milvus_port}",
        "milvus_connected": milvus_ok,
        "models": [
            {"key": k, "hf_name": c.hf_name, "dim": c.dim, "collection": c.collection}
            for k, c in MODEL_REGISTRY.items()
        ],
        "vectors": counts,
    }


@app.get("/models", tags=["meta"], summary="Danh sách model")
def models():
    return {
        k: {"hf_name": c.hf_name, "dim": c.dim, "collection": c.collection}
        for k, c in MODEL_REGISTRY.items()
    }
