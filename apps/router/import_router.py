"""Router: import dữ liệu vào Milvus (chạy 1 lần)."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from apps.controller import import_controller
from apps.router.schemas import ImportRequest

router = APIRouter(prefix="/import", tags=["import"])


@router.post("", summary="Encode anime & nạp vào Milvus (idempotent, chạy 1 lần)")
def import_data(req: ImportRequest):
    try:
        stats = import_controller.run_import(
            models=req.models, force=req.force, min_members=req.min_members
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"status": "ok", "imported": stats}


@router.get("/stats", summary="Số vector hiện có trong từng collection")
def stats():
    return import_controller.collection_stats()
