"""Controller: nghiệp vụ import dữ liệu vào Milvus."""
from __future__ import annotations

from apps.config import MODEL_REGISTRY
from apps.services import milvus_store
from apps.services.data_loader import load_anime
from apps.services.search_service import ensure_imported


def run_import(
    models: list[str] | None,
    force: bool,
    min_members: int,
) -> list[dict]:
    """Import idempotent. Mặc định bỏ qua model đã có dữ liệu; `force=True` để nạp lại."""
    keys = models or list(MODEL_REGISTRY.keys())
    invalid = [k for k in keys if k not in MODEL_REGISTRY]
    if invalid:
        raise ValueError(f"Model không hợp lệ: {invalid}. Hợp lệ: {list(MODEL_REGISTRY)}")

    # Chỉ tốn công load CSV khi thực sự cần import
    need_import = force or any(milvus_store.count(k) == 0 for k in keys)
    df = load_anime(min_members=min_members) if need_import else None

    return [ensure_imported(key, df=df, force=force, min_members=min_members) for key in keys]


def collection_stats() -> dict[str, int]:
    return {key: milvus_store.count(key) for key in MODEL_REGISTRY}
