"""Lưu trữ & truy vấn vector trên Milvus (dùng MilvusClient API 2.6).

Mỗi model có 1 collection riêng (vì có thể khác số chiều / không gian vector).
"""
from __future__ import annotations

from functools import lru_cache

import math

import numpy as np
from pymilvus import DataType, MilvusClient

from apps.config import MODEL_REGISTRY, settings

_VARCHAR_MAX = 2048


def _safe_float(value: object) -> float:
    try:
        f = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0
    return f if math.isfinite(f) else 0.0


def _safe_int(value: object) -> int:
    f = _safe_float(value)
    return int(f)


@lru_cache(maxsize=1)
def get_client() -> MilvusClient:
    uri = f"http://{settings.milvus_host}:{settings.milvus_port}"
    return MilvusClient(uri=uri)


def collection_name(model_key: str) -> str:
    return MODEL_REGISTRY[model_key].collection


def create_collection(model_key: str, recreate: bool = False) -> str:
    """Tạo collection cho 1 model (HNSW + cosine)."""
    client = get_client()
    cfg = MODEL_REGISTRY[model_key]
    name = cfg.collection

    if client.has_collection(name):
        if not recreate:
            return name
        client.drop_collection(name)

    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field("anime_id", DataType.INT64, is_primary=True)
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=cfg.dim)
    schema.add_field("name", DataType.VARCHAR, max_length=_VARCHAR_MAX)
    schema.add_field("genre", DataType.VARCHAR, max_length=_VARCHAR_MAX)
    schema.add_field("type", DataType.VARCHAR, max_length=64)
    schema.add_field("episodes", DataType.VARCHAR, max_length=32)
    schema.add_field("rating", DataType.FLOAT)
    schema.add_field("members", DataType.INT64)
    schema.add_field("document", DataType.VARCHAR, max_length=_VARCHAR_MAX)

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        index_type="HNSW",
        metric_type="COSINE",
        params={"M": 16, "efConstruction": 200},
    )
    client.create_collection(name, schema=schema, index_params=index_params)
    return name


def insert_embeddings(
    model_key: str,
    rows: list[dict],
    vectors: np.ndarray,
    batch_size: int = 1000,
) -> int:
    """Chèn dữ liệu + vector vào collection. Trả về số bản ghi đã chèn."""
    client = get_client()
    name = collection_name(model_key)
    total = 0
    for i in range(0, len(rows), batch_size):
        chunk = rows[i : i + batch_size]
        vec_chunk = vectors[i : i + batch_size]
        data = []
        for row, vec in zip(chunk, vec_chunk):
            data.append(
                {
                    "anime_id": int(row["anime_id"]),
                    "vector": vec.tolist(),
                    "name": str(row.get("name", ""))[:_VARCHAR_MAX],
                    "genre": str(row.get("genre", ""))[:_VARCHAR_MAX],
                    "type": str(row.get("type", ""))[:64],
                    "episodes": str(row.get("episodes", ""))[:32],
                    "rating": _safe_float(row.get("rating")),
                    "members": _safe_int(row.get("members")),
                    "document": str(row.get("document", ""))[:_VARCHAR_MAX],
                }
            )
        client.insert(collection_name=name, data=data)
        total += len(data)
    client.flush(name)
    return total


def search(
    model_key: str,
    query_vectors: np.ndarray,
    top_k: int = 10,
    output_fields: list[str] | None = None,
) -> list[list[dict]]:
    """Tìm kiếm top-k cho mỗi query vector. Trả về list kết quả theo từng query."""
    client = get_client()
    name = collection_name(model_key)
    client.load_collection(name)
    fields = output_fields or ["anime_id", "name", "genre", "type", "rating", "members"]
    res = client.search(
        collection_name=name,
        data=[v.tolist() for v in query_vectors],
        limit=top_k,
        output_fields=fields,
        search_params={"metric_type": "COSINE", "params": {"ef": max(64, top_k * 4)}},
    )
    results: list[list[dict]] = []
    for hits in res:
        items = []
        for h in hits:
            entity = dict(h.get("entity", {}))
            entity["score"] = float(h.get("distance", 0.0))
            items.append(entity)
        results.append(items)
    return results


def count(model_key: str) -> int:
    client = get_client()
    name = collection_name(model_key)
    if not client.has_collection(name):
        return 0
    client.load_collection(name)
    stats = client.get_collection_stats(name)
    return int(stats.get("row_count", 0))
