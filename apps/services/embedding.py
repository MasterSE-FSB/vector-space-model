"""Bao bọc embedding model (sentence-transformers).

Hỗ trợ 2 model của đồ án và tự động xử lý prefix `query:`/`passage:` của e5.
Model được tải về thư mục `models/<key>` (chỉ tải 1 lần, sau đó dùng offline).
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from apps.config import MODEL_REGISTRY, ModelConfig, settings


def download_model(key: str) -> str:
    """Pull model từ HuggingFace về thư mục local (idempotent)."""
    cfg = _get_cfg(key)
    cfg.local_dir.parent.mkdir(parents=True, exist_ok=True)
    model = SentenceTransformer(cfg.hf_name, token=settings.hf_token)
    model.save(str(cfg.local_dir))
    return str(cfg.local_dir)


def _get_cfg(key: str) -> ModelConfig:
    if key not in MODEL_REGISTRY:
        raise KeyError(f"Model '{key}' chưa được đăng ký. Hợp lệ: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[key]


class EmbeddingModel:
    """Một embedding model + logic encode query/passage."""

    def __init__(self, key: str) -> None:
        self.cfg = _get_cfg(key)
        # Ưu tiên dùng bản local đã tải; fallback về HF hub nếu chưa có
        source = str(self.cfg.local_dir) if self.cfg.local_dir.exists() else self.cfg.hf_name
        self.model = SentenceTransformer(source, token=settings.hf_token)

    @property
    def dim(self) -> int:
        return self.cfg.dim

    def _encode(self, texts: list[str], prefix: str, batch_size: int | None) -> np.ndarray:
        prefixed = [f"{prefix}{t}" for t in texts] if prefix else texts
        emb = self.model.encode(
            prefixed,
            batch_size=batch_size or settings.batch_size,
            normalize_embeddings=True,  # chuẩn hóa L2 -> cosine = inner product
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return emb.astype(np.float32)

    def encode_passages(self, texts: list[str], batch_size: int | None = None) -> np.ndarray:
        return self._encode(texts, self.cfg.passage_prefix, batch_size)

    def encode_queries(self, texts: list[str], batch_size: int | None = None) -> np.ndarray:
        return self._encode(texts, self.cfg.query_prefix, batch_size)


@lru_cache(maxsize=4)
def get_model(key: str) -> EmbeddingModel:
    """Cache model trong tiến trình để FastAPI / notebook tái sử dụng."""
    return EmbeddingModel(key)
