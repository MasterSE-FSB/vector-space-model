"""Cấu hình tập trung cho toàn bộ pipeline (Mục 3) và FastAPI.

Đọc biến môi trường từ file `.env` ở thư mục gốc dự án.
"""
from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Thư mục gốc dự án (…/final-project)
ROOT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT_DIR / "kangle" / "dataset"
MODELS_DIR = ROOT_DIR / "models"


class ModelConfig:
    """Mô tả 1 embedding model.

    e5 yêu cầu prefix `query:` / `passage:`; MiniLM không cần prefix.
    """

    def __init__(
        self,
        key: str,
        hf_name: str,
        dim: int,
        collection: str,
        query_prefix: str = "",
        passage_prefix: str = "",
    ) -> None:
        self.key = key
        self.hf_name = hf_name
        self.dim = dim
        self.collection = collection
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix

    @property
    def local_dir(self) -> Path:
        return MODELS_DIR / self.key


# Đăng ký 2 model dùng cho đồ án
MODEL_REGISTRY: dict[str, ModelConfig] = {
    "e5-small-v2": ModelConfig(
        key="e5-small-v2",
        hf_name="intfloat/e5-small-v2",
        dim=384,
        collection="anime_e5_small_v2",
        query_prefix="query: ",
        passage_prefix="passage: ",
    ),
    "all-MiniLM-L6-v2": ModelConfig(
        key="all-MiniLM-L6-v2",
        hf_name="sentence-transformers/all-MiniLM-L6-v2",
        dim=384,
        collection="anime_minilm_l6_v2",
        query_prefix="",
        passage_prefix="",
    ),
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ROOT_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # HuggingFace
    hf_token: str | None = None

    # Milvus
    milvus_host: str = "localhost"
    milvus_port: str = "19530"

    # Dataset
    anime_csv: str = str(DATASET_DIR / "anime.csv")
    rating_csv: str = str(DATASET_DIR / "rating.csv")

    # Encoding
    batch_size: int = 256


settings = Settings()
