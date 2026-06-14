"""Tải & tiền xử lý dataset Anime (Kaggle) cho bài toán Vector Space Model.

Mỗi anime được biến thành một "document" văn bản để embedding & truy vấn ngữ nghĩa.
"""
from __future__ import annotations

import re

import pandas as pd

from apps.config import settings


def _clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    text = text.replace("&quot;", '"').replace("&#039;", "'").replace("&amp;", "&")
    text = re.sub(r"\s+", " ", text)
    return text


def build_document(row: pd.Series) -> str:
    """Sinh đoạn văn mô tả 1 anime từ các trường name/genre/type/episodes.

    Đây là "passage" đưa vào không gian vector để truy vấn ngữ nghĩa.
    """
    name = _clean_text(row.get("name"))
    genre = _clean_text(row.get("genre"))
    atype = _clean_text(row.get("type"))
    episodes = _clean_text(row.get("episodes"))

    parts = [name]
    if genre:
        parts.append(f"Genres: {genre}.")
    if atype:
        parts.append(f"Type: {atype}.")
    if episodes and episodes.lower() != "unknown":
        parts.append(f"Episodes: {episodes}.")
    return " ".join(parts)


def load_anime(csv_path: str | None = None, min_members: int = 0) -> pd.DataFrame:
    """Đọc anime.csv, làm sạch và sinh cột `document`.

    Args:
        csv_path: đường dẫn file (mặc định lấy từ settings).
        min_members: lọc bỏ anime có quá ít người xem (giảm nhiễu, tùy chọn).
    """
    path = csv_path or settings.anime_csv
    df = pd.read_csv(path)

    # Chuẩn hóa kiểu & làm sạch
    df = df.dropna(subset=["name"]).copy()
    df["name"] = df["name"].map(_clean_text)
    df["genre"] = df["genre"].fillna("").map(_clean_text)
    df["type"] = df["type"].fillna("").map(_clean_text)
    df["members"] = pd.to_numeric(df["members"], errors="coerce").fillna(0).astype(int)
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

    if min_members > 0:
        df = df[df["members"] >= min_members].copy()

    df = df.drop_duplicates(subset=["anime_id"]).reset_index(drop=True)
    df["document"] = df.apply(build_document, axis=1)
    df["genre_set"] = df["genre"].map(
        lambda g: frozenset(x.strip() for x in g.split(",") if x.strip())
    )
    return df
