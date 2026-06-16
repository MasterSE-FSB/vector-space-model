"""Traditional sparse Vector Space Model baseline using TF-IDF."""
from __future__ import annotations

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TFIDFSearchService:
    """Traditional Vector Space Model using sparse TF-IDF vectors."""

    def __init__(
        self,
        max_features: int = 20000,
        ngram_range: tuple[int, int] = (1, 2),
        min_df: int = 2,
    ):
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            norm="l2",
        )
        self.df: pd.DataFrame | None = None
        self.matrix = None

    def fit(self, df: pd.DataFrame, text_col: str = "document"):
        if text_col not in df.columns:
            raise ValueError(f"Missing text column: {text_col}")

        self.df = df.reset_index(drop=True).copy()
        self.matrix = self.vectorizer.fit_transform(
            self.df[text_col].fillna("").astype(str)
        )
        return self

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        if self.df is None or self.matrix is None:
            raise RuntimeError("TFIDFSearchService must be fitted before search.")

        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.matrix).flatten()
        top_indices = scores.argsort()[::-1][:top_k]

        return [self._row_to_result(idx, float(scores[idx])) for idx in top_indices]

    def search_similar_to_anime(
        self,
        anime_name: str,
        top_k: int = 10,
    ) -> list[dict]:
        if self.df is None or self.matrix is None:
            raise RuntimeError("TFIDFSearchService must be fitted before search.")

        if "name" not in self.df.columns:
            raise ValueError("DataFrame must contain a 'name' column.")

        matched = self.df[
            self.df["name"].astype(str).str.lower().str.contains(
                anime_name.lower(),
                na=False,
                regex=False,
            )
        ]
        if matched.empty:
            raise ValueError(f"No anime found: {anime_name}")

        query_idx = matched.index[0]
        scores = cosine_similarity(self.matrix[query_idx], self.matrix).flatten()
        ranked_indices = scores.argsort()[::-1]
        top_indices = [idx for idx in ranked_indices if idx != query_idx][:top_k]

        return [self._row_to_result(idx, float(scores[idx])) for idx in top_indices]

    def _row_to_result(self, idx: int, score: float) -> dict:
        if self.df is None:
            raise RuntimeError("TFIDFSearchService must be fitted before search.")

        row = self.df.iloc[idx]
        result = {
            "anime_id": self._safe_int(row.get("anime_id", idx)),
            "name": row.get("name", ""),
            "genre": row.get("genre", ""),
            "type": row.get("type", ""),
            "score": score,
        }

        if "members" in row.index:
            result["members"] = self._safe_int(row.get("members"))

        for rating_col in ("rating", "Rating", "Score", "score"):
            if rating_col in row.index:
                result["rating"] = self._safe_float(row.get(rating_col))
                break

        if "Popularity" in row.index:
            result["popularity"] = self._safe_int(row.get("Popularity"))

        return result

    @staticmethod
    def _safe_int(value):
        try:
            if pd.isna(value):
                return None
            return int(value)
        except Exception:
            return None

    @staticmethod
    def _safe_float(value):
        try:
            if pd.isna(value):
                return None
            return float(value)
        except Exception:
            return None
