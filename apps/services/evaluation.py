"""Đánh giá & so sánh chất lượng truy vấn ngữ nghĩa giữa các model.

Vì dataset không có nhãn liên quan (relevance) sẵn, ta xây dựng "ground-truth giả lập"
dựa trên độ trùng lặp thể loại (genre) giữa các anime — đây là tín hiệu liên quan
hợp lý cho bài toán "tìm anime tương tự".

Các chỉ số: Precision@k, Recall@k, MRR, nDCG@k.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from apps.services import milvus_store
from apps.services.embedding import get_model


def jaccard(a: frozenset, b: frozenset) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def build_relevance(
    df: pd.DataFrame,
    query_ids: list[int] | None = None,
    threshold: float = 0.5,
) -> dict[int, set[int]]:
    """Tập liên quan của mỗi query = anime trong corpus có Jaccard(genre) >= threshold.

    Chỉ tính cho các `query_ids` (mặc định: toàn bộ) nhưng đối chiếu trên TOÀN corpus
    `df`, nên metric phản ánh đúng khả năng truy hồi trên kho dữ liệu thật.

    Trả về dict: anime_id -> set(anime_id liên quan).
    """
    ids = df["anime_id"].to_numpy()
    genre_sets = df["genre_set"].tolist()
    id_to_idx = {int(a): i for i, a in enumerate(ids)}

    if query_ids is None:
        query_ids = [int(a) for a in ids]

    relevance: dict[int, set[int]] = {}
    for qid in query_ids:
        qi = id_to_idx[qid]
        gi = genre_sets[qi]
        rel: set[int] = set()
        if gi:
            for j in range(len(df)):
                if j == qi:
                    continue
                if jaccard(gi, genre_sets[j]) >= threshold:
                    rel.add(int(ids[j]))
        relevance[qid] = rel
    return relevance


def precision_at_k(retrieved: list[int], relevant: set[int], k: int) -> float:
    if k == 0:
        return 0.0
    top = retrieved[:k]
    return sum(1 for r in top if r in relevant) / k


def recall_at_k(retrieved: list[int], relevant: set[int], k: int) -> float:
    if not relevant:
        return 0.0
    top = retrieved[:k]
    return sum(1 for r in top if r in relevant) / len(relevant)


def reciprocal_rank(retrieved: list[int], relevant: set[int]) -> float:
    for idx, r in enumerate(retrieved, start=1):
        if r in relevant:
            return 1.0 / idx
    return 0.0


def ndcg_at_k(retrieved: list[int], relevant: set[int], k: int) -> float:
    top = retrieved[:k]
    dcg = sum(1.0 / np.log2(idx + 1) for idx, r in enumerate(top, start=1) if r in relevant)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(idx + 1) for idx in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_model(
    model_key: str,
    df: pd.DataFrame,
    relevance: dict[int, set[int]],
    query_ids: list[int],
    k: int = 10,
) -> dict:
    """Chạy đánh giá retrieval cho 1 model trên tập query đã cho.

    Query = document của anime; loại bỏ chính nó khỏi kết quả trước khi chấm điểm.
    """
    model = get_model(model_key)
    id_to_doc = dict(zip(df["anime_id"], df["document"]))
    query_docs = [id_to_doc[qid] for qid in query_ids]

    t0 = time.perf_counter()
    qvecs = model.encode_queries(query_docs)
    encode_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    raw = milvus_store.search(model_key, qvecs, top_k=k + 1)
    search_time = time.perf_counter() - t1

    precisions, recalls, rrs, ndcgs = [], [], [], []
    for qid, hits in zip(query_ids, raw):
        retrieved = [int(h["anime_id"]) for h in hits if int(h["anime_id"]) != qid][:k]
        rel = relevance.get(qid, set())
        if not rel:
            continue
        precisions.append(precision_at_k(retrieved, rel, k))
        recalls.append(recall_at_k(retrieved, rel, k))
        rrs.append(reciprocal_rank(retrieved, rel))
        ndcgs.append(ndcg_at_k(retrieved, rel, k))

    n = len(precisions)
    return {
        "model": model_key,
        "num_queries": n,
        f"Precision@{k}": round(float(np.mean(precisions)), 4) if n else 0.0,
        f"Recall@{k}": round(float(np.mean(recalls)), 4) if n else 0.0,
        "MRR": round(float(np.mean(rrs)), 4) if n else 0.0,
        f"nDCG@{k}": round(float(np.mean(ndcgs)), 4) if n else 0.0,
        "encode_time_s": round(encode_time, 3),
        "search_time_s": round(search_time, 3),
        "ms_per_query": round(1000 * (encode_time + search_time) / max(len(query_ids), 1), 2),
    }


def compare_models(
    model_keys: list[str],
    df: pd.DataFrame,
    relevance: dict[int, set[int]],
    query_ids: list[int],
    k: int = 10,
) -> pd.DataFrame:
    rows = [evaluate_model(mk, df, relevance, query_ids, k=k) for mk in model_keys]
    return pd.DataFrame(rows).set_index("model")
