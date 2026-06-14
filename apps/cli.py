"""CLI tiện ích cho các tác vụ chạy 1 lần.

Ví dụ:
    python -m apps.cli pull-models      # tải 2 model HF về thư mục models/
    python -m apps.cli import-data      # encode anime & nạp vào Milvus (bỏ qua nếu đã có)
    python -m apps.cli import-data -f   # ép nạp lại (xoá dữ liệu cũ trước)
    python -m apps.cli stats            # xem số vector trong Milvus
"""
from __future__ import annotations

import argparse
import json

from apps.config import MODEL_REGISTRY
from apps.services import milvus_store
from apps.services.data_loader import load_anime
from apps.services.embedding import download_model
from apps.services.search_service import ensure_imported


def cmd_pull_models(_: argparse.Namespace) -> None:
    for key in MODEL_REGISTRY:
        path = download_model(key)
        print(f"[pull] {key} -> {path}")


def cmd_import_data(args: argparse.Namespace) -> None:
    # Chỉ load CSV khi thực sự cần (tránh tốn tài nguyên nếu đã import đủ)
    need = args.force or any(milvus_store.count(k) == 0 for k in MODEL_REGISTRY)
    df = load_anime(min_members=args.min_members) if need else None
    if df is not None:
        print(f"[import] loaded {len(df):,} anime documents")
    for key in MODEL_REGISTRY:
        stats = ensure_imported(key, df=df, force=args.force, min_members=args.min_members)
        print("[import]", json.dumps(stats, ensure_ascii=False))


def cmd_stats(_: argparse.Namespace) -> None:
    for key, cfg in MODEL_REGISTRY.items():
        print(f"{key:18s} {cfg.collection:24s} -> {milvus_store.count(key):,} vectors")


def main() -> None:
    parser = argparse.ArgumentParser(description="Anime VSM utilities")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("pull-models").set_defaults(func=cmd_pull_models)

    p_import = sub.add_parser("import-data")
    p_import.add_argument("--min-members", type=int, default=0)
    p_import.add_argument(
        "-f", "--force", action="store_true",
        help="ép nạp lại dù collection đã có dữ liệu (xoá trước khi nạp)",
    )
    p_import.set_defaults(func=cmd_import_data)

    sub.add_parser("stats").set_defaults(func=cmd_stats)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
