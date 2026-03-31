from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from config import OUTPUTS_DIR
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.evidence_builder import (
    build_compact_evidence,
    build_structured_evidence,
)


def build_query_dataframe_from_args(args: argparse.Namespace) -> pd.DataFrame:
    row = {
        "source_dataset": "query",
        "tectonic_setting": args.tectonic_setting or "unknown",
        "location": args.location or "unknown",
        "rock_name": args.rock_name or "unknown",
        "rock_texture": args.rock_texture or "unknown",
        "mineral": args.mineral or "unknown",
        "rim_core": args.rim_core or "unknown",
        "primary_secondary": args.primary_secondary or "unknown",
        "SiO2": args.SiO2,
        "TiO2": args.TiO2,
        "Al2O3": args.Al2O3,
        "Cr2O3": args.Cr2O3,
        "FeO": args.FeO,
        "MnO": args.MnO,
        "MgO": args.MgO,
        "CaO": args.CaO,
        "Na2O": args.Na2O,
        "K2O": args.K2O,
    }

    petrography = args.petrography or ""
    note = args.note or ""
    row["text_context"] = f"Petrography: {petrography} | Note: {note}"

    return pd.DataFrame([row])


def print_results(results) -> None:
    if not results:
        print("No results found.")
        return

    print("\n=== Top Retrieved Samples ===")
    for result in results:
        row = result.row
        print(f"\nRank: {result.rank}")
        print(f"Score: {result.score:.4f}")
        print(f"Record ID: {result.record_id}")
        print(f"Mineral: {row.get('mineral', 'unknown')}")
        print(f"Rock Name: {row.get('rock_name', 'unknown')}")
        print(f"Tectonic Setting: {row.get('tectonic_setting', 'unknown')}")
        print(f"Location: {row.get('location', 'unknown')}")
        print(f"Rock Texture: {row.get('rock_texture', 'unknown')}")
        print(f"Rim/Core: {row.get('rim_core', 'unknown')}")
        print(f"Primary/Secondary: {row.get('primary_secondary', 'unknown')}")
        print(f"Text Context: {row.get('text_context', '')}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retrieve top-k similar petrology samples from the dense FAISS index."
    )

    parser.add_argument("--top_k", type=int, default=5)

    # Metadata
    parser.add_argument("--tectonic_setting", type=str, default="unknown")
    parser.add_argument("--location", type=str, default="unknown")
    parser.add_argument("--rock_name", type=str, default="unknown")
    parser.add_argument("--rock_texture", type=str, default="unknown")
    parser.add_argument("--mineral", type=str, default="unknown")
    parser.add_argument("--rim_core", type=str, default="unknown")
    parser.add_argument("--primary_secondary", type=str, default="unknown")
    parser.add_argument("--petrography", type=str, default="")
    parser.add_argument("--note", type=str, default="")

    # Chemistry
    parser.add_argument("--SiO2", type=float, default=None)
    parser.add_argument("--TiO2", type=float, default=None)
    parser.add_argument("--Al2O3", type=float, default=None)
    parser.add_argument("--Cr2O3", type=float, default=None)
    parser.add_argument("--FeO", type=float, default=None)
    parser.add_argument("--MnO", type=float, default=None)
    parser.add_argument("--MgO", type=float, default=None)
    parser.add_argument("--CaO", type=float, default=None)
    parser.add_argument("--Na2O", type=float, default=None)
    parser.add_argument("--K2O", type=float, default=None)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    query_df = build_query_dataframe_from_args(args)

    retriever = DenseRetriever().load()
    results = retriever.search(query_df=query_df, top_k=args.top_k)

    print_results(results)

    compact_evidence = build_compact_evidence(results)
    structured_evidence = build_structured_evidence(results)

    compact_path = OUTPUTS_DIR / "retrieval_evidence.txt"
    json_path = OUTPUTS_DIR / "retrieval_results.json"

    compact_path.write_text(compact_evidence, encoding="utf-8")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(structured_evidence, f, indent=2, ensure_ascii=False)

    print("\nSaved:")
    print(f"- {compact_path}")
    print(f"- {json_path}")


if __name__ == "__main__":
    main()