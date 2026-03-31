from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from config import OUTPUTS_DIR
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.evidence_builder import (
    build_compact_evidence,
    build_structured_evidence,
)
from src.summarization.controlled_summary import summarize_retrieval_results


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="End-to-end petrology explanation pipeline: retrieval + controlled summarization."
    )

    parser.add_argument("--top_k", type=int, default=5)

    parser.add_argument("--tectonic_setting", type=str, default="unknown")
    parser.add_argument("--location", type=str, default="unknown")
    parser.add_argument("--rock_name", type=str, default="unknown")
    parser.add_argument("--rock_texture", type=str, default="unknown")
    parser.add_argument("--mineral", type=str, default="unknown")
    parser.add_argument("--rim_core", type=str, default="unknown")
    parser.add_argument("--primary_secondary", type=str, default="unknown")
    parser.add_argument("--petrography", type=str, default="")
    parser.add_argument("--note", type=str, default="")

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

    print("=== Petrology Explanation Pipeline ===")

    print("\n[1/4] Building query...")
    query_df = build_query_dataframe_from_args(args)
    print(query_df.to_string(index=False))

    # Save query input
    query_input_path = OUTPUTS_DIR / "query_input.json"
    query_dict = query_df.iloc[0].to_dict()

    with open(query_input_path, "w", encoding="utf-8") as f:
        json.dump(
            query_dict,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("\nSaved query input:")
    print(f"- {query_input_path}")

    print("\n[2/4] Loading retriever and searching...")
    retriever = DenseRetriever().load()
    results = retriever.search(query_df=query_df, top_k=args.top_k)

    if not results:
        print("No retrieval results found.")
        return

    print(f"Retrieved {len(results)} neighbors.")

    print("\nTop matches:")
    for result in results:
        row = result.row
        print(
            f"- Rank {result.rank} | score={result.score:.4f} | "
            f"record_id={result.record_id} | "
            f"mineral={row.get('mineral', 'unknown')} | "
            f"rock_name={row.get('rock_name', 'unknown')} | "
            f"tectonic_setting={row.get('tectonic_setting', 'unknown')}"
        )

    print("\n[3/4] Building evidence...")
    compact_evidence = build_compact_evidence(results)
    structured_evidence = build_structured_evidence(results)

    print("\n[4/4] Generating controlled summary...")
    summary_artifacts = summarize_retrieval_results(structured_evidence)

    print("\n=== Controlled Summary ===\n")
    print(summary_artifacts.summary)

    compact_path = OUTPUTS_DIR / "retrieval_evidence.txt"
    retrieval_json_path = OUTPUTS_DIR / "retrieval_results.json"
    summary_txt_path = OUTPUTS_DIR / "controlled_summary.txt"
    summary_json_path = OUTPUTS_DIR / "controlled_summary.json"

    compact_path.write_text(compact_evidence, encoding="utf-8")

    with open(retrieval_json_path, "w", encoding="utf-8") as f:
        json.dump(structured_evidence, f, indent=2, ensure_ascii=False)

    summary_txt_path.write_text(summary_artifacts.summary, encoding="utf-8")

    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "summary": summary_artifacts.summary,
                "stats": summary_artifacts.stats,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("\nSaved outputs:")
    print(f"- {query_input_path}")
    print(f"- {compact_path}")
    print(f"- {retrieval_json_path}")
    print(f"- {summary_txt_path}")
    print(f"- {summary_json_path}")


if __name__ == "__main__":
    main()