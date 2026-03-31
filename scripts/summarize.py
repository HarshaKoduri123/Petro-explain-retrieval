from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from config import OUTPUTS_DIR
from src.summarization.controlled_summary import summarize_retrieval_results


def main() -> None:
    retrieval_json = OUTPUTS_DIR / "retrieval_results.json"
    summary_txt = OUTPUTS_DIR / "controlled_summary.txt"
    summary_json = OUTPUTS_DIR / "controlled_summary.json"

    if not retrieval_json.exists():
        raise FileNotFoundError(
            f"Retrieval results not found: {retrieval_json}\n"
            f"Run `python scripts\\retrieve.py ...` first."
        )

    with open(retrieval_json, "r", encoding="utf-8") as f:
        results = json.load(f)

    artifacts = summarize_retrieval_results(results)

    summary_txt.write_text(artifacts.summary, encoding="utf-8")

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "summary": artifacts.summary,
                "stats": artifacts.stats,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("\n=== Controlled Summary ===\n")
    print(artifacts.summary)
    print("\nSaved:")
    print(f"- {summary_txt}")
    print(f"- {summary_json}")


if __name__ == "__main__":
    main()