from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from config import (
    QIN_CLEANED_FILE,
    QIN_FILE,
    SIEBACH_CLEANED_FILE,
    SIEBACH_FILE_MAP,
    MERGED_FILE,
)
from src.data.loaders import load_qin_raw, load_selected_siebach_raw
from src.data.cleaning import clean_qin, clean_siebach, basic_filtering, add_record_id
from src.data.merging import merge_qin_siebach
from src.utils.io import print_df_info, save_parquet


def main() -> None:
    print("=== Stage 1: Preparing data ===")

    # Load raw
    print("\nLoading Qin raw data...")
    qin_raw = load_qin_raw(QIN_FILE)
    print_df_info("Qin raw", qin_raw)

    print("\nLoading selected Siebach raw data...")
    siebach_raw = load_selected_siebach_raw(SIEBACH_FILE_MAP)
    print_df_info("Siebach raw (selected subset)", siebach_raw)

    # Clean
    print("\nCleaning Qin...")
    qin_clean = clean_qin(qin_raw)
    qin_clean = basic_filtering(qin_clean)
    qin_clean = add_record_id(qin_clean)
    print_df_info("Qin cleaned", qin_clean)

    print("\nCleaning Siebach...")
    siebach_clean = clean_siebach(siebach_raw)
    siebach_clean = basic_filtering(siebach_clean)
    siebach_clean = add_record_id(siebach_clean)
    print_df_info("Siebach cleaned", siebach_clean)

    # Save cleaned
    print("\nSaving cleaned intermediate files...")
    save_parquet(qin_clean, QIN_CLEANED_FILE)
    save_parquet(siebach_clean, SIEBACH_CLEANED_FILE)

    # Merge
    print("\nMerging Qin + Siebach...")
    merged = merge_qin_siebach(qin_clean, siebach_clean)

    # Rebuild clean global record_id after merge
    merged = merged.drop(columns=["record_id"], errors="ignore").reset_index(drop=True)
    merged = add_record_id(merged)

    print_df_info("Merged dataset", merged)

    print("\nSaving merged dataset...")
    save_parquet(merged, MERGED_FILE)

    print("\nDone.")
    print(f"Saved Qin cleaned    -> {QIN_CLEANED_FILE}")
    print(f"Saved Siebach cleaned-> {SIEBACH_CLEANED_FILE}")
    print(f"Saved merged         -> {MERGED_FILE}")


if __name__ == "__main__":
    main()