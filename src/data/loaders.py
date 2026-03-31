from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional

import pandas as pd

from config import DEBUG_MAX_ROWS_PER_FILE


def _safe_read_csv(
    file_path: Path,
    usecols: Optional[Iterable[str]] = None,
    nrows: Optional[int] = None,
) -> pd.DataFrame:

    if nrows is None:
        nrows = DEBUG_MAX_ROWS_PER_FILE

    return pd.read_csv(
        file_path,
        usecols=usecols,
        nrows=nrows,
        low_memory=False,
        encoding="utf-8",
    )


def load_qin_raw(qin_file: Path) -> pd.DataFrame:
    df = _safe_read_csv(qin_file)
    return df


def load_siebach_raw_file(file_path: Path) -> pd.DataFrame:

    if not file_path.exists():
        raise FileNotFoundError(f"Siebach file not found: {file_path}")

    df = _safe_read_csv(file_path)
    return df


def load_selected_siebach_raw(file_map: Dict[str, Path]) -> pd.DataFrame:
    frames = []

    for mineral_group, file_path in file_map.items():
        print(f"Loading Siebach file: {file_path.name}")
        df = load_siebach_raw_file(file_path)
        df["mineral_source_group"] = mineral_group
        frames.append(df)

    merged = pd.concat(frames, ignore_index=True)
    return merged