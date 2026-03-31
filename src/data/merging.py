from __future__ import annotations

import pandas as pd


def merge_qin_siebach(qin_df: pd.DataFrame, siebach_df: pd.DataFrame) -> pd.DataFrame:
    merged = pd.concat([qin_df, siebach_df], ignore_index=True)
    return merged