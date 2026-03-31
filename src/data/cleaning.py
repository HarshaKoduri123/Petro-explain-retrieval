from __future__ import annotations

import pandas as pd
import numpy as np

from config import COMMON_METADATA_COLUMNS, COMMON_OXIDE_COLUMNS
from src.utils.constants import CATEGORY_MISSING_TOKEN, TEXT_MISSING_TOKEN


def _coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _ensure_columns(df: pd.DataFrame, columns: list[str], fill_value=np.nan) -> pd.DataFrame:
    for col in columns:
        if col not in df.columns:
            df[col] = fill_value
    return df


def _normalize_text_column(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .fillna(TEXT_MISSING_TOKEN)
        .str.strip()
    )


def _normalize_category_column(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .fillna(CATEGORY_MISSING_TOKEN)
        .str.strip()
        .replace("", CATEGORY_MISSING_TOKEN)
    )


def clean_qin(df: pd.DataFrame) -> pd.DataFrame:
    
    df = df.copy()
    rename_map = {
        "CITATIONS": "citation",
        "SAMPLE NAME": "sample_name",
        "TECTONIC SETTING": "tectonic_setting",
        "LOCATION": "location",
        "LOCATION COMMENT": "location_comment",
        "ROCK NAME": "rock_name",
        "ROCK TEXTURE": "rock_texture",
        "MINERAL": "mineral",
        "SPOT": "spot",
        "RIM/CORE (MINERAL GRAINS)": "rim_core",
        "PRIMARY/SECONDARY": "primary_secondary",
        "Petrography": "petrography",
        "Note": "note",
        "SIO2(WT%)": "SiO2",
        "TIO2(WT%)": "TiO2",
        "AL2O3(WT%)": "Al2O3",
        "CR2O3(WT%)": "Cr2O3",
        "FEOT(WT%)": "FeO",
        "CAO(WT%)": "CaO",
        "MGO(WT%)": "MgO",
        "MNO(WT%)": "MnO",
        "NA2O(WT%)": "Na2O",
        "K2O(WT%)": "K2O",
    }
    df = df.rename(columns=rename_map)

    df["source_dataset"] = "Qin"
    df["source_file"] = "2024-007_AVAW2Y_Qin_data.csv"
    df = _ensure_columns(df, COMMON_OXIDE_COLUMNS, fill_value=np.nan)
    df = _ensure_columns(df, COMMON_METADATA_COLUMNS, fill_value=np.nan)

    for col in COMMON_OXIDE_COLUMNS:
        df[col] = _coerce_numeric(df[col])

    category_cols = [
        "source_dataset",
        "source_file",
        "sample_name",
        "citation",
        "tectonic_setting",
        "location",
        "location_comment",
        "rock_name",
        "rock_texture",
        "mineral",
        "spot",
        "rim_core",
        "primary_secondary",
    ]
    text_cols = ["petrography", "note"]

    for col in category_cols:
        df[col] = _normalize_category_column(df[col])

    for col in text_cols:
        df[col] = _normalize_text_column(df[col])

    df["text_context"] = (
        "Petrography: " + df["petrography"] + " | "
        "Note: " + df["note"]
    )

    keep_cols = COMMON_METADATA_COLUMNS + COMMON_OXIDE_COLUMNS + ["text_context"]
    return df[keep_cols]


def clean_siebach(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    rename_map = {
        "CITATION": "citation",
        "SAMPLENAME": "sample_name",
        "TECTONICSETTING": "tectonic_setting",
        "LOCATION": "location",
        "LOCATIONCOMMENT": "location_comment",
        "ROCKNAME": "rock_name",
        "ROCKTEXTURE": "rock_texture",
        "MINERAL": "mineral",
        "SPOT": "spot",
        "RIM_CORE_MINERALGRAINS_": "rim_core",
        "PRIMARY_SECONDARY": "primary_secondary",
    }
    df = df.rename(columns=rename_map)

    df["source_dataset"] = "Siebach"
    if "source_file" not in df.columns:
        df["source_file"] = df.get("mineral_source_group", "Siebach_subset")

    df = _ensure_columns(df, COMMON_OXIDE_COLUMNS, fill_value=np.nan)
    df = _ensure_columns(df, COMMON_METADATA_COLUMNS, fill_value=np.nan)

    if "petrography" not in df.columns:
        df["petrography"] = TEXT_MISSING_TOKEN
    if "note" not in df.columns:
        df["note"] = TEXT_MISSING_TOKEN

    for col in COMMON_OXIDE_COLUMNS:
        df[col] = _coerce_numeric(df[col])

    category_cols = [
        "source_dataset",
        "source_file",
        "sample_name",
        "citation",
        "tectonic_setting",
        "location",
        "location_comment",
        "rock_name",
        "rock_texture",
        "mineral",
        "spot",
        "rim_core",
        "primary_secondary",
    ]
    text_cols = ["petrography", "note"]

    for col in category_cols:
        df[col] = _normalize_category_column(df[col])

    for col in text_cols:
        df[col] = _normalize_text_column(df[col])

    df["text_context"] = (
        "Petrography: " + df["petrography"] + " | "
        "Note: " + df["note"]
    )

    keep_cols = COMMON_METADATA_COLUMNS + COMMON_OXIDE_COLUMNS + ["text_context"]
    return df[keep_cols]


def basic_filtering(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()
    numeric_presence = df[COMMON_OXIDE_COLUMNS].notna().sum(axis=1)
    df = df.loc[numeric_presence >= 3].reset_index(drop=True)
    return df


def add_record_id(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.insert(0, "record_id", range(1, len(df) + 1))
    return df