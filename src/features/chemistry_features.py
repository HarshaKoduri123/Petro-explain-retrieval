from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import COMMON_OXIDE_COLUMNS


@dataclass
class ChemistryFeatureOutput:
    matrix: np.ndarray
    feature_names: list[str]


class ChemistryFeatureBuilder:

    def __init__(self, oxide_columns: Iterable[str] | None = None) -> None:
        self.oxide_columns = list(oxide_columns or COMMON_OXIDE_COLUMNS)
        self.pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

    def fit(self, df: pd.DataFrame) -> "ChemistryFeatureBuilder":
        x = df[self.oxide_columns].copy()
        self.pipeline.fit(x)
        return self

    def transform(self, df: pd.DataFrame) -> ChemistryFeatureOutput:
        x = df[self.oxide_columns].copy()
        matrix = self.pipeline.transform(x).astype(np.float32)
        return ChemistryFeatureOutput(matrix=matrix, feature_names=self.oxide_columns)

    def fit_transform(self, df: pd.DataFrame) -> ChemistryFeatureOutput:
        self.fit(df)
        return self.transform(df)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "oxide_columns": self.oxide_columns,
                "pipeline": self.pipeline,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "ChemistryFeatureBuilder":
        obj = joblib.load(path)
        builder = cls(obj["oxide_columns"])
        builder.pipeline = obj["pipeline"]
        return builder