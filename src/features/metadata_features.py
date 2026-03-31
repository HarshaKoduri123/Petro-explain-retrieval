from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer

from config import COMMON_METADATA_COLUMNS


@dataclass
class MetadataFeatureOutput:
    matrix: np.ndarray
    feature_names: list[str]


class MetadataFeatureBuilder:

    def __init__(self, metadata_columns: Iterable[str] | None = None) -> None:
        self.metadata_columns = list(metadata_columns or COMMON_METADATA_COLUMNS)
        self.vectorizer = DictVectorizer(sparse=True)

    def _rows_to_dicts(self, df: pd.DataFrame) -> list[dict[str, str]]:
        subset = df[self.metadata_columns].fillna("unknown").astype(str)
        return subset.to_dict(orient="records")

    def fit(self, df: pd.DataFrame) -> "MetadataFeatureBuilder":
        records = self._rows_to_dicts(df)
        self.vectorizer.fit(records)
        return self

    def transform(self, df: pd.DataFrame) -> MetadataFeatureOutput:
        records = self._rows_to_dicts(df)
        sparse_matrix = self.vectorizer.transform(records)
        matrix = sparse_matrix.astype(np.float32).toarray()
        feature_names = list(self.vectorizer.get_feature_names_out())
        return MetadataFeatureOutput(matrix=matrix, feature_names=feature_names)

    def fit_transform(self, df: pd.DataFrame) -> MetadataFeatureOutput:
        self.fit(df)
        return self.transform(df)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "metadata_columns": self.metadata_columns,
                "vectorizer": self.vectorizer,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "MetadataFeatureBuilder":
        obj = joblib.load(path)
        builder = cls(obj["metadata_columns"])
        builder.vectorizer = obj["vectorizer"]
        return builder