from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


@dataclass
class TextFeatureOutput:
    matrix: np.ndarray
    feature_names: list[str]


class TextFeatureBuilder:

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        text_column: str = "text_context",
        batch_size: int = 128,
    ) -> None:
        self.model_name = model_name
        self.text_column = text_column
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

    def fit(self, df: pd.DataFrame) -> "TextFeatureBuilder":
        return self

    def transform(self, df: pd.DataFrame) -> TextFeatureOutput:
        texts = (
            df[self.text_column]
            .fillna("")
            .astype(str)
            .str.strip()
            .tolist()
        )

        matrix = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)

        feature_names = [f"text_emb_{i}" for i in range(matrix.shape[1])]
        return TextFeatureOutput(matrix=matrix, feature_names=feature_names)

    def fit_transform(self, df: pd.DataFrame) -> TextFeatureOutput:
        self.fit(df)
        return self.transform(df)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model_name": self.model_name,
                "text_column": self.text_column,
                "batch_size": self.batch_size,
                "embedding_dim": self.embedding_dim,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "TextFeatureBuilder":
        obj = joblib.load(path)
        builder = cls(
            model_name=obj["model_name"],
            text_column=obj["text_column"],
            batch_size=obj["batch_size"],
        )
        builder.embedding_dim = obj["embedding_dim"]
        return builder