from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import pandas as pd

from config import EMBEDDINGS_DIR, FAISS_DIR
from src.features.chemistry_features import ChemistryFeatureBuilder
from src.features.metadata_features import MetadataFeatureBuilder
from src.features.text_features import TextFeatureBuilder


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, eps, None)


@dataclass
class RetrievalResult:
    rank: int
    score: float
    record_id: int
    row: dict[str, Any]


class DenseRetriever:
    def __init__(
        self,
        faiss_index_path: str | Path | None = None,
        sample_ids_path: str | Path | None = None,
        retrieval_metadata_path: str | Path | None = None,
        chemistry_builder_path: str | Path | None = None,
        metadata_builder_path: str | Path | None = None,
        text_builder_path: str | Path | None = None,
    ) -> None:
        self.faiss_index_path = Path(faiss_index_path or (FAISS_DIR / "sample_index.faiss"))
        self.sample_ids_path = Path(sample_ids_path or (FAISS_DIR / "sample_ids.npy"))
        self.retrieval_metadata_path = Path(
            retrieval_metadata_path or (FAISS_DIR / "retrieval_metadata.parquet")
        )

        self.chemistry_builder_path = Path(
            chemistry_builder_path or (EMBEDDINGS_DIR / "chemistry_builder.joblib")
        )
        self.metadata_builder_path = Path(
            metadata_builder_path or (EMBEDDINGS_DIR / "metadata_builder.joblib")
        )
        self.text_builder_path = Path(
            text_builder_path or (EMBEDDINGS_DIR / "text_builder.joblib")
        )

        self.index: faiss.Index | None = None
        self.sample_ids: np.ndarray | None = None
        self.metadata_df: pd.DataFrame | None = None

        self.chemistry_builder: ChemistryFeatureBuilder | None = None
        self.metadata_builder: MetadataFeatureBuilder | None = None
        self.text_builder: TextFeatureBuilder | None = None

    def load(self) -> "DenseRetriever":
        if not self.faiss_index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {self.faiss_index_path}")
        if not self.sample_ids_path.exists():
            raise FileNotFoundError(f"Sample IDs file not found: {self.sample_ids_path}")
        if not self.retrieval_metadata_path.exists():
            raise FileNotFoundError(
                f"Retrieval metadata not found: {self.retrieval_metadata_path}"
            )

        self.index = faiss.read_index(str(self.faiss_index_path))
        self.sample_ids = np.load(self.sample_ids_path)
        self.metadata_df = pd.read_parquet(self.retrieval_metadata_path)

        self.chemistry_builder = ChemistryFeatureBuilder.load(self.chemistry_builder_path)
        self.metadata_builder = MetadataFeatureBuilder.load(self.metadata_builder_path)
        self.text_builder = TextFeatureBuilder.load(self.text_builder_path)

        return self

    def _ensure_loaded(self) -> None:
        if (
            self.index is None
            or self.sample_ids is None
            or self.metadata_df is None
            or self.chemistry_builder is None
            or self.metadata_builder is None
            or self.text_builder is None
        ):
            raise RuntimeError("DenseRetriever not loaded. Call `.load()` first.")

    def build_query_embedding(self, query_df: pd.DataFrame) -> np.ndarray:
        self._ensure_loaded()

        chemistry_out = self.chemistry_builder.transform(query_df)
        metadata_out = self.metadata_builder.transform(query_df)
        text_out = self.text_builder.transform(query_df)

        fused = np.concatenate(
            [
                chemistry_out.matrix,
                metadata_out.matrix,
                text_out.matrix,
            ],
            axis=1,
        ).astype(np.float32)

        fused = l2_normalize(fused)
        return fused

    def search(self, query_df: pd.DataFrame, top_k: int = 5) -> list[RetrievalResult]:
        self._ensure_loaded()

        if len(query_df) != 1:
            raise ValueError("search expects a single-row query DataFrame.")

        query_vec = self.build_query_embedding(query_df)
        scores, indices = self.index.search(query_vec, top_k)

        scores = scores[0]
        indices = indices[0]

        results: list[RetrievalResult] = []

        for rank, (score, idx) in enumerate(zip(scores, indices), start=1):
            if idx < 0:
                continue

            record_id = int(self.sample_ids[idx])

            matched = self.metadata_df[self.metadata_df["record_id"] == record_id]
            if matched.empty:
                continue

            row_dict = matched.iloc[0].to_dict()

            results.append(
                RetrievalResult(
                    rank=rank,
                    score=float(score),
                    record_id=record_id,
                    row=row_dict,
                )
            )

        return results