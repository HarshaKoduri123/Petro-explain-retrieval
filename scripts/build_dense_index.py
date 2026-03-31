from __future__ import annotations

import sys
from pathlib import Path

import faiss
import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from config import (
    COMMON_METADATA_COLUMNS,
    COMMON_OXIDE_COLUMNS,
    EMBEDDINGS_DIR,
    FAISS_DIR,
    MERGED_FILE,
)
from src.features.chemistry_features import ChemistryFeatureBuilder
from src.features.metadata_features import MetadataFeatureBuilder
from src.features.text_features import TextFeatureBuilder


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, eps, None)


def main() -> None:
    print("=== Stage 2: Building dense retrieval index ===")

    if not MERGED_FILE.exists():
        raise FileNotFoundError(
            f"Merged file not found: {MERGED_FILE}\n"
            f"Run `python scripts\\prepare_data.py` first."
        )

    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    FAISS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading merged data from: {MERGED_FILE}")
    df = pd.read_parquet(MERGED_FILE)
    print(f"Loaded merged dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    # ---------------------------
    # Chemistry features
    # ---------------------------
    print("\nBuilding chemistry features...")
    chemistry_builder = ChemistryFeatureBuilder()
    chemistry_out = chemistry_builder.fit_transform(df)
    np.save(EMBEDDINGS_DIR / "chemistry_embeddings.npy", chemistry_out.matrix)
    chemistry_builder.save(EMBEDDINGS_DIR / "chemistry_builder.joblib")
    print(f"Chemistry matrix shape: {chemistry_out.matrix.shape}")

    # ---------------------------
    # Metadata features
    # ---------------------------
    print("\nBuilding metadata features...")
    metadata_builder = MetadataFeatureBuilder()
    metadata_out = metadata_builder.fit_transform(df)
    np.save(EMBEDDINGS_DIR / "metadata_embeddings.npy", metadata_out.matrix)
    metadata_builder.save(EMBEDDINGS_DIR / "metadata_builder.joblib")
    print(f"Metadata matrix shape: {metadata_out.matrix.shape}")

    # ---------------------------
    # Text features
    # ---------------------------
    print("\nBuilding text embeddings...")
    text_builder = TextFeatureBuilder(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        text_column="text_context",
        batch_size=128,
    )
    text_out = text_builder.fit_transform(df)
    np.save(EMBEDDINGS_DIR / "text_embeddings.npy", text_out.matrix)
    text_builder.save(EMBEDDINGS_DIR / "text_builder.joblib")
    print(f"Text matrix shape: {text_out.matrix.shape}")

    # ---------------------------
    # Feature fusion
    # ---------------------------
    print("\nFusing features...")
    fused = np.concatenate(
        [
            chemistry_out.matrix,
            metadata_out.matrix,
            text_out.matrix,
        ],
        axis=1,
    ).astype(np.float32)

    fused = l2_normalize(fused)
    np.save(EMBEDDINGS_DIR / "fused_embeddings.npy", fused)
    print(f"Fused matrix shape: {fused.shape}")

    # ---------------------------
    # Build FAISS index
    # ---------------------------
    print("\nBuilding FAISS index...")
    dim = fused.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(fused)

    faiss.write_index(index, str(FAISS_DIR / "sample_index.faiss"))
    print(f"FAISS index saved: {FAISS_DIR / 'sample_index.faiss'}")

    # ---------------------------
    # Save record ids
    # ---------------------------
    if "record_id" not in df.columns:
        raise ValueError("Merged dataset must contain 'record_id'.")

    record_ids = df["record_id"].to_numpy()
    np.save(FAISS_DIR / "sample_ids.npy", record_ids)

    # ---------------------------
    # Save retrieval metadata
    # ---------------------------
    retrieval_cols = ["record_id"] + COMMON_METADATA_COLUMNS + COMMON_OXIDE_COLUMNS + ["text_context"]
    retrieval_cols = [col for col in retrieval_cols if col in df.columns]

    retrieval_frame = df[retrieval_cols].copy()
    retrieval_frame.to_parquet(FAISS_DIR / "retrieval_metadata.parquet", index=False)

    # ---------------------------
    # Save feature manifest
    # ---------------------------
    joblib.dump(
        {
            "chemistry_feature_names": chemistry_out.feature_names,
            "metadata_feature_names": metadata_out.feature_names,
            "text_feature_names": text_out.feature_names,
            "fused_dim": dim,
        },
        EMBEDDINGS_DIR / "feature_manifest.joblib",
    )

    print(f"Saved chemistry embeddings : {EMBEDDINGS_DIR / 'chemistry_embeddings.npy'}")
    print(f"Saved metadata embeddings  : {EMBEDDINGS_DIR / 'metadata_embeddings.npy'}")
    print(f"Saved text embeddings      : {EMBEDDINGS_DIR / 'text_embeddings.npy'}")
    print(f"Saved fused embeddings     : {EMBEDDINGS_DIR / 'fused_embeddings.npy'}")
    print(f"Saved retrieval metadata   : {FAISS_DIR / 'retrieval_metadata.parquet'}")
    print(f"Saved sample ids           : {FAISS_DIR / 'sample_ids.npy'}")


if __name__ == "__main__":
    main()