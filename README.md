
# Petro Explain Retrieval

Petro Explain Retrieval is a retrieval-based petrology explanation system that generates grounded interpretations from mineral chemistry, metadata, and short petrographic text. Instead of predicting directly, the system retrieves the most similar samples from known datasets and summarizes their collective behavior into a structured explanation.

---

## What This Project Does

This system follows a simple pipeline: raw petrology datasets are cleaned and merged into a unified schema, features are built from chemistry, metadata, and text, embeddings are fused and indexed using FAISS, and query samples are matched against stored records. The retrieved neighbors are then summarized into a controlled explanation describing dominant mineral type, host-rock associations, chemistry trends, and retrieval confidence.

The explanation is grounded in real retrieved evidence, which makes outputs interpretable and traceable.

---

## Data Used

The project currently uses two major sources: the Qin mantle clinopyroxene dataset and selected mineral subsets from the Siebach GEOROC MIST dataset. Can be download from https://georoc.eu/georoc/expert/ . To maintain stability and reduce noise, only selected mineral groups are used: CLINOPYROXENES, OLIVINES, ORTHOPYROXENES, GARNETS, and SPINELS.

Only shared oxide chemistry fields are used across datasets to maintain compatibility.

---

## Key Configuration Constraints

All configuration values are stored in `config.py`. Important constraints include a fixed set of oxide chemistry columns (SiO2, TiO2, Al2O3, Cr2O3, FeO, MnO, MgO, CaO, Na2O, K2O), a unified metadata schema (mineral, rock_name, tectonic_setting, location, rock_texture, rim_core, primary_secondary), and dataset paths for Qin and Siebach sources.

Filtering removes rows with insufficient chemistry values to ensure meaningful retrieval behavior. Feature fusion combines chemistry, metadata, and text embeddings before normalization.

---

## System Flow

The system runs in stages: `prepare_data.py` loads and cleans Qin and Siebach datasets and saves aligned parquet files. `build_dense_index.py` converts cleaned data into chemistry, metadata, and text features, merges them into embeddings, and builds a FAISS search index. `run.py` accepts a query composition, retrieves top matching samples, builds structured evidence, and generates a controlled summary.

All outputs are stored in the `artifacts/outputs/` directory.

---

## Output Files

After running the full pipeline, outputs include retrieved neighbor records, structured JSON evidence, and a readable explanation summary. Files are automatically written into the outputs folder, making it easy to inspect model behavior.

Typical output includes dominant mineral identification, host-rock similarity patterns, oxide value ranges, and similarity confidence scores.

---

## Result Interpretation

The generated summary reflects collective behavior of the retrieved neighbors. Strong conclusions usually come from repeated mineral identity, consistent host-rock patterns, and stable oxide chemistry ranges. Weak or missing fields (such as tectonic setting) are handled conservatively to avoid unsupported claims.

The explanation should be interpreted as evidence-based guidance, not an absolute geological classification.

---

## Known Limitations

Current text coverage is limited because many dataset entries lack detailed petrographic notes. Tectonic setting information is frequently missing. Fusion is currently rule-based rather than learned, and retrieval is dense-only without reranking. These constraints make the system chemistry-dominant in its current version.

Future work may include hybrid retrieval, rule-based geological reasoning, reranking strategies, and integration of image modalities.

---

## Setup Instructions

Create a virtual environment, activate it, and install dependencies including PyTorch with CUDA support, pandas, numpy, scikit-learn, faiss-cpu, sentence-transformers, pyarrow, joblib, and tqdm. After installation, ensure dataset files are placed into the `data/raw/` directory following the expected folder structure.

---

## How To Run

Run the pipeline in three main steps. First execute `python scripts/prepare_data.py` to clean and merge datasets. Then run `python scripts/build_dense_index.py` to create embeddings and build the FAISS index. Finally execute `python run.py` with query parameters such as mineral type and oxide values to generate retrieval and explanation outputs.

Example: python run.py --top_k 5 --mineral CLINOPYROXENE --tectonic_setting "mid-ocean ridge" --rock_name basalt --rock_texture porphyritic --rim_core core --primary_secondary primary --SiO2 51.2 --TiO2 0.4 --Al2O3 3.8 --Cr2O3 0.9 --FeO 5.7 --MnO 0.1 --MgO 15.4 --CaO 21.3 --Na2O 0.5 --K2O 0.01 --petrography "mantle-derived clinopyroxene with preserved core chemistry" --note "high Mg and Ca with low alkalis"

The final explanation and supporting files will appear automatically in the outputs directory.

---

## Current Status

The system successfully performs dataset alignment, feature construction, dense retrieval, and controlled explanation generation. It provides a stable baseline for retrieval-based petrology interpretation and can be extended with additional modalities or advanced reasoning modules.
