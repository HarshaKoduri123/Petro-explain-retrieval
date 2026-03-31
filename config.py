from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

# Artifact directories
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
EMBEDDINGS_DIR = ARTIFACTS_DIR / "embeddings"
FAISS_DIR = ARTIFACTS_DIR / "faiss"
OUTPUTS_DIR = ARTIFACTS_DIR / "outputs"

# Raw source folders
QIN_DIR = RAW_DIR / "2024-007_AVAW2Y_Qin_Data"
SIEBACH_DIR = RAW_DIR / "2025-002_Siebach-et-al_Data"

QIN_FILE = QIN_DIR / "2024-007_AVAW2Y_Qin_data.csv"

# Selected Siebach subset
SELECTED_MINERALS = [
    "CLINOPYROXENES",
    "OLIVINES",
    "ORTHOPYROXENES",
    "GARNETS",
    "SPINELS",
]

SIEBACH_FILE_MAP = {
    mineral: SIEBACH_DIR / f"2025-002_Siebach-et-al_MIST_Results_2024-12-SGFTFN_{mineral}.csv"
    for mineral in SELECTED_MINERALS
}

COMMON_OXIDE_COLUMNS = [
    "SiO2",
    "TiO2",
    "Al2O3",
    "Cr2O3",
    "FeO",
    "MnO",
    "MgO",
    "CaO",
    "Na2O",
    "K2O",
]


COMMON_METADATA_COLUMNS = [
    "source_dataset",
    "tectonic_setting",
    "location",
    "rock_name",
    "rock_texture",
    "mineral",
    "rim_core",
    "primary_secondary",
]

# Output files
QIN_CLEANED_FILE = INTERIM_DIR / "qin_cleaned.parquet"
SIEBACH_CLEANED_FILE = INTERIM_DIR / "siebach_selected_cleaned.parquet"
MERGED_FILE = INTERIM_DIR / "merged_samples.parquet"


DEBUG_MAX_ROWS_PER_FILE = None
