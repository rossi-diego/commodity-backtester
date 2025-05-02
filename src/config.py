from pathlib import Path

# Folders
PROJECT_FOLDER = Path(__file__).resolve().parents[1]
DATA_FOLDER = PROJECT_FOLDER / "data"
RESULTS_FOLDER = PROJECT_FOLDER / "results"

# Data files
YFINANCE_RAW = DATA_FOLDER / "yfinance_raw.parquet"
YFINANCE_CLEAN = DATA_FOLDER / "yfinance_clean.parquet"
YFINANCE_DESCRIBE = DATA_FOLDER / "yfinance_describe.csv"

# Results files
