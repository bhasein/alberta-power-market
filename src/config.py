from pathlib import Path

# project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / 'data'

RAW_DIR = DATA_DIR / "raw"
PREPROCESSING_DIR = DATA_DIR / "preprocessing"
PROCESSED_DIR = DATA_DIR / "processed"
DICTIONARIES_DIR = DATA_DIR / "dictionaries"

# raw directories
MARKET_DIR = RAW_DIR / "market"
WEATHER_DIR = RAW_DIR / "weather"
GENERATION_DIR = RAW_DIR / "generation"
INTERTIE_DIR = RAW_DIR / "intertie"

# notebook directory
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# canonical processed files
MASTER_DATA_FILE = PROCESSED_DIR / "alberta_energy_master_2020_2025_hourly_utc.csv"
MASTER_DICTIONARY_FILE = DICTIONARIES_DIR / "alberta_energy_master_2020_2025_dictionary.csv"

# canonical time settings
TIMEZONE = 'UTC'
TIMESTAMP_COL = "timestamp"

# weather locations
WEATHER_LOCATIONS = [
    "calgary",
    "edmonton",
    "lethbridge",
    "medicine_hat",
]