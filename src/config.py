from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"

RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

MARKET_DIR = RAW_DIR / "market"
WEATHER_DIR = RAW_DIR / "weather"

NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

WEATHER_LOCATIONS = [
    "calgary",
    "edmonton",
    "lethbridge",
    "medicine_hat"
]