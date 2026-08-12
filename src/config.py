"""Project-wide configuration: paths, dates, and constants."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

START_DATE = "2010-01-01"

# Annualization factor for daily data.
TRADING_DAYS_PER_YEAR = 252

# Forecast horizons (trading days) for forward realized volatility targets.
TARGET_HORIZONS = (5, 10, 21)
PRIMARY_HORIZON = 5

# Chronological split fractions: train / validation / test.
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15

RANDOM_SEED = 42


def ensure_directories() -> None:
    """Create all output directories if they do not exist."""
    for directory in (RAW_DATA_DIR, PROCESSED_DATA_DIR, FIGURES_DIR):
        directory.mkdir(parents=True, exist_ok=True)
