"""Market data acquisition via yfinance, with local CSV caching.

All series are daily. SPY provides OHLCV + adjusted close, ^VIX provides
the CBOE volatility index, and ^TNX / ^IRX provide Treasury-yield proxies
(10-year and 13-week CBOE yield indices). The macro series are optional:
the rest of the pipeline works if they are unavailable.
"""

from __future__ import annotations

import logging

import pandas as pd
import yfinance as yf

from src.config import RAW_DATA_DIR, START_DATE, ensure_directories

logger = logging.getLogger(__name__)

SPY_TICKER = "SPY"
VIX_TICKER = "^VIX"
MACRO_TICKERS = {"^TNX": "yield_10y", "^IRX": "yield_3m"}


def _download(ticker: str, start: str) -> pd.DataFrame:
    """Download daily bars for one ticker and return a flat-column frame."""
    df = yf.download(ticker, start=start, auto_adjust=False, progress=False)
    if df is None or df.empty:
        raise RuntimeError(f"No data returned for {ticker}")
    # Newer yfinance versions return (field, ticker) MultiIndex columns.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.DatetimeIndex(df.index).tz_localize(None)
    df.index.name = "Date"
    return df.sort_index()


def _load_or_download(ticker: str, filename: str, start: str, force: bool) -> pd.DataFrame:
    """Return cached CSV if present, otherwise download and cache it."""
    ensure_directories()
    path = RAW_DATA_DIR / filename
    if path.exists() and not force:
        logger.info("Loading cached %s from %s", ticker, path)
        return pd.read_csv(path, index_col="Date", parse_dates=True)
    df = _download(ticker, start)
    df.to_csv(path)
    logger.info("Downloaded %s: %d rows (%s to %s)", ticker, len(df), df.index[0].date(), df.index[-1].date())
    return df


def load_spy(start: str = START_DATE, force: bool = False) -> pd.DataFrame:
    """Load SPY OHLCV + adjusted close."""
    return _load_or_download(SPY_TICKER, "spy.csv", start, force)


def load_vix(start: str = START_DATE, force: bool = False) -> pd.DataFrame:
    """Load VIX index levels (close is the standard quoted level)."""
    return _load_or_download(VIX_TICKER, "vix.csv", start, force)


def load_macro(start: str = START_DATE, force: bool = False) -> pd.DataFrame | None:
    """Load optional Treasury-yield proxies. Returns None if unavailable."""
    frames = []
    for ticker, name in MACRO_TICKERS.items():
        try:
            df = _load_or_download(ticker, f"{name}.csv", start, force)
            frames.append(df["Close"].rename(name))
        except Exception as exc:  # macro data is optional by design
            logger.warning("Skipping macro series %s (%s): %s", name, ticker, exc)
    if not frames:
        return None
    return pd.concat(frames, axis=1)


def build_market_dataset(
    start: str = START_DATE,
    include_macro: bool = True,
    force: bool = False,
) -> pd.DataFrame:
    """Merge SPY, VIX, and optional macro series into one daily frame.

    The result is indexed by trading date with columns:
    open, high, low, close, adj_close, volume, vix, [yield_10y, yield_3m].
    VIX and macro columns are forward-filled at most 5 days to bridge
    isolated holiday mismatches, never back-filled (no look-ahead).
    """
    spy = load_spy(start, force)
    data = pd.DataFrame(
        {
            "open": spy["Open"],
            "high": spy["High"],
            "low": spy["Low"],
            "close": spy["Close"],
            "adj_close": spy["Adj Close"],
            "volume": spy["Volume"],
        }
    )

    vix = load_vix(start, force)
    data["vix"] = vix["Close"].reindex(data.index).ffill(limit=5)

    if include_macro:
        macro = load_macro(start, force)
        if macro is not None:
            for col in macro.columns:
                data[col] = macro[col].reindex(data.index).ffill(limit=5)

    data = data.dropna(subset=["adj_close", "vix"])
    return data


def summarize_dataset(data: pd.DataFrame) -> str:
    """Human-readable dataset summary used in logs and the README."""
    lines = [
        f"Observations: {len(data):,}",
        f"Date range:   {data.index[0].date()} to {data.index[-1].date()}",
        f"Columns:      {', '.join(data.columns)}",
        f"Missing values per column:\n{data.isna().sum().to_string()}",
    ]
    return "\n".join(lines)
