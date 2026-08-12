# Data

All data is downloaded automatically by the pipeline — nothing needs to be
placed here manually.

```
data/
├── raw/          # cached CSVs downloaded from Yahoo Finance (gitignored)
└── processed/    # engineered feature matrix (gitignored)
```

## Sources

| Series | Ticker | Source | Notes |
|---|---|---|---|
| S&P 500 ETF (OHLCV + adjusted close) | `SPY` | Yahoo Finance via `yfinance` | daily bars from 2010 |
| CBOE Volatility Index | `^VIX` | Yahoo Finance via `yfinance` | close = standard quoted level |
| 10-year Treasury yield proxy | `^TNX` | Yahoo Finance via `yfinance` | optional macro feature |
| 13-week T-bill yield proxy | `^IRX` | Yahoo Finance via `yfinance` | optional macro feature |

The macro series are optional: if either download fails, the pipeline logs
a warning and continues without them.

To force a fresh download (e.g. to extend the sample to today):

```bash
python run_pipeline.py --refresh
```

No API keys are required for any data source.
