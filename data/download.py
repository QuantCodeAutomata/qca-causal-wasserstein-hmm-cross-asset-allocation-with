"""
Standalone data download script.

Downloads daily adjusted close prices for the five cross-asset proxies used
in all experiments and writes them to ``data/prices.csv``.

Usage
-----
    python data/download.py [--start YYYY-MM-DD] [--end YYYY-MM-DD]

The script attempts the Massive REST API first (requires the
``MASSIVE_TOKEN`` environment variable) and falls back to yfinance.

Asset universe (fixed, non-reconstituting)
------------------------------------------
    SPX  → ^GSPC   (S&P 500 index)
    BOND → TLT     (iShares 20+ Year Treasury ETF)
    GOLD → GLD     (SPDR Gold Trust)
    OIL  → USO     (United States Oil Fund)
    USD  → UUP     (Invesco DB US Dollar Index)

Output
------
    data/prices.csv — DatetimeIndex × 5 columns (SPX, BOND, GOLD, OIL, USD)
"""

import argparse
import os
import sys
from pathlib import Path

# Allow import of src/ from any working directory
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import ASSET_TICKERS, GLOBAL_START, GLOBAL_END
from src.data_utils import download_prices


def main() -> None:
    parser = argparse.ArgumentParser(description="Download cross-asset price data")
    parser.add_argument("--start", default=GLOBAL_START, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default=GLOBAL_END, help="End date YYYY-MM-DD")
    args = parser.parse_args()

    print(f"Downloading prices for {list(ASSET_TICKERS.values())} …")
    prices = download_prices(ASSET_TICKERS, args.start, args.end)

    out = Path(__file__).parent / "prices.csv"
    prices.to_csv(out)
    print(f"Saved {len(prices)} rows × {len(prices.columns)} assets → {out}")


if __name__ == "__main__":
    main()
