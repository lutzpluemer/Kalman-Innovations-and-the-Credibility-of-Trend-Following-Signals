"""
Downloads a small, diverse demo panel from Yahoo Finance for the public repo.
This is INTENTIONALLY separate from the paper's licensed EODHD panel -- Yahoo
data has no redistribution restriction, so this script + its output can be
committed to a public repository, unlike the paper's own data.

Run once: python download_demo_data.py
Produces: data/demo_panel.parquet (columns: ticker, date, close, volume)
"""
import yfinance as yf
import pandas as pd
from pathlib import Path

DEMO_TICKERS = [
    "AAPL", "MSFT", "JNJ", "PG", "XOM", "JPM", "KO", "WMT", "DIS", "CAT",
    "BA", "PFE", "NKE", "IBM", "GE",
]

START_DATE = "2005-01-01"
OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "demo_panel.parquet"


def main():
    frames = []
    for tk in DEMO_TICKERS:
        print(f"Downloading {tk} ...")
        df = yf.download(tk, start=START_DATE, progress=False, auto_adjust=True)
        if df.empty:
            print(f"  WARNING: no data for {tk}, skipping.")
            continue
        df = df.reset_index()
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        df = df.rename(columns={"Date": "date", "Close": "close", "Volume": "volume"})
        df["ticker"] = tk
        frames.append(df[["ticker", "date", "close", "volume"]])

    panel = pd.concat(frames, ignore_index=True)
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(OUT_PATH)
    print(f"\nSaved {len(panel)} rows, {panel['ticker'].nunique()} tickers -> {OUT_PATH}")


if __name__ == "__main__":
    main()
