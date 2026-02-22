"""
Generate Intraday CSV Data - yfinance 1m, 5m, 30m, 1h

Downloads OHLCV from yfinance and caches to CSV per timeframe.

Yahoo intraday limits:
  1m  1-minute data   Last 7 days (up to 30 days total via 7-day chunks)
  5m  5-minute data  Last 59 days (within Yahoo 60-day limit)
  30m 30-minute data Last 59 days
  1h  1-hour data    Last 59 days

Best practices to avoid throttling:
  - Batch requests (yf.download(tickers_list) for 5m/30m/1h)
  - Random delays (200ms–1s between calls)
  - Optional session caching (requests_cache)
  - Rotate User-Agent header

Output: ppo/saved_data/csv_data/
  - trading_data_1m.csv, trading_data_5m.csv, trading_data_30m.csv, trading_data_1h.csv
  - data_manifest_1m.json, ... (optional manifest per interval)

Run: python generate_intraday_csv.py [--intervals 1m 5m 30m 1h] [--force]
"""

import sys
import logging
import json
import hashlib
import argparse
import time
import random
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

try:
    import requests_cache
    REQUESTS_CACHE_AVAILABLE = True
except ImportError:
    REQUESTS_CACHE_AVAILABLE = False

# User-Agent rotation to reduce bot detection (best practice)
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Same tickers as generate_csv_data (no BTC-USD for intraday on Yahoo if needed, add if supported)
DEFAULT_TICKERS = [
    "AAPL", "MSFT", "NVDA", "JPM", "SPY", "GLD", "KO", "JNJ", "AMZN", "XLE", "TLT"
]

# Yahoo Finance intraday limits (days)
# 1m: 7 days per request; up to 30 days total via 7-day chunks
INTERVAL_LIMITS = {
    "1m": 7,    # 1-minute: last 7 days (chunked fetch can extend to 30 days)
    "5m": 59,   # 5-minute: stay within Yahoo 60-day limit (59 avoids boundary reject)
    "30m": 59,  # 30-minute: stay within Yahoo 60-day limit
    "1h": 59,   # 1-hour: stay within Yahoo 60-day limit
}
MAX_1M_DAYS_VIA_CHUNKS = 30  # 1m can be extended to 30 days by requesting 7-day chunks
# yfinance interval string (1h can be "1h" or "60m")
YF_INTERVALS = {
    "1m": "1m",
    "5m": "5m",
    "30m": "30m",
    "1h": "1h",
}

# Rate limiting: random delay 200ms–1s between calls (best practice)
DELAY_TICKER_MIN_SEC = 0.2
DELAY_TICKER_MAX_SEC = 1.0
DELAY_INTERVAL_MIN_SEC = 2.0
DELAY_INTERVAL_MAX_SEC = 5.0


def file_hash(path: Path) -> str:
    if not path.exists():
        return ""
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _random_delay(min_sec: float, max_sec: float) -> None:
    """Sleep a random duration (human-like behavior to avoid throttling)."""
    delay = random.uniform(min_sec, max_sec)
    if delay > 0:
        time.sleep(delay)


def _get_session(use_cache: bool = False, cache_expire_sec: int = 300):
    """Build a session with rotated User-Agent; optionally use requests_cache."""
    if REQUESTS_CACHE_AVAILABLE and use_cache:
        session = requests_cache.CachedSession(
            "yfinance_intraday_cache",
            expire_after=cache_expire_sec,
            allowable_codes=(200,),
        )
    else:
        import requests
        session = requests.Session()
    session.headers["User-Agent"] = random.choice(USER_AGENTS)
    return session


def _batched_download_to_long(
    batch_df: pd.DataFrame,
    tickers: list,
) -> pd.DataFrame:
    """Convert yf.download(..., group_by='ticker') multi-column result to long (date, Ticker, open, ...)."""
    if batch_df.empty:
        return pd.DataFrame()
    # group_by='ticker' gives columns (Ticker, OHLCV) or single ticker gives plain OHLCV
    if isinstance(batch_df.columns, pd.MultiIndex):
        all_dfs = []
        level0 = batch_df.columns.get_level_values(0)
        seen = set()
        for ticker in level0.unique():
            if ticker in seen:
                continue
            seen.add(ticker)
            if ticker not in batch_df.columns:
                continue
            sub = batch_df[ticker].copy()
            if sub.empty:
                continue
            sub = sub.reset_index()
            sub = sub.rename(columns={
                "Date": "date", "Datetime": "date",
                "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume",
            })
            needed = ["date", "open", "high", "low", "close", "volume"]
            if not all(c in sub.columns for c in needed):
                continue
            sub["Ticker"] = ticker
            sub = sub[["date", "Ticker", "open", "high", "low", "close", "volume"]]
            sub["date"] = pd.to_datetime(sub["date"])
            all_dfs.append(sub)
        return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
    # Single ticker: columns are Open, High, Low, Close, Volume
    batch_df = batch_df.reset_index()
    batch_df = batch_df.rename(columns={
        "Date": "date", "Datetime": "date",
        "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume",
    })
    if "date" not in batch_df.columns:
        batch_df["date"] = batch_df.index
    needed = ["date", "open", "high", "low", "close", "volume"]
    if not all(c in batch_df.columns for c in needed):
        return pd.DataFrame()
    batch_df["Ticker"] = tickers[0] if tickers else "UNKNOWN"
    return batch_df[["date", "Ticker", "open", "high", "low", "close", "volume"]].copy()


def _normalize_ohlcv_df(df: pd.DataFrame, ticker: str) -> pd.DataFrame | None:
    """Reset index, rename Date/Datetime to date, keep OHLCV + Ticker. Returns None if missing cols."""
    df = df.reset_index()
    rename_map = {
        "Date": "date", "Datetime": "date",
        "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    needed = ["date", "open", "high", "low", "close", "volume"]
    if not all(c in df.columns for c in needed):
        return None
    df["Ticker"] = ticker
    df = df[["date", "Ticker", "open", "high", "low", "close", "volume"]].copy()
    df["date"] = pd.to_datetime(df["date"])
    return df


def download_intraday_yfinance(
    tickers: list,
    interval: str,
    days_back: int,
    session=None,
    use_batch: bool = True,
) -> pd.DataFrame:
    """
    Download intraday OHLCV from yfinance with batch requests and random delays.
    interval: one of 1m, 5m, 30m, 1h
    days_back: max days to fetch (1m: 7 or up to 30 via 7-day chunks; 5m/30m/1h: 59).
    session: optional requests session (User-Agent + cache); used for yf.download and Ticker.
    use_batch: if True, use yf.download(tickers_list) for 5m/30m/1h (fewer API calls).
    """
    if not YFINANCE_AVAILABLE:
        raise RuntimeError("yfinance is required. Install: pip install yfinance")

    yf_interval = YF_INTERVALS.get(interval, interval)
    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=days_back)
    start_str = start_dt.strftime("%Y-%m-%d")
    end_str = end_dt.strftime("%Y-%m-%d")

    # Best practice: batch request for 5m/30m/1h (single API call for all tickers)
    if use_batch and interval != "1m" and len(tickers) > 1:
        logger.info(f"Batch download {interval} for {len(tickers)} tickers...")
        _random_delay(DELAY_TICKER_MIN_SEC, DELAY_TICKER_MAX_SEC)
        try:
            kwargs = {"interval": yf_interval, "start": start_str, "end": end_str, "auto_adjust": True, "group_by": "ticker", "progress": False}
            if session is not None:
                kwargs["session"] = session
            batch_df = yf.download(tickers, **kwargs)
            out = _batched_download_to_long(batch_df, tickers)
            if not out.empty:
                out = out.sort_values(["Ticker", "date"]).reset_index(drop=True)
                logger.info(f"  Batched: {len(out)} rows")
                return out
        except Exception as e:
            logger.warning(f"Batch download failed ({e}), falling back to per-ticker...")
        # Fallback to per-ticker below

    all_dfs = []
    for i, ticker in enumerate(tickers, 1):
        if i > 1:
            _random_delay(DELAY_TICKER_MIN_SEC, DELAY_TICKER_MAX_SEC)

        logger.info(f"[{i}/{len(tickers)}] {ticker} ({interval})...")
        try:
            t = yf.Ticker(ticker)
            if session is not None and hasattr(t, "_session"):
                t._session = session
            if interval == "1m" and days_back > 7:
                # 1m: fetch in 7-day chunks up to 30 days total (Yahoo: 7d per request)
                chunks = []
                total_days = min(days_back, MAX_1M_DAYS_VIA_CHUNKS)
                for chunk_start in range(0, total_days, 7):
                    chunk_end_d = end_dt - timedelta(days=chunk_start)
                    chunk_start_d = end_dt - timedelta(days=min(chunk_start + 7, total_days))
                    if chunk_start > 0:
                        _random_delay(DELAY_TICKER_MIN_SEC, DELAY_TICKER_MAX_SEC)
                    sub = t.history(
                        interval=yf_interval,
                        start=chunk_start_d.strftime("%Y-%m-%d"),
                        end=chunk_end_d.strftime("%Y-%m-%d"),
                        auto_adjust=True,
                    )
                    if not sub.empty:
                        chunks.append(sub)
                df = pd.concat(chunks, axis=0).drop_duplicates().sort_index() if chunks else pd.DataFrame()
            elif interval == "1m":
                df = t.history(interval=yf_interval, period="7d", auto_adjust=True)
            else:
                df = t.history(interval=yf_interval, start=start_str, end=end_str, auto_adjust=True)

            if df.empty:
                logger.warning(f"  No data for {ticker}")
                continue

            df = _normalize_ohlcv_df(df, ticker)
            if df is None:
                logger.warning(f"  Missing columns for {ticker}")
                continue
            all_dfs.append(df)
            logger.info(f"  Rows: {len(df)}")
        except Exception as e:
            logger.warning(f"  Failed {ticker}: {e}")
            continue

    if not all_dfs:
        raise RuntimeError(f"No data downloaded for any ticker at interval {interval}")

    out = pd.concat(all_dfs, ignore_index=True)
    out = out.sort_values(["Ticker", "date"]).reset_index(drop=True)
    return out


def save_intraday_csv(
    df: pd.DataFrame,
    data_dir: Path,
    interval: str,
    manifest: bool = True,
) -> Path:
    """Save DataFrame to trading_data_{interval}.csv and optional manifest."""
    data_dir.mkdir(parents=True, exist_ok=True)
    safe = interval.replace(" ", "_")
    csv_path = data_dir / f"trading_data_{safe}.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved {csv_path} ({len(df)} rows)")

    if manifest:
        manifest_path = data_dir / f"data_manifest_{safe}.json"
        manifest_data = {
            "interval": interval,
            "created_at": datetime.now().isoformat(),
            "num_rows": len(df),
            "num_tickers": int(df["Ticker"].nunique()),
            "date_min": str(df["date"].min()),
            "date_max": str(df["date"].max()),
            "columns": df.columns.tolist(),
            "csv_hash": file_hash(csv_path),
        }
        with open(manifest_path, "w") as f:
            json.dump(manifest_data, f, indent=2)
        logger.info(f"Saved {manifest_path}")

    return csv_path


def main():
    parser = argparse.ArgumentParser(description="Generate intraday CSV from yfinance (1m, 5m, 30m, 1h)")
    parser.add_argument(
        "--intervals",
        nargs="+",
        default=["1m", "5m", "30m", "1h"],
        help="Intervals to download (default: 1m 5m 30m 1h)",
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=DEFAULT_TICKERS,
        help="Ticker symbols",
    )
    parser.add_argument(
        "--data-dir",
        default="ppo/saved_data/csv_data",
        help="Directory for CSV files",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing CSV files",
    )
    parser.add_argument(
        "--no-batch",
        action="store_true",
        help="Disable batch download (request one ticker at a time for 5m/30m/1h).",
    )
    parser.add_argument(
        "--cache",
        action="store_true",
        help="Use requests_cache to avoid re-downloading same data (pip install requests-cache).",
    )
    parser.add_argument(
        "--cache-expire",
        type=int,
        default=300,
        metavar="SEC",
        help="Cache expiry in seconds when --cache is used (default: 300).",
    )
    parser.add_argument(
        "--1m-days",
        type=int,
        default=7,
        choices=[7, 14, 21, 30],
        metavar="N",
        dest="one_m_days",
        help="For 1m only: days to fetch via 7-day chunks (7, 14, 21, or 30; default 7).",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    intervals = [s.strip().lower() for s in args.intervals]

    session = _get_session(use_cache=getattr(args, "cache", False), cache_expire_sec=getattr(args, "cache_expire", 300))
    if getattr(args, "cache", False) and REQUESTS_CACHE_AVAILABLE:
        logger.info("Using requests_cache to avoid re-downloading same data within cache window")

    for idx_iv, iv in enumerate(intervals):
        if iv not in INTERVAL_LIMITS:
            logger.warning(f"Unknown interval '{iv}', skip. Use one of: 1m, 5m, 30m, 1h")
            continue

        days_back = INTERVAL_LIMITS[iv]
        if iv == "1m":
            days_back = getattr(args, "one_m_days", 7)
        safe = iv.replace(" ", "_")
        csv_path = data_dir / f"trading_data_{safe}.csv"

        if csv_path.exists() and not args.force:
            logger.info(f"Skip {iv}: {csv_path} exists (use --force to overwrite)")
            continue

        # Random delay between interval batches (best practice: human-like)
        if idx_iv > 0:
            _random_delay(DELAY_INTERVAL_MIN_SEC, DELAY_INTERVAL_MAX_SEC)

        logger.info(f"Downloading {iv} (max {days_back} days)...")
        try:
            df = download_intraday_yfinance(
                args.tickers,
                iv,
                days_back,
                session=session,
                use_batch=not getattr(args, "no_batch", False),
            )
            save_intraday_csv(df, data_dir, iv, manifest=True)
        except Exception as e:
            logger.error(f"Failed {iv}: {e}")
            raise

    logger.info("Done. Intraday CSVs: " + ", ".join(str(p) for p in data_dir.glob("trading_data_*.csv")))


if __name__ == "__main__":
    main()
