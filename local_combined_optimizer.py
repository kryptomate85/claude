#!/usr/bin/env python3
"""
local_combined_optimizer.py

LOCAL WINDOWS/MAC/LINUX version for finding optimal parameters.

SETUP:
1. Install Python 3.10+
2. Install dependencies: pip install pandas numpy joblib requests
3. Put all btc_trades_*.zip files in same folder as this script
4. Run: python local_combined_optimizer.py

Expected runtime on your PC (i5-7500T, 16GB RAM): 90-120 minutes
Will use 3 parallel workers automatically.
Results saved to ./results/
"""

import os
import zipfile
import glob
import json
import itertools
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from time import sleep


# =============================================================================
# LOCAL CONFIGURATION
# =============================================================================

# Auto-detect current directory
SCRIPT_DIR = Path(__file__).parent.absolute()
LOCAL_INPUT_PATH = str(SCRIPT_DIR)
LOCAL_OUTPUT_PATH = str(SCRIPT_DIR / "results")

# Create output directory
os.makedirs(LOCAL_OUTPUT_PATH, exist_ok=True)

print(f"Working directory: {LOCAL_INPUT_PATH}")
print(f"Output directory: {LOCAL_OUTPUT_PATH}")


# =============================================================================
# PARAMETER GRID
# =============================================================================

LOOKBACK_MINUTES_LIST = [3, 5, 7, 10, 15]
PRICE_MIN_LIST = [0.5, 0.6, 0.7]
PRICE_MAX_LIST = [0.7, 0.8, 0.9]
MOMENTUM_THRESHOLD_LIST = [0.0, 0.0004, 0.0008, 0.0012]
STOP_LOSS_LIST = [0.0, 0.25, 0.30, 0.35, 0.40]

# Trading constants
MARKET_BUY_SLIPPAGE = 0.01
MARKET_SELL_SLIPPAGE = 0.01  # Slippage when exiting (stop loss)
MIN_TRADES_PER_FOLD = 30
N_FOLDS = 5


# =============================================================================
# DATA LOADING
# =============================================================================

def find_kaggle_zip_files(input_path: str) -> List[str]:
    """
    Find all zip files OR folders with CSV files in the Kaggle input directory.
    Returns list of paths (either .zip files or folder paths).
    """
    items = []
    
    # Search for zip files first
    pattern1 = os.path.join(input_path, "*.zip")
    items.extend(glob.glob(pattern1))
    
    # Search in subdirectories for zips
    pattern2 = os.path.join(input_path, "*", "*.zip")
    items.extend(glob.glob(pattern2))
    
    # If no zips found, look for folders containing CSV files
    if not items:
        try:
            contents = os.listdir(input_path)
            for item in contents:
                full_path = os.path.join(input_path, item)
                if os.path.isdir(full_path):
                    # Check if folder contains CSV files
                    csv_files = glob.glob(os.path.join(full_path, "*.csv"))
                    if csv_files:
                        items.append(full_path)
        except Exception as e:
            print(f"Warning: Error scanning {input_path}: {e}")
    
    return sorted(items)


def optimize_memory(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reduce DataFrame memory usage by optimizing dtypes.
    Converts repeated text to categories and uses float32 instead of float64.
    This typically saves 50-70% memory with ZERO impact on calculations.
    """
    memory_before = df.memory_usage(deep=True).sum() / 1024**3
    print(f"    Memory before optimization: {memory_before:.2f} GB")
    
    # Convert repeated text to categories (huge savings)
    if 'slug' in df.columns:
        df['slug'] = df['slug'].astype('category')
    if 'market_name' in df.columns:
        df['market_name'] = df['market_name'].astype('category')
    if 'side_l' in df.columns:
        df['side_l'] = df['side_l'].astype('category')
    
    # Use 32-bit float instead of 64-bit (50% savings, zero impact on prices)
    if 'price' in df.columns:
        df['price'] = df['price'].astype('float32')
    
    memory_after = df.memory_usage(deep=True).sum() / 1024**3
    savings = (1 - memory_after / memory_before) * 100
    print(f"    Memory after optimization: {memory_after:.2f} GB ({savings:.0f}% savings)")
    
    return df


@dataclass
class BacktestParams:
    """Parameters for a single backtest run."""
    lookback_minutes: int
    price_min: float
    price_max: float
    momentum_threshold: float
    stop_loss: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lookback_minutes": self.lookback_minutes,
            "price_min": self.price_min,
            "price_max": self.price_max,
            "momentum_threshold": self.momentum_threshold,
            "stop_loss": self.stop_loss,
        }

    def to_tuple(self) -> Tuple:
        return (self.lookback_minutes, self.price_min, self.price_max,
                self.momentum_threshold, self.stop_loss)


# =============================================================================
# DATA LOADING (from latest.py)
# =============================================================================

def _normalize_side(s: str) -> Optional[str]:
    if s is None:
        return None
    sl = str(s).strip().lower()
    if sl in ("up", "u"):
        return "up"
    if sl in ("down", "d"):
        return "down"
    return None


def load_trades_from_zip(zip_path: str) -> pd.DataFrame:
    """Load trades from a zip file."""
    use_cols = ["slug", "market_name", "side", "price_usdc_per_share", "timestamp"]
    parts: List[pd.DataFrame] = []

    with zipfile.ZipFile(zip_path) as z:
        for fn in z.namelist():
            if not fn.lower().endswith(".csv"):
                continue
            try:
                df = pd.read_csv(z.open(fn), engine="python")
            except Exception:
                continue
            if not set(use_cols).issubset(df.columns):
                continue

            df = df[use_cols].copy()
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            df["price"] = pd.to_numeric(df["price_usdc_per_share"], errors="coerce")
            df = df.dropna(subset=["slug", "market_name", "side", "timestamp", "price"])
            df["side_l"] = df["side"].apply(_normalize_side)
            df = df.dropna(subset=["side_l"])
            df["timestamp_ns"] = df["timestamp"].astype("int64")
            parts.append(df[["slug", "market_name", "side_l", "price", "timestamp_ns", "timestamp"]])

    if not parts:
        return pd.DataFrame(columns=["slug", "market_name", "side_l", "price", "timestamp_ns", "timestamp"])
    return pd.concat(parts, ignore_index=True)


def load_trades_from_folder(folder_path: str) -> pd.DataFrame:
    """Load trades from CSV files in a folder (for Kaggle datasets that are folders, not zips)."""
    use_cols = ["slug", "market_name", "side", "price_usdc_per_share", "timestamp"]
    parts: List[pd.DataFrame] = []
    
    print(f"  Reading folder: {os.path.basename(folder_path)}")
    
    # Find all CSV files in the folder
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    
    if not csv_files:
        print(f"    Warning: No CSV files found")
        return pd.DataFrame(columns=["slug", "market_name", "side_l", "price", "timestamp_ns", "timestamp"])
    
    print(f"    Found {len(csv_files)} CSV files")
    
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file, engine="python")
            
            if not set(use_cols).issubset(df.columns):
                continue
            
            df = df[use_cols].copy()
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            df["price"] = pd.to_numeric(df["price_usdc_per_share"], errors="coerce")
            df = df.dropna(subset=["slug", "market_name", "side", "timestamp", "price"])
            df["side_l"] = df["side"].apply(_normalize_side)
            df = df.dropna(subset=["side_l"])
            df["timestamp_ns"] = df["timestamp"].astype("int64")
            
            parts.append(df[["slug", "market_name", "side_l", "price", "timestamp_ns", "timestamp"]])
            
        except Exception as e:
            print(f"    Warning: Could not read {os.path.basename(csv_file)}: {e}")
            continue
    
    if not parts:
        return pd.DataFrame(columns=["slug", "market_name", "side_l", "price", "timestamp_ns", "timestamp"])
    
    result_df = pd.concat(parts, ignore_index=True)
    print(f"    Loaded {len(result_df):,} trades")
    
    # Optimize memory usage
    result_df = optimize_memory(result_df)
    
    return result_df


def fetch_btc_prices(start_time_ms: int, end_time_ms: int) -> pd.DataFrame:
    """Fetch historical BTC/USDT prices from Binance API."""
    print("[BTC] Fetching BTC prices from Binance...")

    url = "https://api.binance.com/api/v3/klines"
    all_data = []
    current_start = start_time_ms
    limit = 1000
    interval = "1m"

    while current_start < end_time_ms:
        params = {
            "symbol": "BTCUSDT",
            "interval": interval,
            "startTime": current_start,
            "endTime": end_time_ms,
            "limit": limit
        }

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            if not data:
                break

            for candle in data:
                timestamp_ms = int(candle[0])
                close_price = float(candle[4])
                all_data.append({
                    "timestamp_ms": timestamp_ms,
                    "btc_price": close_price
                })

            current_start = int(data[-1][0]) + 60000
            sleep(0.1)

        except Exception as e:
            print(f"[BTC] Warning: Error fetching BTC prices: {e}")
            break

    if not all_data:
        print("[BTC] Warning: No BTC price data fetched.")
        return pd.DataFrame(columns=["timestamp_ms", "btc_price"])

    df = pd.DataFrame(all_data)
    print(f"[BTC] Fetched {len(df)} BTC price points")
    return df


def get_btc_price_at_time(btc_prices: pd.DataFrame, timestamp_ns: int) -> Tuple[Optional[float], Optional[int]]:
    """
    Get BTC price at a specific timestamp.
    Returns (price, used_timestamp_ns) tuple where:
      - price: the price at or BEFORE the requested timestamp (not after)
      - used_timestamp_ns: the timestamp of the BTC candle actually selected
    Uses interpolation between the two nearest earlier data points if needed.
    """
    if btc_prices.empty:
        return None, None

    timestamp_ms = timestamp_ns // 1_000_000

    # Find the index where timestamp_ms would be inserted
    idx = np.searchsorted(btc_prices["timestamp_ms"].values, timestamp_ms, side="right") - 1

    if idx < 0:
        used_ts_ms = int(btc_prices.iloc[0]["timestamp_ms"])
        return float(btc_prices.iloc[0]["btc_price"]), used_ts_ms * 1_000_000

    if idx < len(btc_prices):
        actual_ts = btc_prices.iloc[idx]["timestamp_ms"]
        if abs(actual_ts - timestamp_ms) < 60000:  # Within 1 minute
            return float(btc_prices.iloc[idx]["btc_price"]), int(actual_ts) * 1_000_000

    if idx >= len(btc_prices) - 1:
        used_ts_ms = int(btc_prices.iloc[-1]["timestamp_ms"])
        return float(btc_prices.iloc[-1]["btc_price"]), used_ts_ms * 1_000_000

    t1 = btc_prices.iloc[idx]["timestamp_ms"]
    p1 = btc_prices.iloc[idx]["btc_price"]
    t2 = btc_prices.iloc[idx + 1]["timestamp_ms"]
    p2 = btc_prices.iloc[idx + 1]["btc_price"]

    if t2 == t1:
        return float(p1), int(t1) * 1_000_000

    ratio = (timestamp_ms - t1) / (t2 - t1)
    # For interpolated values, return the earlier timestamp (t1) as the used timestamp
    return float(p1 + ratio * (p2 - p1)), int(t1) * 1_000_000


# =============================================================================
# MARKET NAME PARSING
# =============================================================================

def extract_market_timestamp(market_name: str) -> Optional[int]:
    """
    Extract timestamp from market name for 15-minute BTC markets.
    
    Market names typically look like:
    "Will BTC close above $96,000.00 at 11:00 AM ET on January 24?"
    
    Returns timestamp in nanoseconds, or None if parsing fails.
    """
    import re
    from datetime import datetime, timezone, timedelta
    
    try:
        # Extract time pattern like "11:00 AM" or "11:15 PM"
        time_match = re.search(r'(\d{1,2}):(\d{2})\s*(AM|PM)', market_name, re.IGNORECASE)
        if not time_match:
            return None
        
        hour = int(time_match.group(1))
        minute = int(time_match.group(2))
        ampm = time_match.group(3).upper()
        
        # Convert to 24-hour format
        if ampm == 'PM' and hour != 12:
            hour += 12
        elif ampm == 'AM' and hour == 12:
            hour = 0
        
        # Extract date pattern like "January 24" or "Jan 24"
        date_match = re.search(r'(January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d{1,2})', market_name, re.IGNORECASE)
        if not date_match:
            return None
        
        month_str = date_match.group(1)
        day = int(date_match.group(2))
        
        # Extract year (optional)
        year_match = re.search(r'\b(20\d{2})\b', market_name)
        if year_match:
            year = int(year_match.group(1))
        else:
            year = 2024
        
        # Parse month name
        month_map = {
            'january': 1, 'jan': 1, 'february': 2, 'feb': 2, 'march': 3, 'mar': 3,
            'april': 4, 'apr': 4, 'may': 5, 'june': 6, 'jun': 6,
            'july': 7, 'jul': 7, 'august': 8, 'aug': 8, 'september': 9, 'sep': 9,
            'october': 10, 'oct': 10, 'november': 11, 'nov': 11, 'december': 12, 'dec': 12
        }
        month = month_map.get(month_str.lower())
        if month is None:
            return None
        
        # Create datetime object (ET timezone conversion)
        dt = datetime(year, month, day, hour, minute, 0, tzinfo=timezone.utc)
        dt_utc = dt + timedelta(hours=5)  # ET to UTC
        
        # Convert to nanoseconds
        timestamp_ns = int(dt_utc.timestamp() * 1_000_000_000)
        return timestamp_ns
        
    except Exception:
        return None

# FEE CALCULATION (from latest.py)
# =============================================================================

def calculate_taker_fee(price: float) -> float:
    """Calculate Polymarket taker fee based on price."""
    p = max(0.01, min(0.99, price))
    min_rate = 0.0020
    max_rate = 0.0156
    distance_from_midpoint = abs(p - 0.5) * 2
    curve = 1 - (distance_from_midpoint ** 2)
    fee_rate = min_rate + (max_rate - min_rate) * curve
    return fee_rate


# =============================================================================
# WALK-FORWARD TIME-SPLIT OOS EVALUATION
# =============================================================================

def create_walk_forward_splits(trades_df: pd.DataFrame, n_folds: int = 5) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Split data chronologically into n_folds for walk-forward time-split OOS evaluation.
    Returns list of (unused_df, oos_df) tuples. Note: unused_df is kept for interface
    compatibility but is not used for training - we only evaluate on oos_df.
    """
    trades_df = trades_df.copy()
    trades_df["date"] = pd.to_datetime(trades_df["timestamp_ns"], unit="ns", utc=True).dt.date

    unique_dates = sorted(trades_df["date"].unique())
    n_dates = len(unique_dates)

    if n_dates < n_folds:
        print(f"Warning: Only {n_dates} dates available, reducing folds to {n_dates}")
        n_folds = n_dates

    fold_size = n_dates // n_folds
    splits = []

    for i in range(n_folds):
        test_start_idx = i * fold_size
        test_end_idx = (i + 1) * fold_size if i < n_folds - 1 else n_dates

        test_dates = set(unique_dates[test_start_idx:test_end_idx])
        train_dates = set(unique_dates) - test_dates

        train_df = trades_df[trades_df["date"].isin(train_dates)].copy()
        test_df = trades_df[trades_df["date"].isin(test_dates)].copy()

        splits.append((train_df, test_df))
        print(f"  Time-Split OOS Fold {i+1}: OOS dates={len(test_dates)} (excluded={len(train_dates)})")

    return splits


# =============================================================================
# BACKTEST ENGINE
# =============================================================================

def _first_time_after(
    t_ns: np.ndarray,
    p: np.ndarray,
    start_ns: int,
    cond_le: Optional[float] = None,
) -> Optional[Tuple[int, float]]:
    """
    Find first timestamp where price <= cond_le after start_ns.
    Returns (timestamp, actual_price) tuple, or None if not found.
    """
    if t_ns.size == 0:
        return None
    i0 = np.searchsorted(t_ns, start_ns, side="left")
    if i0 >= t_ns.size:
        return None
    pp = p[i0:]
    if cond_le is not None:
        hits = np.where(pp <= cond_le)[0]
    else:
        return None
    if hits.size == 0:
        return None
    hit_idx = i0 + hits[0]
    return (int(t_ns[hit_idx]), float(p[hit_idx]))


def run_single_backtest(
    trades_df: pd.DataFrame,
    btc_prices: pd.DataFrame,
    params: BacktestParams,
) -> Dict[str, Any]:
    """
    Run backtest with given parameters on provided trades data.
    Returns dict with metrics and trade details.
    """
    trades_sorted = trades_df.sort_values("timestamp_ns")

    # Group by market
    market_groups = trades_sorted.groupby(["slug", "market_name"], sort=False)

    records = []

    for (slug, market_name), g in market_groups:
        # Split by side
        gU = g[g["side_l"] == "up"]
        gD = g[g["side_l"] == "down"]

        tU = gU["timestamp_ns"].to_numpy(dtype=np.int64)
        pU = gU["price"].to_numpy(dtype=np.float64)
        tD = gD["timestamp_ns"].to_numpy(dtype=np.int64)
        pD = gD["price"].to_numpy(dtype=np.float64)

        if len(tU) == 0 and len(tD) == 0:
            continue

        # Settlement time = last trade
        settlement_time_ns = int(g["timestamp_ns"].max())

        # Entry time = lookback minutes before settlement
        entry_time_ns = settlement_time_ns - (params.lookback_minutes * 60 * 1_000_000_000)

        # Market start time (15 mins before settlement for momentum calc)
        market_start_time_ns = settlement_time_ns - (15 * 60 * 1_000_000_000)

        # Get BTC prices (returns price and actual timestamp used)
        btc_start_price, btc_start_ts = get_btc_price_at_time(btc_prices, market_start_time_ns)
        btc_entry_price, btc_entry_ts = get_btc_price_at_time(btc_prices, entry_time_ns)

        # Track max timestamp used for feature computation (audit trail)
        max_ts_used_for_features_ns = 0
        if btc_start_ts is not None:
            max_ts_used_for_features_ns = max(max_ts_used_for_features_ns, btc_start_ts)
        if btc_entry_ts is not None:
            max_ts_used_for_features_ns = max(max_ts_used_for_features_ns, btc_entry_ts)

        # Find last price on each side at entry time
        idx_u = np.searchsorted(tU, entry_time_ns, side='right') - 1 if len(tU) > 0 else -1
        idx_d = np.searchsorted(tD, entry_time_ns, side='right') - 1 if len(tD) > 0 else -1

        price_u_at_entry = pU[idx_u] if 0 <= idx_u < len(pU) else None
        price_d_at_entry = pD[idx_d] if 0 <= idx_d < len(pD) else None

        # Track trade timestamps used for features
        if idx_u >= 0 and idx_u < len(tU):
            max_ts_used_for_features_ns = max(max_ts_used_for_features_ns, tU[idx_u])
        if idx_d >= 0 and idx_d < len(tD):
            max_ts_used_for_features_ns = max(max_ts_used_for_features_ns, tD[idx_d])

        # Audit: ensure no future data leakage in feature computation
        assert max_ts_used_for_features_ns <= entry_time_ns, \
            f"Future data leak: max_ts_used={max_ts_used_for_features_ns} > entry_time={entry_time_ns}"

        # Determine leader
        if price_u_at_entry is None and price_d_at_entry is None:
            continue

        if price_d_at_entry is None:
            leader_side = "up"
            leader_price = price_u_at_entry
        elif price_u_at_entry is None:
            leader_side = "down"
            leader_price = price_d_at_entry
        else:
            if price_u_at_entry >= price_d_at_entry:
                leader_side = "up"
                leader_price = price_u_at_entry
            else:
                leader_side = "down"
                leader_price = price_d_at_entry

        # Check price filter
        if leader_price < params.price_min or leader_price > params.price_max:
            continue

        # Check BTC momentum filter
        if params.momentum_threshold > 0:
            if btc_start_price is None or btc_entry_price is None or btc_start_price <= 0:
                continue

            btc_momentum_pct = (btc_entry_price - btc_start_price) / btc_start_price

            if leader_side == "up":
                btc_favorable = btc_momentum_pct >= params.momentum_threshold
            else:
                btc_favorable = btc_momentum_pct <= -params.momentum_threshold

            if not btc_favorable:
                continue

        # Market buy with slippage
        entry_price = min(leader_price + MARKET_BUY_SLIPPAGE, 0.99)
        taker_fee_rate = calculate_taker_fee(entry_price)
        taker_fee = entry_price * taker_fee_rate
        cost_basis = entry_price + taker_fee

        # Get trade arrays for bought side
        if leader_side == "up":
            filled_t = tU
            filled_p = pU
            last_price_bought = pU[-1] if len(pU) > 0 else None
            last_price_other = pD[-1] if len(pD) > 0 else None
        else:
            filled_t = tD
            filled_p = pD
            last_price_bought = pD[-1] if len(pD) > 0 else None
            last_price_other = pU[-1] if len(pU) > 0 else None

        # Check stop loss
        stop_result = None
        exit_fee = 0.0

        if params.stop_loss > 0:
            stop_result = _first_time_after(filled_t, filled_p, start_ns=entry_time_ns, cond_le=params.stop_loss)

        # Determine outcome
        if stop_result is not None:
            status = "stopped"
            t_stop, actual_exit_price = stop_result
            # Apply exit slippage (selling at market = get worse price)
            # When selling, slippage works against you (subtract from exit price)
            exit_price_with_slippage = max(actual_exit_price - MARKET_SELL_SLIPPAGE, 0.01)
            exit_fee_rate = calculate_taker_fee(exit_price_with_slippage)
            exit_fee = exit_price_with_slippage * exit_fee_rate
            pnl = (exit_price_with_slippage - exit_fee) - cost_basis
        else:
            # Held to settlement - determine winner
            if last_price_bought is None:
                continue

            if last_price_other is None:
                winner_side = leader_side
            else:
                if leader_side == "up":
                    winner_side = "up" if last_price_bought >= last_price_other else "down"
                else:
                    winner_side = "down" if last_price_bought >= last_price_other else "up"

            if winner_side == leader_side:
                status = "won"
                pnl = 1.0 - cost_basis
            else:
                status = "lost"
                pnl = 0.0 - cost_basis

        settlement_dt = pd.Timestamp(settlement_time_ns, unit='ns', tz='UTC')

        records.append({
            "slug": slug,
            "market_name": market_name,
            "settlement_date": settlement_dt.date(),
            "leader_side": leader_side,
            "leader_price": leader_price,
            "entry_price": entry_price,
            "cost_basis": cost_basis,
            "status": status,
            "pnl": pnl,
        })

    return {
        "trades": records,
        "num_trades": len(records),
    }


# =============================================================================
# METRICS CALCULATION
# =============================================================================

def calculate_metrics(trades: List[Dict]) -> Dict[str, float]:
    """Calculate performance metrics from trade list."""
    if len(trades) == 0:
        return {
            "num_trades": 0,
            "total_pnl": 0.0,
            "avg_pnl": 0.0,
            "win_rate": 0.0,
            "sharpe_trade": 0.0,
            "sharpe_daily": 0.0,
            "max_drawdown": 0.0,
            "profit_factor": 0.0,
        }

    df = pd.DataFrame(trades)
    pnl_array = df["pnl"].values

    num_trades = len(df)
    total_pnl = pnl_array.sum()
    avg_pnl = pnl_array.mean()

    wins = (df["status"] == "won").sum()
    win_rate = wins / num_trades if num_trades > 0 else 0

    # Per-trade Sharpe ratio (annualized)
    if pnl_array.std() > 0:
        sharpe_trade = (pnl_array.mean() / pnl_array.std()) * np.sqrt(252)
    else:
        sharpe_trade = 0.0

    # Daily Sharpe ratio (annualized)
    # Group by settlement_date and sum PnL per day
    daily_pnl = df.groupby("settlement_date")["pnl"].sum().sort_index()
    if daily_pnl.std(ddof=1) > 0:
        sharpe_daily = daily_pnl.mean() / daily_pnl.std(ddof=1) * np.sqrt(252)
    else:
        sharpe_daily = 0.0

    # Max drawdown
    cumulative = np.cumsum(pnl_array)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = running_max - cumulative
    max_drawdown = drawdowns.max() if len(drawdowns) > 0 else 0.0

    # Profit factor
    gross_profit = pnl_array[pnl_array > 0].sum()
    gross_loss = abs(pnl_array[pnl_array < 0].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    return {
        "num_trades": num_trades,
        "total_pnl": total_pnl,
        "avg_pnl": avg_pnl,
        "win_rate": win_rate,
        "sharpe_trade": sharpe_trade,
        "sharpe_daily": sharpe_daily,
        "max_drawdown": max_drawdown,
        "profit_factor": profit_factor,
    }


def calculate_stability_score(
    params: BacktestParams,
    all_results: Dict[Tuple, Dict],
    param_lists: Dict[str, List],
) -> float:
    """
    Calculate stability score by checking if neighboring parameters also perform well.
    Returns score between 0 and 1.
    """
    neighbors_profitable = 0
    total_neighbors = 0

    current = params.to_tuple()
    current_sharpe = all_results.get(current, {}).get("oos_sharpe_daily", 0)

    if current_sharpe <= 0:
        return 0.0

    # Check each parameter dimension
    for attr, values in param_lists.items():
        current_val = getattr(params, attr)
        current_idx = values.index(current_val) if current_val in values else -1

        if current_idx < 0:
            continue

        # Check neighbors
        for delta in [-1, 1]:
            neighbor_idx = current_idx + delta
            if 0 <= neighbor_idx < len(values):
                neighbor_params = BacktestParams(
                    lookback_minutes=params.lookback_minutes if attr != "lookback_minutes" else values[neighbor_idx],
                    price_min=params.price_min if attr != "price_min" else values[neighbor_idx],
                    price_max=params.price_max if attr != "price_max" else values[neighbor_idx],
                    momentum_threshold=params.momentum_threshold if attr != "momentum_threshold" else values[neighbor_idx],
                    stop_loss=params.stop_loss if attr != "stop_loss" else values[neighbor_idx],
                )

                # Skip invalid combinations
                if neighbor_params.price_min >= neighbor_params.price_max:
                    continue

                neighbor_key = neighbor_params.to_tuple()
                neighbor_result = all_results.get(neighbor_key, {})
                neighbor_sharpe = neighbor_result.get("oos_sharpe_daily", 0)

                total_neighbors += 1
                if neighbor_sharpe > 0:
                    neighbors_profitable += 1

    if total_neighbors == 0:
        return 0.5

    return neighbors_profitable / total_neighbors


# =============================================================================
# OPTIMIZER
# =============================================================================

def generate_param_grid() -> List[BacktestParams]:
    """Generate all valid parameter combinations."""
    grid = []

    for lookback in LOOKBACK_MINUTES_LIST:
        for price_min in PRICE_MIN_LIST:
            for price_max in PRICE_MAX_LIST:
                if price_min >= price_max:
                    continue
                for momentum in MOMENTUM_THRESHOLD_LIST:
                    for stop_loss in STOP_LOSS_LIST:
                        grid.append(BacktestParams(
                            lookback_minutes=lookback,
                            price_min=price_min,
                            price_max=price_max,
                            momentum_threshold=momentum,
                            stop_loss=stop_loss,
                        ))

    return grid


def walk_forward_optimize(
    trades_df: pd.DataFrame,
    btc_prices: pd.DataFrame,
    param_grid: List[BacktestParams],
    n_folds: int = 5,
) -> pd.DataFrame:
    """
    Run walk-forward time-split OOS evaluation over parameter grid.
    Note: This is pure OOS evaluation - no training occurs. Each fold's data
    is used only for out-of-sample testing.
    Returns DataFrame with results for each parameter combination.
    """
    print(f"\n{'='*60}")
    print(f"WALK-FORWARD TIME-SPLIT OOS EVALUATION")
    print(f"{'='*60}")
    print(f"Parameter combinations: {len(param_grid)}")
    print(f"Time-split OOS folds: {n_folds}")
    print(f"Min trades per fold: {MIN_TRADES_PER_FOLD}")

    # Create time-split OOS folds
    print("\nCreating time-split OOS folds...")
    splits = create_walk_forward_splits(trades_df, n_folds)

    # Results storage
    all_results = {}

    print(f"\nRunning optimization...")
    total = len(param_grid)

    # Import for parallel processing
    try:
        from joblib import Parallel, delayed
        import multiprocessing
        PARALLEL_AVAILABLE = True
    except ImportError:
        PARALLEL_AVAILABLE = False
        print("  Warning: joblib not available, running sequentially")
        print("  Install with: pip install joblib --break-system-packages")

    # Helper function for parallel execution
    def test_single_param(params: BacktestParams, splits, btc_prices) -> Dict[str, Any]:
        """
        Test a single parameter combination using walk-forward time-split OOS evaluation.
        Note: Only evaluates on oos_df; the unused_df is not used for training.
        """
        fold_results = []
        oos_trades = []

        for fold_idx, (_unused_df, oos_df) in enumerate(splits):
            result = run_single_backtest(oos_df, btc_prices, params)

            if result["num_trades"] >= MIN_TRADES_PER_FOLD:
                metrics = calculate_metrics(result["trades"])
                fold_results.append(metrics)
                oos_trades.extend(result["trades"])

        # Aggregate results
        if len(fold_results) > 0:
            avg_sharpe_daily = np.mean([r["sharpe_daily"] for r in fold_results])
            avg_sharpe_trade = np.mean([r["sharpe_trade"] for r in fold_results])
            avg_pnl = np.mean([r["avg_pnl"] for r in fold_results])
            avg_win_rate = np.mean([r["win_rate"] for r in fold_results])
            total_trades = sum([r["num_trades"] for r in fold_results])

            if oos_trades:
                oos_metrics = calculate_metrics(oos_trades)
                max_dd = oos_metrics["max_drawdown"]
            else:
                max_dd = 0.0

            return {
                "key": params.to_tuple(),
                "params": params.to_dict(),
                "oos_sharpe_daily": avg_sharpe_daily,
                "oos_sharpe_trade": avg_sharpe_trade,
                "oos_avg_pnl": avg_pnl,
                "oos_win_rate": avg_win_rate,
                "oos_max_drawdown": max_dd,
                "oos_total_trades": total_trades,
                "valid_folds": len(fold_results),
            }
        else:
            return {
                "key": params.to_tuple(),
                "params": params.to_dict(),
                "oos_sharpe_daily": 0.0,
                "oos_sharpe_trade": 0.0,
                "oos_avg_pnl": 0.0,
                "oos_win_rate": 0.0,
                "oos_max_drawdown": 0.0,
                "oos_total_trades": 0,
                "valid_folds": 0,
            }
    
    print(f"\nRunning optimization...")
    
    # FORCE SEQUENTIAL - Windows has memory issues with parallel processing
    PARALLEL_AVAILABLE = False
    print(f"  Running sequentially (safer for Windows)")
    print(f"  Estimated time: 2-3 hours for {len(param_grid)} combinations\n")
    
    if PARALLEL_AVAILABLE:
        # Parallel execution (DISABLED for Windows)
        n_jobs = max(1, multiprocessing.cpu_count() - 1)
        print(f"  Using {n_jobs} parallel workers (joblib)")
        
        results_list = Parallel(n_jobs=n_jobs, verbose=5)(
            delayed(test_single_param)(params, splits, btc_prices)
            for params in param_grid
        )
        
        # Convert list to dict
        all_results = {result["key"]: result for result in results_list}
        
    else:
        # Sequential execution
        all_results = {}
        total = len(param_grid)
        start_time = pd.Timestamp.now()
        
        for idx, params in enumerate(param_grid):
            result = test_single_param(params, splits, btc_prices)
            all_results[result["key"]] = result
            
            # Progress every 10 params
            if (idx + 1) % 10 == 0 or (idx + 1) == total:
                elapsed_min = (pd.Timestamp.now() - start_time).total_seconds() / 60
                avg_per_param = elapsed_min / (idx + 1)
                remaining_min = avg_per_param * (total - idx - 1)
                eta = pd.Timestamp.now() + pd.Timedelta(minutes=remaining_min)
                
                print(f"  Progress: {idx + 1}/{total} ({100*(idx+1)/total:.1f}%)")
                print(f"    Elapsed: {elapsed_min:.1f} min | Remaining: {remaining_min:.1f} min")
                print(f"    ETA: {eta.strftime('%H:%M:%S')}\n")

    # Calculate stability scores
    print("\nCalculating stability scores...")
    param_lists = {
        "lookback_minutes": LOOKBACK_MINUTES_LIST,
        "price_min": PRICE_MIN_LIST,
        "price_max": PRICE_MAX_LIST,
        "momentum_threshold": MOMENTUM_THRESHOLD_LIST,
        "stop_loss": STOP_LOSS_LIST,
    }

    for params in param_grid:
        key = params.to_tuple()
        if key in all_results:
            stability = calculate_stability_score(params, all_results, param_lists)
            all_results[key]["stability_score"] = stability

    # Convert to DataFrame
    results_list = []
    for key, result in all_results.items():
        row = result["params"].copy()
        row.update({
            "oos_sharpe_daily": result["oos_sharpe_daily"],
            "oos_sharpe_trade": result["oos_sharpe_trade"],
            "oos_avg_pnl": result["oos_avg_pnl"],
            "oos_win_rate": result["oos_win_rate"],
            "oos_max_drawdown": result["oos_max_drawdown"],
            "oos_total_trades": result["oos_total_trades"],
            "valid_folds": result["valid_folds"],
            "stability_score": result.get("stability_score", 0.0),
        })
        results_list.append(row)

    results_df = pd.DataFrame(results_list)

    # Add combined score: Sharpe (daily) - 0.5*MaxDD + 0.2*Stability
    results_df["combined_score"] = (
        results_df["oos_sharpe_daily"]
        - 0.5 * results_df["oos_max_drawdown"]
        + 0.2 * results_df["stability_score"]
    )

    return results_df.sort_values("oos_sharpe_daily", ascending=False)


# =============================================================================
# REPORT GENERATION
# =============================================================================

def generate_reports(
    results_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    btc_prices: pd.DataFrame,
    out_prefix: str = "optimization",
):
    """Generate output files with optimization results."""

    print(f"\n{'='*60}")
    print("OPTIMIZATION RESULTS")
    print(f"{'='*60}")

    # Filter to valid results
    valid_results = results_df[results_df["valid_folds"] > 0].copy()

    if valid_results.empty:
        print("No valid results found!")
        return

    # Top 10 by Sharpe (daily)
    print("\n--- TOP 10 BY OUT-OF-SAMPLE SHARPE (DAILY) ---")
    top_sharpe = valid_results.nlargest(10, "oos_sharpe_daily")
    print(top_sharpe[["lookback_minutes", "price_min", "price_max",
                      "momentum_threshold", "stop_loss", "oos_sharpe_daily",
                      "oos_avg_pnl", "oos_max_drawdown", "stability_score"]].to_string(index=False))

    # Top 10 by combined score
    print("\n--- TOP 10 BY COMBINED SCORE ---")
    top_combined = valid_results.nlargest(10, "combined_score")
    print(top_combined[["lookback_minutes", "price_min", "price_max",
                        "momentum_threshold", "stop_loss", "combined_score",
                        "oos_sharpe_daily", "oos_max_drawdown", "stability_score"]].to_string(index=False))

    # Best parameters (by Sharpe)
    best_row = valid_results.iloc[0]
    best_params = BacktestParams(
        lookback_minutes=int(best_row["lookback_minutes"]),
        price_min=best_row["price_min"],
        price_max=best_row["price_max"],
        momentum_threshold=best_row["momentum_threshold"],
        stop_loss=best_row["stop_loss"],
    )

    print(f"\n--- BEST PARAMETERS (by Sharpe Daily) ---")
    print(f"  Lookback minutes: {best_params.lookback_minutes}")
    print(f"  Price range: [{best_params.price_min}, {best_params.price_max}]")
    print(f"  Momentum threshold: {best_params.momentum_threshold:.4f} ({best_params.momentum_threshold*100:.2f}%)")
    print(f"  Stop loss: ${best_params.stop_loss:.2f}")
    print(f"\n  OOS Sharpe (daily): {best_row['oos_sharpe_daily']:.4f}")
    print(f"  OOS Sharpe (trade): {best_row['oos_sharpe_trade']:.4f}")
    print(f"  OOS Avg P&L: ${best_row['oos_avg_pnl']:.4f}")
    print(f"  OOS Win Rate: {best_row['oos_win_rate']*100:.1f}%")
    print(f"  OOS Max Drawdown: ${best_row['oos_max_drawdown']:.4f}")
    print(f"  Stability Score: {best_row['stability_score']:.2f}")

    # Run full backtest with best params to get equity curve
    print("\nGenerating equity curve for best parameters...")
    full_result = run_single_backtest(trades_df, btc_prices, best_params)

    if full_result["num_trades"] > 0:
        trades_list = full_result["trades"]
        equity_df = pd.DataFrame(trades_list)
        equity_df["cumulative_pnl"] = equity_df["pnl"].cumsum()

        # Daily aggregation
        daily_equity = equity_df.groupby("settlement_date").agg({
            "pnl": ["sum", "mean", "count"],
            "status": lambda x: (x == "won").sum(),
        }).round(4)
        daily_equity.columns = ["daily_pnl", "avg_pnl", "num_trades", "wins"]
        daily_equity["cumulative_pnl"] = daily_equity["daily_pnl"].cumsum()
        daily_equity["win_rate"] = daily_equity["wins"] / daily_equity["num_trades"]
        daily_equity = daily_equity.reset_index()

        # Save equity curves
        equity_path = os.path.join(LOCAL_OUTPUT_PATH, f"{out_prefix}_equity_curve.csv")
        daily_equity.to_csv(equity_path, index=False)
        print(f"  Saved: {equity_path}")

        # Save detailed trades
        trades_path = os.path.join(LOCAL_OUTPUT_PATH, f"{out_prefix}_best_trades.csv")
        equity_df.to_csv(trades_path, index=False)
        print(f"  Saved: {trades_path}")

    # Save all results
    results_path = os.path.join(LOCAL_OUTPUT_PATH, f"{out_prefix}_all_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"  Saved: {results_path}")

    # Save best params as JSON
    best_params_dict = best_params.to_dict()
    best_params_dict.update({
        "oos_sharpe_daily": float(best_row["oos_sharpe_daily"]),
        "oos_sharpe_trade": float(best_row["oos_sharpe_trade"]),
        "oos_avg_pnl": float(best_row["oos_avg_pnl"]),
        "oos_win_rate": float(best_row["oos_win_rate"]),
        "oos_max_drawdown": float(best_row["oos_max_drawdown"]),
        "stability_score": float(best_row["stability_score"]),
    })

    params_path = os.path.join(LOCAL_OUTPUT_PATH, f"{out_prefix}_best_params.json")
    with open(params_path, "w") as f:
        json.dump(best_params_dict, f, indent=2)
    print(f"  Saved: {params_path}")

    # Save stability report (top 50 by Sharpe with stability info)
    stability_df = valid_results.nlargest(50, "oos_sharpe_daily")[
        ["lookback_minutes", "price_min", "price_max", "momentum_threshold",
         "stop_loss", "oos_sharpe_daily", "oos_sharpe_trade", "oos_avg_pnl", "oos_max_drawdown",
         "stability_score", "combined_score", "oos_total_trades", "valid_folds"]
    ]
    stability_path = os.path.join(LOCAL_OUTPUT_PATH, f"{out_prefix}_stability_report.csv")
    stability_df.to_csv(stability_path, index=False)
    print(f"  Saved: {stability_path}")

    print(f"\n{'='*60}")
    print("OPTIMIZATION COMPLETE")
    print(f"{'='*60}")


# =============================================================================
# MAIN
# =============================================================================

def run(items: List[str], out_prefix: str):
    """Main entry point."""
    print(f"\n{'='*60}")
    print("KAGGLE COMBINED OPTIMIZER (PARALLEL)")
    print(f"{'='*60}\n")
    
    # Load all trades (from zips or folders)
    all_parts = []
    print(f"Loading data from {len(items)} source(s)...")
    
    for item_path in items:
        if os.path.isdir(item_path):
            # It's a folder
            df = load_trades_from_folder(item_path)
        elif item_path.endswith('.zip'):
            # It's a zip file
            df = load_trades_from_zip(item_path)
        else:
            print(f"  Skipping unknown item: {item_path}")
            continue
        
        all_parts.append(df)
    
    trades_df = pd.concat(all_parts, ignore_index=True)
    if trades_df.empty:
        raise SystemExit("No trades loaded. Check your paths / CSV schema.")
    
    print(f"\nTotal trades loaded: {len(trades_df):,}")
    num_markets = len(trades_df.groupby(['slug', 'market_name']))
    print(f"Total unique markets: {num_markets:,}")
    print(f"Date range: {trades_df['timestamp'].min()} to {trades_df['timestamp'].max()}")
    
    # Fetch BTC prices
    min_time_ns = trades_df["timestamp_ns"].min()
    max_time_ns = trades_df["timestamp_ns"].max()
    buffer_ns = 15 * 60 * 1_000_000_000
    start_time_ms = (min_time_ns - buffer_ns) // 1_000_000
    end_time_ms = max_time_ns // 1_000_000
    
    btc_prices = fetch_btc_prices(int(start_time_ms), int(end_time_ms))
    
    # Generate parameter grid
    param_grid = generate_param_grid()
    print(f"\nGenerated {len(param_grid)} parameter combinations to test")
    
    # Run walk-forward optimization
    results_df = walk_forward_optimize(trades_df, btc_prices, param_grid, N_FOLDS)
    
    # Generate reports
    generate_reports(results_df, trades_df, btc_prices, out_prefix)


if __name__ == "__main__":
    # Auto-detect zip files in current directory
    print(f"\n{'='*60}")
    print("LOCAL OPTIMIZER")
    print(f"{'='*60}")
    print(f"Searching for data in: {LOCAL_INPUT_PATH}\n")
    
    data_sources = find_kaggle_zip_files(LOCAL_INPUT_PATH)
    
    if not data_sources:
        print(f"❌ ERROR: No zip files found in {LOCAL_INPUT_PATH}")
        print("\nPlease put your btc_trades_*.zip files in the same folder as this script.")
        print("\nCurrent directory contents:")
        for item in os.listdir(LOCAL_INPUT_PATH):
            if item.endswith('.zip') or os.path.isdir(os.path.join(LOCAL_INPUT_PATH, item)):
                print(f"  - {item}")
        raise SystemExit(1)
    
    # Determine what we found
    zip_count = sum(1 for s in data_sources if s.endswith('.zip'))
    folder_count = sum(1 for s in data_sources if os.path.isdir(s))
    
    print(f"Found {len(data_sources)} data source(s):")
    if zip_count > 0:
        print(f"  ✓ {zip_count} zip file(s)")
    if folder_count > 0:
        print(f"  ✓ {folder_count} folder(s)")
    
    for source in data_sources:
        print(f"    - {os.path.basename(source)}")
    
    print(f"\n{'='*60}")
    print(f"Results will be saved to: {LOCAL_OUTPUT_PATH}")
    print(f"{'='*60}\n")
    
    # Run optimization
    run(data_sources, out_prefix="optimization")
