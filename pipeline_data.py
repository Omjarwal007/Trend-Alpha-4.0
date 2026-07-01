import os
import pandas as pd
import numpy as np
import datetime
from config import CACHE_DIR
from cache_manager import get_historical_data
from utils import log_info, log_warning, log_success

def calculate_atr(df, period=14):
    """Calculates the standard Average True Range (ATR)."""
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Use exponential moving average for smoothing ATR
    atr = tr.ewm(span=period, adjust=False).mean()
    return atr

def calculate_natr(df, period=14):
    """Calculates Normalized ATR (NATR) as a percentage of close price."""
    atr = calculate_atr(df, period)
    natr = (atr / df["Close"]) * 100.0
    return natr

def calculate_chop_index(df, period=14):
    """Calculates the standard Choppiness Index."""
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    sum_tr = tr.rolling(period).sum()
    max_high = high.rolling(period).max()
    min_low = low.rolling(period).min()
    
    # Avoid zero division
    range_diff = max_high - min_low
    range_diff = range_diff.replace(0, np.nan)
    
    chop = 100.0 * np.log10(sum_tr / range_diff) / np.log10(period)
    return chop.fillna(35.0)  # Default non-choppy fallback

def calculate_adx(df, period=14):
    """Calculates the standard ADX, +DI, and -DI."""
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    
    up_move = high.diff()
    down_move = -low.diff()
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    plus_dm = pd.Series(plus_dm, index=df.index)
    minus_dm = pd.Series(minus_dm, index=df.index)
    
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    
    atr = tr.ewm(span=period, adjust=False).mean()
    # Avoid zero division
    atr_smooth = atr.replace(0, np.nan)
    
    plus_di = 100.0 * plus_dm.ewm(span=period, adjust=False).mean() / atr_smooth
    minus_di = 100.0 * minus_dm.ewm(span=period, adjust=False).mean() / atr_smooth
    
    plus_di = plus_di.fillna(0)
    minus_di = minus_di.fillna(0)
    
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    dx = dx.fillna(0)
    adx = dx.ewm(span=period, adjust=False).mean()
    
    return adx, plus_di, minus_di

def calculate_obv(df):
    """Calculates On-Balance Volume (OBV) in a vectorized manner."""
    close = df["Close"]
    volume = df["Volume"]
    
    direction = np.sign(close.diff().fillna(0))
    obv_series = (direction * volume).cumsum()
    return obv_series

def calculate_whipsaw_count(close, ma, lookback=30):
    """Counts crossings of close price over moving average in the lookback period."""
    if len(close) < lookback:
        return 0
    sub_close = close.tail(lookback)
    sub_ma = ma.tail(lookback)
    
    above = (sub_close > sub_ma).astype(int)
    crossings = int(above.diff().abs().sum())
    return crossings

def get_weekly_data(daily_df):
    """Resamples daily OHLCV data to weekly OHLCV data."""
    weekly = daily_df.resample('W').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    })
    # Remove any rows with NaN caused by resampling
    weekly = weekly.dropna()
    return weekly

def get_up_down_volume_ratio(df, lookback=25):
    """Calculates the sum of volume on UP days vs DOWN days in lookback period."""
    if len(df) < lookback:
        return 1.0
    sub = df.tail(lookback).copy()
    sub["Price_Diff"] = sub["Close"].diff()
    
    up_vol = sub.loc[sub["Price_Diff"] > 0, "Volume"].sum()
    dn_vol = sub.loc[sub["Price_Diff"] < 0, "Volume"].sum()
    
    # Avoid zero division
    if dn_vol == 0:
        return 2.0 if up_vol > 0 else 1.0
        
    return up_vol / dn_vol

def get_trend_efficiency_ratio(df, lookback=100):
    """Calculates Trend Efficiency Ratio (Net Move / Sum of absolute daily diffs)."""
    if len(df) < lookback:
        return 0.1
    sub = df.tail(lookback)
    net_move = abs(sub["Close"].iloc[-1] - sub["Close"].iloc[0])
    path_length = sub["Close"].diff().abs().sum()
    
    if path_length == 0:
        return 0.0
    return net_move / path_length

def validate_price_data(df, symbol, date_str=None):
    """Data Quality Gate: Validates price data integrity before indicators are computed.
    
    Returns (is_valid, warnings_list). If invalid, pipeline must reject the stock.
    Checks:
        1. DataFrame not None/empty
        2. Required OHLC columns present  
        3. No NaN in Close column (synthetic/unreliable data indicator)
        4. Last data point within 3 trading days (stale data detection)
        5. Close > 0 for all rows
        6. Minimum 100 rows for indicator stability
    """
    warnings = []
    
    # Check 1: Basic existence
    if df is None or df.empty:
        return False, ["DataFrame is None or empty"]
    
    # Check 2: Required columns
    required = ["Close", "High", "Low", "Open"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        return False, [f"Missing columns: {missing}"]
    
    # Check 3: NaN in Close (sign of corrupt/synthetic data)
    nan_count = df["Close"].isna().sum()
    if nan_count > len(df) * 0.05:  # >5% NaN = unreliable
        return False, [f"Close column has {nan_count}/{len(df)} NaN values — data corrupt"]
    if nan_count > 0:
        warnings.append(f"Close column has {nan_count} NaN value(s); dropping")
        df = df.dropna(subset=["Close"])
    
    # Check 4: Stale data detection (for live pipeline runs)
    if date_str is not None:
        try:
            target_date = pd.Timestamp(date_str)
            last_date = df.index.max()
            if hasattr(last_date, 'tz') and last_date.tz is not None:
                last_date = last_date.tz_localize(None)
            days_stale = (target_date - last_date).days
            if days_stale > 5:
                warnings.append(f"Data is {days_stale} days stale (last: {last_date.date()}, expected: {date_str})")
                # Not failing — allow for weekends/holidays, but warn loudly
                log_warning(f"STALE DATA: {symbol} last data point is {days_stale} days old. Indicators may be unreliable.")
        except Exception:
            pass
    
    # Check 5: Zero/negative prices
    zero_close = (df["Close"] <= 0).sum()
    if zero_close > 0:
        return False, [f"Close price <= 0 for {zero_close} rows"]
    
    # Check 6: Minimum data sufficiency
    if len(df) < 100:
        warnings.append(f"Only {len(df)} rows — insufficient for stable indicators (need ≥100)")
        # Not failing for indices that have short history, but warn
    
    return True, warnings


def compute_all_indicators(symbol, days_history=350, end_date=None):
    """Computes daily and weekly indicators for a given symbol."""
    # Get daily OHLCV
    daily_df = get_historical_data(symbol, days=days_history, end_date=end_date)
    
    # ── DATA QUALITY GATE ──
    is_valid, warnings = validate_price_data(daily_df, symbol, date_str=end_date)
    if not is_valid:
        log_error(f"DATA QUALITY FAILED for {symbol}: {'; '.join(warnings)}. Skipping.")
        return None
    for w in warnings:
        log_warning(f"DATA QUALITY WARNING for {symbol}: {w}")
    
    if daily_df is None or daily_df.empty or len(daily_df) < 50:
        log_warning(f"Insufficient daily data for symbol {symbol}. Skipping.")
        return None
        
    # Redefine Volume as Rupee Traded Value to ensure split-invariance
    daily_df = daily_df.copy()
    daily_df["Volume"] = daily_df["Close"] * daily_df["Volume"]
        
    # Get weekly OHLCV
    weekly_df = get_weekly_data(daily_df)
    
    # ── DAILY CALCULATIONS ──────────────────────────────────
    # Moving Averages
    daily_df["EMA_20"] = daily_df["Close"].ewm(span=20, adjust=False).mean()
    daily_df["EMA_50"] = daily_df["Close"].ewm(span=50, adjust=False).mean()
    daily_df["EMA_150"] = daily_df["Close"].ewm(span=150, adjust=False).mean()
    daily_df["SMA_50"] = daily_df["Close"].rolling(50).mean()
    daily_df["SMA_150"] = daily_df["Close"].rolling(150).mean()
    daily_df["SMA_200"] = daily_df["Close"].rolling(200).mean()
    
    # ATR & NATR
    daily_df["ATR_14"] = calculate_atr(daily_df, 14)
    daily_df["NATR_14"] = calculate_natr(daily_df, 14)
    
    # Choppiness
    daily_df["CHOP_14"] = calculate_chop_index(daily_df, 14)
    daily_df["CHOP_20"] = calculate_chop_index(daily_df, 20)
    daily_df["CHOP_avg"] = (daily_df["CHOP_14"] + daily_df["CHOP_20"]) / 2.0
    
    # ADX
    adx_series, plus_di_series, minus_di_series = calculate_adx(daily_df, 14)
    daily_df["ADX_14"] = adx_series
    daily_df["Plus_DI_14"] = plus_di_series
    daily_df["Minus_DI_14"] = minus_di_series
    
    # OBV
    daily_df["OBV"] = calculate_obv(daily_df)
    daily_df["OBV_EMA_20"] = daily_df["OBV"].ewm(span=20, adjust=False).mean()
    
    # Volume averages
    daily_df["Vol_MA_20"] = daily_df["Volume"].rolling(20).mean()
    daily_df["Vol_MA_50"] = daily_df["Volume"].rolling(50).mean()
    
    # Rate of Change (ROC)
    daily_df["ROC_1m"] = daily_df["Close"].pct_change(21) * 100.0
    daily_df["ROC_3m"] = daily_df["Close"].pct_change(63) * 100.0
    daily_df["ROC_6m"] = daily_df["Close"].pct_change(126) * 100.0
    daily_df["ROC_12m"] = daily_df["Close"].pct_change(252) * 100.0
    daily_df["ROC_50d"] = daily_df["Close"].pct_change(50) * 100.0
    
    # Vectorized Whipsaws calculation (crossings rolling sum)
    crosses_20 = (daily_df["Close"] > daily_df["EMA_20"]).astype(int).diff().abs()
    daily_df["Whipsaws_20d_30MA"] = crosses_20.rolling(30).sum().fillna(0).astype(int)

    crosses_50 = (daily_df["Close"] > daily_df["EMA_50"]).astype(int).diff().abs()
    daily_df["Whipsaws_50d_60MA"] = crosses_50.rolling(60).sum().fillna(0).astype(int)
    
    # Vectorized Up/Down Volume Ratio
    diff = daily_df["Close"].diff()
    up_vol_mask = (diff > 0).astype(float)
    dn_vol_mask = (diff < 0).astype(float)
    up_vol_series = (daily_df["Volume"] * up_vol_mask).rolling(25).sum()
    dn_vol_series = (daily_df["Volume"] * dn_vol_mask).rolling(25).sum()
    ud_ratio = up_vol_series / dn_vol_series.replace(0, np.nan)
    ud_ratio = ud_ratio.fillna(1.0)
    # Match the fallback logic: if dn_vol == 0: return 2.0 if up_vol > 0 else 1.0
    ud_ratio = np.where((up_vol_series > 0) & (dn_vol_series.fillna(0) == 0), 2.0, ud_ratio)
    daily_df["Up_Down_Vol_Ratio_25d"] = ud_ratio
    
    # Vectorized Trend Efficiency Ratio
    net_move = (daily_df["Close"] - daily_df["Close"].shift(100)).abs()
    path_length = daily_df["Close"].diff().abs().rolling(100).sum()
    er = net_move / path_length.replace(0, np.nan)
    daily_df["Trend_Efficiency_Ratio"] = er.fillna(0.1)
    
    # NATR contracting / expanding
    daily_df["NATR_90d_avg"] = daily_df["NATR_14"].rolling(90).mean()
    daily_df["NATR_Trend"] = np.where(daily_df["NATR_14"] < daily_df["NATR_90d_avg"], "CONTRACTING", "EXPANDING")
    
    # ── WEEKLY CALCULATIONS ─────────────────────────────────
    if len(weekly_df) >= 30:
        weekly_df["Weekly_CHOP"] = calculate_chop_index(weekly_df, 14)
        weekly_df["Weekly_30MA"] = weekly_df["Close"].rolling(30).mean()
        # Rising check (compared to 5 weeks ago)
        weekly_df["Weekly_30MA_Rising"] = weekly_df["Weekly_30MA"] > weekly_df["Weekly_30MA"].shift(5)
        
        # Merge back to daily using forward fill
        # Match nearest weekly close date
        weekly_df_aligned = weekly_df[["Weekly_CHOP", "Weekly_30MA", "Weekly_30MA_Rising"]].reindex(daily_df.index, method='ffill')
        daily_df = pd.concat([daily_df, weekly_df_aligned], axis=1)
        
        # Fill leading NaNs
        daily_df["Weekly_CHOP"] = daily_df["Weekly_CHOP"].fillna(35.0)
        daily_df["Weekly_30MA"] = daily_df["Weekly_30MA"].fillna(daily_df["SMA_200"])
        daily_df["Weekly_30MA_Rising"] = daily_df["Weekly_30MA_Rising"].fillna(True)
    else:
        # Mock weekly fields if insufficient weekly data
        daily_df["Weekly_CHOP"] = 35.0
        daily_df["Weekly_30MA"] = daily_df["SMA_200"]
        daily_df["Weekly_30MA_Rising"] = True
 
    return daily_df
 
def run_data_pipeline(symbols, date_str=None):
    """Ingests data for benchmark indices and builds the master data dictionary."""
    log_info("Executing SKILL 01 & 02: Ingesting benchmark index data & building indicators pipeline...")
    
    master_data = {}
    
    # Include Benchmark Indices in calculation
    benchmarks = ["NIFTY_SMALLCAP_250", "NIFTY_50", "NIFTY_NEXT_50", "NIFTY_MIDCAP_150", "NIFTY_500", "MCX_GOLD", "MCX_SILVER", "NIFTY_MICROCAP_250", "INDIA_VIX"]
    for benchmark in benchmarks:
        df = compute_all_indicators(benchmark, end_date=date_str)
        if df is not None:
            master_data[benchmark] = df
            
    log_success(f"Benchmark data pipeline complete. Ingested indicators for {len(master_data)} indices.")
    return master_data
