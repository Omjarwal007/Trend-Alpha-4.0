import os
import pandas as pd
import numpy as np
import datetime
from utils import log_info, log_warning, log_success, log_error

# Local cache paths
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# Maximum cache age in hours before forced refresh
MAX_CACHE_AGE_HOURS = 6  # Stocks can be cached up to 6 hours
MAX_INDEX_CACHE_AGE_HOURS = 2  # Indices refresh every 2 hours

def is_cache_stale(symbol, max_age_hours=None):
    """Check if cached data for a symbol is stale.
    Returns True if cache is missing or older than max_age_hours."""
    if max_age_hours is None:
        max_age_hours = MAX_CACHE_AGE_HOURS
        if any(idx in symbol.upper() for idx in ["NIFTY", "SENSEX", "MCX_", "^"]):
            max_age_hours = MAX_INDEX_CACHE_AGE_HOURS
    
    cache_path = os.path.join(CACHE_DIR, f"{symbol}.parquet")
    if not os.path.exists(cache_path):
        return True
    
    cache_age = (datetime.datetime.now() - datetime.datetime.fromtimestamp(os.path.getmtime(cache_path)))
    return cache_age.total_seconds() > (max_age_hours * 3600)

def get_cache_age(symbol):
    """Returns cache age in hours. Returns -1 if no cache exists."""
    cache_path = os.path.join(CACHE_DIR, f"{symbol}.parquet")
    if not os.path.exists(cache_path):
        return -1
    return (datetime.datetime.now() - datetime.datetime.fromtimestamp(os.path.getmtime(cache_path))).total_seconds() / 3600.0


# ── SCREENER.IN STALENESS TRACKER ──────────────────────────────────────
# Prevents silent use of hardcoded PE/PB fallbacks when screener.in is down.
# After 7 days without a successful fetch, the pipeline must fail visibly.

_SCREENER_CACHE_FILE = os.path.join(CACHE_DIR, "screener_last_success.json")

def update_screener_cache_timestamp():
    """Record that screener.in was successfully fetched just now."""
    try:
        import json
        with open(_SCREENER_CACHE_FILE, "w") as f:
            json.dump({"last_success": datetime.datetime.now().isoformat()}, f)
    except Exception:
        pass

def is_screener_cache_stale(max_days=7):
    """Returns True if screener.in hasn't been successfully fetched in max_days."""
    try:
        import json
        if not os.path.exists(_SCREENER_CACHE_FILE):
            return True  # Never fetched = stale
        with open(_SCREENER_CACHE_FILE, "r") as f:
            data = json.load(f)
        last = datetime.datetime.fromisoformat(data.get("last_success", "2000-01-01"))
        return (datetime.datetime.now() - last).days > max_days
    except Exception:
        return True  # Corrupt file = stale

def get_historical_data(symbol, days=350, end_date=None):
    df = _get_historical_data_raw(symbol, days, end_date)
    if df is not None and not df.empty:
        df = df.dropna(subset=["Close", "High", "Low", "Open"])
    return df

def _get_historical_data_raw(symbol, days=350, end_date=None):
    """Fetches historical price data from cache, screener.in, yfinance, or mfapi.
    Returns None if ALL data sources are exhausted — never fabricates prices."""
    if end_date is None:
        end_date = datetime.date.today()
    elif isinstance(end_date, str):
        end_date = pd.Timestamp(end_date).date()
    elif isinstance(end_date, pd.Timestamp):
        end_date = end_date.date()
    elif isinstance(end_date, (datetime.date, datetime.datetime)):
        end_date = end_date.date()
        
    start_date = end_date - datetime.timedelta(days=days)
    
    cache_file = os.path.join(CACHE_DIR, f"{symbol}_history.csv")
    
    # Check if local cache exists and was modified today
    cache_exists = os.path.exists(cache_file)
    cache_is_fresh = False
    
    if cache_exists:
        try:
            mtime = datetime.date.fromtimestamp(os.path.getmtime(cache_file))
            if (datetime.date.today() - mtime).days < 1:
                cache_is_fresh = True
        except Exception:
            pass
            
    # We only use cache directly if it is fresh AND covers the required window.
    # Exception: if it is a purely historical range in the past, we don't care if the file is fresh,
    # as long as it has the required data window.
    is_historical_request = end_date < (datetime.date.today() - datetime.timedelta(days=2))
    can_use_cache = cache_exists and (cache_is_fresh or is_historical_request)
    
    if can_use_cache:
        try:
            df = pd.read_csv(cache_file, parse_dates=["Date"])
            df.set_index("Date", inplace=True)
            df.index = pd.to_datetime(df.index)
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            
            # Tag data source for pipeline integrity tracking
            if "data_source" not in df.columns:
                df["data_source"] = "cache"
                
            # Verify cache max date relative to market close
            is_today = (end_date == datetime.date.today())
            is_after_market = False
            if is_today:
                now = datetime.datetime.now()
                # Indian Market closes at 15:30. Let's use 15:45 PM for data availability.
                if now.hour > 15 or (now.hour == 15 and now.minute >= 45):
                    is_after_market = True
                    
            expected_max_date = end_date
            if is_today and not is_after_market:
                expected_max_date = end_date - datetime.timedelta(days=1)
                
            if expected_max_date.weekday() == 5:  # Saturday
                expected_max_date -= datetime.timedelta(days=1)
            elif expected_max_date.weekday() == 6:  # Sunday
                expected_max_date -= datetime.timedelta(days=2)
                
            cache_max_date = df.index.max().date()
            if cache_max_date < expected_max_date:
                log_info(f"Cached data for {symbol} is outdated (max date: {cache_max_date}, expected: {expected_max_date}). Forcing re-fetch.")
                raise ValueError("Cache is outdated")
                
            # Verify that the cached data covers the requested start_date and end_date
            # Allow cache if it goes up to within 2 days of the requested end_date (handles incomplete current day data)
            if df.index.min().date() <= start_date and df.index.max().date() >= (end_date - datetime.timedelta(days=2)):
                return df
            # Cache exists but doesn't go back far enough — fall through to fresh fetch
            log_info(f"Cache for {symbol} too short (range: {df.index.min().date()} to {df.index.max().date()}); re-fetching {days} days.")
        except Exception as e:
            log_warning(f"Failed to read cache for {symbol}: {e}")

    log_info(f"Retrieving price data for {symbol}...")
    
    # Check if index and fetch from Screener.in chart API
    if symbol in ["NIFTY_50", "CNX50", "NIFTY_500", "CNX500", "NIFTY_NEXT_50", "JUNIORBEES", "NIFTY_MIDCAP_150", "MID150", "NIFTY_SMALLCAP_250", "SMALLCA250", "NIFTY_MICROCAP_250", "NFMICRO250", "MICROCAP250"]:
        try:
            from screener_fetcher import fetch_screener_index_history
            df = fetch_screener_index_history(symbol, days=days)
            if df is not None and not df.empty:
                df["data_source"] = "screener_in"
                try:
                    df.to_csv(cache_file)
                except Exception as e:
                    log_warning(f"Failed to save cache for {symbol}: {e}")
                return df
        except Exception as e:
            log_warning(f"Screener index history fetch failed for {symbol}: {e}")
            
    # Fast-path for Mutual Funds (6-digit codes) to bypass slow yfinance failures
    clean_sym = str(symbol).replace(".NS", "").replace(".BO", "").strip()
    is_amfi = clean_sym.isdigit() and len(clean_sym) == 6
    if is_amfi:
        log_info(f"Symbol {symbol} is a 6-digit AMFI code. Bypassing yfinance to use mfapi directly...")
        try:
            from mf_fetcher import fetch_mf_data_auto
            mf_df = fetch_mf_data_auto(symbol)
            if mf_df is not None and not mf_df.empty:
                mf_df["data_source"] = "mfapi"
                try:
                    mf_df.to_csv(cache_file)
                except Exception as save_err:
                    pass
                return mf_df
        except Exception as mf_err:
            log_warning(f"Fast-path mfapi fetch failed for {symbol}: {mf_err}")
            
    # Try fetching live from yfinance with retry mechanism and timeout
    if not is_amfi:
        try:
            import yfinance as yf
            import time
            import concurrent.futures
            ticker_symbol = symbol
            if ticker_symbol.endswith(".NS"):
                ticker_symbol = ticker_symbol[:-3]
                
            is_global_or_index = (
                ticker_symbol.startswith("^") or
                "=" in ticker_symbol or
                "-" in ticker_symbol or
                ticker_symbol.upper() in [
                    "DBB", "CPER", "GLD", "SLV",
                    "SMH", "SOXX", "AIQ", "GRID", "SRVR", "XLU", "BOTZ", "ROBO", "XBI", "IBB", "ARKG",
                    "WGMI", "BLOK", "LIT", "REMX", "OZEM", "URA", "BUG", "CIBR", "ITA", "WCLD"
                ]
            )
            
            if ticker_symbol not in [
                "MCX_GOLD", "MCX_SILVER", "NIFTY_50", "NIFTY_MIDCAP_150", "NIFTY_NEXT_50", "NIFTY_SMALLCAP_250", "NIFTY_500",
                "SBI_SMALL_CAP", "NIPPON_SMALL_CAP", "HDFC_SMALL_CAP", "AXIS_SMALL_CAP", "KOTAK_SMALL_CAP", "MOTILAL_SMLCAP_250",
                "INDIA_VIX"
            ] and not ticker_symbol.endswith(".BO") and not is_global_or_index:
                ticker_symbol = f"{ticker_symbol}.NS"
            elif symbol == "NIFTY_50":
                ticker_symbol = "^NSEI"
            elif symbol == "NIFTY_500":
                ticker_symbol = "^CRSLDX"
            elif symbol == "NIFTY_NEXT_50":
                ticker_symbol = "JUNIORBEES.NS"
            elif symbol == "NIFTY_MIDCAP_150":
                ticker_symbol = "MID150CASE.NS"
            elif symbol == "NIFTY_SMALLCAP_250":
                ticker_symbol = "SMALLCAP.NS"
            elif symbol == "MCX_GOLD":
                ticker_symbol = "GC=F"
            elif symbol == "MCX_SILVER":
                ticker_symbol = "SI=F"
            elif symbol == "SBI_SMALL_CAP":
                ticker_symbol = "0P0000XW1B.BO"
            elif symbol == "NIPPON_SMALL_CAP":
                ticker_symbol = "0P0000PTGR.BO"
            elif symbol == "HDFC_SMALL_CAP":
                ticker_symbol = "0P0000XVAA.BO"
            elif symbol == "AXIS_SMALL_CAP":
                ticker_symbol = "0P00011MAX.BO"
            elif symbol == "KOTAK_SMALL_CAP":
                ticker_symbol = "0P0000XV6I.BO"
            elif symbol == "MOTILAL_SMLCAP_250":
                ticker_symbol = "0P0001NJAY.BO"
            elif symbol == "INDIA_VIX":
                ticker_symbol = "^INDIAVIX"
                
            ticker = yf.Ticker(ticker_symbol)
            
            df = None
            for attempt in range(3):
                try:
                    yf_end = (end_date + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
                    # Wrap history() with a timeout using a thread pool (works on Windows)
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(
                            ticker.history,
                            start=start_date.strftime("%Y-%m-%d"),
                            end=yf_end
                        )
                        try:
                            df = future.result(timeout=45)  # 45s timeout per attempt
                        except concurrent.futures.TimeoutError:
                            log_warning(f"yfinance history() timed out for {symbol} (attempt {attempt+1})")
                            continue
                    if df is not None and not df.empty and len(df) >= 50:
                        break
                except Exception as attempt_err:
                    if attempt == 2:
                        raise attempt_err
                    time.sleep(1)
            
            if df is None or df.empty or len(df) < 50:
                raise ValueError("Empty or insufficient data from yfinance.")
            
            # Clean index
            df.index = pd.to_datetime(df.index)
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            df.index.name = "Date"
            df = df[["Open", "High", "Low", "Close", "Volume"]]
            df["data_source"] = "yfinance"
    
            # Backfill missing pre-history using Nifty 50 (^NSEI) returns if ETF has short history
            first_date = df.index.min().date()
            target_start = start_date
            if (first_date - target_start).days > 30 and symbol in ["NIFTY_MIDCAP_150", "NIFTY_SMALLCAP_250"]:
                log_info(f"Backfilling missing history for {symbol} from {target_start} to {first_date} using Nifty 50 (^NSEI) returns...")
                try:
                    n50_ticker = yf.Ticker("^NSEI")
                    yf_end = (end_date + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
                    n50_df = n50_ticker.history(start=target_start.strftime("%Y-%m-%d"), end=yf_end)
                    if not n50_df.empty:
                        n50_df.index = pd.to_datetime(n50_df.index)
                        if n50_df.index.tz is not None:
                            n50_df.index = n50_df.index.tz_localize(None)
                        
                        n50_pre = n50_df.loc[:pd.Timestamp(first_date)].copy()
                        if len(n50_pre) > 1:
                            etf_first_close = df["Close"].dropna().iloc[0]
                            n50_first_close = n50_pre["Close"].iloc[-1]
                            
                            scaling = etf_first_close / n50_first_close
                            reconstructed = pd.DataFrame(index=n50_pre.index[:-1])
                            reconstructed.index.name = "Date"
                            
                            for col in ["Open", "High", "Low", "Close"]:
                                reconstructed[col] = n50_pre[col].iloc[:-1] * scaling
                            reconstructed["Volume"] = n50_pre["Volume"].iloc[:-1]
                            
                            df = pd.concat([reconstructed, df])
                            df = df.sort_index()
                except Exception as backfill_err:
                    log_warning(f"Backfill failed for {symbol}: {backfill_err}")
    
            # ── DATA INTEGRITY: Prices used as-is from upstream data sources ──
            # Scale factors previously present here have been removed.
            # Price manipulation disguises data source failures and corrupts
            # downstream indicators (RS, ATR, EMAs). If upstream data is wrong,
            # the pipeline must fail visibly, not silently "correct" numbers.
            
            # Save cache
            try:
                df.to_csv(cache_file)
            except Exception as e:
                log_warning(f"Failed to save cache for {symbol}: {e}")
                
            log_success(f"Successfully fetched live data for {symbol}.")
            return df
        except Exception as e:
            log_warning(f"Could not retrieve yfinance data for {symbol} ({e}).")
            
            # --- Mutual Fund API Fallback ---
            log_info(f"Attempting to route {symbol} to mfapi.in...")
            try:
                from mf_fetcher import fetch_mf_data_auto
                mf_df = fetch_mf_data_auto(symbol)
                if mf_df is not None and not mf_df.empty:
                    mf_df["data_source"] = "mfapi_fallback"
                    try:
                        mf_df.to_csv(cache_file)
                    except Exception as save_err:
                        log_warning(f"Failed to save mfapi cache for {symbol}: {save_err}")
                    log_success(f"Successfully fetched mfapi data for {symbol}.")
                    return mf_df
            except Exception as mf_err:
                log_warning(f"mfapi fallback failed for {symbol}: {mf_err}")
                
    if cache_exists:
        log_info(f"Falling back to existing cache for {symbol}.")
        try:
            df = pd.read_csv(cache_file, parse_dates=["Date"])
            df.set_index("Date", inplace=True)
            df.index = pd.to_datetime(df.index)
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            if "data_source" not in df.columns:
                df["data_source"] = "stale_cache_fallback"
            return df
        except Exception as read_err:
            log_warning(f"Failed to read fallback cache for {symbol}: {read_err}")
            
    log_error(f"CRITICAL: ALL data sources exhausted for {symbol}. No synthetic fallback — returning None to prevent trading on fabricated data.")
    return None
