import os
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from utils import log_info, log_warning, log_error, categorize_fund_by_name, get_global_mf_name_map, extract_fund_keywords
from config import MAX_CORE_ETFS, CORE_FRICTION_PENALTY_PCT, BASE_DIR, CORE_ETF_ZONES
from universe_generator import generate_curated_universe
from cache_manager import get_historical_data
from scipy.stats import linregress

import concurrent.futures

def fetch_single_ticker(ticker, lookback_days, date_str):
    try:
        df = get_historical_data(ticker, days=lookback_days, end_date=date_str)
        if df is not None and not df.empty:
            end_date = pd.to_datetime(date_str)
            start_date = end_date - timedelta(days=lookback_days)
            df_filtered = df[(df.index >= start_date) & (df.index <= end_date)].copy()
            if len(df_filtered) > 63:  # Must have > 3 months of trading history
                return ticker, df_filtered
    except Exception as e:
        log_warning(f"Failed to fetch data for {ticker}: {e}")
    return ticker, None

def fetch_etf_data(date_str, lookback_days=365):
    """
    Fetches historical data for the dynamically generated ETF/Mutual Fund Universe concurrently.
    """
    curated_symbols = generate_curated_universe()
    tickers = list(set(curated_symbols + ["^NSEI"])) # Include Nifty for benchmark
    
    log_info(f"Fetching data for {len(tickers)} Curated Core ETFs/Indices concurrently...")
    
    etf_data = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=30) as executor:
        futures = {executor.submit(fetch_single_ticker, ticker, lookback_days, date_str): ticker for ticker in tickers}
        
        for future in concurrent.futures.as_completed(futures):
            ticker, df = future.result()
            if df is not None:
                etf_data[ticker] = df
                
    log_info(f"Successfully fetched valid history for {len(etf_data)} funds.")
    return etf_data

def compute_advanced_momentum(etf_df, benchmark_df, ticker=""):
    """
    Calculates Alpha, Beta, RS Line Slope, Volatility, and the Multi-Window Anti-Fade Logic.
    Returns a dictionary of metrics.
    """
    if len(etf_df) < 63 or len(benchmark_df) < 63:
        return None
        
    # Align dates
    df_aligned = etf_df.join(benchmark_df["Close"], how="inner", rsuffix="_bench")
    if len(df_aligned) < 63:
        return None
        
    df_aligned["Return"] = df_aligned["Close"].pct_change()
    df_aligned["Bench_Return"] = df_aligned["Close_bench"].pct_change()
    df_aligned.dropna(inplace=True)
    
    if len(df_aligned) < 60:
        return None

    # Beta & Alpha (6-month approx 126 days)
    lookback = min(126, len(df_aligned))
    recent_df = df_aligned.tail(lookback)
    
    cov = np.cov(recent_df["Return"], recent_df["Bench_Return"])[0][1]
    var = np.var(recent_df["Bench_Return"])
    beta = cov / var if var != 0 else 1.0
    
    ann_return = (recent_df["Return"].mean() * 252)
    bench_ann_return = (recent_df["Bench_Return"].mean() * 252)
    risk_free_rate = 0.07 # 7% India Risk Free
    
    alpha = ann_return - (risk_free_rate + beta * (bench_ann_return - risk_free_rate))
    
    # Daily Volatility (Annualized)
    volatility = df_aligned["Return"].std() * np.sqrt(252) * 100.0
    if volatility == 0 or pd.isna(volatility):
        return None
        
    current_price = float(df_aligned["Close"].iloc[-1])
    
    # RS Line (Relative Strength vs Benchmark)
    df_aligned["RS_Line"] = df_aligned["Close"] / df_aligned["Close_bench"]
    current_rs = float(df_aligned["RS_Line"].iloc[-1])
    
    # ─── 1. MULTI-WINDOW BLEND (15D, 1M, 2M, 3M, 6M Relative Returns) ───
    def get_rs_return(days):
        if len(df_aligned) <= days:
            start_idx = 0
        else:
            start_idx = -days
        start_rs = float(df_aligned["RS_Line"].iloc[start_idx])
        return (current_rs - start_rs) / start_rs if start_rs > 0 else 0

    ret_15d = get_rs_return(15)
    ret_1m = get_rs_return(21)
    ret_2m = get_rs_return(42)
    ret_3m = get_rs_return(63)
    ret_6m = get_rs_return(126)
    ret_12m = get_rs_return(252)

    # Volatility-Adjusted Composite Momentum Score
    weighted_score = (0.20 * ret_15d) + (0.30 * ret_1m) + (0.25 * ret_2m) + (0.15 * ret_3m) + (0.10 * ret_6m)
    
    # 1-Month Short-Term Volatility (Annualized)
    vol_1m = df_aligned["Return"].tail(21).std() * np.sqrt(252) * 100.0
    
    # Quality Score (Sharpe-Momentum) - Uses the highest risk (1M vs 12M) to penalize aggressively
    max_risk = max(volatility, vol_1m if not pd.isna(vol_1m) else volatility)
    quality_score = weighted_score / max(0.1, max_risk / 100.0)
    
    # ─── 2. THE ACCELERATION RATIO ───
    accel_ratio = ret_3m / ret_12m if ret_12m > 0 else 0.0
    is_exhausted = accel_ratio < 0.6 and ret_12m > 0
    is_accelerating = accel_ratio > 1.2
    
    # ─── 3. UNDERPERFORMER U-TURN (RS BREAKOUT) ───
    df_aligned["RS_SMA_50"] = df_aligned["RS_Line"].rolling(window=50, min_periods=20).mean()
    df_aligned["SMA_50"] = df_aligned["Close"].rolling(window=50, min_periods=20).mean()
    df_aligned["SMA_200"] = df_aligned["Close"].rolling(window=200, min_periods=50).mean()
    
    sma_50 = float(df_aligned["SMA_50"].iloc[-1]) if not pd.isna(df_aligned["SMA_50"].iloc[-1]) else current_price
    sma_200 = float(df_aligned["SMA_200"].iloc[-1]) if not pd.isna(df_aligned["SMA_200"].iloc[-1]) else sma_50
    
    # Check if RS 50-SMA slope is positive
    rs_sma_50_current = float(df_aligned["RS_SMA_50"].iloc[-1])
    rs_sma_50_prev = float(df_aligned["RS_SMA_50"].iloc[-5]) if len(df_aligned) > 50 else rs_sma_50_current
    
    u_turn_signal = (current_price > sma_50) and (rs_sma_50_current > rs_sma_50_prev) and (ret_12m < 0.05)
    
    # ─── 4. ETF VS MF VOLATILITY BUFFERS ───
    high_252 = float(df_aligned["Close"].tail(252).max())
    drawdown_252 = (high_252 - current_price) / high_252 if high_252 > 0 else 0
    
    is_hard_exit = False
    if str(ticker).endswith(".NS") or str(ticker).endswith(".BO"):
        # ETF: 2.5x ATR Chandelier Exit
        if "High" in etf_df.columns and "Low" in etf_df.columns:
            tr1 = df_aligned["High"] - df_aligned["Low"]
            tr2 = (df_aligned["High"] - df_aligned["Close"].shift(1)).abs()
            tr3 = (df_aligned["Low"] - df_aligned["Close"].shift(1)).abs()
            df_aligned["TR"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr_14 = df_aligned["TR"].rolling(14).mean().iloc[-1]
            
            high_22 = float(df_aligned["High"].tail(22).max())
            chandelier_stop = high_22 - (2.5 * atr_14)
            is_hard_exit = current_price < chandelier_stop
        else:
            # Fallback if no High/Low
            is_hard_exit = drawdown_252 > 0.10
    else:
        # MF: 5% Drawdown for 15 consecutive days
        df_aligned["Drawdown"] = (high_252 - df_aligned["Close"]) / high_252
        is_deep_dd = (df_aligned["Drawdown"] > 0.05).astype(int)
        
        consecutive_dd = 0
        for val in reversed(is_deep_dd.tail(20).values):
            if val == 1:
                consecutive_dd += 1
            else:
                break
        is_hard_exit = consecutive_dd >= 15

    # Count consecutive days below 50-DMA
    is_below_50 = (df_aligned["Close"] < df_aligned["SMA_50"]).astype(int)
    consecutive_below = 0
    for val in reversed(is_below_50.tail(20).values):
        if val == 1:
            consecutive_below += 1
        else:
            break
            
    is_soft_exit_50dma = current_price < sma_50
    is_hard_exit_50dma = consecutive_below >= 10
    is_hard_exit = is_hard_exit or is_hard_exit_50dma

    is_negative_velocity_1m = ret_1m < 0
    
    # ─── 5. REGIME STAGE CLASSIFICATION ───
    gap_from_200 = (current_price - sma_200) / sma_200 if sma_200 > 0 else 0
    
    # Check 1M vol spike relative to 12M vol
    is_vol_spiking = vol_1m > (volatility * 1.3)
    
    if current_price < sma_200 and not u_turn_signal:
        if is_hard_exit:
            regime = "Stage 4 (Declining)"
        else:
            regime = "Stage 1 (Basing)"
    elif current_price > sma_50 and is_accelerating:
        regime = "Stage 2 (Advancing)"
    elif ret_12m > 0.15 and is_vol_spiking:
        regime = "Stage 3 (Topping)"
    else:
        regime = "Stage 2 (Neutral)"
        
    is_overextended = gap_from_200 > 0.30
    
    # --- COMBINED TREND FILTERS ---
    is_trending = not is_hard_exit
    is_buy_eligible = is_trending and not is_soft_exit_50dma and not is_negative_velocity_1m and not is_overextended
    
    return {
        "Close": current_price,
        "Alpha": alpha * 100.0,
        "Beta": beta,
        "Volatility": volatility,
        "Ret_1M": ret_1m,
        "Ret_3M": ret_3m,
        "Ret_6M": ret_6m,
        "Ret_12M": ret_12m,
        "Weighted_Score": weighted_score,
        "Quality_Score": quality_score,
        "Drawdown_252": drawdown_252,
        "Consecutive_Below_50DMA": consecutive_below,
        "Is_Exhausted": is_exhausted,
        "Is_Accelerating": is_accelerating,
        "U_Turn_Signal": u_turn_signal,
        "Regime_Stage": regime,
        "Is_Trending": is_trending,
        "Is_Buy_Eligible": is_buy_eligible
    }

def run_core_momentum_engine(date_str, existing_core_holdings=None):
    from db_manager import save_pipeline_stage
    """
    Executes the Two-Stage Core Pipeline:
    1. Fetches dynamically curated universe.
    2. Calculates Alpha, Beta, RS Slope, and Universe RS Rating.
    """
    log_info("Executing Stage 2: Advanced Core ETF/MF Mathematical Filter...")
    
    if existing_core_holdings is None:
        existing_core_holdings = []
        
    etf_data = fetch_etf_data(date_str)
    
    if "^NSEI" not in etf_data:
        log_error("Benchmark ^NSEI data missing, aborting ETF Core run.")
        return pd.DataFrame()
        
    benchmark_df = etf_data["^NSEI"]
    
    results = []
    
    for ticker, df in etf_data.items():
        if ticker == "^NSEI":
            continue
            
        metrics = compute_advanced_momentum(df, benchmark_df, ticker=ticker)
        if metrics is None:
            continue
            
        metrics["Symbol"] = ticker
        metrics["Is_Held"] = ticker in existing_core_holdings
        results.append(metrics)
        
    if not results:
        log_error("No valid ETF metrics calculated.")
        return pd.DataFrame()
        
    df_results = pd.DataFrame(results)
    
    # Calculate RS Rating (Percentile ranking based on Quality Score)
    df_results["RS_Rating"] = df_results["Quality_Score"].rank(pct=True) * 100.0
    
    # ── TA 4.0: Compute scores for ALL candidates (not just Is_Trending) ──
    # This ensures the full 301-candidate universe is available for momentum-based filtering
    _full_universe = df_results.copy()
    _full_universe["Score"] = _full_universe.apply(lambda r: r["RS_Rating"] * 0.5 if r.get("Is_Exhausted", False) else r["RS_Rating"], axis=1)
    _full_universe.sort_values("Score", ascending=False, inplace=True)
    _full_universe["Rank"] = range(1, len(_full_universe) + 1)
    mf_name_map = get_global_mf_name_map()
    _full_universe["Name"] = _full_universe["Symbol"].apply(lambda x: mf_name_map.get(str(x), str(x)))
    _full_universe["Category"] = _full_universe["Symbol"].apply(lambda x: CORE_ETF_ZONES.get(str(x), "Unknown"))
    # Save full universe for TA 4.0 dashboard selection (all 301 candidates with scores)
    _full_cols = ["Rank", "Symbol", "Name", "Category", "Score", "RS_Rating", "Weighted_Score", "Drawdown_252", "Volatility", "Is_Trending", "Is_Buy_Eligible", "Close"]
    save_pipeline_stage(_full_universe[_full_cols], "L1_Core_Universe", date_str)
    output_dir_fu = os.path.join(BASE_DIR, "output", date_str)
    _full_universe[_full_cols].to_csv(os.path.join(output_dir_fu, "L1_Core_Universe.csv"), index=False)
    
    # ── TA 4.0: Core Selection — Max 5, Max 1 Per Category, Momentum-Filtered ──
    # Replaces old Is_Trending filter, hysteresis, and RS_Rating-based retention
    # Uses ALL 301 candidates scored in _full_universe
    
    # Compute 1M and 3M returns from cached price history for each candidate
    _cache_dir_s = os.path.join(BASE_DIR, "cache")
    _sel_dt_s = pd.to_datetime(date_str)
    _cat_1m_sum = {}
    _cat_3m_sum = {}
    _cat_counts = {}
    _returns_cache = {}
    
    for _, _ur in _full_universe.iterrows():
        _sym = str(_ur["Symbol"])
        _cat = str(_ur.get("Category", "")).replace(".xlsx","").replace("..",".").replace("_"," ").strip().lower()
        _hfile = os.path.join(_cache_dir_s, f"{_sym}_history.csv")
        if os.path.exists(_hfile):
            try:
                _hdf = pd.read_csv(_hfile, parse_dates=["Date"])
                _hdf = _hdf.dropna(subset=["Close"]).set_index("Date")["Close"]
                _hdf = _hdf[_hdf.index <= _sel_dt_s].sort_index()
                if len(_hdf) >= 2:
                    _cut_1m = _sel_dt_s - pd.Timedelta(days=28)
                    _cut_3m = _sel_dt_s - pd.Timedelta(days=84)
                    _h1 = _hdf[_hdf.index >= _cut_1m]
                    _h3 = _hdf[_hdf.index >= _cut_3m]
                    _ret_1m = (_h1.iloc[-1] / _h1.iloc[0] - 1.0) if len(_h1) >= 2 else None
                    _ret_3m = (_h3.iloc[-1] / _h3.iloc[0] - 1.0) if len(_h3) >= 2 else None
                    _returns_cache[_sym] = (_ret_1m, _ret_3m, _cat)
                    if _ret_1m is not None:
                        _cat_1m_sum[_cat] = _cat_1m_sum.get(_cat, 0.0) + _ret_1m
                        _cat_counts[_cat] = _cat_counts.get(_cat, 0) + 1
                    if _ret_3m is not None:
                        _cat_3m_sum[_cat] = _cat_3m_sum.get(_cat, 0.0) + _ret_3m
            except:
                pass
    
    # Compute average category momentum
    _cat_1m_avg = {c: _cat_1m_sum[c] / _cat_counts.get(c, 1) for c in _cat_1m_sum}
    _cat_3m_avg = {c: _cat_3m_sum[c] / _cat_counts.get(c, 1) for c in _cat_3m_sum}
    
    # Filter: exclude categories where BOTH 1M AND 3M avg returns are negative
    _eligible_cats = set()
    for _cat in _cat_1m_avg:
        _c1m = _cat_1m_avg.get(_cat)
        _c3m = _cat_3m_avg.get(_cat)
        if (_c1m is not None and _c1m < 0) and (_c3m is not None and _c3m < 0):
            log_info(f"TA 4.0 Category Filter: Excluded '{_cat}' (1M={_c1m*100:+.1f}%, 3M={_c3m*100:+.1f}% — both negative)")
            continue
        _eligible_cats.add(_cat)
    
    # Pick best 1 fund per eligible category by Score, then take top 5
    _best_per_cat = {}
    for _, _ur in _full_universe.iterrows():
        _sym = str(_ur["Symbol"])
        _cat = str(_ur.get("Category", "")).replace(".xlsx","").replace("..",".").replace("_"," ").strip().lower()
        if _cat in _eligible_cats:
            _score = float(_ur.get("Score", 0))
            if _cat not in _best_per_cat or _score > _best_per_cat[_cat]["Score"]:
                _best_per_cat[_cat] = _ur
    
    _ta4_selected = sorted(_best_per_cat.values(), key=lambda r: r["Score"], reverse=True)[:MAX_CORE_ETFS]
    _selected_syms = [str(r["Symbol"]) for r in _ta4_selected]
    
    log_info(f"TA 4.0 Core Selection: {len(_eligible_cats)} eligible categories, {len(_ta4_selected)} funds selected (max {MAX_CORE_ETFS})")
    if _ta4_selected:
        for _r in _ta4_selected:
            _sym_s = str(_r["Symbol"])
            _ret1, _ret3, _ = _returns_cache.get(_sym_s, (None, None, ""))
            _r1m_s = f"{_ret1*100:+.1f}%" if _ret1 is not None else "N/A"
            _r3m_s = f"{_ret3*100:+.1f}%" if _ret3 is not None else "N/A"
            _cat_s = str(_r.get("Category", "")).replace(".xlsx","").replace("_"," ").title()
            log_info(f"  → {str(_r['Symbol']):>8s} | {_cat_s:<35s} | Score={_r['Score']:.1f} | 1M={_r1m_s} | 3M={_r3m_s}")
    
    # Build final selection from the TA 4.0 list
    if not _full_universe.empty:
        final_selection = _full_universe[_full_universe["Symbol"].isin(_selected_syms)].copy()
    else:
        final_selection = pd.DataFrame()
    
    # --- STAGE C: ALLOCATION WEIGHTING (Volatility + Risk + RS Momentum) ---
    if not final_selection.empty:
        final_selection.sort_values("Score", ascending=False, inplace=True)
        final_selection["Rank"] = range(1, len(final_selection) + 1)
        final_selection["Core_Weight"] = 0.0
        
        # Volatility Scaling
        final_selection["Inv_Vol"] = 1.0 / final_selection["Volatility"].replace(0, 1.0)
        
        # Combined Weight = Quality Score (RS Momentum / Risk) * Inverse Volatility
        final_selection["Combined_Weight"] = final_selection["Quality_Score"] * final_selection["Inv_Vol"]
        
        total_weight = final_selection["Combined_Weight"].sum()
        if total_weight > 0:
            final_selection["Core_Weight"] = final_selection["Combined_Weight"] / total_weight
        else:
            final_selection["Core_Weight"] = 1.0 / len(final_selection)
            
    top_etfs = final_selection.copy()
    
    # Save output to SQLite Database
    cols = ["Rank", "Symbol", "Name", "Category", "Score", "RS_Rating", "Weighted_Score", "Drawdown_252", "Volatility", "Is_Trending", "Is_Buy_Eligible", "Close"]
    alloc_cols = ["Rank", "Symbol", "Name", "Category", "Score", "RS_Rating", "Weighted_Score", "Drawdown_252", "Volatility", "Core_Weight", "Is_Trending", "Close"]
    
    # Optional fallback for compatibility if they still want a hard copy log
    output_dir = os.path.join(BASE_DIR, "output", date_str)
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "L1_Core_Allocations.csv")
    if not top_etfs.empty:
        top_etfs[alloc_cols].to_csv(out_path, index=False)
    
    # Save the final top allocations
    save_pipeline_stage(top_etfs[alloc_cols] if not top_etfs.empty else top_etfs, "L1_Core_Allocations", date_str)
    
    log_info(f"Core Engine selected {len(top_etfs)} funds. Saved to SQLite database.")
    
    return top_etfs

if __name__ == "__main__":
    # Test script
    import sys
    d_str = sys.argv[1] if len(sys.argv) > 1 else datetime.now().strftime("%Y-%m-%d")
    run_core_momentum_engine(d_str)
