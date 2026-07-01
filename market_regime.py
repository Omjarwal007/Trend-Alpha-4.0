import os
import pandas as pd
import numpy as np
from config import SECTORS, THEMES, SYMBOLS
from utils import log_info, log_success, log_warning
from cache_manager import get_historical_data
from pipeline_data import calculate_chop_index

def determine_index_trend(df):
    """Evaluates trend parameters for index data."""
    if df is None or df.empty:
        return {"above_150": False, "rising_150": False, "distance_150": 0.0, "chop": 35.0, "whipsaw": 0, "bullish": False}
        
    last_close = float(df["Close"].iloc[-1])
    ma_150_col = "EMA_150" if "EMA_150" in df.columns else "SMA_150"
    ma_150 = float(df[ma_150_col].iloc[-1])
    above_150 = last_close > ma_150
    
    # 150 MA rising check (slope over 10 days)
    rising_150 = float(df[ma_150_col].iloc[-1]) > float(df[ma_150_col].iloc[-10])
    is_bullish = above_150 and rising_150
    
    distance_150 = ((last_close - ma_150) / ma_150) * 100.0
    chop = float(df["CHOP_avg"].iloc[-1])
    whipsaw = int(df["Whipsaws_20d_30MA"].iloc[-1])
    
    return {
        "above_150": above_150,
        "rising_150": rising_150,
        "distance_150": distance_150,
        "chop": chop,
        "whipsaw": whipsaw,
        "bullish": is_bullish
    }

def run_market_regime(pipeline_data, date_str=None):
    """Calculates index trend parameters and determines the unified market regime."""
    log_info("Executing SKILL 03 & 04: Calculating Market Regime & Breadth status...")
    
    # Extract index frames
    nifty50_df = pipeline_data.get("NIFTY_50")
    niftynext50_df = pipeline_data.get("NIFTY_NEXT_50")
    nifty150_df = pipeline_data.get("NIFTY_MIDCAP_150")
    nifty250_df = pipeline_data.get("NIFTY_SMALLCAP_250")
    niftymicro250_df = pipeline_data.get("NIFTY_MICROCAP_250")
    nifty500_df = pipeline_data.get("NIFTY_500", nifty50_df)
    
    n50_params = determine_index_trend(nifty50_df)
    nnext50_params = determine_index_trend(niftynext50_df)
    n150_params = determine_index_trend(nifty150_df)
    n250_params = determine_index_trend(nifty250_df)
    nmicro250_params = determine_index_trend(niftymicro250_df)
    
    # ── CALCULATE MARKET BREADTH (using Chartink universe + bhavcopy cache for broader coverage) ──
    above_20dma = 0
    above_50dma = 0
    above_150dma = 0
    above_200dma = 0
    total_stocks = 0
    
    # Reconstruct daily breadth history for Breadth Thrust calculation
    thrust_pct_series = []
    
    # Use CHARTINK stock set (more comprehensive) than just 29 SYMBOLS
    # Try to load broader universe from Chartink or bhavcopy
    breadth_symbols = list(SYMBOLS)  # Start with 29 benchmarks
    
    # Try to expand with Chartink universe (if available)
    try:
        from screener_fetcher import fetch_chartink_universe
        chartink_stocks = fetch_chartink_universe(date_str)
        if chartink_stocks and len(chartink_stocks) > len(breadth_symbols):
            # Add unique Chartink stocks
            chartink_symbols = [s.replace(".NS","") for s in chartink_stocks if s]
            for cs in chartink_symbols:
                if cs not in breadth_symbols:
                    breadth_symbols.append(cs)
        log_info(f"Market breadth using {len(breadth_symbols)} stocks ({len(chartink_stocks) if chartink_stocks else 0} from Chartink)")
    except Exception as e:
        log_warning(f"Could not expand breadth universe: {e}")
    
    # Try to expand with bhavcopy cache stocks (all NSE stocks)
    try:
        import glob
        cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Hermes", "bhavcopy_cache")
        if os.path.isdir(cache_dir):
            parquet_files = sorted(glob.glob(os.path.join(cache_dir, "bhav_*.parquet")))
            if parquet_files:
                latest_bhav = pd.read_parquet(parquet_files[-1])
                if "SYMBOL" in latest_bhav.columns:
                    bhav_symbols = latest_bhav["SYMBOL"].unique().tolist()
                    # Add at most 200 unique symbols from bhavcopy
                    added = 0
                    for bs in bhav_symbols:
                        if bs not in breadth_symbols and added < 200:
                            breadth_symbols.append(bs)
                            added += 1
                    log_info(f"Expanded breadth to {len(breadth_symbols)} stocks (+{added} from bhavcopy)")
    except Exception as e:
        log_warning(f"Could not expand via bhavcopy: {e}")
    
    # Build benchmark_histories from the expanded breadth_symbols list
    benchmark_histories = {}
    for s in breadth_symbols:
        df_hist = None
        if pipeline_data:
            df_hist = pipeline_data.get(s)
        if df_hist is None:
            df_hist = get_historical_data(s, end_date=date_str)
        if df_hist is not None and not df_hist.empty and len(df_hist) >= 50:
            benchmark_histories[s] = df_hist
            if len(benchmark_histories) >= 200:  # Cap at 200 for performance
                break
            
    total_stocks = len(benchmark_histories)
    if total_stocks == 0:
        total_stocks = 1 # avoid division by zero
        
    # Reconstruct breadth over last 50 sessions
    for idx in range(-50, 0):
        a50 = 0
        valid_count = 0
        for s, df in benchmark_histories.items():
            if len(df) >= abs(idx):
                close_val = df["Close"].iloc[idx]
                ma50_val = df["Close"].rolling(50).mean().iloc[idx]
                if close_val > ma50_val:
                    a50 += 1
                valid_count += 1
        pct = (a50 / valid_count * 100.0) if valid_count > 0 else 50.0
        thrust_pct_series.append(pct)
        
    # Latest breadths
    for s, df in benchmark_histories.items():
        last_close = df["Close"].iloc[-1]
        ema20 = df["Close"].ewm(span=20, adjust=False).mean().iloc[-1]
        sma50 = df["Close"].rolling(50).mean().iloc[-1]
        sma150 = df["Close"].rolling(150).mean().iloc[-1]
        sma200 = df["Close"].rolling(200).mean().iloc[-1]
        
        if last_close > ema20: above_20dma += 1
        if last_close > sma50: above_50dma += 1
        if last_close > sma150: above_150dma += 1
        if last_close > sma200: above_200dma += 1
        
    pct_20 = (above_20dma / total_stocks) * 100.0
    pct_50 = (above_50dma / total_stocks) * 100.0
    pct_150 = (above_150dma / total_stocks) * 100.0
    pct_200 = (above_200dma / total_stocks) * 100.0
    
    breadth_score = (pct_20 + pct_50 + pct_150 + pct_200) / 4.0
    
    # Check Breadth Thrust Signal (rising from <=40% to >=70% within 20 sessions)
    breadth_thrust = False
    for i in range(len(thrust_pct_series) - 20):
        segment = thrust_pct_series[i:i+20]
        if segment[0] <= 40.0 and segment[-1] >= 70.0:
            breadth_thrust = True
            break
            
    # NEW: Breadth Confirmation (current level is sustained high)
    recent_avg = sum(thrust_pct_series[-10:]) / 10 if len(thrust_pct_series) >= 10 else breadth_score
    breadth_confirmed = breadth_score >= 70.0 and breadth_score >= recent_avg - 5.0
            
    # Classify Breadth Regimes
    if breadth_score > 80.0:
        breadth_regime = "Powerful Bull"
        exposure_multiplier = 1.50
    elif breadth_score > 65.0:
        breadth_regime = "Bull"
        exposure_multiplier = 1.00
    elif breadth_score > 50.0:
        breadth_regime = "Neutral"
        exposure_multiplier = 0.75
    elif breadth_score > 35.0:
        breadth_regime = "Weak"
        exposure_multiplier = 0.50
    else:
        breadth_regime = "Risk Off"
        exposure_multiplier = 0.25
        
    # ── UNIFIED REGIME ENGINE (SKILL 03) ──────────────────────────
    nifty_pass = n50_params["above_150"] and n50_params["rising_150"] and n50_params["chop"] < 55.0 and n50_params["whipsaw"] <= 2
    small250_pass = n250_params["above_150"] and n250_params["rising_150"] and n250_params["chop"] < 55.0 and n250_params["whipsaw"] <= 3
    
    new_highs_count = 0
    new_lows_count = 0
    
    for s, df in benchmark_histories.items():
        last_close = float(df["Close"].iloc[-1])
        high_252 = float(df["Close"].rolling(252).max().iloc[-1]) if len(df) >= 252 else float(df["Close"].max())
        low_252 = float(df["Close"].rolling(252).min().iloc[-1]) if len(df) >= 252 else float(df["Close"].min())
        
        if last_close >= high_252 * 0.98:
            new_highs_count += 1
        if last_close <= low_252 * 1.02:
            new_lows_count += 1
            
    leadership_ratio = new_highs_count / max(new_lows_count, 1)
    
    # Determine Nifty 500 200 SMA trend
    nifty500_below_200 = False
    if nifty500_df is not None and not nifty500_df.empty:
        last_close_n500 = float(nifty500_df["Close"].iloc[-1])
        sma_200_n500 = float(nifty500_df["SMA_200"].iloc[-1]) if "SMA_200" in nifty500_df.columns else float(nifty500_df["Close"].rolling(200).mean().iloc[-1])
        nifty500_below_200 = last_close_n500 < sma_200_n500

    # Calculate Nifty 500 150 EMA trend for cash allocation rules
    nifty500_below_150_sloping_down = False
    if nifty500_df is not None and not nifty500_df.empty:
        n500_cp = nifty500_df.copy()
        if "EMA_150" not in n500_cp.columns:
            n500_cp["EMA_150"] = n500_cp["Close"].ewm(span=150, adjust=False).mean()
        
        last_close_n500 = float(n500_cp["Close"].iloc[-1])
        last_ema_150_n500 = float(n500_cp["EMA_150"].iloc[-1])
        prev_ema_150_n500 = float(n500_cp["EMA_150"].iloc[-10]) if len(n500_cp) >= 10 else last_ema_150_n500
        nifty500_below_150_sloping_down = (last_close_n500 < last_ema_150_n500) and (last_ema_150_n500 < prev_ema_150_n500)

    # Determine Nifty 500 200 SMA trend for Global Off-Switch (Module A)
    # Using Nifty 500 instead of Nifty 50 to prevent false-positive shutdowns when broader market is strong
    nifty50_below_200_3d = False
    if nifty500_df is not None and len(nifty500_df) >= 3:
        n500_closes = nifty500_df["Close"].iloc[-3:].values
        if "SMA_200" in nifty500_df.columns:
            n500_sma200 = nifty500_df["SMA_200"].iloc[-3:].values
        else:
            n500_sma200 = nifty500_df["Close"].rolling(200).mean().iloc[-3:].values
            
        # Check if close is below 200 SMA for all of the last 3 days
        valid_pairs = [(c, s) for c, s in zip(n500_closes, n500_sma200) if pd.notna(s)]
        if valid_pairs and all(c < s for c, s in valid_pairs):
            nifty50_below_200_3d = True
        
    breadth_below_30 = pct_200 < 30.0
    stop_new_buys = breadth_below_30 or nifty50_below_200_3d
    if stop_new_buys:
        if nifty50_below_200_3d:
            log_warning(f"MARKET OFF-SWITCH TRIGGERED: NIFTY 500 Below 200-SMA for 3 consecutive days! (SYSTEM RED)")
        if breadth_below_30:
            log_warning(f"MARKET OFF-SWITCH TRIGGERED: 200 DMA Breadth ({pct_200:.1f}%) < 30%")
        
    # NEW: Count bullish indices (0-4 scale)
    bull_votes = sum([
        n50_params["bullish"],
        nnext50_params["bullish"],
        n150_params["bullish"],
        n250_params["bullish"]
    ])

    # NEW: Weighted regime determination
    if nifty50_below_200_3d:
        market_regime = "CRISIS" if n250_params["chop"] > 61.8 else "BEAR"
    elif not n50_params["above_150"] and not n250_params["above_150"] and breadth_score < 35.0:
        market_regime = "CRISIS" if n250_params["chop"] > 61.8 else "BEAR"
    elif not n250_params["above_150"]:
        market_regime = "CORRECTION"
    elif bull_votes >= 3 and breadth_score > 80.0:
        market_regime = "BULL"            # 3-4 indices BULL + strong breadth
    elif bull_votes >= 3 and breadth_score > 60.0:
        market_regime = "EARLY_BULL"      # 3-4 indices BULL + moderate breadth
    elif bull_votes >= 2 and breadth_score > 60.0:
        market_regime = "LATE_BULL"       # 2 indices BULL + moderate breadth
    elif bull_votes >= 2 and breadth_score > 40.0:
        market_regime = "SIDEWAYS"        # Mixed signals
    else:
        market_regime = "SIDEWAYS"        # Weak/no leadership
        
    bullish_indices = []
    if n50_params["bullish"]: bullish_indices.append("NIFTY_50")
    if nnext50_params["bullish"]: bullish_indices.append("NIFTY_NEXT_50")
    if n150_params["bullish"]: bullish_indices.append("NIFTY_MIDCAP_150")
    if n250_params["bullish"]: bullish_indices.append("NIFTY_SMALLCAP_250")
    if nmicro250_params["bullish"]: bullish_indices.append("NIFTY_MICROCAP_250")

    regime_results = {
        "nifty_above_150": n50_params["above_150"],
        "midsmall_above_150": n250_params["above_150"],
        "nifty_chop": n50_params["chop"],
        "midsmall_chop": n250_params["chop"],
        "nifty_pass": nifty_pass,
        "midsmall_pass": small250_pass,
        "breadth_score": breadth_score,
        "breadth_regime": breadth_regime,
        "exposure_multiplier": exposure_multiplier,
        "breadth_thrust_active": breadth_thrust,
        "breadth_confirmed": breadth_confirmed,
        "leadership_ratio": leadership_ratio,
        "new_highs_52w": new_highs_count,
        "new_lows_52w": new_lows_count,
        "market_regime": market_regime,
        "stop_new_buys": stop_new_buys,
        "nifty500_below_200": nifty500_below_200,
        "breadth_below_30": breadth_below_30,
        "nifty50_bullish": n50_params["bullish"],
        "niftynext50_bullish": nnext50_params["bullish"],
        "nifty150_bullish": n150_params["bullish"],
        "nifty250_bullish": n250_params["bullish"],
        "microcap250_bullish": nmicro250_params["bullish"],
        "bullish_indices": bullish_indices,
        "bull_votes": bull_votes,
        "nifty50_below_200_3d": nifty50_below_200_3d,
        "nifty500_below_150_sloping_down": nifty500_below_150_sloping_down
    }
    
    # Fetch index fundamentals dynamically from Screener.in
    # Track whether any fundamental fetch succeeded for staleness detection
    _fundamentals_fetched_live = False
    def _try_fetch_fundamentals(idx_name):
        nonlocal _fundamentals_fetched_live
        try:
            from screener_fetcher import fetch_screener_fundamentals
            result = fetch_screener_fundamentals(idx_name) or {}
            src = result.get("Data_Source", "")
            # Count as "live" if data came from an actual web fetch, not a hardcoded DB
            if src not in ("", "Default fallback", "Hardcoded fallback", "FUNDAMENTAL_DB", "DYNAMIC_CSV"):
                _fundamentals_fetched_live = True
            return result
        except Exception as e:
            log_warning(f"Failed to fetch index fundamentals for {idx_name}: {e}")
            return {}
    
    n500_funda = _try_fetch_fundamentals("NIFTY_500")
    small250_funda = _try_fetch_fundamentals("NIFTY_SMALLCAP_250")

    # Fetch Nifty 50 fundamentals
    n50_funda = _try_fetch_fundamentals("NIFTY_50")

    # Fetch Nifty Next 50 fundamentals
    next50_funda = _try_fetch_fundamentals("NIFTY_NEXT_50")

    # Fetch Midcap 150 fundamentals
    mid150_funda = _try_fetch_fundamentals("NIFTY_MIDCAP_150")

    # Determine fallback prices from index history dataframes
    n50_df_price = float(nifty50_df["Close"].iloc[-1]) if nifty50_df is not None and not nifty50_df.empty else 24000.0
    nnext50_df_price = float(niftynext50_df["Close"].iloc[-1]) if niftynext50_df is not None and not niftynext50_df.empty else 75500.0
    if 0 < nnext50_df_price < 2000:
        nnext50_df_price *= 100.0
    n150_df_price = float(nifty150_df["Close"].iloc[-1]) if nifty150_df is not None and not nifty150_df.empty else 22600.0
    if 0 < n150_df_price < 100:
        n150_df_price *= 1000.0
    n250_df_price = float(nifty250_df["Close"].iloc[-1]) if nifty250_df is not None and not nifty250_df.empty else 17000.0
    n500_df_price = float(nifty500_df["Close"].iloc[-1]) if nifty500_df is not None and not nifty500_df.empty else 22600.0

    regime_results.update({
        "nifty500_pe": n500_funda.get("PE_Ratio") or 22.4,
        "nifty500_pb": n500_funda.get("Price_to_Book") or 3.45,
        "nifty500_div_yield": n500_funda.get("Dividend_Yield") or 1.08,
        "nifty500_cagr_1yr": n500_funda.get("CAGR_1Yr") or -3.15,
        "nifty500_cagr_5yr": n500_funda.get("CAGR_5Yr") or 10.7,
        "nifty500_cagr_10yr": n500_funda.get("CAGR_10Yr") or 12.8,
        "nifty500_mcap_cr": n500_funda.get("Market_Cap_Cr") or 41141029.0,
        "nifty500_price": n500_funda.get("Current_Price") or n500_df_price,

        "smallcap250_pe": small250_funda.get("PE_Ratio") or 28.1,
        "smallcap250_pb": small250_funda.get("Price_to_Book") or 3.82,
        "smallcap250_div_yield": small250_funda.get("Dividend_Yield") or 0.78,
        "smallcap250_cagr_1yr": small250_funda.get("CAGR_1Yr") or 0.25,
        "smallcap250_cagr_5yr": small250_funda.get("CAGR_5Yr") or 15.2,
        "smallcap250_cagr_10yr": small250_funda.get("CAGR_10Yr") or 15.5,
        "smallcap250_mcap_cr": small250_funda.get("Market_Cap_Cr") or 10000000.0,
        "smallcap250_price": small250_funda.get("Current_Price") or n250_df_price,

        "nifty50_pe": n50_funda.get("PE_Ratio") or 20.5,
        "nifty50_pb": n50_funda.get("Price_to_Book") or 3.2,
        "nifty50_div_yield": n50_funda.get("Dividend_Yield") or 1.3,
        "nifty50_cagr_1yr": n50_funda.get("CAGR_1Yr") or -2.0,
        "nifty50_cagr_5yr": n50_funda.get("CAGR_5Yr") or 12.0,
        "nifty50_cagr_10yr": n50_funda.get("CAGR_10Yr") or 11.0,
        "nifty50_mcap_cr": n50_funda.get("Market_Cap_Cr") or 180000000.0,
        "nifty50_price": n50_funda.get("Current_Price") or n50_df_price,

        "niftynext50_pe": next50_funda.get("PE_Ratio") or 26.0,
        "niftynext50_pb": next50_funda.get("Price_to_Book") or 4.5,
        "niftynext50_div_yield": next50_funda.get("Dividend_Yield") or 0.9,
        "niftynext50_cagr_1yr": next50_funda.get("CAGR_1Yr") or -1.5,
        "niftynext50_cagr_5yr": next50_funda.get("CAGR_5Yr") or 14.0,
        "niftynext50_cagr_10yr": next50_funda.get("CAGR_10Yr") or 13.0,
        "niftynext50_mcap_cr": next50_funda.get("Market_Cap_Cr") or 25000000.0,
        "niftynext50_price": next50_funda.get("Current_Price") or nnext50_df_price,

        "midcap150_pe": mid150_funda.get("PE_Ratio") or 30.0,
        "midcap150_pb": mid150_funda.get("Price_to_Book") or 4.0,
        "midcap150_div_yield": mid150_funda.get("Dividend_Yield") or 0.7,
        "midcap150_cagr_1yr": mid150_funda.get("CAGR_1Yr") or -0.5,
        "midcap150_cagr_5yr": mid150_funda.get("CAGR_5Yr") or 18.0,
        "midcap150_cagr_10yr": mid150_funda.get("CAGR_10Yr") or 16.0,
        "midcap150_mcap_cr": mid150_funda.get("Market_Cap_Cr") or 35000000.0,
        "midcap150_price": mid150_funda.get("Current_Price") or n150_df_price
    })
    
    # Fetch Microcap 250 fundamentals
    micro250_funda = _try_fetch_fundamentals("NIFTY_MICROCAP_250")

    regime_results.update({
        "microcap250_pe": micro250_funda.get("PE_Ratio") or 27.1,
        "microcap250_pb": micro250_funda.get("Price_to_Book") or 3.43,
        "microcap250_div_yield": micro250_funda.get("Dividend_Yield") or 0.65,
        "microcap250_cagr_1yr": micro250_funda.get("CAGR_1Yr") or -1.32,
        "microcap250_cagr_5yr": micro250_funda.get("CAGR_5Yr") or 16.0,
        "microcap250_cagr_10yr": micro250_funda.get("CAGR_10Yr") or 14.0,
        "microcap250_mcap_cr": micro250_funda.get("Market_Cap_Cr") or 2017417.0,
        "microcap250_price": micro250_funda.get("Current_Price") or 24000.0
    })
    
    # SCREENER.IN STALENESS CHECK: If no live fundamentals were fetched AND
    # cache is >7 days stale, the pipeline is running on unreliable hardcoded defaults.
    if not _fundamentals_fetched_live:
        try:
            from cache_manager import is_screener_cache_stale
            if is_screener_cache_stale(max_days=7):
                log_error("CRITICAL: Screener.in fundamentals >7 days stale. PE/PB/valuations are hardcoded defaults. Pipeline running blind.")
                regime_results["screener_data_stale"] = True
            else:
                regime_results["screener_data_stale"] = False
        except Exception:
            regime_results["screener_data_stale"] = False
    else:
        try:
            from cache_manager import update_screener_cache_timestamp
            update_screener_cache_timestamp()
            regime_results["screener_data_stale"] = False
        except Exception:
            pass
    
    log_success(f"Regime Check Completed: {market_regime} | Breadth: {breadth_regime} ({breadth_score:.1f}%)")
    return regime_results

def run_sector_rotation(pipeline_data, date_str=None):
    """Calculates sector and industry relative strength scores and rankings using the benchmark universe."""
    log_info("Executing SKILL 05: Sector Rotation & Hierarchical RS rankings...")
    
    # Compute relative strength rankings based on the 29 benchmark stocks
    small250_df = pipeline_data.get("NIFTY_SMALLCAP_250")
    if small250_df is not None and not small250_df.empty:
        # Calculate 6m returns of benchmark index
        bench_close_start = float(small250_df["Close"].iloc[-126]) if len(small250_df) >= 126 else float(small250_df["Close"].iloc[0])
        bench_close_end = float(small250_df["Close"].iloc[-1])
        bench_6m_roc = (bench_close_end - bench_close_start) / bench_close_start * 100.0
    else:
        bench_6m_roc = 10.0
        
    stock_records = []
    for s in SYMBOLS:
        df = None
        if pipeline_data:
            df = pipeline_data.get(s)
        if df is None:
            df = get_historical_data(s, end_date=date_str)
        if df is not None and not df.empty and len(df) >= 126:
            close_start = float(df["Close"].iloc[-126])
            close_end = float(df["Close"].iloc[-1])
            roc_6m = (close_end - close_start) / close_start * 100.0
            rs_score = roc_6m - bench_6m_roc
            
            sector = SECTORS.get(s, "Unknown")
            theme = THEMES.get(s, "Generic Theme")
            
            stock_records.append({
                "Symbol": s,
                "Sector": sector,
                "Theme": theme,
                "ROC_6m": roc_6m,
                "RS_Score": rs_score
            })
            
    df_stocks = pd.DataFrame(stock_records)
    
    if not df_stocks.empty:
        # Group by Sector and calculate mean RS_Score
        sector_rs = df_stocks.groupby("Sector")["RS_Score"].mean().reset_index()
        sector_rs["Sector_Rank"] = sector_rs["RS_Score"].rank(pct=True) * 100.0
        
        # Group by Theme and calculate mean RS_Score
        theme_rs = df_stocks.groupby("Theme")["RS_Score"].mean().reset_index()
        theme_rs["Theme_Rank"] = theme_rs["RS_Score"].rank(pct=True) * 100.0
        
        # Merge rankings back to stocks dataframe
        df_stocks = df_stocks.merge(sector_rs[["Sector", "Sector_Rank"]], on="Sector")
        df_stocks = df_stocks.merge(theme_rs[["Theme", "Theme_Rank"]], on="Theme")
        
        # Map sector and theme names directly to their rankings (supporting dynamic stocks)
        sector_rank_map = dict(zip(sector_rs["Sector"], sector_rs["Sector_Rank"]))
        theme_rank_map = dict(zip(theme_rs["Theme"], theme_rs["Theme_Rank"]))
    else:
        sector_rank_map = {}
        theme_rank_map = {}
        
    log_success("Sector rotation complete.")
    return sector_rank_map, theme_rank_map
