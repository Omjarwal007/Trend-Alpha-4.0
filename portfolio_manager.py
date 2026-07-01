import os
import pandas as pd
import numpy as np
import datetime
from config import (
    DEFAULT_PORTFOLIO_CAPITAL, BASE_RISK_PER_TRADE_PCT, MAX_SINGLE_STOCK_CASH_PCT,
    MAX_SINGLE_STOCK_ABS_PCT, MAX_SECTOR_PCT, MAX_INDUSTRY_PCT, MAX_THEME_PCT,
    MAX_PORTFOLIO_HEAT, MAX_OPEN_POSITIONS, EQUITY_HAIRCUT, LIQUID_BONDS_HAIRCUT,
    METALS_MARGIN_REQ, THEMES, BASE_DIR, CAP_CATEGORY_LIMITS, CYCLICAL_SECTOR_KEYWORDS,
    DRAWDOWN_YELLOW_PCT, DRAWDOWN_ORANGE_PCT, DRAWDOWN_RED_PCT, SECTORS,
    NEW_BUY_MIN_SCORE, TOP_N_STOCKS, CORE_ALLOCATION_PCT
)
from utils import log_info, log_success, log_warning
from cache_manager import get_historical_data
from pipeline_data import calculate_atr, calculate_natr
from monitoring_engine import calculate_rs_line

def run_drawdown_governor(peak_val, current_val):
    """Computes active risk limits based on portfolio drawdown level (SKILL 10)."""
    drawdown_pct = ((peak_val - current_val) / peak_val) * 100.0 if peak_val > 0 else 0.0
    
    yellow = abs(DRAWDOWN_YELLOW_PCT) * 100.0
    orange = abs(DRAWDOWN_ORANGE_PCT) * 100.0
    red = abs(DRAWDOWN_RED_PCT) * 100.0
    
    if drawdown_pct >= red:
        action = "FULL STOP — SYSTEM PAUSE"
        risk_mult = 0.0
        rules = ["Close ALL equity positions", "Move to 100% Liquid Bonds + Gold/Silver futures hedge", "Minimum 30-day cooling-off period"]
    elif drawdown_pct >= orange:
        action = "DEFENSIVE"
        risk_mult = 0.0
        rules = ["Close all TIER 2 and TIER 3 positions", "Retain only highest-conviction TIER 1 (max 10 stocks) with tight stops", "Zero new entries until drawdown recovers < 8%", "MTF leverage = 0", "Exit all overlays"]
    elif drawdown_pct >= yellow:
        action = "REDUCE"
        risk_mult = 0.50
        rules = ["Stop adding NEW positions", "Reduce MTF leverage to zero", "Reduce metals overlays by 50%", "Max risk per new trade drops to 0.5% temporarily"]
    else:
        action = "NORMAL"
        risk_mult = 1.00
        rules = ["No drawdown restrictions active"]
        
    return {
        "drawdown_pct": drawdown_pct,
        "action": action,
        "risk_multiplier": risk_mult,
        "rules": rules
    }

def calculate_initial_stop(entry_price, atr_14, natr_pct, conviction_tier, breakout_low=None, ma_50=None):
    """Calculates stop distance and checks safety gates (SKILL 02)."""
    if natr_pct < 2.0:
        base_mult = 1.5
    elif natr_pct < 3.0:
        base_mult = 2.0
    elif natr_pct < 4.0:
        base_mult = 2.5
    else:
        base_mult = 3.0
        
    tier_adj = {
        'TIER 1 — HIGH CONVICTION': 0.0,
        'TIER 2 — MEDIUM CONVICTION': 0.25,
        'TIER 3 — LOW CONVICTION': 0.50,
        'REJECTED': 0.50
    }.get(conviction_tier, 0.50)
    
    mult = base_mult + tier_adj
    atr_stop = entry_price - (mult * atr_14)
    struct_stop = breakout_low * 0.99 if breakout_low is not None else 0.0
    dma50_stop = ma_50 * 0.97 if ma_50 is not None else 0.0
    initial_stop = max(atr_stop, struct_stop, dma50_stop)
    
    max_stop_boundary = entry_price * 0.92
    warning_flag = False
    
    if initial_stop < max_stop_boundary:
        initial_stop = max_stop_boundary
        warning_flag = True
        
    stop_distance_pct = ((entry_price - initial_stop) / entry_price) * 100.0
    return initial_stop, stop_distance_pct, warning_flag

def check_metals_overlay_regime(mcx_gold_df, mcx_silver_df):
    """Calculates MCX Gold and MCX Silver trends and overlays (SKILL 11)."""
    if mcx_gold_df is None or mcx_gold_df.empty or mcx_silver_df is None or mcx_silver_df.empty:
        return {"gold_pass": False, "silver_pass": False, "total_alloc_pct": 0.0, "gold_alloc_pct": 0.0, "silver_alloc_pct": 0.0}
        
    gold_close = float(mcx_gold_df["Close"].iloc[-1])
    gold_150 = float(mcx_gold_df["SMA_150"].iloc[-1]) if "SMA_150" in mcx_gold_df.columns else gold_close
    gold_pass = gold_close > gold_150
    gold_alloc = 25.0 if gold_pass else 0.0
    
    silver_close = float(mcx_silver_df["Close"].iloc[-1])
    silver_150 = float(mcx_silver_df["SMA_150"].iloc[-1]) if "SMA_150" in mcx_silver_df.columns else silver_close
    silver_pass = silver_close > silver_150
    silver_alloc = 10.0 if silver_pass else 0.0
    
    total_alloc = gold_alloc + silver_alloc
    return {
        "gold_pass": gold_pass,
        "silver_pass": silver_pass,
        "total_alloc_pct": total_alloc,
        "gold_alloc_pct": gold_alloc,
        "silver_alloc_pct": silver_alloc
    }

def compute_stock_correlation(s1, s2, pipeline_data=None, date_str=None):
    """Calculates the correlation coefficient of daily returns between two stock symbols over the last 100 sessions."""
    df1 = None
    df2 = None
    if pipeline_data:
        df1 = pipeline_data.get(s1)
        df2 = pipeline_data.get(s2)
    if df1 is None:
        df1 = get_historical_data(s1, end_date=date_str)
    if df2 is None:
        df2 = get_historical_data(s2, end_date=date_str)
    
    if df1 is None or df1.empty or df2 is None or df2.empty:
        return 0.0
    if len(df1) < 50 or len(df2) < 50:
        return 0.0
        
    closes1 = df1["Close"].tail(100).copy()
    closes2 = df2["Close"].tail(100).copy()
    if hasattr(closes1.index, "tz") and closes1.index.tz is not None:
        closes1.index = closes1.index.tz_localize(None)
    if hasattr(closes2.index, "tz") and closes2.index.tz is not None:
        closes2.index = closes2.index.tz_localize(None)
    pct1 = closes1.pct_change().dropna()
    pct2 = closes2.pct_change().dropna()
    
    df_corr = pd.concat([pct1, pct2], axis=1).dropna()
    if df_corr.empty or len(df_corr) < 10:
        return 0.0
        
    corr = np.corrcoef(df_corr.iloc[:, 0], df_corr.iloc[:, 1])[0, 1]
    return float(corr) if not np.isnan(corr) else 0.0

def compute_stock_beta(symbol, benchmark_df, pipeline_data=None, date_str=None):
    """Calculates stock Beta relative to the mid-small cap benchmark over the last 100 sessions."""
    try:
        df_stock = None
        if pipeline_data:
            df_stock = pipeline_data.get(symbol)
        if df_stock is None:
            df_stock = get_historical_data(symbol, end_date=date_str)
        if df_stock is not None and not df_stock.empty and benchmark_df is not None and not benchmark_df.empty:
            df_stock_copy = df_stock.copy()
            benchmark_df_copy = benchmark_df.copy()
            if hasattr(df_stock_copy.index, "tz") and df_stock_copy.index.tz is not None:
                df_stock_copy.index = df_stock_copy.index.tz_localize(None)
            if hasattr(benchmark_df_copy.index, "tz") and benchmark_df_copy.index.tz is not None:
                benchmark_df_copy.index = benchmark_df_copy.index.tz_localize(None)
            
            common_idx = df_stock_copy.index.intersection(benchmark_df_copy.index).tail(100)
            if len(common_idx) >= 15:
                stock_ret = df_stock_copy.loc[common_idx, "Close"].pct_change().dropna()
                bench_ret = benchmark_df_copy.loc[common_idx, "Close"].pct_change().dropna()
                
                df_ret = pd.concat([stock_ret, bench_ret], axis=1).dropna()
                cov = np.cov(df_ret.iloc[:, 0], df_ret.iloc[:, 1])[0, 1]
                var = np.var(df_ret.iloc[:, 1])
                if var > 0:
                    return float(cov / var)
    except Exception:
        pass
    return 1.0

def compute_stock_volatility(symbol, pipeline_data=None, date_str=None):
    """Calculates annualized volatility of daily returns over the last 100 sessions."""
    try:
        df_stock = None
        if pipeline_data:
            df_stock = pipeline_data.get(symbol)
        if df_stock is None:
            df_stock = get_historical_data(symbol, end_date=date_str)
        if df_stock is not None and not df_stock.empty:
            stock_ret = df_stock["Close"].tail(100).pct_change().dropna()
            if len(stock_ret) >= 15:
                return float(np.std(stock_ret) * np.sqrt(252) * 100.0)
    except Exception:
        pass
    return 25.0


def get_regime_controls(regime_status):
    """Translate market regime output into exposure and risk multipliers."""
    regime = (regime_status or {}).get("market_regime", "SIDEWAYS")
    controls = {
        "BULL":       {"risk_multiplier": 1.25, "cash_multiplier": 1.00, "mtf_allowed": True},
        "EARLY_BULL": {"risk_multiplier": 1.00, "cash_multiplier": 0.95, "mtf_allowed": True},
        "LATE_BULL":  {"risk_multiplier": 0.85, "cash_multiplier": 0.80, "mtf_allowed": False},
        "SIDEWAYS":   {"risk_multiplier": 0.50, "cash_multiplier": 0.50, "mtf_allowed": False},
        "CORRECTION": {"risk_multiplier": 0.25, "cash_multiplier": 0.25, "mtf_allowed": False},
        "BEAR":       {"risk_multiplier": 0.00, "cash_multiplier": 0.00, "mtf_allowed": False},
        "CRISIS":     {"risk_multiplier": 0.00, "cash_multiplier": 0.00, "mtf_allowed": False},
    }
    return controls.get(regime, controls["SIDEWAYS"]) | {"market_regime": regime}


def rs_tilt_alloc_pct(rs_val, cap_cat):
    """Dynamic Composite Position Sizing — Minervini / O'Neil / Livermore scale.

    These legends ran CONCENTRATED portfolios with their best ideas at 15-25%.
    We follow their tiered conviction logic:
      RS > 1.50  → 15% target  (Market leader: Minervini goes all-in here)
      RS > 1.00  → 12%         (Strong outperformer: O'Neil full position)
      RS > 0.60  → 10%         (Healthy RS: add-to-winner territory)
      RS > 0.30  →  7%         (Standard conviction)
      RS 0.10-0.30→  5%        (Pilot buy: O'Neil 'starter half-position')

    For RS > 1.0 (proven leaders), the cap-category ceiling is doubled
    (Minervini would hold 20%+ in his best ideas — we use 15% as our max).
    """
    if rs_val > 1.50:
        base_pct = 15.0   # Market leader — go big
    elif rs_val > 1.00:
        base_pct = 12.0   # Strong outperformer
    elif rs_val > 0.60:
        base_pct = 10.0   # Healthy RS
    elif rs_val > 0.30:
        base_pct = 7.0    # Standard
    else:
        base_pct = 5.0    # Pilot / starter buy

    cat_limits = CAP_CATEGORY_LIMITS.get(cap_cat, CAP_CATEGORY_LIMITS["LARGE_CAP"])
    std_max_pct = cat_limits["max_single_pct"] * 100.0

    # Minervini override: proven leaders (RS > 1.0) can go to 2x standard cap, max 15%
    if rs_val > 1.0:
        effective_max = min(std_max_pct * 2.0, 15.0)
    else:
        effective_max = std_max_pct

    return min(base_pct, effective_max)


def get_minervini_max_pct(market_regime):
    """Returns max single-stock allocation % by market regime.

    In hot bull markets, Minervini / O'Neil concentrated 15-25% per position.
    We use a risk-managed version of this concentration philosophy:
      BULL        → 15% cap   (6-8 positions, deeply concentrated)
      EARLY_BULL  → 12% cap   (8-10 positions, building concentration)
      LATE_BULL   → 10% cap   (tighten as chop risk rises)
      SIDEWAYS    →  8% cap   (spread risk across more names)
      CORRECTION  →  6% cap   (defence mode, small sizes)
      BEAR/CRISIS →  0% cap   (no new buys at all)
    """
    regime_max = {
        "BULL":       15.0,
        "EARLY_BULL": 12.0,
        "LATE_BULL":  10.0,
        "SIDEWAYS":    8.0,
        "CORRECTION":  6.0,
        "BEAR":        0.0,
        "CRISIS":      0.0,
    }
    return regime_max.get(market_regime, 8.0)


def estimate_slippage(position_value, adtv_20, cap_category="MID_CAP", ann_vol_pct=30.0):
    """Estimates expected execution slippage as % of trade value.

    Based on NSE market microstructure: small positions relative to ADTV
    incur minimal impact; large positions can suffer 0.5-2% slippage.

    Slippage tiers (Indian mid/small cap empirical):
        Position/ADTV < 1%   → 0.05% base (+ volatility adjustment)
        Position/ADTV 1-5%   → 0.15% base
        Position/ADTV 5-10%  → 0.40% base
        Position/ADTV > 10%  → 1.00% base + warning

    Cap category multiplier:
        MEGA_CAP  → 0.5x  (highly liquid)
        LARGE_CAP → 0.7x
        MID_CAP   → 1.0x
        SMALL_CAP → 1.5x  (wider spreads, thinner order books)

    Volatility multiplier (>40% annualized = volatile day):
        ann_vol < 25%  → 0.8x
        ann_vol 25-40% → 1.0x
        ann_vol > 40%  → 1.3x

    Returns:
        (slippage_pct, warning_flag, detail_string)
    """
    if adtv_20 <= 0:
        return 2.0, True, "ADTV unavailable — assuming worst-case 2% slippage"

    pct_of_adtv = (position_value / adtv_20) * 100.0

    # Base slippage by position-size tier
    if pct_of_adtv < 1.0:
        base_slip = 0.05
    elif pct_of_adtv < 5.0:
        base_slip = 0.15
    elif pct_of_adtv < 10.0:
        base_slip = 0.40
    else:
        base_slip = 1.00  # Danger zone

    # Cap category multiplier
    cat_mult = {"MEGA_CAP": 0.5, "LARGE_CAP": 0.7, "MID_CAP": 1.0, "SMALL_CAP": 1.5}
    cat_factor = cat_mult.get(cap_category, 1.0)

    # Volatility multiplier
    if ann_vol_pct < 25.0:
        vol_factor = 0.8
    elif ann_vol_pct <= 40.0:
        vol_factor = 1.0
    else:
        vol_factor = 1.3

    slippage_pct = base_slip * cat_factor * vol_factor
    slippage_pct = round(slippage_pct, 3)

    warning_flag = slippage_pct > 0.50
    detail = f"{slippage_pct:.2f}% (pos/ADTV={pct_of_adtv:.1f}%, {cap_category}, vol={ann_vol_pct:.0f}%)"

    return slippage_pct, warning_flag, detail


def run_portfolio_construction(eligible_stocks, regime_status, pipeline_data, drawdown_status, portfolio_value=DEFAULT_PORTFOLIO_CAPITAL, date_str=None, base_risk_pct=None, existing_holdings=None):
    """Assembles the complete base portfolio, sizes trades in Core (76%) and Satellite (19%) buckets, and layers overlays/MTF leverage with advanced execution audit safeguards."""
    log_info("Executing SKILL 08, 09, 10, 11, 12: Structuring Core-Satellite portfolio allocation and risk...")
    
    high_corr_pairs_count = 0
    correlation_penalty_pct = 0.0
    
    if not date_str:
        date_str = datetime.date.today().strftime("%Y-%m-%d")
        
    # Load previous allocations DataFrame to extract prior position details
    df_prev = None
    try:
        out_dir_root = os.path.join(BASE_DIR, "output")
        if os.path.exists(out_dir_root):
            dates = sorted([d for d in os.listdir(out_dir_root) if os.path.isdir(os.path.join(out_dir_root, d)) and d < date_str])
            if dates:
                prev_date = dates[-1]
                for filename in ["L7_MAAC_Allocations.csv"]:
                    path_prev = os.path.join(out_dir_root, prev_date, filename)
                    if os.path.exists(path_prev):
                        df_prev = pd.read_csv(path_prev)
                        # Re-populate existing_holdings from previous allocations only if not already provided
                        if existing_holdings is None:
                            if "Allocation_%" in df_prev.columns:
                                existing_holdings = df_prev[df_prev["Allocation_%"] > 0]["Symbol"].tolist()
                            
                        # --- Filter out manual veto removals ---
                        veto_file = os.path.join(BASE_DIR, "Veto_add_remove.csv")
                        if os.path.exists(veto_file):
                            try:
                                v_df = pd.read_csv(veto_file)
                                v_last = v_df.drop_duplicates(subset=['Symbol'], keep='last')
                                removed = v_last[v_last['Action'] == 'VETO_REMOVE']['Symbol'].tolist()
                                existing_holdings = [s for s in existing_holdings if s not in removed]
                            except:
                                pass
                                
                        log_info(f"Loaded {len(existing_holdings)} existing holdings from {prev_date} ({filename}): {existing_holdings}")
                        break
    except Exception as e:
        log_warning(f"Portfolio constructor failed to load previous allocations: {e}")

    # Fetch NIFTY 50 history for RS Line computation
    nifty_df = pipeline_data.get("NIFTY_50")
    if nifty_df is None or nifty_df.empty:
        nifty_df = get_historical_data("NIFTY_50", end_date=date_str)

    # Force-retain all existing holdings with positive RS (RS Line > 0)
    retained_positions = []
    retained_symbols = set()
    
    core_retained_cash = 0.0
    satellite_retained_cash = 0.0
    
    # Track concentrations for sector/theme/cap category
    sector_weights = {}
    theme_weights = {}
    cap_category_weights = {}
    
    for symbol in existing_holdings:
        df_stock = pipeline_data.get(symbol)
        if df_stock is None or df_stock.empty:
            df_stock = get_historical_data(symbol, end_date=date_str)
            
        rs_val, rs_status = calculate_rs_line(symbol, df_stock, nifty_df)
        
        if rs_status == "HOLD":
            # Force-retain this position
            prev_row = None
            if df_prev is not None and not df_prev.empty:
                match_rows = df_prev[df_prev["Symbol"] == symbol]
                if not match_rows.empty:
                    prev_row = match_rows.iloc[0]
                    
            if prev_row is not None:
                prev_alloc = float(prev_row.get("Allocation_%", 0.0))
                entry_price = float(prev_row.get("Entry_Price", 0.0))
                stop_loss = float(prev_row.get("Stop_Loss", 0.0))
                bucket = str(prev_row.get("Bucket", "CORE" if prev_alloc > 2.0 else "SATELLITE"))
                sector = SECTORS.get(symbol)
                if not sector or sector == "Diversified":
                    sector = str(prev_row.get("Sector", "Diversified"))
                theme = THEMES.get(symbol)
                if not theme or theme == "Generic Theme":
                    theme = str(prev_row.get("Theme", "Generic Theme"))
                tier = str(prev_row.get("Tier", "TIER 2 — MEDIUM CONVICTION"))
                cap_cat = str(prev_row.get("Cap_Category", "LARGE_CAP"))
                primary_track = str(prev_row.get("Primary_Track", "NONE"))
                market_cap = float(prev_row.get("Market_Cap_Cr", 0.0))
                roe = float(prev_row.get("ROE", 15.0))
                de = float(prev_row.get("Debt_to_Equity", 0.0))
                chop_avg = float(prev_row.get("CHOP_avg", 50.0))
                weekly_chop = float(prev_row.get("Weekly_CHOP", 50.0))
                whipsaws = float(prev_row.get("Whipsaws_50d", 0.0))
                extension = float(prev_row.get("Extension_From_50DMA", 0.0))
            else:
                # Fallbacks
                prev_alloc = 4.0
                entry_price = float(df_stock["Close"].iloc[-1]) if df_stock is not None and not df_stock.empty else 100.0
                stop_loss = entry_price * 0.92
                bucket = "CORE"
                sector = SECTORS.get(symbol, "Diversified")
                theme = THEMES.get(symbol, "Generic Theme")
                tier = "TIER 2 — MEDIUM CONVICTION"
                cap_cat = "LARGE_CAP"
                primary_track = "NONE"
                market_cap = 0.0
                roe = 15.0
                de = 0.0
                chop_avg = 50.0
                weekly_chop = 50.0
                whipsaws = 0.0
                extension = 0.0
                
            today_close = float(df_stock["Close"].iloc[-1]) if df_stock is not None and not df_stock.empty else entry_price

            # ── PYRAMID ADD-TO-WINNER (Minervini / O'Neil / Livermore) ──────────────
            # Real legends ADD to positions as they keep proving strength.
            # If RS is rising and strong, we pyramid up — never average down.
            #   RS > 1.5  → boost allocation by 40% (Minervini: "load the truck")
            #   RS > 1.0  → boost by 25%   (O'Neil: add second tranche)
            #   RS > 0.6  → boost by 10%   (Livermore: small add on confirmation)
            #   RS ≤ 0.6  → hold at previous allocation flat
            market_regime_now = (regime_status or {}).get("market_regime", "SIDEWAYS")
            minervini_max = get_minervini_max_pct(market_regime_now)

            if rs_val > 1.5:
                pyramid_boost = 1.40
            elif rs_val > 1.0:
                pyramid_boost = 1.25
            elif rs_val > 0.60:
                pyramid_boost = 1.10
            else:
                pyramid_boost = 1.0

            prev_alloc_pyramided = min(prev_alloc * pyramid_boost, minervini_max)
            if prev_alloc_pyramided <= 0:
                prev_alloc_pyramided = prev_alloc

            # Recalculate quantity using the pyramided target allocation
            quantity = int((prev_alloc_pyramided / 100.0 * portfolio_value) / today_close)
            pos_value = quantity * today_close
            alloc_pct = (pos_value / portfolio_value) * 100.0
            
            stop_dist_pct = ((today_close - stop_loss) / today_close) * 100.0 if today_close > 0 else 8.0
            actual_risk = quantity * (today_close - stop_loss)
            actual_risk_pct = (actual_risk / portfolio_value) * 100.0
            
            atr_val = entry_price * 0.03
            natr_val = 3.0
            if df_stock is not None and not df_stock.empty and len(df_stock) >= 15:
                atr_val = float(calculate_atr(df_stock, 14).iloc[-1])
                natr_val = float(calculate_natr(df_stock, 14).iloc[-1])
                
            beta = compute_stock_beta(symbol, nifty_df, pipeline_data, date_str=date_str)
            vol = compute_stock_volatility(symbol, pipeline_data, date_str=date_str)
            
            pos_dict = {
                "Symbol": symbol,
                "Sector": sector,
                "Theme": theme,
                "Tier": tier,
                "Bucket": bucket,
                "Entry_Price": entry_price, # Keep original entry price
                "Quantity": quantity,
                "Position_Value": pos_value,
                "Allocation_Pct": alloc_pct,
                "Stop_Loss": stop_loss,
                "Stop_Distance_Pct": stop_dist_pct,
                "Actual_Risk": actual_risk,
                "Actual_Risk_Pct": actual_risk_pct,
                "NATR": natr_val,
                "ATR_14": atr_val,
                "Stop_Warning": False,
                "ROE": roe,
                "Debt_to_Equity": de,
                "CHOP_avg": chop_avg,
                "Weekly_CHOP": weekly_chop,
                "Whipsaws_50d": whipsaws,
                "Extension_From_50DMA": extension,
                "Beta": beta,
                "Volatility": vol,
                "Cap_Category": cap_cat,
                "Primary_Track": primary_track,
                "Market_Cap_Cr": market_cap,
                "Trim_Flag": "TRIM 25%" if extension > 25.0 else ""
            }
            retained_positions.append(pos_dict)
            retained_symbols.add(symbol)
            
            if bucket == "CORE":
                core_retained_cash += pos_value
            else:
                satellite_retained_cash += pos_value
                
            # Update weights for concentration checks
            sector_weights[sector] = sector_weights.get(sector, 0.0) + pos_value
            theme_weights[theme] = theme_weights.get(theme, 0.0) + pos_value
            cap_category_weights[cap_cat] = cap_category_weights.get(cap_cat, 0.0) + pos_value

    # 1. Compute Base Risk per trade scaled by Drawdown Governor
    regime_controls = get_regime_controls(regime_status)
    base_risk_mult = drawdown_status["risk_multiplier"] * regime_controls["risk_multiplier"]
    equity_cash_multiplier = regime_controls["cash_multiplier"]
    base_risk = base_risk_pct if base_risk_pct is not None else BASE_RISK_PER_TRADE_PCT
    
    # 2. Rebalancing Hysteresis and Macro Off-Switch (for new buys only)
    buys_raw = eligible_stocks[eligible_stocks["Entry_Eligible"] == True].copy()
    
    selected_buys = []
    stop_new_buys = regime_status.get("stop_new_buys", False)
    
    for _, row in buys_raw.iterrows():
        symbol = row["Symbol"]
        rank = row["Final_Rank"]
        
        # New buys only (exclude any symbol that is in existing_holdings)
        if symbol not in existing_holdings:
            score = float(row.get("Factor_Score", row.get("Weighted_Score", 0.0)))
            # New buy entry: score must be > 70 points (User Request)
            if score >= NEW_BUY_MIN_SCORE:
                cap_category = row.get("Cap_Category", "SMALL_CAP")
                
                # Check parent index bullishness
                index_bullish_flag = False
                if cap_category == "MEGA_CAP":
                    if regime_status.get("nifty50_bullish", True) or regime_status.get("niftynext50_bullish", True):
                        index_bullish_flag = True
                elif cap_category == "LARGE_CAP":
                    if regime_status.get("niftynext50_bullish", True):
                        index_bullish_flag = True
                elif cap_category == "MID_CAP":
                    if regime_status.get("nifty150_bullish", True):
                        index_bullish_flag = True
                elif cap_category == "SMALL_CAP":
                    if regime_status.get("nifty250_bullish", True):
                        index_bullish_flag = True
                
                sector_rank = row.get("sector_rank", 50.0)
                is_sector_bullish = sector_rank >= 50.0
                is_exceptional = row.get("is_exceptional_bull", False)
                
                # Allow buy if parent index is bullish, or sector is bullish, or it has independent strength
                allow_new_buy = index_bullish_flag or is_sector_bullish or is_exceptional
                
                if stop_new_buys:
                    log_warning(f"Macro Off-Switch (Global Breadth): Blocking new buy of {symbol} (Rank {rank}).")
                elif not allow_new_buy:
                    log_warning(f"Macro Off-Switch (Stock Level): Blocking new buy of {symbol} (Rank {rank}) - parent index is bearish, sector rank ({sector_rank:.1f}%) < 50%, and no exceptional momentum.")
                else:
                    selected_buys.append(row)
            else:
                pass
                    
    buys = pd.DataFrame(selected_buys) if selected_buys else pd.DataFrame(columns=buys_raw.columns)
    
    if equity_cash_multiplier <= 0.0:
        buys = buys.iloc[0:0].copy()
        
    portfolio_positions = []
    
    midsmall_df = pipeline_data.get("NIFTY_SMALLCAP_250")
    
    log_info(f"Retrieving volatility and risk parameters for {len(buys)} portfolio candidates...")
    
    # Filter to only allow new entries with Final_Rank <= TOP_N_STOCKS
    if not buys.empty and "Final_Rank" in buys.columns:
        buys = buys[buys["Final_Rank"] <= TOP_N_STOCKS].copy()

    # Dynamic Position Limits
    # Master Portfolio: VAM-GQ (20) + VAM-B (20) + Core (5) = 45
    # The core/sat split below is within the VAM-GQ quality-gated track:
    #   "core" = higher-rank VAM-GQ candidates allocated as base core
    #   "satellite" = lower-rank VAM-GQ candidates allocated as satellite
    max_open_pos = MAX_OPEN_POSITIONS
    core_limit = 5
    sat_limit = 40
    
    core_retained_positions = [p for p in retained_positions if p["Bucket"] == "CORE"]
    satellite_retained_positions = [p for p in retained_positions if p["Bucket"] == "SATELLITE"]
    
    core_slots_left = max(0, core_limit - len(core_retained_positions))
    sat_slots_left = max(0, sat_limit - len(satellite_retained_positions))
    
    buys_core = buys.head(core_slots_left).copy()
    buys_satellite = buys.iloc[core_slots_left:core_slots_left+sat_slots_left].copy()
    
    # ── CORE BUCKET (Max 76% Cash Allocation, 1% Risk Per Position) ──
    core_positions = list(core_retained_positions)
    core_cash_allocated = core_retained_cash
    
    new_core_positions = []
    new_core_cash_allocated = 0.0
    
    for _, row in buys_core.iterrows():
        symbol = row["Symbol"]
        entry = row["Close"]
        tier = row["Tier"]
        cap_cat = row.get("Cap_Category", "LARGE_CAP")
        primary_track = row.get("Primary_Track", "NONE")
        is_ipo_stock = row.get("is_ipo", False)
        
        # Get cap-category sizing parameters
        cat_limits = CAP_CATEGORY_LIMITS.get(cap_cat, CAP_CATEGORY_LIMITS["LARGE_CAP"])
        cap_risk_mult = cat_limits["risk_mult"]
        cap_max_single = cat_limits["max_single_pct"]
        cap_max_category = cat_limits["max_category_pct"]
        
        # Apply cyclical track risk reduction
        track_risk_mult = 0.75 if primary_track == "T1_CYCLICAL" else 1.0
        
        # Fetch history
        df_hist = None
        if pipeline_data:
            df_hist = pipeline_data.get(symbol)
        if df_hist is None:
            df_hist = get_historical_data(symbol, end_date=date_str)
        if df_hist is not None and not df_hist.empty and len(df_hist) >= 15:
            atr_val = float(calculate_atr(df_hist, 14).iloc[-1])
            natr_val = float(calculate_natr(df_hist, 14).iloc[-1])
            ma50_val = float(df_hist["Close"].ewm(span=50, adjust=False).mean().iloc[-1])
        else:
            atr_val = entry * 0.03
            natr_val = 3.0
            ma50_val = entry * 0.95
            
        initial_stop, stop_dist_pct, stop_warning = calculate_initial_stop(
            entry, atr_val, natr_val, tier, ma_50=ma50_val
        )
        
        # ── RS-TILT EQUAL-WEIGHT SIZING (Minervini Scale) ───────────────────────
        # RS Line is our sole exit — no price stop exists. Size from RS conviction.
        # Higher RS → larger initial weight. Minervini max cap applied by regime.
        rs_val_entry, _ = calculate_rs_line(symbol, df_hist, nifty_df)
        target_alloc_pct = rs_tilt_alloc_pct(rs_val_entry, cap_cat)

        # Apply cyclical track risk reduction (T1 Cyclical gets 75% of target)
        if track_risk_mult < 1.0:
            target_alloc_pct = round(target_alloc_pct * track_risk_mult, 2)

        # 50% Sizing Penalty for IPOs
        if is_ipo_stock:
            target_alloc_pct *= 0.5

        # Apply Minervini regime-based concentration cap (overrides cat cap for leaders)
        minervini_max = get_minervini_max_pct((regime_status or {}).get("market_regime", "SIDEWAYS"))
        if minervini_max > 0:
            target_alloc_pct = min(target_alloc_pct, minervini_max)

        position_value = portfolio_value * (target_alloc_pct / 100.0)
        raw_qty = position_value / entry

        # Stop distance kept for risk attribution reporting only (not for sizing)
        stop_dist_val = entry - initial_stop
        if stop_dist_val <= 0:
            stop_dist_val = entry * 0.08  # Notional 8% risk distance fallback

        # Check actual risk cap against base_risk
        max_qty_risk = (portfolio_value * base_risk) / stop_dist_val
        if raw_qty > max_qty_risk:
            raw_qty = max_qty_risk
            position_value = raw_qty * entry

        # Check single stock cap (category-aware)
        max_value_cap = portfolio_value * cap_max_single
        if is_ipo_stock:
            max_value_cap *= 0.5

        if position_value > max_value_cap:
            position_value = max_value_cap
            raw_qty = position_value / entry
            
        # 4. ADTV Liquidity Cap (Position <= 10% of 20-day ADTV)
        if df_hist is not None and not df_hist.empty:
            if pipeline_data and symbol in pipeline_data:
                adtv_20 = df_hist["Volume"].tail(20).mean()
            else:
                df_hist_copy = df_hist.copy()
                df_hist_copy["Volume"] = df_hist_copy["Close"] * df_hist_copy["Volume"]
                adtv_20 = df_hist_copy["Volume"].tail(20).mean()
        else:
            adtv_20 = 0.0
            
        if adtv_20 > 0:
            max_adtv_cap = adtv_20 * 0.10
            if position_value > max_adtv_cap:
                log_info(f"ADTV Sizing Cap: Capping position size of {symbol} at 10% of 20-day ADTV (₹{max_adtv_cap:,.2f}).")
                position_value = max_adtv_cap
                raw_qty = position_value / entry
                
        # Check category allocation cap
        current_cat_val = cap_category_weights.get(cap_cat, 0.0)
        max_cat_val = portfolio_value * cap_max_category
        if current_cat_val + position_value > max_cat_val:
            allowed_cat_val = max_cat_val - current_cat_val
            if allowed_cat_val <= 0:
                log_warning(f"{symbol}: {cap_cat} allocation cap reached. Skipping.")
                continue
            position_value = allowed_cat_val
            raw_qty = position_value / entry
            
        # Correlation limit check
        for pos in core_positions:
            corr_val = compute_stock_correlation(symbol, pos["Symbol"], pipeline_data, date_str=date_str)
            if corr_val > 0.80:
                high_corr_pairs_count += 1
                correlation_penalty_pct += 0.5  # Add 0.5% heat penalty per high corr pair
                log_warning(f"High Correlation ({corr_val:.2f}) between {symbol} and {pos['Symbol']}. Applying combined exposure constraints.")
                combined_value = pos["Position_Value"] + position_value
                max_combined_cap = portfolio_value * MAX_SINGLE_STOCK_ABS_PCT
                if combined_value > max_combined_cap:
                    allowed_value = max_combined_cap - pos["Position_Value"]
                    if allowed_value <= 0:
                        position_value = 0.0
                        raw_qty = 0.0
                    else:
                        position_value = allowed_value
                        raw_qty = position_value / entry
                        
        # 5. Sector and Industry Concentration Caps (Diversified = no cap; others 25%)
        sector = row.get("Sector", "Unknown")
        sector_pct_cap = 1.00 if "Diversified" in sector else 0.25
        current_sector_val = sector_weights.get(sector, 0.0)
        if current_sector_val + position_value > (portfolio_value * sector_pct_cap):
            log_warning(f"Sector Cap Exceeded: Skipping {symbol} (Sector: {sector}) to prevent exceeding {int(sector_pct_cap * 100)}% limit.")
            continue
            
        theme = row.get("Theme", "Generic Theme")
        current_theme_val = theme_weights.get(theme, 0.0)
        if current_theme_val + position_value > (portfolio_value * MAX_THEME_PCT):
            log_warning(f"Theme Cap Exceeded: Skipping {symbol} (Theme: {theme}) to prevent exceeding {MAX_THEME_PCT*100:.0f}% limit.")
            continue
            
        # Update weights
        sector_weights[sector] = sector_weights.get(sector, 0.0) + position_value
        theme_weights[theme] = theme_weights.get(theme, 0.0) + position_value
        cap_category_weights[cap_cat] = cap_category_weights.get(cap_cat, 0.0) + position_value
        
        final_qty = int(raw_qty)
        final_value = final_qty * entry
        actual_risk = final_qty * stop_dist_val
        actual_risk_pct = (actual_risk / portfolio_value) * 100.0
        
        if final_qty <= 0:
            continue
            
        new_core_cash_allocated += final_value
        beta = compute_stock_beta(symbol, midsmall_df, pipeline_data, date_str=date_str)
        vol = compute_stock_volatility(symbol, pipeline_data, date_str=date_str)
        
        new_core_positions.append({
            "Symbol": symbol,
            "Sector": sector,
            "Theme": theme,
            "Tier": tier,
            "Bucket": "CORE",
            "Entry_Price": entry,
            "Quantity": final_qty,
            "Position_Value": final_value,
            "Allocation_Pct": (final_value / portfolio_value) * 100.0,
            "Stop_Loss": initial_stop,
            "Stop_Distance_Pct": stop_dist_pct,
            "Actual_Risk": actual_risk,
            "Actual_Risk_Pct": actual_risk_pct,
            "NATR": natr_val,
            "ATR_14": atr_val,
            "Stop_Warning": stop_warning,
            "ROE": row["ROE"],
            "Debt_to_Equity": row["Debt_to_Equity"],
            "CHOP_avg": row["CHOP_avg"],
            "Weekly_CHOP": row["Weekly_CHOP"],
            "Whipsaws_50d": row["Whipsaws_50d"],
            "Extension_From_50DMA": row["Extension_From_50DMA"],
            "Beta": beta,
            "Volatility": vol,
            "Cap_Category": cap_cat,
            "Primary_Track": primary_track,
            "Market_Cap_Cr": row.get("Market_Cap_Cr", 0.0),
            "Trim_Flag": "TRIM 25%" if float(row.get("Extension_From_50DMA", 0.0)) > 25.0 else ""
        })
        
    # Scale only the new Core positions if total core cash exceeds max_core_cash limit
    max_core_cash = portfolio_value * CORE_ALLOCATION_PCT * equity_cash_multiplier
    available_core_cash = max(0.0, max_core_cash - core_retained_cash)
    if new_core_cash_allocated > available_core_cash:
        scale_ratio = available_core_cash / new_core_cash_allocated if new_core_cash_allocated > 0 else 0.0
        for pos in new_core_positions:
            pos["Quantity"] = int(pos["Quantity"] * scale_ratio)
            pos["Position_Value"] = pos["Quantity"] * pos["Entry_Price"]
            pos["Allocation_Pct"] = (pos["Position_Value"] / portfolio_value) * 100.0
            pos["Actual_Risk"] = pos["Quantity"] * (pos["Entry_Price"] - pos["Stop_Loss"])
            pos["Actual_Risk_Pct"] = (pos["Actual_Risk"] / portfolio_value) * 100.0
        new_core_positions = [p for p in new_core_positions if p["Quantity"] > 0]
        
    core_positions.extend(new_core_positions)
    core_cash_allocated = sum(p["Position_Value"] for p in core_positions)
        
    # ── SATELLITE BUCKET (Max 19% Cash Allocation, 0.25% Risk Per Position) ──
    satellite_positions = list(satellite_retained_positions)
    satellite_cash_allocated = satellite_retained_cash
    
    new_satellite_positions = []
    new_satellite_cash_allocated = 0.0
    
    for _, row in buys_satellite.iterrows():
        symbol = row["Symbol"]
        entry = row["Close"]
        tier = row["Tier"]
        cap_cat = row.get("Cap_Category", "LARGE_CAP")
        primary_track = row.get("Primary_Track", "NONE")
        is_ipo_stock = row.get("is_ipo", False)
        
        cat_limits = CAP_CATEGORY_LIMITS.get(cap_cat, CAP_CATEGORY_LIMITS["LARGE_CAP"])
        cap_risk_mult = cat_limits["risk_mult"]
        cap_max_category = cat_limits["max_category_pct"]
        
        track_risk_mult = 0.75 if primary_track == "T1_CYCLICAL" else 1.0
        
        df_hist = None
        if pipeline_data:
            df_hist = pipeline_data.get(symbol)
        if df_hist is None:
            df_hist = get_historical_data(symbol, end_date=date_str)
        if df_hist is not None and not df_hist.empty and len(df_hist) >= 15:
            atr_val = float(calculate_atr(df_hist, 14).iloc[-1])
            natr_val = float(calculate_natr(df_hist, 14).iloc[-1])
            ma50_val = float(df_hist["Close"].ewm(span=50, adjust=False).mean().iloc[-1])
        else:
            atr_val = entry * 0.03
            natr_val = 3.0
            ma50_val = entry * 0.95
            
        initial_stop, stop_dist_pct, stop_warning = calculate_initial_stop(
            entry, atr_val, natr_val, tier, ma_50=ma50_val
        )
        
        # ── RS-TILT SATELLITE SIZING ──────────────────────────────────────────────
        # Satellite positions are starter/speculative size. RS <= 0.30 gets 1.5%, others get 2%.
        rs_val_entry, _ = calculate_rs_line(symbol, df_hist, nifty_df)
        target_alloc_pct_sat = 1.5 if rs_val_entry <= 0.30 else 2.0

        if is_ipo_stock:
            target_alloc_pct_sat *= 0.5

        position_value = portfolio_value * (target_alloc_pct_sat / 100.0)
        raw_qty = position_value / entry

        # Stop distance kept for risk attribution reporting only (not for sizing)
        stop_dist_val = entry - initial_stop
        if stop_dist_val <= 0:
            stop_dist_val = entry * 0.08  # Notional 8% risk distance fallback

        # Check actual risk cap against base_risk (Satellite risk is 25% of base_risk)
        max_qty_risk = (portfolio_value * base_risk * 0.25) / stop_dist_val
        if raw_qty > max_qty_risk:
            raw_qty = max_qty_risk
            position_value = raw_qty * entry

        max_value_cap = portfolio_value * 0.02
        if is_ipo_stock:
            max_value_cap *= 0.5

        if position_value > max_value_cap:
            position_value = max_value_cap
            raw_qty = position_value / entry
            
        if df_hist is not None and not df_hist.empty:
            if pipeline_data and symbol in pipeline_data:
                adtv_20 = df_hist["Volume"].tail(20).mean()
            else:
                df_hist_copy = df_hist.copy()
                df_hist_copy["Volume"] = df_hist_copy["Close"] * df_hist_copy["Volume"]
                adtv_20 = df_hist_copy["Volume"].tail(20).mean()
        else:
            adtv_20 = 0.0
            
        if adtv_20 > 0:
            max_adtv_cap = adtv_20 * 0.10
            if position_value > max_adtv_cap:
                log_info(f"ADTV Sizing Cap: Capping position size of {symbol} at 10% of 20-day ADTV (₹{max_adtv_cap:,.2f}).")
                position_value = max_adtv_cap
                raw_qty = position_value / entry
                
        current_cat_val = cap_category_weights.get(cap_cat, 0.0)
        max_cat_val = portfolio_value * cap_max_category
        if current_cat_val + position_value > max_cat_val:
            allowed_cat_val = max_cat_val - current_cat_val
            if allowed_cat_val <= 0:
                continue
            position_value = allowed_cat_val
            raw_qty = position_value / entry
            
        for pos in (core_positions + satellite_positions):
            corr_val = compute_stock_correlation(symbol, pos["Symbol"], pipeline_data, date_str=date_str)
            if corr_val > 0.80:
                log_warning(f"High Correlation ({corr_val:.2f}) between {symbol} and {pos['Symbol']}. Applying combined exposure constraints.")
                combined_value = pos["Position_Value"].item() if hasattr(pos["Position_Value"], 'item') else pos["Position_Value"]
                combined_value += position_value
                max_combined_cap = portfolio_value * MAX_SINGLE_STOCK_ABS_PCT
                if combined_value > max_combined_cap:
                    allowed_value = max_combined_cap - pos["Position_Value"]
                    if allowed_value <= 0:
                        position_value = 0.0
                        raw_qty = 0.0
                    else:
                        position_value = allowed_value
                        raw_qty = position_value / entry
                        
        sector = row.get("Sector", "Unknown")
        sector_pct_cap = 1.00 if "Diversified" in sector else 0.25
        current_sector_val = sector_weights.get(sector, 0.0)
        if current_sector_val + position_value > (portfolio_value * sector_pct_cap):
            log_warning(f"Sector Cap Exceeded: Skipping {symbol} (Sector: {sector}) to prevent exceeding {int(sector_pct_cap * 100)}% limit.")
            continue
            
        theme = row.get("Theme", "Generic Theme")
        current_theme_val = theme_weights.get(theme, 0.0)
        if current_theme_val + position_value > (portfolio_value * MAX_THEME_PCT):
            log_warning(f"Theme Cap Exceeded: Skipping {symbol} (Theme: {theme}) to prevent exceeding {MAX_THEME_PCT*100:.0f}% limit.")
            continue
            
        sector_weights[sector] = sector_weights.get(sector, 0.0) + position_value
        theme_weights[theme] = theme_weights.get(theme, 0.0) + position_value
        cap_category_weights[cap_cat] = cap_category_weights.get(cap_cat, 0.0) + position_value
        
        final_qty = int(raw_qty)
        final_value = final_qty * entry
        actual_risk = final_qty * stop_dist_val
        actual_risk_pct = (actual_risk / portfolio_value) * 100.0
        
        if final_qty <= 0:
            continue
            
        new_satellite_cash_allocated += final_value
        beta = compute_stock_beta(symbol, midsmall_df, pipeline_data, date_str=date_str)
        vol = compute_stock_volatility(symbol, pipeline_data, date_str=date_str)
        
        new_satellite_positions.append({
            "Symbol": symbol,
            "Sector": sector,
            "Theme": theme,
            "Tier": tier,
            "Bucket": "SATELLITE",
            "Entry_Price": entry,
            "Quantity": final_qty,
            "Position_Value": final_value,
            "Allocation_Pct": (final_value / portfolio_value) * 100.0,
            "Stop_Loss": initial_stop,
            "Stop_Distance_Pct": stop_dist_pct,
            "Actual_Risk": actual_risk,
            "Actual_Risk_Pct": actual_risk_pct,
            "NATR": natr_val,
            "ATR_14": atr_val,
            "Stop_Warning": stop_warning,
            "ROE": row["ROE"],
            "Debt_to_Equity": row["Debt_to_Equity"],
            "CHOP_avg": row["CHOP_avg"],
            "Weekly_CHOP": row["Weekly_CHOP"],
            "Whipsaws_50d": row["Whipsaws_50d"],
            "Extension_From_50DMA": row["Extension_From_50DMA"],
            "Beta": beta,
            "Volatility": vol,
            "Cap_Category": cap_cat,
            "Primary_Track": primary_track,
            "Market_Cap_Cr": row.get("Market_Cap_Cr", 0.0),
            "Trim_Flag": "TRIM 25%" if float(row.get("Extension_From_50DMA", 0.0)) > 25.0 else ""
        })
        
    # Scale only the new Satellite positions if total satellite cash exceeds max_satellite_cash limit
    # Satellite bucket capped at ACTIVE_ALLOCATION_PCT (35%) of portfolio value per config.
    # MTF leverage, if enabled by regime, is applied as a SEPARATE overlay — not hidden
    # inside the satellite allocation bucket.
    from config import ACTIVE_ALLOCATION_PCT
    max_satellite_cash = portfolio_value * ACTIVE_ALLOCATION_PCT * equity_cash_multiplier
    available_satellite_cash = max(0.0, max_satellite_cash - satellite_retained_cash)
    if new_satellite_cash_allocated > available_satellite_cash:
        scale_ratio = available_satellite_cash / new_satellite_cash_allocated if new_satellite_cash_allocated > 0 else 0.0
        for pos in new_satellite_positions:
            pos["Quantity"] = int(pos["Quantity"] * scale_ratio)
            pos["Position_Value"] = pos["Quantity"] * pos["Entry_Price"]
            pos["Allocation_Pct"] = (pos["Position_Value"].item() if hasattr(pos["Position_Value"], 'item') else pos["Position_Value"]) / portfolio_value * 100.0
            pos["Actual_Risk"] = pos["Quantity"] * (pos["Entry_Price"] - pos["Stop_Loss"])
            pos["Actual_Risk_Pct"] = (pos["Actual_Risk"] / portfolio_value) * 100.0
        new_satellite_positions = [p for p in new_satellite_positions if p["Quantity"] > 0]
        
    satellite_positions.extend(new_satellite_positions)
    satellite_cash_allocated = sum(p["Position_Value"] for p in satellite_positions)
    
    # Combine positions
    portfolio_positions = core_positions + satellite_positions

    # Drawdown and regime states can force risk down even when stock-level signals still pass.
    drawdown_action = drawdown_status.get("action", "NORMAL")
    market_regime = regime_controls["market_regime"]
    
    # We apply these risk filters ONLY to new buys!
    filtered_new_positions = new_core_positions + new_satellite_positions
    if drawdown_action == "FULL STOP — SYSTEM PAUSE" or market_regime in {"BEAR", "CRISIS"}:
        filtered_new_positions = []
    elif drawdown_action == "DEFENSIVE":
        filtered_new_positions = [p for p in filtered_new_positions if p["Tier"] == "TIER 1 — HIGH CONVICTION"]
    elif drawdown_action == "REDUCE" or market_regime == "CORRECTION":
        filtered_new_positions = [p for p in filtered_new_positions if p["Tier"] != "TIER 3 — LOW CONVICTION"]
        
    portfolio_positions = retained_positions + filtered_new_positions
    
    # ── TIER CONCENTRATION BALANCE AUDIT ───────────────────────
    if len(portfolio_positions) > 0:
        tier1_positions = [pos for pos in portfolio_positions if pos["Tier"] == "TIER 1 — HIGH CONVICTION"]
        t1_count = len(tier1_positions)
        max_allowed_total = int(t1_count / 0.40)
        
        while len(portfolio_positions) > max_allowed_total:
            pruned = False
            for idx in range(len(portfolio_positions) - 1, -1, -1):
                # Skip retained symbols during pruning audits
                if portfolio_positions[idx]["Symbol"] in retained_symbols:
                    continue
                if portfolio_positions[idx]["Tier"] == "TIER 3 — LOW CONVICTION":
                    log_warning(f"Concentration Audit: Pruning Tier 3 position {portfolio_positions[idx]['Symbol']} to meet the 40% Tier 1 minimum threshold.")
                    portfolio_positions.pop(idx)
                    pruned = True
                    break
            
            if not pruned:
                for idx in range(len(portfolio_positions) - 1, -1, -1):
                    # Skip retained symbols during pruning audits
                    if portfolio_positions[idx]["Symbol"] in retained_symbols:
                        continue
                    if portfolio_positions[idx]["Tier"] != "TIER 1 — HIGH CONVICTION":
                        log_warning(f"Concentration Audit: Pruning Tier 2 position {portfolio_positions[idx]['Symbol']} to meet the 40% Tier 1 minimum threshold.")
                        portfolio_positions.pop(idx)
                        pruned = True
                        break
            if not pruned:
                break
    
    # ── CALCULATE PORTFOLIO CORRELATION MATRIX ────────────────
    portfolio_symbols = [pos["Symbol"] for pos in portfolio_positions]
    n_assets = len(portfolio_symbols)
    
    correlation_penalty = 0.0
    high_corr_pairs = 0
    
    if n_assets > 0:
        data = np.zeros((n_assets, n_assets))
        for i in range(n_assets):
            data[i, i] = 1.0
            for j in range(i + 1, n_assets):
                s1 = portfolio_symbols[i]
                s2 = portfolio_symbols[j]
                c_val = compute_stock_correlation(s1, s2, pipeline_data, date_str=date_str)
                data[i, j] = c_val
                data[j, i] = c_val
                
                if c_val > 0.70:
                    high_corr_pairs += 1
                    correlation_penalty += 0.005
        corr_matrix_df = pd.DataFrame(data, index=portfolio_symbols, columns=portfolio_symbols)
        
        out_dir = os.path.join(BASE_DIR, "output", date_str)
        os.makedirs(out_dir, exist_ok=True)
        try:
            corr_matrix_df.to_csv(os.path.join(out_dir, "Portfolio_Correlation_Matrix.csv"))
        except PermissionError:
            log_warning("Permission denied writing to Portfolio_Correlation_Matrix.csv. Writing to backup: Portfolio_Correlation_Matrix_LOCKED.csv")
            try:
                corr_matrix_df.to_csv(os.path.join(out_dir, "Portfolio_Correlation_Matrix_LOCKED.csv"))
            except Exception as e:
                log_error(f"Failed to write backup correlation matrix: {e}")
        log_success(f"Exported Portfolio Correlation Matrix for {n_assets} assets.")
        
    # ── PRECIOUS METALS OVERLAY (SKILL 11) ───────────────────────
    gold_df = pipeline_data.get("MCX_GOLD")
    silver_df = pipeline_data.get("MCX_SILVER")
    metals_results = check_metals_overlay_regime(gold_df, silver_df)
    
    if drawdown_status["action"] == "FULL STOP — SYSTEM PAUSE":
        metals_results["total_alloc_pct"] = 0.0
        metals_results["gold_alloc_pct"] = 0.0
        metals_results["silver_alloc_pct"] = 0.0
        
    # ── MTF LEVERAGE ENGINE (SKILL 12) ───────────────────────────
    midsmall_df = pipeline_data.get("NIFTY_SMALLCAP_250")
    if midsmall_df is not None and not midsmall_df.empty:
        ms_close = float(midsmall_df["Close"].iloc[-1])
        ms_ema_col = "EMA_150" if "EMA_150" in midsmall_df.columns else "SMA_150"
        ms_ema = float(midsmall_df[ms_ema_col].iloc[-1]) if ms_ema_col in midsmall_df.columns else ms_close
        above_150 = ms_close > ms_ema
        rising_150 = True
        if len(midsmall_df) >= 10:
            rising_150 = float(midsmall_df[ms_ema_col].iloc[-1]) > float(midsmall_df[ms_ema_col].iloc[-10])
        midsmall_bullish = above_150 and rising_150
    else:
        midsmall_bullish = False
        
    mtf_alloc_pct = 25.0 if (midsmall_bullish and regime_controls["mtf_allowed"]) else 0.0
    
    if drawdown_status["action"] in {"FULL STOP — SYSTEM PAUSE", "DEFENSIVE", "REDUCE"}:
        mtf_alloc_pct = 0.0
        
    leverage_penalty = 0.01 * (mtf_alloc_pct / 25.0)
    
    # All Bearish Check: 95% cash + 5% liquid bees only
    all_bearish = (not midsmall_bullish) and (not metals_results["gold_pass"]) and (not metals_results["silver_pass"])
    if all_bearish and equity_cash_multiplier <= 0.0:
        portfolio_positions = retained_positions
        
    if not portfolio_positions:
        base_heat = 0.0
        portfolio_heat = 0.0
        correlation_penalty = 0.0
        leverage_penalty = 0.0
        high_corr_pairs = 0
        active_core_pct = 0.0
        active_satellite_pct = 0.0
        active_vam_gq_pct = 0.0
    else:
        base_heat = sum(pos["Actual_Risk_Pct"] for pos in portfolio_positions)
        portfolio_heat = base_heat + (correlation_penalty * 100.0) + (leverage_penalty * 100.0)
        active_core_pct = sum(pos["Allocation_Pct"] for pos in portfolio_positions if pos["Bucket"] == "CORE")
        active_satellite_pct = sum(pos["Allocation_Pct"] for pos in portfolio_positions if pos["Bucket"] == "SATELLITE")
        active_vam_gq_pct = active_core_pct + active_satellite_pct

    # ── SEBI margin checker ──────────────────────────────────────
    # In BULL / EARLY_BULL: reduce liquid bees to 3% — Minervini stays fully invested.
    # In all other regimes: keep standard 5% SEBI buffer.
    _regime_for_bees = regime_controls.get("market_regime", "SIDEWAYS")
    bees_pct = 0.03 if _regime_for_bees in {"BULL", "EARLY_BULL"} else 0.05
    bees_value = portfolio_value * bees_pct

    non_cash_collateral = (active_vam_gq_pct / 100.0) * portfolio_value * (1.0 - EQUITY_HAIRCUT)
    cash_equiv_collateral = bees_value * (1.0 - LIQUID_BONDS_HAIRCUT)

    overlay_exposure_pct = metals_results["total_alloc_pct"]
    fo_margin_required = portfolio_value * (overlay_exposure_pct / 100.0) * METALS_MARGIN_REQ

    cash_required_50_50 = fo_margin_required * 0.50
    sebi_compliant = cash_equiv_collateral >= cash_required_50_50

    portfolio_blueprint = {
        "active_core_equities_pct": active_core_pct,
        "active_satellite_equities_pct": active_satellite_pct,
        "active_vam_gq_equities_pct": active_vam_gq_pct,
        "liquid_bees_pct": bees_pct * 100.0,          # Regime-adjusted LIQUIDBEES allocation %
        "liquid_bonds_pct": bees_pct * 100.0,          # Alias used by dashboard donut chart
        "mtf_leverage_pct": mtf_alloc_pct,
        "gold_futures_pct": metals_results["gold_alloc_pct"],
        "silver_futures_pct": metals_results["silver_alloc_pct"],
        "total_exposure_pct": active_vam_gq_pct + mtf_alloc_pct + (metals_results["total_alloc_pct"].item() if hasattr(metals_results["total_alloc_pct"], 'item') else metals_results["total_alloc_pct"]) + (bees_pct * 100.0),
        "cash_pct": max(0.0, 100.0 - active_vam_gq_pct - mtf_alloc_pct - (metals_results["total_alloc_pct"].item() if hasattr(metals_results["total_alloc_pct"], 'item') else metals_results["total_alloc_pct"]) - (bees_pct * 100.0)),
        "portfolio_value": portfolio_value,
        "liquid_bees_rupees": bees_value,
        "cash_rupees": max(0.0, portfolio_value * (1.0 - (active_vam_gq_pct + mtf_alloc_pct + (metals_results["total_alloc_pct"].item() if hasattr(metals_results["total_alloc_pct"], 'item') else metals_results["total_alloc_pct"]) + (bees_pct * 100.0)) / 100.0)),
        "core_rupees": (active_core_pct / 100.0) * portfolio_value,
        "satellite_rupees": (active_satellite_pct / 100.0) * portfolio_value,
        "gold_rupees": (metals_results["gold_alloc_pct"] / 100.0) * portfolio_value,
        "silver_rupees": (metals_results["silver_alloc_pct"] / 100.0) * portfolio_value,
        "mtf_rupees": (mtf_alloc_pct / 100.0) * portfolio_value,
        "market_regime": regime_controls["market_regime"],
        "regime_risk_multiplier": regime_controls["risk_multiplier"],
        "regime_equity_cash_multiplier": equity_cash_multiplier,
        "portfolio_heat_pct": portfolio_heat,
        "base_heat_pct": base_heat,
        "correlation_penalty_pct": correlation_penalty * 100.0,
        "leverage_penalty_pct": leverage_penalty * 100.0,
        "high_corr_pairs_count": high_corr_pairs,
        "non_cash_collateral": non_cash_collateral,
        "cash_equiv_collateral": cash_equiv_collateral,
        "fo_margin_required": fo_margin_required,
        "sebi_compliant": sebi_compliant,
        "sebi_shortfall_pct": max(0.0, (cash_required_50_50 - cash_equiv_collateral) / max(cash_required_50_50, 1.0) * 100.0),
        "excess_cash_margin": cash_equiv_collateral - cash_required_50_50
    }
    
    log_success("Portfolio structuring and sizing completed successfully.")
    return portfolio_positions, portfolio_blueprint
