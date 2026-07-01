import json
import os
import pandas as pd
import numpy as np
import datetime
from screener_fetcher import (
    fetch_screener_fundamentals, get_dynamic_sector, get_dynamic_theme,
    fetch_chartink_universe, fetch_market_cap, fetch_delivery_data
)
from utils import log_info, log_success, log_warning
from cache_manager import get_historical_data
from pipeline_data import get_up_down_volume_ratio, calculate_adx, calculate_obv, calculate_chop_index, calculate_atr, calculate_natr
from config import (
    FACTOR_THRESHOLDS, SELECTION_FILTER_MODE, MIN_FACTOR_SCORE, ACTIVE_THEMES,
    MIN_MARKET_CAP_CR, CYCLICAL_SECTOR_KEYWORDS,
    MIN_ADTV_CR, MIN_ADX_14, MAX_ANNUAL_VOL_PCT,
    QUALITY_GATE_BFSI, QUALITY_GATE_STANDARD,
    FACTOR_WEIGHTS, TOP_N_STOCKS
)


# ── ASM/GSM SURVEILLANCE LIST HELPER ──────────────────────────
# NSE publishes ASM/GSM lists at:
# https://www.nseindia.com/regulations/market-surveillance
# Download as CSV and save as asm_gsm_stocks.csv in project root
# One column, one symbol per row (no header required)
_ASM_GSM_CACHE = None
_ASM_GSM_PATH = None
_ASM_GSM_CACHE_FILE = None  # backup cache for fail-closed


def _get_asm_gsm_paths():
    global _ASM_GSM_PATH, _ASM_GSM_CACHE_FILE
    if _ASM_GSM_PATH is None:
        base = os.path.dirname(os.path.abspath(__file__))
        _ASM_GSM_PATH = os.path.join(base, "asm_gsm_stocks.csv")
        _ASM_GSM_CACHE_FILE = os.path.join(base, "asm_gsm_stocks.cache.json")
    return _ASM_GSM_PATH, _ASM_GSM_CACHE_FILE


def _save_asm_gsm_cache(symbols: set):
    """Save successfully loaded ASM/GSM list to a cache file for fail-closed fallback."""
    _, cache_path = _get_asm_gsm_paths()
    try:
        with open(cache_path, "w") as f:
            json.dump(list(symbols), f)
    except Exception:
        pass


def _load_asm_gsm_cache() -> set:
    """Load backup cache from last successful run."""
    _, cache_path = _get_asm_gsm_paths()
    try:
        if os.path.exists(cache_path):
            with open(cache_path) as f:
                data = json.load(f)
            return set(data)
    except Exception:
        pass
    return None  # No cache available


def _load_asm_gsm_list() -> set:
    """Load ASM/GSM stock symbols from local CSV. Cached after first load.

    FAIL-SAFE / FAIL-OPEN design (allowing empty list):
    - If CSV exists:
        - If it parses successfully -> use it (even if empty, 0 symbols)
        - If it is empty/only comments (raises EmptyDataError) -> use empty set (0 symbols under surveillance)
        - If there is another error parsing it -> try backup cache, else log warning and fail-open (0 symbols)
    - If CSV is missing:
        - Try backup cache from last run
        - If no backup cache exists -> log warning and fail-open (0 symbols under surveillance)
    """
    global _ASM_GSM_CACHE, _ASM_GSM_PATH, _ASM_GSM_CACHE_FILE
    if _ASM_GSM_CACHE is not None:
        return _ASM_GSM_CACHE

    asm_path, cache_path = _get_asm_gsm_paths()

    # Try primary CSV
    if os.path.exists(asm_path):
        try:
            try:
                df = pd.read_csv(asm_path, header=None, comment="#")
                df = df.dropna(how="all")
                if not df.empty:
                    symbols = set(df.iloc[:, 0].astype(str).str.strip().str.upper())
                else:
                    symbols = set()
            except Exception as e:
                # Handle pd.errors.EmptyDataError or empty files safely
                symbols = set()
                
            _ASM_GSM_CACHE = symbols
            log_info(f"Loaded {len(symbols)} ASM/GSM stocks from {asm_path}")
            _save_asm_gsm_cache(symbols)
            return symbols
        except Exception as e:
            log_warning(f"ASM/GSM CSV parse failed: {e}")

    # Fallback: try backup cache
    cached = _load_asm_gsm_cache()
    if cached is not None:
        log_warning(f"ASM/GSM primary list unavailable — using backup cache ({len(cached)} stocks)")
        _ASM_GSM_CACHE = cached
        return cached

    # Fail-closed: block all stocks if both primary and backup fail
    log_warning("🚨 ASM/GSM list is unavailable and no backup cache exists. SYSTEM FAIL-CLOSED: All new entries blocked.")
    _ASM_GSM_CACHE = {"__ALL_BLOCKED__"}
    return _ASM_GSM_CACHE


def _is_asm_gsm_stock(symbol: str) -> bool:
    """Check if a stock is under NSE ASM/GSM surveillance or data is unavailable."""
    asm_set = _load_asm_gsm_list()
    if "__ALL_BLOCKED__" in asm_set:
        return True  # Fail-closed: reject everything if no data
    return symbol.upper() in asm_set


# ── QUALITY HARD GATE ────────────────────────────────────────────────────────

def _apply_quality_gate(symbol: str, f_data: dict, is_financial: bool, sector: str = ""):
    """
    TWO-TRACK hard elimination quality gate.

    Routes BFSI stocks through QUALITY_GATE_BFSI (Net NPA, CAR, ROA, Promoter Pledge)
    and non-BFSI stocks through QUALITY_GATE_STANDARD (D/E, 3Yr CFO/PAT, ROCE, Promoter Pledge).

    ROCE check is WAIVED for cyclical sectors (metals, infra, defence, power, etc.)
    because momentum systems buy cyclical recoveries precisely BEFORE ROCE normalises.
    D/E and CFO/PAT checks still apply to cyclicals.

    Returns:
        (passed: bool, rejection_reason: str)

    DESIGN NOTE: This gate is NON-BYPASSABLE. Neither is_exceptional_bull nor any
    other momentum override can skip Quality. A stock must prove structural financial
    health before entering the ranking system.
    """
    pledge = f_data.get("Promoter_Pledge_%", 0.0)
    sector_lower = sector.lower()

    if is_financial:
        # ── BFSI PATH ─────────────────────────────────────────────
        net_npa  = f_data.get("Net_NPA_%", None)
        car      = f_data.get("CAR_%", None)
        roa      = f_data.get("ROA_%", None)
        pcr      = f_data.get("PCR_%", None)
        casa     = f_data.get("CASA_%", None)
        gate     = QUALITY_GATE_BFSI

        failures = []
        if net_npa is None:
            failures.append("Missing Data: Net NPA")
        elif net_npa >= gate["Net_NPA_max_pct"]:
            failures.append(f"Net NPA {net_npa:.2f}% >= {gate['Net_NPA_max_pct']}%")
            
        if car is None:
            failures.append("Missing Data: CAR")
        elif car <= gate["CAR_min_pct"]:
            failures.append(f"CAR {car:.1f}% <= {gate['CAR_min_pct']}% (RBI floor 11.5%)")
            
        if roa is None:
            failures.append("Missing Data: ROA")
        elif roa <= gate["ROA_min_pct"]:
            failures.append(f"ROA {roa:.2f}% <= {gate['ROA_min_pct']}%")
            
        if pcr is not None and pcr != 0.0 and pcr <= gate.get("PCR_min_pct", 70.0):
            failures.append(f"PCR {pcr:.1f}% <= {gate.get('PCR_min_pct', 70.0)}%")
            
        if casa is not None and casa != 0.0 and casa <= gate.get("CASA_min_pct", 35.0):
            failures.append(f"CASA {casa:.1f}% <= {gate.get('CASA_min_pct', 35.0)}%")
            
        if pledge is None:
            failures.append("Missing Data: Promoter Pledge")
        elif pledge >= gate["Promoter_Pledge_max_pct"]:
            failures.append(f"Promoter Pledge {pledge:.1f}% >= {gate['Promoter_Pledge_max_pct']}%")

        if failures:
            return False, "[BFSI Quality Gate] " + " | ".join(failures)
        return True, ""

    else:
        # ── NON-BFSI PATH ───────────────────────────────────────────
        de           = f_data.get("Debt_to_Equity", None)
        cfo_pat_3yr  = f_data.get("CFO_PAT_3Yr_Avg", None)
        roce         = f_data.get("ROCE_3Yr_Avg", None)
        gate         = QUALITY_GATE_STANDARD

        # Is this stock in a cyclical sector? (Track 1 targets)
        is_cyclical = any(kw in sector_lower for kw in CYCLICAL_SECTOR_KEYWORDS)

        failures = []
        if de is None:
            failures.append("Missing Data: Debt to Equity")
        elif de >= gate["DE_max"]:
            failures.append(f"D/E {de:.2f} >= {gate['DE_max']}")
            
        # 3-Year Avg CFO/PAT > 0: positive cash generation on average
        if cfo_pat_3yr is None:
            failures.append("Missing Data: 3Yr CFO/PAT")
        elif cfo_pat_3yr <= gate["CFO_PAT_3Yr_min"]:
            failures.append(f"3Yr CFO/PAT {cfo_pat_3yr:.2f} <= {gate['CFO_PAT_3Yr_min']} (persistently negative cash generation)")
            
        # ROCE check: waived for cyclical sectors — they're bought before ROCE normalises
        if roce is None:
            failures.append("Missing Data: ROCE")
        elif not is_cyclical and roce <= gate["ROCE_min_pct"]:
            failures.append(f"ROCE {roce:.1f}% <= {gate['ROCE_min_pct']}%")
        elif is_cyclical and roce <= gate["ROCE_min_pct"]:
            log_info(f"  [{symbol}] ROCE {roce:.1f}% below {gate['ROCE_min_pct']}% — WAIVED (cyclical sector: {sector})")
            
        if pledge is None:
            failures.append("Missing Data: Promoter Pledge")
        elif pledge >= gate["Promoter_Pledge_max_pct"]:
            failures.append(f"Promoter Pledge {pledge:.1f}% >= {gate['Promoter_Pledge_max_pct']}%")

        if failures:
            return False, "[Quality Gate] " + " | ".join(failures)
        return True, ""


# ── REJECTED RECORD HELPER ───────────────────────────────────────────────

def make_rejected_record(symbol, close, volume, reason, sector="Diversified", theme="General Momentum", screeners="", mcap_cr=0.0, cap_category="BELOW_MIN", is_ipo=False):
    """Generates a fully populated rejected stock record to maintain column schema uniformity."""
    try:
        f_data = fetch_screener_fundamentals(symbol)
    except Exception:
        f_data = None
        
    if not f_data:
        f_data = {}
        
    return {
        "Symbol": symbol,
        "Close": close,
        "Volume": volume,
        "Composite_Score": 0.0,
        "Opportunity_Score": 0.0,
        "Tier": "REJECTED",
        "Stage": 0,
        "Entry_Eligible": False,
        "ATR_14": 0.0,
        "NATR_14": 0.0,
        "NATR_Trend": "UNKNOWN",
        "ADX_14": 0.0,
        "ADX_Bullish": False,
        "OBV_Rising": False,
        "RS_vs_Nifty50": 0.0,
        "Independent_Alpha_Pass": False,
        "Emerging_Recovery_Aligned": False,
        "Up_Down_Vol_Ratio": 0.0,
        "Extension_Score": 0.0,
        "Extension_From_50DMA": 0.0,
        "CHOP_avg": 35.0,
        "Weekly_CHOP": 35.0,
        "Whipsaws_50d": 0,
        "ROE": f_data.get("ROE_%", 0.0),
        "Debt_to_Equity": f_data.get("Debt_to_Equity", 0.0),
        "Sector": sector,
        "Theme": theme,
        "Screeners": screeners,
        "Rejection_Reason": reason,
        "Data_Source": f_data.get("Data_Source", "N/A"),
        "Data_Flags": f_data.get("Data_Flags", "None"),
        "Delivery_Below_Threshold": False,
        "Market_Cap_Cr": mcap_cr,
        "Cap_Category": cap_category,
        "Factors_Passed": "",
        "Factor_Count": 0,
        "Factor_Score": 0,
        "Factor_Details": "{}",
        "Delivery_Pct": 0.0,
        "Final_Rank": 999,
        "is_ipo": is_ipo,
        "is_exceptional_bull": False,
        "is_vcp_setup": False,
        "stage2_aligned": False,
        "sma_50": 0.0,
        "sma_200": 0.0,
        "high_52w": 0.0,
        "low_52w": 0.0,
        "sales_growth": 0.0,
        "profit_growth": 0.0,
        "roce_3yr": 0.0
    }


# ── 7-FACTOR SELECTION FRAMEWORK EVALUATOR ────────────────────────────────

def score_7_factors(data):
    """
    Unified Institutional Scoring Engine — Pure Percentile Ranking.
    
    Replaces absolute capping with cross-sectional and sector-neutral percentiles.
    Returns:
        (factor_scores_dict, factor_details_dict, weighted_score)
    """
    factor_scores = {}
    details = {}

    # ── F1: Sectoral Trend (15%) ─────────────────────────────────────────
    f1_score = float(data.get("sector_rank", 50.0))
    factor_scores["F1_SECTORAL_TREND"] = f1_score
    details["F1_SECTORAL_TREND"] = {
        "score": round(f1_score, 1),
        "reason": f"Sector Percentile Rank: {f1_score:.1f}/100"
    }

    # ── F2: Thematic Trend (9%) ─────────────────────────────────────────
    theme = data.get("theme", "General Momentum")
    f2_score = float(data.get("theme_rank", 50.0))
    factor_scores["F2_THEMATIC_TREND"] = f2_score
    details["F2_THEMATIC_TREND"] = {
        "score": round(f2_score, 1),
        "reason": f"Theme '{theme}' | Percentile Rank: {f2_score:.1f}/100"
    }

    # ── F3: Momentum (38%) ────────────────────────────────────────────────
    # Uses pure Clenow Volatility-Adjusted Rank
    f3_score = float(data.get("rank_mom", 50.0))
    factor_scores["F3_MOMENTUM"] = f3_score
    details["F3_MOMENTUM"] = {
        "score": round(f3_score, 1),
        "reason": f"Clenow Momentum Percentile Rank: {f3_score:.1f}/100"
    }

    # ── F4: Growth (13%) ─────────────────────────────────────────────────
    # Uses Sector-Neutral Fundamental Rank
    f4_score = float(data.get("rank_growth", 50.0))
    factor_scores["F4_GROWTH"] = f4_score
    details["F4_GROWTH"] = {
        "score": round(f4_score, 1),
        "reason": f"Sector-Neutral Growth Percentile: {f4_score:.1f}/100"
    }

    # ── F5: Quality (NOW A HARD GATE — NOT SCORED) ────────────────────────
    pass

    # ── F6: Delivery Confirmation (12%) ───────────────────────────────────
    # Uses cross-sectional rank of delivery accumulation
    f6_score = float(data.get("rank_delivery", 50.0))
    factor_scores["F6_DELIVERY_CONFIRMATION"] = f6_score
    details["F6_DELIVERY_CONFIRMATION"] = {
        "score": round(f6_score, 1),
        "reason": f"Delivery Accumulation Percentile: {f6_score:.1f}/100"
    }

    # ── F7: PEAD (6%) ───────────────────────────────────────────────────
    f7_score = float(data.get("rank_pead", 50.0))
    factor_scores["F7_PEAD"] = f7_score
    details["F7_PEAD"] = {
        "score": round(f7_score, 1),
        "reason": f"PEAD Catalyst Percentile: {f7_score:.1f}/100"
    }

    # ── F8: FII/DII Conviction (7%) ──────────────────────────────────────
    f8_score = float(data.get("rank_fii", 50.0))
    factor_scores["F8_FII_DII_CONVICTION"] = f8_score
    details["F8_FII_DII_CONVICTION"] = {
        "score": round(f8_score, 1),
        "reason": f"Institutional Flow Percentile: {f8_score:.1f}/100"
    }

    # ── FINAL WEIGHTED SCORE ──────────────────────────────────────────────
    weighted_score = sum(
        factor_scores[k] * FACTOR_WEIGHTS[k]
        for k in FACTOR_WEIGHTS
        if k in factor_scores
    )
    weighted_score = min(100.0, max(0.0, weighted_score))

    return factor_scores, details, weighted_score


# ── LEGACY COMPATIBILITY WRAPPER ─────────────────────────────────────────
def evaluate_7_factors(data):
    """
    Legacy wrapper — kept so dashboard/other callers don't break.
    Delegates to score_7_factors() and returns (passed_factors, details, factor_count).
    'passed_factors' = factors with score >= 50 (i.e., above midpoint).
    """
    factor_scores, details, weighted_score = score_7_factors(data)
    # Convert to legacy format: passed = factors above 50/100 score
    passed = [k for k, v in factor_scores.items() if v >= 50.0]
    # Convert details to legacy format (add 'passed' key)
    legacy_details = {}
    for k, v in details.items():
        legacy_details[k] = {
            "passed": factor_scores[k] >= 50.0,
            "score": v["score"],
            "reason": v["reason"]
        }
    return passed, legacy_details, len(passed)




def run_stock_selection(pipeline_data=None, sector_ranks=None, theme_ranks=None, date_str=None, chartink_universe=None):
    """
    Ingests pre-qualified stocks from Chartink screeners, evaluates each through
    7 sequential elimination filters (Hard Gates), computes 4 Alpha Scans,
    percentile-ranks them, and sorts/allocates Tiers.
    """
    log_info("Executing 11-Step stock selection and ranking engine...")

    # 1. Fetch Chartink universe if not already provided
    if chartink_universe is None:
        chartink_universe = fetch_chartink_universe(date_str)

    # Veto Cooldown Filter: Exclude manually removed stocks for 10 days
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    veto_file = os.path.join(BASE_DIR, "Veto_add_remove.csv")
    if os.path.exists(veto_file):
        try:
            v_df = pd.read_csv(veto_file)
            if not v_df.empty and 'Date' in v_df.columns:
                target_date = pd.to_datetime(date_str) if date_str else pd.Timestamp.now()
                v_df['Date'] = pd.to_datetime(v_df['Date'], errors='coerce')
                recent_vetoes = v_df[(v_df['Action'] == 'VETO_REMOVE') & ((target_date - v_df['Date']).dt.days <= 10) & ((target_date - v_df['Date']).dt.days >= 0)]
                if not recent_vetoes.empty:
                    banned_symbols = set(recent_vetoes['Symbol'].tolist())
                    if isinstance(chartink_universe, dict):
                        chartink_universe = {sym: data for sym, data in chartink_universe.items() if sym not in banned_symbols}
                    else:
                        chartink_universe = [sym for sym in chartink_universe if sym not in banned_symbols]
                    log_info(f"Veto Cooldown applied: Excluded {len(banned_symbols)} stocks from ranking.")
        except Exception as e:
            log_warning(f"Failed to process Veto cooldowns: {e}")


    # Get Nifty Smallcap 250 6m ROC for Relative Strength (peer benchmark)
    bench_roc_6m = 15.0
    smallcap250_hist = None
    if pipeline_data:
        smallcap250_hist = pipeline_data.get("NIFTY_SMALLCAP_250")
    if smallcap250_hist is None:
        smallcap250_hist = get_historical_data("NIFTY_SMALLCAP_250", end_date=date_str)
    if smallcap250_hist is not None and len(smallcap250_hist) >= 126:
        smallcap250_close = smallcap250_hist["Close"]
        bench_roc_6m = (smallcap250_close.iloc[-1] - smallcap250_close.iloc[-126]) / smallcap250_close.iloc[-126] * 100.0

    # Get Nifty 50 data for Independent Alpha check
    nifty50_hist = None
    if pipeline_data:
        nifty50_hist = pipeline_data.get("NIFTY_50")
    if nifty50_hist is None:
        nifty50_hist = get_historical_data("NIFTY_50", end_date=date_str)

    # Identify which indices are bullish (Close > 150 EMA and rising 150 EMA)
    index_bullish = {}
    for index_name in ["NIFTY_50", "NIFTY_NEXT_50", "NIFTY_MIDCAP_150", "NIFTY_SMALLCAP_250"]:
        idx_df = None
        if pipeline_data:
            idx_df = pipeline_data.get(index_name)
        if idx_df is None:
            idx_df = get_historical_data(index_name, end_date=date_str)
        
        is_bullish = False
        if idx_df is not None and not idx_df.empty:
            if "EMA_150" not in idx_df.columns:
                idx_df = idx_df.copy()
                idx_df["EMA_150"] = idx_df["Close"].ewm(span=150, adjust=False).mean()
            
            last_close = float(idx_df["Close"].iloc[-1])
            last_ema = float(idx_df["EMA_150"].iloc[-1])
            above_150 = last_close > last_ema
            
            # Rising EMA_150 check (slope over 10 days)
            rising_150 = True
            if len(idx_df) >= 10:
                rising_150 = float(idx_df["EMA_150"].iloc[-1]) > float(idx_df["EMA_150"].iloc[-10])
            
            is_bullish = above_150 and rising_150
            log_info(f"Index check: {index_name} | Close: {last_close:.2f} | 150 EMA: {last_ema:.2f} | Rising: {rising_150} | Bullish: {is_bullish}")
        else:
            log_warning(f"Index data missing for {index_name}. Defaulting to True (Bullish).")
            is_bullish = True
            
        index_bullish[index_name] = is_bullish

    allowed_categories = set()
    if index_bullish.get("NIFTY_50", True):
        allowed_categories.add("MEGA_CAP")
    if index_bullish.get("NIFTY_NEXT_50", True):
        allowed_categories.add("MEGA_CAP")
        allowed_categories.add("LARGE_CAP")
    if index_bullish.get("NIFTY_MIDCAP_150", True):
        allowed_categories.add("MID_CAP")
    if index_bullish.get("NIFTY_SMALLCAP_250", True):
        allowed_categories.add("SMALL_CAP")

    # ── PASS 1: Collect delivery_pct and RS per stock to compute sector averages ──
    delivery_by_sector = {}   # sector -> list of delivery_pct
    symbol_delivery_cache = {}

    # Fallback storage for Sector/Theme ranks if not provided
    temp_sector_rs = {}
    temp_theme_rs = {}

    for _sym, _sdata in chartink_universe.items():
        _deliv = fetch_delivery_data(_sym)
        _dpct = _deliv.get("delivery_pct", 45.0) if isinstance(_deliv, dict) else 45.0
        _sec = get_dynamic_sector(_sym)
        _thm = get_dynamic_theme(_sym)
        symbol_delivery_cache[_sym] = _dpct
        delivery_by_sector.setdefault(_sec, []).append(_dpct)

        # Calculate a quick 50-day RS vs Nifty50 for fallback ranking
        if (sector_ranks is None or theme_ranks is None) and nifty50_hist is not None:
            _hist = pipeline_data.get(_sym) if pipeline_data else get_historical_data(_sym, days=100, end_date=date_str)
            if _hist is not None and len(_hist) >= 50:
                common_idx = _hist.index.intersection(nifty50_hist.index)
                if len(common_idx) >= 50:
                    rs_ratio = _hist.loc[common_idx, "Close"] / nifty50_hist.loc[common_idx, "Close"]
                    _rs = float((rs_ratio.iloc[-1] - rs_ratio.iloc[-50]) / rs_ratio.iloc[-50] * 100.0)
                    temp_sector_rs.setdefault(_sec, []).append(_rs)
                    temp_theme_rs.setdefault(_thm, []).append(_rs)

    # Compute sector average deliveries
    sector_avg_delivery_map = {
        sec: float(sum(vals) / len(vals))
        for sec, vals in delivery_by_sector.items() if vals
    }

    # Apply Fallback Ranks if missing
    if sector_ranks is None:
        if temp_sector_rs:
            sec_avgs = {k: sum(v)/len(v) for k, v in temp_sector_rs.items()}
            sorted_secs = sorted(sec_avgs.keys(), key=lambda k: sec_avgs[k])
            sector_ranks = {k: (i / max(1, len(sorted_secs) - 1)) * 100.0 for i, k in enumerate(sorted_secs)}
        else:
            sector_ranks = {}
            
    if theme_ranks is None:
        if temp_theme_rs:
            thm_avgs = {k: sum(v)/len(v) for k, v in temp_theme_rs.items()}
            sorted_thms = sorted(thm_avgs.keys(), key=lambda k: thm_avgs[k])
            theme_ranks = {k: (i / max(1, len(sorted_thms) - 1)) * 100.0 for i, k in enumerate(sorted_thms)}
        else:
            theme_ranks = {}

    # Track results
    eligible_raw_records = []
    rejected_records = []

    # ── PASS 2: Sequential scanning of the stock universe ──
    for symbol, data in chartink_universe.items():
        screeners = data["Screeners"]
        close_price = data["Close"]
        volume_raw = data["Volume"]
        sector = get_dynamic_sector(symbol)
        theme = get_dynamic_theme(symbol)

        # ── 1. Market Capitalization Filter (>= ₹1,000 Crores) ──
        # Evaluated early before loading heavy history to save resources
        mcap_cr, cap_category = fetch_market_cap(symbol)
        if mcap_cr < MIN_MARKET_CAP_CR:
            log_warning(f"REJECTED {symbol}: Market cap ₹{mcap_cr:,.0f} Cr < ₹{MIN_MARKET_CAP_CR:,} Cr minimum.")
            rejected_records.append(
                make_rejected_record(
                    symbol, close_price, close_price * volume_raw,
                    f"Market Cap ₹{mcap_cr:.0f} Cr < {MIN_MARKET_CAP_CR} Cr",
                    sector, theme, ",".join(screeners), mcap_cr, cap_category
                )
            )
            continue

        # Load daily price history early to compute exceptional indicators
        hist_df = None
        if pipeline_data:
            hist_df = pipeline_data.get(symbol)
        if hist_df is None:
            hist_df = get_historical_data(symbol, days=1100, end_date=date_str)

        if hist_df is None or hist_df.empty:
            log_warning(f"REJECTED {symbol}: No price history found.")
            rejected_records.append(
                make_rejected_record(
                    symbol, close_price, close_price * volume_raw,
                    "No price history found",
                    sector, theme, ",".join(screeners), mcap_cr, cap_category
                )
            )
            continue

        # Compute Trend & Momentum parameters for potential override
        is_exceptional_bull = False
        closes_temp = hist_df["Close"]
        if len(closes_temp) >= 15:
            temp_close = float(closes_temp.iloc[-1])
            temp_ema_150 = closes_temp.ewm(span=150, adjust=False).mean().iloc[-1]
            temp_ema_200 = closes_temp.ewm(span=200, adjust=False).mean().iloc[-1]
            trend_bullish = temp_close > temp_ema_150 and temp_close > temp_ema_200
            
            roc_1m = (closes_temp.iloc[-1] - closes_temp.iloc[-21]) / closes_temp.iloc[-21] * 100.0 if len(closes_temp) >= 21 else 0.0
            roc_3m = (closes_temp.iloc[-1] - closes_temp.iloc[-63]) / closes_temp.iloc[-63] * 100.0 if len(closes_temp) >= 63 else 0.0
            roc_6m = (closes_temp.iloc[-1] - closes_temp.iloc[-126]) / closes_temp.iloc[-126] * 100.0 if len(closes_temp) >= 126 else 0.0
            roc_12m = (closes_temp.iloc[-1] - closes_temp.iloc[-252]) / closes_temp.iloc[-252] * 100.0 if len(closes_temp) >= 252 else 0.0
            mom_score = 0.40 * roc_3m + 0.30 * roc_6m + 0.20 * roc_12m + 0.10 * roc_1m
            
            rs_vs_nifty50_temp = 0.0
            if nifty50_hist is not None:
                common_idx = closes_temp.index.intersection(nifty50_hist.index)
                if len(common_idx) >= 50:
                    rs_ratio = closes_temp.loc[common_idx] / nifty50_hist.loc[common_idx, "Close"]
                    rs_vs_nifty50_temp = float((rs_ratio.iloc[-1] - rs_ratio.iloc[-50]) / rs_ratio.iloc[-50] * 100.0)
            
            # Exception Criteria: Trend Bullish AND Extraordinary Momentum AND Benchmark Outperformance.
            # Raised from loose OR condition (mom >= 35 OR RS >= 15) to a strict AND gate.
            # This override is meant for stocks that have truly exceptional price action
            # despite failing the ADX/Volatility technical filter. It still cannot bypass
            # Quality Gates (D/E, ROCE, CFO/PAT), Trend Gate (200 EMA), or any other hard gate.
            # Requirements:
            #   - Confirmed uptrend (above both 150 EMA and 200 EMA)
            #   - Weighted momentum score >= 50 (40% 3m + 30% 6m + 20% 12m + 10% 1m)
            #   - 50-day RS vs Nifty >= 20% (stock is crushing the benchmark)
            is_exceptional_bull = trend_bullish and mom_score >= 50.0 and rs_vs_nifty50_temp >= 20.0
            if is_exceptional_bull:
                log_success(f"Detected exceptional trend & momentum for {symbol} (Mom Score: {mom_score:.1f}, RS vs N50: {rs_vs_nifty50_temp:.1f}%)")

        # ── 3. SECTOR/THEME TREND GATE (replaces old index restriction) ──
        # Check if the stock's sector or theme has positive trend
        # This aligns with F1 (Sectoral Trend) and F2 (Thematic Trend) factors
        if cap_category not in allowed_categories:
            sector_rank = sector_ranks.get(sector, 50.0) if sector_ranks else 50.0
            theme_rank = theme_ranks.get(theme, 50.0) if theme_ranks else 50.0
            in_active_theme = any(active.lower() in theme.lower() for active in ACTIVE_THEMES)
            
            # Pass if: sector rank > 50% OR theme rank > 50% OR theme is active
            if sector_rank >= 50.0 or theme_rank >= 50.0 or in_active_theme:
                log_success(f"BYPASS SECTOR/THEME REJECTION for {symbol}: Sector rank {sector_rank:.1f}%, Theme rank {theme_rank:.1f}% or Active theme '{theme}'")
            else:
                rejection_reason = f"Sector/Theme weak: sector rank {sector_rank:.1f}%, theme rank {theme_rank:.1f}%"
                log_warning(f"REJECTED {symbol}: {rejection_reason}")
                rejected_records.append(
                    make_rejected_record(
                        symbol, close_price, close_price * volume_raw,
                        rejection_reason,
                        sector, theme, ",".join(screeners), mcap_cr, cap_category
                    )
                )
                continue

        # To determine if the stock is a real IPO/new listing, check history over 1100 calendar days (~3 years)
        hist_check_df = hist_df
        history_len = len(hist_check_df) if hist_check_df is not None else 0

        # ── 2. IPO Seasoning Gate ──
        # 252 trading days ≈ 1 calendar year of trading history.
        # Momentum systems cannot reliably trade stocks with <1yr of price history:
        # lock-up expiries, incomplete price discovery, and no ATR/volatility baseline.
        if history_len < 252:
            log_warning(f"REJECTED {symbol}: IPO seasoning < 1 year ({history_len} trading days).")
            rejected_records.append(
                make_rejected_record(
                    symbol, close_price, close_price * volume_raw,
                    f"IPO seasoning < 1 year ({history_len} days)",
                    sector, theme, ",".join(screeners), mcap_cr, cap_category, is_ipo=True
                )
            )
            continue

        is_ipo = history_len < 756  # 1 to 3 years listed

        # Redefine Volume in historical df as Rupee Traded Value
        hist_df = hist_df.copy()
        hist_df["Volume"] = hist_df["Close"] * hist_df["Volume"]
        volume = close_price * volume_raw  # today's Rupee Traded Value

        # ── 3. ADTV Filter (30-day Average Daily Turnover >= ₹10 Crores) ──
        adtv_30 = hist_df["Volume"].tail(30).mean()
        if adtv_30 < MIN_ADTV_CR * 10000000:
            log_warning(f"REJECTED {symbol}: 30-day ADTV ₹{adtv_30/10000000:.2f} Cr < ₹{MIN_ADTV_CR:.0f} Cr.")
            rejected_records.append(
                make_rejected_record(
                    symbol, close_price, volume,
                    f"ADTV ₹{adtv_30/10000000:.2f} Cr < {MIN_ADTV_CR} Cr",
                    sector, theme, ",".join(screeners), mcap_cr, cap_category, is_ipo=is_ipo
                )
            )
            continue

        # ── 3b. ASM/GSM SURVEILLANCE GATE — added per Claude Sonnet 4 review ──
        # Guards against stocks under SEBI Additional/Graded Surveillance Measures
        # which have sudden 100% margin requirements and price band freezes
        if _is_asm_gsm_stock(symbol):
            log_warning(f"REJECTED {symbol}: Under ASM/GSM surveillance.")
            rejected_records.append(
                make_rejected_record(
                    symbol, close_price, volume,
                    "Under ASM/GSM surveillance — capital trap risk",
                    sector, theme, ",".join(screeners), mcap_cr, cap_category, is_ipo=is_ipo
                )
            )
            continue

        # Fetch fundamentals
        f_data = fetch_screener_fundamentals(symbol)
        is_financial = any(w in sector.lower() for w in ["bank", "financial", "nbfc"])
        if "capital market" in theme.lower() or "capital market" in sector.lower():
            is_financial = False

        # ── 4. TWO-TRACK QUALITY HARD GATE ──────────────────────────────────────
        # BFSI path: Net NPA < 1.75% | CAR > 12% | ROA > 0.80% | Pledge < 15%
        # Standard path: D/E < 1.5 | 3Yr CFO/PAT > 0 | ROCE > 8% (cyclicals exempt) | Pledge < 20%
        # NON-BYPASSABLE: is_exceptional_bull cannot override this gate.
        quality_passed, quality_reason = _apply_quality_gate(symbol, f_data, is_financial, sector)
        if not quality_passed:
            log_warning(f"REJECTED {symbol}: {quality_reason}")
            rejected_records.append(
                make_rejected_record(
                    symbol, close_price, volume,
                    quality_reason,
                    sector, theme, ",".join(screeners), mcap_cr, cap_category, is_ipo=is_ipo
                )
            )
            continue

        # Pull ROCE for downstream scoring (still needed in eligible record)
        roce_3yr = f_data.get("ROCE_3Yr_Avg", 15.0)
        cfo_pat_3yr = f_data.get("CFO_PAT_3Yr_Avg", 0.80)
        debt_to_eq = f_data.get("Debt_to_Equity", 0.0) if f_data else 0.0

        # ── 7. Trend Filter (Price > 200 EMA) ──
        # NOTE: Not bypassable. Trend alignment is a core requirement.
        ema_200 = hist_df["Close"].ewm(span=200, adjust=False).mean().iloc[-1]
        if close_price <= ema_200:
            log_warning(f"REJECTED {symbol}: Close ₹{close_price:.2f} <= 200 EMA ₹{ema_200:.2f}.")
            rejected_records.append(
                make_rejected_record(
                    symbol, close_price, volume,
                    f"Close ₹{close_price:.2f} <= 200 EMA ₹{ema_200:.2f}",
                    sector, theme, ",".join(screeners), mcap_cr, cap_category, is_ipo=is_ipo
                )
            )
            continue

        # ── 8. Volatility & Trend Strength Filter (ADX-14 > 20, Annualized Volatility < 60%) ──
        # NOTE: This is the ONLY gate that exceptional_bull can bypass.
        # All quality gates (D/E, ROCE, CFO/PAT, Trend) are hard requirements.
        adx_series, plus_di_series, minus_di_series = calculate_adx(hist_df, 14)
        adx_val = adx_series.iloc[-1] if not adx_series.empty else 0.0
        plus_di_val = plus_di_series.iloc[-1] if not plus_di_series.empty else 0.0
        minus_di_val = minus_di_series.iloc[-1] if not minus_di_series.empty else 0.0
        adx_bullish = adx_val > MIN_ADX_14 and plus_di_val > minus_di_val

        # Annualized Volatility
        daily_returns = hist_df["Close"].pct_change()
        vol_window = min(252, len(hist_df) - 1)
        ann_vol = daily_returns.tail(vol_window).std() * np.sqrt(252) * 100.0

        if adx_val <= MIN_ADX_14 or ann_vol >= MAX_ANNUAL_VOL_PCT:
            if is_exceptional_bull:
                log_success(f"BYPASS ADX/VOL FILTER for {symbol}: Exceptional trend & momentum override. ADX={adx_val:.1f}, Vol={ann_vol:.1f}%")
            else:
                log_warning(f"REJECTED {symbol}: Volatility {ann_vol:.1f}% >= 60% or ADX {adx_val:.1f} <= 20.")
                rejected_records.append(
                    make_rejected_record(
                        symbol, close_price, volume,
                        f"ADX {adx_val:.1f} <= 20 or Vol {ann_vol:.1f}% >= 60%",
                        sector, theme, ",".join(screeners), mcap_cr, cap_category, is_ipo=is_ipo
                    )
                )
                continue

        # ── 9. Delivery Confirmation Gate (New — aligns with F6_DELIVERY_CONFIRMATION) ──
        # Hard minimum: delivery% must be >= 30% to pass hard gate
        # F6 in scoring has higher bar (40%), this is just the entry floor
        delivery_data = fetch_delivery_data(symbol)
        delivery_pct = delivery_data.get("delivery_pct", 0.0) if isinstance(delivery_data, dict) else delivery_data
        if delivery_pct is not None and delivery_pct < 30.0:
            log_warning(f"REJECTED {symbol}: Delivery% {delivery_pct:.1f}% < 30%.")
            rejected_records.append(
                make_rejected_record(
                    symbol, close_price, volume,
                    f"Delivery% {delivery_pct:.1f}% < 30%",
                    sector, theme, ",".join(screeners), mcap_cr, cap_category, is_ipo=is_ipo
                )
            )
            continue

        # ── 10. FII/DII Activity Gate ──
        fii_change = f_data.get("FII_DII_Net_Change", 0.0)
        fii_total = f_data.get("Total_FII_DII_Holding_%", 0.0)
        
        if fii_change < -1.0:
            log_warning(f"REJECTED {symbol}: FII/DII heavily selling (Change: {fii_change:.2f}%)")
            rejected_records.append(
                make_rejected_record(
                    symbol, close_price, volume,
                    f"FII/DII heavily selling (Change: {fii_change:.2f}%)",
                    sector, theme, ",".join(screeners), mcap_cr, cap_category, is_ipo=is_ipo
                )
            )
            continue

        # ── ELIGIBLE CANDIDATE FOUND ──────────────────
        # Calculate Technical Indicators needed for legacy track checks & scoring
        natr_series = calculate_natr(hist_df, 14)
        natr_val = natr_series.iloc[-1] if not natr_series.empty else 0.0
        natr_window = min(90, len(natr_series))
        natr_90d_avg = natr_series.tail(natr_window).mean() if not natr_series.empty else natr_val
        natr_trend = "CONTRACTING" if natr_val < natr_90d_avg else "EXPANDING"
        chop_14 = calculate_chop_index(hist_df, 14)
        chop_20 = calculate_chop_index(hist_df, 20)
        chop_avg = (chop_14.iloc[-1] + chop_20.iloc[-1]) / 2.0 if not chop_14.empty and not chop_20.empty else 35.0

        df_weekly = hist_df.resample('W').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'})
        weekly_chop = 35.0
        if len(df_weekly) >= 15:
            chop_w = calculate_chop_index(df_weekly, 14)
            if not chop_w.empty:
                weekly_chop = float(chop_w.iloc[-1])

        atr_val = calculate_atr(hist_df, 14).iloc[-1]
        
        closes = hist_df["Close"]
        ma_50_series = closes.rolling(50).mean()
        ma_50_last = ma_50_series.iloc[-1]
        extension_from_50dma = float((close_price - ma_50_last) / ma_50_last * 100.0) if ma_50_last > 0 else 0.0

        whipsaw_20ma = int((closes > closes.rolling(20).mean()).astype(int).diff().abs().tail(30).sum())
        whipsaw_50ma = int((closes > closes.rolling(50).mean()).astype(int).diff().abs().tail(60).sum())

        stage2_aligned = False
        emerging_recovery_aligned = False
        is_vcp_setup = False
        sma_50 = 0.0
        sma_200 = 0.0
        high_52w = close_price
        low_52w = close_price
        if len(hist_df) >= 200:
            ema_50_series = closes.ewm(span=50, adjust=False).mean()
            ema_50 = ema_50_series.iloc[-1]
            ema_50_prev = ema_50_series.iloc[-5] if len(ema_50_series) >= 5 else ema_50_series.iloc[-2]
            ema_50_rising = ema_50 > ema_50_prev
            
            sma_50_series = closes.rolling(50).mean()
            sma_50 = sma_50_series.iloc[-1]
            sma_150_series = closes.rolling(150).mean()
            sma_150 = sma_150_series.iloc[-1]
            sma_200_series = closes.rolling(200).mean()
            sma_200 = sma_200_series.iloc[-1]
            
            # Rising checks (compare to 22 trading days ago ~30 calendar days)
            sma_50_prev = sma_50_series.iloc[-22] if len(sma_50_series) >= 22 else sma_50_series.iloc[0]
            sma_50_rising = sma_50 > sma_50_prev
            
            sma_150_prev = sma_150_series.iloc[-22] if len(sma_150_series) >= 22 else sma_150_series.iloc[0]
            sma_150_rising = sma_150 > sma_150_prev
            
            sma_200_prev = sma_200_series.iloc[-22] if len(sma_200_series) >= 22 else sma_200_series.iloc[0]
            sma_200_rising = sma_200 > sma_200_prev
            
            # 52-week High and Low bounds (252 trading days)
            window_252 = min(252, len(closes))
            low_52w = float(closes.tail(window_252).min())
            high_52w = float(closes.tail(window_252).max())
            
            above_low_25pct = close_price >= low_52w * 1.25
            within_high_25pct = close_price >= high_52w * 0.75
            
            # Volume confirmation: average volume in last 10 days > average volume in last 90 days
            vol_10d_avg = hist_df["Volume"].tail(10).mean()
            vol_90d_avg = hist_df["Volume"].tail(90).mean()
            vol_confirmation = vol_10d_avg > vol_90d_avg
            
            stage2_aligned = (
                close_price > sma_50
                and close_price > sma_150
                and close_price > sma_200
                and sma_50 > sma_150 > sma_200
                and sma_50_rising
                and sma_150_rising
                and sma_200_rising
                and above_low_25pct
                and within_high_25pct
                and vol_confirmation
            )
            
            # VCP Setup flag: tight 2-week range (high-low in last 10 days is within 15% of close price)
            high_10d = float(closes.tail(10).max())
            low_10d = float(closes.tail(10).min())
            is_vcp_setup = (high_10d - low_10d) <= close_price * 0.15
            
            emerging_recovery_aligned = (close_price > ema_50) and (close_price > sma_200) and (ema_50 > sma_200) and ema_50_rising

        vwap_location = "Bullish Zone" if close_price > ema_200 else "Accumulation"
        is_exit_candidate = False
        exit_reason = "None"

        # OBV and slopes
        obv_series = calculate_obv(hist_df)
        def compute_slope_20d(series):
            if len(series) < 20: return 0.0
            y = series.tail(20).values
            x = np.arange(20)
            try:
                slope, _ = np.polyfit(x, y, 1)
                return float(slope)
            except:
                return 0.0

        obv_slope = compute_slope_20d(obv_series)
        obv_ema20 = obv_series.ewm(span=20, adjust=False).mean()
        obv_rising = obv_series.iloc[-1] > obv_ema20.iloc[-1] if not obv_series.empty and not obv_ema20.empty else False

        up_down_ratio = get_up_down_volume_ratio(hist_df, 25)
        dfi = obv_slope * up_down_ratio

        rs_vs_nifty50 = 0.0
        independent_alpha_pass = False
        if nifty50_hist is not None:
            common_idx = hist_df.index.intersection(nifty50_hist.index)
            if len(common_idx) >= 50:
                rs_ratio = hist_df.loc[common_idx, "Close"] / nifty50_hist.loc[common_idx, "Close"]
                rs_vs_nifty50 = float((rs_ratio.iloc[-1] - rs_ratio.iloc[-50]) / rs_ratio.iloc[-50] * 100.0)
                independent_alpha_pass = rs_vs_nifty50 > 0.0

        # Delivery data
        delivery_pct = symbol_delivery_cache.get(symbol, 45.0)
        delivery_below_threshold = delivery_pct < 30.0

        # Delivery volume ratio: today's delivery volume vs 90d avg total volume
        delivery_pct_val = float(delivery_pct) if delivery_pct is not None else 45.0
        vol_90d_avg = hist_df["Volume"].tail(90).mean() if len(hist_df) >= 90 else adtv_30
        delivery_vol_ratio = (volume * (delivery_pct_val / 100.0)) / vol_90d_avg if vol_90d_avg > 0 else 1.0

        # Calculate opportunity score factors
        roc_1m = (closes.iloc[-1] - closes.iloc[-21]) / closes.iloc[-21] * 100.0 if len(closes) >= 21 else 0.0
        roc_3m = (closes.iloc[-1] - closes.iloc[-63]) / closes.iloc[-63] * 100.0 if len(closes) >= 63 else 0.0
        roc_6m = (closes.iloc[-1] - closes.iloc[-126]) / closes.iloc[-126] * 100.0 if len(closes) >= 126 else 0.0
        roc_12m = (closes.iloc[-1] - closes.iloc[-252]) / closes.iloc[-252] * 100.0 if len(closes) >= 252 else 0.0
        momentum_score = 0.40 * roc_3m + 0.30 * roc_6m + 0.20 * roc_12m + 0.10 * roc_1m
        rs_score = roc_6m - bench_roc_6m

        # ── COMPUTE THE 4 ALPHA SCANS ──

        # 1. Institutional Volatility-Adjusted Momentum Score (VAM-GQ 63-Day Risk-Adjusted Return)
        # Using pure 63-day return divided by annualized volatility
        def calc_vam_gq_mom(prices):
            if len(prices) >= 45:
                _ret = (prices.iloc[-1] / prices.iloc[0]) - 1.0
                _vol = prices.pct_change().std() * np.sqrt(252)
                if _vol > 0:
                    return _ret / _vol
            return 0.0
            
        score_3m = calc_vam_gq_mom(hist_df["Close"].tail(63))
        raw_mom_score = score_3m

        # 2. Fundamental Growth Score
        sales_growth = f_data.get("Sales_Growth_%", 15.0)
        profit_growth = f_data.get("Profit_Growth_%", 15.0)
        raw_growth_score = (sales_growth + profit_growth) / 2.0

        # 3. PEAD Score — REAL earnings detection from price/volume patterns + bhavcopy
        # Strategy: Detect institutional accumulation events (earnings-like patterns)
        # using volume surge + price gap + subsequent drift
        lookback_len = min(65, len(hist_df) - 3)
        if lookback_len > 0:
            # Find the most significant volume+price surge event in lookback period
            recent_vol = hist_df["Volume"].iloc[-lookback_len-3 : -3].copy()
            
            # Identify candidate earnings events: top 5% volume days with positive close
            vol_threshold = recent_vol.quantile(0.95)
            high_vol_days = recent_vol[recent_vol >= vol_threshold]
            
            if len(high_vol_days) > 0:
                # Among high-volume days, pick the one with largest price gap (open vs prev close)
                best_event = None
                best_score = 0
                
                for ev_date in high_vol_days.index:
                    ev_loc = hist_df.index.get_loc(ev_date)
                    if ev_loc < 1 or ev_loc >= len(hist_df) - 5:
                        continue
                    
                    ev_open = hist_df["Open"].iloc[ev_loc]
                    ev_close = hist_df["Close"].iloc[ev_loc]
                    prev_close = hist_df["Close"].iloc[ev_loc - 1]
                    
                    # Price gap (gap up = earnings beat, gap down = miss)
                    gap_pct = (ev_open / prev_close - 1.0) * 100.0
                    
                    # Post-event drift (5-day return after event)
                    post_close_5 = hist_df["Close"].iloc[min(ev_loc + 5, len(hist_df)-1)]
                    drift_5d = (post_close_5 / ev_close - 1.0) * 100.0
                    
                    # Composite event score: gap magnitude + post-drift
                    event_score = abs(gap_pct) * 2.0 + abs(drift_5d)
                    if event_score > best_score:
                        best_score = event_score
                        best_event = ev_loc
                
                if best_event is not None:
                    earnings_loc = best_event
                    t_days = len(hist_df) - 1 - earnings_loc
                    
                    # Compute surprise from price reaction
                    ev_open = hist_df["Open"].iloc[earnings_loc]
                    ev_close = hist_df["Close"].iloc[earnings_loc]
                    prev_close = hist_df["Close"].iloc[earnings_loc - 1]
                    gap_pct = (ev_open / prev_close - 1.0) * 100.0
                    
                    # Positive gap = positive surprise proxy
                    if gap_pct > 1.0:
                        surprise = min(30.0, gap_pct * 3.0)  # Cap at 30% surprise
                    elif gap_pct < -1.0:
                        surprise = max(-30.0, gap_pct * 2.0)  # Negative surprise
                    else:
                        surprise = 0.0  # No significant gap — not an earnings event
                    
                    # Volume multiplier
                    pre_start = max(0, earnings_loc - 20)
                    pre_vol = hist_df["Volume"].iloc[pre_start:earnings_loc].mean()
                    post_end = min(len(hist_df), earnings_loc + 3)
                    post_vol = hist_df["Volume"].iloc[earnings_loc:post_end].mean()
                    vol_multiplier = post_vol / pre_vol if pre_vol > 0 else 1.0
                else:
                    t_days = 0
                    vol_multiplier = 1.0
                    surprise = 0.0
            else:
                t_days = 0
                vol_multiplier = 1.0
                surprise = 0.0
        else:
            t_days = 0
            vol_multiplier = 1.0
            surprise = 0.0

        raw_pead_score = surprise * vol_multiplier * np.exp(-0.03576 * t_days)

        # 4. Smart Money/Volume Ratio
        recent_20 = hist_df.tail(20).copy()
        recent_20["Price_Diff"] = recent_20["Close"].diff()
        up_vol_sum = recent_20.loc[recent_20["Price_Diff"] > 0, "Volume"].sum()
        dn_vol_sum = recent_20.loc[recent_20["Price_Diff"] < 0, "Volume"].sum()
        raw_sm_score = up_vol_sum / dn_vol_sum if dn_vol_sum > 0 else (2.0 if up_vol_sum > 0 else 1.0)

        # Legacy structures
        avg_vol_20 = hist_df["Volume"].tail(20).mean()
        def vol_breakout_at(mult):
            return volume >= avg_vol_20 * mult

        eligible_raw_records.append({
            "Symbol": symbol, "Close": close_price, "Volume": volume,
            "Sector": sector, "Theme": theme, "Screeners": ",".join(screeners),
            "Market_Cap_Cr": mcap_cr, "Cap_Category": cap_category,
            "Data_Source": f_data.get("Data_Source", "N/A"), "Data_Flags": f_data.get("Data_Flags", ""),
            "whipsaw_20ma": whipsaw_20ma, "whipsaw_50ma": whipsaw_50ma,
            "chop_avg": chop_avg, "weekly_chop": weekly_chop, "natr_val": natr_val,
            "natr_trend": natr_trend,
            "extension_from_50dma": extension_from_50dma, "atr_val": atr_val,
            "stage2_aligned": stage2_aligned, "emerging_recovery_aligned": emerging_recovery_aligned,
            "vwap": ema_200, "vwap_location": vwap_location, "is_exit_candidate": is_exit_candidate, "exit_reason": exit_reason,
            "rs_return_6m": roc_6m, "dfi": dfi, "adx_val": adx_val, "adx_bullish": adx_bullish, "obv_rising": obv_rising,
            "rs_vs_nifty50": rs_vs_nifty50, "independent_alpha_pass": independent_alpha_pass,
            "up_down_ratio": up_down_ratio, "delivery_pct": delivery_pct, "delivery_below_threshold": delivery_below_threshold,
            "delivery_vol_ratio": delivery_vol_ratio,
            "ROE": f_data.get("ROE_%", 15.0), "Debt_to_Equity": debt_to_eq, "is_ipo": is_ipo,
            "is_financial": is_financial, "sales_growth": sales_growth, "profit_growth": profit_growth,
            "opm": f_data.get("OPM_%", 15.0), "surprise": surprise, "fii_dii_net": f_data.get("FII_DII_Net_Change", 0.0),
            # Scoring raw factors
            "raw_mom_score": raw_mom_score, "raw_growth_score": raw_growth_score,
            "raw_pead_score": raw_pead_score, "raw_sm_score": raw_sm_score,
            "momentum_score": momentum_score, "rs_score": rs_score, "vol_breakout_at": vol_breakout_at,
            "is_exceptional_bull": is_exceptional_bull,
            "is_vcp_setup": is_vcp_setup,
            "stage2_aligned": stage2_aligned,
            "sma_50": sma_50,
            "sma_200": sma_200,
            "high_52w": high_52w,
            "low_52w": low_52w,
            "roce_3yr": roce_3yr,
            "sector_rank": sector_ranks.get(sector, 50.0) if sector_ranks else 50.0,
            "theme_rank": theme_ranks.get(theme, 50.0) if theme_ranks else 50.0,
            "cfo_pat_3yr": cfo_pat_3yr,
            "days_since_earnings": t_days
        })

    # ── TIERING AND RANKING FOR ELIGIBLE STOCKS ──
    eligible_processed_records = []
    if eligible_raw_records:
        df_el = pd.DataFrame(eligible_raw_records)

        # Percentile rank each factor
        df_el["rank_mom"] = df_el["raw_mom_score"].rank(pct=True) * 100.0
        
        # Sector-Neutral Growth Rank
        df_el["rank_growth"] = df_el.groupby("Sector")["raw_growth_score"].rank(pct=True) * 100.0
        # Fallback to global rank if sector group is too small
        df_el["rank_growth"] = df_el["rank_growth"].fillna(df_el["raw_growth_score"].rank(pct=True) * 100.0)
        
        df_el["rank_pead"] = df_el["raw_pead_score"].rank(pct=True) * 100.0
        
        # Pre-calculate pure percentile ranks for institutional/delivery scoring
        df_el["rank_delivery"] = df_el["delivery_vol_ratio"].rank(pct=True) * 100.0
        df_el["rank_fii"] = df_el["fii_dii_net"].rank(pct=True) * 100.0

        # Legacy placeholders
        df_el["Composite_Score"] = 0.0
        df_el["Opportunity_Score"] = 0.0
        df_el["Final_Rank"] = 999

        for idx, row in df_el.iterrows():
            row_dict = row.to_dict()
            # Score 8 factors (continuous 0-100 per factor, weighted composite)
            factor_scores, factor_details, weighted_score = score_7_factors(row_dict)

            # Final Score exactly equals the weighted_score (removed double scoring)
            final_score = weighted_score

            # Legacy compatibility: passed_factors = factors scoring >= 50/100
            passed_factors = [k for k, v in factor_scores.items() if v >= 50.0]
            factor_score_legacy = len(passed_factors)

            # Store weighted score and final score back into the row dict for sorting
            row_dict["Weighted_Score"] = weighted_score
            row_dict["Final_Score"] = final_score
            eligible_processed_records.append({
                **row_dict,
                "passed_factors": passed_factors,
                "factor_details_obj": factor_details,
                "factor_score_legacy": factor_score_legacy,
            })

    # ── WEIGHTED RANKING: sort by Final_Score and pick top 50 ──
    eligible_processed_records_out = []
    if eligible_processed_records:
        # Sort all candidates by Final_Score descending
        eligible_processed_records.sort(key=lambda r: r.get("Final_Score", 0.0), reverse=True)

        for rank_pos, row_dict in enumerate(eligible_processed_records, start=1):
            passed_factors = row_dict.pop("passed_factors", [])
            factor_details = row_dict.pop("factor_details_obj", {})
            factor_score_legacy = row_dict.pop("factor_score_legacy", 0)
            final_score = row_dict.get("Final_Score", 0.0)
            weighted_score = row_dict.get("Weighted_Score", 0.0)

            # Top N stocks are Entry_Eligible; rest are watchlist only
            eligible = rank_pos <= TOP_N_STOCKS

            # Tier by rank position (not factor count)
            if rank_pos <= 15:
                tier = "TIER 1 — HIGH CONVICTION"
            elif rank_pos <= 35:
                tier = "TIER 2 — MEDIUM CONVICTION"
            elif rank_pos <= TOP_N_STOCKS:
                tier = "TIER 3 — LOW CONVICTION"
            else:
                tier = "WATCHLIST"

            reason = (
                f"RANKED #{rank_pos} — Weighted Score: {weighted_score:.1f}/100 | Final Score: {final_score:.1f}"
                if eligible else
                f"WATCHLIST #{rank_pos} — Score: {final_score:.1f} (outside Top {TOP_N_STOCKS})"
            )

            record = {
                "Symbol": row_dict["Symbol"], "Close": row_dict["Close"], "Volume": row_dict["Volume"],
                "Composite_Score": row_dict["Composite_Score"], "Opportunity_Score": row_dict["Opportunity_Score"],
                "Weighted_Score": round(weighted_score, 2), "Final_Score": round(final_score, 2),
                "Tier": tier, "Stage": 2 if eligible else 1, "Entry_Eligible": eligible,
                "ATR_14": row_dict["atr_val"], "NATR_14": row_dict["natr_val"],
                "NATR_Trend": row_dict["natr_trend"],
                "ADX_14": round(row_dict["adx_val"], 2), "ADX_Bullish": row_dict["adx_bullish"], "OBV_Rising": row_dict["obv_rising"],
                "RS_vs_Nifty50": round(row_dict["rs_vs_nifty50"], 2), "Independent_Alpha_Pass": row_dict["independent_alpha_pass"],
                "Emerging_Recovery_Aligned": row_dict["emerging_recovery_aligned"], "Up_Down_Vol_Ratio": row_dict["up_down_ratio"],
                "Extension_Score": row_dict["Composite_Score"] / 10.0, "Extension_From_50DMA": row_dict["extension_from_50dma"],
                "CHOP_avg": row_dict["chop_avg"], "Weekly_CHOP": row_dict["weekly_chop"], "Whipsaws_50d": row_dict["whipsaw_50ma"],
                "ROE": row_dict["ROE"], "Debt_to_Equity": row_dict["Debt_to_Equity"], "Sector": row_dict["Sector"], "Theme": row_dict["Theme"],
                "Screeners": row_dict["Screeners"], "Rejection_Reason": reason,
                "Data_Source": row_dict["Data_Source"], "Data_Flags": row_dict["Data_Flags"], "Delivery_Below_Threshold": row_dict["delivery_below_threshold"],
                "Market_Cap_Cr": row_dict["Market_Cap_Cr"], "Cap_Category": row_dict["Cap_Category"],
                "Factors_Passed": ",".join(passed_factors), "Factor_Count": len(passed_factors),
                "Factor_Score": round(weighted_score, 2),  # weighted 0-100 score (replaces old 0-8 count)
                "Factor_Details": json.dumps(factor_details),
                "Delivery_Pct": row_dict["delivery_pct"], "Final_Rank": rank_pos, "is_ipo": row_dict["is_ipo"],
                "is_exceptional_bull": row_dict["is_exceptional_bull"],
                "is_vcp_setup": row_dict.get("is_vcp_setup", False),
                "stage2_aligned": row_dict.get("stage2_aligned", False),
                "sma_50": row_dict.get("sma_50", 0.0),
                "sma_200": row_dict.get("sma_200", 0.0),
                "high_52w": row_dict.get("high_52w", 0.0),
                "low_52w": row_dict.get("low_52w", 0.0),
                "sales_growth": row_dict.get("sales_growth", 0.0),
                "profit_growth": row_dict.get("profit_growth", 0.0),
                "roce_3yr": row_dict.get("roce_3yr", 0.0)
            }
            eligible_processed_records_out.append(record)



    # Combine ranked eligible/watchlist records and hard-rejected records
    all_records = eligible_processed_records_out + rejected_records
    df_selected = pd.DataFrame(all_records)


    # Sort final dataframe: Entry_Eligible first, then Rank
    if not df_selected.empty:
        df_selected["sort_eligible"] = df_selected["Entry_Eligible"].astype(int)
        df_selected = df_selected.sort_values(
            by=["sort_eligible", "Final_Rank"],
            ascending=[False, True]
        ).drop(columns=["sort_eligible"]).reset_index(drop=True)
    else:
        df_selected = pd.DataFrame(columns=[
            "Symbol", "Close", "Volume", "Composite_Score", "Opportunity_Score",
            "Weighted_Score", "Final_Score",
            "Tier", "Stage", "Entry_Eligible", "ATR_14", "NATR_14", "NATR_Trend",
            "ADX_14", "ADX_Bullish", "OBV_Rising", "RS_vs_Nifty50", "Independent_Alpha_Pass", "Emerging_Recovery_Aligned",
            "Up_Down_Vol_Ratio", "Extension_Score", "Extension_From_50DMA", "CHOP_avg", "Weekly_CHOP",
            "Whipsaws_50d", "ROE", "Debt_to_Equity", "Sector", "Theme", "Screeners",
            "Rejection_Reason", "Data_Source", "Data_Flags", "Delivery_Below_Threshold",
            "Market_Cap_Cr", "Cap_Category", "Factors_Passed", "Factor_Count", "Factor_Score",
            "Factor_Details", "Delivery_Pct", "Final_Rank", "is_ipo", "is_exceptional_bull",
            "is_vcp_setup", "stage2_aligned", "sma_50", "sma_200", "high_52w", "low_52w",
            "sales_growth", "profit_growth", "roce_3yr"
        ])

    qualified = len(df_selected[df_selected['Entry_Eligible'] == True]) if not df_selected.empty else 0
    t1 = len(df_selected[df_selected['Tier'] == 'TIER 1 — HIGH CONVICTION']) if not df_selected.empty else 0
    t2 = len(df_selected[df_selected['Tier'] == 'TIER 2 — MEDIUM CONVICTION']) if not df_selected.empty else 0
    t3 = len(df_selected[df_selected['Tier'] == 'TIER 3 — LOW CONVICTION']) if not df_selected.empty else 0
    log_success(f"Weighted Ranking complete. Top {TOP_N_STOCKS} selected: T1={t1} | T2={t2} | T3={t3} | Total eligible={qualified} (from {len(df_selected)} scanned)")
    return df_selected

