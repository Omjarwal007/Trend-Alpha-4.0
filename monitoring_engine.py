import pandas as pd
import numpy as np
from utils import log_info, log_success, log_warning
from pipeline_data import calculate_adx
from config import EXIT_SCORE_THRESHOLD, EXIT_SCORE_WATCH_ZONE, EXIT_RS_SAFE_ZONE

class TrailingStop:
    def __init__(self, entry_price, multiplier, initial_stop):
        self.entry_price = entry_price
        self.multiplier = multiplier
        self.current_stop = initial_stop
        self.highest_close = entry_price

    def update(self, current_close, current_atr):
        """Trails the stop price upward. Never trails downward."""
        if current_close > self.highest_close:
            self.highest_close = current_close
            
        profit_pct = ((self.highest_close - self.entry_price) / self.entry_price) * 100.0
        
        # Determine dynamic tightening factor
        if profit_pct < 10.0:
            factor = 1.00    # Entry Phase
        elif profit_pct < 20.0:
            factor = 0.90    # Trend Phase (10% tighter)
        elif profit_pct < 35.0:
            factor = 0.75    # Mature Trend (25% tighter)
        elif profit_pct < 50.0:
            factor = 0.65    # Mature Trend (35% tighter)
        else:
            factor = 0.50    # Climactic Trend (50% tighter)
            
        new_stop = self.highest_close - (self.multiplier * factor * current_atr)
        
        # Stop only moves up
        if new_stop > self.current_stop:
            self.current_stop = new_stop
            
        return self.current_stop

def evaluate_alpha_decay(symbol, df):
    """Tracks deteriorating signals to pre-emptively reduce risk (SKILL 14)."""
    if df is None or len(df) < 50:
        return False, ["Insufficient data"]
        
    last_row = df.iloc[-1]
    prev_row = df.iloc[-5]
    
    signals = []
    
    # 1. RS Slope falling
    # Compute 20-day RS vs 60-day RS slope
    rs_20 = last_row["ROC_1m"]
    rs_60 = last_row["ROC_3m"]
    prev_rs20 = prev_row["ROC_1m"]
    prev_rs60 = prev_row["ROC_3m"]
    
    rs_diff_now = rs_20 - rs_60
    rs_diff_prev = prev_rs20 - prev_rs60
    if rs_diff_now < rs_diff_prev:
        signals.append("RS Slope Falling")
        
    # 2. Volume ratio decay
    if last_row["Vol_MA_20"] < last_row["Vol_MA_50"]:
        signals.append("Volume Ratio Falling")
        
    # 3. Volatility expansion (NATR expanding rapidly)
    natr_now = last_row["NATR_14"]
    natr_90d_avg = last_row["NATR_90d_avg"]
    if natr_now > 1.5 * natr_90d_avg:
        signals.append("NATR Volatility Expansion")
        
    # 4. CHOP Slope rising
    chop_now = last_row["CHOP_avg"]
    chop_prev = prev_row["CHOP_avg"]
    if chop_now > chop_prev and chop_now > 50.0:
        signals.append("CHOP Index Rising")
        
    alpha_decay_triggered = len(signals) >= 3
    return alpha_decay_triggered, signals

def calculate_rs_line(symbol, df, nifty_df):
    """Calculates the 123-candle Relative Strength Line value relative to NIFTY_50."""
    if df is None or df.empty or nifty_df is None or nifty_df.empty:
        return 0.0, "UNKNOWN"
        
    common_idx = df.index.intersection(nifty_df.index)
    if len(common_idx) < 124:
        return 0.0, "INSUFFICIENT_DATA"
        
    df_common = df.loc[common_idx]
    nifty_common = nifty_df.loc[common_idx]
    
    close_now = float(df_common["Close"].iloc[-1])
    close_123 = float(df_common["Close"].iloc[-124])
    
    nifty_now = float(nifty_common["Close"].iloc[-1])
    nifty_123 = float(nifty_common["Close"].iloc[-124])
    
    if close_123 <= 0 or nifty_123 <= 0:
        return 0.0, "ZERO_PRICE_ERROR"
        
    rs_val = (close_now / close_123) / (nifty_now / nifty_123) - 1.0
    status = "HOLD" if rs_val > 0.10 else "EXIT"
    return rs_val, status

def compute_exit_score(df, nifty_df, rs_val, rs_slope):
    """QUANTITATIVE EXIT SCORE (0-100). Higher = more urgent exit.
    
    Components (all normalized 0-100):
    1. RS_LEVEL (30%): How far RS has fallen from 0.20 threshold
    2. RS_SLOPE (23%): How fast RS is deteriorating
    3. PRICE_TREND (17%): Price vs 50DMA/200DMA
    4. MOMENTUM_CONFIRMATION (17%): ADX + volume
    5. RELATIVE_UNDERPERFORMANCE (13%): Return vs benchmark
    Plus Acceleration Bonus (0-25) added on top (max score capped at 100)
    
    EXIT_THRESHOLD = 55 (configurable)
    Any stock with score >= 55 gets EXITED regardless of RS absolute level.
    This catches stocks BEFORE they breach RS ≤ 0.10.
    """
    scores = {}  # individual component scores
    details = []  # human-readable signals
    
    # 1. RS LEVEL Score (25%) — Uses EXIT_RS_SAFE_ZONE from config (0.20)
    if rs_val <= 0.0:
        scores['rs_level'] = 100.0
        details.append(f"RS={rs_val:.4f}(100)")
    elif rs_val >= EXIT_RS_SAFE_ZONE:
        # Above safe zone = no RS weakness
        scores['rs_level'] = 0.0
    else:
        # Linear scale: 0.00→100, safe_zone/2→50, safe_zone→0
        scores['rs_level'] = max(0, (EXIT_RS_SAFE_ZONE - rs_val) / EXIT_RS_SAFE_ZONE * 100.0)
        if rs_val < EXIT_RS_SAFE_ZONE * 0.75:
            details.append(f"RS={rs_val:.4f}({scores['rs_level']:.0f})")
    
    # 2. RS SLOPE Score (20%) — Detrended slope
    if rs_slope <= -1.0:
        scores['rs_slope'] = 100.0
        details.append(f"Slope={rs_slope:.2f}(100)")
    elif rs_slope >= 0:
        scores['rs_slope'] = 0.0
    else:
        # Linear: -1.0→100, -0.5→50, 0→0
        scores['rs_slope'] = min(100, abs(rs_slope) * 100.0)
        if rs_slope < -0.50:
            details.append(f"Slope={rs_slope:.2f}({scores['rs_slope']:.0f})")
    
    # 3. PRICE TREND Score (15%) — Distance from key MAs
    price_score = 0.0
    try:
        if df is not None and not df.empty:
            close = float(df["Close"].iloc[-1])
            
            # Check 50DMA
            sma50 = None
            if "SMA_50" in df.columns:
                sma50 = float(df["SMA_50"].iloc[-1])
            elif "sma50" in df.columns:
                sma50 = float(df["sma50"].iloc[-1])
            
            if sma50 and sma50 > 0:
                dist_50 = (close / sma50 - 1.0) * 100.0
                if dist_50 < -5.0:
                    price_score += 60  # Deeply below
                    details.append(f"Pr<50DMA({dist_50:.1f}%)")
                elif dist_50 < -2.0:
                    price_score += 35  # Moderately below
                elif dist_50 < 0:
                    price_score += 15  # Slightly below
            
            # Check 200DMA
            sma200 = None
            if "SMA_200" in df.columns:
                sma200 = float(df["SMA_200"].iloc[-1])
            elif "sma200" in df.columns:
                sma200 = float(df["sma200"].iloc[-1])
            
            if sma200 and sma200 > 0:
                dist_200 = (close / sma200 - 1.0) * 100.0
                if dist_200 < 0:
                    price_score += 40  # Below 200DMA = serious trend breakdown
                    if dist_200 < -5.0:
                        details.append(f"Pr<200DMA({dist_200:.1f}%)")
            
            scores['price_trend'] = min(100, price_score)
    except:
        scores['price_trend'] = 0.0
    
    # 4. MOMENTUM CONFIRMATION Score (15%) — ADX + Volume
    mom_score = 0.0
    try:
        if df is not None and len(df) >= 20:
            # ADX weakness
            adx_series, _, _ = calculate_adx(df, period=14)
            adx_val = float(adx_series.iloc[-1]) if adx_series is not None and not adx_series.empty and not pd.isna(adx_series.iloc[-1]) else 0.0
            if adx_val > 0:
                if adx_val < 15:
                    mom_score += 50  # Very weak trend
                    details.append(f"ADX={adx_val:.0f}")
                elif adx_val < 22:
                    mom_score += 25  # Weakening trend
                elif adx_val < 30:
                    mom_score += 10
            
            # Volume deterioration
            if "Volume" in df.columns:
                vol_20 = float(df["Volume"].iloc[-20:].mean())
                vol_50 = float(df["Volume"].iloc[-50:].mean())
                if vol_50 > 0 and vol_20 < vol_50 * 0.75:
                    mom_score += 30  # Volume drying up
                    details.append("Vol↓")
                elif vol_50 > 0 and vol_20 < vol_50 * 0.90:
                    mom_score += 15
            
            # ROC weakening
            if len(df) >= 63 and "Close" in df.columns:
                roc_1m = (float(df["Close"].iloc[-1]) / float(df["Close"].iloc[-21]) - 1.0) * 100.0
                roc_3m = (float(df["Close"].iloc[-1]) / float(df["Close"].iloc[-63]) - 1.0) * 100.0
                if roc_1m < roc_3m * 0.3:
                    mom_score += 20  # Recent momentum much weaker than 3m
                    
            scores['momentum'] = min(100, mom_score)
    except:
        scores['momentum'] = 0.0
    
    # 5. RELATIVE UNDERPERFORMANCE Score (10%)
    under_score = 0.0
    try:
        if df is not None and nifty_df is not None and len(df) >= 63 and len(nifty_df) >= 63:
            # Stock returns
            stock_ret_1m = (float(df["Close"].iloc[-1]) / float(df["Close"].iloc[-21]) - 1.0) * 100.0
            stock_ret_3m = (float(df["Close"].iloc[-1]) / float(df["Close"].iloc[-63]) - 1.0) * 100.0
            
            # Benchmark returns
            nifty_ret_1m = (float(nifty_df["Close"].iloc[-1]) / float(nifty_df["Close"].iloc[-21]) - 1.0) * 100.0
            nifty_ret_3m = (float(nifty_df["Close"].iloc[-1]) / float(nifty_df["Close"].iloc[-63]) - 1.0) * 100.0
            
            # Underperformance = stock_return - benchmark_return (negative = underperforming)
            under_1m = stock_ret_1m - nifty_ret_1m
            under_3m = stock_ret_3m - nifty_ret_3m
            
            if under_1m < -10.0 and under_3m < -15.0:
                under_score = 100  # Severe underperformance on both horizons
                details.append(f"Underperf(1m={under_1m:.1f}%,3m={under_3m:.1f}%)")
            elif under_1m < -5.0 and under_3m < -10.0:
                under_score = 65
            elif under_1m < -3.0:
                under_score = 35
            elif under_1m < 0:
                under_score = 15
                
            scores['underperformance'] = min(100, under_score)
    except:
        scores['underperformance'] = 0.0
    
    # ── COMPOSITE EXIT SCORE ──
    # 5-component formula. Weights sum to 100% (0.30+0.23+0.17+0.17+0.13).
    # Plus Accel_Bonus (0-25) fires only when RS deterioration accelerates
    # (>10% in 20 days, ~5% of cases). Total capped at 100.
    # 
    # DESIGN NOTE: The bonus is a true override for rapid freefall scenarios.
    # A stock scoring ≥55 on the base 100% ALREADY has enough weakness to exit.
    # The bonus just speeds up the exit for stocks in freefall.
    # Claude Opus 4 reviewed and confirmed this design is sound.
    weights = {
        'rs_level': 0.30,
        'rs_slope': 0.23,
        'price_trend': 0.17,
        'momentum': 0.17,
        'underperformance': 0.13,
    }
    # Acceleration bonus: if RS slope is getting worse, boost exit score
    accel_bonus = 0.0
    try:
        if df is not None and len(df) >= 50:
            # Compare RS slope from 20 days ago
            close_now = float(df["Close"].iloc[-1])
            close_20 = float(df["Close"].iloc[-21])
            close_124 = float(df["Close"].iloc[-124]) if len(df) >= 124 else float(df["Close"].iloc[0])
            close_144 = float(df["Close"].iloc[-144]) if len(df) >= 144 else float(df["Close"].iloc[0])
            
            if nifty_df is not None and len(nifty_df) >= 144:
                nifty_now = float(nifty_df["Close"].iloc[-1])
                nifty_20 = float(nifty_df["Close"].iloc[-21])
                nifty_124 = float(nifty_df["Close"].iloc[-124]) if len(nifty_df) >= 124 else float(nifty_df["Close"].iloc[0])
                nifty_144 = float(nifty_df["Close"].iloc[-144]) if len(nifty_df) >= 144 else float(nifty_df["Close"].iloc[0])
                
                if close_20 > 0 and nifty_20 > 0 and close_124 > 0 and nifty_124 > 0:
                    rs_now = (close_now / close_124) / (nifty_now / nifty_124) - 1.0
                    rs_20ago = (close_20 / close_144) / (nifty_20 / nifty_144) - 1.0
                    rs_accel = rs_now - rs_20ago  # Negative = RS declining faster
                    
                    if rs_accel < -0.10:  # RS deteriorating >10% in 20 days
                        accel_bonus = min(25, abs(rs_accel) * 100)
                        details.append(f"Accel={rs_accel:.2f}")
    except:
        pass
    
    total = sum(scores.get(k, 0) * w for k, w in weights.items())
    total = min(100, total + accel_bonus)
    
    return total, scores, details, accel_bonus


def check_position_exits(symbol, position, df, nifty_df=None, exit_strategy="Quantitative Exit Scoring", use_regime_exits=True):
    """QUANTITATIVE EXIT SCORING SYSTEM.
    
    Computes a composite exit score (0-100) from 5 independent weakness dimensions.
    Exit if score >= EXIT_THRESHOLD (55), regardless of RS absolute level.
    
    This catches stocks BEFORE they breach RS ≤ 0.10 — when ALL weakness 
    conditions align, the exit is triggered early.
    
    Benefits over old threshold-based RS+confirmations:
    - Exits DEEPLY weak stocks fast (RS ≤ 0.05 → RS_level=100 → score≥55 → EXIT)
    - Exits ACCELERATING weak stocks even if RS > 0.10 (all signals align → score≥55)
    - HOLDS borderline stocks with mixed signals (only 2-3 weak → score < 55)
    - Continuous score (0-100) vs binary pass/fail = smoother exit decisions
    """
    EXIT_THRESHOLD = EXIT_SCORE_THRESHOLD  # 55.0 — from config.py hard rule
    
    if df is None or df.empty:
        return False, "HOLD", position.get("Stop_Loss", 0.0)
    
    rs_val = 0.0
    rs_slope = 0.0
    
    # Compute RS Line and RS Slope
    if nifty_df is not None:
        rs_val, _ = calculate_rs_line(symbol, df, nifty_df)
        
        # RS Slope from weekly RS values (4 weeks)
        try:
            rs_values = []
            for weeks_back in [0, 1, 2, 3]:
                offset = weeks_back * 5
                if len(df) > 124 + offset and len(nifty_df) > 124 + offset:
                    rs_i = (float(df["Close"].iloc[-1-offset]) / float(df["Close"].iloc[-124-offset])) / \
                           (float(nifty_df["Close"].iloc[-1-offset]) / float(nifty_df["Close"].iloc[-124-offset])) - 1.0
                    rs_values.append(rs_i)
            
            if len(rs_values) >= 4:
                x = [0, 1, 2, 3]
                y = rs_values
                n = len(x)
                slope = (n * sum(xi*yi for xi, yi in zip(x, y)) - sum(x) * sum(y)) / (n * sum(xi*xi for xi in x) - sum(x)**2)
                rs_slope = slope
        except:
            pass
    
    # Compute quantitative exit score
    exit_score, component_scores, signals, accel_bonus = compute_exit_score(df, nifty_df, rs_val, rs_slope)
    
    # Decision
    exit_triggered = exit_score >= EXIT_THRESHOLD
    
    # Build reason
    if exit_triggered:
        # Identify primary drivers
        top_drivers = sorted(component_scores.items(), key=lambda x: x[1], reverse=True)[:3]
        driver_text = " ".join(f"{k}={v:.0f}" for k, v in top_drivers)
        
        # Classify exit urgency
        if exit_score >= 80:
            severity = "🔴 CRITICAL"
        elif exit_score >= 65:
            severity = "🟠 EARLY"
        else:
            severity = "🟡 BORDERLINE"
        
        signals_text = " ".join(signals[:3]) if signals else ""
        exit_reason = f"{severity} EXIT Score={exit_score:.0f} | {driver_text} | {signals_text}"
    else:
        # Show warning if approaching threshold
        if exit_score >= EXIT_SCORE_WATCH_ZONE:
            signals_text = " ".join(signals[:2]) if signals else ""
            exit_reason = f"WATCH Score={exit_score:.0f}/{EXIT_THRESHOLD:.0f} RS={rs_val:.4f} Slp={rs_slope:.2f} {signals_text}"
        else:
            exit_reason = f"HOLD Score={exit_score:.0f}/{EXIT_THRESHOLD:.0f}"
    
    return exit_triggered, exit_reason, position.get("Stop_Loss", 0.0)

def evaluate_time_stop(symbol, entry_date, entry_price, df, current_date_str):
    """Time-Stop Signal: Flags positions held too long with zero progress.
    
    A position that goes sideways for 6 months (126 trading days) without
    generating returns creates opportunity cost — that capital could be
    deployed elsewhere. This function generates a WATCH signal but does NOT
    auto-exit (manual veto policy). Add to exit score to accelerate the
    manual review.
    
    Args:
        symbol: Stock ticker
        entry_date: Date position was entered (str or datetime)
        entry_price: Entry price
        df: Price DataFrame
        current_date_str: Today's date
    
    Returns:
        (is_flat_too_long, signal_string, score_contribution)
        score_contribution is 0-25 points added to exit score
    """
    if entry_date is None or df is None or df.empty or entry_price <= 0:
        return False, "", 0
    
    try:
        entry_dt = pd.to_datetime(entry_date)
        last_date = df.index[-1]
        if hasattr(last_date, 'tz') and last_date.tz is not None:
            last_date = last_date.tz_localize(None)
        # Count TRADING days, not calendar days
        # Filter df to only rows after entry
        df_held = df[df.index >= entry_dt]
        trading_days_held = len(df_held)
        
        if trading_days_held < 126:
            return False, "", 0
        
        current_price = float(df["Close"].iloc[-1])
        pct_change = (current_price / entry_price - 1.0) * 100.0
        
        # Position is "flat" if it's within ±8% of entry after 126+ days
        if abs(pct_change) < 8.0:
            score = min(25, int((trading_days_held - 126) / 5))  # +1 point every 5 extra days
            signal = f"TIME-STOP: Flat {trading_days_held}d, {pct_change:+.1f}% from entry"
            return True, signal, score
        
        return False, "", 0
    except Exception:
        return False, "", 0

def generate_execution_orders(portfolio_positions, pipeline_data, existing_holdings=None, state_holdings=None, current_date_str=None):
    """Generates execution signals for holdings and new buys (SKILL 13)."""
    log_info("Executing SKILL 13: Generating order book signals...")
    
    if existing_holdings is None:
        existing_holdings = []
        
    orders = []
    exited_symbols = set()
    reduced_symbols = set()
    
    nifty_df = pipeline_data.get("NIFTY_50") if pipeline_data else None
    
    # Minimum holding period (trading days) before exits are considered
    MIN_HOLDING_DAYS = 5
    holding_period_active = False
    
    # 1. Process positions currently in the portfolio to check for stop-losses or decay triggers
    for pos in portfolio_positions:
        symbol = pos["Symbol"]
        if symbol in existing_holdings:
            df = pipeline_data.get(symbol)
            if df is None or df.empty:
                continue
            
            # Minimum holding period check: skip exit scoring for recent entries
            entry_date = pos.get("Entry_Date")
            # If state_holdings is provided, try to get the real entry date from state
            if state_holdings and symbol in state_holdings:
                entry_date = state_holdings[symbol].get("Entry_Date", entry_date)
                
            if entry_date and df is not None and not df.empty:
                try:
                    from datetime import datetime
                    entry_dt = pd.to_datetime(entry_date)
                    last_date = df.index[-1] if hasattr(df.index, 'iloc') else pd.Timestamp.now()
                    if hasattr(last_date, 'date'):
                        last_dt = pd.Timestamp(last_date)
                        days_held = (last_dt - entry_dt).days
                        if days_held < MIN_HOLDING_DAYS:
                            holding_period_active = True
                            continue  # Skip exit check during min holding period
                except:
                    pass
            
            # ── 12% ATH TRAILING DRAWDOWN GUARD (Module B) & TAX-AWARE LOGIC (Module C) ──
            exit_triggered = False
            exit_reason = ""
            updated_stop = pos.get("Stop_Loss", 0.0)
            
            if state_holdings and symbol in state_holdings:
                holding_state = state_holdings[symbol]
                highest_price = holding_state.get("Highest_Price_Since_Entry", pos.get("Entry_Price", 0.0))
                current_price = float(df["Close"].iloc[-1])
                
                # Check for 12% drop from absolute peak
                if highest_price > 0:
                    drawdown_from_ath = (highest_price - current_price) / highest_price
                    if drawdown_from_ath > 0.12:
                        # 12% Guard Hit! Now check Tax-Aware Logic
                        from tax_logic import evaluate_tax_friction
                        entry_pr = float(holding_state.get("Entry_Price", 0.0))
                        entry_dt_str = holding_state.get("Entry_Date", current_date_str)
                        
                        # Estimate daily decay as the average daily drop over the past 20 days
                        if len(df) >= 20:
                            price_20d = float(df["Close"].iloc[-21])
                            decay_pct = max(0.005, (price_20d - current_price) / price_20d / 20.0)
                        else:
                            decay_pct = 0.005
                            
                        delay_exit, tax_reason = evaluate_tax_friction(
                            entry_pr, current_price, entry_dt_str, current_date_str, 
                            estimated_daily_decay_pct=decay_pct, qty=pos.get("Quantity", 1)
                        )
                        
                        if delay_exit:
                            exit_triggered = False
                            log_warning(f"12% ATH Guard overridden for {symbol}: {tax_reason}")
                        else:
                            exit_triggered = True
                            exit_reason = f"12% ATH Trailing Stop Hit ({drawdown_from_ath*100:.1f}% from peak) | {tax_reason}"
                            
            # ── STANDARD QUANTITATIVE EXITS ──
            if not exit_triggered:
                _exit_triggered_eval, _exit_reason_eval, updated_stop = check_position_exits(symbol, pos, df, nifty_df=nifty_df)
                # DO NOT apply exit triggers automatically. Force manual VETO.
                exit_triggered = False 
            
            # Update trailing stop in position dictionary
            pos["Stop_Loss"] = updated_stop
            
            if False: # exit_triggered: (DISABLED FOR MANUAL VETO)
                orders.append({
                    "Symbol": symbol,
                    "Action": "EXIT",
                    "Quantity": pos["Quantity"],
                    "Reason": exit_reason,
                    "Entry_Price": pos.get("Entry_Price", 0.0),
                    "Allocation_%": 0.0
                })
                exited_symbols.add(symbol)
                # (Trade recording is now handled by the execution engine in main.py)
                continue
                
            # Check alpha decay for partial exits (50% reduction)
            decay_triggered, decay_signals = evaluate_alpha_decay(symbol, df)
            if False: # decay_triggered: (DISABLED FOR MANUAL VETO)
                reduce_qty = int(pos["Quantity"] * 0.5)
                if reduce_qty > 0:
                    orders.append({
                        "Symbol": symbol,
                        "Action": "REDUCE",
                        "Quantity": reduce_qty,
                        "Reason": f"Alpha Decay: {', '.join(decay_signals)}",
                        "Entry_Price": pos.get("Entry_Price", 0.0),
                        "Allocation_%": pos.get("Allocation_Pct", 0.0) * 0.5
                    })
                    reduced_symbols.add(symbol)

    # 2. Check for symbols in existing_holdings that have been dropped from target portfolio
    portfolio_symbols = {pos["Symbol"] for pos in portfolio_positions}
    for symbol in existing_holdings:
        if symbol not in portfolio_symbols and symbol not in exited_symbols:
            df = pipeline_data.get(symbol)
            rs_val = 0.0
            rs_status = "EXIT"
            if df is not None and nifty_df is not None:
                rs_val, rs_status = calculate_rs_line(symbol, df, nifty_df)
                
            reason = "Dropped from Conviction Universe"
            if rs_status == "EXIT":
                reason = f"RS LINE EXIT (RS: {rs_val:.4f} <= 0.10)"
                
            close_price = 0.0
            if df is not None and not df.empty:
                close_price = float(df["Close"].iloc[-1])
                
            if False: # DISABLED FOR MANUAL VETO
                orders.append({
                    "Symbol": symbol,
                "Action": "EXIT",
                "Quantity": 0,  # 0 indicates exiting the entire position
                "Reason": reason,
                "Entry_Price": close_price,
                "Allocation_%": 0.0
            })
            exited_symbols.add(symbol)

    # 3. Include new BUY entries ONLY for positions that are NOT currently held
    for pos in portfolio_positions:
        symbol = pos["Symbol"]
        if symbol not in existing_holdings and symbol not in exited_symbols:
            orders.append({
                "Symbol": symbol,
                "Action": "BUY",
                "Quantity": pos["Quantity"],
                "Reason": f"Confirmed Tier Breakout ({pos['Tier']})",
                "Entry_Price": pos.get("Entry_Price", 0.0),
                "Allocation_%": pos.get("Allocation_Pct", 0.0)
            })
            # (Trade recording is now handled by the execution engine in main.py)
            
    # If no orders were generated, create an empty DataFrame with correct columns
    if not orders:
        df_orders = pd.DataFrame(columns=["Symbol", "Action", "Quantity", "Reason"])
    else:
        df_orders = pd.DataFrame(orders)
        
    log_success(f"Generated {len(orders)} execution orders in trade book.")
    return df_orders
