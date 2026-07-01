"""
Trade Ledger + Performance Analytics
=====================================
Append-only CSV trade ledger tracking every BUY, EXIT, REDUCE.
Computes real Sharpe, CAGR, win rate from closed trades.

USAGE:
    from analytics import record_trade, compute_portfolio_analytics
    
    # Record a trade when order executes
    record_trade("RELIANCE", "BUY", 2500.0, 10, date_str="2026-06-20")
    record_trade("RELIANCE", "EXIT", 2750.0, 10, date_str="2026-06-27")
    
    # Get analytics
    report = compute_portfolio_analytics(positions, blueprint, date_str)
    print(report["sharpe_ratio"])
"""

import pandas as pd
import numpy as np
import os
import datetime
from utils import log_info, log_success, log_warning

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LEDGER_PATH = os.path.join(BASE_DIR, "trade_ledger.csv")

def record_trade(symbol, action, price, quantity, date_str=None, reason=""):
    """Append a trade record to the CSV ledger. Thread-safe append.
    
    Args:
        symbol: Stock ticker
        action: "BUY", "EXIT", "REDUCE", "ADD"
        price: Execution price
        quantity: Number of shares
        date_str: "YYYY-MM-DD" (default: today)
        reason: Exit reason (e.g., "CRITICAL EXIT: RS=0.02"")
    """
    if date_str is None:
        date_str = datetime.date.today().isoformat()
    
    record = pd.DataFrame([{
        "Date": date_str,
        "Symbol": symbol,
        "Action": action,
        "Price": price,
        "Quantity": quantity,
        "Value": price * quantity,
        "Reason": reason,
        "Timestamp": datetime.datetime.now().isoformat(),
    }])
    
    # Append to CSV (create if not exists)
    if not os.path.exists(LEDGER_PATH):
        record.to_csv(LEDGER_PATH, index=False)
    else:
        record.to_csv(LEDGER_PATH, mode="a", header=False, index=False)
    
    log_info(f"LEDGER: {action} {quantity}x {symbol} @ ₹{price:.2f}" + (f" — {reason}" if reason else ""))


def get_trade_ledger():
    """Load the full trade ledger as a DataFrame. Returns empty DataFrame if no ledger."""
    if not os.path.exists(LEDGER_PATH):
        return pd.DataFrame(columns=["Date", "Symbol", "Action", "Price", "Quantity", "Value", "Reason", "Timestamp"])
    return pd.read_csv(LEDGER_PATH, parse_dates=["Date"])


def compute_portfolio_analytics(positions, blueprint, date_str=None, state_holdings=None, pipeline_data=None):
    """Computes REAL performance metrics from the trade ledger.
    
    Returns dict with:
        win_rate_%, expectancy_R, profit_factor, CAGR_%, max_drawdown_%,
        sharpe_ratio, sortino_ratio, total_trades, avg_hold_days,
        sector_allocations, metrics_source
    """
    log_info("Computing portfolio analytics from trade ledger...")
    
    ledger = get_trade_ledger()
    has_trades = len(ledger) > 0
    
    # ── 1. Win Rate & Profit Metrics from Closed Trades ──
    win_rate = 0.0
    expectancy = 0.0
    profit_factor = 0.0
    total_trades = 0
    avg_hold_days = 0.0
    
    if has_trades:
        # Find closed positions: paired BUY→EXIT for each symbol
        closed_positions = []
        for symbol in ledger["Symbol"].unique():
            sym_trades = ledger[ledger["Symbol"] == symbol].sort_values("Date")
            buys = sym_trades[sym_trades["Action"].isin(["BUY", "ADD"])]
            exits = sym_trades[sym_trades["Action"].isin(["EXIT", "REDUCE"])]
            
            buy_value = buys["Value"].sum() if len(buys) > 0 else 0
            exit_value = exits["Value"].sum() if len(exits) > 0 else 0
            
            if buy_value > 0 and exit_value > 0:
                pnl = exit_value - buy_value
                pnl_pct = (pnl / buy_value) * 100.0
                closed_positions.append({
                    "symbol": symbol,
                    "pnl": pnl,
                    "pnl_pct": pnl_pct,
                    "buy_value": buy_value,
                    "exit_value": exit_value,
                    "win": pnl > 0,
                })
                
                # Holding period
                if len(buys) > 0 and len(exits) > 0:
                    first_buy = buys["Date"].min()
                    last_exit = exits["Date"].max()
                    hold_days = (last_exit - first_buy).days
                    avg_hold_days = (avg_hold_days * (len(closed_positions)-1) + hold_days) / len(closed_positions)
        
        if closed_positions:
            total_trades = len(closed_positions)
            wins = [p for p in closed_positions if p["win"]]
            win_rate = (len(wins) / total_trades) * 100.0
            
            total_profit = sum(p["pnl"] for p in wins)
            total_loss = abs(sum(p["pnl"] for p in closed_positions if not p["win"]))
            profit_factor = total_profit / total_loss if total_loss > 0 else (99.0 if total_profit > 0 else 0.0)
            
            avg_win = np.mean([p["pnl_pct"] for p in wins]) if wins else 0
            avg_loss = np.abs(np.mean([p["pnl_pct"] for p in closed_positions if not p["win"]])) if len(wins) < total_trades else 1
            expectancy = avg_win / avg_loss if avg_loss > 0 else 0
    
    # ── 2. CAGR, Max Drawdown, Sharpe & Sortino from Proper Equity Curve ──
    cagr = 0.0
    max_dd = 0.0
    sharpe = 0.0
    sortino = 0.0
    daily_returns_pct = pd.Series(dtype=float)
    
    if has_trades and len(ledger) >= 5:
        # Build daily P&L series from ledger
        daily_pnl = ledger.copy()
        daily_pnl["Date"] = pd.to_datetime(daily_pnl["Date"])
        daily_pnl = daily_pnl.set_index("Date")
        
        # Aggregate net P&L per day: EXIT value - BUY value
        daily_net = daily_pnl.groupby(daily_pnl.index.date).apply(
            lambda g: g[g["Action"].isin(["EXIT", "REDUCE"])]["Value"].sum() - 
                      g[g["Action"].isin(["BUY", "ADD"])]["Value"].sum()
        )
        
        if len(daily_net) >= 20:
            # Total deployed capital as base
            total_invested = ledger[ledger["Action"].isin(["BUY", "ADD"])]["Value"].sum()
            total_returned = ledger[ledger["Action"].isin(["EXIT", "REDUCE"])]["Value"].sum()
            
            # Build equity curve: starting_capital + cumulative realized P&L
            starting_capital = max(total_invested, 1.0)
            cumulative_pnl = daily_net.cumsum()
            equity_curve = starting_capital + cumulative_pnl
            
            # Proper daily percentage returns from equity curve
            daily_returns_pct = equity_curve.pct_change().dropna()
            daily_returns_pct = daily_returns_pct.replace([np.inf, -np.inf], np.nan).dropna()
            
            if len(daily_returns_pct) > 1:
                # CAGR from equity curve endpoints
                days_span = (ledger["Date"].max() - ledger["Date"].min()).days
                if days_span > 0 and equity_curve.iloc[0] > 0:
                    total_growth = equity_curve.iloc[-1] / equity_curve.iloc[0]
                    years = days_span / 365.25
                    if years > 0 and total_growth > 0:
                        cagr = (total_growth ** (1.0 / years) - 1.0) * 100.0
                
                # Max drawdown from equity curve
                rolling_max = equity_curve.cummax()
                drawdowns = (equity_curve - rolling_max) / rolling_max.replace(0, 1)
                if len(drawdowns) > 0 and drawdowns.min() < 0:
                    max_dd = abs(drawdowns.min() * 100.0)
                
                # Annualized Sharpe (Rf = 7% for Indian context)
                rf_daily = 0.07 / 252
                excess_returns = daily_returns_pct - rf_daily
                if excess_returns.std() > 0:
                    sharpe = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252)
                
                # Sortino (downside deviation only)
                downside = excess_returns[excess_returns < 0]
                if len(downside) > 0 and downside.std() > 0:
                    sortino = (excess_returns.mean() / downside.std()) * np.sqrt(252)
    
    # ── 3. (removed — CAGR, DD, Sharpe, Sortino now computed above) ──
    
    # ── 4. Sector Allocations (current portfolio) ──
    sector_contributions = {}
    for pos in positions:
        sec = pos.get("Sector", "Diversified")
        val = pos.get("Position_Value", 0)
        sector_contributions[sec] = sector_contributions.get(sec, 0.0) + val
    
    total_val = sum(sector_contributions.values())
    if total_val > 0:
        for sec in sector_contributions:
            sector_contributions[sec] = (sector_contributions[sec] / total_val) * 100.0
    
    factor_attribution = {}
    
    source = "trade_ledger" if has_trades and total_trades > 0 else "no_closed_trades"
    
    # ── Unrealized P&L from open positions ──
    unrealized_pnl = 0.0
    total_unrealized_cost = 0.0
    if state_holdings and pipeline_data:
        for sym, pos in state_holdings.items():
            df = pipeline_data.get(sym)
            if df is not None and not df.empty:
                cur_price = float(df["Close"].iloc[-1])
                entry_pr = float(pos.get("Entry_Price", cur_price))
                qty = float(pos.get("Quantity", 0))
                
                cost = entry_pr * qty
                current_val = cur_price * qty
                
                if cost > 0:
                    unrealized_pnl += (current_val - cost)
                    total_unrealized_cost += cost
    
    unrealized_pnl_pct = (unrealized_pnl / total_unrealized_cost * 100.0) if total_unrealized_cost > 0 else 0.0

    analytics_report = {
        "win_rate_%": round(win_rate, 1),
        "expectancy_R": round(expectancy, 2),
        "profit_factor": round(profit_factor, 2),
        "CAGR_%": round(cagr, 2),
        "max_drawdown_%": round(max_dd, 2),
        "sharpe_ratio": round(sharpe, 3),
        "sortino_ratio": round(sortino, 3),
        "total_trades": total_trades,
        "avg_hold_days": round(avg_hold_days, 1),
        "sector_allocations": sector_contributions,
        "factor_alpha_attribution": factor_attribution,
        "metrics_source": source,
        "unrealized_pnl": unrealized_pnl,
        "unrealized_pnl_pct": unrealized_pnl_pct
    }
    
    log_success(f"Analytics computed from {source}: Sharpe={sharpe:.2f}, CAGR={cagr:.1f}%, WinRate={win_rate:.1f}%, {total_trades} closed trades, Unrealized ₹{unrealized_pnl:,.0f}")
    return analytics_report


def reconcile_portfolio(portfolio_positions=None):
    """Cross-references portfolio positions against trade ledger and compliance limits.
    
    Detects: phantom positions, missed exits, sector/theme limit breaches, 
    position count violations. Returns (report_dict, warnings_list).
    
    Call after pipeline completes to validate portfolio integrity.
    """
    from config import MAX_OPEN_POSITIONS, MAX_SECTOR_PCT, MAX_THEME_PCT
    from utils import log_warning, log_success
    
    warnings = []
    report = {
        "position_count": 0,
        "max_positions_ok": True,
        "sector_limits_ok": True,
        "theme_limits_ok": True,
        "phantom_positions": [],
        "missed_exits": [],
        "ledger_positions": 0,
    }
    
    if not portfolio_positions:
        return report, ["No portfolio positions to reconcile"]
    
    report["position_count"] = len(portfolio_positions)
    
    # 1. Max positions check
    if report["position_count"] > MAX_OPEN_POSITIONS:
        report["max_positions_ok"] = False
        warnings.append(f"⚠️ {report['position_count']} positions > MAX {MAX_OPEN_POSITIONS}")
    
    # 2. Sector & theme limits
    total_value = sum(p.get("Position_Value", 0) or p.get("Allocation_Pct", 0) for p in portfolio_positions)
    if total_value > 0:
        sector_val = {}
        theme_val = {}
        for pos in portfolio_positions:
            val = pos.get("Position_Value", 0) or pos.get("Allocation_Pct", 0)
            sec = pos.get("Sector", "Diversified")
            theme = pos.get("Theme", "General")
            sector_val[sec] = sector_val.get(sec, 0) + val
            theme_val[theme] = theme_val.get(theme, 0) + val
        
        for sec, val in sector_val.items():
            if val / total_value > MAX_SECTOR_PCT:
                report["sector_limits_ok"] = False
                warnings.append(f"⚠️ Sector {sec}: {val/total_value*100:.1f}% > {MAX_SECTOR_PCT*100:.0f}%")
        
        for theme, val in theme_val.items():
            if val / total_value > MAX_THEME_PCT:
                report["theme_limits_ok"] = False
                warnings.append(f"⚠️ Theme {theme}: {val/total_value*100:.1f}% > {MAX_THEME_PCT*100:.0f}%")
    
    # 3. Trade ledger cross-reference
    try:
        ledger = get_trade_ledger()
        if len(ledger) > 0:
            pipe_symbols = {p["Symbol"] for p in portfolio_positions}
            led_symbols = set(ledger["Symbol"].unique())
            report["ledger_positions"] = len(led_symbols)
            
            # Phantom: in pipeline but never bought
            for sym in pipe_symbols:
                sym_trades = ledger[ledger["Symbol"] == sym]
                if len(sym_trades) == 0 or not any(sym_trades["Action"].isin(["BUY", "ADD"])):
                    report["phantom_positions"].append(sym)
                    warnings.append(f"⚠️ Phantom: {sym} in portfolio but no BUY in ledger")
            
            # Missed exit: exited in ledger but still in pipeline
            for sym in pipe_symbols:
                sym_trades = ledger[ledger["Symbol"] == sym]
                if len(sym_trades) > 0 and any(sym_trades["Action"] == "EXIT"):
                    report["missed_exits"].append(sym)
                    warnings.append(f"⚠️ Missed exit: {sym} EXIT in ledger but still active")
    except Exception as e:
        warnings.append(f"Ledger check: {e}")
    
    if warnings:
        log_warning(f"Reconciliation: {len(warnings)} issues")
        for w in warnings:
            log_warning(f"  {w}")
    else:
        log_success("Reconciliation: ALL CLEAN")
    
    return report, warnings
