"""
Pre-Trade Checklist — Validation Before BUY Orders Execute
============================================================
Runs BEFORE new BUY orders are added to the portfolio.
Checks 5 conditions before allowing a new position:
1. Market regime allows new entries
2. Stock not already held
3. No critical exits in the same stock recently
4. Current position count within limits
5. Drawdown governor not in full-stop mode
"""
import os, json
import pandas as pd
from datetime import datetime

from config import OUTPUT_DIR
PIPELINE_BASE = OUTPUT_DIR

def pre_trade_check(symbol, portfolio_positions, regime_status, drawdown_status, date_str=None):
    """Run pre-trade checklist. Returns (passed: bool, warnings: list).
    
    Only blocks if conditions are CRITICAL. Warns on MEDIUM issues.
    """
    from config import MAX_OPEN_POSITIONS
    
    warnings = []
    passed = True
    
    # Check 1: Drawdown governor
    dd_action = drawdown_status.get("action", "NORMAL") if drawdown_status else "NORMAL"
    if dd_action in ["FULL STOP — SYSTEM PAUSE", "DEFENSIVE"]:
        warnings.append(f"🔴 BLOCKED: Drawdown governor = {dd_action}")
        passed = False
    elif dd_action == "REDUCE":
        warnings.append(f"🟡 CAUTION: Drawdown governor = {dd_action} — only replace exits, no net new")
    
    # Check 2: Position count
    current_count = len(portfolio_positions)
    if current_count >= MAX_OPEN_POSITIONS:
        warnings.append(f"🔴 BLOCKED: {current_count} positions ≥ MAX {MAX_OPEN_POSITIONS}")
        passed = False
    elif current_count >= MAX_OPEN_POSITIONS - 2:
        warnings.append(f"🟡 CAUTION: {current_count}/{MAX_OPEN_POSITIONS} positions filled")
    
    # Check 3: Symbol not already held
    held_symbols = {p.get("Symbol", "") for p in portfolio_positions}
    if symbol in held_symbols:
        if str(symbol).isdigit() and len(str(symbol)) == 6:
            warnings.append(f"🟡 SKIP: {symbol} (CORE ETF) already in portfolio")
        else:
            warnings.append(f"🟡 SKIP: {symbol} already in portfolio")
        passed = False
    
    # Check 4: Recent exits in same symbol (don't re-buy too soon)
    try:
        from analytics import get_trade_ledger
        ledger = get_trade_ledger()
        if not ledger.empty:
            sym_exits = ledger[(ledger["Symbol"] == symbol) & (ledger["Action"] == "EXIT")]
            if not sym_exits.empty:
                last_exit = pd.to_datetime(sym_exits["Date"].max())
                days_since = (datetime.now() - last_exit).days
                if days_since < 30:
                    warnings.append(f"🟡 CAUTION: {symbol} was exited {days_since}d ago — 30d cooldown recommended")
    except Exception:
        pass
    
    # Check 5: Market regime
    if regime_status:
        regime = regime_status.get("market_regime", "")
        if regime in ["RISK_OFF", "WEAK"]:
            warnings.append(f"🔴 BLOCKED: Market regime = {regime} — no new entries")
            passed = False
        elif regime == "NEUTRAL":
            warnings.append(f"🟡 CAUTION: Market regime = {regime} — reduce position size")
    
    return passed, warnings


def print_checklist(symbol, passed, warnings):
    """Pretty-print pre-trade checklist."""
    print(f"\n  📋 PRE-TRADE CHECKLIST — {symbol}")
    print(f"  {'─'*50}")
    if passed:
        print(f"  ✅ ALL CHECKS PASSED — Proceed with BUY")
    else:
        print(f"  ❌ BLOCKED — Cannot enter")
    for w in warnings:
        print(f"  {w}")
    print()


def run_pre_trade_checklist(portfolio_positions, regime_status, drawdown_status, orders_df=None, date_str=None):
    """Run pre-trade checklist for all pending BUY orders.
    
    Filters orders_df to only include BUY orders that pass all checks.
    Returns filtered orders_df + list of blocked symbols.
    """
    from config import MAX_OPEN_POSITIONS
    
    if orders_df is None or orders_df.empty:
        return orders_df, []
    
    buys = orders_df[orders_df["Action"] == "BUY"].copy() if "Action" in orders_df.columns else pd.DataFrame()
    if buys.empty:
        return orders_df, []
    
    blocked = []
    approved = []
    
    for _, row in buys.iterrows():
        symbol = row.get("Symbol", "")
        passed, warnings = pre_trade_check(symbol, portfolio_positions, regime_status, drawdown_status, date_str)
        
        if passed:
            approved.append(symbol)
        else:
            blocked.append({"symbol": symbol, "warnings": warnings})
            # Remove from orders_df
            orders_df = orders_df[~((orders_df["Symbol"] == symbol) & (orders_df["Action"] == "BUY"))]
    
    if blocked:
        print(f"\n  🚫 PRE-TRADE CHECKLIST: {len(blocked)} BUY(s) blocked")
        for b in blocked:
            for w in b["warnings"]:
                print(f"  {w}")
    
    return orders_df, blocked
