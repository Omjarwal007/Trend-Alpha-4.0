"""
Portfolio Reconciliation + Backtest vs Live Comparison
======================================================
"""
import pandas as pd
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def get_backtest_results():
    """Load backtest summary from the latest backtest output."""
    backtest_dir = os.path.join(BASE_DIR, "Backtesting")
    if not os.path.isdir(backtest_dir):
        # Try alternative locations
        for f in os.listdir(BASE_DIR):
            if "backtest" in f.lower() and (f.endswith(".csv") or f.endswith(".json")):
                path = os.path.join(BASE_DIR, f)
                if f.endswith(".json"):
                    import json
                    with open(path) as fh:
                        return json.load(fh)
                else:
                    return pd.read_csv(path).to_dict(orient="records")
        return None
    
    # Find latest backtest result file
    bt_files = sorted([f for f in os.listdir(backtest_dir) if f.endswith(".csv") or f.endswith(".json")])
    if not bt_files:
        return None
    
    latest = os.path.join(backtest_dir, bt_files[-1])
    if latest.endswith(".json"):
        import json
        with open(latest) as fh:
            return json.load(fh)
    return pd.read_csv(latest).to_dict(orient="records")


def compare_backtest_vs_live(backtest_ret=None, live_ledger=None, date_range_days=90):
    """Compare backtest performance vs live trade ledger.
    
    Returns:
        dict with: backtest_metrics, live_metrics, tracking_error, alpha, beta
    """
    from analytics import get_trade_ledger, compute_portfolio_analytics
    
    result = {
        "backtest": {"has_data": False},
        "live": {"has_data": False},
        "tracking_error": None,
        "alpha": None,
        "beta": None,
        "difference": {},
    }
    
    # 1. Get live metrics from trade ledger
    ledger = get_trade_ledger()
    if len(ledger) > 0 and "Action" in ledger.columns:
        exits = ledger[ledger["Action"] == "EXIT"]
        buys = ledger[ledger["Action"] == "BUY"]
        if len(exits) > 0 and len(buys) > 0:
            total_cost = buys["Value"].sum()
            total_return = exits["Value"].sum()
            if total_cost > 0:
                live_ret = ((total_return - total_cost) / total_cost) * 100.0
                # Count winning trades
                # Simple: for each symbol compare buy cost vs exit value
                win_count = 0
                total_closed = 0
                for sym in set(exits["Symbol"].unique()) & set(buys["Symbol"].unique()):
                    buy_val = buys[buys["Symbol"] == sym]["Value"].sum()
                    exit_val = exits[exits["Symbol"] == sym]["Value"].sum()
                    if buy_val > 0 and exit_val > 0:
                        total_closed += 1
                        if exit_val > buy_val:
                            win_count += 1
                
                result["live"] = {
                    "has_data": True,
                    "total_return_pct": round(live_ret, 2),
                    "total_closed_trades": total_closed,
                    "win_rate_pct": round((win_count / total_closed * 100), 1) if total_closed > 0 else 0,
                    "total_trades": total_closed,
                }
    
    # 2. Get backtest metrics
    bt = get_backtest_results()
    if bt is not None:
        if isinstance(bt, dict):
            result["backtest"] = {
                "has_data": True,
                "cagr_pct": bt.get("CAGR", bt.get("cagr", 0)),
                "sharpe": bt.get("Sharpe", bt.get("sharpe", 0)),
                "max_dd_pct": bt.get("MaxDD", bt.get("max_dd", bt.get("max_drawdown", 0))),
                "win_rate_pct": bt.get("WinRate", bt.get("win_rate", 0)),
                "total_trades": bt.get("Trades", bt.get("trades", 0)),
            }
        elif isinstance(bt, list) and len(bt) > 0:
            result["backtest"] = {
                "has_data": True,
                "cagr_pct": bt[0].get("CAGR", 0),
                "sharpe": bt[0].get("Sharpe", 0),
                "max_dd_pct": bt[0].get("MaxDD", 0),
                "win_rate_pct": bt[0].get("WinRate", 0),
                "total_trades": bt[0].get("Trades", 0),
            }
    
    # 3. Compute comparison
    if result["live"]["has_data"] and result["backtest"]["has_data"]:
        diff = {}
        for key in ["total_return_pct", "win_rate_pct", "total_trades"]:
            if key in result["live"] and key in result["backtest"]:
                lv = result["live"].get(key, 0)
                bv = result["backtest"].get(key, 0)
                if isinstance(lv, (int, float)) and isinstance(bv, (int, float)):
                    diff[key] = round(lv - bv, 2)
        
        # Tracking error approximation
        result["tracking_error"] = abs(diff.get("total_return_pct", 0))
        result["difference"] = diff
    
    return result


# Also re-export get_trade_ledger for convenience
from analytics import get_trade_ledger, record_trade
