"""
Post-Exit P&L Review — Weekly/Monthly Performance Report
=========================================================
Reads trade_ledger.csv and generates a structured P&L report:
- Per-stock win/loss with P&L%
- Win rate, profit factor, avg hold time
- Best/worst performers
- Monthly breakdown
"""
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

LEDGER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trade_ledger.csv")
REPORT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

def generate_pnl_report(date_str=None, days_back=90):
    """Generate P&L report from trade ledger.
    
    Args:
        date_str: Report date (default: today)
        days_back: How many days of history to include
    
    Returns: dict with report data
    """
    if date_str is None:
        date_str = datetime.today().strftime("%Y-%m-%d")
    
    if not os.path.exists(LEDGER_PATH):
        return {"error": "No trade ledger found — run pipeline first"}
    
    ledger = pd.read_csv(LEDGER_PATH, parse_dates=["Date"])
    cutoff = pd.Timestamp(date_str) - timedelta(days=days_back)
    ledger = ledger[ledger["Date"] >= cutoff]
    
    if ledger.empty:
        return {"error": f"No trades in last {days_back} days"}
    
    report = {
        "report_date": date_str,
        "period_days": days_back,
        "total_trades": 0,
        "win_count": 0,
        "loss_count": 0,
        "win_rate_pct": 0,
        "profit_factor": 0,
        "avg_hold_days": 0,
        "total_pnl_pct": 0,
        "best_trade": None,
        "worst_trade": None,
        "monthly_breakdown": [],
        "stock_details": [],
    }
    
    # Analyze each closed position
    stock_pnls = []
    monthly_pnl = {}
    
    for sym in ledger["Symbol"].unique():
        sym_data = ledger[ledger["Symbol"] == sym].sort_values("Date")
        buys = sym_data[sym_data["Action"].isin(["BUY", "ADD"])]
        exits = sym_data[sym_data["Action"] == "EXIT"]
        
        if len(buys) == 0 or len(exits) == 0:
            continue
        
        total_buy_val = buys["Value"].sum()
        total_exit_val = exits["Value"].sum()
        total_buy_qty = buys["Quantity"].sum()
        
        if total_buy_val == 0:
            continue
        
        avg_buy_price = total_buy_val / total_buy_qty if total_buy_qty > 0 else 0
        avg_exit_price = total_exit_val / exits["Quantity"].sum() if exits["Quantity"].sum() > 0 else 0
        pnl_val = total_exit_val - total_buy_val
        pnl_pct = (pnl_val / total_buy_val) * 100.0
        
        first_buy = buys["Date"].min()
        last_exit = exits["Date"].max()
        hold_days = (last_exit - first_buy).days
        
        exit_reason = exits.iloc[0].get("Reason", "") if not exits.empty else ""
        
        stock_data = {
            "symbol": sym,
            "pnl_pct": round(pnl_pct, 2),
            "pnl_val": round(pnl_val, 2),
            "hold_days": hold_days,
            "avg_buy": round(avg_buy_price, 2),
            "avg_exit": round(avg_exit_price, 2),
            "is_win": pnl_val > 0,
            "exit_reason": str(exit_reason)[:100],
        }
        stock_pnls.append(stock_data)
        
        # Monthly aggregation
        exit_month = last_exit.strftime("%Y-%m")
        if exit_month not in monthly_pnl:
            monthly_pnl[exit_month] = {"trades": 0, "wins": 0, "pnl": 0}
        monthly_pnl[exit_month]["trades"] += 1
        monthly_pnl[exit_month]["wins"] += 1 if pnl_val > 0 else 0
        monthly_pnl[exit_month]["pnl"] += pnl_val
    
    if not stock_pnls:
        return {"error": "No closed positions found"}
    
    # Summary
    total_trades = len(stock_pnls)
    wins = [s for s in stock_pnls if s["is_win"]]
    losses = [s for s in stock_pnls if not s["is_win"]]
    
    report["total_trades"] = total_trades
    report["win_count"] = len(wins)
    report["loss_count"] = len(losses)
    report["win_rate_pct"] = round(len(wins) / total_trades * 100, 1) if total_trades > 0 else 0
    report["avg_hold_days"] = round(np.mean([s["hold_days"] for s in stock_pnls]), 1)
    report["total_pnl_pct"] = round(np.mean([s["pnl_pct"] for s in stock_pnls]), 2)
    
    total_profit = sum(s["pnl_val"] for s in wins)
    total_loss = abs(sum(s["pnl_val"] for s in losses))
    report["profit_factor"] = round(total_profit / total_loss, 2) if total_loss > 0 else (99.0 if total_profit > 0 else 0)
    
    # Best/worst
    report["best_trade"] = max(stock_pnls, key=lambda x: x["pnl_pct"])
    report["worst_trade"] = min(stock_pnls, key=lambda x: x["pnl_pct"])
    
    # Stock details sorted by P&L
    report["stock_details"] = sorted(stock_pnls, key=lambda x: x["pnl_pct"], reverse=True)
    
    # Monthly breakdown
    report["monthly_breakdown"] = [{"month": m, **d} for m, d in sorted(monthly_pnl.items())]
    
    return report


def print_pnl_report(report):
    """Print formatted P&L report."""
    if "error" in report:
        print(f"\n❌ {report['error']}\n")
        return
    
    print(f"\n{'='*70}")
    print(f"  📈 P&L REPORT — Trend Alfa v3")
    print(f"  Period: Last {report['period_days']} days | Report date: {report['report_date']}")
    print(f"{'='*70}")
    
    # Summary cards
    wr = report["win_rate_pct"]
    wr_color = "🟢" if wr >= 50 else "🔴"
    pf = report["profit_factor"]
    
    print(f"\n  {wr_color}  Win Rate:   {wr}% ({report['win_count']}W / {report['loss_count']}L)")
    print(f"  📊  Profit Factor: {pf:.2f}" + (" (good)" if pf >= 1.5 else " (needs improvement)" if pf >= 1.0 else " (poor)"))
    print(f"  ⏱️   Avg Hold:     {report['avg_hold_days']} days")
    print(f"  💰  Avg P&L:       {report['total_pnl_pct']:+.2f}%")
    print(f"  📋  Total Trades:  {report['total_trades']}")
    
    # Best / Worst
    if report["best_trade"]:
        b = report["best_trade"]
        print(f"\n  🏆  BEST TRADE:  {b['symbol']}  {b['pnl_pct']:+.2f}%  (held {b['hold_days']}d)")
    if report["worst_trade"]:
        w = report["worst_trade"]
        print(f"  💀  WORST TRADE: {w['symbol']}  {w['pnl_pct']:+.2f}%  (held {w['hold_days']}d)")
    
    # Stock details table
    print(f"\n  {'Symbol':<16} {'P&L%':<10} {'P&L₹':<12} {'Days':<6} {'Result':<8}")
    print(f"  {'─'*55}")
    for s in report["stock_details"]:
        icon = "✅" if s["is_win"] else "❌"
        pnl_s = f"{s['pnl_pct']:+.2f}%"
        print(f"  {icon} {s['symbol']:<14} {pnl_s:<10} ₹{s['pnl_val']:<10.0f} {s['hold_days']:<6} ")
    
    # Monthly breakdown
    if report["monthly_breakdown"]:
        print(f"\n  📅  MONTHLY BREAKDOWN")
        print(f"  {'─'*45}")
        for m in report["monthly_breakdown"]:
            icon = "✅" if m["pnl"] >= 0 else "❌"
            print(f"  {icon} {m['month']:<10} {m['trades']:>2} trades  {m['wins']:>2} wins  ₹{m['pnl']:>+8.0f}")
    
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    import sys
    days = 90
    if len(sys.argv) > 1:
        days = int(sys.argv[1])
    report = generate_pnl_report(days_back=days)
    print_pnl_report(report)
