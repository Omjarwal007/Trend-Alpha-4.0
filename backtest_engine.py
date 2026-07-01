"""
Backtesting Harness — Trend Alfa 4.0
====================================
Framework skeleton for walk-forward backtesting of the multi-factor
trend-following system. Establishes patterns for historical validation.

STATUS: SKELETON — requires implementation of _run_single_window()
with actual pipeline integration + trade simulation.

USAGE (after implementation):
    python backtest_engine.py --start 2020-01-01 --end 2024-12-31 --capital 10000000
"""

import pandas as pd
import numpy as np
import datetime
import os
import json

from config import DEFAULT_PORTFOLIO_CAPITAL, BASE_DIR

# ── BACKTEST CONFIGURATION ──────────────────────────────────────────────
# Walk-forward parameters
TRAIN_WINDOW_YEARS = 3       # Training period for factor weight optimization
TEST_WINDOW_YEARS = 1         # Out-of-sample testing period
MIN_TRAINING_DAYS = 756       # Minimum trading days before first test window
PORTFOLIO_CAPITAL = DEFAULT_PORTFOLIO_CAPITAL  # ₹1 Cr default

# Transaction cost model (Indian markets)
STT_RATE = 0.001              # 0.1% Securities Transaction Tax (delivery)
BROKERAGE_RATE = 0.0001       # 0.01% per side
EXCHANGE_CHARGES = 0.000035   # 0.0035% NSE + SEBI + stamp
SLIPPAGE_BASIS_PTS = 0.0005   # 5 bps minimum slippage assumption
TOTAL_ROUND_TRIP_COST = 2 * (STT_RATE + BROKERAGE_RATE + EXCHANGE_CHARGES + SLIPPAGE_BASIS_PTS)
# ~0.13% per round-trip before impact costs

# LTCG tax (equity delivery > 1 year)
LTCG_TAX_RATE = 0.10          # 10% on gains > ₹1L per year
STCG_TAX_RATE = 0.15          # 15% on short-term gains


def compute_metrics(equity_curve, benchmark_curve=None, risk_free_rate=0.07):
    """Computes standard backtest performance metrics from an equity curve.
    
    Args:
        equity_curve: pd.Series of daily portfolio values (index = dates)
        benchmark_curve: optional pd.Series of benchmark values
        risk_free_rate: annual risk-free rate (7% for India)
    
    Returns:
        dict with CAGR, Sharpe, Sortino, MaxDD, Calmar, WinRate, etc.
    """
    if len(equity_curve) < 252:
        return {"error": "Insufficient data — need ≥1 year of daily returns"}
    
    daily_returns = equity_curve.pct_change().dropna()
    total_days = len(daily_returns)
    years = total_days / 252
    
    # CAGR
    start_val = equity_curve.iloc[0]
    end_val = equity_curve.iloc[-1]
    cagr = (end_val / start_val) ** (1 / years) - 1.0
    
    # Volatility
    ann_vol = daily_returns.std() * np.sqrt(252)
    
    # Sharpe Ratio
    excess_returns = daily_returns - risk_free_rate / 252
    sharpe = (excess_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() > 0 else 0
    
    # Sortino Ratio (downside deviation only)
    downside = daily_returns[daily_returns < 0]
    downside_std = downside.std() * np.sqrt(252) if len(downside) > 0 else ann_vol
    sortino = (daily_returns.mean() * 252 - risk_free_rate) / downside_std if downside_std > 0 else 0
    
    # Max Drawdown
    rolling_max = equity_curve.expanding().max()
    drawdowns = (equity_curve - rolling_max) / rolling_max
    max_dd = drawdowns.min()
    
    # Calmar Ratio
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0
    
    # Win Rate (from trade ledger, not daily)
    win_rate = 0.0
    profit_factor = 0.0
    total_trades = 0
    
    # Benchmark comparison
    alpha = 0.0
    beta = 1.0
    if benchmark_curve is not None and len(benchmark_curve) >= 252:
        bench_returns = benchmark_curve.pct_change().dropna()
        aligned = pd.concat([daily_returns, bench_returns], axis=1).dropna()
        if len(aligned) > 60:
            stock_ret = aligned.iloc[:, 0]
            bench_ret = aligned.iloc[:, 1]
            cov = np.cov(stock_ret, bench_ret)[0, 1]
            var = np.var(bench_ret)
            beta = cov / var if var > 0 else 1.0
            bench_cagr = (benchmark_curve.iloc[-1] / benchmark_curve.iloc[0]) ** (1 / years) - 1.0
            alpha = cagr - (risk_free_rate + beta * (bench_cagr - risk_free_rate))
    
    return {
        "cagr_pct": round(cagr * 100, 2),
        "ann_vol_pct": round(ann_vol * 100, 2),
        "sharpe_ratio": round(sharpe, 2),
        "sortino_ratio": round(sortino, 2),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "calmar_ratio": round(calmar, 2),
        "alpha_pct": round(alpha * 100, 2),
        "beta": round(beta, 2),
        "win_rate_pct": round(win_rate, 2),
        "profit_factor": round(profit_factor, 2),
        "total_trades": total_trades,
        "years_tested": round(years, 1),
        "start_date": str(equity_curve.index[0].date()),
        "end_date": str(equity_curve.index[-1].date()),
    }


def walk_forward_backtest(start_date, end_date, capital=PORTFOLIO_CAPITAL):
    """Runs walk-forward cross-validation of the Trend Alfa pipeline.
    
    Splits the date range into training (3yr) + testing (1yr) windows,
    rolling forward. For each window:
        1. Train: optimize factor weights on training data
        2. Test: run pipeline on out-of-sample test data
        3. Record: simulate trades, compute equity curve
    
    Args:
        start_date: 'YYYY-MM-DD' overall backtest start
        end_date: 'YYYY-MM-DD' overall backtest end
        capital: starting portfolio capital
    
    Returns:
        dict with full_results, summary_metrics, window_details
    """
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    
    train_window = pd.DateOffset(years=TRAIN_WINDOW_YEARS)
    test_window = pd.DateOffset(years=TEST_WINDOW_YEARS)
    
    # Generate walk-forward windows
    windows = []
    current_test_start = start + train_window  # First test period starts after training
    
    while current_test_start + test_window <= end:
        test_start = current_test_start
        test_end = min(test_start + test_window, end)
        train_start = test_start - train_window
        
        windows.append({
            "train_start": train_start.strftime("%Y-%m-%d"),
            "train_end": (test_start - pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            "test_start": test_start.strftime("%Y-%m-%d"),
            "test_end": test_end.strftime("%Y-%m-%d"),
        })
        
        current_test_start = test_end
    
    print(f"Walk-forward backtest: {len(windows)} windows from {start_date} to {end_date}")
    print(f"Training: {TRAIN_WINDOW_YEARS}yr | Testing: {TEST_WINDOW_YEARS}yr | Capital: ₹{capital:,.0f}")
    print(f"Transaction cost: {TOTAL_ROUND_TRIP_COST*100:.2f}% per round-trip\n")
    
    full_results = []
    all_metrics = []
    
    for i, window in enumerate(windows):
        print(f"Window {i+1}/{len(windows)}: Train [{window['train_start']} to {window['train_end']}] → Test [{window['test_start']} to {window['test_end']}]")
        
        # TODO: Implement _run_single_window()
        # result = _run_single_window(window, capital)
        # full_results.append(result)
        # all_metrics.append(result["metrics"])
        
        print(f"  [SKELETON] Window structure ready — implement _run_single_window() to execute")
    
    # Aggregate OOS metrics across all windows
    # TODO: Compute weighted average of OOS metrics
    # summary = aggregate_window_metrics(all_metrics)
    
    return {
        "windows": windows,
        "total_windows": len(windows),
        "train_years": TRAIN_WINDOW_YEARS,
        "test_years": TEST_WINDOW_YEARS,
        "start_date": start_date,
        "end_date": end_date,
        "capital": capital,
        "transaction_cost_pct": round(TOTAL_ROUND_TRIP_COST * 100, 3),
        "status": "SKELETON — implement _run_single_window()",
    }


def _run_single_window(window, capital):
    """IMPLEMENTATION REQUIRED: Run pipeline for a single walk-forward window.
    
    This function needs to:
    1. Load historical data for the window's training period
    2. Optimize factor weights on training data (if doing adaptive weights)
    3. Run daily pipeline on test data using walk-forward (no look-ahead)
    4. Simulate trades with transaction costs, slippage, and taxes
    5. Track equity curve, trades, and drawdowns
    6. Return metrics for this window
    
    Args:
        window: dict with train_start, train_end, test_start, test_end
        capital: starting capital for this window
    
    Returns:
        dict with equity_curve, trades, metrics, window
    """
    # PLACEHOLDER — implement me
    train_start = pd.Timestamp(window["train_start"])
    test_start = pd.Timestamp(window["test_start"])
    test_end = pd.Timestamp(window["test_end"])
    
    raise NotImplementedError(
        "Backtesting engine not yet implemented. "
        "Wire the daily pipeline (main.py) into this function, "
        "adding transaction cost simulation and walk-forward safeguards."
    )


def monte_carlo_simulation(equity_curve, n_simulations=1000, horizon_days=252):
    """Runs Monte Carlo simulation on historical daily returns.
    
    Resamples daily returns with replacement to generate N simulated
    equity paths. Reports confidence intervals for terminal value,
    max drawdown, and CAGR.
    
    Args:
        equity_curve: pd.Series of historical daily portfolio values
        n_simulations: number of simulation paths
        horizon_days: forward projection in trading days
    
    Returns:
        dict with simulation statistics
    """
    daily_returns = equity_curve.pct_change().dropna()
    if len(daily_returns) < 100:
        return {"error": "Need ≥100 days of returns for reliable simulation"}
    
    terminal_values = []
    max_drawdowns = []
    cagrs = []
    
    np.random.seed(42)
    
    for _ in range(n_simulations):
        # Bootstrap daily returns
        sampled = np.random.choice(daily_returns.values, size=horizon_days, replace=True)
        
        # Build equity curve
        equity = [equity_curve.iloc[-1]]
        for r in sampled:
            equity.append(equity[-1] * (1 + r))
        equity_series = pd.Series(equity)
        
        terminal_values.append(equity[-1])
        
        # Max drawdown
        rolling_max = equity_series.expanding().max()
        dd = (equity_series - rolling_max) / rolling_max
        max_drawdowns.append(dd.min())
        
        # CAGR
        cagr = (equity[-1] / equity[0]) ** (252 / horizon_days) - 1.0
        cagrs.append(cagr)
    
    terminal_values = np.array(terminal_values)
    max_drawdowns = np.array(max_drawdowns)
    cagrs = np.array(cagrs)
    
    return {
        "horizon_days": horizon_days,
        "n_simulations": n_simulations,
        "terminal_value": {
            "median": np.median(terminal_values),
            "p5": np.percentile(terminal_values, 5),
            "p25": np.percentile(terminal_values, 25),
            "p75": np.percentile(terminal_values, 75),
            "p95": np.percentile(terminal_values, 95),
        },
        "cagr_pct": {
            "median": round(np.median(cagrs) * 100, 2),
            "p5": round(np.percentile(cagrs, 5) * 100, 2),
            "p95": round(np.percentile(cagrs, 95) * 100, 2),
        },
        "max_drawdown_pct": {
            "median": round(np.median(max_drawdowns) * 100, 2),
            "p5": round(np.percentile(max_drawdowns, 5) * 100, 2),
            "worst_case": round(np.min(max_drawdowns) * 100, 2),
        },
    }


def sensitivity_analysis(base_config, param_ranges, metric_fn, n_steps=10):
    """Runs parameter sensitivity analysis across specified ranges.
    
    For each parameter in param_ranges, varies it across n_steps evenly
    spaced values while holding other params at base_config values.
    Computes metric_fn(config) at each point.
    
    Args:
        base_config: dict of parameter names → base values
        param_ranges: dict of parameter names → (min, max) tuples
        metric_fn: callable that takes config dict and returns metric float
        n_steps: number of steps per parameter sweep
    
    Returns:
        dict mapping param_name → list of (value, metric) pairs
    """
    results = {}
    
    for param, (min_val, max_val) in param_ranges.items():
        sweep = []
        for step in range(n_steps):
            value = min_val + (max_val - min_val) * step / (n_steps - 1)
            config = base_config.copy()
            config[param] = value
            try:
                metric = metric_fn(config)
                sweep.append((value, metric))
            except Exception as e:
                sweep.append((value, None))
        results[param] = sweep
    
    return results


# ── CLI ENTRY POINT ────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Trend Alfa 4.0 Backtesting Engine")
    parser.add_argument("--start", default="2020-01-01", help="Backtest start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2024-12-31", help="Backtest end date (YYYY-MM-DD)")
    parser.add_argument("--capital", type=float, default=10000000, help="Starting capital in INR")
    parser.add_argument("--monte-carlo", action="store_true", help="Run Monte Carlo simulation")
    parser.add_argument("--sensitivity", action="store_true", help="Run sensitivity analysis")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("  TREND ALFA 4.0 — BACKTESTING HARNESS (SKELETON)")
    print("=" * 60)
    print()
    print("STATUS: Framework structure is ready.")
    print("TODO: Implement _run_single_window() to execute actual backtests.")
    print()
    
    result = walk_forward_backtest(args.start, args.end, args.capital)
    
    print(f"\nWalk-forward windows generated: {result['total_windows']}")
    print(f"Transaction cost model: {result['transaction_cost_pct']:.2f}% per RT")
    print(f"\nNext steps:")
    print("  1. Implement _run_single_window() in backtest_engine.py")
    print("  2. Wire pipeline stages with look-ahead safeguards")
    print("  3. Add trade simulation with cost model")
    print("  4. Run walk-forward on 5+ years of historical data")
    print("  5. Validate factor weights with OOS Sharpe > 0.5 target")
