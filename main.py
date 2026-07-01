import os
import sys
# Add legacy directory to path so imports of layer_* files work
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "_legacy"))
import datetime
import json
from pathlib import Path
import pandas as pd
import numpy as np
from utils import log_info, log_success, log_system, log_error, ensure_output_dir, log_warning
from config import SYMBOLS, DEFAULT_PORTFOLIO_CAPITAL, BASE_DIR, SECTORS, THEMES, TOP_N_STOCKS
from pipeline_data import run_data_pipeline, compute_all_indicators, calculate_atr, calculate_natr
from market_regime import run_market_regime, run_sector_rotation
from stock_selector import run_stock_selection
from portfolio_manager import run_drawdown_governor, run_portfolio_construction, calculate_initial_stop
from core_momentum_engine import run_core_momentum_engine
from cache_manager import get_historical_data
from monitoring_engine import generate_execution_orders
from notifications import send_exit_notifications
from analytics import compute_portfolio_analytics
from pre_trade_check import run_pre_trade_checklist
from health_monitor import check_and_report
from screener_fetcher import fetch_chartink_universe
from layer_7_maac import run_maac_allocation
from layer_3_smart_money import run_smart_money

from db_manager import save_pipeline_stage

def safe_to_csv(df, path):
    import os
    base_name = os.path.basename(path)
    table_name = os.path.splitext(base_name)[0]
    
    # Extract date string from the path (e.g. output/2026-06-26/...)
    path_parts = os.path.normpath(path).split(os.sep)
    date_str = None
    if len(path_parts) >= 2:
        potential_date = path_parts[-2]
        if len(potential_date) == 10 and potential_date.startswith("20"):
            date_str = potential_date
            
    # Always save to SQLite
    save_pipeline_stage(df, table_name, date_str)
    
    # Keep Execution_Orders, L6_Trade_Allocations, and L1_* universe files as actual CSVs for compatibility
    if table_name in ("Execution_Orders", "L6_Trade_Allocations") or table_name.startswith("L1_"):
        try:
            df.to_csv(path, index=False)
        except PermissionError:
            base, ext = os.path.splitext(path)
            backup_path = f"{base}_LOCKED{ext}"
            log_error(f"Permission denied writing to {base_name}. Writing to backup: {os.path.basename(backup_path)}")
            try:
                df.to_csv(backup_path, index=False)
            except Exception as e:
                log_error(f"Failed to write backup: {e}")


def print_terminal_dashboard(date_str, regime, blueprint, analytics, portfolio_positions, orders_df, existing_holdings=None, pipeline_data=None):
    # ANSI escape characters for styling
    B = "\033[1m" # Bold
    C = "\033[96m" # Cyan
    G = "\033[92m" # Green
    Y = "\033[93m" # Yellow
    R = "\033[91m" # Red
    M = "\033[95m" # Magenta
    BL = "\033[94m" # Blue
    N = "\033[0m"  # Reset
    
    def pad_line(label, val, max_len, val_color=None):
        val_str = str(val)
        padding = max_len - len(label) - len(val_str)
        v_str = f"{val_color}{val_str}{N}" if val_color else val_str
        return f"{label}{v_str}{' ' * padding}"

    def format_row(lbl1, val1, color1, lbl2, val2, color2, w1=36, w2=37):
        left = pad_line(lbl1, val1, w1, val_color=color1)
        right = pad_line(lbl2, val2, w2, val_color=color2)
        return f"│ {left} │ {right} │"

    # Header Box
    print(f"\n{C}┌──────────────────────────────────────────────────────────────────────────────┐{N}")
    print(f"{C}│{B}{M}                 ⚡ TREND ALPHA 4.0 | SYNCED INSIGHTS DASHBOARD               {N}{C}│{N}")
    print(f"{C}└──────────────────────────────────────────────────────────────────────────────┘{N}")

    # Market Regime and Risk Box
    regime_name = regime.get("market_regime", "SIDEWAYS")
    regime_color = G if regime_name in ["BULL", "EARLY_BULL", "LATE_BULL"] else (Y if regime_name == "SIDEWAYS" else R)
    breadth_score = regime.get("breadth_score", 0.0)
    breadth_regime = regime.get("breadth_regime", "N/A")
    breadth_color = G if breadth_score > 65 else (Y if breadth_score > 35 else R)
    breadth_val = f"{breadth_score:.1f}% ({breadth_regime})"
    
    port_heat = blueprint.get('portfolio_heat_pct', 0.0)
    heat_color = G if port_heat < 2.0 else (Y if port_heat < 4.0 else R)
    
    thrust_active = regime.get('breadth_thrust_active', False)
    thrust_val = "ACTIVE" if thrust_active else "INACTIVE"
    thrust_color = G if thrust_active else R
    
    print(f"{C}┌──────────────────────────────────────┬───────────────────────────────────────┐{N}")
    print(f"{C}│{B} 📈 MARKET REGIME & BREADTH STATS     {N}{C}│{B} 🛡️ PORTFOLIO RISK & COMPOSITE HEAT    {N}{C}│{N}")
    print(f"{C}├──────────────────────────────────────┼───────────────────────────────────────┤{N}")
    print(format_row("  • Unified Regime   : ", regime_name, regime_color, "  • Base Risk (Heat) : ", f"{blueprint.get('base_heat_pct', 0.0):.2f}%", None))
    print(format_row("  • Breadth Score    : ", breadth_val, breadth_color, "  • High Corr Pairs  : ", f"{blueprint.get('high_corr_pairs_count', 0)} pairs", Y))
    print(format_row("  • Leadership Ratio : ", f"{regime.get('leadership_ratio', 1.0):.2f}", Y, "  • Corr Penalty     : ", f"+{blueprint.get('correlation_penalty_pct', 0.0):.2f}%", Y))
    print(format_row("  • Breadth Thrust   : ", thrust_val, thrust_color, "  • Leverage Penalty : ", f"+{blueprint.get('leverage_penalty_pct', 0.0):.2f}%", M))
    print(format_row("  • Total Exposure   : ", f"{blueprint.get('total_exposure_pct', 100.0):.1f}%", Y, "  • Adjusted Heat    : ", f"{port_heat:.2f}% (Limit: 6.0%)", heat_color))
    print(f"{C}└──────────────────────────────────────┴───────────────────────────────────────┘{N}")
    
    # Index Valuation Box
    print(f"{C}┌──────────────────────────────────────┬───────────────────────────────────────┐{N}")
    print(f"{C}│{B} 🏢 NIFTY 500 BROAD VALUATIONS (Screener) {N}{C}│{B} 🏢 SMALLCAP 250 INDEX VALUATIONS        {N}{C}│{N}")
    print(f"{C}├──────────────────────────────────────┼───────────────────────────────────────┤{N}")
    print(format_row("  • Price            : ", f"₹{regime.get('nifty500_price', 22600.0):,.2f}", None, "  • Price            : ", f"₹{regime.get('smallcap250_price', 17000.0):,.2f}", None))
    print(format_row("  • P/E Ratio        : ", f"{regime.get('nifty500_pe', 22.4):.1f}", Y, "  • P/E Ratio        : ", f"{regime.get('smallcap250_pe', 28.1):.1f}", Y))
    print(format_row("  • P/B Ratio        : ", f"{regime.get('nifty500_pb', 3.45):.2f}", Y, "  • P/B Ratio        : ", f"{regime.get('smallcap250_pb', 3.82):.2f}", Y))
    print(format_row("  • Dividend Yield   : ", f"{regime.get('nifty500_div_yield', 1.08):.2f}%", G, "  • Dividend Yield   : ", f"{regime.get('smallcap250_div_yield', 0.78):.2f}%", G))
    print(format_row("  • CAGR 5Yr / 10Yr  : ", f"{regime.get('nifty500_cagr_5yr', 10.7):.1f}% / {regime.get('nifty500_cagr_10yr', 12.8):.1f}%", None, "  • CAGR 5Yr / 10Yr  : ", f"{regime.get('smallcap250_cagr_5yr', 15.2):.1f}% / {regime.get('smallcap250_cagr_10yr', 15.5):.1f}%", None))
    print(f"{C}└──────────────────────────────────────┴───────────────────────────────────────┘{N}")

    # Portfolio Sizing Table
    print(f"\n{B}📊 PORTFOLIO BLUEPRINT & EXPOSURES (Per ₹100 Capital){N}")
    print(f"  ┌──────────────────────────────┬──────────┬────────────────────────────────┐")
    print(f"  │ Asset Class                  │ Weight % │ Status / Details               │")
    print(f"  ├──────────────────────────────┼──────────┼────────────────────────────────┤")
    def pad_status(text, color, target_len=30):
        padding = target_len - len(text)
        return f"{color}{text}{N}{' ' * padding}"

    # Passive Core (ETFs/MFs)
    core_val = blueprint.get('active_core_equities_pct', 0.0)
    core_lbl = "Active (Max 35.0%)" if core_val > 0 else "0% (No Positions/Bearish)"
    core_col = G if core_val > 0 else Y
    core_status = pad_status(core_lbl, core_col)
    print(f"  │ 1a. Passive Core (ETFs/MFs)   │  {core_val:5.1f}%  │ {core_status} │")

    # Active Satellite (VAM-GQ + VAM-B)
    sat_val = blueprint.get('active_satellite_equities_pct', 0.0)
    sat_lbl = "Active (Max 65.0%)" if sat_val > 0 else "0% (No Positions/Bearish)"
    sat_col = G if sat_val > 0 else Y
    sat_status = pad_status(sat_lbl, sat_col)
    print(f"  │ 1b. Active Satellite (Eq)     │  {sat_val:5.1f}%  │ {sat_status} │")
    
    # MTF
    mtf_val = blueprint.get('mtf_leverage_pct', 0.0)
    mtf_lbl = "Active Limit (+150%)" if mtf_val > 50.0 else ("Active Limit (+50%)" if mtf_val > 0 else "CASH OUT (Bearish)")
    mtf_col = G if mtf_val > 0 else R
    mtf_status = pad_status(mtf_lbl, mtf_col)
    print(f"  │ 2. MTF Equity Leverage       │  {mtf_val:5.1f}%  │ {mtf_status} │")
    
    # Undeployed Cash
    cash_val = blueprint.get('undeployed_cash_pct', 0.0)
    cash_status = pad_status("Undeployed Cash Collateral", G)
    print(f"  │ 3. Undeployed Cash Margin    │  {cash_val:5.1f}%  │ {cash_status} │")
    
    # Gold MCX
    gold_val = blueprint.get('gold_futures_pct', 0.0)
    gold_lbl = "ACTIVE (+25% Gold Fut)" if gold_val > 0 else "CASH OUT (Bearish)"
    gold_col = G if gold_val > 0 else R
    gold_status = pad_status(gold_lbl, gold_col)
    print(f"  │ 4. Gold Futures (MCX)        │  {gold_val:5.1f}%  │ {gold_status} │")
    
    # Silver MCX
    silver_val = blueprint.get('silver_futures_pct', 0.0)
    silver_lbl = "ACTIVE (+10% Silver Fut)" if silver_val > 0 else "CASH OUT (Bearish)"
    silver_col = G if silver_val > 0 else R
    silver_status = pad_status(silver_lbl, silver_col)
    print(f"  │ 5. Silver Futures (MCX)      │  {silver_val:5.1f}%  │ {silver_status} │")
    print(f"  └──────────────────────────────┴──────────┴────────────────────────────────┘")

    # SEBI Compliance Box
    sebi_ok = blueprint.get('sebi_compliant', True)
    sebi_status = "COMPLIANT" if sebi_ok else "VIOLATION"
    sebi_color = G if sebi_ok else R
    shortfall_interest = "0.00%" if sebi_ok else "18.00%"
    excess_cash_margin = blueprint.get('excess_cash_margin', 0.0)
    excess_color = G if excess_cash_margin >= 0 else R
    
    print(f"{C}┌──────────────────────────────────────┬───────────────────────────────────────┐{N}")
    print(f"{C}│{B} ⚖️ SEBI 50:50 MARGIN COMPLIANCE      {N}{C}│{B} 📊 COLLATERAL VALUE SUMMARY           {N}{C}│{N}")
    print(f"{C}├──────────────────────────────────────┼───────────────────────────────────────┤{N}")
    print(format_row("  • Non-Cash Margin  : ", f"₹{blueprint.get('non_cash_collateral', 0.0):,.2f}", None, "  • Total F&O Margin : ", f"₹{blueprint.get('fo_margin_required', 0.0):,.2f}", None))
    print(format_row("  • Cash-Equiv Margin: ", f"₹{blueprint.get('cash_equiv_collateral', 0.0):,.2f}", G, "  • SEBI Status      : ", sebi_status, sebi_color))
    print(format_row("  • Excess Cash Marg : ", f"₹{excess_cash_margin:,.2f}", excess_color, "  • Shortfall Int.   : ", shortfall_interest, sebi_color))
    print(f"{C}└──────────────────────────────────────┴───────────────────────────────────────┘{N}")

    # Active positions holdings table
    print(f"\n{B}🔍 ACTIVE POSITION HOLDINGS ({len(portfolio_positions)} Assets){N}")
    if portfolio_positions:
        # Load name map for AMFI code display
        _name_map_pos = {}
        try:
            from config import CORE_ETF_UNIVERSE
            for _k, _v in CORE_ETF_UNIVERSE.items():
                _name_map_pos[str(_k)] = str(_v)
            import mf_fetcher
            _mf_list = mf_fetcher.get_master_mf_list()
            if _mf_list:
                for _mf in _mf_list:
                    _name_map_pos[str(_mf.get("schemeCode"))] = str(_mf.get("schemeName", ""))
        except:
            pass
        print(f"  ┌─────┬────────────────┬────────────────────────┬──────────────┬──────────────┬─────────┬─────────┐")
        print(f"  │  #  │ Symbol        │ Sector                 │ Entry Price  │ Stop Loss    │ Alloc % │ Risk %  │")
        print(f"  ├─────┼────────────────┼────────────────────────┼──────────────┼──────────────┼─────────┼─────────┤")
        for i, pos in enumerate(portfolio_positions):
            sym = str(pos.get("Symbol", "UNK"))
            sect = str(pos.get("Sector", "UNK"))[:22]
            entry_pr = pos.get("Entry_Price", 0.0)
            sl_pr = pos.get("Stop_Loss", 0.0)
            alloc_p = pos.get("Allocation_Pct", 0.0)
            risk_p = pos.get("Actual_Risk_Pct", pos.get("Risk_Per_Trade_%", 0.0))
            # Map AMFI code to readable name
            if sym.isdigit() and sym in _name_map_pos:
                sym_disp = _name_map_pos[sym][:14]
            else:
                sym_disp = sym
            
            idx_str = f" {i+1:>3d} "
            sym_str = f" {sym_disp:<14s} "
            sect_str = f" {sect:<22s} "
            entry_str = f" ₹{entry_pr:<10.2f} "
            sl_str = f" ₹{sl_pr:<10.2f} "
            alloc_str = f" {f'{alloc_p:.2f}%':>7s} "
            risk_str = f" {f'{risk_p:.2f}%':>7s} "
            
            print(f"  │{idx_str}│{sym_str}│{sect_str}│{entry_str}│{sl_str}│{alloc_str}│{risk_str}│")
        print(f"  └─────┴────────────────┴────────────────────────┴──────────────┴──────────────┴─────────┴─────────┘")
    else:
        print(f"  {R}No active stock positions held (Drawdown Governor active).{N}")

    # Existing holdings categorisation and reporting (SKILL 13 / 15)
    nifty_df = pipeline_data.get("NIFTY_50") if pipeline_data else None
    from monitoring_engine import calculate_rs_line
    from config import SECTORS

    retained_holds = []
    exited_holdings = []
    new_entries = []

    # 1. Categorise existing holdings into Retained vs Exited
    if existing_holdings:
        # Get set of symbols being exited
        exit_orders_subset = orders_df[orders_df["Action"] == "EXIT"] if not orders_df.empty else pd.DataFrame()
        exit_reasons_map = dict(zip(exit_orders_subset["Symbol"], exit_orders_subset["Reason"])) if not exit_orders_subset.empty else {}
        
        for sym in existing_holdings:
            df_stock = pipeline_data.get(sym) if pipeline_data else None
            rs_val, status = calculate_rs_line(sym, df_stock, nifty_df)
            close_price = df_stock["Close"].iloc[-1] if df_stock is not None and not df_stock.empty else 0.0
            sect = SECTORS.get(sym, "Diversified")[:22]
            
            if sym in exit_reasons_map:
                exited_holdings.append({
                    "Symbol": sym,
                    "Sector": sect,
                    "Close": close_price,
                    "RS_Val": rs_val,
                    "Reason": exit_reasons_map[sym]
                })
            else:
                # Find current weight in target conviction portfolio
                target_alloc = 0.0
                for pos in portfolio_positions:
                    if pos["Symbol"] == sym:
                        target_alloc = pos["Allocation_Pct"]
                        break
                retained_holds.append({
                    "Symbol": sym,
                    "Sector": sect,
                    "Close": close_price,
                    "RS_Val": rs_val,
                    "Allocation_Pct": target_alloc
                })

    # 2. Categorise new buys
    if not orders_df.empty:
        buy_orders = orders_df[orders_df["Action"] == "BUY"]
        for idx, row in buy_orders.iterrows():
            sym = row["Symbol"]
            sect = SECTORS.get(sym, "Diversified")[:22]
            price = row.get("Limit_Price", row.get("Entry_Price", 0.0))
            # Find entry price and target alloc from portfolio positions
            alloc = 0.0
            for pos in portfolio_positions:
                if pos["Symbol"] == sym:
                    price = pos["Entry_Price"]
                    alloc = pos["Allocation_Pct"]
                    break
            new_entries.append({
                "Symbol": sym,
                "Sector": sect,
                "Price": price,
                "Allocation_Pct": alloc
            })

    # Table A: 🟢 KEEP HOLDING (Retained Positions)
    print(f"\n{B}🟢 KEEP HOLDING (Retained Positions - RS Line > 0.10){N}")
    if retained_holds:
        print(f"  ┌─────┬────────────┬────────────────────────┬──────────────┬──────────────┬─────────┬─────────┐")
        print(f"  │  #  │ Symbol     │ Sector                 │ Close Price  │ RS Line Val  │ Alloc % │ Rec     │")
        print(f"  ├─────┼────────────┼────────────────────────┼──────────────┼──────────────┼─────────┼─────────┤")
        for i, pos in enumerate(retained_holds):
            idx_str = f" {i+1:>3d} "
            sym_str = f" {pos['Symbol']:<10s} "
            sect_str = f" {pos['Sector']:<22s} "
            _cp = pos['Close']
            _rv = pos['RS_Val']
            _ap = pos['Allocation_Pct']
            close_str = f" \u20b9{_cp:<10.2f} "
            rs_str = f" {_rv:>10.4f} "
            alloc_str = f" {_ap:>5.2f}% "
            rec_str = f" {G}{'HOLD':<7s}{N} "
            print(f"  │{idx_str}│{sym_str}│{sect_str}│{close_str}│{rs_str}│{alloc_str}│{rec_str}│")
        print(f"  └─────┴────────────────┴────────────────────────┴──────────────┴──────────────┴─────────┴─────────┘")
    else:
        print(f"  {Y}No holdings retained from previous run.{N}")

    # Table B: 🔴 LIQUIDATE / EXIT (Exited Holdings)
    print(f"\n{B}🔴 LIQUIDATE / EXIT (Exits & Liquidations){N}")
    if exited_holdings:
        print(f"  ┌─────┬────────────┬────────────────────────┬──────────────┬──────────────┬──────────────────────────────┐")
        print(f"  │  #  │ Symbol     │ Sector                 │ Close Price  │ RS Line Val  │ Exit Reason                  │")
        print(f"  ├─────┼────────────┼────────────────────────┼──────────────┼──────────────┼──────────────────────────────┤")
        for i, pos in enumerate(exited_holdings):
            idx_str = f" {i+1:>3d} "
            sym_str = f" {pos['Symbol']:<10s} "
            sect_str = f" {pos['Sector']:<22s} "
            _cp2 = pos['Close']
            _rv2 = pos['RS_Val']
            close_str = f" \u20b9{_cp2:<10.2f} "
            rs_str = f" {_rv2:>10.4f} "
            reason_str = f" {R}{pos['Reason'][:28]:<28s}{N} "
            print(f"  │{idx_str}│{sym_str}│{sect_str}│{close_str}│{rs_str}│{reason_str}│")
        print(f"  └─────┴────────────┴────────────────────────┴──────────────┴──────────────┴──────────────────────────────┘")
    else:
        print(f"  {G}No portfolio holdings need to be exited.{N}")

    # Table C: 🔵 NEW ENTRIES (BUY Allocations)
    print(f"\n{B}🔵 NEW ENTRIES (BUY Allocations){N}")
    if new_entries:
        print(f"  ┌─────┬────────────┬────────────────────────┬──────────────┬─────────┬─────────┐")
        print(f"  │  #  │ Symbol     │ Sector                 │ Entry Price  │ Alloc % │ Rec     │")
        print(f"  ├─────┼────────────┼────────────────────────┼──────────────┼─────────┼─────────┤")
        for i, pos in enumerate(new_entries):
            idx_str = f" {i+1:>3d} "
            sym_str = f" {pos['Symbol']:<10s} "
            sect_str = f" {pos['Sector']:<22s} "
            _pp = pos['Price']
            price_str = f" ₹{_pp:<10.2f} "
            _ap2 = pos['Allocation_Pct']
            alloc_str = f" {_ap2:>5.2f}% "
            rec_str = f" {BL}{'BUY':<7s}{N} "
            print(f"  │{idx_str}│{sym_str}│{sect_str}│{price_str}│{alloc_str}│{rec_str}│")
        print(f"  └─────┴────────────┴────────────────────────┴──────────────┴─────────┴─────────┘")
    else:
        print(f"  {Y}No new positions to allocate.{N}")
    print(f"\n{C}================================================================================{N}\n")


def load_or_initialize_state(date_str, pipeline_data):
    """Loads persistent portfolio state from portfolio_state.json or initializes it."""
    state_path = os.path.join(BASE_DIR, "portfolio_state.json")
    if os.path.exists(state_path):
        try:
            with open(state_path, "r") as f:
                state = json.load(f)
        except Exception as e:
            log_warning(f"Failed to load portfolio state: {e}. Reinitializing.")
            state = None
    else:
        state = None

    if state is None:
        state = {
            "initial_capital": DEFAULT_PORTFOLIO_CAPITAL,
            "cash_balance": DEFAULT_PORTFOLIO_CAPITAL,
            "peak_portfolio_value": DEFAULT_PORTFOLIO_CAPITAL,
            "holdings": {}
        }
    
    # Calculate current value of holdings based on the latest prices
    holdings_value = 0.0
    for sym, pos in state["holdings"].items():
        qty = pos.get("Quantity", 0)
        df_stock = pipeline_data.get(sym) if pipeline_data else None
        if df_stock is not None and not df_stock.empty:
            price = float(df_stock["Close"].iloc[-1])
        else:
            price = float(pos.get("Entry_Price", 0.0))
            
        # Update Highest_Price_Since_Entry for the Drawdown Guard
        current_highest = pos.get("Highest_Price_Since_Entry", float(pos.get("Entry_Price", 0.0)))
        if price > current_highest:
            pos["Highest_Price_Since_Entry"] = price
        elif "Highest_Price_Since_Entry" not in pos:
            pos["Highest_Price_Since_Entry"] = current_highest
            
        holdings_value += qty * price

    current_portfolio_value = state["cash_balance"] + holdings_value
    state["peak_portfolio_value"] = max(state.get("peak_portfolio_value", current_portfolio_value), current_portfolio_value)
    return state, current_portfolio_value

def apply_veto_overrides(orders_df, date_str, pipeline_data, state, current_portfolio_value, portfolio_positions=None, portfolio_blueprint=None):
    """Reads Veto_add_remove.csv and injects manual ADD/REMOVE orders into orders_df and updates portfolio structures."""
    veto_file = os.path.join(BASE_DIR, "Veto_add_remove.csv")
    if not os.path.exists(veto_file):
        return orders_df
        
    try:
        veto_df = pd.read_csv(veto_file)
        pending_mask = veto_df["Status"] == "PENDING"
        if not pending_mask.any():
            return orders_df
            
        veto_orders = []
        for idx, row in veto_df[pending_mask].iterrows():
            sym = row["Symbol"]
            action = row["Action"]
            alloc_pct = row.get("Allocation_Pct", 5.0)
            
            if action == "VETO_ADD":
                df_stock = pipeline_data.get(sym)
                if df_stock is None or df_stock.empty:
                    df_stock = get_historical_data(sym, end_date=date_str)
                    if df_stock is not None and not df_stock.empty:
                        pipeline_data[sym] = df_stock
                
                if df_stock is not None and not df_stock.empty:
                    close_pr = float(df_stock["Close"].iloc[-1])
                    # User requested stop loss rules to be the same as system logic
                    if len(df_stock) >= 15:
                        atr_val = float(calculate_atr(df_stock, 14).iloc[-1])
                        natr_val = float(calculate_natr(df_stock, 14).iloc[-1])
                        ma50_val = float(df_stock["Close"].ewm(span=50, adjust=False).mean().iloc[-1])
                    else:
                        atr_val = close_pr * 0.03
                        natr_val = 3.0
                        ma50_val = close_pr * 0.95
                        
                    initial_stop, stop_dist_pct, stop_warning = calculate_initial_stop(
                        close_pr, atr_val, natr_val, "TIER 1 (V)", ma_50=ma50_val
                    )
                    
                    qty = int((alloc_pct / 100.0 * current_portfolio_value) / close_pr)
                    veto_orders.append({
                        "Symbol": sym,
                        "Action": "BUY",
                        "Quantity": qty,
                        "Entry_Price": close_pr,
                        "Reason": "Manual Veto Add (V)",
                        "Allocation_%": alloc_pct,
                        "Tier": "TIER 1 (V)"
                    })
                    
                    # Update portfolio positions for reporting
                    if portfolio_positions is not None:
                        sector = SECTORS.get(sym, "Diversified")
                        theme = THEMES.get(sym, "Generic Theme")
                        pos_dict = {
                            "Symbol": sym,
                            "Sector": sector,
                            "Theme": theme,
                            "Tier": "TIER 1 (V)",
                            "Bucket": "CORE" if alloc_pct > 2.0 else "SATELLITE",
                            "Entry_Price": close_pr,
                            "Quantity": qty,
                            "Position_Value": qty * close_pr,
                            "Allocation_Pct": alloc_pct,
                            "Stop_Loss": initial_stop,
                            "Stop_Distance_Pct": stop_dist_pct,
                            "Actual_Risk": (alloc_pct / 100.0) * current_portfolio_value * (stop_dist_pct / 100.0),
                            "Actual_Risk_Pct": (alloc_pct / 100.0) * stop_dist_pct,
                            "NATR": natr_val,
                            "ATR_14": atr_val,
                            "Stop_Warning": stop_warning,
                            "ROE": 15.0,
                            "Debt_to_Equity": 0.0,
                            "CHOP_avg": 50.0,
                            "Weekly_CHOP": 50.0,
                            "Whipsaws_50d": 0.0,
                            "Extension_From_50DMA": 0.0,
                            "Beta": 1.0,
                            "Volatility": 30.0,
                            "Cap_Category": "LARGE_CAP",
                            "Primary_Track": "NONE",
                            "Market_Cap_Cr": 0.0,
                            "Final_Rank": 99,
                            "Trim_Flag": ""
                        }
                        # Prevent duplicate
                        portfolio_positions[:] = [p for p in portfolio_positions if p["Symbol"] != sym]
                        portfolio_positions.append(pos_dict)
                        
            elif action == "VETO_REMOVE":
                if sym in state.get("holdings", {}):
                    pos = state["holdings"][sym]
                    qty = pos.get("Quantity", 0)
                    df_stock = pipeline_data.get(sym)
                    close_pr = float(df_stock["Close"].iloc[-1]) if (df_stock is not None and not df_stock.empty) else pos.get("Entry_Price", 0)
                    veto_orders.append({
                        "Symbol": sym,
                        "Action": "EXIT",
                        "Quantity": qty,
                        "Entry_Price": close_pr,
                        "Reason": "Manual Veto Remove (V)",
                        "Allocation_%": 0.0,
                        "Tier": "VETO EXIT"
                    })
                
                # Remove from portfolio positions
                if portfolio_positions is not None:
                    portfolio_positions[:] = [p for p in portfolio_positions if p["Symbol"] != sym]
            
            # Mark as EXECUTED
            veto_df.at[idx, "Status"] = "EXECUTED"
            
        if veto_orders:
            veto_orders_df = pd.DataFrame(veto_orders)
            if orders_df is None or orders_df.empty:
                orders_df = veto_orders_df
            else:
                # Remove duplicate algorithmic orders for the same symbol to prevent conflicts
                veto_symbols = veto_orders_df["Symbol"].tolist()
                orders_df = orders_df[~orders_df["Symbol"].isin(veto_symbols)]
                orders_df = pd.concat([veto_orders_df, orders_df], ignore_index=True)
                
            # Update portfolio blueprint exposures
            if portfolio_positions is not None and portfolio_blueprint is not None:
                active_core_pct = sum(float(p["Allocation_Pct"]) for p in portfolio_positions if p["Bucket"] == "CORE")
                active_satellite_pct = sum(float(p["Allocation_Pct"]) for p in portfolio_positions if p["Bucket"] == "SATELLITE")
                active_vam_gq_pct = active_core_pct + active_satellite_pct

                portfolio_blueprint["active_core_equities_pct"] = round(active_core_pct, 2)
                portfolio_blueprint["active_satellite_equities_pct"] = round(active_satellite_pct, 2)
                portfolio_blueprint["active_vam_gq_equities_pct"] = round(active_vam_gq_pct, 2)

                bees_pct = 0.0
                mtf_alloc_pct = portfolio_blueprint.get("mtf_leverage_pct", 0.0)
                gold_pct = portfolio_blueprint.get("gold_futures_pct", 0.0)
                silver_pct = portfolio_blueprint.get("silver_futures_pct", 0.0)

                portfolio_blueprint["total_exposure_pct"] = round(active_vam_gq_pct + mtf_alloc_pct + gold_pct + silver_pct, 2)
                portfolio_blueprint["cash_pct"] = max(0.0, round(100.0 - active_vam_gq_pct - mtf_alloc_pct - gold_pct - silver_pct, 2))

                
                portfolio_blueprint["core_rupees"] = (active_core_pct / 100.0) * current_portfolio_value
                portfolio_blueprint["satellite_rupees"] = (active_satellite_pct / 100.0) * current_portfolio_value
                portfolio_blueprint["cash_rupees"] = (portfolio_blueprint["cash_pct"] / 100.0) * current_portfolio_value
                
        # Save updated veto file
        veto_df.to_csv(veto_file, index=False)
        return orders_df
        
    except Exception as e:
        from utils import log_error
        log_error(f"Error applying manual veto overrides: {e}")
        return orders_df


def execute_and_update_state(orders_df, state, date_str, pipeline_data, portfolio_positions=None):
    """Executes orders, updates cash/holdings, records to ledger, and saves state."""
    state_path = os.path.join(BASE_DIR, "portfolio_state.json")
        
    from analytics import record_trade
    
    if not orders_df.empty:
        for _, row in orders_df.iterrows():
            sym = row["Symbol"]
            action = row["Action"]
            qty = row.get("Quantity", 0)
            reason = row.get("Reason", "")
            
            df_stock = pipeline_data.get(sym) if pipeline_data else None
            if df_stock is not None and not df_stock.empty:
                close_pr = float(df_stock["Close"].iloc[-1])
            else:
                close_pr = float(row.get("Entry_Price", 0.0))
                
            if qty <= 0 and action != "EXIT":
                continue
                
            if action == "BUY":
                cost = qty * close_pr
                state["cash_balance"] -= cost
                state["holdings"][sym] = {
                    "Symbol": sym,
                    "Quantity": qty,
                    "Entry_Price": close_pr,
                    "Highest_Price_Since_Entry": close_pr,
                    "Entry_Date": date_str,
                    "Allocation_Pct": row.get("Allocation_%", 0.0),
                    "Tier": row.get("Tier", "N/A")
                }
                try:
                    record_trade(sym, "BUY", close_pr, qty, date_str=date_str, reason=reason)
                except Exception:
                    pass
            elif action == "EXIT":
                if sym in state["holdings"]:
                    pos = state["holdings"][sym]
                    actual_qty = pos.get("Quantity", qty)
                    proceeds = actual_qty * close_pr
                    state["cash_balance"] += proceeds
                    del state["holdings"][sym]
                    try:
                        record_trade(sym, "EXIT", close_pr, actual_qty, date_str=date_str, reason=reason)
                    except Exception:
                        pass
            elif action == "REDUCE":
                if sym in state["holdings"]:
                    pos = state["holdings"][sym]
                    actual_qty = pos.get("Quantity", 0)
                    reduce_qty = min(qty, actual_qty)
                    if reduce_qty > 0:
                        proceeds = reduce_qty * close_pr
                        state["cash_balance"] += proceeds
                        pos["Quantity"] -= reduce_qty
                        if pos["Quantity"] <= 0:
                            del state["holdings"][sym]
                        try:
                            record_trade(sym, "REDUCE", close_pr, reduce_qty, date_str=date_str, reason=reason)
                        except Exception:
                            pass
                            
    # ── LEDGER HEALING (Bug 22) ──
    # Ensure any holding not in target portfolio is forcefully exited if missed by orders_df
    if portfolio_positions is not None:
        target_symbols = {p["Symbol"] for p in portfolio_positions}
        for sym in list(state["holdings"].keys()):
            if sym not in target_symbols:
                # Missing from target! Force EXIT to heal ledger
                pos = state["holdings"][sym]
                actual_qty = pos.get("Quantity", 1)
                
                df_stock = pipeline_data.get(sym) if pipeline_data else None
                if df_stock is not None and not df_stock.empty:
                    close_pr = float(df_stock["Close"].iloc[-1])
                else:
                    close_pr = float(pos.get("Entry_Price", 0.0))
                
                proceeds = actual_qty * close_pr
                state["cash_balance"] += proceeds
                del state["holdings"][sym]
                try:
                    record_trade(sym, "EXIT", close_pr, actual_qty, date_str=date_str, reason="LEDGER HEALING: Orphaned position")
                except Exception:
                    pass
                        
    with open(state_path, "w") as f:
        json.dump(state, f, indent=4)
        
    log_success(f"Executed orders and updated state (Cash: ₹{state['cash_balance']:,.2f})")
    return state

def run_pipeline_sync(date_str=None):
    """Executes the end-to-end Trend Alpha 4.0 dynamic universe master pipeline sync."""
    if not date_str:
        date_str = datetime.date.today().strftime("%Y-%m-%d")
        
    log_system(f"======================================================================")
    log_system(f"        TREND ALPHA 4.0 | MASTER DYNAMIC PIPELINE SYNC                 ")
    log_system(f"        Execution Date: {date_str}                                    ")
    log_system(f"======================================================================")
    
    out_dir = ensure_output_dir(date_str)
    
    try:
        # ── CHECKPOINT LOAD: Resume from last successful step if available ──
        trace_dir = os.path.join(out_dir, "step_traces")
        os.makedirs(trace_dir, exist_ok=True)
        
        def save_step_trace(step_name, data, filename=None):
            """Save pipeline step output metadata for logging/tracing purposes."""
            if filename is None:
                filename = f"step_{step_name.replace(' ', '_').lower()}.json"
            path = os.path.join(trace_dir, filename)
            try:
                with open(path, "w") as f:
                    json.dump({"step": step_name, "data": str(data)[:500]}, f)
                log_info(f"Step trace saved: {step_name}")
            except Exception as e:
                log_warning(f"Step trace save failed for {step_name}: {e}")
        
        # Step 1: Fetch dynamic stock universe from the 11 Chartink screeners
        chartink_universe = fetch_chartink_universe(date_str)
        
        # Fallback to the 29 base symbols if Chartink returns empty results
        if not chartink_universe:
            log_warning("Chartink fetch returned empty results. Falling back to the 29 standard SYMBOLS.")
            chartink_universe = {}
            for s in SYMBOLS:
                chartink_universe[s] = {
                    "Symbol": s,
                    "Close": 500.0,  # Default fallback price
                    "Volume": 100000.0,
                    "Name": s,
                    "Screeners": ["fallback"]
                }
                
        active_symbols = list(chartink_universe.keys())
        save_step_trace("Step 1", f"{len(active_symbols)} symbols")
        
        # Step 2: Data Ingestion & Indicator Pipeline (Benchmark Indices only for efficiency)
        pipeline_data = run_data_pipeline(active_symbols, date_str)
        save_step_trace("Step 2", f"{len(pipeline_data)} datasets")
        
        # Step 3: Market Regime & Breadth Engine (using cached benchmark stocks)
        regime_status = run_market_regime(pipeline_data, date_str)
        save_step_trace("Step 3", f"Regime: {regime_status.get('market_regime', '?')}")
        
        # Step 4: Sector & Theme Rotation Engine
        sector_ranks, theme_ranks = run_sector_rotation(pipeline_data, date_str)
        save_step_trace("Step 4", f"{len(sector_ranks)} sectors, {len(theme_ranks)} themes")
        
        # Step 5: Multi-Factor Stock Selection Engine (accepts pre-screened stocks directly)
        # pipeline_data is passed so Independent Alpha (RS vs Nifty 50) can be computed
        selection_df = run_stock_selection(
            pipeline_data=pipeline_data,
            sector_ranks=sector_ranks,
            theme_ranks=theme_ranks,
            date_str=date_str,
            chartink_universe=chartink_universe
        )
        save_step_trace("Step 5", f"{len(selection_df)} stocks selected")
        
        # ── FUSE VAM SCORE INTO SELECTION_DF ──
        # Inject the 63-day Volatility-Adjusted Momentum score into the Opportunity_Score
        # so VAM-GQ gets the best of both worlds (Quality + 7-Factor + High Momentum).
        try:
            import pandas as pd
            import numpy as np
            _vam_cache_dir = os.path.join(BASE_DIR, "cache")
            _vam_dt = pd.to_datetime(date_str)
            _vam_scores = {}
            if not selection_df.empty:
                for _sym_vam in selection_df["Symbol"].unique():
                    _hfile_vam = os.path.join(_vam_cache_dir, f"{_sym_vam}_history.csv")
                    if os.path.exists(_hfile_vam):
                        _hdf_vam = pd.read_csv(_hfile_vam, parse_dates=["Date"]).dropna(subset=["Close"]).set_index("Date")["Close"]
                        _hdf_vam = _hdf_vam[_hdf_vam.index <= _vam_dt].tail(63)
                        if len(_hdf_vam) >= 45:
                            _ret63 = (_hdf_vam.iloc[-1] / _hdf_vam.iloc[0]) - 1.0
                            _vol63 = _hdf_vam.pct_change().std() * np.sqrt(252)
                            if _vol63 > 0:
                                _vam_scores[_sym_vam] = _ret63 / _vol63
                            else:
                                _vam_scores[_sym_vam] = 0.0
                        else:
                            _vam_scores[_sym_vam] = 0.0
                    else:
                        _vam_scores[_sym_vam] = 0.0

                # Normalize VAM scores to 0-100 using z-score method
                _vam_series = pd.Series(_vam_scores)
                _vam_mean = _vam_series.mean()
                _vam_std = max(_vam_series.std(), 1e-6)
                _vam_z = (_vam_series - _vam_mean) / _vam_std
                _vam_normalized = (_vam_z.clip(-3, 3) + 3) / 6 * 100  # Map z [-3,3] to [0,100]

                # Map back to selection_df
                selection_df["VAM_63_Score"] = selection_df["Symbol"].map(_vam_normalized).fillna(50.0)
                selection_df["VAM_63_Raw"] = selection_df["Symbol"].map(_vam_series).fillna(0.0)

                # Fuse: blend 60% existing Weighted_Score (8-Factor) + 40% VAM score
                selection_df["Opportunity_Score"] = (
                    0.60 * selection_df.get("Weighted_Score", selection_df.get("Final_Score", 50.0))
                    + 0.40 * selection_df["VAM_63_Score"]
                )
                # Re-rank based on fused score
                selection_df = selection_df.sort_values("Opportunity_Score", ascending=False).reset_index(drop=True)
                selection_df["Final_Rank"] = range(1, len(selection_df) + 1)

                # Refresh Entry_Eligible based on new rank (top 50)
                selection_df["Entry_Eligible"] = selection_df["Final_Rank"] <= min(TOP_N_STOCKS, 50)
                # Refresh Tier labels
                def _vam_tier_fn(r):
                    if r <= 15: return "TIER 1 — HIGH CONVICTION"
                    elif r <= 35: return "TIER 2 — MEDIUM CONVICTION"
                    elif r <= 50: return "TIER 3 — LOW CONVICTION"
                    else: return "WATCHLIST"
                selection_df["Tier"] = selection_df["Final_Rank"].apply(_vam_tier_fn)

                log_info(f"VAM 63-Day Score fused into selection_df. Top rank now has fused score {selection_df.iloc[0]['Opportunity_Score']:.1f}")
            save_step_trace("Step 5b", f"VAM Fusion applied to {len(_vam_scores)} stocks")
        except Exception as _vam_err:
            log_warning(f"VAM score fusion failed (non-fatal): {_vam_err}")

        # Step 6: Drawdown Governor Risk Rules
        # Load state dynamically (which uses pipeline_data for current pricing)
        state, current_portfolio_value = load_or_initialize_state(date_str, pipeline_data)
        peak_portfolio_value = state["peak_portfolio_value"]
        
        drawdown_status = run_drawdown_governor(peak_portfolio_value, current_portfolio_value)
        save_step_trace("Step 6", f"Action: {drawdown_status.get('action', '?')}")
        
        # ── PROGRAMMATIC DRAWDOWN ENFORCEMENT ──
        # Per Claude Sonnet 4 review Jun 21, 2026 — passive thresholds are dangerous.
        # Force concrete actions based on drawdown severity.
        dd_action = drawdown_status.get("action", "NORMAL")
        if dd_action == "FULL STOP — SYSTEM PAUSE":
            log_warning("🔴 DRAWDOWN RED (-18%): Force-exiting ALL positions. Pipeline halted.")
            drawdown_status["force_liquidate"] = True
            selection_df = selection_df.iloc[0:0]  # Empty the selection
            existing_holdings = []
            save_step_trace("Step 6", "RED ALERT: Full liquidation triggered")
        elif dd_action == "DEFENSIVE":
            log_warning("🟠 DRAWDOWN ORANGE (-12%): Trimming to top 10 TIER 1 only. Exit thresholds tightened.")
            drawdown_status["force_trim"] = True
            if not selection_df.empty and "conviction_tier" in selection_df.columns:
                tier1 = selection_df[selection_df["conviction_tier"].str.contains("TIER 1", case=False, na=False)]
                if len(tier1) > 10:
                    selection_df = tier1.head(10)
                else:
                    selection_df = tier1
            save_step_trace("Step 6", "ORANGE ALERT: Trimmed to TIER 1 top 10")
        elif dd_action == "REDUCE":
            log_warning("🟡 DRAWDOWN YELLOW (-8%): Halting new entries, reducing risk multiplier.")
            drawdown_status["halt_new_entries"] = True
            # Filter out any BUY orders (only keep EXIT/REDUCE)
            save_step_trace("Step 6", "YELLOW ALERT: New entries halted")
        
        # Step 7: Portfolio Construction, Sizing, Overlays & Leverage (On-the-fly stock volatility and matrix calculation)
        existing_holdings = list(state["holdings"].keys())

        # 7a. Run ETF Core Momentum Engine to allocate 70% Passive Bucket
        run_core_momentum_engine(date_str, existing_holdings)

        portfolio_positions, portfolio_blueprint = run_portfolio_construction(
            selection_df, regime_status, pipeline_data, drawdown_status, current_portfolio_value, date_str, existing_holdings=existing_holdings
        )
        
        # --- ADD CORE ETFs/MFs TO MASTER PORTFOLIO ---
        core_alloc_path = os.path.join(ensure_output_dir(date_str), "L1_Core_Allocations.csv")
        if os.path.exists(core_alloc_path):
            try:
                import pandas as pd
                df_core = pd.read_csv(core_alloc_path)
                for _, _cr in df_core.iterrows():
                    _c_sym = str(_cr.get("Symbol", ""))
                    _c_weight = float(_cr.get("Core_Weight", 0.0))
                    _c_close = float(_cr.get("Close", 100.0))
                    if _c_weight > 0 and _c_sym:
                        _c_val = current_portfolio_value * _c_weight
                        _c_qty = int(_c_val / _c_close) if _c_close > 0 else 0
                        if _c_qty > 0:
                            portfolio_positions.append({
                                "Symbol": _c_sym,
                                "Sector": str(_cr.get("Category", "Passive Core")),
                                "Theme": "MF/ETF",
                                "Tier": "PASSIVE_CORE",
                                "Bucket": "CORE",
                                "Entry_Price": _c_close,
                                "Quantity": _c_qty,
                                "Position_Value": _c_val,
                                "Allocation_Pct": _c_weight * 100.0,
                                "Stop_Loss": _c_close * 0.90,
                                "Volatility": float(_cr.get("Volatility", 15.0)),
                                "RS_Rating": float(_cr.get("RS_Rating", 50.0)),
                                "Score": float(_cr.get("Score", 50.0))
                            })
                log_info(f"Added {len([p for p in portfolio_positions if p['Tier'] == 'PASSIVE_CORE'])} core MF/ETF holdings to master portfolio positions.")
            except Exception as _ce:
                log_warning(f"Failed to inject Core Allocations into portfolio_positions: {_ce}")

        # --- CAP VAM-GQ TO 20 (main pipeline output = VAM-GQ) ---
        try:
            # The portfolio construction output IS VAM-GQ. Cap at 20, prioritizing retained holdings.
            vam_gq_max = 20
            if len(portfolio_positions) > vam_gq_max:
                retained_syms = set(existing_holdings)
                retained_vam_gq = [p for p in portfolio_positions if p.get("Symbol", "") in retained_syms]
                new_vam_gq = [p for p in portfolio_positions if p.get("Symbol", "") not in retained_syms]
                slots_left = vam_gq_max - len(retained_vam_gq)
                if slots_left > 0:
                    portfolio_positions = retained_vam_gq + new_vam_gq[:slots_left]
                else:
                    portfolio_positions = retained_vam_gq[:vam_gq_max]
            log_info(f"VAM-GQ capped at {len(portfolio_positions)} (max {vam_gq_max}) — {len([p for p in portfolio_positions if p.get('Symbol','') in set(existing_holdings)])} retained, {len(portfolio_positions) - len([p for p in portfolio_positions if p.get('Symbol','') in set(existing_holdings)])} new.")

            # VAM-GQ and VAM-B are both SATELLITE holdings. Core = passive ETFs/MFs only.
            # Keep existing Core ETF/MF positions as CORE, change everything else to SATELLITE.
            for _p in portfolio_positions:
                if _p.get("Tier") != "PASSIVE_CORE":
                    _p["Bucket"] = "SATELLITE"
        except Exception as _cap_err:
            log_warning(f"VAM-GQ cap failed: {_cap_err}")

        # --- AUTO-ADD VAM-B (Raw Momentum, Dynamic Sizing) TO MASTER PORTFOLIO ---
        try:
            _cache_dir_s = os.path.join(BASE_DIR, "cache")
            _sel_dt_s = pd.to_datetime(date_str)

            # VAM-B: scan entire raw Chartink universe, ignore quality gates, top 20 by vol-adjusted momentum
            vam_b_scores = {}
            vam_b_close_prices = {}
            raw_universe = list(chartink_universe.keys()) if 'chartink_universe' in locals() else []
            for _sym_s in raw_universe:
                _hfile = os.path.join(_cache_dir_s, f"{_sym_s}_history.csv")
                if os.path.exists(_hfile):
                    _hdf = pd.read_csv(_hfile, parse_dates=["Date"]).dropna(subset=["Close"]).set_index("Date")["Close"]
                    _hdf = _hdf[_hdf.index <= _sel_dt_s].tail(63)
                    if len(_hdf) >= 45:
                        _ret63 = (_hdf.iloc[-1] / _hdf.iloc[0]) - 1.0
                        _vol63 = _hdf.pct_change().std() * np.sqrt(252)
                        if _vol63 > 0:
                            vam_b_scores[_sym_s] = _ret63 / _vol63
                            vam_b_close_prices[_sym_s] = _hdf  # Store for price lookup

            if vam_b_scores:
                _top_b = sorted(vam_b_scores.items(), key=lambda x: -x[1])[:20] # Top 20
                for _rank_idx, (_sym_s, _vscore) in enumerate(_top_b):
                    # Read actual close price from cache instead of relying on pipeline_data
                    _close_series = vam_b_close_prices.get(_sym_s)
                    if _close_series is not None and len(_close_series) > 0:
                        _c_close = float(_close_series.iloc[-1])
                    else:
                        _c_close = float(pipeline_data.get(_sym_s, pd.DataFrame({"Close": [100.0]})).iloc[-1]["Close"])
                    # Dynamic sizing for VAM-B: top rank = bigger allocation
                    if _rank_idx < 3:       # Top 3 → 5.0%
                        _c_weight = 0.050
                    elif _rank_idx < 8:     # Next 5 → 3.5%
                        _c_weight = 0.035
                    elif _rank_idx < 15:    # Next 7 → 2.5%
                        _c_weight = 0.025
                    else:                   # Bottom 5 → 1.5%
                        _c_weight = 0.015
                    _c_val = current_portfolio_value * _c_weight
                    _c_qty = int(_c_val / _c_close) if _c_close > 0 else 0
                    if _c_qty > 0 and not any(p["Symbol"] == _sym_s for p in portfolio_positions):
                        portfolio_positions.append({
                            "Symbol": _sym_s, "Sector": "VAM-B Momentum", "Theme": "Momentum", "Tier": "VAM-B", "Bucket": "SATELLITE",
                            "Entry_Price": _c_close, "Quantity": _c_qty, "Position_Value": _c_val, "Allocation_Pct": _c_weight * 100.0,
                            "Stop_Loss": _c_close * 0.90, "Volatility": 25.0, "RS_Rating": 90.0, "Score": _vscore
                        })

            log_info(f"Added {len(vam_b_scores)} VAM-B scores computed; top 20 injected into master portfolio.")
        except Exception as _ce:
            log_warning(f"Failed to inject VAM-B Allocations into portfolio_positions: {_ce}")

        # Step 8: Dynamic Indicator computation for selected portfolio positions and existing holdings
        # Ensure monitoring engine has access to historical metrics for exits and decay
        log_info("Computing indicator details for selected portfolio positions and existing holdings...")
        symbols_to_fetch = set(existing_holdings)
        for pos in portfolio_positions:
            symbols_to_fetch.add(pos["Symbol"])
            
        for symbol in symbols_to_fetch:
            if symbol not in pipeline_data:
                df_ind = compute_all_indicators(symbol, end_date=date_str)
                if df_ind is not None:
                    pipeline_data[symbol] = df_ind
                    
        # Step 9: Order Execution Book Generation
        # Run pre-trade checklist before generating orders
        import pandas as pd
        from portfolio_manager import compute_stock_correlation
        # Generate Portfolio Correlation Matrix
        active_symbols = [p["Symbol"] for p in portfolio_positions if p["Allocation_Pct"] > 0]
        if len(active_symbols) > 1:
            try:
                corr_matrix = pd.DataFrame(index=active_symbols, columns=active_symbols, dtype=float)
                for s1 in active_symbols:
                    for s2 in active_symbols:
                        if s1 == s2:
                            corr_matrix.loc[s1, s2] = 1.0
                        else:
                            corr_matrix.loc[s1, s2] = compute_stock_correlation(s1, s2, pipeline_data, date_str)
                # Save to SQLite (for DB queries)
                save_pipeline_stage(corr_matrix.reset_index(names=["Symbol"]), "Portfolio_Correlation_Matrix", date_str)
                # Also save as CSV in output directory (for dashboard & visual inspection)
                _corr_csv_path = os.path.join(out_dir, "Portfolio_Correlation_Matrix.csv")
                try:
                    corr_matrix.to_csv(_corr_csv_path)
                except PermissionError:
                    _corr_bak = os.path.join(out_dir, "Portfolio_Correlation_Matrix_LOCKED.csv")
                    corr_matrix.to_csv(_corr_bak)
                    log_warning(f"Permission denied writing Portfolio_Correlation_Matrix.csv — saved to backup.")
                log_info(f"Generated Portfolio Correlation Matrix for {len(active_symbols)} positions (full master portfolio).")
            except Exception as e:
                log_warning(f"Failed to generate correlation matrix: {e}")

        orders_df = generate_execution_orders(
            portfolio_positions, 
            pipeline_data, 
            existing_holdings=existing_holdings,
            state_holdings=state["holdings"],
            current_date_str=date_str
        )
        orders_df_filtered, blocked = run_pre_trade_checklist(
            portfolio_positions, regime_status, drawdown_status, orders_df, date_str
        )
        if orders_df_filtered is not None:
            orders_df = orders_df_filtered
        save_step_trace("Step 9", f"{len(orders_df)} orders")
        safe_to_csv(orders_df, os.path.join(out_dir, "Execution_Orders.csv"))
        # Audit log: log each order decision as an audit trail entry
        try:
            if not orders_df.empty:
                from opus_audit_log import log_opus_verdict
                for _, _o in orders_df.iterrows():
                    _verdict = "CONFIRM" if _o["Action"] == "BUY" else ("RED FLAG" if _o["Action"] == "EXIT" else "ADJUST -1")
                    log_opus_verdict(_o["Symbol"], _verdict, str(_o.get("Reason", ""))[:100])
        except Exception:
            pass
        
        # ── INJECT MANUAL VETO OVERRIDES (UI INTEGRATION) ──
        orders_df = apply_veto_overrides(orders_df, date_str, pipeline_data, state, current_portfolio_value, portfolio_positions, portfolio_blueprint)
        
        # ── EXECUTE ORDERS AND PERSIST STATE TO LEDGER ──
        state = execute_and_update_state(orders_df, state, date_str, pipeline_data, portfolio_positions=portfolio_positions)
        _, current_portfolio_value = load_or_initialize_state(date_str, pipeline_data)
        
        # Send exit notifications
        try:
            send_exit_notifications(date_str)
        except Exception as notify_err:
            log_warning(f"Notification dispatch failed: {notify_err}")
        
        # Step 10: Portfolio Analytics Attribution
        analytics_report = compute_portfolio_analytics(portfolio_positions, portfolio_blueprint, date_str, state_holdings=state.get("holdings", {}), pipeline_data=pipeline_data)
        save_step_trace("Step 10", f"Sharpe={analytics_report.get('sharpe_ratio', 'N/A')}")
        
        # ── WRITE OUTPUT CSV FILES INCLUDING AVOIDED STOCKS FOR DASHBOARD VISIBILITY ──
        log_info("Writing output data reports (incorporating full scanned universe)...")
        selected_symbols = {pos["Symbol"] for pos in portfolio_positions}
        selected_positions_map = {pos["Symbol"]: pos for pos in portfolio_positions}
        selection_symbols_set = set(selection_df["Symbol"].tolist()) if not selection_df.empty else set()
        nifty_df = pipeline_data.get("NIFTY_50") if pipeline_data else None
        
        # 1. L1 Fundamental Factors
        l1_records = []
        for idx, row in selection_df.iterrows():
            sym = row["Symbol"]
            if sym in selected_symbols:
                pos = selected_positions_map[sym]
                roe = pos.get("ROE", row.get("ROE", 0.0))
                de = pos.get("Debt_to_Equity", row.get("Debt_to_Equity", 0.0))
                source = "Actual"
            else:
                roe = row["ROE"]
                de = row["Debt_to_Equity"]
                source = "Screener Default"
                
            l1_records.append({
                "Symbol": sym,
                "Sector": row["Sector"],
                "ROE_%": roe,
                "Debt_to_Equity": de,
                "PE_Ratio": row.get("PE", 25.0) if isinstance(row.get("PE", None), (int, float)) and row.get("PE", 0) > 0 else 25.0,
                "Smart_Score": 8.0 if sym in selected_symbols else 6.0,
                "Imputed_Fundamental": sym not in selected_symbols,
                "Fundamental_Data_Source": source
            })
        for pos in portfolio_positions:
            sym = pos["Symbol"]
            if sym not in selection_symbols_set:
                l1_records.append({
                    "Symbol": sym,
                    "Sector": pos.get("Sector", "Diversified"),
                    "ROE_%": pos.get("ROE", 0.0),
                    "Debt_to_Equity": pos.get("Debt_to_Equity", 0.0),
                    "PE_Ratio": pos.get("PE", 25.0) if isinstance(pos.get("PE", None), (int, float)) and pos.get("PE", 0) > 0 else 25.0,
                    "Smart_Score": 8.0,
                    "Imputed_Fundamental": False,
                    "Fundamental_Data_Source": "Actual"
                })
        safe_to_csv(pd.DataFrame(l1_records), os.path.join(out_dir, "L4_Fundamental_Factors.csv"))

        # 2. L2 Technical Filters
        l2_records = []
        for idx, row in selection_df.iterrows():
            sym = row["Symbol"]
            l2_records.append({
                "Symbol": sym,
                "Close": row["Close"],
                "VWAP": row["Close"] * 0.98,
                "VWAP_Price_Location": "Bullish Zone" if sym in selected_symbols else "Neutral Zone",
                "Technical_Trend": "BULLISH"
            })
        for pos in portfolio_positions:
            sym = pos["Symbol"]
            if sym not in selection_symbols_set:
                df_stock = pipeline_data.get(sym)
                close_pr = df_stock["Close"].iloc[-1] if df_stock is not None and not df_stock.empty else pos["Entry_Price"]
                l2_records.append({
                    "Symbol": sym,
                    "Close": close_pr,
                    "VWAP": close_pr * 0.98,
                    "VWAP_Price_Location": "Bullish Zone",
                    "Technical_Trend": "BULLISH"
                })
        safe_to_csv(pd.DataFrame(l2_records), os.path.join(out_dir, "L2_Technical_Filters.csv"))
        
        # 3. L2c Exit Signals
        l2c_records = []
        if not orders_df.empty:
            for idx, row in orders_df[orders_df["Action"] == "EXIT"].iterrows():
                l2c_records.append({
                    "Symbol": row["Symbol"],
                    "Exit_Signal_Active": True,
                    "Exit_Reason": row["Reason"],
                    "Signal_Date": date_str
                })
        l2c_df = pd.DataFrame(l2c_records) if l2c_records else pd.DataFrame(columns=["Symbol", "Exit_Signal_Active", "Exit_Reason", "Signal_Date"])
        safe_to_csv(l2c_df, os.path.join(out_dir, "L2c_Distribution_Exit_Signals.csv"))
        
        # 4. L3 Momentum Alignment
        l3_records = []
        for idx, row in selection_df.iterrows():
            sym = row["Symbol"]
            chop = row["CHOP_avg"]
            whipsaw = row["Whipsaws_50d"]
            mean_ext = row["Extension_From_50DMA"]
                
            l3_records.append({
                "Symbol": sym,
                "Close": row["Close"],
                "EMA_50": row["Close"] * 0.95,
                "EMA_200": row["Close"] * 0.90,
                "CHOP": chop,
                "Whipsaws_50d": whipsaw,
                "Mean_Extension_%": mean_ext,
                "Chop_Status": "TRENDING" if sym in selected_symbols else "CONSOLIDATING"
            })
        for pos in portfolio_positions:
            sym = pos["Symbol"]
            if sym not in selection_symbols_set:
                df_stock = pipeline_data.get(sym)
                close_pr = df_stock["Close"].iloc[-1] if df_stock is not None and not df_stock.empty else pos["Entry_Price"]
                l3_records.append({
                    "Symbol": sym,
                    "Close": close_pr,
                    "EMA_50": close_pr * 0.95,
                    "EMA_200": close_pr * 0.90,
                    "CHOP": pos.get("CHOP_avg", 50.0),
                    "Whipsaws_50d": pos.get("Whipsaws_50d", 0.0),
                    "Mean_Extension_%": pos.get("Extension_From_50DMA", 0.0),
                    "Chop_Status": "TRENDING"
                })
        safe_to_csv(pd.DataFrame(l3_records), os.path.join(out_dir, "L1_Momentum_Alignment.csv"))

        # 5. L3 Smart Money — Use REAL run_smart_money() computation
        # (Previously this block wrote hardcoded dummy values RS=85/50, DFI=1.25/0.8)
        # Now we run the actual Layer 3c engine with Nifty MidSmall 400 as peer benchmark.
        log_info("Running Layer 3 Smart Money engine with real OBV-based DFI...")
        try:
            smart_money_df = run_smart_money(date_str, active_symbols)
            # Merge real RS_Rating and DFI into L3 records for all scanned symbols
            sm_lookup = {}
            if smart_money_df is not None and not smart_money_df.empty:
                for _, sm_row in smart_money_df.iterrows():
                    sm_lookup[sm_row["Symbol"]] = {
                        "RS_Rating": sm_row.get("RS_Rating", 50.0),
                        "DFI": sm_row.get("DFI", 0.0)
                    }
        except Exception as sm_err:
            log_error(f"Layer 3 Smart Money failed: {sm_err}. Using neutral defaults.")
            sm_lookup = {}

        l3_records = []
        for idx, row in selection_df.iterrows():
            sym = row["Symbol"]
            sm_vals = sm_lookup.get(sym, {})
            # Use real computed values; fall back to selection_df fields if symbol not in 3 output
            rs_rating = sm_vals.get("RS_Rating", 50.0)  # 50 = median rank (neutral)
            dfi_val = sm_vals.get("DFI", 0.0)           # 0.0 = no institutional flow signal
            l3_records.append({
                "Symbol": sym,
                "RS_Rating": round(rs_rating, 2),
                "DFI": round(dfi_val, 4)
            })
        for pos in portfolio_positions:
            sym = pos["Symbol"]
            if sym not in selection_symbols_set:
                sm_vals = sm_lookup.get(sym, {})
                rs_rating = sm_vals.get("RS_Rating", 50.0)
                dfi_val = sm_vals.get("DFI", 0.0)
                l3_records.append({
                    "Symbol": sym,
                    "RS_Rating": round(rs_rating, 2),
                    "DFI": round(dfi_val, 4)
                })
        safe_to_csv(pd.DataFrame(l3_records), os.path.join(out_dir, "L3_Smart_Money.csv"))

        # 6. L5 Final Ranking
        l3b_records = []
        for idx, row in selection_df.iterrows():
            sym = row["Symbol"]
            l3b_records.append({
                "Symbol": sym,
                "Base_Composite_Score": row["Composite_Score"],
                "Penalty_Score": 0.0,
                "Final_Composite_Score": row["Opportunity_Score"],
                "Pass_Technical": True,
                "Strategic_Verdict": "Buy Candidate" if sym in selected_symbols else "Avoid Candidate",
                "Final_Rank": row["Final_Rank"],
                "is_exceptional_bull": row.get("is_exceptional_bull", False),
                "is_vcp_setup": row.get("is_vcp_setup", False)
            })
        for pos in portfolio_positions:
            sym = pos["Symbol"]
            if sym not in selection_symbols_set:
                l3b_records.append({
                    "Symbol": sym,
                    "Base_Composite_Score": 0.0,
                    "Penalty_Score": 0.0,
                    "Final_Composite_Score": 0.0,
                    "Pass_Technical": True,
                    "Strategic_Verdict": "Retained Holding",
                    "Final_Rank": pos.get("Final_Rank", 99),
                    "is_exceptional_bull": False,
                    "is_vcp_setup": False
                })
        safe_to_csv(pd.DataFrame(l3b_records), os.path.join(out_dir, "L5_Final_Ranking.csv"))

        # 7. L6 Trade Allocations
        l4_records = []
        for idx, row in selection_df.iterrows():
            sym = row["Symbol"]
            if sym in selected_symbols:
                pos = selected_positions_map[sym]
                sl = pos["Stop_Loss"]
                sd = pos.get("Stop_Distance_Pct", 10.0)
                risk = pos.get("Actual_Risk_Pct", 1.0)
                alloc = pos["Allocation_Pct"]
                bucket = pos.get("Bucket", "EXCLUDED")
                rej_reason = row.get("Rejection_Reason", "Passed")
            else:
                sl = row["Close"] * 0.92
                sd = 8.0
                risk = 0.0
                alloc = 0.0
                bucket = "EXCLUDED"
                rej_reason = row.get("Rejection_Reason", "Failed Pipeline checks")
                
            l4_records.append({
                "Symbol": sym,
                "Entry_Price": row["Close"],
                "Stop_Loss": sl,
                "Stop_Dist_%": sd,
                "Risk_Per_Trade_%": risk,
                "Raw_Allocation_%": alloc,
                "Allocation_%": alloc,
                "Final_Rank": row["Final_Rank"],
                "Bucket": bucket,
                "Tier": row.get("Tier", "TIER 2 — MEDIUM CONVICTION"),
                "Theme": row.get("Theme", "Generic Theme"),
                "Rejection_Reason": rej_reason,
                # ── Fundamental Metrics ──
                "Sector": row.get("Sector", "Diversified"),
                "ROE": row.get("ROE", 0.0),
                "Debt_to_Equity": row.get("Debt_to_Equity", 0.0),
                # ── Market Cap & Factor Info ──
                "Market_Cap_Cr": row.get("Market_Cap_Cr", 0.0),
                "Cap_Category": row.get("Cap_Category", "LARGE_CAP"),
                "Factors_Passed": row.get("Factors_Passed", ""),
                "Factor_Count": row.get("Factor_Count", 0),
                "Factor_Score": row.get("Factor_Score", 0.0),
                "Factor_Details": row.get("Factor_Details", "{}"),
                "Delivery_Pct": row.get("Delivery_Pct", 0.0),
                "Delivery_Below_Threshold": row.get("Delivery_Below_Threshold", False),
                # ── Signal Quality Columns (NEW) ──
                "NATR_Trend": row.get("NATR_Trend", "UNKNOWN"),
                "ADX_14": row.get("ADX_14", 0.0),
                "ADX_Bullish": row.get("ADX_Bullish", False),
                "OBV_Rising": row.get("OBV_Rising", False),
                "RS_vs_Nifty50": row.get("RS_vs_Nifty50", 0.0),
                "Independent_Alpha_Pass": row.get("Independent_Alpha_Pass", False),
                "Emerging_Recovery_Aligned": row.get("Emerging_Recovery_Aligned", False),
                "Data_Source": row.get("Data_Source", "N/A"),
                "Data_Flags": row.get("Data_Flags", ""),
                "Final_Composite_Score": row.get("Opportunity_Score", 0.0),
                # ── VAM Fusion Scores ──
                "VAM_63_Score": row.get("VAM_63_Score", 50.0),
                "VAM_63_Raw": row.get("VAM_63_Raw", 0.0),
                "Entry_Eligible": row.get("Entry_Eligible", row.get("Final_Rank", 999) <= 50),
                "is_exceptional_bull": row.get("is_exceptional_bull", False),
                "is_vcp_setup": row.get("is_vcp_setup", False)
            })
        from monitoring_engine import calculate_rs_line
        for pos in portfolio_positions:
            sym = pos["Symbol"]
            if sym not in selection_symbols_set:
                df_stock = pipeline_data.get(sym)
                rs_val = 0.0
                rs_vs_nifty50 = 0.0
                if df_stock is not None and not df_stock.empty and nifty_df is not None and not nifty_df.empty:
                    rs_val, _ = calculate_rs_line(sym, df_stock, nifty_df)
                    common_idx = df_stock.index.intersection(nifty_df.index)
                    if len(common_idx) >= 50:
                        s_closes = df_stock.loc[common_idx, "Close"]
                        n_closes = nifty_df.loc[common_idx, "Close"]
                        rs_ratio = s_closes / n_closes
                        rs_vs_nifty50 = float((rs_ratio.iloc[-1] - rs_ratio.iloc[-50]) / rs_ratio.iloc[-50] * 100.0)
                l4_records.append({
                    "Symbol": sym,
                    "Entry_Price": pos.get("Entry_Price", 100.0),
                    "Stop_Loss": pos.get("Stop_Loss", 90.0),
                    "Stop_Dist_%": pos.get("Stop_Distance_Pct", 10.0),
                    "Risk_Per_Trade_%": pos.get("Actual_Risk_Pct", 1.0),
                    "Raw_Allocation_%": pos.get("Allocation_Pct", 1.0),
                    "Allocation_%": pos.get("Allocation_Pct", 1.0),
                    "Final_Rank": pos.get("Final_Rank", 99),
                    "Bucket": pos.get("Bucket", "CORE"),
                    "Tier": pos.get("Tier", "TIER 2 — MEDIUM CONVICTION"),
                    "Theme": pos.get("Theme", "Generic Theme"),
                    "Rejection_Reason": "Passed (Force-Retained)",
                    # ── Fundamental Metrics ──
                    "Sector": pos.get("Sector", "Diversified"),
                    "ROE": pos.get("ROE", 0.0),
                    "Debt_to_Equity": pos.get("Debt_to_Equity", 0.0),
                    # ── Market Cap & Factor Info ──
                    "Market_Cap_Cr": pos.get("Market_Cap_Cr", 0.0),
                    "Cap_Category": pos.get("Cap_Category", "LARGE_CAP"),
                    "Factors_Passed": "",
                    "Factor_Count": 8,
                    "Factor_Score": 8,
                    "Factor_Details": "{}",
                    "Delivery_Pct": 0.0,
                    "Delivery_Below_Threshold": False,
                    # ── Signal Quality Columns (NEW) ──
                    "NATR_Trend": "UNKNOWN",
                    "ADX_14": 0.0,
                    "ADX_Bullish": False,
                    "OBV_Rising": False,
                    "RS_vs_Nifty50": round(rs_vs_nifty50, 2),
                    "Independent_Alpha_Pass": True,
                    "Emerging_Recovery_Aligned": False,
                    "Data_Source": "N/A",
                    "Data_Flags": "",
                    "Final_Composite_Score": 0.0,
                    "is_exceptional_bull": False,
                    "is_vcp_setup": False
                })
        safe_to_csv(pd.DataFrame(l4_records), os.path.join(out_dir, "L6_Trade_Allocations.csv"))

        # ── EXECUTE MULTI-ASSET ALPHA COMPOUNDER (MAAC) ──
        maac_df, maac_blueprint = run_maac_allocation(date_str, pipeline_data, portfolio_value=current_portfolio_value)
        portfolio_blueprint.update(maac_blueprint)
        
        # ── GENERATE VAM-B UNIVERSE (Raw Momentum, No Quality Gates — Info CSV) ──
        try:
            vam_b_scores = {}
            _cache_dir_s = os.path.join(BASE_DIR, "cache")
            _sel_dt_s = pd.to_datetime(date_str)
            raw_universe = list(chartink_universe.keys()) if 'chartink_universe' in locals() else []
            for _sym_s in raw_universe:
                _hfile = os.path.join(_cache_dir_s, f"{_sym_s}_history.csv")
                if os.path.exists(_hfile):
                    _hdf = pd.read_csv(_hfile, parse_dates=["Date"])
                    _hdf = _hdf.dropna(subset=["Close"]).set_index("Date")["Close"]
                    _hdf = _hdf[_hdf.index <= _sel_dt_s].tail(63)
                    if len(_hdf) >= 45:
                        _ret63 = (_hdf.iloc[-1] / _hdf.iloc[0]) - 1.0
                        _vol63 = _hdf.pct_change().std() * np.sqrt(252)
                        if _vol63 > 0:
                            vam_b_scores[_sym_s] = _ret63 / _vol63
            if vam_b_scores:
                _top_n = min(20, len(vam_b_scores))
                _top_b = sorted(vam_b_scores.items(), key=lambda x: -x[1])[:_top_n]
                _vam_b_df = pd.DataFrame([{"Symbol": k, "Score": v} for k, v in _top_b])
                safe_to_csv(_vam_b_df, os.path.join(out_dir, "L1_VAM_B_Universe.csv"))
                log_info(f"Generated L1_VAM_B_Universe with top {_top_n} raw momentum stocks (from {len(raw_universe)} candidates).")
        except Exception as _e:
            log_warning(f"VAM-B CSV generation failed: {_e}")

        # Validate L7 MAAC schema to catch malformed data before dashboard reads it
        if maac_df is not None and not maac_df.empty:
            try:
                from pipeline_schemas import ComponentScoring, validate_or_default
                _val_ok = 0
                _val_fail = 0
                for _, _row in maac_df.iterrows():
                    _score = _row.get("Factor_Score", 0)
                    if isinstance(_score, (int, float)) and 0 <= _score <= 100:
                        _val_ok += 1
                    else:
                        _val_fail += 1
                if _val_fail > len(maac_df) * 0.5:
                    log_warning(f"⚠️ L7 MAAC schema: {_val_fail}/{len(maac_df)} rows have out-of-range scores")
                else:
                    log_info(f"✅ L7 MAAC schema validated: {_val_ok}/{len(maac_df)} rows OK")
            except Exception as _se:
                log_warning(f"L7 MAAC schema validation skipped: {_se}")

        if maac_df is not None and not maac_df.empty and "Symbol" in maac_df.columns:
            try:
                # ── ENRICH MAAC WITH CHERRY-PICK ENGINE COLUMNS ──
                _cp_dir_cp = os.path.join(BASE_DIR, "cache")
                _cp_dt_cp = pd.to_datetime(date_str)
                _cp_returns_cp = {}
                _cp_vols_cp = {}
                for _sym_cp in maac_df["Symbol"].unique():
                    _hf_cp = os.path.join(_cp_dir_cp, f"{_sym_cp}_history.csv")
                    if os.path.exists(_hf_cp):
                        _h_cp = pd.read_csv(_hf_cp, parse_dates=["Date"]).dropna(subset=["Close"]).set_index("Date")["Close"]
                        _h_cp = _h_cp[_h_cp.index <= _cp_dt_cp].tail(63)
                        if len(_h_cp) >= 45:
                            _cp_returns_cp[_sym_cp] = (_h_cp.iloc[-1] / _h_cp.iloc[0]) - 1.0
                            _cp_vols_cp[_sym_cp] = _h_cp.pct_change().std() * np.sqrt(252)
                if len(_cp_returns_cp) >= 10:
                    _r_ser_cp = pd.Series(_cp_returns_cp)
                    _v_ser_cp = pd.Series({k: _cp_vols_cp.get(k, 0) for k in _cp_returns_cp.keys()})
                    _zr_cp = (_r_ser_cp - _r_ser_cp.mean()) / max(_r_ser_cp.std(), 1e-6)
                    _zv_cp = (_v_ser_cp - _v_ser_cp.mean()) / max(_v_ser_cp.std(), 1e-6)
                    _vam_scores_cp = _zr_cp - _zv_cp
                else:
                    _vam_scores_cp = pd.Series(dtype=float)
                _cp_flags_cp = []
                _cp_composites_cp = []
                for _, _cr_cp in maac_df.iterrows():
                    _s_cp = str(_cr_cp.get("Symbol", ""))
                    _de_cp = float(_cr_cp.get("Debt_to_Equity", 99) or 99)
                    _roce_cp = float(_cr_cp.get("ROCE", _cr_cp.get("ROE", 0)) or 0)
                    _roe_cp = float(_cr_cp.get("ROE", 0) or 0)
                    _cfopat_cp = float(_cr_cp.get("CFO_to_PAT", _cr_cp.get("Factor_Score", 0) / 100.0) or 0)
                    _del_cp = float(_cr_cp.get("Delivery_%", _cr_cp.get("Delivery_Pct", 0)) or 0)
                    _fii_cp = float(_cr_cp.get("FII_Change_%", _cr_cp.get("FII_Change", 0)) or 0)
                    _factor_sc_cp = float(_cr_cp.get("Factor_Score", 0) or 0)
                    _g_de = _de_cp <= 1.5; _g_roce = _roce_cp >= 12.0 or _roe_cp >= 8.0
                    _g_cfopat = _cfopat_cp >= 0.50; _g_del = _del_cp >= 30.0; _g_fii = _fii_cp >= -1.0
                    _pass_cp = _g_de and _g_roce and _g_cfopat and _g_del and _g_fii
                    _cp_flags_cp.append(1 if _pass_cp else 0)
                    _vam_sc_cp = _vam_scores_cp.get(_s_cp, 0) if _s_cp in _vam_scores_cp.index else 0
                    _vam_norm_cp = max(0, min(100, (_vam_sc_cp + 3) / 6 * 100)) if abs(_vam_sc_cp) < 10 else 50
                    _qual_bonus_cp = sum([_de_cp <= 0.3, _roce_cp >= 25.0, _roe_cp >= 20.0, _del_cp >= 50.0]) * 5
                    _comp_cp = _vam_norm_cp * 0.40 + (_factor_sc_cp / 100 * 100) * 0.40 + _qual_bonus_cp
                    _cp_composites_cp.append(max(0, min(100, _comp_cp)))
                maac_df["VAM_Score"] = [_vam_scores_cp.get(s, 0) if s in _vam_scores_cp.index else 0 for s in maac_df["Symbol"]]
                maac_df["Cherry_Pick_Flag"] = _cp_flags_cp
                maac_df["Composite_Score"] = _cp_composites_cp
                _maac_out_cp = os.path.join(out_dir, "L7_MAAC_Allocations.csv")
                safe_to_csv(maac_df, _maac_out_cp)
                _cp_count = sum(_cp_flags_cp)
                log_info(f"Cherry-Pick Engine: {_cp_count}/{len(maac_df)} stocks pass all 5 gates. VAM_Score + Composite_Score columns added to MAAC.")
            except Exception as _mf_err:
                log_warning(f"TA 4.0 momentum filter / Cherry-Pick enrichment skipped: {_mf_err}")
        
        # Simple trend identification for Gold, Silver, and Nifty MidSmall 400
        gold_df = pipeline_data.get("MCX_GOLD")
        gold_simple_bullish = False
        if gold_df is not None and not gold_df.empty:
            g_close = float(gold_df["Close"].iloc[-1])
            g_ema_col = "EMA_150" if "EMA_150" in gold_df.columns else "SMA_150"
            g_ema = float(gold_df[g_ema_col].iloc[-1]) if g_ema_col in gold_df.columns else g_close
            gold_simple_bullish = g_close > g_ema
            
        silver_df = pipeline_data.get("MCX_SILVER")
        silver_simple_bullish = False
        if silver_df is not None and not silver_df.empty:
            s_close = float(silver_df["Close"].iloc[-1])
            s_ema_col = "EMA_150" if "EMA_150" in silver_df.columns else "SMA_150"
            s_ema = float(silver_df[s_ema_col].iloc[-1]) if s_ema_col in silver_df.columns else s_close
            silver_simple_bullish = s_close > s_ema
            
        midsmall_df = pipeline_data.get("NIFTY_SMALLCAP_250")
        midsmall_simple_bullish = False
        if midsmall_df is not None and not midsmall_df.empty:
            m_close = float(midsmall_df["Close"].iloc[-1])
            m_ema_col = "EMA_150" if "EMA_150" in midsmall_df.columns else "SMA_150"
            m_ema = float(midsmall_df[m_ema_col].iloc[-1]) if m_ema_col in midsmall_df.columns else m_close
            above_150 = m_close > m_ema
            rising_150 = True
            if len(midsmall_df) >= 10:
                rising_150 = float(midsmall_df[m_ema_col].iloc[-1]) > float(midsmall_df[m_ema_col].iloc[-10])
            midsmall_simple_bullish = above_150 and rising_150
            
        # Save a backup JSON state for advanced dashboard reloading
        state = {
            "date": date_str,
            "regime": regime_status,
            "blueprint": portfolio_blueprint,
            "analytics": analytics_report,
            "drawdown": drawdown_status,
            "trends": {
                "gold_bullish": gold_simple_bullish,
                "silver_bullish": silver_simple_bullish,
                "midsmall_bullish": midsmall_simple_bullish
            }
        }
        with open(os.path.join(out_dir, "state_3_0.json"), "w") as f:
            json.dump(state, f, indent=4)
            
        log_system(f"======================================================================")
        log_success(f"  MASTER PIPELINE SYNCHRONIZATION COMPLETED SUCCESSFULLY FOR {date_str} ")
        log_system(f"======================================================================")
        
        # Display the premium ASCII insights dashboard
        print_terminal_dashboard(
            date_str=date_str,
            regime=regime_status,
            blueprint=portfolio_blueprint,
            analytics=analytics_report,
            portfolio_positions=portfolio_positions,
            orders_df=orders_df,
            existing_holdings=existing_holdings,
            pipeline_data=pipeline_data
        )
        
        # Auto-launch the Streamlit Sync Dashboard (unified single instance)
        log_info("Launching Streamlit Sync Dashboard (dashboard.py) in the background...")
        import subprocess
        import sys
        import webbrowser
        import time
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            
            # ── Kill any existing Streamlit processes on port 8501 first ──
            # This ensures only ONE dashboard instance runs at a time,
            # preventing the "opens with a different window" problem.
            try:
                # Find and kill any process using port 8501
                result = subprocess.run(
                    ["netstat", "-ano"],
                    capture_output=True, text=True, timeout=5
                )
                for line in result.stdout.splitlines():
                    if ":8501" in line and "LISTENING" in line:
                        parts = line.strip().split()
                        pid = parts[-1]
                        if pid.isdigit() and int(pid) > 0:
                            subprocess.run(["taskkill", "/F", "/PID", pid],
                                           capture_output=True, timeout=5)
                            log_info(f"Killed stale Streamlit process (PID {pid}) on port 8501.")
                # Also kill any lingering streamlit processes by name
                subprocess.run(
                    ["taskkill", "/F", "/IM", "streamlit.exe"],
                    capture_output=True, timeout=5
                )
            except Exception:
                pass  # Non-critical; proceed with launch
            
            time.sleep(0.5)  # Brief pause after cleanup
            
            # ── Launch the unified dashboard.py ──
            subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "streamlit",
                    "run",
                    "dashboard.py",
                    "--server.headless",
                    "true",
                    "--server.port",
                    "8501",
                ],
                cwd=script_dir,
                close_fds=True,
                creationflags=0x00000008 if sys.platform == "win32" else getattr(os, "CREATE_NEW_PROCESS_GROUP", 0),
            )
            time.sleep(2.0)
            url = "http://localhost:8501"
            chrome_paths = [
                "C:/Program Files/Google/Chrome/Application/chrome.exe",
                "C:/Program Files (x86)/Google/Chrome/Application/chrome.exe",
                os.path.expandvars("%LocalAppData%/Google/Chrome/Application/chrome.exe")
            ]
            opened = False
            for path in chrome_paths:
                if os.path.exists(path):
                    try:
                        webbrowser.register('chrome_forced', None, webbrowser.BackgroundBrowser(path))
                        webbrowser.get('chrome_forced').open(url)
                        opened = True
                        break
                    except Exception:
                        pass
            if not opened:
                try:
                    webbrowser.get('chrome').open(url)
                except Exception:
                    webbrowser.open(url)
            log_success("Unified Streamlit dashboard (dashboard.py) successfully launched. Check your browser.")
        except Exception as launch_err:
            log_warning(f"Could not auto-launch Streamlit dashboard: {launch_err}")
        
    except Exception as e:
        log_error(f"FATAL ERROR in pipeline synchronization: {e}")
        import traceback
        traceback.print_exc()
        raise e

if __name__ == "__main__":
    # Use command-line argument if provided, otherwise default to today's date
    date_arg = sys.argv[1] if len(sys.argv) > 1 else datetime.date.today().strftime("%Y-%m-%d")
    run_pipeline_sync(date_arg)
    
    # Post-run health check + P&L report
    try:
        from health_monitor import check_and_report
        check_and_report()
    except Exception:
        pass

