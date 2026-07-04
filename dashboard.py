import numpy as np
from tax_logic import STCG_RATE, LTCG_RATE
from config import NEW_BUY_MIN_SCORE, EXIT_SCORE_THRESHOLD, EXIT_RS_SAFE_ZONE
import zipfile
import io
import plotly.io as pio
import plotly.graph_objects as go
import plotly.express as px
import os
import sys
import glob
import json
import yfinance as yf
import streamlit as st
from datetime import datetime, timedelta
import re
import pandas as pd
from db_manager import load_pipeline_stage
def read_data_smart(filepath_or_buffer, **kwargs):
    if not isinstance(filepath_or_buffer, str):
        return pd.read_csv(filepath_or_buffer, **kwargs)
    import os
    base_name = os.path.basename(filepath_or_buffer)
    table_name = os.path.splitext(base_name)[0]
    # Skip SQLite lookup for well-known non-pipeline files (index cache, etc.)
    _non_pipeline_prefixes = ("NIFTY_", "INDIA_VIX", "^NSEI", "^BSESN", "^CNXSC")
    if not table_name.startswith(_non_pipeline_prefixes):
        # Try SQLite
        df = load_pipeline_stage(table_name)
        if df is not None and not df.empty:
            if table_name == 'Portfolio_Correlation_Matrix' and 'Symbol' in df.columns:
                df.set_index('Symbol', inplace=True)
                df.index.name = None
            # Handle index_col argument if passed
            if 'index_col' in kwargs and kwargs['index_col'] == 0:
                if isinstance(df.index, pd.RangeIndex):
                    if df.index.name != list(df.columns)[0]:
                        df.set_index(df.columns[0], inplace=True)
            return df
    # Fallback to CSV
    if os.path.exists(filepath_or_buffer):
        return pd.read_csv(filepath_or_buffer, **kwargs)
    return pd.DataFrame()
def download_csv_button(df, filename, label="📥 Download CSV", key=None):
    if df is not None and not df.empty:
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(label=label, data=csv, file_name=filename, mime="text/csv", key=key)
def create_zip_of_folder(folder_path):
    """Zip output folder CSVs PLUS generated analysis exports."""
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        # 1. All CSV files from the output folder
        if os.path.exists(folder_path):
            for file_name in sorted(os.listdir(folder_path)):
                if file_name.endswith('.csv'):
                    file_path = os.path.join(folder_path, file_name)
                    zip_file.write(file_path, arcname=f"pipeline/{file_name}")
        # 2. VAM-B ranked stocks (from existing L1_VAM_B_Universe.csv)
        _vamb_path = os.path.join(folder_path, "L1_VAM_B_Universe.csv")
        if os.path.exists(_vamb_path):
            zip_file.write(_vamb_path, arcname="analysis/VAM-B_Ranked_Stocks.csv")
        # 3. VAM-GQ ranked stocks (generate from MAAC with quality gates)
        _maac_path_m = os.path.join(folder_path, "L7_MAAC_Allocations.csv")
        if os.path.exists(_maac_path_m):
            try:
                _df_gq = pd.read_csv(_maac_path_m)
                # Filter stocks (non-numeric symbols, eligible entries)
                _df_gq = _df_gq[~_df_gq["Symbol"].astype(str).str.match(r"^\d+$")]
                # Score columns to include
                _gq_cols = ["Symbol", "Sector", "Final_Composite_Score", "CIO_Verdict", "CIO_Score",
                            "ADX_14", "RS_vs_Nifty50", "Allocation_%", "Entry_Price",
                            "ROE", "Debt_to_Equity", "Delivery_Pct", "Market_Cap_Cr", "Cap_Category"]
                _gq_cols = [c for c in _gq_cols if c in _df_gq.columns]
                _df_gq = _df_gq[_gq_cols].sort_values("Final_Composite_Score", ascending=False) if "Final_Composite_Score" in _df_gq.columns else _df_gq[_gq_cols]
                _csv_gq = _df_gq.to_csv(index=False).encode('utf-8')
                zip_file.writestr("analysis/VAM-GQ_Ranked_Stocks.csv", _csv_gq)
            except:
                pass
        # 4. Master Analyzer table output (generate from MAAC data directly)
        try:
            _maac_path_m2 = os.path.join(folder_path, "L7_MAAC_Allocations.csv")
            if os.path.exists(_maac_path_m2):
                _ma_df2 = pd.read_csv(_maac_path_m2)
                _ma_df2 = _ma_df2[~_ma_df2["Symbol"].astype(str).str.match(r"^\d+$")].copy()
                if not _ma_df2.empty and "RS_vs_Nifty50" in _ma_df2.columns:
                    # RS Line = RS_vs_Nifty50 / 100, Live RS = RS_vs_Nifty50
                    _ma_df2["RS_LINE"] = (pd.to_numeric(_ma_df2["RS_vs_Nifty50"], errors="coerce") / 100.0).round(2)
                    _ma_df2["LIVE_RS"] = pd.to_numeric(_ma_df2["RS_vs_Nifty50"], errors="coerce").round(2)
                    # Rank by RS descending
                    _ma_df2["RS_RANK"] = _ma_df2["RS_LINE"].rank(ascending=False).astype(int)
                    _ma_df2 = _ma_df2.sort_values("RS_RANK")
                    # Price from Entry_Price
                    _ma_df2["PRICE"] = pd.to_numeric(_ma_df2.get("Entry_Price", pd.Series(0)), errors="coerce").round(1)
                    # ADX, OBV, CIO
                    _ma_df2["ADX"] = pd.to_numeric(_ma_df2.get("ADX_14", 0), errors="coerce").fillna(0).astype(int)
                    _ma_df2["OBV"] = _ma_df2["OBV_Rising"].apply(lambda x: "↑" if str(x).upper() == "TRUE" else "↓") if "OBV_Rising" in _ma_df2.columns else ""
                    _ma_df2["CIO"] = _ma_df2.get("CIO_Verdict", "—").fillna("—")
                    _ma_df2["SECTOR"] = _ma_df2.get("Sector", "")
                    # Output columns
                    _out_cols2 = ["RS_RANK", "Symbol", "SECTOR", "PRICE", "LIVE_RS", "RS_LINE",
                                  "ADX", "OBV", "CIO"]
                    _out_cols2 = [c for c in _out_cols2 if c in _ma_df2.columns]
                    _csv_ma2 = _ma_df2[_out_cols2].to_csv(index=False).encode('utf-8')
                    zip_file.writestr("analysis/Master_Analyzer_Table.csv", _csv_ma2)
        except:
            pass
        # 5. Satellite Ranking table (VAM-B + VAM-GQ merged ranking)
        try:
            import json as _json
            _sat_rows = []
            # VAM-B from L1_VAM_B_Universe.csv
            _vamb_path2 = os.path.join(folder_path, "L1_VAM_B_Universe.csv")
            if os.path.exists(_vamb_path2):
                _df_vamb2 = pd.read_csv(_vamb_path2)
                _df_vamb2 = _df_vamb2[~_df_vamb2["Symbol"].astype(str).str.match(r"^\d+$")].head(20)
                for _, _r in _df_vamb2.iterrows():
                    _sym = str(_r["Symbol"])
                    _vam = float(_r.get("VAM_Score", _r.get("Score", 0)))
                    _sat_rows.append({"symbol": _sym, "track": "VAM-B", "vam_score": _vam or 0})
            # VAM-GQ from MAAC with Factor_Details
            _maac_path_sat = os.path.join(folder_path, "L7_MAAC_Allocations.csv")
            if os.path.exists(_maac_path_sat):
                _df_gq2 = pd.read_csv(_maac_path_sat)
                _df_gq2 = _df_gq2[~_df_gq2["Symbol"].astype(str).str.match(r"^\d+$")]
                _df_gq2 = _df_gq2.sort_values("Final_Composite_Score", ascending=False).head(20)
                for _, _r in _df_gq2.iterrows():
                    _sym = str(_r["Symbol"])
                    _fs = float(_r.get("Final_Composite_Score", _r.get("Factor_Score", 0)) or 0)
                    _mom = _sec = _del = _gro = _pea = _fii = ""
                    try:
                        _fd = _json.loads(str(_r.get("Factor_Details", "{}")))
                        _mom = int(_fd.get("F3_MOMENTUM", {}).get("score", 0) or 0)
                        _sec = int((_fd.get("F1_SECTORAL_TREND", {}).get("score", 0) or 0) * 0.6 + (_fd.get("F2_THEMATIC_TREND", {}).get("score", 0) or 0) * 0.4)
                        _del = int(_fd.get("F6_DELIVERY_CONFIRMATION", {}).get("score", 0) or 0)
                        _gro = int(_fd.get("F4_GROWTH", {}).get("score", 0) or 0)
                        _pea = int(_fd.get("F7_PEAD", {}).get("score", 0) or 0)
                        _fii = int(_fd.get("F8_FII_DII_CONVICTION", {}).get("score", 0) or 0)
                    except:
                        pass
                    _found = False
                    for _s in _sat_rows:
                        if _s["symbol"] == _sym:
                            _s["track"] = "VAM-B + VAM-GQ"
                            _s["fs"] = _fs
                            _s["mom"] = _mom; _s["sec"] = _sec; _s["del"] = _del
                            _s["gro"] = _gro; _s["pea"] = _pea; _s["fii"] = _fii
                            _found = True
                            break
                    if not _found:
                        _sat_rows.append({"symbol": _sym, "track": "VAM-GQ", "fs": _fs,
                            "mom": _mom, "sec": _sec, "del": _del, "gro": _gro, "pea": _pea, "fii": _fii})
            # Score normalization and ranking
            if _sat_rows:
                _scores = [s.get("vam_score", s.get("fs", 0)) for s in _sat_rows]
                _min_s = min(_scores); _max_s = max(_scores); _range_s = max(_max_s - _min_s, 0.01)
                _sat_out = []
                for _idx, _s in enumerate(_sat_rows, 1):
                    _us = _s.get("vam_score", _s.get("fs", 0))
                    if _s["track"] == "VAM-B + VAM-GQ":
                        _us = _s.get("vam_score", 0) * 0.5 + _s.get("fs", 0) * 2.0
                    _comp = max(0, min(100, (_us - _min_s) / _range_s * 100))
                    _sv = round(_s.get("vam_score", _s.get("fs", 0)), 2) if _s["track"] == "VAM-B" else round(_s.get("fs", 0), 1)
                    _sat_out.append({
                        "Rank": _idx, "Symbol": _s["symbol"], "Track": _s["track"],
                        "Score": _sv, "Mom": _s.get("mom", ""), "Sec+Thm": _s.get("sec", ""),
                        "Delivery": _s.get("del", ""), "Growth": _s.get("gro", ""),
                        "PEAD": _s.get("pea", ""), "FII/DII": _s.get("fii", ""),
                        "Composite": round(_comp, 1),
                    })
                _csv_sat = pd.DataFrame(_sat_out).to_csv(index=False).encode('utf-8')
                zip_file.writestr("analysis/Satellite_Ranking.csv", _csv_sat)
        except:
            pass
    return zip_buffer.getvalue()
@st.cache_data(ttl=3600)
def get_global_mf_name_map():
    mf_name_map = {}
    try:
        from config import CORE_ETF_UNIVERSE
        for k, v in CORE_ETF_UNIVERSE.items():
            mf_name_map[str(k)] = str(v)
        import mf_fetcher
        master_mfs = mf_fetcher.get_master_mf_list()
        if master_mfs:
            for mf in master_mfs:
                mf_name_map[str(mf.get("schemeCode"))] = str(mf.get("schemeName", ""))
        # Fallback: scan ALL L1_Core_Universe.csv files for codes not yet mapped
        _core_univ_dirs = []
        if os.path.exists(base_output_dir):
            for _d in sorted(os.listdir(base_output_dir), reverse=True):
                _p = os.path.join(base_output_dir, _d, "L1_Core_Universe.csv")
                if os.path.exists(_p):
                    _core_univ_dirs.append(_p)
        for _csv_path in _core_univ_dirs[:3]:  # check last 3 dates
            try:
                _df_cu = pd.read_csv(_csv_path)
                if "Symbol" in _df_cu.columns and "Name" in _df_cu.columns:
                    for _, _r in _df_cu.iterrows():
                        _sym = str(_r["Symbol"])
                        if _sym not in mf_name_map:
                            mf_name_map[_sym] = str(_r["Name"])
            except:
                pass
        # Index/ETF ticker map (from core_universe_processor)
        _etf_map = {
            "nifty 50": "NIFTYBEES.NS", "nifty next 50": "JUNIORBEES.NS",
            "nifty 100": "NIFTY100.NS", "nifty midcap 150": "MID150BEES.NS",
            "nifty smallcap 250": "SMALLCAP.NS", "nifty microcap 250": "NIFTYMICROCAP250.NS",
            "nifty bank": "BANKBEES.NS", "nifty auto": "AUTOBEES.NS",
            "nifty it": "ITBEES.NS", "nifty pharma": "PHARMABEES.NS",
            "nifty fmcg": "CONSUMBEES.NS", "nifty infra": "INFRABEES.NS",
            "nifty psu bank": "PSUBNKBEES.NS", "nifty private bank": "PVTBANIETF.NS",
            "nifty healthcare": "HEALTHY.NS", "nifty realty": "NETFREALTY.NS",
            "nifty100 low vol 30": "LOWVOLIETF.NS", "nifty alpha 50": "ALPHABEES.NS",
            "nifty200 momentum 30": "MOM50.NS", "nifty midcap150 momentum 50": "MOMOM100.NS",
            "nifty cpse": "CPSEETF.NS", "nifty commodities": "COMMOIETF.NS",
            "nifty gold": "GOLDBEES.NS", "nifty silver": "SILVERBEES.NS",
        }
        for _name, _ticker in _etf_map.items():
            mf_name_map[_ticker] = _name.title()
    except Exception:
        pass
    return mf_name_map
# Fix for plotly json serialization with buggy local orjson installation
pio.json.config.default_engine = "json"
# Setup global premium dark theme for Plotly
theme_layout = go.Layout(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(13, 21, 39, 0.35)",
    font=dict(family="'Inter', sans-serif", color="#cbd5e1"),
    title=dict(font=dict(family="'Outfit', sans-serif", size=20, color="#ffffff")),
    xaxis=dict(
        gridcolor="rgba(255, 255, 255, 0.05)",
        zerolinecolor="rgba(255, 255, 255, 0.08)",
        tickfont=dict(family="'Inter', sans-serif", size=11, color="#94a3b8")
    ),
    yaxis=dict(
        gridcolor="rgba(255, 255, 255, 0.05)",
        zerolinecolor="rgba(255, 255, 255, 0.08)",
        tickfont=dict(family="'Inter', sans-serif", size=11, color="#94a3b8")
    ),
    legend=dict(
        font=dict(family="'Inter', sans-serif", size=11, color="#cbd5e1"),
        bgcolor="rgba(13, 21, 39, 0.7)",
        bordercolor="rgba(255, 255, 255, 0.05)",
        borderwidth=1
    )
)
pio.templates["premium_dark"] = pio.templates["plotly_dark"]
pio.templates["premium_dark"].layout.update(theme_layout)
pio.templates.default = "premium_dark"
# Set page config to premium wide layout
st.set_page_config(
    page_title="Trend Alpha 4.0 | Institutional Portfolio OS Terminal",
    page_icon="data:image/png;base64,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",
    layout="wide",
    initial_sidebar_state="expanded"
)
# Custom Glassmorphic CSS Injection with Outfit/Inter typography
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600;700&display=swap');
    /* Global Overrides */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    /* Custom Scrollbar for premium dark mode */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: rgba(15, 23, 42, 0.3);
    }
    ::-webkit-scrollbar-thumb {
        background: rgba(255, 255, 255, 0.08);
        border-radius: 4px;
        border: 1px solid transparent;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(34, 211, 238, 0.25);
    }
    /* Background & Gradient */
    .stApp {
        background: radial-gradient(circle at 15% 25%, #0A0F1E 0%, #121828 80%, #000000 100%);
        color: #e2e8f0;
    }
    /* Sidebar Styling for dark mode alignment */
    [data-testid="stSidebar"] {
        background-color: #060a13 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05) !important;
    }
    [data-testid="stSidebarNav"] {
        background-color: transparent !important;
    }
    /* Premium alert banner overrides */
    [data-testid="stAlert"] {
        background: rgba(15, 23, 42, 0.45) !important;
        border-radius: 16px !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3) !important;
    }
    /* Glassmorphic Cards 2.0 with Radial Gradient & Glow */
    .glass-card {
        background: radial-gradient(circle at top left, rgba(255, 255, 255, 0.03) 0%, rgba(255, 255, 255, 0.01) 100%) !important;
        backdrop-filter: blur(16px) !important;
        -webkit-backdrop-filter: blur(16px) !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        border-radius: 20px !important;
        padding: 22px;
        margin-bottom: 20px;
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.03) !important;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .glass-card:hover {
        transform: translateY(-3px);
        border-color: rgba(34, 211, 238, 0.3) !important;
        box-shadow: 0 20px 45px -10px rgba(34, 211, 238, 0.2), inset 0 1px 0 rgba(255, 255, 255, 0.05) !important;
    }
    .kpi-card {
        height: 195px !important;
        margin-bottom: 10px !important;
        display: flex;
        flex-direction: column;
        justify-content: flex-start;
    }
    /* Harmonized Neon Pill Badges */
    .badge-buy {
        background: rgba(34, 211, 238, 0.08) !important;
        color: #22d3ee !important;
        border: 1px solid rgba(34, 211, 238, 0.25) !important;
        box-shadow: 0 0 10px rgba(34, 211, 238, 0.05) !important;
        padding: 4px 10px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.82rem;
    }
    .badge-hold {
        background: rgba(251, 191, 36, 0.08) !important;
        color: #fbbf24 !important;
        border: 1px solid rgba(251, 191, 36, 0.25) !important;
        box-shadow: 0 0 10px rgba(251, 191, 36, 0.05) !important;
        padding: 4px 10px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.82rem;
    }
    .badge-avoid {
        background: rgba(239, 68, 68, 0.08) !important;
        color: #f87171 !important;
        border: 1px solid rgba(239, 68, 68, 0.25) !important;
        box-shadow: 0 0 10px rgba(239, 68, 68, 0.05) !important;
        padding: 4px 10px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.82rem;
    }
    .badge-normal {
        background: rgba(16, 185, 129, 0.08) !important;
        color: #34d399 !important;
        border: 1px solid rgba(16, 185, 129, 0.25) !important;
        box-shadow: 0 0 10px rgba(16, 185, 129, 0.05) !important;
        padding: 4px 10px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.82rem;
    }
    .stProgress > div > div > div > div {
        background-color: #0ea5e9;
    }
    /* Premium Unified Table Designs (Screening, Orders, Elimination, Ledger, Custom) */
    .premium-table, .screening-table, .orders-table, .elimination-table, .unified-ledger-table {
        display: table !important;
        width: 100% !important;
        border-collapse: collapse !important;
        background: rgba(13, 21, 39, 0.25) !important;
        border-radius: 12px !important;
        overflow: hidden !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        font-size: 0.88rem !important;
        table-layout: fixed !important;
    }
    .premium-table thead, .screening-table thead, .orders-table thead, .elimination-table thead, .unified-ledger-table thead {
        display: table-header-group !important;
    }
    .premium-table tbody, .screening-table tbody, .orders-table tbody, .elimination-table tbody, .unified-ledger-table tbody {
        display: table-row-group !important;
    }
    .premium-table tr, .screening-table tr, .orders-table tr, .elimination-table tr, .unified-ledger-table tr {
        display: table-row !important;
        border-bottom: 1px solid rgba(255, 255, 255, 0.04) !important;
        transition: background 0.2s ease !important;
    }
    .premium-table tr:hover, .screening-table tr:hover, .orders-table tr:hover, .elimination-table tr:hover, .unified-ledger-table tr:hover {
        background: rgba(255, 255, 255, 0.02) !important;
    }
    .premium-table th, .screening-table th, .orders-table th, .elimination-table th, .unified-ledger-table th,
    .premium-table td, .screening-table td, .orders-table td, .elimination-table td, .unified-ledger-table td {
        display: table-cell !important;
        padding: 12px 10px !important;
        box-sizing: border-box !important;
        vertical-align: middle !important;
    }
    .premium-table th, .screening-table th, .orders-table th, .elimination-table th, .unified-ledger-table th {
        background: rgba(15, 23, 42, 0.6) !important;
        color: #e2e8f0 !important;
        font-weight: 600 !important;
        font-family: 'Outfit', sans-serif !important;
        border-bottom: 1.5px solid rgba(255, 255, 255, 0.08) !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
    }
    /* Strict column widths and text alignment */
    .col-rank { width: 4% !important; text-align: left !important; }
    .col-symbol { width: 12% !important; text-align: left !important; }
    .col-verdict { width: 10% !important; text-align: left !important; }
    .col-tracks { width: 7% !important; text-align: center !important; }
    .col-score { width: 7% !important; text-align: center !important; }
    .col-natr { width: 12% !important; text-align: center !important; }
    .col-adx { width: 7% !important; text-align: center !important; }
    .col-obv { width: 10% !important; text-align: center !important; }
    .col-rs { width: 8% !important; text-align: center !important; }
    .col-alpha { width: 9% !important; text-align: center !important; }
    .col-deliv { width: 7% !important; text-align: right !important; }
    .col-alloc { width: 7% !important; text-align: right !important; }
    .col-order-sym { width: 15% !important; text-align: left !important; }
    .col-order-act { width: 15% !important; text-align: center !important; }
    .col-order-qty { width: 15% !important; text-align: right !important; }
    .col-order-reason { width: 55% !important; text-align: left !important; }
    .col-el-rank { width: 5% !important; text-align: center !important; }
    .col-el-sym { width: 10% !important; text-align: left !important; }
    .col-el-cap { width: 9% !important; text-align: center !important; }
    .col-el-status { width: 9% !important; text-align: center !important; }
    .col-el-reason { width: 27% !important; text-align: left !important; }
    .col-el-score { width: 7% !important; text-align: center !important; }
    .col-el-mcap { width: 9% !important; text-align: right !important; }
    .col-el-de { width: 7% !important; text-align: right !important; }
    .col-el-roe { width: 7% !important; text-align: right !important; }
    .col-el-adx { width: 5% !important; text-align: center !important; }
    .col-el-deliv { width: 5% !important; text-align: right !important; }
    /* Modern Pill Navigation for Streamlit Tabs */
    button[data-baseweb="tab"] {
        color: #94a3b8 !important;
        font-family: 'Outfit', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        padding: 12px 24px !important;
        border: 1px solid transparent !important;
        border-radius: 99px !important;
        background: rgba(255, 255, 255, 0.03) !important;
        margin-right: 12px !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    button[data-baseweb="tab"]:hover {
        color: #818cf8 !important;
        background: rgba(255, 255, 255, 0.07) !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #ffffff !important;
        background: linear-gradient(135deg, rgba(129, 140, 248, 0.2) 0%, rgba(192, 132, 252, 0.2) 100%) !important;
        border-color: rgba(129, 140, 248, 0.3) !important;
        box-shadow: 0 0 15px rgba(129, 140, 248, 0.15) !important;
        font-weight: 700 !important;
    }
    /* Style standard Streamlit Metric components as cards */
    [data-testid="stMetric"] {
        background: radial-gradient(circle at top left, rgba(255, 255, 255, 0.03) 0%, rgba(255, 255, 255, 0.01) 100%) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        border-radius: 16px !important;
        padding: 16px 20px !important;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.03) !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    [data-testid="stMetric"]:hover {
        transform: translateY(-2px) !important;
        border-color: rgba(34, 211, 238, 0.3) !important;
        box-shadow: 0 15px 40px -5px rgba(34, 211, 238, 0.15), inset 0 1px 0 rgba(255, 255, 255, 0.05) !important;
    }
    [data-testid="stMetricValue"] {
        color: #f1f5f9 !important;
        font-family: 'Outfit', sans-serif !important;
        font-weight: 700 !important;
        font-size: 1.8rem !important;
        margin-top: 4px !important;
    }
    [data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
        font-family: 'Outfit', sans-serif !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        font-size: 0.78rem !important;
        letter-spacing: 0.8px !important;
    }
    [data-testid="stMetricDelta"] {
        font-family: monospace !important;
        font-weight: 600 !important;
    }
    .pulse-dot {
        display: inline-block;
        width: 10px;
        height: 10px;
        background-color: #10b981;
        border-radius: 50%;
        box-shadow: 0 0 8px #10b981;
        animation: pulse-glow 2s infinite alternate ease-in-out;
    }
    @keyframes pulse-glow {
        0% { transform: scale(0.85); box-shadow: 0 0 4px rgba(16, 185, 129, 0.4); opacity: 0.7; }
        100% { transform: scale(1.15); box-shadow: 0 0 12px rgba(16, 185, 129, 1); opacity: 1; }
    }
</style>
""", unsafe_allow_html=True)
# Main Title & Hero Banner
st.markdown("""
<div style="background: radial-gradient(circle at 0% 0%, #0d1527 0%, #070a13 70%, #020306 100%);
     padding: 35px 45px;
     border-radius: 24px;
     margin-bottom: 30px;
     border: 1px solid rgba(34, 211, 238, 0.15);
     box-shadow: 0 20px 50px rgba(0, 0, 0, 0.8), inset 0 1px 0 rgba(255, 255, 255, 0.05), 0 0 30px rgba(34, 211, 238, 0.03);
     display: flex;
     align-items: center;
     justify-content: space-between;
     gap: 30px;
     flex-wrap: wrap;">
    <div style="display: flex; align-items: center; gap: 30px; flex-wrap: wrap;">
        <!-- Glowing container for logo -->
        <div style="position: relative; display: flex; align-items: center; justify-content: center;
                    width: 110px; height: 110px; border-radius: 24px;
                    background: linear-gradient(135deg, rgba(251, 191, 36, 0.12) 0%, rgba(34, 211, 238, 0.12) 100%);
                    border: 1px solid rgba(255, 255, 255, 0.12);
                    box-shadow: 0 0 25px rgba(34, 211, 238, 0.15), 0 0 40px rgba(34, 211, 238, 0.08) inset;">
            <img style="height: 92px; width: 92px; border-radius: 20px; object-fit: contain;" src="data:image/png;base64,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" />
        </div>
        <div>
            <!-- Main Title with logo matching gradient -->
            <h1 style="margin: 0; font-size: 3.6rem; font-weight: 800; font-family: 'Outfit', 'Inter', sans-serif; letter-spacing: -0.03em; line-height: 1.1;">
                <span style="background: linear-gradient(135deg, #FBBF24 0%, #F59E0B 40%, #22D3EE 100%);
                             -webkit-background-clip: text;
                             -webkit-text-fill-color: transparent;
                             filter: drop-shadow(0 2px 10px rgba(34, 211, 238, 0.25));">
                    Trend Alpha 4.0
                </span>
            </h1>
            <!-- Subtitle -->
            <p style="margin: 10px 0 0 0; color: #cbd5e1; font-size: 1.35rem; font-weight: 500; font-family: 'Outfit', 'Inter', sans-serif; letter-spacing: -0.01em; opacity: 0.9;">
                Multi-Role Institutional Asset Sizing & Quantitative Control Panel
            </p>
        </div>
    </div>
    <!-- Premium Status Pill -->
    <div style="display: flex; align-items: center; gap: 12px; background: rgba(16, 185, 129, 0.06);
                border: 1px solid rgba(16, 185, 129, 0.25); padding: 12px 24px; border-radius: 99px;
                box-shadow: 0 0 25px rgba(16, 185, 129, 0.05); align-self: center;">
        <span class="pulse-dot"></span>
        <span style="color: #10b981; font-family: 'Outfit', 'Inter', sans-serif; font-weight: 700; font-size: 0.95rem; letter-spacing: 0.08em; text-transform: uppercase;">
            SYNCED & SECURE
        </span>
    </div>
</div>
""", unsafe_allow_html=True)
# Sidebar configurations
st.sidebar.markdown("""
<div style="text-align: center; margin-bottom: 15px;">
    <h3 style="color: #818cf8; margin-bottom: 0;">⚙️ Operations Desk</h3>
</div>
""", unsafe_allow_html=True)
def _resolve_pipeline_file(filename):
    """Find the requested pipeline file, falling back to previous dates if the selected date is incomplete."""
    _d = selected_date
    for _attempt in [_d] + [(d[1] if isinstance(d, tuple) else d) for d in available_dates if (d[1] if isinstance(d, tuple) else d) < _d]:
        _p = os.path.join(base_output_dir, _attempt, filename)
        if os.path.exists(_p):
            return _p, _attempt
    return None, None
base_output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
available_dates = []
if os.path.exists(base_output_dir):
    for item in sorted(os.listdir(base_output_dir), reverse=True):
        item_path = os.path.join(base_output_dir, item)
        if os.path.isdir(item_path) and not item.startswith("--"):
            # Check if date is a non-trading day (weekend)
            try:
                _dt = pd.to_datetime(item)
                _is_weekend = _dt.weekday() >= 5  # Saturday=5, Sunday=6
            except:
                _is_weekend = False
            # Check for minimum complete pipeline files
            _has_maac = os.path.exists(os.path.join(item_path, "L7_MAAC_Allocations.csv"))
            _has_universe = os.path.exists(os.path.join(item_path, "L1_Core_Universe.csv"))
            _has_corr = os.path.exists(os.path.join(item_path, "Portfolio_Correlation_Matrix.csv"))
            _has_state = os.path.exists(os.path.join(item_path, "state_3_0.json"))
            if _has_maac:
                _prefix = "🗓️ " if _is_weekend else ""
                if _has_universe and _has_corr and _has_state:
                    available_dates.append((_prefix + item, item) if _prefix else item)
                else:
                    available_dates.append(("⚠️ " + item, item))
# No reverse — already sorted newest-first from the sorted(..., reverse=True) loop
if not available_dates:
    available_dates = [("2026-06-05 (fallback)", "2026-06-05")]
# Default to the first complete date (not marked with ⚠️)
_default_idx = 0
for _i, _d in enumerate(available_dates):
    _label = _d[0] if isinstance(_d, tuple) else _d
    if not _label.startswith("⚠️"):
        _default_idx = _i
        break
selected_date = st.sidebar.selectbox("Synchronization Date", available_dates, index=_default_idx, format_func=lambda x: x[0] if isinstance(x, tuple) else x)
if isinstance(selected_date, tuple):
    selected_date = selected_date[1]
# Define paths and load state immediately to prevent NameError in sidebar widgets
OUTPUT_DIR = os.path.join(base_output_dir, selected_date)
maac_path = os.path.join(OUTPUT_DIR, "L7_MAAC_Allocations.csv")
state_path = os.path.join(OUTPUT_DIR, "state_3_0.json")
corr_path = os.path.join(OUTPUT_DIR, "Portfolio_Correlation_Matrix.csv")
orders_path = os.path.join(OUTPUT_DIR, "Execution_Orders.csv")
def load_df_safe(path):
    if os.path.exists(path):
        try:
            df = read_data_smart(path)
            if not df.empty and "Symbol" in df.columns:
                veto_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Veto_add_remove.csv")
                if os.path.exists(veto_file):
                    try:
                        v_df = read_data_smart(veto_file)
                        v_last = v_df.drop_duplicates(subset=['Symbol'], keep='last')
                        removed = v_last[v_last['Action'] == 'VETO_REMOVE']['Symbol'].tolist()
                        df = df[~df['Symbol'].isin(removed)]
                    except:
                        pass
            return df
        except Exception as e:
            st.warning(f"Error loading {os.path.basename(path)}: {e}")
    return pd.DataFrame()
df_maac = load_df_safe(maac_path)
df_orders = load_df_safe(orders_path)
if df_maac.empty:
    st.error("MAAC Allocations dataset is unavailable. Run the integrated pipeline (`main.py`) to generate data.")
    st.stop()
# ── Load state backup if exists ──
state_loaded = False
state_data = {}
if os.path.exists(state_path):
    try:
        with open(state_path, "r") as f:
            state_data = json.load(f)
            state_loaded = True
    except Exception:
        pass
# Setup default values if state load fails
regime = state_data.get("regime", {}) if state_loaded else {}
blueprint = state_data.get("blueprint", {}) if state_loaded else {}
analytics = state_data.get("analytics", {}) if state_loaded else {}
drawdown = state_data.get("drawdown", {}) if state_loaded else {}
trends = state_data.get("trends", {}) if state_loaded else {}
_synced_global = state_loaded
st.sidebar.markdown("---")
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Portfolio Performance")
src = analytics.get("metrics_source", "no_ledger")
trades = analytics.get("total_trades", 0)
unrealized = analytics.get("unrealized_pnl", 0.0)
unrealized_pct = analytics.get("unrealized_pnl_pct", 0.0)
unr_c = "#34d399" if unrealized > 0 else ("#ef4444" if unrealized < 0 else "#94a3b8")
if src != "no_ledger" and trades > 0:
    sharpe = analytics.get("sharpe_ratio", 0)
    cagr = analytics.get("CAGR_%", 0)
    win_rate = analytics.get("win_rate_%", 0)
    profit_factor = analytics.get("profit_factor", 0)
    max_dd = analytics.get("max_drawdown_%", 0)
    avg_hold = analytics.get("avg_hold_days", 0)
    sharpe_c = "#34d399" if sharpe >= 1.0 else ("#fbbf24" if sharpe >= 0.5 else "#ef4444")
    cagr_c = "#34d399" if cagr >= 15 else ("#fbbf24" if cagr >= 5 else "#ef4444")
    win_c = "#34d399" if win_rate >= 50 else ("#fbbf24" if win_rate >= 35 else "#ef4444")
    html = f"<div style='background:radial-gradient(circle at top left, rgba(255,255,255,0.04) 0%, rgba(255,255,255,0.01) 100%); border:1px solid rgba(255,255,255,0.08); border-radius:14px; padding:14px 16px; box-shadow: 0 10px 30px rgba(0,0,0,0.3);'>"
    html += f"<div style='display:flex;justify-content:space-between;margin-bottom:8px;padding-bottom:8px;border-bottom:1px dashed rgba(255,255,255,0.1);'><span style='color:#cbd5e1;font-size:0.85rem;font-weight:600;'>Unrealized P&L</span><span style='color:{unr_c};font-weight:800;font-size:0.9rem;'>₹{unrealized:,.0f} ({unrealized_pct:+.1f}%)</span></div>"
    html += f"<div style='display:flex;justify-content:space-between;margin-bottom:4px;'><span style='color:#94a3b8;font-size:0.75rem;'>Sharpe</span><span style='color:{sharpe_c};font-weight:700;'>{sharpe:.2f}</span></div>"
    html += f"<div style='display:flex;justify-content:space-between;margin-bottom:4px;'><span style='color:#94a3b8;font-size:0.75rem;'>CAGR</span><span style='color:{cagr_c};font-weight:700;'>{cagr:.1f}%</span></div>"
    html += f"<div style='display:flex;justify-content:space-between;margin-bottom:4px;'><span style='color:#94a3b8;font-size:0.75rem;'>Win Rate</span><span style='color:{win_c};font-weight:700;'>{win_rate:.1f}%</span></div>"
    html += f"<div style='display:flex;justify-content:space-between;margin-bottom:4px;'><span style='color:#94a3b8;font-size:0.75rem;'>Profit Factor</span><span style='color:#a78bfa;font-weight:700;'>{ profit_factor:.2f}</span></div>"
    html += f"<div style='display:flex;justify-content:space-between;margin-bottom:4px;'><span style='color:#94a3b8;font-size:0.75rem;'>Max DD</span><span style='color:#f87171;font-weight:700;'>{ max_dd:.1f}%</span></div>"
    html += f"<div style='display:flex;justify-content:space-between;'><span style='color:#94a3b8;font-size:0.75rem;'>Avg Hold</span><span style='color:#94a3b8;font-weight:700;'>{ avg_hold:.0f}d</span></div>"
    html += f"<div style='margin-top:6px;padding-top:6px;border-top:1px solid rgba(255,255,255,0.06);'><span style='color:#64748b;font-size:0.65rem;'>{trades} closed trades</span></div></div>"
    st.sidebar.markdown(html, unsafe_allow_html=True)
else:
    html = f"<div style='background:radial-gradient(circle at top left, rgba(255,255,255,0.04) 0%, rgba(255,255,255,0.01) 100%); border:1px solid rgba(255,255,255,0.08); border-radius:14px; padding:14px 16px; box-shadow: 0 10px 30px rgba(0,0,0,0.3);'>"
    html += f"<div style='display:flex;justify-content:space-between;margin-bottom:4px;'><span style='color:#cbd5e1;font-size:0.85rem;font-weight:600;'>Unrealized P&L (MTM)</span><span style='color:{unr_c};font-weight:800;font-size:0.9rem;'>₹{unrealized:,.0f} ({unrealized_pct:+.1f}%)</span></div>"
    html += f"<div style='margin-top:6px;padding-top:6px;border-top:1px solid rgba(255,255,255,0.06);'><span style='color:#64748b;font-size:0.75rem;'>📭 0 closed trades yet (Realized stats offline)</span></div></div>"
    st.sidebar.markdown(html, unsafe_allow_html=True)
st.sidebar.subheader("Portfolio Parameters")
portfolio_capital = st.sidebar.number_input(
    "Total Portfolio Capital (₹)",
    min_value=100000.0,
    value=10000000.0,
    step=100000.0,
    format="%.2f"
)
# ── Helper UI Functions ─────────────────────────────────────────────────────
def stat_box(label, value, sub="", color="#818cf8", accent=None):
    accent_style = f"border-left:3px solid {accent};" if accent else ""
    return f"""
    <div style="background:rgba(15,23,42,0.6);border:1px solid rgba(255,255,255,0.06);
         border-radius:10px;padding:16px;{accent_style}text-align:center;">
        <div style="font-size:0.72rem;color:#64748b;text-transform:uppercase;letter-spacing:0.05em;
             font-weight:600;margin-bottom:6px;">{label}</div>
        <div style="font-family:'Outfit';font-weight:800;font-size:1.6rem;color:{color};
             margin-bottom:4px;">{value}</div>
        <div style="font-size:0.72rem;color:#475569;">{sub}</div>
    </div>"""
def sec_title(icon, text):
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:8px;margin:16px 0 12px;">
        <span style="font-size:1.1rem;">{icon}</span>
        <span style="font-family:'Outfit';font-weight:700;color:#818cf8;font-size:1rem;">{text}</span>
    </div>""", unsafe_allow_html=True)
# ── SHARED FUNCTIONS (for Hermes merged tabs) ──
_HERMES_OUTPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
@st.cache_data(ttl=120)
def _h_get_portfolio():
    all_stocks = set()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, "output")
    def add_from_csv(filename, top_n=None):
        path = os.path.join(output_dir, filename)
        if os.path.exists(path):
            try:
                df = read_data_smart(path)
                if "Symbol" in df.columns:
                    if top_n and "Score" in df.columns:
                        df = df.sort_values(by="Score", ascending=False).head(top_n)
                    for s in df["Symbol"]:
                        all_stocks.add(str(s).strip())
            except:
                pass
    # Add from universes — VAM-GQ top 20 by Factor_Score, VAM-B from L1
    add_from_csv("L1_VAM_B_Universe.csv", top_n=20)
    # VAM-GQ: take top 20 ranked stocks (not just Entry_Eligible) for unified view
    l6_path = os.path.join(output_dir, "L6_Trade_Allocations.csv")
    if os.path.exists(l6_path):
        try:
            l6_df = read_data_smart(l6_path)
            if "Symbol" in l6_df.columns:
                if "Factor_Score" in l6_df.columns:
                    l6_df = l6_df.sort_values("Factor_Score", ascending=False)
                # Take top 20 by rank, skip rejected (score=0) for cleaner view
                l6_top = l6_df[l6_df["Factor_Score"] > 0].head(20)
                for s in l6_top["Symbol"]:
                    all_stocks.add(str(s).strip())
        except:
            pass
    add_from_csv("L1_Core_Universe.csv")
    # Add from active portfolio state
    state_file = os.path.join(base_dir, "portfolio_state.json")
    if os.path.exists(state_file):
        try:
            import json
            with open(state_file, 'r') as f:
                state = json.load(f)
                for r in state.get("active_ledger", []):
                    if r.get("status") in ("HOLD", "NEW BUY"):
                        all_stocks.add(str(r.get("Symbol")).strip())
        except:
            pass
    # --- Filter out manual veto removals ---
    veto_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Veto_add_remove.csv")
    if os.path.exists(veto_file):
        try:
            v_df = read_data_smart(veto_file)
            v_last = v_df.drop_duplicates(subset=['Symbol'], keep='last')
            removed = v_last[v_last['Action'] == 'VETO_REMOVE']['Symbol'].tolist()
            for r in removed:
                if r in all_stocks:
                    all_stocks.remove(r)
        except:
            pass
    all_stocks = {s for s in all_stocks if not __import__("re").match(r"^(NIFTY_|STRATEGY_|INDIA_VIX|MCX_)", str(s), __import__("re").IGNORECASE)}
    return sorted(list(all_stocks))
@st.cache_data(ttl=300)
def _h_get_rs_df(_refresh_key=0):
    """Fetch RS data for portfolio stocks.
    Priority:
    1. Pipeline L7_MAAC_Allocations.csv (RS_vs_Nifty50 / 100 = raw RS)
    2. Pipeline L6_Trade_Allocations.csv (same source)
    3. Fallback: live yfinance download
    _refresh_key forces cache bust when changed.
    """
    # ── Priority 1: Read RS from pipeline output ──
    _base_out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    _pipeline_rs = {}
    for _try_file in ["L7_MAAC_Allocations.csv", "L6_Trade_Allocations.csv"]:
        _paths = sorted(
            glob.glob(os.path.join(_base_out, "*", _try_file)),
            reverse=True
        )
        if _paths:
            try:
                _pdf = read_data_smart(_paths[0])
                if not _pdf.empty and "RS_vs_Nifty50" in _pdf.columns and "Symbol" in _pdf.columns:
                    for _, _pr in _pdf.iterrows():
                        _s = str(_pr["Symbol"]).strip()
                        if not _s or _s.isdigit() or any(c.islower() for c in _s if c.isalpha()):
                            continue
                        _rs_val = pd.to_numeric(_pr.get("RS_vs_Nifty50", 0), errors="coerce")
                        if pd.notna(_rs_val) and _rs_val != 0.0:
                            _price = float(_pr.get("Entry_Price", _pr.get("Close", 0)) or 0)
                            _stock_ret = float(_pr.get("Return_63d", 0) or 0)
                            # Pipeline RS_vs_Nifty50 is ×100 scale; /100 to get raw RS
                            _pipeline_rs[_s] = {
                                "RS": round(_rs_val / 100.0, 4),
                                "Stock_123d%": round(_stock_ret * 100, 1) if _stock_ret else 0.0,
                                "Price": round(_price, 1)
                            }
            except Exception:
                pass
        if _pipeline_rs:
            break  # found usable data
    if _pipeline_rs:
        _rows = []
        # Compute nifty 123d% once from cache for completeness
        _nifty_ret = 0.0
        try:
            from cache_manager import get_historical_data
            _nf = get_historical_data("^NSEI")
            if _nf is not None and not _nf.empty and len(_nf) >= 124:
                _nc = _nf["Close"]
                _nifty_ret = round((float(_nc.iloc[-1]) / float(_nc.iloc[-124]) - 1.0) * 100, 1)
        except Exception:
            pass
        for _sym, _d in _pipeline_rs.items():
            _rows.append((
                _sym,
                _d["RS"],
                _d["Stock_123d%"],
                _nifty_ret,
                _d["Price"]
            ))
        return pd.DataFrame(_rows, columns=["Stock", "RS", "Stock_123d%", "Nifty_123d%", "Price"])
    # ── Priority 2: Fallback to live yfinance download ──
    portfolio = _h_get_portfolio()
    if not portfolio:
        return pd.DataFrame()
    # Robust Nifty download
    nifty_c = pd.Series(dtype=float)
    for _nifty_attempt in range(2):
        try:
            _na = _nifty_attempt == 0
            nifty = yf.download("^NSEI", period="1y", progress=False, auto_adjust=_na)
            if nifty is not None and isinstance(nifty, pd.DataFrame) and not nifty.empty:
                if isinstance(nifty.columns, pd.MultiIndex):
                    nifty_c = nifty.xs("Close", axis=1, level=0).squeeze()
                elif "Close" in nifty.columns:
                    nifty_c = nifty["Close"].squeeze()
                elif "Adj Close" in nifty.columns:
                    nifty_c = nifty["Adj Close"].squeeze()
                nifty_c = pd.Series(nifty_c).dropna()
                if len(nifty_c) >= 124:
                    break
        except:
            pass
    results = []
    tickers = [sym + ".NS" for sym in portfolio]
    all_data = {}
    if len(tickers) > 0:
        _chunk_size = 5
        for _chunk_start in range(0, len(tickers), _chunk_size):
            _chunk = tickers[_chunk_start:_chunk_start + _chunk_size]
            try:
                _chunk_data = yf.download(_chunk, period="1y", progress=False, auto_adjust=True, group_by="ticker")
                if _chunk_data is not None and isinstance(_chunk_data, pd.DataFrame) and not _chunk_data.empty:
                    if len(_chunk) == 1:
                        all_data[_chunk[0]] = _chunk_data
                    elif hasattr(_chunk_data.columns, "levels"):
                        for _t in _chunk:
                            if _t in _chunk_data.columns.levels[0]:
                                all_data[_t] = _chunk_data[_t]
            except:
                pass
    for sym in portfolio:
        if not sym or not isinstance(sym, str):
            continue
        if sym.isdigit():
            results.append((sym, None, None, None, None))
            continue
        if len(sym) > 15 or ' ' in sym or any(c.islower() for c in sym if c.isalpha()):
            results.append((sym, None, None, None, None))
            continue
        try:
            ticker_sym = sym + ".NS"
            d = all_data.get(ticker_sym, None)
            if d is None or (isinstance(d, pd.DataFrame) and d.empty):
                for _attempt in range(2):
                    try:
                        d = yf.download(ticker_sym, period="1y", progress=False, auto_adjust=True)
                        if d is not None and isinstance(d, pd.DataFrame) and not d.empty:
                            break
                    except:
                        pass
            if d is None or (isinstance(d, pd.DataFrame) and d.empty):
                results.append((sym, None, None, None, None))
                continue
            if isinstance(d.columns, pd.MultiIndex):
                c = d.xs("Close", axis=1, level=0).squeeze()
            elif "Close" in d.columns:
                c = d["Close"].squeeze()
            elif "Adj Close" in d.columns:
                c = d["Adj Close"].squeeze()
            else:
                c = d.iloc[:, 0].squeeze()
            c = pd.Series(c).dropna()
            if len(c) < 124:
                results.append((sym, None, None, None, None))
                continue
            cp = float(c.iloc[-1]); p123 = float(c.iloc[-124]); sr = cp / p123
            ni = nifty_c.index.get_indexer([c.index[-1]], method="ffill")[0]
            if ni < 123:
                results.append((sym, None, None, None, None))
                continue
            nr = float(nifty_c.iloc[ni]) / float(nifty_c.iloc[ni - 123])
            rs = (sr / nr) - 1.0
            results.append((sym, round(rs, 4), round((sr - 1) * 100, 1), round((nr - 1) * 100, 1), round(cp, 1)))
        except:
            results.append((sym, None, None, None, None))
    return pd.DataFrame(results, columns=["Stock", "RS", "Stock_123d%", "Nifty_123d%", "Price"])
@st.cache_data(ttl=300)
def _h_get_historical():
    """Build holdings history from L7_MAAC_Allocations.csv with allocation % as value."""
    all_h = []
    if not os.path.exists(_HERMES_OUTPUT):
        return pd.DataFrame(all_h)
    for d in sorted(os.listdir(_HERMES_OUTPUT)):
        # Try L7_MAAC_Allocations first (has Allocation_%)
        _p = os.path.join(_HERMES_OUTPUT, d, "L7_MAAC_Allocations.csv")
        try:
            _df_m = pd.read_csv(_p)
            for _, _r in _df_m.iterrows():
                _sym = str(_r.get("Symbol", "")).strip()
                _alloc = float(_r.get("Allocation_%", 0) or 0)
                _rs = float(_r.get("RS_vs_Nifty50", 0) or 0)
                _factor = float(_r.get("Factor_Score", 0) or 0)
                if _sym and not _sym.isdigit() and not __import__("re").match(r"^(NIFTY_|STRATEGY_|INDIA_VIX|MCX_)", _sym, __import__("re").IGNORECASE):
                    all_h.append({"Date": d, "Stock": _sym,
                                  "Allocation": _alloc, "RS": _rs, "Factor": _factor})
        except:
            # Fallback: read symbols from correlation matrix (legacy)
            _cp = os.path.join(_HERMES_OUTPUT, d, "Portfolio_Correlation_Matrix.csv")
            if os.path.exists(_cp):
                try:
                    for _s in read_data_smart(_cp, nrows=0).columns[1:]:
                        _s = str(_s).strip()
                        if _s.isdigit() or __import__("re").match(r"^(NIFTY_|STRATEGY_|INDIA_VIX|MCX_)", _s, __import__("re").IGNORECASE):
                            continue
                        all_h.append({"Date": d, "Stock": _s, "Allocation": 0, "RS": 0, "Factor": 0})
                except:
                    pass
    return pd.DataFrame(all_h)
@st.cache_data(ttl=300)
def _h_get_nifty_2y():
    try:
        import yfinance as yf
        ns = yf.download("^NSEI", period="2y", auto_adjust=True, progress=False)
        return ns
    except Exception:
        return pd.DataFrame()
@st.cache_data(ttl=900)
def _fetch_yf_commodities_news_data():
    """Fetch live commodities and global index data via yfinance.
    Uses individual Ticker.history() calls (more reliable than batch download)."""
    try:
        import yfinance as yf
        import pandas as pd
        from datetime import datetime, timedelta
        # Build commodities DataFrame manually (individual tickers are more reliable)
        _tickers_cc = {"Gold": "GC=F", "Silver": "SI=F", "Crude Oil": "CL=F", "USD/INR": "INR=X", "US 10Y": "^TNX"}
        _cc_dfs = []
        for _name, _t in _tickers_cc.items():
            try:
                _t_obj = yf.Ticker(_t)
                _h = _t_obj.history(period="5d", auto_adjust=True)
                if not _h.empty:
                    _h = _h[["Close"]].rename(columns={"Close": _t})
                    _cc_dfs.append(_h)
            except:
                pass
        _cc_data = pd.concat(_cc_dfs, axis=1, sort=True).sort_index() if _cc_dfs else pd.DataFrame()
        # Build global indices DataFrame
        _news_tickers = {"S&P 500": "^GSPC", "NASDAQ": "^IXIC", "Dow Jones": "^DJI", "Hang Seng": "^HSI", "Shanghai": "000001.SS"}
        _news_dfs = []
        for _name, _t in _news_tickers.items():
            try:
                _t_obj = yf.Ticker(_t)
                _h = _t_obj.history(period="5d", auto_adjust=True)
                if not _h.empty:
                    _h = _h[["Close"]].rename(columns={"Close": _t})
                    _news_dfs.append(_h)
            except:
                pass
        _news_data = pd.concat(_news_dfs, axis=1).sort_index() if _news_dfs else pd.DataFrame()
        return _cc_data, _news_data
    except Exception:
        return pd.DataFrame(), pd.DataFrame()
RS_THRESHOLD_H = 0.10
cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
def _load_index_hist(symbol, filename):
    path = os.path.join(cache_dir, filename)
    df = pd.DataFrame()
    # Try loading via cache_manager to ensure freshness
    try:
        from cache_manager import get_historical_data
        df = get_historical_data(symbol)
    except Exception:
        pass
    # Fallback to direct file read if cache_manager failed
    if df is None or df.empty:
        if os.path.exists(path):
            try:
                df = read_data_smart(path, parse_dates=["Date"]).sort_values("Date")
            except Exception:
                df = pd.DataFrame()
    if not df.empty:
        if "Date" not in df.columns:
            df = df.reset_index()
            if "index" in df.columns and "Date" not in df.columns:
                df = df.rename(columns={"index": "Date"})
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
    if not df.empty and "Close" in df.columns:
        df["Close"] = pd.to_numeric(df["Close"])
        # Adjust Nifty Next 50 ETF scaling
        if symbol == "NIFTY_NEXT_50":
            if df["Close"].iloc[-1] < 2000:
                df["Close"] = df["Close"] * 100.0
        elif symbol == "NIFTY_MIDCAP_150":
            if df["Close"].iloc[-1] < 100:
                df["Close"] = df["Close"] * 1000.0
        return df
    return pd.DataFrame()
def _get_price_and_change(df):
    if df.empty or len(df) < 2:
        return 0.0, 0.0
    last = float(df["Close"].iloc[-1])
    prev = float(df["Close"].iloc[-2])
    chg = ((last / prev) - 1.0) * 100.0 if prev > 0 else 0.0
    return last, chg
# Helper to register manual Veto Actions (Add/Remove)
def register_veto_action(symbol, action, alloc_pct=0.0):
    if not symbol:
        st.warning("Please enter a valid stock symbol.")
        return
    symbol = symbol.upper().strip()
    # Correct path calculation based on dashboard.py location
    veto_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Veto_add_remove.csv")
    date_str = pd.Timestamp.now().strftime("%Y-%m-%d")
    timestamp_str = pd.Timestamp.now().isoformat()
    entry = pd.DataFrame([{
        "Date": date_str,
        "Symbol": symbol,
        "Action": action,
        "Allocation_Pct": alloc_pct,
        "Status": "PENDING",
        "Timestamp": timestamp_str
    }])
    if os.path.exists(veto_file):
        entry.to_csv(veto_file, mode='a', header=False, index=False)
    else:
        entry.to_csv(veto_file, mode='w', header=True, index=False)
    if action == "VETO_ADD":
        st.success(f"✅ Registered {symbol} for manual ADD ({alloc_pct}% alloc) in next pipeline run.")
    else:
        st.success(f"🛑 Registered {symbol} for manual REMOVE in next pipeline run.")
    import time
    time.sleep(0.5)
    st.cache_data.clear()
    st.rerun()
# ── Calculate Global Strategy Performance Metrics for KPI Cards ──
global_strat_ret = "0.00%"
global_strat_dd = "0.00%"
global_strat_vol = "0.00%"
global_strat_sharpe = "0.00"
# ── TA 4.0 Blended Strategy — session state for cross-tab KPI sync ──
if "ta4_ret" not in st.session_state:
    st.session_state.ta4_ret = global_strat_ret
    st.session_state.ta4_dd = global_strat_dd
    st.session_state.ta4_sharpe = global_strat_sharpe
    st.session_state.ta4_vol = global_strat_vol
_global_perf_error = ""
try:
    cache_dir_g = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
    output_dir_g = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    nifty50_path_g = os.path.join(cache_dir_g, "NIFTY_50_history.csv")
    if os.path.exists(nifty50_path_g):
        df_nifty_g = read_data_smart(nifty50_path_g)
        df_nifty_g["Date"] = pd.to_datetime(df_nifty_g["Date"])
        df_nifty_g = df_nifty_g.sort_values("Date").reset_index(drop=True)
        end_date_g = pd.to_datetime(selected_date)
        nifty_dates = pd.to_datetime(df_nifty_g["Date"])
        if not (nifty_dates == end_date_g).any():
            past_dates = nifty_dates[nifty_dates <= end_date_g]
            if not past_dates.empty:
                end_date_g = past_dates.max()
            else:
                end_date_g = nifty_dates.max()
        import re
        run_dates_g = []
        for d in sorted(os.listdir(output_dir_g)):
            if os.path.isdir(os.path.join(output_dir_g, d)) and re.match(r"^\d{4}-\d{2}-\d{2}$", d):
                if os.path.exists(os.path.join(output_dir_g, d, "L7_MAAC_Allocations.csv")):
                    run_dates_g.append(d)
        if run_dates_g:
            inception_date_g = pd.to_datetime(run_dates_g[0]) - pd.Timedelta(days=1)
        else:
            inception_date_g = pd.to_datetime("2026-05-27")
        start_date_g = inception_date_g
        if start_date_g > end_date_g:
            start_date_g = end_date_g - pd.Timedelta(days=1)
        nifty_min_date_g = df_nifty_g["Date"].min()
        if start_date_g < nifty_min_date_g:
            start_date_g = nifty_min_date_g
        df_nifty_range_g = df_nifty_g[(df_nifty_g["Date"] >= start_date_g) & (df_nifty_g["Date"] <= end_date_g)]
        trading_dates_g = df_nifty_range_g["Date"].tolist()
        if len(trading_dates_g) >= 2:
            import glob
            csv_files_g = glob.glob(os.path.join(cache_dir_g, "*_history.csv"))
            series_list_g = {}
            _valid_syms_set_g = set()
            for _d in run_dates_g:
                for filename in ["L7_MAAC_Allocations.csv"]:
                    path = os.path.join(output_dir_g, _d, filename)
                    if os.path.exists(path):
                        df_tmp = read_data_smart(path)
                        col_tmp = next((c for c in ["Allocation_%", "Allocation_Pct", "Alloc_%"] if c in df_tmp.columns), None)
                        if col_tmp:
                            _valid_syms_set_g.update(df_tmp[df_tmp[col_tmp] > 0]["Symbol"].tolist())
                            break
                # TA 4.0 Blended: also load Core symbols from L1_Core_Allocations.csv
                _core_path = os.path.join(output_dir_g, _d, "L1_Core_Allocations.csv")
                if os.path.exists(_core_path):
                    df_core_tmp = read_data_smart(_core_path)
                    if "Symbol" in df_core_tmp.columns and "Core_Weight" in df_core_tmp.columns:
                        _valid_syms_set_g.update(df_core_tmp[df_core_tmp["Core_Weight"] > 0]["Symbol"].tolist())
            if state_loaded and "holdings" in state_data:
                _valid_syms_set_g.update(state_data["holdings"].keys())
            for file in csv_files_g:
                sym = os.path.basename(file).replace("_history.csv", "")
                if re.match(r"^(NIFTY_|STRATEGY_|INDIA_VIX|MCX_)", sym, re.IGNORECASE):
                    continue
                if _valid_syms_set_g and sym not in _valid_syms_set_g:
                    continue
                try:
                    df_s = read_data_smart(file)
                    df_s["Date"] = pd.to_datetime(df_s["Date"])
                    s = df_s.dropna(subset=["Close"]).set_index("Date")["Close"]
                    if not s.empty:
                        series_list_g[sym] = s
                except:
                    pass
            if series_list_g:
                price_matrix_g = pd.DataFrame(series_list_g).sort_index().ffill()
                portfolio_value_g = 100.0
                portfolio_history_g = [{"Date": trading_dates_g[0].strftime("%Y-%m-%d"), "Value": 100.0}]
                # TA 4.0 Blended: separate Core and Satellite allocation dicts
                sat_allocations_g = None
                core_allocations_g = {}
                valid_past_runs_g = [d for d in run_dates_g if pd.to_datetime(d) < trading_dates_g[0]]
                if valid_past_runs_g:
                    _d = valid_past_runs_g[-1]
                elif run_dates_g:
                    _d = run_dates_g[0]
                else:
                    _d = None
                if _d:
                    # Load Satellite (MAAC) allocations
                    for filename in ["L7_MAAC_Allocations.csv"]:
                        path = os.path.join(output_dir_g, _d, filename)
                        if os.path.exists(path):
                            df_tmp = read_data_smart(path)
                            col_tmp = next((c for c in ["Allocation_%", "Allocation_Pct", "Alloc_%"] if c in df_tmp.columns), None)
                            if col_tmp:
                                df_active = df_tmp[df_tmp[col_tmp] > 0]
                                sat_allocations_g = dict(zip(df_active["Symbol"], df_active[col_tmp] / 100.0))
                                break
                    # Load Core allocations from L1_Core_Allocations.csv
                    _core_path = os.path.join(output_dir_g, _d, "L1_Core_Allocations.csv")
                    if os.path.exists(_core_path):
                        df_core = read_data_smart(_core_path)
                        if "Symbol" in df_core.columns and "Core_Weight" in df_core.columns:
                            for _, _cr in df_core[df_core["Core_Weight"] > 0].iterrows():
                                core_allocations_g[str(_cr["Symbol"])] = float(_cr["Core_Weight"])
                    # Normalize Core weights to sum to 1.0
                    _core_sum = sum(core_allocations_g.values())
                    if _core_sum > 0:
                        for _sym in list(core_allocations_g.keys()):
                            core_allocations_g[_sym] /= _core_sum
                # Read blueprint leverage params (with sensible defaults)
                _bp_core_pct = float(blueprint.get("active_core_equities_pct", 60.0)) if isinstance(blueprint, dict) else 60.0
                _bp_mtf_pct = float(blueprint.get("mtf_leverage_pct", 150.0)) if isinstance(blueprint, dict) else 150.0
                _bp_total_exp = _bp_core_pct + _bp_mtf_pct
                for i in range(1, len(trading_dates_g)):
                    prev_date = trading_dates_g[i - 1]
                    curr_date = trading_dates_g[i]
                    curr_date_str = curr_date.strftime("%Y-%m-%d")
                    # ── Compute Core daily return (unleveraged) ──
                    core_daily_ret = 0.0
                    core_wt_count = 0.0
                    if core_allocations_g:
                        for sym, weight in core_allocations_g.items():
                            if sym in price_matrix_g.columns:
                                s_val = price_matrix_g.loc[:prev_date, sym]
                                p_prev = float(s_val.iloc[-1]) if not s_val.empty and pd.notna(s_val.iloc[-1]) else None
                                s_val_curr = price_matrix_g.loc[:curr_date, sym]
                                p_curr = float(s_val_curr.iloc[-1]) if not s_val_curr.empty and pd.notna(s_val_curr.iloc[-1]) else None
                                if p_prev is not None and p_curr is not None and p_prev > 0:
                                    core_daily_ret += weight * ((p_curr / p_prev) - 1.0)
                                    core_wt_count += weight
                        if core_wt_count > 0:
                            core_daily_ret /= core_wt_count
                    # ── Compute Satellite daily return (leveraged) ──
                    sat_daily_ret = 0.0
                    sat_wt_used = 0.0
                    if sat_allocations_g:
                        sat_raw_ret = 0.0
                        sat_raw_wt = 0.0
                        for sym, weight in sat_allocations_g.items():
                            if sym in price_matrix_g.columns:
                                s_val = price_matrix_g.loc[:prev_date, sym]
                                p_prev = float(s_val.iloc[-1]) if not s_val.empty and pd.notna(s_val.iloc[-1]) else None
                                s_val_curr = price_matrix_g.loc[:curr_date, sym]
                                p_curr = float(s_val_curr.iloc[-1]) if not s_val_curr.empty and pd.notna(s_val_curr.iloc[-1]) else None
                                if p_prev is not None and p_curr is not None and p_prev > 0:
                                    sat_raw_ret += weight * ((p_curr / p_prev) - 1.0)
                                    sat_raw_wt += weight
                        if sat_raw_wt > 0:
                            sat_daily_ret = (sat_raw_ret / sat_raw_wt) * sum(sat_allocations_g.values())
                            sat_wt_used = sum(sat_allocations_g.values())
                    # ── TA 4.0 Blended: Core + Leveraged Satellite ──
                    # Formula: (Core_pct × core_ret + MTF_pct × sat_ret) / (Core_pct + MTF_pct)
                    # This correctly amplifies Satellite returns by the leverage ratio
                    if _bp_total_exp > 0 and sat_wt_used > 0:
                        daily_ret = (_bp_core_pct / 100.0 * core_daily_ret + _bp_mtf_pct / 100.0 * sat_daily_ret) / (_bp_total_exp / 100.0)
                    elif core_wt_count > 0:
                        daily_ret = core_daily_ret
                    elif sat_wt_used > 0:
                        daily_ret = sat_daily_ret
                else:
                    daily_ret = 0.0
                portfolio_value_g = portfolio_value_g * (1.0 + daily_ret)
                portfolio_history_g.append({"Date": curr_date_str, "Value": portfolio_value_g})
                # Rebalance allocations on pipeline run dates
                if curr_date_str in run_dates_g:
                    # Update Satellite allocations
                    for filename in ["L7_MAAC_Allocations.csv"]:
                        path = os.path.join(output_dir_g, curr_date_str, filename)
                        if os.path.exists(path):
                            df_tmp = read_data_smart(path)
                            col_tmp = next((c for c in ["Allocation_%", "Allocation_Pct", "Alloc_%"] if c in df_tmp.columns), None)
                            if col_tmp:
                                df_active = df_tmp[df_tmp[col_tmp] > 0]
                                sat_allocations_g = dict(zip(df_active["Symbol"], df_active[col_tmp] / 100.0))
                                break
                    # Update Core allocations
                    _core_path = os.path.join(output_dir_g, curr_date_str, "L1_Core_Allocations.csv")
                    core_allocations_g = {}
                    if os.path.exists(_core_path):
                        df_core = read_data_smart(_core_path)
                        if "Symbol" in df_core.columns and "Core_Weight" in df_core.columns:
                            for _, _cr in df_core[df_core["Core_Weight"] > 0].iterrows():
                                core_allocations_g[str(_cr["Symbol"])] = float(_cr["Core_Weight"])
                        _core_sum = sum(core_allocations_g.values())
                        if _core_sum > 0:
                            for _sym in list(core_allocations_g.keys()):
                                core_allocations_g[_sym] /= _core_sum
                    # Refresh blueprint leverage params for dynamic regime adjustment
                    _bp_core_pct = float(blueprint.get("active_core_equities_pct", 60.0)) if isinstance(blueprint, dict) else 60.0
                    _bp_mtf_pct = float(blueprint.get("mtf_leverage_pct", 150.0)) if isinstance(blueprint, dict) else 150.0
                    _bp_total_exp = _bp_core_pct + _bp_mtf_pct
                df_port_g = pd.DataFrame(portfolio_history_g)
                if not df_port_g.empty:
                    raw_g = pd.to_numeric(df_port_g["Value"], errors="coerce").ffill().fillna(100.0).values
                    total_ret_g = float(raw_g[-1] - 100.0)
                    cum_max_g = np.maximum.accumulate(raw_g)
                    dd_g = (raw_g - cum_max_g) / cum_max_g * 100.0
                    max_dd_g = float(np.min(dd_g)) if len(dd_g) > 0 else 0.0
                    daily_rets_g = pd.Series(raw_g).pct_change().dropna()
                    daily_rets_g = daily_rets_g.replace([np.inf, -np.inf], np.nan).dropna()
                    if len(daily_rets_g) >= 5 and daily_rets_g.std() > 0:
                        std_dev_g = daily_rets_g.std() * np.sqrt(252) * 100.0
                        _rf_daily_g = 0.065 / 252  # 6.5% p.a. risk-free rate (India T-bill proxy)
                        sharpe_g = ((daily_rets_g.mean() - _rf_daily_g) / daily_rets_g.std()) * np.sqrt(252)
                    else:
                        std_dev_g = 0.0
                        sharpe_g = 0.0
                global_strat_ret = f"{total_ret_g:+.2f}%"
                global_strat_dd = f"{max_dd_g:.2f}%"
                global_strat_vol = f"{std_dev_g:.2f}%"
                global_strat_sharpe = f"{sharpe_g:.2f}"
                # Also push to session state as fallback before Performance Analyzer sync
                if "ta4_ret" in st.session_state:
                    st.session_state.ta4_ret = global_strat_ret
                    st.session_state.ta4_dd = global_strat_dd
                    st.session_state.ta4_sharpe = global_strat_sharpe
                    st.session_state.ta4_vol = global_strat_vol
except Exception as _e_gp:
    _global_perf_error = str(_e_gp)
# Display global perf computation warning early for visibility
if _global_perf_error:
    st.warning(f"⚠️ **Strategy performance computation incomplete:** {_global_perf_error}. CAGR/Sharpe/DD may show defaults.")
# Compute inception label for KPI display (dynamically from first pipeline run date)
try:
    _inception_label = inception_date_g.strftime("%b %d, %Y")
except Exception:
    _inception_label = "May 27, 2026"
# Load India VIX
vix_df = _load_index_hist("INDIA_VIX", "INDIA_VIX_history.csv")
vix_price, vix_chg = _get_price_and_change(vix_df)
# ── Market Mood Index (MMI) — Short & Long Term ──
# (Tickertape MMI scraping removed to prevent blocking main UI thread. Replaced with local MMI calculations inside tab_cio.)
# ── Research Link Buttons Helper ─────────────────────────────────────
def _render_research_links(layout="row", compact=False):
    """Render pill-style research platform buttons. Call from any tab."""
    if compact:
        _ps = "6px 12px"
        _fs = "0.75rem"
    else:
        _ps = "7px 16px"
        _fs = "0.82rem"
    _mb = "12px" if compact else "16px"
    st.markdown(f"""
    <div style="display:flex;gap:8px;margin-bottom:{_mb};flex-wrap:wrap;">
        <a href="https://www.morningstar.in/tools/ECFundscreener.aspx" target="_blank" rel="noopener noreferrer" style="text-decoration:none;">
            <span style="display:inline-flex;align-items:center;gap:5px;background:rgba(99,102,241,0.12);border:1px solid rgba(99,102,241,0.3);border-radius:10px;padding:{_ps};font-family:'Inter',sans-serif;font-size:{_fs};font-weight:600;color:#818cf8;cursor:pointer;">⭐ MS Fund Screener</span>
        </a>
        <a href="https://www.etmoney.com/mutual-funds/equity" target="_blank" rel="noopener noreferrer" style="text-decoration:none;">
            <span style="display:inline-flex;align-items:center;gap:5px;background:rgba(16,185,129,0.12);border:1px solid rgba(16,185,129,0.3);border-radius:10px;padding:{_ps};font-family:'Inter',sans-serif;font-size:{_fs};font-weight:600;color:#34d399;cursor:pointer;">💰 ET Money</span>
        </a>
        <a href="https://www.tickertape.in/screener/mutual-fund" target="_blank" rel="noopener noreferrer" style="text-decoration:none;">
            <span style="display:inline-flex;align-items:center;gap:5px;background:rgba(251,191,36,0.12);border:1px solid rgba(251,191,36,0.3);border-radius:10px;padding:{_ps};font-family:'Inter',sans-serif;font-size:{_fs};font-weight:600;color:#fbbf24;cursor:pointer;">📊 Tickertape</span>
        </a>
        <a href="https://chartink.com/watchlist_dashboard" target="_blank" rel="noopener noreferrer" style="text-decoration:none;">
            <span style="display:inline-flex;align-items:center;gap:5px;background:rgba(59,130,246,0.12);border:1px solid rgba(59,130,246,0.3);border-radius:10px;padding:{_ps};font-family:'Inter',sans-serif;font-size:{_fs};font-weight:600;color:#60a5fa;cursor:pointer;">📈 Chartink</span>
        </a>
        <a href="https://www.screener.in/watchlist/153801/" target="_blank" rel="noopener noreferrer" style="text-decoration:none;">
            <span style="display:inline-flex;align-items:center;gap:5px;background:rgba(168,85,247,0.12);border:1px solid rgba(168,85,247,0.3);border-radius:10px;padding:{_ps};font-family:'Inter',sans-serif;font-size:{_fs};font-weight:600;color:#c084fc;cursor:pointer;">📋 Screener.in</span>
        </a>
        <a href="https://trendlyne.com/" target="_blank" rel="noopener noreferrer" style="text-decoration:none;">
            <span style="display:inline-flex;align-items:center;gap:5px;background:rgba(236,72,153,0.12);border:1px solid rgba(236,72,153,0.3);border-radius:10px;padding:{_ps};font-family:'Inter',sans-serif;font-size:{_fs};font-weight:600;color:#f472b6;cursor:pointer;">🔍 Trendlyne</span>
        </a>
        <a href="https://in.tradingview.com/markets/stocks-india/" target="_blank" rel="noopener noreferrer" style="text-decoration:none;">
            <span style="display:inline-flex;align-items:center;gap:5px;background:rgba(34,211,238,0.12);border:1px solid rgba(34,211,238,0.3);border-radius:10px;padding:{_ps};font-family:'Inter',sans-serif;font-size:{_fs};font-weight:600;color:#22d3ee;cursor:pointer;">📉 TradingView</span>
        </a>
    </div>
    """, unsafe_allow_html=True)
def render_unified_veto_ui(tab_key="default"):
    """Compact veto add/remove UI rendered in each tab.
    Reads/writes Veto_add_remove.csv for manual overrides."""
    _vk = f"veto_{tab_key}"
    st.markdown("""
    <div style="background:rgba(59,130,246,0.05);border:1px solid rgba(59,130,246,0.15);border-radius:10px;padding:10px 14px;margin-bottom:16px;">
        <div style="font-size:0.75rem;font-weight:700;color:#818cf8;text-transform:uppercase;letter-spacing:0.04em;">⚡ Manual Override</div>
        <div style="font-size:0.7rem;color:#64748b;margin-top:2px;">Add or remove any stock bypassing algorithm — takes effect on next pipeline run.</div>
    </div>
    """, unsafe_allow_html=True)
    _c1, _c2, _c3 = st.columns([2, 1.5, 1.5])
    with _c1:
        _s = st.text_input("Symbol", key=f"{_vk}_sym", placeholder="e.g. RELIANCE")
    with _c2:
        _a = st.number_input("Alloc %", 0.5, 50.0, 5.0, 0.5, key=f"{_vk}_alloc")
    with _c3:
        st.markdown('<div style="height:2px;"></div>', unsafe_allow_html=True)
        _ba = st.button("➕ Add", key=f"{_vk}_add", use_container_width=True)
        _br = st.button("➖ Remove", key=f"{_vk}_rem", use_container_width=True)
        if _ba and _s.strip():
            register_veto_action(_s.strip().upper(), "VETO_ADD", _a)
        if _br and _s.strip():
            register_veto_action(_s.strip().upper(), "VETO_REMOVE", 0.0)
# Create tabs for role-based terminal segregation
tab_cio, tab_active, tab_core, tab_ta, tab_vams, tab_orch_rs_5d, tab_bt, tab_global = st.tabs([
    "🌐 Market Intelligence",
    "💼 Master Portfolio (Active Positions)",
    "🏛️ Core Allocations Tab",
    "💼 VAM-GQ (Volatility adjusted momentum - growth and quality)",
    "⚡ VAM-B (Volatility adjusted momentum - Blended)",
    "🚀 Master Analyzer tab",
    "📊 Performance Analyzer",
    "🌍 Global & Thematic",
])
# ──────────────────────────────────────────────────────────────────────────────
# TAB 1: CIO EXECUTIVE CONTROL PANEL
# ──────────────────────────────────────────────────────────────────────────────
with tab_cio:
    _render_research_links(compact=True)
    render_unified_veto_ui("tab_cio")
    st.caption("📡 **Role:** Market Regime · Breadth · Alpha Scores · Macro Dashboard")
    if not state_loaded:
        st.error("⚠️ **System Alert**: Core pipeline state could not be synchronized. Displayed metrics may be stale.")
    if regime.get("nifty500_below_150_sloping_down", False):
        st.warning("⚠️ **NIFTY 500 Defensive Cash Cushion Active**: Nifty 500 is below its 150 EMA and sloping downward. Cash position is forced to **50%**, with Core (65%) and Satellite (35%) buckets scaled proportionally to 32.5% and 17.5% respectively.")
    # Pre-fetch cached yfinance data to avoid synchronous block on every Streamlit rerun
    _cc_data, _news_data = _fetch_yf_commodities_news_data()
    # ════════════════════════════════════════════════════════════════════════════
    # TICKERTAPE-ALIGNED 3-TIER MMI SYSTEM
    # ════════════════════════════════════════════════════════════════════════════
    # ── Data Inputs ──
    _vix_l = vix_price if vix_price else 15.0
    _vix_chg = vix_chg if vix_chg else 0.0
    _breadth = regime.get("breadth_score", 50.0)
    _pct20 = regime.get("pct_20", 50.0)
    _pct200 = regime.get("pct_200", 50.0)
    _bull_votes = regime.get("bull_votes", 2)
    _nh_nl = regime.get("nh_nl_ratio", 1.0)
    _fii_proxy = 0.0
    try:
        _fii_vals = df_maac.get("FII_Change_%", df_maac.get("FII_Change", pd.Series([0]))).dropna()
        _fii_proxy = (_fii_vals > 0).mean() * 100 if len(_fii_vals) > 0 else 50
    except: _fii_proxy = 50
    # ── Exact Tickertape Momentum: (90D EMA - 30D EMA) / 90D EMA ──
    _tt_mom = 50
    try:
        _n50_df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", "NIFTY_50_history.csv"))
        _n50_df["Date"] = pd.to_datetime(_n50_df["Date"])
        _n50_df = _n50_df[_n50_df["Date"] <= pd.to_datetime(selected_date)].sort_values("Date")
        if not _n50_df.empty and len(_n50_df) >= 90:
            _n50_df["EMA30"] = _n50_df["Close"].ewm(span=30).mean()
            _n50_df["EMA90"] = _n50_df["Close"].ewm(span=90).mean()
            _mom_val = ((_n50_df["EMA90"].iloc[-1] - _n50_df["EMA30"].iloc[-1]) / _n50_df["EMA90"].iloc[-1]) * 100
            _tt_mom = max(0, min(100, 50 + _mom_val * 20))  # +1% → 70, -1% → 30
    except: _tt_mom = 50
    # ── Exact Tickertape Gold Demand: Gold vs Nifty 2-week relative return ──
    _tt_gold = 50
    try:
        _gold_df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", "MCX_GOLD_history.csv"))
        _gold_df["Date"] = pd.to_datetime(_gold_df["Date"])
        _gold_df = _gold_df[_gold_df["Date"] <= pd.to_datetime(selected_date)].tail(14).sort_values("Date")
        _n50_g = _n50_df.tail(14) if '_n50_df' in dir() else None
        if _gold_df is not None and not _gold_df.empty and len(_gold_df) >= 2 and _n50_g is not None and len(_n50_g) >= 2:
            _g_ret = (_gold_df["Close"].iloc[-1] / _gold_df["Close"].iloc[0] - 1) * 100
            _n_ret = (_n50_g["Close"].iloc[-1] / _n50_g["Close"].iloc[0] - 1) * 100
            _gold_diff = _g_ret - _n_ret  # positive = gold outperforming (fear)
            _tt_gold = max(0, min(100, 50 - _gold_diff * 10))  # gold up → fear (low score)
    except: _tt_gold = 50
    # ── Tier 1: Tickertape MMI (6 Components, Equal Weight) ──
    # 1) FII Activity
    _c1_fii = max(0, min(100, _fii_proxy))
    # 2) Volatility & Skew: VIX inversion (VIX 28→0, VIX 12→100)
    _c2_vix = max(0, min(100, (28 - _vix_l) / 16 * 100))
    # 3) EXACT Tickertape Momentum: (90D EMA - 30D EMA) / 90D EMA
    _c3_mom = _tt_mom
    # 4) Market Breadth
    _c4_breadth = max(0, min(100, _breadth))
    # 5) Price Strength: 52W high/low
    _c5_strength = max(0, min(100, min(100, _nh_nl * 20))) if _nh_nl > 0 else 50
    # 6) EXACT Tickertape Gold Demand: Gold vs Nifty 2-week
    _c6_gold = _tt_gold
    # Composite: Equal weight (Tickertape methodology)
    _st_mmi = (_c1_fii + _c2_vix + _c3_mom + _c4_breadth + _c5_strength + _c6_gold) / 6.0
    _st_mmi = max(0, min(100, _st_mmi))
    # Tickertape zones: <30 Extreme Fear, 30-45 Fear, 45-55 Neutral, 55-70 Greed, >70 Extreme Greed
    if _st_mmi >= 70: _st_zone, _st_c = "Extreme Greed", "#f59e0b"
    elif _st_mmi >= 55: _st_zone, _st_c = "Greed", "#10b981"
    elif _st_mmi >= 45: _st_zone, _st_c = "Neutral", "#94a3b8"
    elif _st_mmi >= 30: _st_zone, _st_c = "Fear", "#f97316"
    else: _st_zone, _st_c = "Extreme Fear", "#ef4444"
    # ── Tier 2: Medium-Term MMI (Momentum Engine) ──
    _mt_breadth = _breadth
    _mt_sector = regime.get("above_50ema_sectors", 0.5) * 100
    _mt_mmi = (_mt_breadth * 0.5 + _mt_sector * 0.3 + 50 * 0.2)
    _mt_mmi = max(0, min(100, _mt_mmi))
    if _mt_mmi >= 70: _mt_zone, _mt_c = "Extreme Greed", "#f59e0b"
    elif _mt_mmi >= 55: _mt_zone, _mt_c = "Greed", "#10b981"
    elif _mt_mmi >= 45: _mt_zone, _mt_c = "Neutral", "#94a3b8"
    elif _mt_mmi >= 30: _mt_zone, _mt_c = "Fear", "#f97316"
    else: _mt_zone, _mt_c = "Extreme Fear", "#ef4444"
    # ── Tier 3: Long-Term MMI (Structural Regime) ──
    _lt_200 = _pct200
    _lt_leadership = (_bull_votes / 4.0) * 100
    _lt_mmi = (_lt_200 * 0.5 + _lt_leadership * 0.3 + 50 * 0.2)
    _lt_mmi = max(0, min(100, _lt_mmi))
    _lt_zone = "Contraction" if _lt_mmi < 40 else "Neutral" if _lt_mmi < 60 else "Expansion"
    _lt_c = "#ef4444" if _lt_mmi < 40 else "#94a3b8" if _lt_mmi < 60 else "#10b981"
    # ── Alpha Score ──
    _alpha_qg = (_st_mmi * 0.30 + _mt_mmi * 0.50 + _lt_mmi * 0.20) / 10.0
    _alpha_qg = max(0, min(10, _alpha_qg))
    # Compute VAM-B Alpha from L1_VAM_B_Universe.csv average VAM score
    _alpha_vamb = 5.0  # default neutral
    _vamb_av_path = os.path.join(OUTPUT_DIR, "L1_VAM_B_Universe.csv")
    if not os.path.exists(_vamb_av_path):
        _vamb_av_path, _ = _resolve_pipeline_file("L1_VAM_B_Universe.csv") if not os.path.exists(_vamb_av_path) else (_vamb_av_path, None)
    if os.path.exists(_vamb_av_path):
        try:
            _df_vamb_a = pd.read_csv(_vamb_av_path)
            if not _df_vamb_a.empty and "Score" in _df_vamb_a.columns:
                _avg_vam = _df_vamb_a["Score"].mean()
                _alpha_vamb = max(0, min(10, (_avg_vam + 2.0) / 4.0 * 10.0))  # normalize ~ -2..+2 range to 0-10
        except:
            pass
    # Compute Core Alpha from L1_Core_Allocations.csv weighted momentum
    _alpha_core = 5.0  # default neutral
    _core_a_path = os.path.join(OUTPUT_DIR, "L1_Core_Allocations.csv")
    if not os.path.exists(_core_a_path):
        _core_a_path, _ = _resolve_pipeline_file("L1_Core_Allocations.csv") if not os.path.exists(_core_a_path) else (_core_a_path, None)
    if os.path.exists(_core_a_path):
        try:
            _df_core_a = pd.read_csv(_core_a_path)
            if not _df_core_a.empty and "Core_Weight" in _df_core_a.columns and "Score" in _df_core_a.columns:
                _weighted_sc = (_df_core_a["Core_Weight"] * _df_core_a["Score"]).sum() / _df_core_a["Core_Weight"].sum()
                _alpha_core = max(0, min(10, _weighted_sc / 10.0))  # Score is 0-100, normalize to 0-10
        except:
            pass
    # Unified composite = average of all three
    _alpha_unified = (_alpha_qg + _alpha_vamb + _alpha_core) / 3.0
    # Display all 4 scores as cards
    _strat_scores = [
        ("⚡ QG-VAM Alpha", "30% Short · 50% Medium · 20% Long-Term", _alpha_qg,
         "Market regime & breadth signal. Blends short/medium/long MMI."),
        ("📈 VAM-B Alpha", "Pure momentum signal", _alpha_vamb,
         "Average volatility-adjusted momentum across the raw VAM-B universe."),
        ("🏛️ Core Alpha", "Quality & momentum blend", _alpha_core,
         "Weighted composite score of Core ETF/MF holdings' trailing momentum."),
        ("🔷 Unified Composite", "QG-VAM + VAM-B + Core", _alpha_unified,
         "Equal-weighted average of all three strategy alphas for overall positioning."),
    ]
    _alpha_cols = st.columns(4)
    for _ac, (_at, _asub, _av, _adesc) in zip(_alpha_cols, _strat_scores):
        _ac_c = "#10b981" if _av >= 8 else "#f59e0b" if _av >= 5 else "#ef4444"
        _ac_zone = "Go Heavy" if _av >= 8 else "Be Selective" if _av >= 5 else "Protect Capital"
        _ac.markdown(f"""
        <div style="background:linear-gradient(135deg,rgba({','.join(str(int(_ac_c.lstrip('#')[i:i + 2], 16)) for i in (0, 2, 4))},0.10),rgba(15,23,42,0.4));border:1px solid {_ac_c}40;border-radius:14px;padding:12px 14px;margin-bottom:12px;height:205px;">
            <div style="font-family:'Outfit';font-size:0.85rem;font-weight:700;color:#f1f5f9;">{_at}</div>
            <div style="font-size:0.65rem;color:#94a3b8;margin-top:1px;">{_asub}</div>
            <div style="font-family:'Outfit';font-size:1.8rem;font-weight:800;color:{_ac_c};margin:4px 0 2px;">{_av:.1f}<span style="font-size:0.7rem;color:#94a3b8;">/10</span></div>
            <span style="background:{_ac_c}20;color:{_ac_c};padding:1px 8px;border-radius:6px;font-size:0.65rem;font-weight:700;">{_ac_zone}</span>
            <div style="margin-top:6px;background:rgba(15,23,42,0.5);border-radius:4px;height:4px;overflow:hidden;">
                <div style="width:{_av * 10:.0f}%;background:linear-gradient(90deg,#ef4444,#f97316,#f59e0b,#10b981);height:100%;border-radius:4px;"></div>
            </div>
            <div style="display:flex;justify-content:space-between;font-size:0.58rem;color:#64748b;margin-top:2px;">
                <span>Protect</span><span>Selective</span><span>Go Heavy</span>
            </div>
            <div style="font-size:0.65rem;color:#94a3b8;margin-top:6px;line-height:1.4;">{_adesc}</div>
        </div>
        """, unsafe_allow_html=True)
    def _make_gauge(_score, _zone, _color, _title, _sub):
        """Build a Tickertape-style half-circle gauge."""
        # Tickertape zone configuration
        _zones = [
            (0, 30, "#22c55e", "EXTREME FEAR"),   # green = opportunity
            (30, 55, "#f97316", "FEAR"),           # orange
            (55, 70, "#f97316", "GREED"),          # orange (same zone, gauge fills here)
            (70, 100, "#ef4444", "EXTREME GREED"),  # red = caution
        ]
        # Gauge arc color = the zone the score falls in
        _bar_c = _color
        return go.Figure(go.Indicator(
            mode="gauge+number",
            value=_score,
            number=dict(suffix="", font=dict(size=34, color=_bar_c, family="Outfit")),
            title=dict(
                text=f"<b>{_title}</b><br><span style='font-size:14px;color:{_bar_c};font-weight:700;'>{_zone}</span>"
                f"<br><span style='font-size:10px;color:#64748b;'>{_sub}</span>",
                font=dict(size=13, color="#cbd5e1", family="Inter")),
            gauge=dict(
                axis=dict(range=[0, 100], tickwidth=1, tickcolor="#64748b",
                         tickfont=dict(size=9, color="#475569"),
                         tickvals=[0, 25, 50, 75, 100],
                         ticktext=["0", "25", "50", "75", "100"]),
                bar=dict(color=_bar_c, thickness=0.65),
                bgcolor="rgba(0,0,0,0)", borderwidth=0,
                shape="angular",
                steps=[
                    dict(range=[0, 30], color="rgba(34,197,94,0.35)", line=dict(color="#22c55e", width=1)),
                    dict(range=[30, 55], color="rgba(249,115,22,0.25)", line=dict(color="#f97316", width=1)),
                    dict(range=[55, 70], color="rgba(249,115,22,0.35)", line=dict(color="#f97316", width=1)),
                    dict(range=[70, 100], color="rgba(239,68,68,0.30)", line=dict(color="#ef4444", width=1)),
                ],
                threshold=dict(
                    line=dict(color=_bar_c, width=5),
                    thickness=0.85, value=_score
                )
            )
        ))
    _fig_s = _make_gauge(_st_mmi, _st_zone, _st_c, "Tier 1: Tactical Pulse", "1—15D | 6-Component MMI")
    _fig_m = _make_gauge(_mt_mmi, _mt_zone, _mt_c, "Tier 2: Momentum Engine", "1—6M | Breadth+Leadership")
    _fig_l = _make_gauge(_lt_mmi, _lt_zone, _lt_c, "Tier 3: Structural Regime", "6M—2Y | 200d EMA+Macro")
    for _f in [_fig_s, _fig_m, _fig_l]:
        _f.update_layout(template="plotly_dark", height=300,
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=30, r=30, t=55, b=20), font=dict(family="Inter"))
    # Display
    _g1, _g2, _g3 = st.columns(3)
    with _g1:
        st.plotly_chart(_fig_s, use_container_width=True, config={'displayModeBar': False})
        st.markdown(f"""
        <div class='glass-card' style='padding:12px 16px;margin-top:-18px;border-left:4px solid {_st_c};'>
            <div style='font-size:0.85rem;font-weight:700;color:{_st_c};'>📊 Short-Term MMI (Tickertape Method)</div>
            <div style='font-size:0.72rem;color:#94a3b8;margin-top:4px;line-height:1.6;'>
                <b>MMI: {_st_mmi:.1f}</b> · VIX: <b>{_vix_l:.1f}</b> ({_vix_chg:+.1f}%) · Breadth: <b>{_breadth:.0f}%</b>
            </div>
            <div style='font-size:0.68rem;color:#64748b;margin-top:5px;padding:7px;background:rgba(15,23,42,0.4);border-radius:6px;'>
                <b style='color:{_st_c};'>6 Components:</b>
                <span style='color:#e2e8f0;'>FII</span> {_c1_fii:.0f} ·
                <span style='color:#e2e8f0;'>VIX</span> {_c2_vix:.0f} ·
                <span style='color:#e2e8f0;'>Mom</span> {_c3_mom:.0f} ·
                <span style='color:#e2e8f0;'>Brth</span> {_c4_breadth:.0f} ·
                <span style='color:#e2e8f0;'>52W</span> {_c5_strength:.0f} ·
                <span style='color:#e2e8f0;'>Gold</span> {_c6_gold:.0f}
            </div>
        </div>
        """, unsafe_allow_html=True)
    with _g2:
        st.plotly_chart(_fig_m, use_container_width=True, config={'displayModeBar': False})
        _mt_sig = "Power Setup — max out positions" if _mt_mmi > 70 else "Be selective — quality-only" if _mt_mmi > 45 else "Defensive — rotate to Low-Vol"
        st.markdown(f"""
        <div class='glass-card' style='padding:12px 16px;margin-top:-18px;border-left:4px solid {_mt_c};'>
            <div style='font-size:0.85rem;font-weight:700;color:{_mt_c};'>📈 Medium-Term MMI — Momentum Engine</div>
            <div style='font-size:0.72rem;color:#94a3b8;margin-top:4px;line-height:1.6;'>
                Breadth: <b>{_breadth:.0f}%</b> · Sector Breadth: <b>{_mt_sector:.0f}%</b>
            </div>
            <div style='font-size:0.68rem;color:#64748b;margin-top:5px;padding:7px;background:rgba(15,23,42,0.4);border-radius:6px;'>
                <b style='color:{_mt_c};'>Signal:</b> {_mt_sig}. {'Deep breadth supports positions.' if _mt_sector > 50 else 'Thin leadership — reduce sizes.'}
            </div>
        </div>
        """, unsafe_allow_html=True)
    with _g3:
        st.plotly_chart(_fig_l, use_container_width=True, config={'displayModeBar': False})
        _lt_sig = "Risk-On — full deployment" if _lt_mmi > 60 else "Risk-Off — raise cash"
        st.markdown(f"""
        <div class='glass-card' style='padding:12px 16px;margin-top:-18px;border-left:4px solid {_lt_c};'>
            <div style='font-size:0.85rem;font-weight:700;color:{_lt_c};'>🏛️ Long-Term MMI — Structural Regime</div>
            <div style='font-size:0.72rem;color:#94a3b8;margin-top:4px;line-height:1.6;'>
                200d Breadth: <b>{_pct200:.0f}%</b> · Leadership: <b>{_bull_votes}/4</b>
            </div>
            <div style='font-size:0.68rem;color:#64748b;margin-top:5px;padding:7px;background:rgba(15,23,42,0.4);border-radius:6px;'>
                <b style='color:{_lt_c};'>Signal:</b> {_lt_sig}. {'Growth stocks favored.' if _lt_mmi > 60 else 'Gold/USDINR headwinds — margin risk.'}
            </div>
        </div>
        """, unsafe_allow_html=True)
    # ── MACRO MARKET REGIME ──
    st.subheader("🌐 Market Regime & Drawdown Status")
    col_reg1, col_reg2, col_vix, col_reg3, col_reg4 = st.columns(5)
    # Unified Regime Card
    with col_reg1:
        reg_val = regime.get("market_regime", "SIDEWAYS")
        reg_col = "#10b981" if "BULL" in reg_val else ("#f59e0b" if "SIDEWAYS" in reg_val else "#ef4444")
        bull_votes = regime.get("bull_votes", 0)
        st.markdown(f"""
        <div class="glass-card kpi-card" style="border-left: 5px solid {reg_col};">
            <span style="color: #94a3b8; font-size: 0.8rem; font-weight: 600; text-transform: uppercase;">Unified Market Regime</span>
            <h2 style="margin: 8px 0 3px 0; color: {reg_col}; font-size: 2.2rem; font-weight: 700;">{reg_val}</h2>
            <div style="font-size: 0.78rem; color: #94a3b8; margin-bottom: 6px;">
                Leadership: <b>{bull_votes}/4 Indices BULL</b>
            </div>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 4px; font-size: 0.72rem; color: #e2e8f0; line-height: 1.3;">
                <div>Mega: <b style="color: {'#10b981' if regime.get('nifty50_bullish', False) else '#ef4444'};">{"BULL" if regime.get("nifty50_bullish", False) else "BEAR"}</b></div>
                <div>Large: <b style="color: {'#10b981' if regime.get('niftynext50_bullish', False) else '#ef4444'};">{"BULL" if regime.get("niftynext50_bullish", False) else "BEAR"}</b></div>
                <div>Mid: <b style="color: {'#10b981' if regime.get('nifty150_bullish', False) else '#ef4444'};">{"BULL" if regime.get("nifty150_bullish", False) else "BEAR"}</b></div>
                <div>Small: <b style="color: {'#10b981' if regime.get('nifty250_bullish', False) else '#ef4444'};">{"BULL" if regime.get("nifty250_bullish", False) else "BEAR"}</b></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    # Breadth & Thrust Card
    with col_reg2:
        breadth_score = regime.get("breadth_score", 0.0)
        thrust_active = regime.get("breadth_thrust_active", False)
        breadth_confirmed = regime.get("breadth_confirmed", False)
        badge_style = 'padding: 2px 6px; font-size: 0.7rem; border-radius: 12px; display: inline-block; white-space: nowrap;'
        thrust_tag = f'<span class="badge-normal" style="{badge_style}">THRUST ACTIVE</span>' if thrust_active else f'<span class="badge-hold" style="{badge_style}">NO THRUST</span>'
        confirm_tag = f'<span class="badge-normal" style="{badge_style}">BREADTH CONFIRMED</span>' if breadth_confirmed else f'<span class="badge-hold" style="{badge_style}">UNCONFIRMED</span>'
        st.markdown(f"""
        <div class="glass-card kpi-card">
            <span style="color: #94a3b8; font-size: 0.8rem; font-weight: 600; text-transform: uppercase;">Market Breadth Score</span>
            <h2 style="margin: 8px 0 3px 0; color: #a855f7; font-size: 2.2rem; font-weight: 700;">{breadth_score:.1f}%</h2>
            <div style="font-size: 0.82rem; color: #94a3b8; line-height: 1.3;">
                Regime: <b>{regime.get("breadth_regime", "Risk Off")}</b>
                <div style="margin-top: 6px; display: flex; flex-direction: column; gap: 4px; align-items: flex-start;">
                    {thrust_tag}
                    {confirm_tag}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    # India VIX Card with dynamic percentile-based thresholds
    with col_vix:
        badge_style = 'padding: 2px 6px; font-size: 0.7rem; border-radius: 12px; display: inline-block; white-space: nowrap;'
        # Compute VIX thresholds dynamically from history if available
        _vix_hi = vix_df["Close"].dropna() if not vix_df.empty and "Close" in vix_df.columns else pd.Series(dtype=float)
        if len(_vix_hi) >= 60:
            _vix_p75 = float(_vix_hi.quantile(0.75))
            _vix_p50 = float(_vix_hi.quantile(0.50))
        else:
            _vix_p75, _vix_p50 = 20.0, 15.0  # fallback defaults
        vix_badge = f'<span class="badge-normal" style="{badge_style}">🟢 LOW FEAR</span>'
        vix_color = "#10b981"
        if vix_price >= _vix_p75:
            vix_badge = f'<span class="badge-avoid" style="{badge_style}">🔴 HIGH FEAR</span>'
            vix_color = "#ef4444"
        elif vix_price >= _vix_p50:
            vix_badge = f'<span class="badge-hold" style="{badge_style}">🟡 MODERATE</span>'
            vix_color = "#f59e0b"
        st.markdown(f"""
        <div class="glass-card kpi-card" style="border-left: 5px solid {vix_color};">
            <span style="color: #94a3b8; font-size: 0.8rem; font-weight: 600; text-transform: uppercase;">India VIX (Volatility)</span>
            <h2 style="margin: 8px 0 3px 0; color: {vix_color}; font-size: 2.2rem; font-weight: 700;">{vix_price:.2f}</h2>
            <div style="font-size: 0.82rem; color: #94a3b8; line-height: 1.4;">
                1D Change: <b style="color: {'#10b981' if vix_chg < 0 else '#ef4444'};">{vix_chg:+.2f}%</b>
                <div style="margin-top: 6px;">{vix_badge}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    # Drawdown Circuit Breaker Card
    with col_reg3:
        dd_pct = drawdown.get("drawdown_pct", 0.0)
        dd_action = drawdown.get("action", "NORMAL")
        badge_style = 'padding: 2px 6px; font-size: 0.7rem; border-radius: 12px; display: inline-block; white-space: nowrap;'
        dd_badge = f'<span class="badge-normal" style="{badge_style}">🟢 NORMAL</span>'
        if dd_action == "HALT":
            dd_badge = f'<span class="badge-hold" style="{badge_style}">🟡 HALT</span>'
        elif dd_action == "EMERGENCY":
            dd_badge = f'<span class="badge-avoid" style="{badge_style}">🔴 EMERGENCY</span>'
        st.markdown(f"""
        <div class="glass-card kpi-card" style="border-left: 5px solid {"#10b981" if dd_action == "NORMAL" else ("#f59e0b" if dd_action == "HALT" else "#ef4444")};">
            <span style="color: #94a3b8; font-size: 0.8rem; font-weight: 600; text-transform: uppercase;">Portfolio Drawdown</span>
            <h2 style="margin: 8px 0 3px 0; color: #f1f5f9; font-size: 2.2rem; font-weight: 700;">{dd_pct:.2f}%</h2>
            <div style="font-size: 0.82rem; color: #94a3b8; line-height: 1.4;">
                Mult: <b>{drawdown.get("risk_multiplier", 1.0):.2f}x</b>
                <div style="margin-top: 6px;">Gov Action: {dd_badge}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    # Leadership & Exposure Multiplier Card
    with col_reg4:
        breadth_mult = regime.get("exposure_multiplier", 0.25)
        gov_mult = drawdown.get("risk_multiplier", 1.0)
        final_mult = breadth_mult * gov_mult
        st.markdown(f"""
        <div class="glass-card kpi-card">
            <span style="color: #94a3b8; font-size: 0.8rem; font-weight: 600; text-transform: uppercase;">Exposure Sizing Multiplier</span>
            <h2 style="margin: 8px 0 3px 0; color: #06b6d4; font-size: 2.2rem; font-weight: 700;">{final_mult:.2f}x</h2>
            <div style="font-size: 0.75rem; color: #e2e8f0; line-height: 1.35; margin-top: 5px;">
                Breadth: <b>{breadth_mult:.2f}x</b> × Gov: <b>{gov_mult:.2f}x</b><br/>
                L-Ratio: <b>{regime.get("leadership_ratio", 0.0):.2f}</b><br/>
                NH/NL: <b>{regime.get("new_highs_52w", 0)}/{regime.get("new_lows_52w", 0)}</b>
            </div>
        </div>
        """, unsafe_allow_html=True)
    # ──────────────────────────────────────────────────────────────────────────
    # INDEX FUNDA-TECHNICAL MONITOR (Redesigned – All Indices)
    # ──────────────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🏢 Index Funda-Technical Monitor")
    # ── Load price history for all indices to compute live change % ──
    # Note: loader and helper functions moved to top of file for global availability
    idx_histories = {
        "Nifty 50": _load_index_hist("NIFTY_50", "NIFTY_50_history.csv"),
        "Nifty Next 50": _load_index_hist("NIFTY_NEXT_50", "NIFTY_NEXT_50_history.csv"),
        "Nifty Midcap 150": _load_index_hist("NIFTY_MIDCAP_150", "NIFTY_MIDCAP_150_history.csv"),
        "Nifty Smallcap 250": _load_index_hist("NIFTY_SMALLCAP_250", "NIFTY_SMALLCAP_250_history.csv"),
        "Nifty Microcap 250": _load_index_hist("NIFTY_MICROCAP_250", "NIFTY_MICROCAP_250_history.csv"),
    }
    # ── Index Configuration (keys, colors, regime data keys, defaults) ──
    index_config = [
        {
            "name": "Nifty 50", "icon": "🔵", "accent": "#3b82f6",
            "prefix": "nifty50", "bullish_key": "nifty50_bullish",
            "defaults": {"pe": 20.5, "pb": 3.2, "div_yield": 1.3, "cagr_5yr": 12.0, "cagr_10yr": 11.0, "price": 23600.0}
        },
        {
            "name": "Nifty Next 50", "icon": "🟢", "accent": "#10b981",
            "prefix": "niftynext50", "bullish_key": "niftynext50_bullish",
            "defaults": {"pe": 26.0, "pb": 4.5, "div_yield": 0.9, "cagr_5yr": 14.0, "cagr_10yr": 13.0, "price": 75500.0}
        },
        {
            "name": "Nifty Midcap 150", "icon": "🟡", "accent": "#f59e0b",
            "prefix": "midcap150", "bullish_key": "nifty150_bullish",
            "defaults": {"pe": 30.0, "pb": 4.0, "div_yield": 0.7, "cagr_5yr": 18.0, "cagr_10yr": 16.0, "price": 22600.0}
        },
        {
            "name": "Nifty Smallcap 250", "icon": "🟣", "accent": "#a855f7",
            "prefix": "smallcap250", "bullish_key": "nifty250_bullish",
            "defaults": {"pe": 28.1, "pb": 3.82, "div_yield": 0.78, "cagr_5yr": 15.2, "cagr_10yr": 15.5, "price": 17000.0}
        },
        {
            "name": "Nifty Microcap 250", "icon": "🔴", "accent": "#ef4444",
            "prefix": "microcap250", "bullish_key": "microcap250_bullish",
            "defaults": {"pe": 27.1, "pb": 3.43, "div_yield": 0.65, "cagr_5yr": 16.0, "cagr_10yr": 14.0, "price": 24000.0}
        },
    ]
    # ── 1. Top-Level KPI Price Strip (Live Prices + Daily Change) ──
    kpi_cols = st.columns(5)
    for i, cfg in enumerate(index_config):
        hist_df = idx_histories.get(cfg["name"], pd.DataFrame())
        price, chg = _get_price_and_change(hist_df)
        regime_price_key = f"{cfg['prefix']}_price"
        regime_val = regime.get(regime_price_key, 0.0)
        is_simulated = False
        if price > 0.0:
            price_display = price
            # Adjust scaling if the source returned raw values instead of scaled ones
            if cfg["name"] == "Nifty Next 50" and price < 2000:
                price_display *= 100.0
            elif cfg["name"] == "Nifty Midcap 150" and price < 100:
                price_display *= 1000.0
        else:
            if regime_val is not None and regime_val > 0.0:
                price_display = regime_val
            else:
                price_display = cfg["defaults"]["price"]
                is_simulated = True
        # Trend from regime or calculated dynamically
        is_bull = None
        if cfg["bullish_key"] and cfg["bullish_key"] in regime:
            is_bull = regime.get(cfg["bullish_key"])
        else:
            # Calculate dynamically on the fly from history data
            if not hist_df.empty and "Close" in hist_df.columns:
                close_series = hist_df["Close"].dropna()
                if len(close_series) >= 150:
                    sma_150 = close_series.rolling(150).mean()
                    last_close = float(close_series.iloc[-1])
                    last_sma = float(sma_150.iloc[-1])
                    prev_sma = float(sma_150.iloc[-10]) if len(sma_150) >= 10 else last_sma
                    is_bull = (last_close > last_sma) and (last_sma > prev_sma)
        if is_bull is None:
            trend_badge = '<span style="background: rgba(255,255,255,0.08); color: #94a3b8; padding: 2px 8px; border-radius: 10px; font-size: 0.7rem; font-weight: 600;">—</span>'
        elif is_bull:
            trend_badge = '<span style="background: rgba(16,185,129,0.15); color: #34d399; padding: 2px 8px; border-radius: 10px; font-size: 0.7rem; font-weight: 600;">▲ BULL</span>'
        else:
            trend_badge = '<span style="background: rgba(239,68,68,0.15); color: #f87171; padding: 2px 8px; border-radius: 10px; font-size: 0.7rem; font-weight: 600;">▼ BEAR</span>'
        if is_simulated:
            simulated_badge = '<span style="background: rgba(245,158,11,0.15); color: #fbbf24; padding: 2px 6px; border-radius: 6px; font-size: 0.6rem; font-weight: 700; margin-left: 6px;" title="Fallback value used because history failed to load">⚠️ SIMULATED</span>'
        else:
            simulated_badge = ''
        chg_color = "#34d399" if chg >= 0 else "#f87171"
        chg_prefix = "+" if chg > 0 else ""
        with kpi_cols[i]:
            st.markdown(f"""
            <div class="glass-card" style="padding: 14px 16px; margin-bottom: 0; border-top: 3px solid {cfg['accent']}; text-align: center;">
                <div style="font-family: 'Outfit'; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 1px; color: #94a3b8; font-weight: 600; margin-bottom: 6px;">{cfg['icon']} {cfg['name']}{simulated_badge}</div>
                <div style="font-family: 'Outfit'; font-size: 1.5rem; font-weight: 700; color: #f1f5f9;">₹{price_display:,.1f}</div>
                <div style="font-size: 0.82rem; color: {chg_color}; font-weight: 600; margin-top: 4px;">{chg_prefix}{chg:.2f}%</div>
                <div style="margin-top: 6px;">{trend_badge}</div>
            </div>
            """, unsafe_allow_html=True)
    # ── 2. Detailed Valuation & Performance Comparison Table ──
    st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
    def _compute_index_returns(df):
        if df.empty or len(df) < 2:
            return {"1w": 0.0, "1m": 0.0, "3m": 0.0, "6m": 0.0, "9m": 0.0, "12m": 0.0}
        df = df.copy()
        if "Date" not in df.columns:
            df = df.reset_index()
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date")
        series = df.set_index("Date")["Close"].dropna()
        if len(series) < 2:
            return {"1w": 0.0, "1m": 0.0, "3m": 0.0, "6m": 0.0, "9m": 0.0, "12m": 0.0}
        last_date = series.index[-1]
        last_val = float(series.iloc[-1])
        def get_pct_change(delta_days):
            target_date = last_date - pd.Timedelta(days=delta_days)
            sub = series[series.index <= target_date]
            if not sub.empty:
                prev_val = float(sub.iloc[-1])
                return ((last_val / prev_val) - 1.0) * 100.0 if prev_val > 0 else 0.0
            else:
                prev_val = float(series.iloc[0])
                return ((last_val / prev_val) - 1.0) * 100.0 if prev_val > 0 else 0.0
        return {
            "1w": get_pct_change(7),
            "1m": get_pct_change(30),
            "3m": get_pct_change(91),
            "6m": get_pct_change(182),
            "9m": get_pct_change(273),
            "12m": get_pct_change(365)
        }
    val_html = []
    val_html.append("""
    <div class="glass-card" style="padding: 0; overflow: hidden; border-radius: 12px;">
    <table class="premium-table">
    <thead>
    <tr style="background: rgba(30, 41, 59, 0.7); border-bottom: 2px solid rgba(255,255,255,0.08);">
    <th style="width: 18%; padding: 10px 12px; text-align: left; color: #94a3b8; font-family: 'Outfit'; font-weight: 600; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.5px;">Index</th>
    <th style="width: 7%; padding: 10px 12px; text-align: right; color: #94a3b8; font-family: 'Outfit'; font-weight: 600; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.5px;">P/E</th>
    <th style="width: 6%; padding: 10px 12px; text-align: right; color: #94a3b8; font-family: 'Outfit'; font-weight: 600; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.5px;">P/B</th>
    <th style="width: 8%; padding: 10px 12px; text-align: right; color: #94a3b8; font-family: 'Outfit'; font-weight: 600; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.5px;">Div Yield</th>
    <th style="width: 6%; padding: 10px 12px; text-align: right; color: #94a3b8; font-family: 'Outfit'; font-weight: 600; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.5px;">1W</th>
    <th style="width: 6%; padding: 10px 12px; text-align: right; color: #94a3b8; font-family: 'Outfit'; font-weight: 600; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.5px;">1M</th>
    <th style="width: 6%; padding: 10px 12px; text-align: right; color: #94a3b8; font-family: 'Outfit'; font-weight: 600; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.5px;">3M</th>
    <th style="width: 6%; padding: 10px 12px; text-align: right; color: #94a3b8; font-family: 'Outfit'; font-weight: 600; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.5px;">6M</th>
    <th style="width: 6%; padding: 10px 12px; text-align: right; color: #94a3b8; font-family: 'Outfit'; font-weight: 600; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.5px;">9M</th>
    <th style="width: 6%; padding: 10px 12px; text-align: right; color: #94a3b8; font-family: 'Outfit'; font-weight: 600; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.5px;">12M</th>
    <th style="width: 8%; padding: 10px 12px; text-align: right; color: #94a3b8; font-family: 'Outfit'; font-weight: 600; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.5px;">CAGR 5Y</th>
    <th style="width: 8%; padding: 10px 12px; text-align: right; color: #94a3b8; font-family: 'Outfit'; font-weight: 600; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.5px;">CAGR 10Y</th>
    <th style="width: 9%; padding: 10px 12px; text-align: center; color: #94a3b8; font-family: 'Outfit'; font-weight: 600; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.5px;">Regime</th>
    </tr>
    </thead>
    <tbody>""")
    def cell_style(val):
        if val > 5.0:
            return "background: rgba(16, 185, 129, 0.22); color: #34d399; font-weight: 600;"
        elif val > 0.0:
            return "background: rgba(16, 185, 129, 0.06); color: #a7f3d0;"
        elif val < -5.0:
            return "background: rgba(239, 68, 68, 0.22); color: #f87171; font-weight: 600;"
        elif val < 0.0:
            return "background: rgba(239, 68, 68, 0.06); color: #fecaca;"
        return "color: #94a3b8;"
    for cfg in index_config:
        pfx = cfg["prefix"]
        dfl = cfg["defaults"]
        pe_val = regime.get(f"{pfx}_pe") or dfl["pe"]
        pb_val = regime.get(f"{pfx}_pb") or dfl["pb"]
        div_val = regime.get(f"{pfx}_div_yield") or dfl["div_yield"]
        cagr5 = regime.get(f"{pfx}_cagr_5yr") or dfl["cagr_5yr"]
        cagr10 = regime.get(f"{pfx}_cagr_10yr") or dfl["cagr_10yr"]
        hist_df = idx_histories.get(cfg["name"], pd.DataFrame())
        ret = _compute_index_returns(hist_df)
        is_bull = regime.get(cfg["bullish_key"], None) if cfg["bullish_key"] else None
        if is_bull is None:
            # Calculate dynamically on the fly from history data
            if not hist_df.empty and "Close" in hist_df.columns:
                close_series = hist_df["Close"].dropna()
                if len(close_series) >= 150:
                    sma_150 = close_series.rolling(150).mean()
                    last_close = float(close_series.iloc[-1])
                    last_sma = float(sma_150.iloc[-1])
                    prev_sma = float(sma_150.iloc[-10]) if len(sma_150) >= 10 else last_sma
                    is_bull = (last_close > last_sma) and (last_sma > prev_sma)
        if is_bull is None:
            regime_cell = '<span style="color: #64748b;">—</span>'
        elif is_bull:
            regime_cell = '<span style="background: rgba(16,185,129,0.15); color: #34d399; padding: 3px 10px; border-radius: 10px; font-size: 0.78rem; font-weight: 600;">BULL</span>'
        else:
            regime_cell = '<span style="background: rgba(239,68,68,0.15); color: #f87171; padding: 3px 10px; border-radius: 10px; font-size: 0.78rem; font-weight: 600;">BEAR</span>'
        pe_color = "#34d399" if pe_val < 22 else ("#fbbf24" if pe_val < 30 else "#f87171")
        val_html.append(f"""<tr style="border-bottom: 1px solid rgba(255,255,255,0.04);">
        <td style="padding: 9px 12px; color: {cfg['accent']}; font-weight: 700; font-family: 'Outfit';">{cfg['icon']} {cfg['name']}</td>
        <td style="padding: 9px 12px; text-align: right; color: {pe_color}; font-family: monospace; font-weight: 600;">{pe_val:.1f}</td>
        <td style="padding: 9px 12px; text-align: right; color: #fbbf24; font-family: monospace;">{pb_val:.2f}</td>
        <td style="padding: 9px 12px; text-align: right; color: #10b981; font-family: monospace;">{div_val:.2f}%</td>
        <td style="padding: 9px 12px; text-align: right; {cell_style(ret['1w'])} font-family: monospace;">{ret['1w']:.1f}%</td>
        <td style="padding: 9px 12px; text-align: right; {cell_style(ret['1m'])} font-family: monospace;">{ret['1m']:.1f}%</td>
        <td style="padding: 9px 12px; text-align: right; {cell_style(ret['3m'])} font-family: monospace;">{ret['3m']:.1f}%</td>
        <td style="padding: 9px 12px; text-align: right; {cell_style(ret['6m'])} font-family: monospace;">{ret['6m']:.1f}%</td>
        <td style="padding: 9px 12px; text-align: right; {cell_style(ret['9m'])} font-family: monospace;">{ret['9m']:.1f}%</td>
        <td style="padding: 9px 12px; text-align: right; {cell_style(ret['12m'])} font-family: monospace; font-weight: 600;">{ret['12m']:.1f}%</td>
        <td style="padding: 9px 12px; text-align: right; color: #60a5fa; font-family: monospace; font-weight: 600;">{cagr5:.1f}%</td>
        <td style="padding: 9px 12px; text-align: right; color: #60a5fa; font-family: monospace;">{cagr10:.1f}%</td>
        <td style="padding: 9px 12px; text-align: center;">{regime_cell}</td>
        </tr>""")
    val_html.append("</tbody></table></div>")
    st.markdown("".join(val_html).replace("\n", ""), unsafe_allow_html=True)
    # ── 3. INDEX RS COMPARISON (vs Nifty 50) ──
    # Trend Chart (All Indices) ──
    st.markdown("<div style='margin-top: 25px;'></div>", unsafe_allow_html=True)
    st.markdown("### 📈 Comparative Index Performance (Base 100)")
    tf_cols = st.columns([2, 5])
    with tf_cols[0]:
        timeframe = st.radio(
            "Select Analysis Window",
            options=["1M", "3M", "6M", "YTD", "1Y", "MAX"],
            index=4,  # default to 1Y
            horizontal=True,
            key="idx_chart_timeframe"
        )
    # Determine start date based on selection and selected_date
    sel_dt = pd.to_datetime(selected_date)
    if timeframe == "1M":
        start_date = sel_dt - pd.Timedelta(days=30)
    elif timeframe == "3M":
        start_date = sel_dt - pd.Timedelta(days=91)
    elif timeframe == "6M":
        start_date = sel_dt - pd.Timedelta(days=182)
    elif timeframe == "YTD":
        start_date = pd.to_datetime(f"{sel_dt.year}-01-01")
    elif timeframe == "1Y":
        start_date = sel_dt - pd.Timedelta(days=365)
    else:
        start_date = None
    chart_indices = [
        ("Nifty 50", "#3b82f6", 2.5),
        ("Nifty Next 50", "#10b981", 2.0),
        ("Nifty Midcap 150", "#f59e0b", 2.0),
        ("Nifty Smallcap 250", "#a855f7", 2.0),
        ("Nifty Microcap 250", "#ef4444", 2.0),
    ]
    series_dict = {}
    for idx_name, _, _ in chart_indices:
        df = idx_histories.get(idx_name, pd.DataFrame())
        if not df.empty and "Date" in df.columns and "Close" in df.columns:
            df = df.copy()
            df["Date"] = pd.to_datetime(df["Date"])
            if start_date is not None:
                df = df[df["Date"] >= start_date]
            df = df.sort_values("Date").drop_duplicates(subset=["Date"])
            series_dict[idx_name] = df.set_index("Date")["Close"]
    if series_dict and "Nifty 50" in series_dict:
        nifty_base = series_dict["Nifty 50"]
        # Align all series by combining them on their common Date index
        df_combined = pd.DataFrame(series_dict).ffill().bfill().dropna()
        if not df_combined.empty:
            df_combined = df_combined.sort_index()
            # Compute cumulative RS line for each index against Nifty 50
            df_rs = pd.DataFrame(index=df_combined.index)
            df_rs["Nifty 50"] = 0.0  # Nifty vs itself = 0 baseline
            nifty_rebased = nifty_base / nifty_base.iloc[0]
            for col in df_combined.columns:
                if col == "Nifty 50":
                    continue
                idx_rebased = df_combined[col] / df_combined[col].iloc[0]
                df_rs[col] = ((idx_rebased / nifty_rebased) - 1.0) * 100.0
            if not df_rs.empty:
                # Filter controls for fig_idx
                all_idx_names = [name for name, _, _ in chart_indices]
                select_all_idx = st.checkbox("Select All Indices", value=False, key="select_all_idx")
                if select_all_idx:
                    selected_idx_names = all_idx_names
                else:
                    selected_idx_names = st.multiselect(
                        "Filter Indices to Display",
                        options=all_idx_names,
                        default=all_idx_names,
                        key="selected_indices_rs"
                    )
                fig_idx = go.Figure()
                for idx_name, color, width in chart_indices:
                    if idx_name in selected_idx_names:
                        rolling_slope = df_rs[idx_name].diff(20).fillna(0.0)
                        if idx_name == "Nifty 50":
                            fig_idx.add_trace(go.Scatter(
                                x=df_rs.index, y=df_rs[idx_name],
                                name=idx_name + " (baseline)",
                                line=dict(color=color, width=width, dash='dash'),
                                customdata=np.stack((rolling_slope.values,), axis=-1),
                                hovertemplate=(
                                    f"<span style='color:{color}; font-weight:bold; font-size:13px;'>{idx_name}</span><br>"
                                    "RS vs Nifty: <b>%{y:+.2f}%</b><br>"
                                    "20D Slope: <b>%{customdata[0]:+.2f}%</b>"
                                    "<extra></extra>"
                                )
                            ))
                        elif idx_name in df_rs.columns:
                            fig_idx.add_trace(go.Scatter(
                                x=df_rs.index, y=df_rs[idx_name],
                                name=idx_name,
                                line=dict(color=color, width=width, shape='spline'),
                                customdata=np.stack((rolling_slope.values,), axis=-1),
                                hovertemplate=(
                                    f"<span style='color:{color}; font-weight:bold; font-size:13px;'>{idx_name}</span><br>"
                                    "RS vs Nifty: <b>%{y:+.2f}%</b><br>"
                                    "20D Slope: <b>%{customdata[0]:+.2f}%</b>"
                                    "<extra></extra>"
                                )
                            ))
                fig_idx.update_layout(
                    hovermode="closest",
                    template="plotly_dark",
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=40, r=20, t=15, b=25),
                    height=420,
                    legend=dict(
                        orientation="h", y=-0.22, x=0.5, xanchor="center",
                        bgcolor="rgba(0,0,0,0)",
                        font=dict(size=11, color="#94a3b8")
                    ),
                    xaxis=dict(
                        showgrid=True, gridcolor="rgba(255,255,255,0.03)", zeroline=False, tickfont=dict(color="#64748b"),
                        showspikes=True, spikethickness=1, spikedash="dot", spikecolor="rgba(255,255,255,0.3)", spikemode="across",
                    ),
                    yaxis=dict(
                        title="RS % (vs Nifty 50)",
                        showgrid=True, gridcolor="rgba(255,255,255,0.03)",
                        zeroline=True, zerolinecolor="rgba(255,255,255,0.15)",
                        tickfont=dict(color="#64748b"),
                        tickformat="+.1f",
                        title_font=dict(color="#64748b", size=11),
                        showspikes=True, spikethickness=1, spikedash="dot", spikecolor="rgba(255,255,255,0.3)", spikemode="across"
                    ),
                    hoverlabel=dict(
                        bgcolor="rgba(15, 23, 42, 0.9)",
                        font=dict(size=12, color="#f1f5f9"),
                        bordercolor="rgba(255, 255, 255, 0.1)"
                    )
                )
                fig_idx.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.2)", line_width=1.5)
                st.plotly_chart(fig_idx, use_container_width=True)
            else:
                st.info("Not enough data (need 124+ days) for RS calculation.")
        else:
            st.info("No overlapping data found for the selected timeframe.")
    else:
        st.info("No index history data available for charting, or Nifty 50 baseline missing.")
    # ── 4. COMMODITIES & CURRENCY SNAPSHOT ──
    st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
    st.markdown("### 💰 Commodities & Currency Snapshot")
    try:
        _tickers_cc = {"Gold": "GC=F", "Silver": "SI=F", "Crude Oil": "CL=F", "USD/INR": "INR=X", "US 10Y": "^TNX"}
        _cc_cols = st.columns(5)
        for i, (_label, _t) in enumerate(_tickers_cc.items()):
            with _cc_cols[i]:
                try:
                    _c = _cc_data[_t].dropna() if _t in _cc_data.columns else pd.Series(dtype=float)
                    if len(_c) >= 2:
                        _cc_px = float(_c.iloc[-1])
                        _cc_chg = ((_cc_px / float(_c.iloc[-2])) - 1.0) * 100.0
                        _cc_c = "#34d399" if _cc_chg >= 0 else "#f87171"
                        _cc_pre = "+" if _cc_chg > 0 else ""
                        st.markdown(f"""
                        <div class="glass-card" style="padding: 12px; text-align: center; border-top: 2px solid {'#f59e0b' if _label == 'Gold' else '#94a3b8'}; margin-bottom:0;">
                            <div style="font-size:0.7rem;color:#94a3b8;font-weight:600;text-transform:uppercase;">{_label}</div>
                            <div style="font-family:'Outfit';font-size:1.1rem;font-weight:700;color:#f1f5f9;margin-top:4px;">{_cc_px:.2f}</div>
                            <div style="font-size:0.78rem;color:{_cc_c};font-weight:600;">{_cc_pre}{_cc_chg:.2f}%</div>
                        </div>""", unsafe_allow_html=True)
                except:
                    st.markdown(f"""
                    <div class="glass-card" style="padding: 12px; text-align: center; border-top: 2px solid #475569; margin-bottom:0;">
                        <div style="font-size:0.7rem;color:#94a3b8;font-weight:600;text-transform:uppercase;">{_label}</div>
                        <div style="font-family:'Outfit';font-size:1.1rem;font-weight:700;color:#475569;margin-top:4px;">—</div>
                    </div>""", unsafe_allow_html=True)
    except:
        pass
    # ── 4b. GLOBAL MARKET NEWS (Collapsible) ──
    with st.expander("🌍 Global Market News & Economic Calendar"):
        try:
            _news_tickers = {"S&P 500": "^GSPC", "NASDAQ": "^IXIC", "Dow Jones": "^DJI", "Hang Seng": "^HSI", "Shanghai": "000001.SS"}
            _news_cols = st.columns(5)
            for i, (_label, _t) in enumerate(_news_tickers.items()):
                with _news_cols[i]:
                    try:
                        _c2 = _news_data[_t].dropna() if _t in _news_data.columns else pd.Series(dtype=float)
                        if len(_c2) >= 2:
                            _n_px = float(_c2.iloc[-1])
                            _n_chg = ((_n_px / float(_c2.iloc[-2])) - 1.0) * 100.0
                            _n_c = "#34d399" if _n_chg >= 0 else "#f87171"
                            st.metric(_label, f"{_n_px:,.0f}", f"{_n_chg:+.2f}%")
                    except:
                        st.metric(_label, "—", "—")
        except:
            pass
        st.markdown("<div style='margin-top:10px;'></div>", unsafe_allow_html=True)
        # Free-text news sources (links)
        st.markdown("""
        <div style="display:flex;gap:10px;flex-wrap:wrap;font-size:0.82rem;">
            <a href="https://www.bloomberg.com/markets" target="_blank" style="color:#60a5fa;">Bloomberg Markets</a>
            <a href="https://www.reuters.com/markets/" target="_blank" style="color:#60a5fa;">Reuters Markets</a>
            <a href="https://economictimes.indiatimes.com/markets" target="_blank" style="color:#60a5fa;">Economic Times</a>
            <a href="https://www.moneycontrol.com/" target="_blank" style="color:#60a5fa;">Moneycontrol</a>
            <a href="https://www.investing.com/economic-calendar/" target="_blank" style="color:#fbbf24;">📅 Economic Calendar</a>
        </div>""", unsafe_allow_html=True)
    # ──────────────────────────────────────────────────────────────────────────
    # SECTORAL & THEMATIC MONITOR
    # ──────────────────────────────────────────────────────────────────────────
    st.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True)
    st.subheader("📊 Sectoral & Thematic Performance Monitor")
    @st.cache_data(ttl=300)
    def _load_sectoral_thematic_data():
        tickers_mapping = {
            "Nifty Bank": ("^NSEBANK", "1272670"),
            "Nifty Private Bank": ("NIFTY_PVT_BANK.NS", "1274024"),
            "Nifty PSU Bank": ("^CNXPSUBANK", "1272693"),
            "Nifty Financial Services": ("NIFTY_FIN_SERVICE.NS", "1272803"),
            "Nifty IT": ("^CNXIT", "1272649"),
            "Nifty Auto": ("^CNXAUTO", "1272796"),
            "Nifty FMCG": ("^CNXFMCG", "1272711"),
            "Nifty Pharma": ("^CNXPHARMA", "1272672"),
            "Nifty Metal": ("^CNXMETAL", "1272797"),
            "Nifty Realty": ("^CNXREALTY", "1272692"),
            "Nifty Energy": ("^CNXENERGY", "1272671"),
            "Nifty Media": ("^CNXMEDIA", "1272799"),
            "Nifty Infrastructure": ("^CNXINFRA", "1272689"),
            "Nifty Consumption": ("^CNXCONSUM", "1272795"),
            "Nifty Services": ("^CNXSERVICE", "1272770"),
            "Nifty MNC": ("^CNXMNC", "1272696"),
            "Nifty PSE": ("^CNXPSE", "1272714"),
            "Nifty Microcap 250": ("NIFTY_MICROCAP_250.NS", "1284386"),
        }
        # Build name->ticker map from tickers_mapping (replaces deleted `tickers` dict)
        tickers = {name: ticker for name, (ticker, _) in tickers_mapping.items()}
        import requests
        import yfinance as yf
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        def fetch_screener_index(company_id):
            if not company_id:
                return None
            url = f"https://www.screener.in/api/company/{company_id}/chart/?q=Price-DMA50-DMA200&days=365"
            try:
                r = requests.get(url, headers=headers, timeout=10)
                if r.status_code == 200:
                    data = r.json()
                    price_values = None
                    if "datasets" in data:
                        for ds in data["datasets"]:
                            if ds.get("metric") == "Price":
                                price_values = ds.get("values", [])
                                break
                    if not price_values and "chart" in data:
                        price_values = [[pt[0], pt[1]] for pt in data["chart"] if len(pt) >= 2]
                    if price_values:
                        df = pd.DataFrame(price_values, columns=["Date", "Close"])
                        df["Date"] = pd.to_datetime(df["Date"])
                        df.set_index("Date", inplace=True)
                        return df["Close"]
            except Exception as e:
                import logging
                logging.warning(f"Screener API fetch failed for {company_id}: {e}")
            return None
        def fetch_and_extend_services():
            try:
                svc_df = yf.download("^CNXSERVICE", period="1y", auto_adjust=True, progress=False)
                if svc_df.empty:
                    return None
                svc_series = svc_df["Close"].squeeze().copy()
                svc_series.index = pd.to_datetime(svc_series.index)
                if svc_series.index.tz is not None:
                    svc_series.index = svc_series.index.tz_localize(None)
                etf_df = yf.download("MOSERVICE.NS", period="1y", auto_adjust=True, progress=False)
                if etf_df.empty:
                    return svc_series
                etf_series = etf_df["Close"].squeeze().copy()
                etf_series.index = pd.to_datetime(etf_series.index)
                if etf_series.index.tz is not None:
                    etf_series.index = etf_series.index.tz_localize(None)
                last_svc_date = svc_series.index[-1]
                extra_dates = etf_series.index[etf_series.index > last_svc_date]
                if len(extra_dates) > 0:
                    last_svc_val = float(svc_series.iloc[-1])
                    etf_sub = etf_series[etf_series.index <= last_svc_date]
                    if etf_sub.empty:
                        return svc_series
                    base_etf_val = float(etf_sub.iloc[-1])
                    extended_svc = svc_series.to_dict()
                    for dt in extra_dates:
                        etf_val = float(etf_series.loc[dt])
                        pct_change = (etf_val / base_etf_val)
                        extended_svc[dt] = last_svc_val * pct_change
                    res_series = pd.Series(extended_svc).sort_index()
                    res_series.name = "^CNXSERVICE"
                    return res_series
                return svc_series
            except Exception:
                try:
                    df = yf.download("^CNXSERVICE", period="1y", auto_adjust=True, progress=False)
                    if not df.empty:
                        res = df["Close"].squeeze()
                        res.index = pd.to_datetime(res.index)
                        if res.index.tz is not None:
                            res.index = res.index.tz_localize(None)
                        return res
                except:
                    pass
            return None
        import concurrent.futures
        def fetch_single_index_data(name, ticker, company_id):
            series = None
            if company_id:
                series = fetch_screener_index(company_id)
            if series is None:
                if name == "Nifty Services":
                    series = fetch_and_extend_services()
                else:
                    try:
                        sdf = yf.download(ticker, period="1y", auto_adjust=True, progress=False)
                        if not sdf.empty:
                            if isinstance(sdf.columns, pd.MultiIndex):
                                series = sdf["Close"].copy()
                        else:
                            series = sdf["Close"].copy()
                        series = series.squeeze()
                        series.index = pd.to_datetime(series.index)
                        if series.index.tz is not None:
                                series.index = series.index.tz_localize(None)
                    except Exception:
                        pass
            if series is not None:
                series.name = ticker
            return series
        try:
            dfs = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                future_to_name = {
                    executor.submit(fetch_single_index_data, name, ticker, company_id): name
                    for name, (ticker, company_id) in tickers_mapping.items()
                }
                for future in concurrent.futures.as_completed(future_to_name):
                    try:
                        series = future.result()
                        if series is not None:
                            dfs.append(series)
                    except Exception:
                        pass
            if dfs:
                df = pd.concat(dfs, axis=1)
                df = df.ffill().bfill()
                return df, tickers
            return pd.DataFrame(), tickers
        except Exception:
            return pd.DataFrame(), tickers
    @st.cache_data(ttl=3600)
    def _load_mf_data():
        mf_tickers_mapping = {name: ticker for name, ticker, color in mutual_funds}
        try:
            dfs = []
            for name, ticker in mf_tickers_mapping.items():
                try:
                    df = yf.download(ticker, period="2y", auto_adjust=True, progress=False)
                    if not df.empty:
                        if isinstance(df.columns, pd.MultiIndex):
                            series = df["Close"].copy()
                        else:
                            series = df["Close"].copy()
                        series = series.squeeze()
                        series.index = pd.to_datetime(series.index)
                        if series.index.tz is not None:
                                series.index = series.index.tz_localize(None)
                        series.name = ticker
                        dfs.append(series)
                except Exception:
                    pass
            if dfs:
                df = pd.concat(dfs, axis=1)
                df = df.ffill().bfill()
                return df
            return pd.DataFrame()
        except Exception:
            return pd.DataFrame()
    sec_df, sec_tickers = _load_sectoral_thematic_data()
    
    if not sec_df.empty:
        sectoral_indices = [
            ("Nifty Bank", "^NSEBANK", "#3b82f6"),
            ("Nifty Private Bank", "NIFTY_PVT_BANK.NS", "#60a5fa"),
            ("Nifty PSU Bank", "^CNXPSUBANK", "#06b6d4"),
            ("Nifty Financial Services", "NIFTY_FIN_SERVICE.NS", "#6366f1"),
            ("Nifty IT", "^CNXIT", "#10b981"),
            ("Nifty Auto", "^CNXAUTO", "#f59e0b"),
            ("Nifty FMCG", "^CNXFMCG", "#ec4899"),
            ("Nifty Pharma", "^CNXPHARMA", "#14b8a6"),
            ("Nifty Metal", "^CNXMETAL", "#84cc16"),
            ("Nifty Realty", "^CNXREALTY", "#ef4444"),
            ("Nifty Energy", "^CNXENERGY", "#f97316"),
            ("Nifty Media", "^CNXMEDIA", "#8b5cf6")
        ]
        
        thematic_indices = [
            ("Nifty Infrastructure", "^CNXINFRA", "#f59e0b"),
            ("Nifty Consumption", "^CNXCONSUM", "#d946ef"),
            ("Nifty Services", "^CNXSERVICE", "#22c55e"),
            ("Nifty MNC", "^CNXMNC", "#f43f5e"),
            ("Nifty PSE", "^CNXPSE", "#0284c7"),
        ]
        
        mutual_funds = [
            ("Quant Small Cap Fund", "0P0000XW4J.BO", "#f59e0b"),
            ("HDFC Small Cap Fund", "0P0000XVAA.BO", "#06b6d4"),
            ("SBI Small Cap Fund", "0P0000XW1A.BO", "#10b981"),
            ("HDFC Mid-Cap Opportunities", "0P0000XW8F.BO", "#ec4899"),
        ]
        
        def _compute_returns(df, ticker):
            if df.empty or ticker not in df.columns:
                return {"price": 0.0, "1d": 0.0, "1w": 0.0, "1m": 0.0, "3m": 0.0, "6m": 0.0, "9m": 0.0, "1y": 0.0}
            series = df[ticker].dropna()
            if len(series) < 2:
                return {"price": 0.0, "1d": 0.0, "1w": 0.0, "1m": 0.0, "3m": 0.0, "6m": 0.0, "9m": 0.0, "1y": 0.0}
            
            # Ensure Date index is DatetimeIndex
            series.index = pd.to_datetime(series.index)
            series = series.sort_index()
            
            last_date = series.index[-1]
            last_val = float(series.iloc[-1])
            
            def get_pct_change(delta_days):
                target_date = last_date - pd.Timedelta(days=delta_days)
                sub = series[series.index <= target_date]
                if not sub.empty:
                    prev_val = float(sub.iloc[-1])
                    return ((last_val / prev_val) - 1.0) * 100.0 if prev_val > 0 else 0.0
                else:
                    prev_val = float(series.iloc[0])
                    return ((last_val / prev_val) - 1.0) * 100.0 if prev_val > 0 else 0.0
                    
            return {
                "price": last_val,
                "1d": ((last_val / float(series.iloc[-2])) - 1.0) * 100.0 if len(series) >= 2 else 0.0,
                "1w": get_pct_change(7),
                "1m": get_pct_change(30),
                "3m": get_pct_change(91),
                "6m": get_pct_change(182),
                "9m": get_pct_change(273),
                "1y": get_pct_change(365)
            }
            
        def _generate_heatmap_table(title, items, data_df):
            html = []
            html.append(f"""
            <div class="glass-card" style="padding: 0; overflow: hidden; border-radius: 12px; margin-bottom: 20px;">
            <div style="background: rgba(30, 41, 59, 0.9); padding: 10px 14px; border-bottom: 1px solid rgba(255,255,255,0.08); font-family: 'Outfit'; font-weight: 700; color: #f1f5f9; font-size: 0.95rem;">
                {title}
            </div>
            <table class="premium-table">
            <thead>
            <tr style="background: rgba(15, 23, 42, 0.4); border-bottom: 1px solid rgba(255,255,255,0.08);">
            <th style="width: 25%; padding: 8px 10px; text-align: left; color: #94a3b8; font-weight: 600;">Index</th>
            <th style="width: 14%; padding: 8px 10px; text-align: right; color: #94a3b8; font-weight: 600;">Price</th>
            <th style="width: 8.5%; padding: 8px 10px; text-align: right; color: #94a3b8; font-weight: 600;">1D</th>
            <th style="width: 8.5%; padding: 8px 10px; text-align: right; color: #94a3b8; font-weight: 600;">1W</th>
            <th style="width: 8.5%; padding: 8px 10px; text-align: right; color: #94a3b8; font-weight: 600;">1M</th>
            <th style="width: 8.5%; padding: 8px 10px; text-align: right; color: #94a3b8; font-weight: 600;">3M</th>
            <th style="width: 8.5%; padding: 8px 10px; text-align: right; color: #94a3b8; font-weight: 600;">6M</th>
            <th style="width: 8.5%; padding: 8px 10px; text-align: right; color: #94a3b8; font-weight: 600;">9M</th>
            <th style="width: 10%; padding: 8px 10px; text-align: right; color: #94a3b8; font-weight: 600;">1Y</th>
            </tr>
            </thead>
            <tbody>""")
            
            def cell_style(val):
                if val > 5.0:
                    return "background: rgba(16, 185, 129, 0.22); color: #34d399; font-weight: 600;"
                elif val > 0.0:
                    return "background: rgba(16, 185, 129, 0.06); color: #a7f3d0;"
                elif val < -5.0:
                    return "background: rgba(239, 68, 68, 0.22); color: #f87171; font-weight: 600;"
                elif val < 0.0:
                    return "background: rgba(239, 68, 68, 0.06); color: #fecaca;"
                return "color: #94a3b8;"
            for name, ticker, color in items:
                ret = _compute_returns(data_df, ticker)
                p_str = f"₹{ret['price']:,.1f}" if ret['price'] > 0 else "—"
                html.append(f"""
                <tr style="border-bottom: 1px solid rgba(255,255,255,0.03);">
                <td style="padding: 8px 10px; color: {color}; font-weight: 700; font-family: 'Outfit';">{name}</td>
                <td style="padding: 8px 10px; text-align: right; color: #f1f5f9; font-family: monospace; font-weight: 600;">{p_str}</td>
                <td style="padding: 8px 10px; text-align: right; {cell_style(ret['1d'])} font-family: monospace;">{ret['1d']:.1f}%</td>
                <td style="padding: 8px 10px; text-align: right; {cell_style(ret['1w'])} font-family: monospace;">{ret['1w']:.1f}%</td>
                <td style="padding: 8px 10px; text-align: right; {cell_style(ret['1m'])} font-family: monospace;">{ret['1m']:.1f}%</td>
                <td style="padding: 8px 10px; text-align: right; {cell_style(ret['3m'])} font-family: monospace;">{ret['3m']:.1f}%</td>
                <td style="padding: 8px 10px; text-align: right; {cell_style(ret['6m'])} font-family: monospace;">{ret['6m']:.1f}%</td>
                <td style="padding: 8px 10px; text-align: right; {cell_style(ret['9m'])} font-family: monospace;">{ret['9m']:.1f}%</td>
                <td style="padding: 8px 10px; text-align: right; {cell_style(ret['1y'])} font-family: monospace; font-weight: 600;">{ret['1y']:.1f}%</td>
                </tr>""")
            html.append("</tbody></table></div>")
            return "".join(html).replace("\n", "")
        col_sec_l, col_sec_r = st.columns([1, 1])
        with col_sec_l:
            st.markdown(_generate_heatmap_table("Sectoral Indices Performance Heatmap", sectoral_indices, sec_df), unsafe_allow_html=True)
        with col_sec_r:
            st.markdown(_generate_heatmap_table("Thematic & Special Indices Performance Heatmap", thematic_indices, sec_df), unsafe_allow_html=True)
            
        # ── Combined Large Chart ──
        st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
        st.markdown("### 📈 Sectoral & Thematic Comparative Trend (Base 100)")
        
        sec_tf_cols = st.columns([2, 5])
        with sec_tf_cols[0]:
            sec_timeframe = st.radio(
                "Select Performance Window",
                options=["1W", "1M", "3M", "6M", "YTD", "1Y", "MAX"],
                index=5,  # default to 1Y
                horizontal=True,
                key="sec_chart_timeframe"
            )
            
        # Determine start date based on selection and selected_date
        sec_sel_dt = pd.to_datetime(selected_date)
        if sec_timeframe == "1W":
            sec_start_date = sec_sel_dt - pd.Timedelta(days=7)
        elif sec_timeframe == "1M":
            sec_start_date = sec_sel_dt - pd.Timedelta(days=30)
        elif sec_timeframe == "3M":
            sec_start_date = sec_sel_dt - pd.Timedelta(days=91)
        elif sec_timeframe == "6M":
            sec_start_date = sec_sel_dt - pd.Timedelta(days=182)
        elif sec_timeframe == "YTD":
            sec_start_date = pd.to_datetime(f"{sec_sel_dt.year}-01-01")
        elif sec_timeframe == "1Y":
            sec_start_date = sec_sel_dt - pd.Timedelta(days=365)
        else:
            sec_start_date = None
            
        df_sec_sub = sec_df.copy()
        df_sec_sub.index = pd.to_datetime(df_sec_sub.index)
        df_sec_sub = df_sec_sub.apply(pd.to_numeric, errors='coerce')
        df_sec_sub = df_sec_sub.dropna(how="all").sort_index()
        
        if sec_start_date is not None:
            df_sec_sub = df_sec_sub[df_sec_sub.index >= sec_start_date]
        
        # Compute RS vs Nifty 50 for each sector (as percentage)
        rs_sec_dict = {}
        ticker_to_name = {v: k for k, v in sec_tickers.items()}
        nifty_sec_close = pd.DataFrame()
        try:
            ns = _h_get_nifty_2y()
            if isinstance(ns.columns, pd.MultiIndex):
                nifty_sec_close = ns["Close"].squeeze().dropna()
            else:
                nifty_sec_close = ns["Close"].squeeze().dropna() if "Close" in ns.columns else ns.squeeze().dropna()
            nifty_sec_close = pd.to_numeric(pd.Series(nifty_sec_close), errors='coerce')
            nifty_sec_close.index = pd.to_datetime(nifty_sec_close.index)
            if sec_start_date is not None:
                nifty_sec_close = nifty_sec_close[nifty_sec_close.index >= sec_start_date]
            
            for name, ticker, color in sectoral_indices + thematic_indices:
                if ticker in df_sec_sub.columns:
                    series = df_sec_sub[ticker].dropna()
                    if not series.empty and not nifty_sec_close.empty:
                        sec_rebased = series / series.iloc[0]
                        nifty_aligned = nifty_sec_close.reindex(series.index, method='ffill')
                        if not nifty_aligned.empty:
                            nifty_rebased = nifty_aligned / nifty_aligned.iloc[0]
                            sec_rs = ((sec_rebased / nifty_rebased) - 1.0) * 100.0
                            rs_sec_dict[name] = sec_rs.dropna()
        except Exception:
            pass
        # ── RS CHART — Smart Ranking (like Strategy Indices) ──
        # Compute current RS values for smart defaults & annotations
        _sec_rankings = []
        if rs_sec_dict:
            for name in rs_sec_dict:
                _last_rs = rs_sec_dict[name].dropna()
                if not _last_rs.empty:
                    _sec_rankings.append((name, _last_rs.iloc[-1]))
            _sec_rankings.sort(key=lambda x: x[1], reverse=True)
        
        # Smart defaults: top 5 + bottom 2 = 7 sectors max
        _top_sec = [r[0] for r in _sec_rankings[:5]]
        _bot_sec = [r[0] for r in _sec_rankings[-2:]]
        _smart_sec_defaults = list(dict.fromkeys(_top_sec + _bot_sec))
        
        select_all_sec = st.checkbox("Select All Sectors/Themes", value=False, key="select_all_sec")
        all_sec_names = [name for name, _, _ in sectoral_indices + thematic_indices]
        
        if select_all_sec:
            selected_sec_names = all_sec_names
        else:
            selected_sec_names = st.multiselect(
                "Filter Sectors/Themes to Display",
                options=all_sec_names,
                default=[n for n in _smart_sec_defaults if n in all_sec_names],
                key="selected_sectors_rs"
            )
        
        fig_sec = go.Figure()
        has_sec_data = False
        
        if rs_sec_dict and _sec_rankings:
            # Build rank lookup
            _rank_map = {name: i+1 for i, (name, _) in enumerate(_sec_rankings)}
            _rs_map = dict(_sec_rankings)
            _top3 = set(r[0] for r in _sec_rankings[:3])
            _bot3 = set(r[0] for r in _sec_rankings[-3:])
            
            for name, ticker, color in sectoral_indices + thematic_indices:
                if name in selected_sec_names and name in rs_sec_dict and not rs_sec_dict[name].empty:
                    series_rs = rs_sec_dict[name]
                    rolling_slope = series_rs.diff(20).fillna(0.0)
                    _rank = _rank_map.get(name, 99)
                    
                    # Visual hierarchy: top 3 = thick solid, bottom 3 = dashed, middle = thin dot
                    if name in _top3:
                        _lw, _ld, _lo = 3.0, 'solid', 1.0
                    elif name in _bot3:
                        _lw, _ld, _lo = 1.5, 'dash', 0.7
                    else:
                        _lw, _ld, _lo = 1.2, 'dot', 0.5
                    
                    fig_sec.add_trace(go.Scatter(
                        x=series_rs.index, y=series_rs.values,
                        name=f"#{_rank} {name}",
                        line=dict(color=color, width=_lw, dash=_ld, shape='spline'),
                        opacity=_lo,
                        customdata=np.stack((rolling_slope.values, _rank * np.ones(len(series_rs))), axis=-1),
                        hovertemplate=(
                            f"<span style='color:{color}; font-weight:bold; font-size:13px;'>#{_rank} {name}</span><br>"
                            "RS vs Nifty: <b>%{y:+.2f}%</b><br>"
                            "20d Slope: <b style='color:%{{#34d399 if customdata[0]>=0 else #f87171}};'>%{{customdata[0]:+.2f}}%</b><br>"
                            "Current RS: <b>%{{y:.1f}}%</b><extra></extra>"
                        ),
                        showlegend=True
                    ))
                    has_sec_data = True
            
            # Top / Bottom annotations
            if _sec_rankings:
                _top_name, _top_val = _sec_rankings[0]
                _bot_name, _bot_val = _sec_rankings[-1]
                fig_sec.add_annotation(
                    x=0.02, y=0.98, xref="paper", yref="paper",
                    text=f"🏆 Top: {_top_name} ({_top_val:+.1f}%)<br>⚠️ Bottom: {_bot_name} ({_bot_val:+.1f}%)",
                    showarrow=False, font=dict(size=11, color="#cbd5e1", family="Inter"),
                    align="left", bgcolor="rgba(15,23,42,0.75)", bordercolor="rgba(255,255,255,0.1)",
                    borderwidth=1, borderpad=6
                )
                # Rank boxes on right side for top 3
                for _ri, (_rn, _rv) in enumerate(_sec_rankings[:3]):
                    fig_sec.add_annotation(
                        x=0.98, y=0.85 - _ri * 0.08, xref="paper", yref="paper",
                        text=f"<b>#{_ri+1}</b> {_rn}<br><span style='font-size:10px;'>{_rv:+.1f}%</span>",
                        showarrow=False, font=dict(size=10, color="#f1f5f9", family="Inter"),
                        align="left", bgcolor="rgba(15,23,42,0.7)", bordercolor="rgba(255,255,255,0.08)",
                        borderwidth=1, borderpad=5
                    )
        
        if not has_sec_data:
            st.info("No sector RS data available for the selected period.")
        else:
            fig_sec.update_layout(
                template="plotly_dark",
                title=dict(text="RS % (vs Nifty 50)", font=dict(size=13, color="#94a3b8", family="Inter")),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                hovermode="x unified",
                hoverlabel=dict(bgcolor="rgba(15,23,42,0.92)", font_size=12, font_family="Inter"),
                margin=dict(l=60, r=200, t=40, b=40),
                height=550,
                font_family="Inter",
                legend=dict(
                    orientation="v", yanchor="top", y=0.98, xanchor="left", x=1.02,
                    font=dict(size=9, color="#94a3b8"), bgcolor="rgba(15,23,42,0.4)",
                    bordercolor="rgba(255,255,255,0.05)", borderwidth=1
                ),
                xaxis=dict(
                    showgrid=True, gridcolor="rgba(255,255,255,0.04)",
                    title="", tickfont=dict(size=10, color="#64748b", family="Inter"),
                    rangeslider=dict(visible=True, thickness=0.05),
                    rangeselector=dict(font_size=10)
                ),
                yaxis=dict(
                    showgrid=True, gridcolor="rgba(255,255,255,0.06)",
                    title="", tickfont=dict(size=10, color="#64748b", family="Inter"),
                    zeroline=True, zerolinecolor="rgba(255,255,255,0.1)",
                    zerolinewidth=1.5
                )
            )
            fig_sec.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.1)", line_width=1)
            st.plotly_chart(fig_sec, use_container_width=True)
    st.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True)
    # ── 5. STRATEGY INDICES PERFORMANCE & RS ──
    st.subheader("🎯 Strategy Indices Performance Monitor")
    # ── STRATEGY INDICES RS CHART ──
    # Screener.in company IDs for Nifty Strategy Indices
    STRATEGY_SCREENER_IDS = {
        "Nifty100 Low Vol 30":          1274556,
        "Nifty200 Momentum 30":         1274792,
        "Nifty200 Alpha 30":            1284754,
        "Nifty100 Alpha 30":            1285243,
        "Nifty Alpha 50":               1272880,
        "Nifty Alpha Low Vol 30":       1284493,
        "Nifty Alpha Quality LV 30":    1285064,
        "Nifty Alpha Quality Value LV 30": 1285065,
        "Nifty Dividend Opp 50":        1272806,
        "Nifty Growth Sectors 15":      1272898,
        "Nifty High Beta 50":           1272879,
        "Nifty Low Vol 50":             1272878,
        "Nifty100 Quality 30":          1274022,
        "Nifty Midcap150 Mom 50":       1284757,
        "Nifty500 Flexicap Q 30":       1285957,
        "Nifty500 Low Vol 50":          1285721,
        "Nifty500 Momentum 50":         1285162,
        "Nifty500 Quality 50":          1285496,
        "Nifty500 Multi MQVLv 50":      1285497,
        "Nifty Midcap150 Q 50":         1284383,
        "Nifty Smallcap250 Q 50":       1285254,
        "Nifty Total Mkt Mom Q 50":     1285963,
        "Nifty500 Multicap Mom Q 50":   1285258,
        "Nifty MidSmall400 Mom Q 100":  1285163,
        "Nifty Smallcap250 Mom Q 100":  1285164,
        "Nifty Quality LV 30":          1285060,
        "Nifty50 Value 20":             1272895,
        "Nifty200 Value 30":            1285256,
        "Nifty500 Value 50":            1285062,
        "Nifty200 Quality 30":          1275141,
    }
    strategy_tickers = [
        ("Nifty100 Low Vol 30",         "#6366f1"),
        ("Nifty200 Momentum 30",        "#10b981"),
        ("Nifty200 Alpha 30",           "#a855f7"),
        ("Nifty100 Alpha 30",           "#3b82f6"),
        ("Nifty Alpha 50",              "#f59e0b"),
        ("Nifty Alpha Low Vol 30",      "#06b6d4"),
        ("Nifty Alpha Quality LV 30",   "#ec4899"),
        ("Nifty Alpha Quality Value LV 30", "#8b5cf6"),
        ("Nifty Dividend Opp 50",       "#84cc16"),
        ("Nifty Growth Sectors 15",     "#f97316"),
        ("Nifty High Beta 50",          "#ef4444"),
        ("Nifty Low Vol 50",            "#0ea5e9"),
        ("Nifty100 Quality 30",         "#14b8a6"),
        ("Nifty Midcap150 Mom 50",      "#f472b6"),
        ("Nifty500 Flexicap Q 30",      "#d946ef"),
        ("Nifty500 Low Vol 50",         "#38bdf8"),
        ("Nifty500 Momentum 50",        "#22c55e"),
        ("Nifty500 Quality 50",         "#eab308"),
        ("Nifty500 Multi MQVLv 50",     "#a3e635"),
        ("Nifty Midcap150 Q 50",        "#2dd4bf"),
        ("Nifty Smallcap250 Q 50",      "#fb923c"),
        ("Nifty Total Mkt Mom Q 50",    "#c084fc"),
        ("Nifty500 Multicap Mom Q 50",  "#fde047"),
        ("Nifty MidSmall400 Mom Q 100", "#67e8f9"),
        ("Nifty Smallcap250 Mom Q 100", "#fca5a5"),
        ("Nifty Quality LV 30",         "#a78bfa"),
        ("Nifty50 Value 20",            "#34d399"),
        ("Nifty200 Value 30",           "#fbbf24"),
        ("Nifty500 Value 50",           "#818cf8"),
        ("Nifty200 Quality 30",         "#fb7185"),
    ]
    strat_rs_dict = {}
    try:
        import requests as strat_req
        import json
        import time
        screener_headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Referer": "https://www.screener.in/",
        }
        # Get Nifty 50 from idx_histories for RS computation (resample to weekly)
        nifty_strat_hist = idx_histories.get("Nifty 50", pd.DataFrame())
        if not nifty_strat_hist.empty and "Date" in nifty_strat_hist.columns and "Close" in nifty_strat_hist.columns:
            nifty_strat_s = nifty_strat_hist.set_index("Date")["Close"].sort_index().resample("W-FRI").last().dropna()
            nifty_strat_ratio = nifty_strat_s / nifty_strat_s.shift(26)  # ~6 months in weeks
        else:
            nifty_strat_ratio = None
        strat_df_list = []
        for name, _ in strategy_tickers:
            try:
                cid = STRATEGY_SCREENER_IDS.get(name)
                if cid is None:
                    continue
                cache_file = os.path.join(cache_dir, f"strategy_{cid}_history.json")
                use_cache = False
                prices = None
                # Check if cache exists and is fresh (less than 4 hours old)
                if os.path.exists(cache_file):
                    file_age = time.time() - os.path.getmtime(cache_file)
                    if file_age < 14400:  # 4 hours
                        try:
                            with open(cache_file, "r") as f:
                                prices = json.load(f)
                            use_cache = True
                        except Exception:
                            pass
                if not use_cache:
                    url = f"https://www.screener.in/api/company/{cid}/chart/?q=Price-DMA50-DMA200&days=500"
                    try:
                        r = strat_req.get(url, headers=screener_headers, timeout=10)
                        if r.status_code == 200:
                            data = r.json()
                            if "datasets" in data:
                                for ds in data["datasets"]:
                                    if ds.get("metric") == "Price":
                                        prices = ds.get("values", [])
                                        # Save to cache
                                        with open(cache_file, "w") as f:
                                            json.dump(prices, f)
                                        break
                    except Exception:
                        # Fallback to cache on network error
                        if os.path.exists(cache_file):
                            try:
                                with open(cache_file, "r") as f:
                                    prices = json.load(f)
                            except Exception:
                                pass
                else:
                    # Fallback to cache on error status codes (e.g. 429)
                    if os.path.exists(cache_file):
                        with open(cache_file, "r") as f:
                            prices = json.load(f)
                if not prices:
                    continue
                pts = {}
                for date_str, val_str in prices:
                    try:
                        pts[pd.Timestamp(date_str)] = float(val_str)
                    except Exception:
                        pass
                if len(pts) > 0:
                    series = pd.Series(pts).sort_index()
                    series.name = name
                    strat_df_list.append(series)
            except Exception:
                pass
    except Exception as e:
        st.error(f"Error loading strategy indices: {e}")
    if 'strat_df_list' in locals() and strat_df_list:
        strat_df = pd.concat(strat_df_list, axis=1)
        strat_df = strat_df.ffill().bfill()
        strat_items = [(n, n, c) for n, c in strategy_tickers]
        st.markdown(_generate_heatmap_table("Strategy Indices Performance Heatmap", strat_items, strat_df), unsafe_allow_html=True)
        
    st.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True)
    st.subheader("📈 Strategy Indices — RS Heatmap")
    # Same timeframe selectors from the main chart
    str_tf = st.radio(
        "Analysis Window",
        options=["1M", "3M", "6M", "YTD", "1Y", "MAX"],
        index=4, horizontal=True, key="strat_tf"
    )
    str_sel_dt = pd.to_datetime(selected_date)
    if str_tf == "1M":      str_start = str_sel_dt - pd.Timedelta(days=30)
    elif str_tf == "3M":    str_start = str_sel_dt - pd.Timedelta(days=91)
    elif str_tf == "6M":    str_start = str_sel_dt - pd.Timedelta(days=182)
    elif str_tf == "YTD":   str_start = pd.to_datetime(f"{str_sel_dt.year}-01-01")
    elif str_tf == "1Y":    str_start = str_sel_dt - pd.Timedelta(days=365)
    else:                   str_start = None
    # ── STRATEGY RS CHART — Smart Defaults with Ranking ──
    # Re-calculate strat_rs_dict based on selected timeframe
    strat_rs_dict = {}
    if 'strat_df' in locals() and not strat_df.empty:
        df_strat_sub = strat_df.copy()
        if str_start is not None:
            df_strat_sub = df_strat_sub[df_strat_sub.index >= str_start]
        nifty_strat_close = pd.DataFrame()
        try:
            ns = _h_get_nifty_2y()
            if isinstance(ns.columns, pd.MultiIndex):
                nifty_strat_close = ns["Close"].squeeze().dropna()
            else:
                nifty_strat_close = ns["Close"].squeeze().dropna() if "Close" in ns.columns else ns.squeeze().dropna()
            nifty_strat_close = pd.to_numeric(pd.Series(nifty_strat_close), errors='coerce')
            nifty_strat_close.index = pd.to_datetime(nifty_strat_close.index)
            if str_start is not None:
                nifty_strat_close = nifty_strat_close[nifty_strat_close.index >= str_start]
        except Exception:
            pass
        for name, color in strategy_tickers:
            if name in df_strat_sub.columns:
                series = df_strat_sub[name].dropna()
                if not series.empty and not nifty_strat_close.empty:
                    s_rebased = series / series.iloc[0]
                    n_aligned = nifty_strat_close.reindex(series.index, method='ffill')
                    if not n_aligned.empty:
                        n_rebased = n_aligned / n_aligned.iloc[0]
                        sec_rs = ((s_rebased / n_rebased) - 1.0) * 100.0
                        strat_rs_dict[name] = sec_rs.dropna()
    
    # Compute current RS values to rank performers
    _strat_rankings = []
    for name, color in strategy_tickers:
        if name in strat_rs_dict and not strat_rs_dict[name].empty:
            _srs = strat_rs_dict[name]
            if len(_srs) > 0:
                _cur_rs = float(_srs.iloc[-1])
                _1m_chg = float(_srs.iloc[-1] - _srs.iloc[-min(22, len(_srs))]) if len(_srs) >= 2 else 0
                _strat_rankings.append((name, color, _cur_rs, _1m_chg))
    _strat_rankings.sort(key=lambda x: x[2], reverse=True)
    
    # Default: show top 5 + bottom 3 performers (8 total = readable)
    _top_n = [r[0] for r in _strat_rankings[:5]]
    _bot_n = [r[0] for r in _strat_rankings[-3:]]
    _smart_defaults = list(dict.fromkeys(_top_n + _bot_n))
    
    all_strat_names = [name for name, _ in strategy_tickers]
    select_all_strat = st.checkbox("Select All Strategy Indices", value=False, key="select_all_strat")
    if select_all_strat:
        selected_strat_names = all_strat_names
    else:
        selected_strat_names = st.multiselect(
            "Filter Strategy Indices to Display",
            options=all_strat_names,
            default=_smart_defaults,
            key="selected_strategies_rs"
        )
    
    # Rank label helper
    _rank_map = {r[0]: (i+1, r[2], r[3]) for i, r in enumerate(_strat_rankings)}
    
    fig_strat = go.Figure()
    has_strat = False
    _opacity_map = {name: 0.95 if i < 5 else (0.6 if i < len(_strat_rankings) - 3 else 0.95) for i, (name, *_) in enumerate(_strat_rankings)}
    
    for name, color in strategy_tickers:
        if name in selected_strat_names and name in strat_rs_dict and not strat_rs_dict[name].empty:
            srs = strat_rs_dict[name]
            if not srs.empty:
                rolling_slope = srs.diff(20).fillna(0.0)
                _rank, _cur_rs, _1m_chg = _rank_map.get(name, (99, 0, 0))
                _lw = 3.0 if _rank <= 3 or _rank > len(_strat_rankings) - 3 else 1.2
                _op = 0.95 if _rank <= 5 else (0.4 if _rank <= len(_strat_rankings) - 4 else 0.95)
                _dash = "solid" if _rank <= 3 else ("dash" if _rank > len(_strat_rankings) - 3 else "dot")
                _display_name = f"#{_rank} {name}"
                fig_strat.add_trace(go.Scatter(
                    x=srs.index, y=srs.values,
                    name=_display_name,
                    line=dict(color=color, width=_lw, shape='spline', dash=_dash),
                    opacity=_op,
                    customdata=np.stack((rolling_slope.values, np.full_like(rolling_slope.values, _rank),
                                         np.full_like(rolling_slope.values, _cur_rs)), axis=-1),
                    hovertemplate=(
                        f"<span style='color:{color}; font-weight:bold; font-size:13px;'>#{_rank} {name}</span><br>"
                        "RS vs Nifty: <b>%{y:+.2f}%</b><br>"
                        "20D Slope: <b>%{customdata[0]:+.2f}%</b><br>"
                        "Current RS: <b>%{customdata[2]:+.2f}%</b>"
                        "<extra></extra>"
                    )
                ))
                has_strat = True
    
    if has_strat:
        # Add current RS rank annotations
        for name, _, rs_val, _ in _strat_rankings[:3]:
            if name in selected_strat_names and name in strat_rs_dict and not strat_rs_dict[name].empty:
                srs = strat_rs_dict[name]
                fig_strat.add_annotation(
                    x=srs.index[-1], y=float(srs.iloc[-1]),
                    text=f"<b>#{_rank_map[name][0]}</b>",
                    showarrow=True, arrowhead=1, arrowsize=1, arrowwidth=1.5,
                    ax=25, ay=-15, font=dict(size=10, color="#f1f5f9"),
                    bgcolor="rgba(15,23,42,0.7)", bordercolor="rgba(255,255,255,0.15)",
                    borderwidth=1, borderpad=3
                )
        
        fig_strat.update_layout(
            hovermode="closest", template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=40, r=30, t=15, b=30), height=550,
            legend=dict(orientation="v", y=0.5, x=1.02, xanchor="left",
                       bgcolor="rgba(15,23,42,0.6)", font=dict(size=9, color="#94a3b8"),
                       borderwidth=1, bordercolor="rgba(255,255,255,0.06)"),
            xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.03)", zeroline=False,
                       tickfont=dict(size=10, color="#64748b"),
                       rangeslider=dict(visible=True, thickness=0.06)),
            yaxis=dict(title="RS % (vs Nifty 50)", showgrid=True, gridcolor="rgba(255,255,255,0.03)",
                       zeroline=True, zerolinecolor="rgba(255,255,255,0.15)",
                       tickfont=dict(size=10, color="#64748b"), tickformat="+.1f",
                       title_font=dict(color="#64748b", size=11)),
            hoverlabel=dict(bgcolor="rgba(15, 23, 42, 0.95)", font=dict(size=11, color="#f1f5f9"),
                           bordercolor="rgba(255,255,255,0.1)")
        )
        fig_strat.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.25)", line_width=1.5)
        
        # Add top performer label
        if _strat_rankings:
            _top = _strat_rankings[0]
            fig_strat.add_annotation(xref="paper", yref="paper", x=0.01, y=0.98,
                text=f"🏆 Top: {_top[0]} (+{_top[2]:.1f}%)",
                showarrow=False, font=dict(size=10, color="#34d399"),
                bgcolor="rgba(15,23,42,0.5)", borderpad=4)
            if len(_strat_rankings) > 1:
                _bot = _strat_rankings[-1]
                fig_strat.add_annotation(xref="paper", yref="paper", x=0.01, y=0.92,
                    text=f"⚠️ Bottom: {_bot[0]} ({_bot[2]:.1f}%)",
                    showarrow=False, font=dict(size=10, color="#f87171"),
                    bgcolor="rgba(15,23,42,0.5)", borderpad=4)
        
        st.plotly_chart(fig_strat, use_container_width=True)
    else:
        st.info("Strategy index data not available for RS calculation (need 124+ days).")
    # ── CONSOLIDATED RS MOMENTUM TABLE (Sectors + Strategies) ──
    st.markdown("<div style='margin-top: 25px;'></div>", unsafe_allow_html=True)
    st.subheader("📊 RS Momentum Analysis — All Indices")
    st.caption("RS % = outperformance vs Nifty 50 (123-day lookback). Positive = outperforming. Direction based on 1M change.")
    all_rs_table = []
    # Add sectors
    for name, ticker, color in sectoral_indices + thematic_indices:
        if name not in rs_sec_dict or rs_sec_dict[name].empty:
            continue
        rs_s = rs_sec_dict[name]
        if len(rs_s) < 22:
            continue
        cur = float(rs_s.iloc[-1])
        w1 = float(rs_s.iloc[-6]) if len(rs_s) >= 6 else cur
        m1 = float(rs_s.iloc[-22]) if len(rs_s) >= 22 else cur
        cw = cur - w1
        cm = cur - m1
        if cm > 2.0:       dr = "🚀 Strong Up"
        elif cm > 0.5:     dr = "⬆️ Rising"
        elif cm > -0.5:    dr = "➡️ Stable"
        elif cm > -2.0:    dr = "⬇️ Declining"
        else:               dr = "🔻 Strong Down"
        all_rs_table.append(("sector", name, color, cur, cw, cm, dr))
    # Add strategies
    for name, color in strategy_tickers:
        if name not in strat_rs_dict or strat_rs_dict[name].empty:
            continue
        rs_s = strat_rs_dict[name]
        if len(rs_s) < 10:
            continue
        cur = float(rs_s.iloc[-1])
        w1 = float(rs_s.iloc[-2]) if len(rs_s) >= 2 else cur
        m1 = float(rs_s.iloc[-5]) if len(rs_s) >= 5 else cur
        cw = cur - w1
        cm = cur - m1
        if cm > 2.0:       dr = "🚀 Strong Up"
        elif cm > 0.5:     dr = "⬆️ Rising"
        elif cm > -0.5:    dr = "➡️ Stable"
        elif cm > -2.0:    dr = "⬇️ Declining"
        else:               dr = "🔻 Strong Down"
        all_rs_table.append(("strategy", name, color, cur, cw, cm, dr))
    if all_rs_table:
        # Sort: by Current RS descending
        all_rs_table.sort(key=lambda r: r[3], reverse=True)
        trs = []
        for cat, name, color, cur, cw, cm, dr in all_rs_table:
            if cat == "sector":
                badge = '<span style="display:inline-block;padding:2px 6px;border-radius:4px;font-size:0.65rem;font-weight:700;margin-right:8px;background:rgba(59,130,246,0.1);color:#60a5fa;border:1px solid rgba(59,130,246,0.2);">SECTOR</span>'
            else:
                badge = '<span style="display:inline-block;padding:2px 6px;border-radius:4px;font-size:0.65rem;font-weight:700;margin-right:8px;background:rgba(129,140,248,0.1);color:#a5b4fc;border:1px solid rgba(129,140,248,0.2);">STRATEGY</span>'
            
            if cur > 0:
                if cm > 0:
                    reason = "Outperforming Nifty 50 with accelerating momentum"
                else:
                    reason = "Outperforming Nifty 50, but momentum is slowing"
            else:
                if cm > 0:
                    reason = "Underperforming Nifty 50, but showing recovery"
                else:
                    reason = "Underperforming Nifty 50 with persistent weakness"
            trs.append(
                f"<tr style='border-bottom:1px solid rgba(255,255,255,0.05);'>"
                f"<td style='padding:10px 12px;color:{color};font-weight:700;text-align:left;'>{badge} {name}</td>"
                f"<td style='padding:10px 12px;color:{'#10b981' if cur>0 else '#ef4444'};font-weight:700;text-align:right;'>{cur:+.2f}%</td>"
                f"<td style='padding:10px 12px;color:{'#10b981' if cw>0 else '#ef4444'};text-align:right;'>{cw:+.2f}%</td>"
                f"<td style='padding:10px 12px;color:{'#10b981' if cm>0 else '#ef4444'};font-weight:600;text-align:right;'>{cm:+.2f}%</td>"
                f"<td style='padding:10px 12px;text-align:left;'>{dr}</td>"
                f"<td style='padding:10px 12px;color:#94a3b8;text-align:left;'>{reason}</td>"
                f"</tr>"
            )
        tbl = f"""
        <div style="overflow-x:auto;margin:10px 0;">
        <table class="premium-table">
        <thead>
        <tr style="background:rgba(99,102,241,0.12);border-bottom:1px solid rgba(255,255,255,0.1);">
            <th style="width: 22%; padding:10px 12px; text-align:left; color:#818cf8;">Index</th>
            <th style="width: 12%; padding:10px 12px; text-align:right; color:#818cf8;">Current RS %</th>
            <th style="width: 10%; padding:10px 12px; text-align:right; color:#818cf8;">Δ 1W</th>
            <th style="width: 10%; padding:10px 12px; text-align:right; color:#818cf8;">Δ 1M</th>
            <th style="width: 12%; padding:10px 12px; text-align:left; color:#818cf8;">Direction</th>
            <th style="width: 34%; padding:10px 12px; text-align:left; color:#818cf8;">Reason / Verdict</th>
        </tr>
        </thead>
        <tbody>{"".join(trs)}</tbody>
        </table>
        </div>
        """
        st.markdown(tbl, unsafe_allow_html=True)
    else:
        st.info("Insufficient data for RS momentum analysis.")
    # ── 7. RISK INDICATORS ── (from Market Intelligence) ──
    reg_mi = regime if 'regime' in dir() else {}
    chop_nifty  = reg_mi.get("nifty_chop", 53.3)
    chop_midsm  = reg_mi.get("midsmall_chop", 66.7)
    sec_title("🌊", "Index Choppiness Regime")
    c1, c2 = st.columns([1, 1])
    with c1:
        indices = ["Nifty 50", "MidSmall 400"]
        chop_vals = [chop_nifty, chop_midsm]
        chop_colors = ["#f87171" if v > 61.8 else "#fbbf24" if v > 50 else "#10b981" for v in chop_vals]
        chop_labels = ["Choppy" if v > 61.8 else "Sideways" if v > 50 else "Trending" for v in chop_vals]
        fig_cg = go.Figure()
        for idx, (name, val, color, label) in enumerate(zip(indices, chop_vals, chop_colors, chop_labels)):
            fig_cg.add_trace(go.Bar(name=name, x=[name], y=[val], marker_color=color,
                text=[f"{val:.1f}<br><b>{label}</b>"], textposition="inside", width=0.4))
        fig_cg.add_hrect(y0=61.8, y1=100, fillcolor="rgba(239,68,68,0.06)", layer="below", line_width=0)
        fig_cg.add_hrect(y0=38.2, y1=61.8, fillcolor="rgba(251,191,36,0.04)", layer="below", line_width=0)
        fig_cg.add_hrect(y0=0, y1=38.2, fillcolor="rgba(16,185,129,0.06)", layer="below", line_width=0)
        fig_cg.add_hline(y=61.8, line_dash="dash", line_color="#f87171", annotation_text="Choppy (61.8)")
        fig_cg.add_hline(y=38.2, line_dash="dash", line_color="#10b981", annotation_text="Trending (38.2)")
        fig_cg.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0,r=0,t=40,b=0), height=320, font_family="Inter",
            yaxis=dict(range=[0,100], gridcolor="rgba(255,255,255,0.04)"),
            xaxis=dict(showgrid=False), showlegend=False,
            title=dict(text="Choppiness Index — Regime Classification", font_size=12, font_color="#64748b"))
        st.plotly_chart(fig_cg, use_container_width=True)
    with c2:
        sec_title("💹", "Multi-Asset Trend Summary")
        eq_bull = reg_mi.get("equity_bullish", True)
        au_bull = reg_mi.get("gold_bullish", True)
        ag_bull = reg_mi.get("silver_bullish", True)
        ma1, ma2, ma3 = st.columns(3)
        for col, name, sub, bullish, color, desc in [
            (ma1, "Indian Equities", "Nifty MidSmall 400", eq_bull, "#3b82f6",
             "Mid-Small cap equity momentum. Above 150 EMA = risk-on allocation active."),
            (ma2, "MCX Gold Futures", "Inflation Hedge", au_bull, "#fbbf24",
             "Gold futures as uncorrelated hedge. Above 150 EMA = +25% commodity overlay."),
            (ma3, "MCX Silver Futures", "High-Beta Metal", ag_bull, "#f472b6",
             "Silver as industrial demand proxy. Above 150 EMA = +10% speculative overlay."),
        ]:
            sc = "#10b981" if bullish else "#ef4444"
            st_txt = "🟢 BULLISH" if bullish else "🔴 BEARISH"
            with col:
                st.markdown(f"""
                <div class="glass-card" style="border-color:{color}18;text-align:center;">
                    <div style="font-family:'Outfit';font-size:1.1rem;font-weight:700;color:{color};">{name}</div>
                    <div style="font-size:0.8rem;color:#475569;">{sub}</div>
                    <div style="font-family:'Outfit';font-size:1.6rem;font-weight:800;color:{sc};">{st_txt}</div>
                    <div style="font-size:0.78rem;color:#475569;line-height:1.5;">{desc}</div>
                </div>""", unsafe_allow_html=True)
    # ── 8. THEME ROTATION ── ──
    sec_title("🎯", "Theme/Sector Rotation")
    import glob, os
    _h_rs_tr = pd.DataFrame()
    _out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    _l6_files = sorted(glob.glob(os.path.join(_out_dir, "*", "L6_Trade_Allocations.csv")))
    if _l6_files:
        _l6_df = read_data_smart(_l6_files[-1])
        if "Symbol" in _l6_df.columns and "RS_vs_Nifty50" in _l6_df.columns:
            _h_rs_tr = _l6_df.rename(columns={"Symbol": "Stock", "RS_vs_Nifty50": "RS"})
            # If all RS values are 0 (e.g. pipeline rejected everything), fall back
            if _h_rs_tr["RS"].nunique() <= 1 and _h_rs_tr["RS"].iloc[0] == 0.0:
                _h_rs_tr = pd.DataFrame()
    if _h_rs_tr.empty:
        _h_rs_tr = _h_get_rs_df()
    # Dynamically map Sectors and Themes if they are missing
    if not _h_rs_tr.empty:
        from config import SECTORS, THEMES
        if "Sector" not in _h_rs_tr.columns:
            _h_rs_tr["Sector"] = _h_rs_tr["Stock"].apply(lambda s: SECTORS.get(s, "Other"))
        if "Theme" not in _h_rs_tr.columns:
            _h_rs_tr["Theme"] = _h_rs_tr["Stock"].apply(lambda s: THEMES.get(s, "Other"))
            
        df_valid = _h_rs_tr.dropna(subset=["Sector", "Theme", "RS"]).copy()
        df_valid = df_valid[~df_valid["Sector"].isin(["PASSIVE_CORE", "ETF", "Strategy", "Other"])]
        df_valid = df_valid[~df_valid["Theme"].isin(["PASSIVE_CORE", "ETF", "Strategy", "Other"])]
        
        # Split sectors like 'Consumer Cyclical - Auto Parts' -> 'Consumer Cyclical'
        df_valid["Broad_Sector"] = df_valid["Sector"].apply(lambda s: str(s).split(" - ")[0].strip())
        
        data = []
        # Group by Sector
        for name, g in df_valid.groupby("Broad_Sector"):
            if len(g) >= 1:
                data.append({"Group": f"🏛️ {name}", "Count": len(g), "Avg RS": g["RS"].mean()})
                
        # Group by Theme
        for name, g in df_valid.groupby("Theme"):
            if len(g) >= 1:
                data.append({"Group": f"🎯 {name}", "Count": len(g), "Avg RS": g["RS"].mean()})
    else:
        data = []
        
    if data:
        theme_df = pd.DataFrame(data).sort_values("Avg RS", ascending=False).head(10)  # Top 10
        fig = px.bar(theme_df, x="Group", y="Avg RS", color="Avg RS",
                     color_continuous_scale=["#ef4444", "#f97316", "#eab308", "#10b981", "#0ea5e9", "#6366f1", "#8b5cf6", "#d946ef"],
                     text=theme_df["Avg RS"].round(2).astype(str), height=550)
                     
        fig.add_hline(y=0.10, line_dash="dash", line_color="#ef4444", 
                      annotation_text="Min RS", annotation_font_color="#ef4444", annotation_position="top left")
        
        # Improve presentation and colors
        fig.update_layout(
            template="plotly_dark", 
            title=dict(text=f"Theme & Sector Rotation ({len(theme_df)} groups)", font=dict(size=22, color="#f8fafc", family="Outfit"), pad=dict(b=20)),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=False, title="", tickangle=-45, tickfont=dict(size=12, color="#cbd5e1", family="Inter")),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.08)", griddash="dash", title="Average RS (Relative Strength)", title_font=dict(color="#cbd5e1", family="Inter")),
            margin=dict(t=80, b=120, l=60, r=40),
            coloraxis_colorbar=dict(title="RS Score", thicknessmode="pixels", thickness=12, title_font=dict(color="#cbd5e1"), tickfont=dict(color="#cbd5e1")),
            hoverlabel=dict(bgcolor="rgba(15, 23, 42, 0.9)", font_size=13, font_family="Inter")
        )
        fig.update_traces(
            marker_line_width=0, 
            textposition="outside",
            textfont=dict(color="#f8fafc", size=13, family="Inter")
        )
        st.plotly_chart(fig, use_container_width=True)
st.markdown("<div style='margin-top: 40px;'></div>", unsafe_allow_html=True)
# ──────────────────────────────────────────────────────────────────────────────
# GLOBAL SHARED LEDGER DATA — computed once, used by tab_pm and tab_core
# ──────────────────────────────────────────────────────────────────────────────
# Initialize df_prev and prev_holdings
_df_prev_global = None
_prev_holdings_global = []
try:
    _dates_global = sorted([d for d in os.listdir(base_output_dir)
                             if os.path.isdir(os.path.join(base_output_dir, d)) and d < selected_date])
    if _dates_global:
        _prev_date_global = _dates_global[-1]
        for _fn in ["L7_MAAC_Allocations.csv"]:
            _pp = os.path.join(base_output_dir, _prev_date_global, _fn)
            if os.path.exists(_pp):
                _df_prev_global = read_data_smart(_pp)
                if "Allocation_%" in _df_prev_global.columns:
                    _prev_holdings_global = _df_prev_global[_df_prev_global["Allocation_%"] > 0]["Symbol"].tolist()
                    break
except Exception:
    pass
# Extract orders globally
_exit_map_global = {}
_buy_reason_map_global = {}
_reduce_map_global = {}
if not df_orders.empty:
    _df_exits_g = df_orders[df_orders["Action"] == "EXIT"]
    _exit_map_global = dict(zip(_df_exits_g["Symbol"], _df_exits_g["Reason"]))
    _df_buys_g = df_orders[df_orders["Action"] == "BUY"]
    _buy_reason_map_global = dict(zip(_df_buys_g["Symbol"], _df_buys_g["Reason"]))
    _df_reduces_g = df_orders[df_orders["Action"] == "REDUCE"]
    _reduce_map_global = dict(zip(_df_reduces_g["Symbol"], _df_reduces_g["Reason"]))
# Build global ledger rows
_mf_name_map_global = get_global_mf_name_map()
_ledger_rows_global = []
_core_rows_global = []
_sat_rows_global = []
_ledger_build_error = ""
# Safe defaults — overwritten inside try block if successful; prevents NameError in tab if try fails early
_all_ledger_syms_g = set()
_ledger_fail_count_g = 0
_ledger_fail_samples_g = []
try:
    from cache_manager import get_historical_data as _get_hist_g
    from monitoring_engine import calculate_rs_line as _calc_rs_g, compute_exit_score as _comp_exit_g
    from config import SECTORS as _SECTORS_G
    _nifty_df_global = _get_hist_g("NIFTY_50", end_date=selected_date)
    # BUG FIX 1: Pre-load L1_Core_Allocations for accurate RS/score data for AMFI funds
    # (calculate_rs_line returns 0.0 for funds due to short NAV history vs 252-day requirement)
    _core_alloc_lookup = {}
    try:
        _ca_path = os.path.join(OUTPUT_DIR, "L1_Core_Allocations.csv")
        if os.path.exists(_ca_path):
            _df_ca_g = pd.read_csv(_ca_path)
        else:
            _df_ca_g = load_pipeline_stage("L1_Core_Allocations")
        if _df_ca_g is not None and not _df_ca_g.empty:
            for _, _car in _df_ca_g.iterrows():
                _core_alloc_lookup[str(_car["Symbol"])] = {
                    "RS_Rating": float(_car.get("RS_Rating", 0.0)),
                    "Score": float(_car.get("Score", 0.0)),
                    "Core_Weight": float(_car.get("Core_Weight", 0.0)),
                    "Name": str(_car.get("Name", "")),
                }
    except Exception:
        pass
    _current_alloc_syms_g = set()
    if not df_maac.empty and "Symbol" in df_maac.columns and "Allocation_%" in df_maac.columns:
        _current_alloc_syms_g = set(df_maac[df_maac["Allocation_%"] > 0]["Symbol"].tolist())
    _prev_holdings_set_g = set(_prev_holdings_global)
    _exit_syms_set_g = set(_exit_map_global.keys())
    _new_buy_syms_set_g = set()
    if not df_orders.empty and "Action" in df_orders.columns and "Symbol" in df_orders.columns:
        _new_buy_syms_set_g = set(df_orders[df_orders["Action"].isin(["BUY", "VETO_ADD"])]["Symbol"].tolist())
        _veto_removes_g = set(df_orders[df_orders["Action"] == "VETO_REMOVE"]["Symbol"].tolist())
        _exit_syms_set_g.update(_veto_removes_g)
        # T2-12: Deduplicate (EXIT beats NEW BUY if both happen same day)
        _new_buy_syms_set_g = _new_buy_syms_set_g - _exit_syms_set_g
    else:
        _new_buy_syms_set_g = _current_alloc_syms_g - _prev_holdings_set_g
    _all_ledger_syms_g = (_prev_holdings_set_g | _current_alloc_syms_g | _exit_syms_set_g | _new_buy_syms_set_g)
    import datetime
    _now_ts = datetime.datetime.now()
    _ledger_fail_count_g = 0
    _ledger_fail_samples_g = []
    for _sym_g in _all_ledger_syms_g:
        try:
            _is_amfi_code_g = str(_sym_g).isdigit()
            _is_satellite_g = not _is_amfi_code_g
            
            if _sym_g in _exit_syms_set_g or (_sym_g in _prev_holdings_set_g and _sym_g not in _current_alloc_syms_g):
                _status_g = "EXIT"
            elif _sym_g in _new_buy_syms_set_g:
                _status_g = "NEW BUY"
            else:
                _status_g = "HOLD"
            _df_stock_g = _get_hist_g(_sym_g, end_date=selected_date)
            _close_price_g = _df_stock_g["Close"].iloc[-1] if _df_stock_g is not None and not _df_stock_g.empty else 0.0
            if _close_price_g == 0.0 and _status_g not in ("EXIT", "NEW BUY"):
                # No price data available — skip silently (symbol may not be cached yet)
                _ledger_fail_count_g += 1
                continue
            # BUG FIX 1: For AMFI codes (mutual funds), use pre-loaded RS_Rating from L1_Core_Allocations
            # calculate_rs_line returns 0.0 for funds (only ~65 NAV points vs 252 needed for RS)
            _is_amfi_code_g = str(_sym_g).isdigit()
            if _is_amfi_code_g and str(_sym_g) in _core_alloc_lookup:
                _core_data_g = _core_alloc_lookup[str(_sym_g)]
                _rs_val_g = _core_data_g["RS_Rating"] / 100.0  # normalize 0–100 → 0–1 for display
                _exit_score_g = max(0.0, 100.0 - _core_data_g["Score"])  # invert: lower = safer hold
            else:
                _rs_val_g, _ = _calc_rs_g(_sym_g, _df_stock_g, _nifty_df_global)
                _exit_score_g, _, _, _ = _comp_exit_g(_df_stock_g, _nifty_df_global, _rs_val_g, 0.0)
            _row_maac_g = None
            if not df_maac.empty:
                _m_g = df_maac[df_maac["Symbol"] == _sym_g]
                if not _m_g.empty:
                    _row_maac_g = _m_g.iloc[0]
            _prev_alloc_g = 0.0
            try:
                if _df_prev_global is not None and not _df_prev_global.empty:
                    _pm_g = _df_prev_global[_df_prev_global["Symbol"] == _sym_g]
                    if not _pm_g.empty:
                        _prev_alloc_g = float(_pm_g.iloc[0].get("Allocation_%", 0.0))
            except Exception:
                pass
            _sector_g = _SECTORS_G.get(_sym_g, str(_row_maac_g.get("Sector", "Diversified")) if _row_maac_g is not None else "Diversified")
            # BUG FIX 6: Prefer Bucket from MAAC row (already set by engine), fall back to heuristic
            _maac_bucket_g = str(_row_maac_g.get("Bucket", "")).upper() if _row_maac_g is not None else ""
            if "CORE" in _maac_bucket_g:
                _bucket_g = "PASSIVE_CORE"
            else:
                _is_etf_g = str(_sym_g) in _mf_name_map_global or str(_sym_g).isdigit()
                _bucket_g = "PASSIVE_CORE" if _is_etf_g else "ACTIVE_SATELLITE"
            _prev_entry_g = 0.0
            if _df_prev_global is not None and not _df_prev_global.empty:
                _pm_g = _df_prev_global[_df_prev_global["Symbol"] == _sym_g]
                if not _pm_g.empty:
                    _prev_entry_g = float(_pm_g.iloc[0].get("Entry_Price", 0.0))
                
            _entry_price_g = float(_row_maac_g.get("Entry_Price", 0.0)) if _row_maac_g is not None else 0.0
            if _entry_price_g == 0.0:
                _entry_price_g = _prev_entry_g if _prev_entry_g > 0.0 else _close_price_g
            _stop_loss_g = float(_row_maac_g.get("Stop_Loss", 0.0)) if _row_maac_g is not None else 0.0
            _stop_dist_g = float(_row_maac_g.get("Stop_Dist_%", 0.0)) if _row_maac_g is not None else 0.0
            _risk_pct_g = float(_row_maac_g.get("Risk_Per_Trade_%", 0.0)) if _row_maac_g is not None else 0.0
            _target_alloc_g = float(_row_maac_g.get("Allocation_%", 0.0)) if _row_maac_g is not None else 0.0
            if _status_g == "EXIT":
                _target_alloc_g = 0.0
            _boost_label_g = ""
            _rank_g = int(_row_maac_g.get("Final_Rank", 99)) if _row_maac_g is not None else 99
            if _status_g == "NEW BUY":
                _boost_label_g = "🆕 Fresh"
            elif _status_g == "EXIT":
                _boost_label_g = "🚫 Exit"
            else:
                _delta_g = _target_alloc_g - _prev_alloc_g
                if _delta_g > 0.3:
                    _boost_label_g = f"+{_delta_g:.1f}% 🔺"
                elif _delta_g < -0.3:
                    _boost_label_g = f"{_delta_g:.1f}% 🔻"
                else:
                    _boost_label_g = "—"
            _actual_port_val = blueprint.get("portfolio_value", portfolio_capital)
            _pos_val_g = (_target_alloc_g / 100.0) * _actual_port_val
            _qty_g = int(_pos_val_g / _entry_price_g) if _entry_price_g > 0 else 0
            _trim_flag_g = _row_maac_g.get("Trim_Flag", "") if (_row_maac_g is not None and "Trim_Flag" in _row_maac_g.index) else ""
            if _status_g == "EXIT":
                _reason_g = _exit_map_global.get(_sym_g)
                if not _reason_g:
                    if _rs_val_g <= 0.10:
                        _reason_g = f"RS Line Exit (RS: {_rs_val_g:.3f} <= 0.10)"
                else:
                        _reason_g = "Dropped from Conviction watchlist"
                _rationale_g = f"🚫 Exit: {_reason_g}"
            elif _status_g == "NEW BUY":
                _factors_passed_g = str(_row_maac_g.get("Factors_Passed", "")) if _row_maac_g is not None else ""
                _factor_score_g = _row_maac_g.get("Factor_Score", 0) if _row_maac_g is not None else 0
                _final_rank_g = _row_maac_g.get("Final_Rank", 999) if _row_maac_g is not None else 999
                if _factors_passed_g:
                    _rationale_g = f"🆕 Buy: Rank #{_final_rank_g} | Score {_factor_score_g:.1f}/100 ({_factors_passed_g})"
                else:
                    _rationale_g = f"🆕 Buy: {_buy_reason_map_global.get(_sym_g, 'Passed screening gates')}"
            else:
                if _sym_g in _reduce_map_global:
                    _rationale_g = f"⚠️ Reduce: {_reduce_map_global[_sym_g]}"
                elif _trim_flag_g == "TRIM 25%":
                    _ext_val_g = float(_row_maac_g.get("Extension_From_50DMA", 0.0)) if _row_maac_g is not None else 0.0
                    _rationale_g = f"⚠️ Trim 25% (Extension: {_ext_val_g:.1f}% > 25%)"
                elif _is_amfi_code_g and str(_sym_g) in _core_alloc_lookup:
                    # BUG FIX (Rationale): For mutual funds use momentum score, not stock exit score
                    _mf_score = _core_alloc_lookup[str(_sym_g)].get("Score", 0.0)
                    _mf_rs = _core_alloc_lookup[str(_sym_g)].get("RS_Rating", 0.0)
                    _rationale_g = f"🏛️ Core Hold — Momentum Score: {_mf_score:.1f}/100 | RS Rating: {_mf_rs:.1f}"
                else:
                    _delta_g = _target_alloc_g - _prev_alloc_g
                    if _delta_g > 0.3:
                        _rationale_g = f"🔺 Pyramided (RS Line: {_rs_val_g:.3f} > 0.60)"
                    else:
                        _rationale_g = f"Holding — Exit Score: {_exit_score_g:.0f}/55"
            # BUG FIX 8: Set clean Name and Display_Symbol for AMFI codes
            _full_name_g = ""
            if str(_sym_g) in _mf_name_map_global:
                _full_name_g = _mf_name_map_global[str(_sym_g)]
            elif str(_sym_g) in _core_alloc_lookup:
                _full_name_g = _core_alloc_lookup[str(_sym_g)].get("Name", "")
            # Display_Symbol: for funds show the clean name; for stocks show the ticker
            if _full_name_g:
                _display_sym_g = _full_name_g  # clean full name (no AMFI code prepended)
            else:
                _display_sym_g = _sym_g
            _ledger_rows_global.append({
                "status": _status_g, "Symbol": _sym_g, "Display_Symbol": _display_sym_g,
                "Name": _full_name_g,  # BUG FIX 8: full clean name key
                "Sector": _sector_g, "Bucket": _bucket_g,
                "Entry_Price": _entry_price_g, "Close": _close_price_g,
                "Stop_Loss": _stop_loss_g, "Stop_Dist_%": _stop_dist_g,
                "Risk_Per_Trade_%": _risk_pct_g, "Prev_Alloc_%": _prev_alloc_g,
                "New_Alloc_%": _target_alloc_g, "Boost_Label": _boost_label_g,
                "RS_Val": _rs_val_g, "Exit_Score": _exit_score_g,
                "Qty": _qty_g, "Pos_Value": _pos_val_g, "Trim_Flag": _trim_flag_g,
                "Rationale": _rationale_g, "Rank": _rank_g
            })
        except Exception as _e_sym:
            _ledger_fail_count_g += 1
            if len(_ledger_fail_samples_g) < 3:
                _ledger_fail_samples_g.append(f"{_sym_g}: {_e_sym}")
    # Sort and rank
    _active_rows_g = [r for r in _ledger_rows_global if r["status"] != "EXIT"]
    _exit_rows_g = [r for r in _ledger_rows_global if r["status"] == "EXIT"]
    _active_rows_g.sort(key=lambda r: r["RS_Val"], reverse=True)
    for _idx_g, _r_g in enumerate(_active_rows_g):
        _r_g["Rank"] = _idx_g + 1
    for _r_g in _exit_rows_g:
        _r_g["Rank"] = 99
    _ledger_rows_global = _active_rows_g + _exit_rows_g
    _ledger_rows_global.sort(key=lambda r: (0 if r["status"] != "EXIT" else 1, r["Rank"], r["Symbol"]))
    _core_rows_global = [r for r in _ledger_rows_global if r["Bucket"] == "PASSIVE_CORE"]
    _sat_rows_global = [r for r in _ledger_rows_global if r["Bucket"] == "ACTIVE_SATELLITE"]
except Exception as _e_ledger:
    _ledger_build_error = str(_e_ledger)
# ──────────────────────────────────────────────────────────────────────────────
# TAB 3: ACTIVE HOLDINGS AND ALLOCATIONS
# ──────────────────────────────────────────────────────────────────────────────
def _normalize_sector(sec_name):
    if not sec_name or not isinstance(sec_name, str):
        return "Diversified"
    # Split on hyphens to get the main sector name (e.g. "Industrials - Electrical Equipment" -> "Industrials")
    parts = re.split(r'\s*-\s*', sec_name)
    return parts[0].strip()
@st.cache_data(ttl=300)
def _get_active_holdings_df(_ledger_rows_json, _vams_qualified_list, _maac_active_syms_list, _mf_map_json="{}"):
    import json as _js_h
    _ledger_rows = _js_h.loads(_ledger_rows_json)
    _vams_qualified = set(_vams_qualified_list)
    _maac_active_syms = set(_maac_active_syms_list)
    _mf_map = _js_h.loads(_mf_map_json)
    _all_active_data = []
    
    for _r in _ledger_rows:
        _alloc = _r.get("New_Alloc_%", 0.0)
        _sym = _r["Symbol"]
        _is_pending_exit = (_alloc <= 0) or (_sym not in _maac_active_syms)
        
        if _r.get("Bucket") == "PASSIVE_CORE":
            _bucket = "Core Holdings"
        elif _sym in _vams_qualified:
            _bucket = "Satellite Holdings"
        else:
            _bucket = "Satellite Holdings"  # TA 4.0 — everything non-core is Satellite
            
        _entry_p = _r.get("Entry_Price", 0.0)
        _close_p = _r.get("Close", 0.0)
        _pnl_pct = ((_close_p - _entry_p) / _entry_p * 100.0) if _entry_p > 0.0 else None
        
        _all_active_data.append({
            "Symbol": _sym,
            "Display_Symbol": _mf_map.get(str(_sym), _r.get("Display_Symbol", _sym)),
            "Sector": _normalize_sector(_r.get("Sector", "Diversified")),
            "Bucket": _bucket,
            "Allocation": _alloc,
            "Current Price": _close_p,
            "Entry_Price": _entry_p,
            "Stop_Loss": _r.get("Stop_Loss", 0.0),
            "RS_Val": _r.get("RS_Val", 0.0),
            "Exit_Score": _r.get("Exit_Score", 0.0),
            "Rationale": _r.get("Rationale", "—"),
            "Qty": _r.get("Qty", 0),
            "Rank": _r.get("Rank", 99),
            "PnL_%": _pnl_pct,
            "Pending_Exit": _is_pending_exit
        })
    return _all_active_data
with tab_active:
    # Ensure _inception_label is defined in tab scope
    if "_inception_label" not in globals() and "_inception_label" not in locals():
        _inception_label = "May 27, 2026"
    st.markdown("""
    <div style="background:linear-gradient(135deg,rgba(59,130,246,0.12),rgba(139,92,246,0.06));
                border:1px solid rgba(139,92,246,0.2); border-radius:14px;
                padding:20px 24px; margin-bottom:24px;">
    <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px;">
        <div style="font-family:'Outfit';font-size:1.2rem;font-weight:700;color:#8b5cf6;">
          💼 Consolidated Active Holdings &amp; Allocations
        </div>
        <span style="background:rgba(16,185,129,0.12);color:#34d399;border:1px solid rgba(16,185,129,0.3);padding:4px 12px;border-radius:20px;font-size:0.78rem;font-weight:700;display:inline-flex;align-items:center;gap:4px;">
          🟢 SYNCED &amp; SECURE
        </span>
    </div>
    <div style="font-size:0.9rem;color:#cbd5e1;margin-top:4px;">
        <b style="color:#fbbf24;">Trend Alpha 4.0 (Core + Satellite)</b> — Beautifully organized view of your entire portfolio: <b style="color:#6366f1;">Core Bucket</b> (ETFs &amp; Mutual Funds) +
        <b style="color:#a78bfa;">Satellite Bucket</b> — VAM-B (Pure Momentum) + VAM-GQ (Quality-Gated Momentum). Allocations are regime-adjusted.
    </div>
    </div>
    """, unsafe_allow_html=True)
    
    _render_research_links(compact=True)
    
    render_unified_veto_ui("tab_active")
    
    st.caption("💰 **Role:** Live Portfolio Positions · Allocations · Risk Ledger · Exit Signals")
    
    # ── Ledger build error display ──────────────────────────────────────────
    if _ledger_build_error:
        st.error(f"🔴 **Ledger Build Failed:** `{_ledger_build_error}`. Run the pipeline (`main.py`) to regenerate data.")
    # Data completeness warning (safe — defaults pre-initialized above the try block)
    if _ledger_fail_count_g > 0:
        _fail_pct = _ledger_fail_count_g / max(len(_all_ledger_syms_g), 1) * 100
        _sample_str = "; ".join(_ledger_fail_samples_g[:2])
        st.warning(f"⚠️ **Data Incomplete:** {_ledger_fail_count_g}/{len(_all_ledger_syms_g)} symbols ({_fail_pct:.0f}%) could not load. These symbols may lack cached price data. {_sample_str}")
    
    # T2-01: Identify VAMS qualified symbols using reliable columns instead of parsing human-readable prose
    _vams_qualified = set()
    if not df_maac.empty:
        for _, _row in df_maac.iterrows():
            if _row.get("Entry_Eligible", False) == True or "VAM" in str(_row.get("Factor_Track", "")).upper():
                _vams_qualified.add(_row["Symbol"])
            else:
                _reason = str(_row.get("Rejection_Reason", "")).strip().upper()
                if "RANKED" in _reason or "PASSED" in _reason:
                    _vams_qualified.add(_row["Symbol"])
                
    _active_ledger = [r for r in _ledger_rows_global if r["status"] != "EXIT"]
    
    _maac_active_syms = set()
    if not df_maac.empty and "Symbol" in df_maac.columns and "Allocation_%" in df_maac.columns:
        _maac_active_syms = set(df_maac[df_maac["Allocation_%"] > 0]["Symbol"].tolist())
    
    _vams_qualified_list = sorted(list(_vams_qualified))
    _maac_active_syms_list = sorted(list(_maac_active_syms))
    import json as _js_h
    _active_ledger_json = _js_h.dumps(_active_ledger)
    _mf_map_json = _js_h.dumps(_mf_name_map_global)
    _all_active_data = _get_active_holdings_df(_active_ledger_json, _vams_qualified_list, _maac_active_syms_list, _mf_map_json)
        
    if not _all_active_data:
        st.info("No active holdings found in current portfolio state.")
    else:
        # ── DYNAMIC PORTFOLIO WEIGHTING UI & LOGIC ──
        st.markdown("""
        <div style="background:rgba(15,23,42,0.6);border:1px solid rgba(255,255,255,0.05);border-radius:12px;padding:16px 20px;margin-bottom:24px;">
            <div style="font-family:'Outfit';font-size:1.1rem;font-weight:700;color:#f8fafc;margin-bottom:4px;">🎛️ Portfolio Weighting</div>
            <div style="color:#94a3b8;font-size:0.85rem;margin-bottom:16px;">
                Drag slider to adjust Core / Satellite allocation split.
            </div>
        """, unsafe_allow_html=True)
        
        _core_pct = st.slider("🎯 Core Weight %", 0, 100, 65, 5,
                              format="%d%%")
        st.caption(f"Satellite: {100 - _core_pct}%  ·  Core: {_core_pct}%")
        target_core_pct = float(_core_pct)
        target_sat_pct = 100.0 - target_core_pct
        target_cash_pct = 0.0
        _do_rebalance = True
            
        st.markdown("</div>", unsafe_allow_html=True)
            
        # Normalize if they don't sum to 100%
        _total_tgt = target_core_pct + target_sat_pct + target_cash_pct
        if abs(_total_tgt - 100.0) > 0.1 and _total_tgt > 0:
            target_core_pct = (target_core_pct / _total_tgt) * 100.0
            target_sat_pct = (target_sat_pct / _total_tgt) * 100.0
            target_cash_pct = (target_cash_pct / _total_tgt) * 100.0
            st.info(f"ℹ️ Weights normalized to sum to 100%: Core {target_core_pct:.1f}%, Sat {target_sat_pct:.1f}%, Cash {target_cash_pct:.1f}%")
            
        # Create lookups from df_maac for dynamic scoring
        _maac_scores = {}
        if not df_maac.empty:
            for _, _row in df_maac.iterrows():
                _sym = _row["Symbol"]
                _fs = _row.get("Factor_Score", 0.0)
                _rs = _row.get("RS_vs_Nifty50", 0.0)
                if pd.isna(_fs): _fs = 0.0
                if pd.isna(_rs): _rs = 0.0
                _maac_scores[_sym] = {"Factor_Score": _fs, "RS_vs_Nifty50": _rs}
                
        # Split items
        _core_items = [item for item in _all_active_data if item["Bucket"] == "Core Holdings" and not item["Pending_Exit"]]
        _sat_items = [item for item in _all_active_data if item["Bucket"] == "Satellite Holdings" and not item["Pending_Exit"]]
        
        # Helper to apply proportional weights
        def _apply_proportional_weights(items, target_pct, score_key="RS_Val", use_maac_key=None):
            if not items: return
            if target_pct <= 0:
                for item in items: item["Allocation"] = 0.0
                return
                
            _scores = []
            for item in items:
                _s = 1.0 # default baseline
                if use_maac_key and item["Symbol"] in _maac_scores:
                    _s = max(0.1, float(_maac_scores[item["Symbol"]][use_maac_key]))
                elif score_key in item:
                    _s = max(0.1, float(item.get(score_key, 1.0)))
                _scores.append(abs(_s)) # proportional weights must be positive
                
            _total_score = sum(_scores)
            if _total_score <= 0:
                # equal weight fallback
                _w = target_pct / len(items)
                for item in items: item["Allocation"] = _w
            else:
                for item, score in zip(items, _scores):
                    item["Allocation"] = (score / _total_score) * target_pct
                    
        # Apply weights: Core based on returns, Satellite based on risk-adjusted momentum
        if _do_rebalance:
            _apply_proportional_weights(_core_items, target_core_pct, score_key="RS_Val", use_maac_key="RS_vs_Nifty50")
            _apply_proportional_weights(_sat_items, target_sat_pct, use_maac_key="Factor_Score")
        
        # Force pending exits to 0
        for item in _all_active_data:
            if item["Pending_Exit"]:
                item["Allocation"] = 0.0
                
        # Override the total_exposure_pct in blueprint to reflect our dynamic weighting
        blueprint["total_exposure_pct"] = target_core_pct + target_sat_pct
        df_act = pd.DataFrame(_all_active_data)
        
        # Top Metric Cards
        tot_alloc = df_act["Allocation"].sum()
            
        # T2-05 & T2-10: Over-allocation strict guard
        if tot_alloc >= 100.0:
            pass
            
        # Format returns with plus sign if positive
        # Use session state (synced from Performance Analyzer) for TA 4.0 blended metrics
        _ta4_ret = st.session_state.get("ta4_ret", global_strat_ret)
        _ta4_dd = st.session_state.get("ta4_dd", global_strat_dd)
        _ta4_sharpe = st.session_state.get("ta4_sharpe", global_strat_sharpe)
        _ta4_vol = st.session_state.get("ta4_vol", global_strat_vol)
        
        try:
            ret_val = float(_ta4_ret.replace("%", ""))
            ret_color = "#34d399" if ret_val >= 0 else "#f87171"
            ret_prefix = "+" if ret_val > 0 else ""
        except Exception:
            ret_color = "#34d399"
            ret_prefix = ""
        # UI Upgrade: 4 KPI Cards with TA 4.0 Blended Core + Satellite branding
        st.markdown(f"""
        <div style="display: flex; gap: 16px; margin: 15px 0 25px 0; flex-wrap: wrap;">
            <div class="glass-card" style="flex: 1; min-width: 200px; padding: 18px; margin-bottom: 0; border-left: 4px solid #10b981; box-shadow: 0 0 15px rgba(16, 185, 129, 0.15);">
                <div style="font-family: 'Outfit'; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 1px; color: #94a3b8; font-weight: 600;">TA 4.0 (Core+Sat) Return</div>
                <div style="font-family: 'Outfit'; font-size: 1.8rem; font-weight: 700; color: {ret_color}; margin-top: 8px;">{ret_prefix}{_ta4_ret}</div>
                <div style="font-size: 0.72rem; color: #64748b; margin-top: 4px;">Since {_inception_label} · Blended</div>
            </div>
            <div class="glass-card" style="flex: 1; min-width: 200px; padding: 18px; margin-bottom: 0; border-left: 4px solid #ef4444; box-shadow: 0 0 15px rgba(239, 68, 68, 0.15);">
                <div style="font-family: 'Outfit'; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 1px; color: #94a3b8; font-weight: 600;">TA 4.0 Max Drawdown</div>
                <div style="font-family: 'Outfit'; font-size: 1.8rem; font-weight: 700; color: #f87171; margin-top: 8px;">{_ta4_dd}</div>
                <div style="font-size: 0.72rem; color: #64748b; margin-top: 4px;">Peak-to-trough risk · Blended</div>
            </div>
            <div class="glass-card" style="flex: 1; min-width: 200px; padding: 18px; margin-bottom: 0; border-left: 4px solid #f59e0b; box-shadow: 0 0 15px rgba(245, 158, 11, 0.15);">
                <div style="font-family: 'Outfit'; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 1px; color: #94a3b8; font-weight: 600;">TA 4.0 Sharpe Ratio</div>
                <div style="font-family: 'Outfit'; font-size: 1.8rem; font-weight: 700; color: #fbbf24; margin-top: 8px;">{_ta4_sharpe}</div>
                <div style="font-size: 0.72rem; color: #64748b; margin-top: 4px;">Risk-adjusted (252d) · Blended</div>
            </div>
            <div class="glass-card" style="flex: 1; min-width: 200px; padding: 18px; margin-bottom: 0; border-left: 4px solid #3b82f6; box-shadow: 0 0 15px rgba(59, 130, 246, 0.15);">
                <div style="font-family: 'Outfit'; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 1px; color: #94a3b8; font-weight: 600;">TA 4.0 Ann. Volatility</div>
                <div style="font-family: 'Outfit'; font-size: 1.8rem; font-weight: 700; color: #38bdf8; margin-top: 8px;">{_ta4_vol}</div>
                <div style="font-size: 0.72rem; color: #64748b; margin-top: 4px;">Annual volatility · Blended</div>
            </div>
        """, unsafe_allow_html=True)
        # ── 6 KPI Metric Cards ──────────────────────────────────────────────
        col_pm1, col_pm2, col_pm3, col_pm4, col_pm5, col_pm6 = st.columns(6)
        base_h  = blueprint.get("base_heat_pct", 0.0)
        adj_h   = blueprint.get("portfolio_heat_pct", 0.0)
        corr_p  = blueprint.get("correlation_penalty_pct", 0.0)
        lev_p   = blueprint.get("leverage_penalty_pct", 0.0)
        pairs   = blueprint.get("high_corr_pairs_count", 0)
        mtf_w   = blueprint.get("mtf_leverage_pct", 0.0)
        _pv      = blueprint.get("portfolio_value", 10000000.0)
        rm      = blueprint.get("regime_risk_multiplier", 1.0)
        regime_label = blueprint.get("market_regime", "SIDEWAYS")
        regime_colors = {"BULL":"#10b981","EARLY_BULL":"#34d399","LATE_BULL":"#a3e635",
                         "SIDEWAYS":"#f59e0b","CORRECTION":"#fb923c","BEAR":"#ef4444","CRISIS":"#dc2626"}
        rm_color = regime_colors.get(regime_label, "#f59e0b")
        heat_color  = "#10b981" if adj_h < 4.0 else ("#f59e0b" if adj_h <= 6.0 else "#ef4444")
        heat_bar    = min(adj_h / 10.0, 1.0) * 100  # Scale: full bar at 10%; danger zone >6% clearly visible at 60%+
        heat_bar_c  = heat_color
        natr_heat = 0.0
        try:
            if not df_maac.empty and "NATR" in df_maac.columns and "Allocation_%" in df_maac.columns:
                _active_h = df_maac[df_maac["Allocation_%"] > 0].copy()
                natr_heat = float((_active_h["Allocation_%"] * _active_h["NATR"].fillna(3.0)).sum())
        except Exception:
            natr_heat = 0.0
        natr_color  = "#10b981" if natr_heat < 20 else ("#f59e0b" if natr_heat <= 30 else "#ef4444")
        natr_label  = "Normal" if natr_heat < 20 else ("Caution" if natr_heat <= 30 else "Reduce")
        def _kpi(label, value_str, sub, accent, bar_pct=None, icon=""):
            bar_html = ""
            if bar_pct is not None:
                bar_html = f'<div style="margin-top:8px;background:rgba(255,255,255,0.05);border-radius:4px;height:4px;overflow:hidden;"><div style="width:{bar_pct:.1f}%;background:{accent};height:100%;border-radius:4px;"></div></div>'
            # UI Upgrade: Cards styled with subtle glow effect based on accent border color
            glow_style = f"border-top:3px solid {accent}; box-shadow: 0 0 10px {accent}1a;"
            return f'<div class="glass-card" style="{glow_style}padding:14px 16px;"><div style="color:#64748b;font-size:0.72rem;font-weight:600;text-transform:uppercase;letter-spacing:0.05em;">{icon}{label}</div><div style="color:{accent};font-size:1.6rem;font-weight:700;font-family:\'Outfit\';margin:6px 0 2px 0;">{value_str}</div><div style="color:#94a3b8;font-size:0.75rem;">{sub}</div>{bar_html}</div>'
        with col_pm1:
            st.markdown(_kpi("Portfolio Heat", f"{adj_h:.2f}%",
                f"Base {base_h:.2f}% | Limit 6.00%", heat_bar_c, heat_bar, "🔥 "), unsafe_allow_html=True)
        with col_pm2:
            st.markdown(_kpi("Correlation Penalty", f"+{corr_p:.2f}%",
                f"{pairs} high-corr pairs (>0.70)", "#f59e0b", min(corr_p/2.0,1.0)*100, "🔗 "), unsafe_allow_html=True)
        with col_pm3:
            st.markdown(_kpi("Leverage Penalty", f"+{lev_p:.2f}%",
                f"MTF Weight: {mtf_w:.1f}%", "#a855f7", min(lev_p/1.0,1.0)*100, "⚡ "), unsafe_allow_html=True)
        with col_pm4:
            st.markdown(_kpi("NATR Vol Heat", f"{natr_heat:.1f}%",
                f"Σ(Alloc%×NATR) | {natr_label}", natr_color, min(natr_heat/30.0,1.0)*100, "📡 "), unsafe_allow_html=True)
        with col_pm5:
            st.markdown(_kpi("Regime Multiplier", f"{rm:.2f}×",
                f"Regime: {regime_label}", rm_color, min(rm/1.5,1.0)*100, "🌐 "), unsafe_allow_html=True)
        with col_pm6:
            st.markdown(_kpi("Portfolio Value", f"₹{_pv/100000:.1f}L",
                f"₹{_pv:,.0f} total capital · {regime_label}", "#06b6d4",
                min(rm/1.5,1.0)*100, "💼 "), unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        
        # ── Regime Readiness Strip ───────────────────────────────────────────
        _regime_action_map = {"BULL":"Go Heavy", "EARLY_BULL":"Accumulate", "LATE_BULL":"Gradual Reduce",
                             "SIDEWAYS":"Stock Pick", "CORRECTION":"Hedged", "BEAR":"Defensive", "CRISIS":"Cash"}
        _regime_action = _regime_action_map.get(regime_label, "Cautious")
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:16px;padding:8px 18px;
                    background:rgba(15,23,42,0.5);border-radius:10px;
                    border-left:4px solid {rm_color};margin:8px 0 18px 0;">
          <span style="color:{rm_color};font-family:'Outfit';font-size:0.8rem;font-weight:700;text-transform:uppercase;letter-spacing:0.5px;">
            ▸ {regime_label}
          </span>
          <span style="color:{rm_color};font-family:'Outfit';font-size:0.85rem;font-weight:600;">
            {_regime_action}
          </span>
          <span style="flex:1;"></span>
          <span style="color:#64748b;font-size:0.72rem;">
            Regime {rm:.2f}× · ℙ Heat {adj_h:.2f}% · {natr_label}
          </span>
          <span style="color:{rm_color};font-size:0.85rem;">▸</span>
        </div>
        """, unsafe_allow_html=True)
        
        # Upper Charts (Side-by-Side)
        col_act1, col_act2 = st.columns([1.1, 1])
        
        with col_act1:
            st.markdown("""<div style="font-family:'Outfit';font-size:1.05rem;font-weight:700;color:#f1f5f9;margin-bottom:12px;">
              🍩 Portfolio Asset Allocation</div>""", unsafe_allow_html=True)
            _pv          = blueprint.get("portfolio_value", 10000000.0)
            _core_pct    = df_act[df_act["Bucket"] == "Core Holdings"]["Allocation"].sum() if not df_act.empty else 0.0
            _vams_pct    = df_act[df_act["Bucket"] == "Satellite Holdings"]["Allocation"].sum() if not df_act.empty else 0.0
            _gold_pct    = blueprint.get("gold_futures_pct", 0.0)
            _silver_pct  = blueprint.get("silver_futures_pct", 0.0)
            _mtf_pct     = blueprint.get("mtf_leverage_pct", 0.0)
            _cash_pct    = max(0.0, 100.0 - (_core_pct + _vams_pct + _gold_pct + _silver_pct + _mtf_pct))
            
            core_total_pct = _core_pct
            vams_total_pct = _vams_pct
            cash_total_pct = _cash_pct
            leverage_total_pct = _mtf_pct + _gold_pct + _silver_pct
            
            core_total_rs = (core_total_pct / 100.0) * _pv
            vams_total_rs = (vams_total_pct / 100.0) * _pv
            cash_total_rs = (cash_total_pct / 100.0) * _pv
            leverage_total_rs = (leverage_total_pct / 100.0) * _pv
            
            _regime_label = blueprint.get("market_regime", "SIDEWAYS")
            # Build donut chart data (Satellite Holdings renamed)
            _slices = [
                ("Core Holdings",       core_total_pct,     "#3b82f6"),
                ("Satellite Holdings",  vams_total_pct,     "#8b5cf6"),
                ("Unallocated Cash",    cash_total_pct,     "#475569"),
            ]
            _slices_nonzero = [(n, v, c) for n, v, c in _slices if v > 0.01]
            
            if _slices_nonzero:
                fig_sizing = go.Figure(data=[go.Pie(
                    labels=[s[0] for s in _slices_nonzero],
                    values=[s[1] for s in _slices_nonzero],
                    hole=0.55,
                    marker=dict(colors=[s[2] for s in _slices_nonzero], line=dict(color="rgba(0,0,0,0.3)", width=1.5)),
                    textposition="outside",
                    texttemplate="%{label}<br><b>%{value:.1f}%</b>",
                    textfont=dict(size=11, family="Inter", color="#f8fafc"),
                    hovertemplate="<b>%{label}</b><br>Weight: %{value:.2f}%<extra></extra>",
                    pull=[0.05 if n == "Core Holdings" else (0.03 if n == "Satellite Holdings" else 0) for n, _, _ in _slices_nonzero]
                )])
                _total_exp = core_total_pct + vams_total_pct
                fig_sizing.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=10, r=10, t=10, b=10),
                    height=380, font_family="Inter", showlegend=False,
                    annotations=[dict(
                        text=f"<b>{_total_exp:.1f}%</b><br><span style='font-size:12px'>Allocated</span>",
                        x=0.5, y=0.5, xanchor="center", yanchor="middle",
                        font=dict(size=24, color="#f1f5f9", family="Outfit"), showarrow=False
                    )]
                )
                st.plotly_chart(fig_sizing, use_container_width=True)
                
            # Capital Allocation Table
            def _rs(v): return f"₹{v:,.0f}"
            def _pc(v): return f"{v:.2f}%"
            _rows = []
            if core_total_pct > 0.01: _rows.append(("🔵 Core Holdings", _pc(core_total_pct), _rs(core_total_rs), "#3b82f6"))
            if vams_total_pct > 0.01: _rows.append(("🟣 Satellite Holdings", _pc(vams_total_pct), _rs(vams_total_rs), "#8b5cf6"))
            if cash_total_pct > 0.01: _rows.append(("⬛ Unallocated Cash", _pc(cash_total_pct), _rs(cash_total_rs), "#64748b"))
            _total_exp_pct  = blueprint.get("total_exposure_pct", 0.0)
            _non_cash_col   = blueprint.get("non_cash_collateral", 0.0)
            _cash_eq_col    = blueprint.get("cash_equiv_collateral", 0.0)
            _fo_margin      = blueprint.get("fo_margin_required", 0.0)
            _sebi_ok        = blueprint.get("sebi_compliant", True)
            _shortfall_pct  = blueprint.get("sebi_shortfall_pct", 0.0)
            if tot_alloc > 100.0:
                _sebi_ok = False
                _shortfall_pct = max(_shortfall_pct, tot_alloc - 100.0)
            _sebi_tag       = f'<span style="background:rgba(16,185,129,0.15);color:#34d399;padding:2px 8px;border-radius:8px;font-size:0.78rem;font-weight:700;">COMPLIANT</span>' if _sebi_ok else f'<span style="background:rgba(239,68,68,0.15);color:#f87171;padding:2px 8px;border-radius:8px;font-size:0.78rem;font-weight:700;">SHORTFALL {_shortfall_pct:.1f}%</span>'
            
            _rows_html = "".join([
                f'<tr style="border-bottom:1px solid rgba(255,255,255,0.04);"><td style="padding:7px 10px;color:{clr};font-weight:600;font-size:0.85rem;">{nm}</td><td style="padding:7px 10px;text-align:right;color:#e2e8f0;font-family:monospace;font-weight:700;">{pc}</td><td style="padding:7px 10px;text-align:right;color:#94a3b8;font-family:monospace;">{rs}</td></tr>'
                for nm, pc, rs, clr in _rows
            ])
            
            st.markdown(f"""
            <div class="glass-card" style="padding:0;overflow:hidden;border-radius:12px;margin-top:8px;">
              <div style="background:rgba(30,41,59,0.9);padding:9px 14px;border-bottom:1px solid rgba(255,255,255,0.08);font-family:'Outfit';font-weight:700;color:#f1f5f9;font-size:0.9rem;">
                Capital Allocation Breakdown &nbsp;·&nbsp; Portfolio: <span style="color:#06b6d4;">₹{_pv:,.0f}</span>
              </div>
              <table class="premium-table">
                <thead><tr style="background:rgba(15,23,42,0.4);"><th style="width: 40%; padding:7px 10px; text-align:left; color:#64748b; font-weight:600; font-size:0.75rem; text-transform:uppercase;">Asset Class</th><th style="width: 25%; padding:7px 10px; text-align:right; color:#64748b; font-weight:600; font-size:0.75rem; text-transform:uppercase;">Weight</th><th style="width: 35%; padding:7px 10px; text-align:right; color:#64748b; font-weight:600; font-size:0.75rem; text-transform:uppercase;">₹ Value</th></tr></thead>
                <tbody>{_rows_html}</tbody>
              </table>
              <div style="background:rgba(15,23,42,0.5);padding:10px 14px;border-top:1px solid rgba(255,255,255,0.06);">
                <table class="premium-table" style="font-size:0.83rem;">
                  <tr><td style="color:#94a3b8;padding:4px 0;">Total Exp Weight</td><td style="text-align:right;font-weight:700;color:{'#f59e0b' if _total_exp_pct > 100 else '#e2e8f0'};">{_total_exp_pct:.2f}%</td></tr>
                  <tr><td style="color:#94a3b8;padding:4px 0;">SEBI 50:50 Compliance</td><td style="text-align:right;">{_sebi_tag}</td></tr>
                </table>
              </div>
            </div>
            """, unsafe_allow_html=True)
            
        with col_act2:
            st.markdown("""<div style="font-family:'Outfit';font-size:1.05rem;font-weight:700;color:#f1f5f9;margin-bottom:12px;">
              🍕 Sector Concentration vs 25% Limit</div>""", unsafe_allow_html=True)
            
            # Group by Sector (already normalized)
            _sector_alloc = df_act.groupby("Sector")["Allocation"].sum().to_dict()
            _sector_alloc = {k: v for k, v in _sector_alloc.items() if k not in ["PASSIVE_CORE", "CORE"] and v > 0}
            
            if _sector_alloc:
                _div_wt = _sector_alloc.get("Diversified", 0.0)
                if _div_wt > 3.0:
                    st.warning(f"⚠️ **Sector Warning:** {_div_wt:.1f}% in 'Diversified' (Unresolved Sector). Check mapping.")
                
                _df_sec = pd.DataFrame(list(_sector_alloc.items()), columns=["Sector", "Weight"])
                _df_sec["Color"]  = _df_sec.apply(lambda r: "#f59e0b" if r["Sector"] == "Diversified" and r["Weight"] > 3.0 else ("#ef4444" if r["Weight"] > 25.0 else ("#818cf8" if r["Weight"] > 15 else "#3b82f6")), axis=1)
                _df_sec = _df_sec.sort_values("Weight", ascending=True)
                fig_sec = go.Figure()
                fig_sec.add_trace(go.Bar(
                    x=_df_sec["Weight"], y=_df_sec["Sector"], orientation="h",
                    marker=dict(color=_df_sec["Color"], opacity=0.85,
                                line=dict(color="rgba(255,255,255,0.1)", width=0.5)),
                    text=_df_sec["Weight"].apply(lambda v: f"  {v:.1f}%"),
                    textposition="outside",
                    textfont=dict(size=11, family="Inter"),
                    showlegend=False,
                    hovertemplate="<b>%{y}</b><br>Weight: %{x:.2f}%<extra></extra>"
                ))
                fig_sec.add_vline(x=25.0, line_dash="dash", line_color="rgba(239,68,68,0.6)", line_width=1.5, annotation_text="Max 25%", annotation_font_color="#f87171", annotation_font_size=10)
                # Add shaded danger zone beyond 25%
                fig_sec.add_vrect(x0=25.0, x1=_df_sec["Weight"].max() * 1.15, fillcolor="rgba(239,68,68,0.04)", line_width=0)
                fig_sec.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=10, r=40, t=10, b=10), height=max(180, len(_df_sec) * 36),
                    barmode="overlay",
                    xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.04)", ticksuffix="%", tickfont=dict(color="#64748b", size=10), zeroline=False),
                    yaxis=dict(showgrid=False, tickfont=dict(color="#94a3b8", size=10))
                )
                st.plotly_chart(fig_sec, use_container_width=True)
                # Compact compliance pills with violation count
                _violations = [(s, w) for s, w in _sector_alloc.items() if w > 25.0]
                _violation_badge = f'<span style="background:rgba(239,68,68,0.15);color:#f87171;padding:2px 10px;border-radius:20px;font-size:0.72rem;font-weight:700;margin-left:8px;">⚠ {len(_violations)} OVER 25%</span>' if _violations else ""
                st.markdown(f"<div style='display:flex;align-items:center;flex-wrap:wrap;gap:6px;margin-bottom:8px;'><span style='color:#94a3b8;font-size:0.8rem;font-weight:600;'>Sector Diversification Pills</span>{_violation_badge}</div>", unsafe_allow_html=True)
                _pills = []
                for sec, weight in sorted(_sector_alloc.items(), key=lambda x: -x[1]):
                    if "Diversified" in sec:
                        _pills.append(f'<span style="background:rgba(129,140,248,0.12);color:#818cf8;border:1px solid rgba(129,140,248,0.3);padding:3px 9px;border-radius:20px;font-size:0.75rem;font-weight:600;">♾ {sec[:20]} {weight:.1f}%</span>')
                    elif weight > 25.0:
                        _pills.append(f'<span style="background:rgba(239,68,68,0.12);color:#f87171;border:1px solid rgba(239,68,68,0.3);padding:3px 9px;border-radius:20px;font-size:0.75rem;font-weight:600;">⚠️ {sec[:20]} {weight:.1f}%</span>')
                    elif weight > 15.0:
                        _pills.append(f'<span style="background:rgba(245,158,11,0.10);color:#fbbf24;border:1px solid rgba(245,158,11,0.2);padding:3px 9px;border-radius:20px;font-size:0.75rem;font-weight:500;">◈ {sec[:20]} {weight:.1f}%</span>')
                else:
                        _pills.append(f'<span style="background:rgba(16,185,129,0.08);color:#34d399;border:1px solid rgba(16,185,129,0.2);padding:3px 9px;border-radius:20px;font-size:0.75rem;font-weight:500;">✓ {sec[:20]} {weight:.1f}%</span>')
                st.markdown(f'<div style="display:flex;flex-wrap:wrap;gap:6px;padding:14px;background:rgba(15,23,42,0.3);border-radius:10px;border:1px solid rgba(255,255,255,0.05);">{" ".join(_pills)}</div>', unsafe_allow_html=True)
            else:
                st.info("No active equity holdings. Concentration analysis offline.")
        st.markdown("<hr style='border:none;border-top:1px solid rgba(255,255,255,0.07);margin:24px 0;'>", unsafe_allow_html=True)
        
        # ── ✨ Portfolio Heat Allocation Map ──────────────────────────────────────
        st.markdown("""
        <div style="display:flex;align-items:center;gap:12px;margin-bottom:6px;">
          <div style="font-family:'Outfit';font-size:1.1rem;font-weight:800;color:#f1f5f9;
                      letter-spacing:-0.3px;">🔥 Portfolio Heat Map</div>
          <div style="font-size:0.75rem;color:#64748b;font-family:'Inter';">Block size = Allocation % · Colour = Heat (green → amber → red)</div>
          <div style="margin-left:auto;display:flex;align-items:center;gap:6px;">
            <span style="width:48px;height:8px;background:linear-gradient(90deg,#059669,#10b981,#f59e0b,#ef4444);border-radius:4px;display:inline-block;"></span>
            <span style="font-size:0.68rem;color:#64748b;">Low Heat</span>
            <span style="font-size:0.68rem;color:#64748b;">→ High Heat</span>
          </div>
        </div>
        """, unsafe_allow_html=True)
        df_act_tree = df_act[df_act["Allocation"] > 0].copy()
        df_act_tree["Treemap_Group"] = df_act_tree.apply(
            lambda r: r["Bucket"] if r["Sector"] == "Diversified" else r["Sector"], axis=1
        )
        df_act_tree = df_act_tree.sort_values(by="Allocation", ascending=False)
        if len(df_act_tree) > 50:
            df_act_tree = df_act_tree.head(50)
        try:
            # ── Portfolio Heat Score: blend allocation weight + RS_Val (0–1) ──
            _max_alloc = df_act_tree["Allocation"].max() if df_act_tree["Allocation"].max() > 0 else 1.0
            _rs_series = pd.to_numeric(df_act_tree.get("RS_Val", pd.Series([0.5]*len(df_act_tree))), errors="coerce").fillna(0.5)
            # heat = 60% allocation normalised + 40% RS score (inverted: high RS = cool)
            df_act_tree["_heat"] = (
                0.6 * (df_act_tree["Allocation"] / _max_alloc) +
                0.4 * (1.0 - _rs_series.clip(0, 1).values)
            ).clip(0, 1)
            # ── Custom continuous colour scale: deep teal → emerald → amber → crimson ──
            _heat_colorscale = [
                [0.00, "#0d9488"],  # teal  – lightest allocation, best RS
                [0.20, "#10b981"],  # emerald
                [0.40, "#84cc16"],  # lime
                [0.60, "#f59e0b"],  # amber
                [0.80, "#f97316"],  # orange
                [1.00, "#dc2626"],  # crimson – heaviest allocation or worst RS
            ]
            # ── Build PnL label if available ──
            def _pnl_tag(row):
                pnl = row.get("PnL_%", None)
                if pnl is None or pd.isna(pnl): return ""
                sign = "+" if pnl >= 0 else ""
                return f"<br><span>{sign}{pnl:.1f}%</span>"
            df_act_tree["_label"] = df_act_tree.apply(
                lambda r: f"{r['Display_Symbol']}<br>{r['Allocation']:.2f}%{_pnl_tag(r)}",
                axis=1
            )
            fig_tree = px.treemap(
                df_act_tree,
                path=[px.Constant("Total Portfolio"), "Bucket", "Treemap_Group", "Display_Symbol"],
                values="Allocation",
                color="_heat",
                color_continuous_scale=_heat_colorscale,
                range_color=[0.0, 1.0],
                hover_data={"Allocation": ":.2f", "RS_Val": ":.3f"},
                custom_data=["Display_Symbol", "Allocation", "Sector", "_heat"],
            )
            fig_tree.update_traces(
                texttemplate="<b>%{customdata[0]}</b><br>%{customdata[1]:.2f}%",
                textposition="middle center",
                textfont=dict(family="Inter", size=12, color="rgba(255,255,255,0.95)"),
                marker=dict(
                    line=dict(width=2.0, color="rgba(2,6,23,0.7)"),
                    cornerradius=4,
                    pad=dict(t=16, l=4, r=4, b=4),
                ),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Sector: %{customdata[2]}<br>"
                    "Allocation: <b>%{customdata[1]:.2f}%</b><br>"
                    "Portfolio Heat: <b>%{customdata[3]:.2f}</b>"
                    "<extra></extra>"
                ),
            )
            fig_tree.update_layout(
                template="plotly_dark",
                margin=dict(t=6, l=6, r=6, b=6),
                height=440,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter"),
                coloraxis_showscale=False,  # we show our own legend above
                coloraxis_colorbar=dict(
                    title=dict(text="Heat", font=dict(color="#94a3b8", size=11)),
                    tickfont=dict(color="#64748b", size=10),
                    thickness=8, len=0.6, x=1.01,
                    bgcolor="rgba(15,23,42,0.0)",
                ),
            )
            st.plotly_chart(fig_tree, use_container_width=True)
            # ── Summary ribbon below the treemap ──────────────────────────────
            _n_hot  = int((df_act_tree["_heat"] > 0.65).sum())
            _n_warm = int(((df_act_tree["_heat"] > 0.35) & (df_act_tree["_heat"] <= 0.65)).sum())
            _n_cool = int((df_act_tree["_heat"] <= 0.35).sum())
            _total_heat = float(df_act_tree["_heat"].mean())
            _heat_label = "🔥 HIGH" if _total_heat > 0.60 else ("☀️ MODERATE" if _total_heat > 0.35 else "❄️ LOW")
            _heat_color = "#ef4444" if _total_heat > 0.60 else ("#f59e0b" if _total_heat > 0.35 else "#10b981")
            st.markdown(f"""
            <div style="display:flex;gap:10px;margin-top:-6px;margin-bottom:12px;align-items:center;flex-wrap:wrap;">
              <div style="background:rgba(15,23,42,0.7);border:1px solid rgba(255,255,255,0.07);
                          border-radius:10px;padding:7px 16px;font-size:0.78rem;font-family:'Inter';
                          color:{_heat_color};font-weight:700;">Portfolio Heat: {_heat_label} ({_total_heat:.2f})</div>
              <div style="background:rgba(220,38,38,0.1);border:1px solid rgba(220,38,38,0.25);
                          border-radius:10px;padding:7px 14px;font-size:0.78rem;font-family:'Inter';color:#f87171;">
                🔴 Hot Positions: <b>{_n_hot}</b></div>
              <div style="background:rgba(245,158,11,0.1);border:1px solid rgba(245,158,11,0.25);
                          border-radius:10px;padding:7px 14px;font-size:0.78rem;font-family:'Inter';color:#fbbf24;">
                🟡 Warm Positions: <b>{_n_warm}</b></div>
              <div style="background:rgba(16,185,129,0.1);border:1px solid rgba(16,185,129,0.25);
                          border-radius:10px;padding:7px 14px;font-size:0.78rem;font-family:'Inter';color:#34d399;">
                🟢 Cool Positions: <b>{_n_cool}</b></div>
            </div>
            """, unsafe_allow_html=True)
        except Exception as _tree_err:
            st.info(f"Heat map temporarily unavailable: {_tree_err}")
        
        # ── Consolidated Position Ledger (Native Dataframe) ─────────────────────
        st.markdown("#### 📋 Consolidated Position Ledger")
        # Prepare dataframe for native display
        _dl_cols_list = ["Display_Symbol", "Bucket", "Sector", "Allocation", "Current Price",
                         "Entry_Price", "Stop_Loss", "RS_Val", "PnL_%", "Rationale", "Pending_Exit"]
        _dl_cols_avail = [c for c in _dl_cols_list if c in df_act.columns]
        _df_disp = df_act[_dl_cols_avail].copy()
        
        # Sort defaults: Bucket then Allocation
        _df_disp = _df_disp.sort_values(by=["Bucket", "Allocation"], ascending=[True, False])
        # Download button above table
        st.download_button(
            label="📥 Export CSV",
            data=_df_disp.drop(columns=["Pending_Exit"], errors='ignore').to_csv(index=False).encode("utf-8"),
            file_name=f"portfolio_ledger_{selected_date}.csv",
            mime="text/csv",
            key="ledger_download_v2"
        )
        st.dataframe(
            _df_disp,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Display_Symbol": st.column_config.TextColumn("Symbol / Fund", width="medium"),
                "Bucket": st.column_config.TextColumn("Bucket"),
                "Sector": st.column_config.TextColumn("Sector"),
                "Allocation": st.column_config.ProgressColumn("Alloc %", format="%.2f%%", min_value=0, max_value=30),
                "Current Price": st.column_config.NumberColumn("Price (₹)", format="%.2f"),
                "Entry_Price": st.column_config.NumberColumn("Entry (₹)", format="%.2f"),
                "Stop_Loss": st.column_config.NumberColumn("Stop Loss (₹)", format="%.2f"),
                "RS_Val": st.column_config.NumberColumn("RS Score", format="%.3f"),
                "PnL_%": st.column_config.NumberColumn("P&L %", format="%.1f%%"),
                "Rationale": st.column_config.TextColumn("Rationale"),
                "Pending_Exit": st.column_config.CheckboxColumn("Pending Exit")
            }
        )
        # ── ✨ Premium Correlation Matrix Heatmap ─────────────────────────────────
        st.markdown("""
        <div style="display:flex;align-items:center;gap:10px;margin-top:28px;margin-bottom:6px;">
          <div style="font-family:'Outfit';font-size:1.1rem;font-weight:800;color:#f1f5f9;
                      letter-spacing:-0.3px;">🧬 Position Correlation Intelligence</div>
          <div style="font-size:0.75rem;color:#64748b;font-family:'Inter';">Hierarchically-clustered · Diverging palette · ρ ≥ 0.70 flagged as risk</div>
        </div>
        """, unsafe_allow_html=True)
        _stale_corr = True  # default: mark stale until we confirm readable data
        if os.path.exists(corr_path):
            # Read CSV directly (SQLite via read_data_smart may have stale old data)
            try:
                df_corr = pd.read_csv(corr_path, index_col=0)
            except Exception:
                df_corr = read_data_smart(corr_path, index_col=0)
            if df_corr is None:
                df_corr = pd.DataFrame()
            if not df_corr.empty:
                _stale_corr = False
                _corr_syms = set(df_corr.columns) & set(df_corr.index)
                _act_syms = set(df_act[~df_act["Pending_Exit"]]["Symbol"]) if "Pending_Exit" in df_act.columns else set(df_act["Symbol"])
                # Filter matrix to only symbols present in current active holdings
                _common_syms = _corr_syms & _act_syms
                _missing_in_matrix = _act_syms - _corr_syms
                _missing_in_portfolio = _corr_syms - _act_syms
                if _common_syms:
                    # Subset matrix to common symbols only
                    df_corr = df_corr.loc[list(_common_syms), list(_common_syms)]
                if _missing_in_matrix:
                    # ── Dynamic correlation fill for missing holdings via yfinance ──
                    _corr_fill_ok = False
                    _fill_err = ""
                    try:
                        _end_dt = pd.Timestamp(selected_date)
                        _start_dt = _end_dt - pd.Timedelta(days=180)
                        _close_data = {}
                        _all_fetch = list(_act_syms)
                        for _sym in _all_fetch:
                            try:
                                _tk = yf.Ticker(f"{_sym}.NS")
                                _hist = _tk.history(start=_start_dt.strftime("%Y-%m-%d"), end=_end_dt.strftime("%Y-%m-%d"))
                                if not _hist.empty:
                                    _close_data[_sym] = _hist["Close"]
                            except Exception:
                                pass
                        if len(_close_data) >= 2:
                            _ret_df = pd.DataFrame({k: v.pct_change() for k, v in _close_data.items()}).dropna()
                            if not _ret_df.empty and len(_ret_df) >= 10:
                                _live_corr = _ret_df.corr()
                                for _mc in list(_missing_in_matrix):
                                    if _mc in _live_corr.columns:
                                        df_corr[_mc] = 0.0
                                        df_corr.loc[_mc] = 0.0
                                        df_corr.loc[_mc, _mc] = 1.0
                                        for _ec in df_corr.columns:
                                            if _ec in _live_corr.columns and _mc in _live_corr.index:
                                                _v = _live_corr.loc[_mc, _ec]
                                                df_corr.loc[_mc, _ec] = _v
                                                df_corr.loc[_ec, _mc] = _v
                                _corr_syms = set(df_corr.columns) & set(df_corr.index)
                                _common_syms = _corr_syms & _act_syms
                                _missing_in_matrix = _act_syms - _corr_syms
                                if not _missing_in_matrix:
                                    _corr_fill_ok = True
                    except Exception as _fe:
                        _fill_err = str(_fe)
                    if _corr_fill_ok:
                        st.success(f"✅ Correlation matrix expanded to all {len(_common_syms)} active holdings via live data.")
                    else:
                        st.markdown(f"""
                    <div style="background:rgba(245,158,11,0.08);border:1px solid rgba(245,158,11,0.25);
                                 border-radius:10px;padding:10px 14px;margin-bottom:12px;font-family:'Inter';">
                      ⚠️ <b style="color:#fbbf24;">{len(_missing_in_matrix)} holding(s) missing from correlation matrix</b>
                      <span style="color:#94a3b8;font-size:0.78rem;"> — Showing matrix for {len(_common_syms)} matched symbols only. Re-run pipeline to refresh full matrix.</span>
                    </div>""", unsafe_allow_html=True)
            if not df_corr.empty:
                _n_assets = len(df_corr)
                if _n_assets > 60:
                    # Large portfolio — show summary stats instead
                    _mask_u = np.triu(np.ones_like(df_corr.values, dtype=bool), k=1)
                    _all_vals = df_corr.values[_mask_u]
                    _avg_corr = float(np.nanmean(_all_vals))
                    _max_corr = float(np.nanmax(_all_vals))
                    _pct_high = float(np.mean(np.abs(_all_vals) >= 0.70)) * 100.0
                    _ccolor = "#ef4444" if _avg_corr > 0.55 else ("#f59e0b" if _avg_corr > 0.35 else "#10b981")
                    st.markdown(f"""
                    <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:14px;">
                      <div style="background:rgba(15,23,42,0.7);border:1px solid rgba(255,255,255,0.07);
                                  border-radius:12px;padding:14px 18px;">
                        <div style="font-size:0.72rem;color:#64748b;font-family:'Inter';text-transform:uppercase;letter-spacing:0.5px;">Avg Pairwise ρ</div>
                        <div style="font-size:1.8rem;font-weight:800;color:{_ccolor};font-family:'Outfit';margin-top:4px;">{_avg_corr:.2f}</div>
                      </div>
                      <div style="background:rgba(15,23,42,0.7);border:1px solid rgba(255,255,255,0.07);
                                  border-radius:12px;padding:14px 18px;">
                        <div style="font-size:0.72rem;color:#64748b;font-family:'Inter';text-transform:uppercase;letter-spacing:0.5px;">Max ρ in Portfolio</div>
                        <div style="font-size:1.8rem;font-weight:800;color:#f87171;font-family:'Outfit';margin-top:4px;">{_max_corr:.2f}</div>
                      </div>
                      <div style="background:rgba(15,23,42,0.7);border:1px solid rgba(255,255,255,0.07);
                                  border-radius:12px;padding:14px 18px;">
                        <div style="font-size:0.72rem;color:#64748b;font-family:'Inter';text-transform:uppercase;letter-spacing:0.5px;">High-Risk Pairs (ρ≥0.70)</div>
                        <div style="font-size:1.8rem;font-weight:800;color:#fbbf24;font-family:'Outfit';margin-top:4px;">{_pct_high:.1f}%</div>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.info(f"🗂️ Portfolio has **{_n_assets} holdings** — view limits disabled (matrix disabled above 60 to prevent lag).")
                else:
                    # ── Hierarchical clustering reorder ──
                    try:
                        import scipy.cluster.hierarchy as sch
                        _d_dist = sch.distance.pdist(df_corr.values)
                        _linkage = sch.linkage(_d_dist, method="ward")
                        _ind_leaves = sch.leaves_list(_linkage)
                        df_corr = df_corr.iloc[_ind_leaves, _ind_leaves]
                    except Exception:
                        try:
                            _mean_c = df_corr.mean().sort_values(ascending=False)
                            df_corr = df_corr.loc[_mean_c.index, _mean_c.index]
                        except Exception:
                            pass
                    # ── Custom diverging colorscale: vibrant blue → white → crimson ──
                    _corr_scale = [
                        [0.00, "#1d4ed8"],  # deep blue  (strong negative)
                        [0.20, "#3b82f6"],  # sky blue
                        [0.40, "#bfdbfe"],  # light blue
                        [0.50, "#f8fafc"],  # near-white (zero correlation)
                        [0.60, "#fecaca"],  # light rose
                        [0.80, "#ef4444"],  # red
                        [1.00, "#991b1b"],  # deep crimson (strong positive)
                    ]
                    # ── Build text annotations for each cell ──
                    _z = df_corr.values
                    _text_matrix = [[f"{_z[r][c]:.2f}" if r != c else "" for c in range(_n_assets)] for r in range(_n_assets)]
                    # ── Map AMFI codes to readable names for Core holdings ──
                    _corr_labels = []
                    for _sym_c in list(df_corr.columns):
                        _mapped = _mf_name_map_global.get(str(_sym_c), str(_sym_c))
                        # Truncate long names for readability
                        if len(_mapped) > 28:
                            _mapped = _mapped[:26] + ".."
                        _corr_labels.append(_mapped)
                    fig_corr = go.Figure(data=go.Heatmap(
                        z=_z,
                        x=_corr_labels,
                        y=_corr_labels,
                        text=_text_matrix,
                        texttemplate="%{text}",
                        textfont=dict(size=9, color="rgba(15,23,42,0.9)", family="Inter"),
                        colorscale=_corr_scale,
                        zmin=-1.0, zmax=1.0,
                        showscale=True,
                        colorbar=dict(
                            title=dict(text="ρ", font=dict(size=13, color="#94a3b8")),
                            tickvals=[-1, -0.5, 0, 0.5, 1],
                            ticktext=["-1.0", "-0.5", "0.0", "+0.5", "+1.0"],
                            tickfont=dict(size=10, color="#64748b"),
                            thickness=10, len=0.75,
                            bgcolor="rgba(15,23,42,0.0)",
                            outlinewidth=0,
                        ),
                        hovertemplate="<b>%{y}</b> ↔ <b>%{x}</b><br>ρ = <b>%{z:.3f}</b><extra></extra>",
                    ))
                    # ── Style axes ──
                    _tick_angle = -55 if _n_assets > 15 else -35
                    _tick_sz = 9 if _n_assets > 20 else 11
                    fig_corr.update_xaxes(
                        tickangle=_tick_angle,
                        tickfont=dict(size=_tick_sz, family="Inter", color="#94a3b8"),
                        showgrid=False, zeroline=False,
                    )
                    fig_corr.update_yaxes(
                        tickfont=dict(size=_tick_sz, family="Inter", color="#94a3b8"),
                        showgrid=False, zeroline=False, autorange="reversed",
                    )
                    fig_corr.update_layout(
                        template="plotly_dark",
                        plot_bgcolor="rgba(8,14,30,0.95)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        margin=dict(l=10, r=60, t=10, b=10),
                        height=max(360, _n_assets * 32 + 60),
                        font=dict(family="Inter"),
                    )
                    st.plotly_chart(fig_corr, use_container_width=True)
                # ── High-correlation pair callout cards ────────────────────────────
                try:
                    _high_corr_pairs = []
                    _cols_c = list(df_corr.columns)
                    for _i_c in range(len(_cols_c)):
                        for _j_c in range(_i_c + 1, len(_cols_c)):
                            _rho = float(df_corr.iloc[_i_c, _j_c])
                            if pd.notna(_rho) and abs(_rho) >= 0.70:
                                _high_corr_pairs.append((_cols_c[_i_c], _cols_c[_j_c], _rho))
                    _high_corr_pairs.sort(key=lambda x: -abs(x[2]))
                    if _high_corr_pairs:
                        _pairs_label = "⚠️ CRITICAL" if any(abs(r) >= 0.85 for _, _, r in _high_corr_pairs) else "🟡 WARNING"
                        _pairs_color = "#ef4444" if any(abs(r) >= 0.85 for _, _, r in _high_corr_pairs) else "#fbbf24"
                        st.markdown(f"""
                        <div style="margin-top:10px;">
                          <div style="font-family:'Outfit';font-size:0.88rem;font-weight:700;color:{_pairs_color};
                                      margin-bottom:8px;">{_pairs_label} — {len(_high_corr_pairs)} High-Correlation Pairs (ρ ≥ 0.70)</div>
                          <div style="display:flex;flex-wrap:wrap;gap:7px;">
                        """, unsafe_allow_html=True)
                        _pair_cards = []
                        for _s1_c, _s2_c, _rho_v in _high_corr_pairs[:24]:
                            _n1_c = _mf_name_map_global.get(str(_s1_c), str(_s1_c))[:20] + (".." if len(_mf_name_map_global.get(str(_s1_c), str(_s1_c))) > 20 else "")
                            _n2_c = _mf_name_map_global.get(str(_s2_c), str(_s2_c))[:20] + (".." if len(_mf_name_map_global.get(str(_s2_c), str(_s2_c))) > 20 else "")
                            _rc = "#ef4444" if abs(_rho_v) >= 0.85 else "#f97316" if abs(_rho_v) >= 0.77 else "#fbbf24"
                            _bg = "rgba(239,68,68,0.09)" if abs(_rho_v) >= 0.85 else "rgba(249,115,22,0.07)" if abs(_rho_v) >= 0.77 else "rgba(245,158,11,0.07)"
                            _bd = f"rgba({239 if abs(_rho_v)>=0.85 else 249},{68 if abs(_rho_v)>=0.85 else 115},{68 if abs(_rho_v)>=0.85 else 22},0.3)"
                            _pair_cards.append(
                                f"<div style='background:{_bg};border:1px solid {_bd};"
                                f"padding:6px 12px;border-radius:10px;font-size:0.77rem;font-family:Inter;color:#e2e8f0;'"
                                f"><span style='color:#94a3b8;'>{_n1_c}</span>"
                                f" <span style='color:{_rc};font-weight:700;'>ρ={_rho_v:+.2f}</span> "
                                f"<span style='color:#94a3b8;'>{_n2_c}</span></div>"
                            )
                        st.markdown("".join(_pair_cards) + "</div></div>", unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div style="background:rgba(16,185,129,0.08);border:1px solid rgba(16,185,129,0.25);
                                     border-radius:10px;padding:12px 16px;font-family:'Inter';font-size:0.82rem;color:#34d399;">
                          ✅ <b>Excellent Diversification</b> — No position pairs exceed the ρ = 0.70 concentration threshold.
                        </div>""", unsafe_allow_html=True)
                except Exception:
                    pass
            elif not _stale_corr:
                st.info("Correlation matrix has empty dimensions.")
        else:
            st.info("Diversification matrix offline. No active positions to correlate.")
# ──────────────────────────────────────────────────────────────────────────────
# TAB 4: PORTFOLIO MANAGER RISK & SIZING PANEL
# ──────────────────────────────────────────────────────────────────────────────
def _render_top_sections(df_maac):
    """Render Weighted Ranking header, KPI strip, factor charts, weight cards & Top 50 table."""
    if df_maac.empty:
        return
    _df_r = df_maac.copy()
    if "Tier" in _df_r.columns:
        _df_r = _df_r[~_df_r["Tier"].astype(str).str.contains("CORE", case=False)]
    if "Symbol" in _df_r.columns:
        _df_r = _df_r[~_df_r["Symbol"].astype(str).str.match(r"^(NIFTY_|STRATEGY_|INDIA_VIX|MCX_)", case=False)]
    # Filter out zombies: stocks with Factor_Score == 0 (missing/dead data from old pipeline)
    if "Factor_Score" in _df_r.columns:
        _zombie_before = len(_df_r)
        _df_r = _df_r[_df_r["Factor_Score"] > 0]
        _zombie_removed = _zombie_before - len(_df_r)
        if _zombie_removed > 100:
            st.warning(f"⚠️ **{_zombie_removed} stocks have Score=0** — pipeline data is stale. Re-run `main.py` to refresh scores.")
    # ALWAYS override Entry_Eligible for display: show top 50 regardless of pipeline cap
    if "Final_Rank" in _df_r.columns:
        _df_r["Entry_Eligible"] = _df_r["Final_Rank"].apply(lambda r: r <= 50 if pd.notna(r) else False)
    st.markdown("""<div class="tab-hero">
    <h2 style="margin:0;font-family:'Outfit';color:#818cf8;font-size:1.4rem;">🏆 Weighted Ranking System — Top 50 Thesis Tracks</h2>
    <p style="margin:4px 0 0;color:#64748b;font-size:0.88rem;">Trend-following positional strategy · Momentum-biased factor weights (F3=35%) · Rank-based tiers · Best 50 opportunities always surfaced</p>
    </div>""", unsafe_allow_html=True)
    _r_elig_df = _df_r[_df_r["Entry_Eligible"] == True] if "Entry_Eligible" in _df_r.columns else pd.DataFrame()
    _r_elig_ct = len(_r_elig_df); _r_t1 = len(_r_elig_df[_r_elig_df["Tier"]=="TIER 1 — HIGH CONVICTION"]) if not _r_elig_df.empty else 0
    _r_t2 = len(_r_elig_df[_r_elig_df["Tier"]=="TIER 2 — MEDIUM CONVICTION"]) if not _r_elig_df.empty else 0
    _r_t3 = len(_r_elig_df[_r_elig_df["Tier"]=="TIER 3 — LOW CONVICTION"]) if not _r_elig_df.empty else 0
    _r_avg_sc = _r_elig_df["Factor_Score"].mean() if not _r_elig_df.empty and "Factor_Score" in _r_elig_df.columns else 0.0
    for _col, (_em, _lbl, _val, _clr) in zip(st.columns(5),
        [("🏆","Top 50 Eligible",str(_r_elig_ct),"#10b981"),("🥇","T1 High Conv.",str(_r_t1),"#10b981"),
         ("🥈","T2 Medium",str(_r_t2),"#fbbf24"),("🥉","T3 Low",str(_r_t3),"#3b82f6"),
         ("📊","Avg Score",f"{_r_avg_sc:.1f}/100","#818cf8")]): _col.markdown(f"""<div class="glass-card" style="text-align:center;padding:12px 8px;border-color:{_clr}30;">
          <div style="font-size:1.2rem;">{_em}</div>
          <div style="font-size:0.6rem;color:#64748b;text-transform:uppercase;">{_lbl}</div>
          <div style="font-family:'Outfit';font-weight:800;color:{_clr};font-size:1.35rem;margin-top:2px;">{_val}</div>
        </div>""", unsafe_allow_html=True)
    st.markdown('<div style="height:8px;"></div>', unsafe_allow_html=True)
    # Factor score averages
    _fav = {"F1_SECTORAL_TREND":[],"F2_THEMATIC_TREND":[],"F3_MOMENTUM":[],"F4_GROWTH":[],"F6_DELIVERY_CONFIRMATION":[],"F7_PEAD":[],"F8_FII_DII_CONVICTION":[]}
    if not _df_r.empty and "Factor_Details" in _df_r.columns:
        for _fd_str in _df_r["Factor_Details"].dropna():
            try:
                _fd = json.loads(str(_fd_str))
                for _fk in _fav:
                    _sc = _fd.get(_fk, {}).get("score", None)
                    if _sc is not None: _fav[_fk].append(float(_sc))
            except: pass
    _fav = {k: (sum(v)/len(v) if v else 0.0) for k, v in _fav.items()}
    _rc1, _rc2 = st.columns(2)
    with _rc1:
        _act_df = _df_r[_df_r["Allocation_%"]>0] if "Allocation_%" in _df_r.columns else pd.DataFrame()
        if not _act_df.empty and "Tier" in _act_df.columns:
            _tc = {"TIER 1 — HIGH CONVICTION":"#10b981","TIER 2 — MEDIUM CONVICTION":"#fbbf24","TIER 3 — LOW CONVICTION":"#3b82f6","WATCHLIST":"#64748b"}
            _tl = {"TIER 1 — HIGH CONVICTION":"T1: High (1-15)","TIER 2 — MEDIUM CONVICTION":"T2: Med (16-35)","TIER 3 — LOW CONVICTION":"T3: Low (36-50)","WATCHLIST":"Watchlist (51+)"}
            _sa = _act_df.groupby("Tier")["Allocation_%"].sum().reset_index()
            _sa["L"] = _sa["Tier"].map(_tl).fillna(_sa["Tier"]); _sa["C"] = _sa["Tier"].map(_tc).fillna("#334155")
            st.markdown("<div style='margin-top: 25px;'></div>", unsafe_allow_html=True)
            st.plotly_chart(go.Figure(data=[go.Pie(labels=_sa["L"],values=_sa["Allocation_%"],hole=0.5,marker_colors=_sa["C"].tolist(),textinfo="percent+label",textfont_size=11)]
                ).update_layout(plot_bgcolor="rgba(0,0,0,0)",paper_bgcolor="rgba(0,0,0,0)",margin=dict(l=0,r=0,t=10,b=0),height=280,font_family="Inter",showlegend=False), use_container_width=True)
        else: st.info("No active tier allocations yet.")
    with _rc2:
        _fb = go.Figure()
        _fb.add_trace(go.Bar(y=["Momentum","Sector","Delivery","Growth","FII/DII","PEAD"],
            x=[_fav["F3_MOMENTUM"],(_fav["F1_SECTORAL_TREND"]+_fav["F2_THEMATIC_TREND"])/2,_fav["F6_DELIVERY_CONFIRMATION"],_fav["F4_GROWTH"],_fav["F8_FII_DII_CONVICTION"],_fav["F7_PEAD"]],
            orientation="h",marker_color=["#fbbf24","#3b82f6","#60a5fa","#a855f7","#ec4899","#34d399"],textposition="outside"))
        _fb.add_vline(x=50,line_dash="dash",line_color="rgba(255,255,255,0.15)")
        _fb.update_layout(plot_bgcolor="rgba(0,0,0,0)",paper_bgcolor="rgba(0,0,0,0)",margin=dict(l=0,r=70,t=10,b=0),height=280,font_family="Inter",xaxis=dict(range=[0,130]),yaxis=dict(showgrid=False),showlegend=False)
        st.plotly_chart(_fb, use_container_width=True)
    # Factor Weight Cards
    st.markdown('<div style="border-top:1px solid rgba(255,255,255,0.05);margin:8px 0 16px;"></div>', unsafe_allow_html=True)
    _FC = [
        ("F3_MOMENTUM", "🚀", "Momentum (35%)", "#fbbf24", 0.35,
         "Stage 2 breakout +40pts · RS vs Nifty50 0-30pts · Mom percentile 0-30pts. Primary edge multiplier."),
        ("F1_SECTORAL_TREND", "🔄", "Sector & Theme (25%)", "#3b82f6", 0.25,
         "Sector percentile rank + Active narrative themes. Top-down tailwinds amplify individual stock moves."),
        ("F6_DELIVERY_CONFIRMATION", "📦", "Delivery & Volume (12%)", "#60a5fa", 0.12,
         "Delivery% 30%→70% = +60pts · Volume ratio breakouts 1.2x+ = +40pts. Detects institutional accumulation."),
        ("F4_GROWTH", "📈", "Earnings Growth (10%)", "#a855f7", 0.10,
         "Sales trajectory 0-40% = 0-50pts · Profit margins 0-40% = 0-50pts. Fundamental fuel for sustained trends."),
        ("F8_FII_DII_CONVICTION", "🏦", "FII / DII Flows (7%)", "#ec4899", 0.07,
         "Net institutional holding change -2% to +5% = 0-100. Smart money confirms the trend."),
        ("F7_PEAD", "⚡", "PEAD Catalyst (3%)", "#34d399", 0.03,
         "Post-Earnings Announcement Drift percentile. Surprises decay in 20-40 days."),
    ]
    for _crow in [_FC[:3], _FC[3:]]:
        _ccols = st.columns(3)
        for _ci, (_fk, _em, _tt, _clr, _wt, _desc) in enumerate(_crow):
            if _fk == "F1_SECTORAL_TREND":
                _as = (_fav.get("F1_SECTORAL_TREND", 0) + _fav.get("F2_THEMATIC_TREND", 0)) / 2
                _cp = int(_r_elig_df["Factors_Passed"].apply(lambda v: ("F1_SECTORAL_TREND" in str(v).split(",")) or ("F2_THEMATIC_TREND" in str(v).split(",")) if pd.notna(v) else False).sum()) if not _r_elig_df.empty and "Factors_Passed" in _r_elig_df.columns else 0
            else:
                _as = _fav.get(_fk, 0.0)
                _cp = int(_r_elig_df["Factors_Passed"].apply(lambda v: _fk in str(v).split(",") if pd.notna(v) else False).sum()) if not _r_elig_df.empty and "Factors_Passed" in _r_elig_df.columns else 0
            with _ccols[_ci]:
                st.markdown(f"""<div class="glass-card" style="border-color:{_clr}28;min-height:220px;padding:20px;margin-bottom:15px;">
                  <div style="display:flex;align-items:center;gap:12px;margin-bottom:12px;">
                    <div style="font-size:1.8rem;padding:8px;background:rgba(255,255,255,0.03);border-radius:10px;">{_em}</div>
                    <div><div style="font-family:'Outfit';font-weight:800;color:{_clr};font-size:1.15rem;">{_tt}</div>
                    <div style="font-size:0.7rem;color:#475569;">Target Weight: {int(_wt*100)}%</div></div>
                  </div>
                  <div style="font-size:0.85rem;color:#94a3b8;margin-bottom:16px;line-height:1.6;min-height:60px;">{_desc}</div>
                  <div style="display:flex;gap:12px;margin-bottom:14px;">
                    <div style="flex:1;text-align:center;background:rgba(255,255,255,0.03);border-radius:8px;padding:10px 4px;border:1px solid rgba(255,255,255,0.04);">
                      <div style="font-size:0.65rem;color:#64748b;text-transform:uppercase;">Avg Score</div>
                      <div style="font-family:'Outfit';font-weight:800;color:{_clr};font-size:1.5rem;">{_as:.0f}</div></div>
                    <div style="flex:1;text-align:center;background:rgba(255,255,255,0.03);border-radius:8px;padding:10px 4px;border:1px solid rgba(255,255,255,0.04);">
                      <div style="font-size:0.65rem;color:#64748b;text-transform:uppercase;">Top50 &ge;50</div>
                      <div style="font-family:'Outfit';font-weight:800;color:{_clr};font-size:1.5rem;">{_cp}</div></div>
                  </div>
                  <div style="background:rgba(255,255,255,0.06);border-radius:6px;height:8px;overflow:hidden;">
                    <div style="background:{_clr};width:{min(100,int(_as))}%;height:100%;border-radius:6px;box-shadow:0 0 10px {_clr}60;"></div></div>
                  <div style="display:flex;justify-content:space-between;margin-top:4px;">
                    <span style="font-size:0.65rem;color:#475569;">0</span>
                    <span style="font-size:0.7rem;color:{_clr};font-weight:700;">{_as:.0f} / 100</span>
                    <span style="font-size:0.65rem;color:#475569;">100</span></div>
                </div>""", unsafe_allow_html=True)
    # Top 50 Ranked Table — full factor breakdown
    st.markdown('<div style="border-top:1px solid rgba(255,255,255,0.05);margin:8px 0 16px;"></div>', unsafe_allow_html=True)
    if not _df_r.empty and "Final_Rank" in _df_r.columns and "Entry_Eligible" in _df_r.columns:
        _top50 = _df_r[_df_r["Entry_Eligible"]==True].sort_values("Final_Rank").head(50)
        if not _top50.empty:
            _tbadge = {
                "TIER 1 — HIGH CONVICTION":'<span style="background:rgba(16,185,129,0.18);color:#34d399;border:1px solid rgba(16,185,129,0.35);padding:2px 7px;border-radius:10px;font-size:0.67rem;font-weight:700;">T1</span>',
                "TIER 2 — MEDIUM CONVICTION":'<span style="background:rgba(251,191,36,0.14);color:#fbbf24;border:1px solid rgba(251,191,36,0.35);padding:2px 7px;border-radius:10px;font-size:0.67rem;font-weight:700;">T2</span>',
                "TIER 3 — LOW CONVICTION":'<span style="background:rgba(59,130,246,0.14);color:#60a5fa;border:1px solid rgba(59,130,246,0.35);padding:2px 7px;border-radius:10px;font-size:0.67rem;font-weight:700;">T3</span>',
                "WATCHLIST":'<span style="background:rgba(100,116,139,0.14);color:#94a3b8;border:1px solid rgba(100,116,139,0.35);padding:2px 7px;border-radius:10px;font-size:0.67rem;font-weight:700;">W</span>',
            }
            _th = ['<div style="overflow-x:auto;"><table class="elimination-table"><thead><tr>']
            for _h in ["#","Symbol","Tier","Score /100","Fused 🧬","🚀 Mom","🔄 Sec+Thm","📦 Delivery","📈 Growth","📰 PEAD","🏦 FII/DII","Sector","Cap"]:
                _th.append(f"<th>{_h}</th>")
            _th.append("</tr></thead><tbody>")
            for _, _row in _top50.iterrows():
                _rp=int(_row.get("Final_Rank",999)); _sym=str(_row.get("Symbol","—")); _tr=str(_row.get("Tier",""))
                _bdg=_tbadge.get(_tr,f'<span style="color:#64748b;">{_tr}</span>')
                _ws=float(_row.get("Factor_Score",_row.get("Weighted_Score",0.0)))
                _fus=float(_row.get("Final_Composite_Score",_row.get("Opportunity_Score",0.0)))
                _sec=str(_row.get("Sector","—")); _cap=str(_row.get("Cap_Category","—")).replace("_"," ") if pd.notna(_row.get("Cap_Category")) else "—"
                _f3=_f1=_f6=_f4=_f7=_f8="—"; _fdr=_row.get("Factor_Details","")
                if _fdr and str(_fdr) not in ("","{}","nan"):
                    try:
                        _fdp=json.loads(str(_fdr))
                        def _fs(k,_d=_fdp): v=_d.get(k,{}).get("score",None); return f"{float(v):.0f}" if v is not None else "—"
                        _f3=_fs("F3_MOMENTUM")
                        _f1_val=_fdp.get("F1_SECTORAL_TREND",{}).get("score",None); _f2_val=_fdp.get("F2_THEMATIC_TREND",{}).get("score",None)
                        if _f1_val is not None and _f2_val is not None: _f1=f"{(float(_f1_val)+float(_f2_val))/2:.0f}"
                        elif _f1_val is not None: _f1=f"{float(_f1_val):.0f}"
                        _f6=_fs("F6_DELIVERY_CONFIRMATION"); _f4=_fs("F4_GROWTH"); _f7=_fs("F7_PEAD"); _f8=_fs("F8_FII_DII_CONVICTION")
                    except: pass
                _bc="#10b981" if _ws>=70 else ("#fbbf24" if _ws>=50 else "#f87171")
                _sbar=(f'<div style="display:flex;align-items:center;gap:5px;">'
                    f'<div style="flex:1;background:rgba(255,255,255,0.07);border-radius:3px;height:6px;overflow:hidden;">'
                    f'<div style="background:{_bc};width:{min(100,_ws):.0f}%;height:100%;border-radius:3px;"></div>'
                    f'</div><span style="color:{_bc};font-weight:700;font-size:0.82rem;min-width:38px;">{_ws:.1f}</span></div>')
                _rc_clr="#fbbf24" if _rp<=15 else ("#60a5fa" if _rp<=35 else "#94a3b8")
                _th.append(
                    f"<tr>"
                    f'<td><span style="font-family:\'Outfit\';font-weight:800;color:{_rc_clr};font-size:0.9rem;">#{_rp}</span></td>'
                    f'<td><span style="font-weight:700;color:#f1f5f9;">{_sym}</span></td>'
                    f"<td>{_bdg}</td><td>{_sbar}</td>"
                    f'<td style="color:#818cf8;font-weight:600;text-align:center;">{_fus:.1f}</td>'
                    f'<td style="color:#fbbf24;font-weight:600;text-align:center;">{_f3}</td>'
                    f'<td style="color:#3b82f6;text-align:center;">{_f1}</td>'
                    f'<td style="color:#60a5fa;text-align:center;">{_f6}</td>'
                    f'<td style="color:#a855f7;text-align:center;">{_f4}</td>'
                    f'<td style="color:#10b981;text-align:center;">{_f7}</td>'
                    f'<td style="color:#ec4899;text-align:center;">{_f8}</td>'
                    f'<td style="color:#94a3b8;font-size:0.77rem;">{_sec}</td>'
                    f'<td style="color:#64748b;font-size:0.74rem;">{_cap}</td></tr>'
                )
            _th.append("</tbody></table></div>")
            st.markdown("".join(_th), unsafe_allow_html=True)
with tab_ta:
    # ── VAM-GQ Tab Header ──
    st.markdown('<div style="border-top:2px solid rgba(255,255,255,0.08);margin:24px 0 16px;"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="background:linear-gradient(135deg,rgba(16,185,129,0.10),rgba(59,130,246,0.06));
                border:1px solid rgba(16,185,129,0.2); border-radius:14px;
                padding:14px 20px; margin-bottom:16px;">
      <div style="font-family:'Outfit';font-size:1.0rem;font-weight:700;color:#34d399;">
        💼 VAM-GQ (Volatility Adjusted Momentum — Growth & Quality) — Screening Lab
      </div>
      <div style="font-size:0.78rem;color:#94a3b8;margin-top:3px;">
        8-Factor pipeline funnel · Qualified stocks pass all gates · Top-ranked flow into <b style="color:#a78bfa;">Satellite Holdings</b>.
      </div>
    </div>
    """, unsafe_allow_html=True)
    
    _render_research_links(compact=True)
    render_unified_veto_ui("tab_ta")
    st.caption("🔬 **Role:** Screening Lab — Quality-Gated Momentum · Factor Scoring · Rank & Eliminate")
    # ── Weighted Ranking, KPI, Factor Cards, Top 50 ──
    _render_top_sections(df_maac)
    # ── Filters + Rejection chart ──
    df_el_all = df_maac.copy()
    if not df_el_all.empty:
        if "Tier" in df_el_all.columns:
            df_el_all = df_el_all[~df_el_all["Tier"].astype(str).str.contains("CORE", case=False)]
        if "Bucket" in df_el_all.columns:
            df_el_all = df_el_all[~df_el_all["Bucket"].astype(str).str.contains("CORE", case=False)]
    def get_el_status(row):
        reason = str(row.get("Rejection_Reason", "")).strip().upper()
        return "Qualified" if ("RANKED" in reason or "PASSED" in reason) else "Disqualified"
    df_el_all["Status"] = df_el_all.apply(get_el_status, axis=1)
    total_scanned = len(df_el_all)
    total_qual = len(df_el_all[df_el_all["Status"] == "Qualified"])
    total_disq = total_scanned - total_qual
    fc1, fc2 = st.columns([1, 2])
    with fc1:
        search_el = st.text_input("Search Symbol...", key="search_el").upper()
        status_filter = st.radio("Status", ["All", "Qualified", "Disqualified"], index=1)
        selected_cap_el = st.multiselect("Cap Tier", ["MEGA_CAP","LARGE_CAP","MID_CAP","SMALL_CAP","BELOW_MIN"],
            default=["MEGA_CAP","LARGE_CAP","MID_CAP","SMALL_CAP","BELOW_MIN"], key="cap_el")
    with fc2:
        st.markdown("#### 📊 Rejection Reasons")
        df_disq = df_el_all[df_el_all["Status"] == "Disqualified"]
        if not df_disq.empty:
            def grp(r):
                r = str(r).lower()
                if any(k in r for k in ["market cap", "adtv", "seasoning", "ipo", "asm", "gsm"]):
                    return "Liquidity & Surveillance"
                if any(k in r for k in ["debt", "d/e", "roce", "roe", "cfo", "pat", "car", "npa", "pledge"]):
                    return "Fundamental Quality"
                if any(k in r for k in ["200 ema", "200_ema", "adx", "vol", "delivery", "sector", "theme"]):
                    return "Trend & Momentum"
                if "fii" in r or "dii" in r:
                    return "Institutional Selling"
                return "Other"
            dr = df_disq["Rejection_Reason"].apply(grp).value_counts().reset_index()
            dr.columns = ["Reason","Count"]
            fig_r = px.bar(dr, x="Count", y="Reason", orientation="h", template="plotly_dark",
                color="Count", color_continuous_scale=px.colors.sequential.Reds)
            fig_r.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=20,r=20,t=10,b=20), height=200, font_family="Inter", coloraxis_showscale=False)
            st.plotly_chart(fig_r, use_container_width=True)
        else:
            st.info("No rejections.")
    st.markdown("---")
    st.markdown("#### 📋 Screening & Elimination Table")
    st.markdown("""
    <div style="margin-bottom:8px;display:flex;gap:8px;flex-wrap:wrap;align-items:center;">
      <span style="color:#64748b;font-size:0.78rem;">⚡ Qualified stocks flow into <b style="color:#a78bfa;">Satellite Holdings</b></span>
      <span style="color:#334155;">·</span>
      <a href="#tab_vams" style="color:#818cf8;font-size:0.78rem;text-decoration:none;">→ View in VAM-B for scoring</a>
      <span style="color:#334155;">·</span>
      <a href="#tab_active" style="color:#6366f1;font-size:0.78rem;text-decoration:none;">→ View in Master Portfolio</a>
    </div>
    """, unsafe_allow_html=True)
    df_el_filtered = df_el_all.copy()
    if search_el: df_el_filtered = df_el_filtered[df_el_filtered["Symbol"].str.contains(search_el)]
    if status_filter == "Qualified": df_el_filtered = df_el_filtered[df_el_filtered["Status"] == "Qualified"]
    elif status_filter == "Disqualified": df_el_filtered = df_el_filtered[df_el_filtered["Status"] == "Disqualified"]
    if selected_cap_el: df_el_filtered = df_el_filtered[df_el_filtered["Cap_Category"].fillna("BELOW_MIN").isin(selected_cap_el)]
    df_el_filtered["sort_s"] = df_el_filtered["Status"].apply(lambda x: 1 if x == "Qualified" else 0)
    df_el_filtered = df_el_filtered.sort_values(by=["sort_s","Final_Rank"], ascending=[False,True])
    tbl = ['<div style="overflow-x:auto;"><table class="elimination-table"><thead><tr>']
    for c in ["Rank","Symbol","Cap","Status","Score","ADX","Deliv%","D/E","ROE","Rationale"]:
          tbl.append(f"<th>{c}</th>")
    tbl.append("</tr></thead><tbody>")
    for _, row in df_el_filtered.iterrows():
          s = row["Status"]
          if s == "Qualified":
              sb = '<span style="background:rgba(16,185,129,0.15);color:#34d399;border:1px solid rgba(16,185,129,0.3);padding:2px 6px;border-radius:10px;font-size:0.72rem;font-weight:600;">PASSED</span>'
              rd = str(int(row.get("Final_Rank", 999)))
              score_val = row.get('Factor_Score', row.get('Final_Composite_Score', None))
              sc = f"{score_val:.1f}/100" if score_val is not None and pd.notna(score_val) else "N/A"
              rt = '<span style="color:#a855f7;">PASSED ALL GATES</span>'
          else:
              sb = '<span style="background:rgba(239,68,68,0.15);color:#f87171;border:1px solid rgba(239,68,68,0.3);padding:2px 6px;border-radius:10px;font-size:0.72rem;font-weight:600;">ELIMINATED</span>'
              rd = '<span style="color:#64748b;">—</span>'
              sc = '<span style="color:#64748b;">—</span>'
              rt = f'<span style="color:#e2e8f0;">{row.get("Rejection_Reason","Failed")}</span>'
          adx_v = row.get("ADX_14", 0.0)
          deliv_v = row.get("Delivery_Pct", 0.0)
          de_v = row.get("Debt_to_Equity", 0.0)
          roe_v = row.get("ROE", 0.0)
          cap = str(row.get("Cap_Category", "—"))
          tbl.append(f"<tr><td style='text-align:center;font-weight:600;'>{rd}</td><td style='font-weight:700;color:#f1f5f9;'>{row['Symbol']}</td><td style='color:#94a3b8;text-align:center;'>{cap}</td><td style='text-align:center;'>{sb}</td><td style='text-align:center;font-weight:600;'>{sc}</td><td style='text-align:center;'>{adx_v:.1f}</td><td style='text-align:right;'>{deliv_v:.1f}%</td><td style='text-align:right;'>{de_v:.2f}</td><td style='text-align:right;'>{roe_v:.1f}%</td><td>{rt}</td></tr>")
    tbl.append("</tbody></table></div>")
    st.markdown("".join(tbl), unsafe_allow_html=True)
    st.markdown("---")
    # ── VAM-GQ Top 20 Allocation ──
    _vamgq_top20 = df_maac.copy()
    if not _vamgq_top20.empty and "Final_Rank" in _vamgq_top20.columns:
        _vamgq_top20 = _vamgq_top20[_vamgq_top20["Final_Rank"].notna() & (_vamgq_top20["Final_Rank"] <= 20)].sort_values("Final_Rank")
    if not _vamgq_top20.empty:
        st.subheader("🏆 VAM-GQ Master Portfolio — Top 20 Allocations")
        st.caption("Core satellite holdings for the 45-position master portfolio (20 VAM-GQ + 20 VAM-B + 5 Core)")
        _v20_html = ['<div style="overflow-x:auto;"><table class="elimination-table"><thead><tr>']
        for _v20_h in ["#","Symbol","Score","Fused","Alloc %","Stop Loss","Dist%","Risk%","Sector","Cap"]:
            _v20_html.append(f"<th>{_v20_h}</th>")
        _v20_html.append("</tr></thead><tbody>")
        for _, _v20r in _vamgq_top20.iterrows():
            _v20_sym = str(_v20r["Symbol"])
            _v20_ws = _v20r.get("Factor_Score",0)
            _v20_fus = _v20r.get("Final_Composite_Score",0)
            _v20_alloc = _v20r.get("Allocation_%",0)
            _v20_sl = _v20r.get("Stop_Loss",0)
            _v20_sd = _v20r.get("Stop_Dist_%",0)
            _v20_rp = _v20r.get("Risk_Per_Trade_%",0)
            _v20_sec = str(_v20r.get("Sector","—"))[:22]
            _v20_cap = str(_v20r.get("Cap_Category","—")).replace("_"," ")
            _v20_html.append(f"<tr><td style='font-weight:700;color:#fbbf24;'>{int(_v20r['Final_Rank'])}</td><td style='font-weight:700;color:#f1f5f9;'>{_v20_sym}</td><td style='color:#34d399;text-align:center;'>{_v20_ws:.1f}</td><td style='color:#818cf8;text-align:center;'>{_v20_fus:.1f}</td><td style='color:#34d399;text-align:center;'>{_v20_alloc:.1f}%</td><td style='color:#f87171;text-align:right;'>₹{_v20_sl:,.0f}</td><td style='color:#f87171;text-align:center;'>{_v20_sd:.1f}%</td><td style='color:#fbbf24;text-align:center;'>{_v20_rp:.2f}%</td><td style='color:#94a3b8;font-size:0.77rem;'>{_v20_sec}</td><td style='color:#64748b;font-size:0.74rem;'>{_v20_cap}</td></tr>")
        _v20_html.append("</tbody></table></div>")
        st.markdown("".join(_v20_html), unsafe_allow_html=True)
    st.markdown("---")
    st.info("Portfolio position ledger available in the Master Portfolio tab.")
# ──────────────────────────────────────────────────────────────────────────────
# TAB 3: CORE HOLDINGS — ETF / MF / INDEX FUNDS
# ──────────────────────────────────────────────────────────────────────────────
with tab_core:
    st.markdown("""
    <div style="background:linear-gradient(135deg,rgba(59,130,246,0.12),rgba(16,185,129,0.06));
                border:1px solid rgba(59,130,246,0.2); border-radius:14px;
                padding:16px 22px; margin-bottom:22px;">
    <div style="font-family:'Outfit';font-size:1.05rem;font-weight:700;color:#60a5fa;">
        🏛️ Core Holdings — Passive ETF, MF & Index Fund Allocation
    </div>
    <div style="font-size:0.82rem;color:#64748b;margin-top:4px;">
        <b style="color:#fbbf24;">TA 4.0 Blended Core Selection</b> · Max <b>5</b> holdings · Max <b>1</b> per category · Momentum-filtered (1M &amp; 3M) · Regime-weighted allocation
    </div>
    </div>
    """, unsafe_allow_html=True)
    
    _render_research_links(compact=True)
    render_unified_veto_ui("tab_core")
    st.caption("🏦 **Role:** Passive ETF/MF Selection · Momentum Rotation · Category Allocation")
    _core_ledger = list(_core_rows_global)
    _core_mf_map = _mf_name_map_global
    if not _synced_global:
        st.warning("⚠️ **Pipeline Incomplete:** The system has not finished writing today's orders. Core ledger metrics and additions may not reflect the final state.")
    # ── KPI Strip ────────────────────────────────────────────────────────────
    _c_hold = sum(1 for r in _core_ledger if r["status"] == "HOLD")
    _c_buy  = sum(1 for r in _core_ledger if r["status"] == "NEW BUY")
    _c_exit = sum(1 for r in _core_ledger if r["status"] == "EXIT")
    _c_dep  = sum(r["New_Alloc_%"] for r in _core_ledger if r["status"] in {"HOLD", "NEW BUY"})
    _c_val  = sum(r["Pos_Value"] for r in _core_ledger if r["status"] in {"HOLD", "NEW BUY"})
    _c_cats = len(set(r.get("Sector", "-") for r in _core_ledger if r["status"] != "EXIT"))
    _ck1, _ck2, _ck3, _ck4, _ck5 = st.columns(5)
    for _col_c, _lbl_c, _val_c, _clr_c in [
        (_ck1, "Core Holdings",   str(_c_hold),           "#10b981"),
        (_ck2, "New Additions",   str(_c_buy),            "#60a5fa"),
        (_ck3, "Exits (Today)",   str(_c_exit),           "#f87171"),
        (_ck4, "Deployed %",      f"{_c_dep:.1f}%",        "#818cf8"),
        (_ck5, "Deployed ₹",      f"₹{_c_val:,.0f}",      "#34d399"),
    ]:
        _col_c.markdown(f"""
        <div class="glass-card" style="text-align:center;padding:12px 8px;border-color:{_clr_c}30;">
            <div style="font-size:0.6rem;color:#64748b;text-transform:uppercase;letter-spacing:0.04em;">{_lbl_c}</div>
            <div style="font-family:'Outfit';font-weight:800;color:{_clr_c};font-size:1.35rem;margin-top:2px;">{_val_c}</div>
        </div>""", unsafe_allow_html=True)
    st.markdown('<div style="height:12px;"></div>', unsafe_allow_html=True)
    # ── TA 4.0 Core Selection — Max 5, Best 1 Per Category, Momentum-Filtered ──
    _cache_dir_core = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
    _sel_dt_core = pd.to_datetime(selected_date)
    
    # Load universe
    _core_univ_path = os.path.join(OUTPUT_DIR, "L1_Core_Universe.csv")
    _core_univ_path, _ = _resolve_pipeline_file("L1_Core_Universe.csv") if not os.path.exists(_core_univ_path) else (_core_univ_path, None)
    # Shared category normalizer used by both selection and display
    def _norm_cat(c):
        return str(c).replace(".xlsx","").replace("..",".").replace("_"," ").strip().lower()
    _ta4_core_selection = []
    if os.path.exists(_core_univ_path):
        try:
            _df_univ = pd.read_csv(_core_univ_path)
            if not _df_univ.empty and "Symbol" in _df_univ.columns:
                # Compute 15D, 1M (~21D), 2M (~42D), 3M (~63D), 6M (~126D) returns for each candidate
                _lookback_6m = _sel_dt_core - pd.Timedelta(days=190)  # buffer for weekends
                
                _returns_data = {}
                for _, _ur in _df_univ.iterrows():
                    _sym = str(_ur["Symbol"])
                    _hist_file = os.path.join(_cache_dir_core, f"{_sym}_history.csv")
                    if os.path.exists(_hist_file):
                        try:
                            _hdf = pd.read_csv(_hist_file, parse_dates=["Date"])
                            _hdf = _hdf.dropna(subset=["Close"]).set_index("Date")["Close"]
                            _hdf = _hdf[_hdf.index <= _sel_dt_core].sort_index()
                            if len(_hdf) >= 2:
                                _cut_15d = _sel_dt_core - pd.tseries.offsets.BDay(15)
                                _cut_1m = _sel_dt_core - pd.tseries.offsets.BDay(21)
                                _cut_2m = _sel_dt_core - pd.tseries.offsets.BDay(42)
                                _cut_3m = _sel_dt_core - pd.tseries.offsets.BDay(63)
                                _cut_6m = _sel_dt_core - pd.tseries.offsets.BDay(126)
                                
                                _h15d = _hdf[_hdf.index >= _cut_15d]
                                _h1m = _hdf[_hdf.index >= _cut_1m]
                                _h2m = _hdf[_hdf.index >= _cut_2m]
                                _h3m = _hdf[_hdf.index >= _cut_3m]
                                _h6m = _hdf[_hdf.index >= _cut_6m]
                                
                                _ret_15d = (_h15d.iloc[-1] / _h15d.iloc[0]) - 1.0 if len(_h15d) >= 2 else 0.0
                                _ret_1m = (_h1m.iloc[-1] / _h1m.iloc[0]) - 1.0 if len(_h1m) >= 2 else 0.0
                                _ret_2m = (_h2m.iloc[-1] / _h2m.iloc[0]) - 1.0 if len(_h2m) >= 2 else 0.0
                                _ret_3m = (_h3m.iloc[-1] / _h3m.iloc[0]) - 1.0 if len(_h3m) >= 2 else 0.0
                                _ret_6m = (_h6m.iloc[-1] / _h6m.iloc[0]) - 1.0 if len(_h6m) >= 2 else 0.0
                                
                                _returns_data[_sym] = (_ret_15d, _ret_1m, _ret_2m, _ret_3m, _ret_6m)
                        except:
                            pass
                
                # Score each candidate
                _candidates = []
                for _, _ur in _df_univ.iterrows():
                    _sym = str(_ur["Symbol"])
                    _cat = _norm_cat(_ur.get("Category", ""))
                    
                    _r_data = _returns_data.get(_sym)
                    if _r_data is not None:
                        _r15d, _r1m, _r2m, _r3m, _r6m = _r_data
                        # Composite Momentum Score (time-decay weighted)
                        # NOTE: Raw returns used (not annualized) → longer-period returns naturally
                        # dominate the score. For true momentum-agnostic ranking, annualize each return
                        # by dividing by its period (e.g., _r15d / (15/252), _r6m / (126/252)).
                        _W_15D = 0.20; _W_1M = 0.30; _W_2M = 0.25; _W_3M = 0.15; _W_6M = 0.10
                        _comp_score = (_W_15D * _r15d) + (_W_1M * _r1m) + (_W_2M * _r2m) + (_W_3M * _r3m) + (_W_6M * _r6m)
                    else:
                        _r1m, _r3m, _comp_score = None, None, float(_ur.get("Score", 0))
                    _name = _mf_name_map_global.get(_sym, str(_ur.get("Name", _sym)))
                    _is_trend = bool(_ur.get("Is_Trending", False))
                    _is_buy = bool(_ur.get("Is_Buy_Eligible", False))
                    
                    if _r1m is not None or _r3m is not None:
                        _candidates.append({
                            "symbol": _sym, "name": _name, "cat": _cat,
                            "score": _comp_score, "ret_1m": _r1m, "ret_3m": _r3m,
                            "trending": _is_trend, "buy_eligible": _is_buy
                        })
                
                # Group by category: compute avg category momentum
                _cat_1m_avg = {}
                _cat_3m_avg = {}
                _cat_counts = {}
                for _c in _candidates:
                    _cat = _c["cat"]
                    _cat_counts[_cat] = _cat_counts.get(_cat, 0) + 1
                    if _c["ret_1m"] is not None:
                        _cat_1m_avg[_cat] = _cat_1m_avg.get(_cat, 0.0) + _c["ret_1m"]
                    if _c["ret_3m"] is not None:
                        _cat_3m_avg[_cat] = _cat_3m_avg.get(_cat, 0.0) + _c["ret_3m"]
                for _cat in _cat_1m_avg:
                    _cat_1m_avg[_cat] /= _cat_counts.get(_cat, 1)
                for _cat in _cat_3m_avg:
                    _cat_3m_avg[_cat] /= _cat_counts.get(_cat, 1)
                
                # Filter: exclude categories where BOTH 1M and 3M avg is negative
                _eligible_cats = set()
                for _c in _candidates:
                    _cat = _c["cat"]
                    _c1m = _cat_1m_avg.get(_cat)
                    _c3m = _cat_3m_avg.get(_cat)
                    # Rule: exclude if BOTH 1M AND 3M category avg are negative
                    if (_c1m is not None and _c1m < 0) and (_c3m is not None and _c3m < 0):
                        continue
                    _eligible_cats.add(_cat)
                
                # Pick best 1 per eligible category (by Score), then take top 5
                _best_per_cat = {}
                for _c in _candidates:
                    if _c["cat"] in _eligible_cats:
                        if _c["cat"] not in _best_per_cat or _c["score"] > _best_per_cat[_c["cat"]]["score"]:
                            _best_per_cat[_c["cat"]] = _c
                
                _ta4_core_selection = sorted(_best_per_cat.values(), key=lambda x: -x["score"])[:5]
        except Exception as _e_sel:
            st.warning(f"⚠️ Core selection engine: {_e_sel}")
    
    # Display TA 4.0 Core Selection
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,rgba(251,191,36,0.10),rgba(245,158,11,0.05));
                border:1px solid rgba(251,191,36,0.25); border-radius:14px;
                padding:18px 22px; margin-bottom:18px;">
    <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;">
        <div style="font-family:'Outfit';font-size:1.1rem;font-weight:700;color:#fbbf24;">
          ⭐ TA 4.0 Core Selection — Max 5 Holdings
        </div>
        <span style="background:rgba(251,191,36,0.12);color:#fbbf24;border:1px solid rgba(251,191,36,0.3);padding:3px 10px;border-radius:16px;font-size:0.72rem;font-weight:700;">
          Max 1 per Category · Momentum-Filtered
        </span>
    </div>
    <div style="font-size:0.82rem;color:#94a3b8;margin-top:6px;">
        Categories with trailing 1M <b style="color:#ef4444;">AND</b> 3M returns both negative are excluded.
        Best-ranked fund selected per eligible category.
    </div>
    </div>
    """, unsafe_allow_html=True)
    
    if _ta4_core_selection:
        _sel_cats = len(_ta4_core_selection)
        _sel_cols = st.columns(min(_sel_cats, 5))
        for _si, _sel in enumerate(_ta4_core_selection):
            _sc = _sel_cols[_si % 5]
            _r1m_s = f"<span style='color:{'#34d399' if _sel['ret_1m'] and _sel['ret_1m'] >= 0 else '#f87171'}'>{_sel['ret_1m']*100:+.1f}%</span>" if _sel['ret_1m'] is not None else "<span style='color:#475569;'>—</span>"
            _r3m_s = f"<span style='color:{'#34d399' if _sel['ret_3m'] and _sel['ret_3m'] >= 0 else '#f87171'}'>{_sel['ret_3m']*100:+.1f}%</span>" if _sel['ret_3m'] is not None else "<span style='color:#475569;'>—</span>"
            _name_short = _sel['name'][:22] + ("…" if len(_sel['name']) > 22 else "")
            _cat_disp = _sel['cat'].title()
            _badge = '<span style="background:rgba(251,191,36,0.15);color:#fbbf24;border:1px solid rgba(251,191,36,0.3);padding:1px 6px;border-radius:8px;font-size:0.6rem;font-weight:700;">#' + str(_si+1) + '</span>'
            _sc.markdown(f"""
            <div class="glass-card" style="padding:14px 12px;border-left:4px solid #fbbf24;box-shadow:0 0 10px rgba(251,191,36,0.10);text-align:center;">
              <div style="font-size:0.65rem;color:#fbbf24;font-weight:700;text-transform:uppercase;letter-spacing:0.04em;">{_cat_disp}</div>
              <div style="font-size:0.78rem;color:#f1f5f9;font-weight:600;margin:6px 0 3px;line-height:1.3;">{_name_short}</div>
              <div style="display:flex;justify-content:center;gap:12px;margin:6px 0;">
                <div><span style="font-size:0.55rem;color:#64748b;text-transform:uppercase;">1M</span><br>{_r1m_s}</div>
                <div><span style="font-size:0.55rem;color:#64748b;text-transform:uppercase;">3M</span><br>{_r3m_s}</div>
                <div><span style="font-size:0.55rem;color:#64748b;text-transform:uppercase;">Score</span><br><span style="font-weight:700;color:#60a5fa;">{_sel['score']:.1f}</span></div>
              </div>
              <div style="font-size:0.6rem;color:#64748b;">{_badge}</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown('<div style="height:6px;"></div>', unsafe_allow_html=True)
    else:
        st.info("⚠️ Core universe data unavailable. Run pipeline to populate L1_Core_Universe.csv.")
    
    st.markdown('<div style="height:18px;"></div>', unsafe_allow_html=True)
    
    # ── Portfolio Composition Donut Chart + Concentration Warning ─────────────
    try:
        _ca_path_vis = os.path.join(OUTPUT_DIR, "L1_Core_Allocations.csv")
        if os.path.exists(_ca_path_vis):
            _df_ca_vis = pd.read_csv(_ca_path_vis)
        else:
            from db_manager import load_pipeline_stage as _lps_vis
            _df_ca_vis = _lps_vis("L1_Core_Allocations") or pd.DataFrame()
        if not _df_ca_vis.empty:
            if "Core_Weight" in _df_ca_vis.columns:
                # ── CSV path (L1_Core_Allocations) — has Category, Core_Weight ──
                _CAT_VIS_MAP = {
                    "global and inetrnation funds": "Global & Intl",
                    "global and international funds": "Global & Intl",
                    "small cap mutual funds": "Small Cap",
                    "mid cap mutual funds": "Mid Cap",
                    "large cap mutual funds": "Large Cap",
                    "flexi cap mutual funds": "Flexi Cap",
                    "broad market etf or index funds": "Broad Market",
                    "thematic etfs and index funds": "Thematic",
                    "thematic etfs and index funds.": "Thematic",
                    "business cycle and special oportunity fund": "Biz Cycle / Opp",
                    "comodities etfs": "Commodities",
                    "sectoral etfs - index funds": "Sectoral",
                    "strategy etfs and  index funds new": "Strategy ETF",
                    "strategy etfs and index funds new": "Strategy ETF",
                }
                def _vcat(raw):
                    c = _norm_cat(raw)
                    return _CAT_VIS_MAP.get(c, c.title())
                _df_ca_vis['_disp_cat'] = _df_ca_vis['Category'].apply(_vcat)
                _df_ca_vis['_disp_name'] = _df_ca_vis.apply(
                    lambda r: _mf_name_map_global.get(str(r['Symbol']), str(r.get('Name', r['Symbol']))), axis=1
                )
                _df_ca_vis['_pct'] = _df_ca_vis['Core_Weight'] * 100.0
            elif '_pct' in _df_ca_vis.columns:
                # Data already prepared from core ledger — skip normalisation
                pass
            # ── Category aggregation for concentration check ──────────────────
            _cat_agg = _df_ca_vis.groupby("_disp_cat")["_pct"].sum().reset_index()
            _cat_agg.columns = ["Category", "Weight%"]
            _cat_agg = _cat_agg.sort_values("Weight%", ascending=False)
            _max_cat = _cat_agg.iloc[0]
            # ── Concentration Warning ─────────────────────────────────────────
            _CONC_THRESHOLD = 35.0
            if _max_cat["Weight%"] >= _CONC_THRESHOLD:
                st.markdown(f"""
                <div style="background:linear-gradient(135deg,rgba(239,68,68,0.12),rgba(245,158,11,0.08));
                            border:1px solid rgba(239,68,68,0.4); border-left:4px solid #ef4444;
                            border-radius:10px; padding:12px 18px; margin:6px 0 16px 0;
                            display:flex; align-items:center; gap:12px;">
                    <span style="font-size:1.4rem;">⚠️</span>
                    <div>
                        <div style="font-family:'Outfit';font-weight:700;color:#fca5a5;font-size:0.9rem;">
                            Category Concentration Alert — Style Drift Risk
                        </div>
                        <div style="font-size:0.8rem;color:#94a3b8;margin-top:3px;">
                            <b style="color:#fbbf24;">{_max_cat['Category']}</b> holds
                            <b style="color:#ef4444;">{_max_cat['Weight%']:.1f}%</b> of the core portfolio
                            — exceeds the {_CONC_THRESHOLD:.0f}% category cap.
                            Consider capping allocations or diversifying into another category.
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            # ── Donut Chart + Category Breakdown side-by-side ─────────────────
            _col_donut, _col_breakdown = st.columns([1, 1], gap="large")
            with _col_donut:
                # Premium colour palette (one per category, consistent)
                _PALETTE = [
                    "#6366f1", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6",
                    "#06b6d4", "#f43f5e", "#84cc16", "#fb923c", "#a78bfa",
                ]
                _labels  = _df_ca_vis["_disp_name"].tolist()
                _parents = _df_ca_vis["_disp_cat"].tolist()
                _values  = _df_ca_vis["_pct"].tolist()
                _colors  = [_PALETTE[i % len(_PALETTE)] for i in range(len(_labels))]
                fig_donut = go.Figure()
                fig_donut.add_trace(go.Pie(
                    labels=_labels,
                    values=_values,
                    hole=0.62,
                    marker=dict(colors=_colors, line=dict(color="rgba(15,23,42,0.9)", width=2)),
                    textinfo="none",
                    hovertemplate="<b>%{label}</b><br>Allocation: <b>%{value:.1f}%</b><extra></extra>",
                    sort=False,
                ))
                # Centre annotation
                fig_donut.update_layout(
                    annotations=[dict(
                        text=f"<b style='font-size:22px'>{len(_labels)}</b><br><span style='font-size:11px;color:#94a3b8'>Holdings</span>",
                        x=0.5, y=0.5, font_size=14, showarrow=False,
                        font=dict(color="#f1f5f9", family="Outfit"),
                    )],
                    showlegend=False,
                    margin=dict(l=10, r=10, t=30, b=10),
                    height=310,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    title=dict(
                        text="<b>Core Portfolio Composition</b>",
                        font=dict(color="#94a3b8", size=13, family="Outfit"),
                        x=0.5, xanchor="center",
                    ),
                )
                st.plotly_chart(fig_donut, use_container_width=True, config={"displayModeBar": False})
            with _col_breakdown:
                st.markdown("""
                <div style="font-family:'Outfit';font-weight:700;font-size:0.85rem;
                            color:#94a3b8;text-transform:uppercase;letter-spacing:0.06em;
                            margin-bottom:10px;margin-top:30px;">
                    Category Breakdown
                </div>""", unsafe_allow_html=True)
                _breakdown_html = []
                for i, (_, _crow) in enumerate(_cat_agg.iterrows()):
                    _cw  = float(_crow["Weight%"])
                    _cc  = _PALETTE[i % len(_PALETTE)]
                    _bar_w = min(100, int(_cw / _cat_agg["Weight%"].max() * 100))
                    _warn = " ⚠️" if _cw >= _CONC_THRESHOLD else ""
                    _breakdown_html.append(f"""
                    <div style="margin-bottom:11px;">
                        <div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:4px;">
                            <span style="font-size:0.78rem;color:#f1f5f9;font-weight:600;">{_crow['Category']}{_warn}</span>
                            <span style="font-family:monospace;font-size:0.82rem;font-weight:700;color:{_cc};">{_cw:.1f}%</span>
                        </div>
                        <div style="background:rgba(255,255,255,0.06);border-radius:4px;height:6px;overflow:hidden;">
                            <div style="width:{_bar_w}%;height:100%;border-radius:4px;
                                        background:linear-gradient(90deg,{_cc}cc,{_cc}55);
                                        transition:width 0.5s ease;"></div>
                        </div>
                    </div>""")
                # Fund-level mini list
                st.markdown("".join(_breakdown_html), unsafe_allow_html=True)
                st.markdown("""<div style="border-top:1px solid rgba(255,255,255,0.05);
                                          margin:10px 0 8px;"></div>""", unsafe_allow_html=True)
                # Individual fund chips
                _chips = ""
                for _, _fr in _df_ca_vis.iterrows():
                    _fi   = list(_df_ca_vis.index).index(_fr.name)
                    _fc   = _PALETTE[_fi % len(_PALETTE)]
                    _fn   = str(_fr["_disp_name"])[:28] + ("…" if len(str(_fr["_disp_name"])) > 28 else "")
                    _fw   = float(_fr["_pct"])
                    _chips += f"""<span style="display:inline-flex;align-items:center;gap:5px;
                                               background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.07);
                                               border-left:3px solid {_fc};border-radius:6px;
                                               padding:4px 9px;margin:3px 3px;font-size:0.72rem;color:#cbd5e1;">
                                    <span style="color:{_fc};font-weight:700;">{_fw:.1f}%</span>
                                    {_fn}
                                 </span>"""
                st.markdown(f'<div style="line-height:2;">{_chips}</div>', unsafe_allow_html=True)
    except Exception as _e_vis:
        st.info(f"📊 Portfolio composition chart unavailable — data incomplete ({_e_vis}). Run pipeline for latest allocations.")
    st.markdown("""
    <style>
    .unified-ledger-table {
        width: 100% !important;
        border-collapse: collapse !important;
        background: rgba(30, 41, 59, 0.15) !important;
        border-radius: 12px !important;
        overflow: hidden !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        font-size: 0.85rem !important;
    }
    .unified-ledger-table th {
        background: rgba(59, 130, 246, 0.15) !important;
        color: #60a5fa !important;
        font-weight: 600 !important;
        font-family: 'Outfit', sans-serif !important;
        padding: 10px 8px !important;
        border-bottom: 2px solid rgba(59, 130, 246, 0.25) !important;
        font-size: 0.78rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
    }
    .unified-ledger-table td {
        padding: 10px 8px !important;
        vertical-align: middle !important;
        border-bottom: 1px solid rgba(255, 255, 255, 0.04) !important;
    }
    .unified-ledger-table tr:hover {
        background: rgba(255, 255, 255, 0.02) !important;
    }
    </style>
    """, unsafe_allow_html=True)
    def _render_ledger_core(rows, title):
        if not rows:
            return
        st.markdown(f"#### {title}")
        ledger_html = ["""<div style="overflow-x: auto; width: 100%; margin-bottom: 25px;">
<table class="unified-ledger-table">
<thead>
<tr>
<th style="width: 8%; text-align: center;">Status</th>
<th style="width: 4%; text-align: center;">Rank</th>
<th style="width: 22%; text-align: left;">Fund / ETF</th>
<th style="width: 12%; text-align: left;">Category</th>
<th style="width: 11%; text-align: right;">Price Profile</th>
<th style="width: 7%; text-align: center;">RS</th>
<th style="width: 12%; text-align: right;">Alloc Profile</th>
<th style="width: 8%; text-align: center;">Change</th>
<th style="width: 11%; text-align: right;">Exposure</th>
<th style="width: 15%; text-align: left;">Rationale</th>
</tr>
</thead>
<tbody>"""]
        for r in rows:
            s = r["status"]
            if s == "HOLD":
                status_badge = '<span style="background:rgba(16,185,129,0.15); color:#34d399; border:1px solid rgba(16,185,129,0.35); padding:3px 9px; border-radius:10px; font-size:0.72rem; font-weight:700;">🟢 HOLD</span>'
                sym_color = "#60a5fa"; row_style = ""
            elif s == "NEW BUY":
                status_badge = '<span style="background:rgba(59,130,246,0.15); color:#60a5fa; border:1px solid rgba(59,130,246,0.35); padding:3px 9px; border-radius:10px; font-size:0.72rem; font-weight:700;">🔵 BUY</span>'
                sym_color = "#818cf8"; row_style = "background: rgba(59,130,246,0.03);"
            else:
                status_badge = '<span style="background:rgba(239,68,68,0.15); color:#f87171; border:1px solid rgba(239,68,68,0.35); padding:3px 9px; border-radius:10px; font-size:0.72rem; font-weight:700;">🔴 EXIT</span>'
                sym_color = "#f87171"; row_style = "background: rgba(239,68,68,0.02); opacity: 0.8;"
            rank_str = str(r["Rank"]) if r["Rank"] < 99 else "-"
            disp = _mf_name_map_global.get(str(r["Symbol"]), r.get("Name", r.get("Display_Symbol", r["Symbol"])))
            sym_str = f'<b style="color:{sym_color}; font-size:0.9rem;">{disp}</b>'
            cat_str = f'<span style="color:#94a3b8;font-size:0.8rem;">{r.get("RealCategory", r.get("Sector", ""))[:25]}</span>'
            if s == "EXIT":
                price_str = f'<div style="font-weight:600; color:#f87171;">Exit: ₹{r["Close"]:,.2f}</div>'
            else:
                price_str = f'<div style="color:#94a3b8; font-size:0.75rem;">Close: ₹{r["Close"]:,.2f}</div>'
            rs_val_c = r.get("RS_Val", 0.0); ex_score_c = r.get("Exit_Score", 0.0)
            rs_color_c = "#10b981" if ex_score_c < 40 else ("#f59e0b" if ex_score_c < 55 else "#ef4444")
            rs_str = f'<span style="font-family:monospace; font-weight:600; color:{rs_color_c};">{rs_val_c:.3f}</span>'
            prev_a_c = r["Prev_Alloc_%"]; new_a_c = r["New_Alloc_%"]
            if s == "EXIT":
                alloc_str = f'''<div style="font-size:0.75rem; color:#94a3b8; margin-bottom:2px;">Prev: {prev_a_c:.2f}% <span style="color:#ef4444; font-weight:700;">→ 0.00%</span></div>
                <div style="width:100%; height:8px; background:rgba(255,255,255,0.05); border-radius:4px; position:relative;">
                    <div style="position:absolute; top:0; left:0; height:100%; background:rgba(239,68,68,0.3); width:{min(100, prev_a_c*4)}%; border-radius:4px;"></div>
                </div>'''
            else:
                alloc_str = f'''<div style="font-size:0.75rem; color:#94a3b8; margin-bottom:2px;">Prev: {prev_a_c:.2f}% <span style="color:#60a5fa; font-weight:700;">→ {new_a_c:.2f}%</span></div>
                <div style="width:100%; height:8px; background:rgba(255,255,255,0.05); border-radius:4px; position:relative;">
                    <div style="position:absolute; top:0; left:0; height:100%; background:rgba(148,163,184,0.3); width:{min(100, prev_a_c*4)}%; border-radius:4px;"></div>
                    <div style="position:absolute; top:0; left:0; height:100%; background:#60a5fa; width:{min(100, new_a_c*4)}%; border-radius:4px; opacity:0.8;"></div>
                </div>'''
            boost = r["Boost_Label"]
            boost_cell = f'<span style="color:#94a3b8; font-size:0.78rem;">{boost}</span>'
            pos_val_c = r["Pos_Value"]
            if s == "EXIT":
                exp_str = '<div style="color:#475569; text-align:right;">-</div>'
            else:
                exp_str = f'<div style="font-weight:700; color:#34d399; font-size:0.82rem;">₹{pos_val_c:,.0f}</div>'
            rat_color = "#f87171" if s == "EXIT" else ("#60a5fa" if s == "NEW BUY" else "#94a3b8")
            rat_str = f'<span style="color:{rat_color}; font-size:0.8rem;">{r["Rationale"]}</span>'
            ledger_html.append(f"""<tr style="{row_style}">
<td style="text-align:center;">{status_badge}</td>
<td style="text-align:center; font-family:monospace; color:#94a3b8;">{rank_str}</td>
<td>{sym_str}</td>
<td>{cat_str}</td>
<td style="text-align:right; font-family:monospace;">{price_str}</td>
<td style="text-align:center;">{rs_str}</td>
<td style="text-align:right; font-family:monospace;">{alloc_str}</td>
<td style="text-align:center;">{boost_cell}</td>
<td style="text-align:right;">{exp_str}</td>
<td>{rat_str}</td>
</tr>""")
        ledger_html.append("</tbody></table></div>")
        st.markdown("".join(ledger_html).replace("\n", ""), unsafe_allow_html=True)
    if _core_ledger:
        try:
            from config import CORE_ETF_ZONES
            
            # Map actual categories for the ledger
            mf_rows = []
            debt_gold_rows = []
            equity_rows = []
            
            for _cr in _core_ledger:
                sym = str(_cr["Symbol"])
                actual_cat = str(CORE_ETF_ZONES.get(sym, _cr.get("Sector", "Uncategorized"))).replace(".xlsx", "").replace("_", " ").title()
                _cr["RealCategory"] = actual_cat
                
                cat_lower = actual_cat.lower()
                if "mutual fund" in cat_lower or "flexi cap" in cat_lower or "biz cycle" in cat_lower or "small cap" in cat_lower or "mid cap" in cat_lower or "large cap" in cat_lower:
                    mf_rows.append(_cr)
                elif "gold" in cat_lower or "silver" in cat_lower or "commodit" in cat_lower or "debt" in cat_lower or "bond" in cat_lower or "liquid" in cat_lower:
                    debt_gold_rows.append(_cr)
                else:
                    equity_rows.append(_cr)
                    
            if not mf_rows and not debt_gold_rows and not equity_rows:
                st.info("No core holdings data available. Run the pipeline to populate.")
                
            _render_ledger_core(equity_rows, "📈 Equity ETFs")
            _render_ledger_core(debt_gold_rows, "🛡️ Debt & Gold")
            _render_ledger_core(mf_rows, "🏦 Mutual Funds")
        except Exception as e:
            st.warning(f"⚠️ **Table Rendering Failed:** {e}. Falling back to flat ledger view.")
            st.subheader("📋 Unified Core Portfolio Allocation Table")
            _render_ledger_core(_core_ledger)
    else:
        st.info("No Core ETF/MF holdings detected. Run the pipeline to populate allocation data.")
    # ── Core Portfolio Candidate Rankings ────────────────────────────────────
    st.markdown('<div style="border-top:1px solid rgba(255,255,255,0.05);margin:24px 0 16px;"></div>', unsafe_allow_html=True)
    sec_title("🛡️", "Core Portfolio Candidate Analysis & Rankings")
    _core_file_tab = os.path.join(OUTPUT_DIR, "L1_Core_Universe.csv")
    _core_alloc_tab = os.path.join(OUTPUT_DIR, "L1_Core_Allocations.csv")
    if not os.path.exists(_core_file_tab):
        # Auto-fallback to latest date that has the file
        _cf_path, _cf_date = _resolve_pipeline_file("L1_Core_Universe.csv")
        if _cf_path:
            _core_file_tab = _cf_path
            st.info(f"📁 Using L1_Core_Universe from **{_cf_date}** (selected date {selected_date} has incomplete pipeline data)")
    if os.path.exists(_core_file_tab):
        # BUG FIX 4: Read directly from dated CSV (bypass SQLite which has no date scope)
        df_core_t = pd.read_csv(_core_file_tab)
        _alloc_syms_t = []
        _alloc_weights_t = {}
        if os.path.exists(_core_alloc_tab):
            # BUG FIX 5: Only run mtime check when both CSV files actually exist on disk
            mtime_univ = os.path.getmtime(_core_file_tab)
            mtime_alloc = os.path.getmtime(_core_alloc_tab)
            if abs(mtime_univ - mtime_alloc) > 300:
                st.warning("⚠️ **Pipeline Interrupted:** Universe and allocation files are from different runs. Displayed data may be out of sync.")
            
            _df_ca_t = pd.read_csv(_core_alloc_tab)  # Also bypass SQLite for date-accuracy
            _alloc_syms_t = [str(s) for s in _df_ca_t["Symbol"].tolist()]
            _alloc_weights_t = dict(zip(_df_ca_t["Symbol"].astype(str), _df_ca_t["Core_Weight"]))
        if not df_core_t.empty:
            # BUG FIX 3: Category name normalization map (strips .xlsx, corrects typos)
            _CAT_DISPLAY_MAP = {
                "global and inetrnation funds": "Global & International Funds",
                "global and international funds": "Global & International Funds",
                "small cap mutual funds": "Small Cap Mutual Funds",
                "mid cap mutual funds": "Mid Cap Mutual Funds",
                "large cap mutual funds": "Large Cap Mutual Funds",
                "flexi cap mutual funds": "Flexi Cap Mutual Funds",
                "broad market etf or index funds": "Broad Market ETF / Index Funds",
                "thematic etfs and index funds": "Thematic ETFs & Index Funds",
                "thematic etfs and index funds.": "Thematic ETFs & Index Funds",
                "business cycle and special oportunity fund": "Business Cycle & Special Opportunities",
                "comodities etfs": "Commodities ETFs",
                "sectoral etfs - index funds": "Sectoral ETFs & Index Funds",
                "strategy etfs and  index funds new": "Strategy ETFs & Index Funds",
                "strategy etfs and index funds new": "Strategy ETFs & Index Funds",
            }
            def _normalize_cat(raw):
                cleaned = str(raw).replace(".xlsx", "").replace("..", ".").replace("_", " ").strip().lower()
                return _CAT_DISPLAY_MAP.get(cleaned, cleaned.title())
            _global_mf_n = _mf_name_map_global  # reuse cached from line 4882
            st.markdown("##### 🏆 Unified Core Ranking (All Categories)")
            # BUG FIX 2: Removed redundant raw Symbol column; merged it as secondary label under Fund Name
            _unified_html = ['<div style="overflow-x:auto; margin-bottom:25px;"><table class="elimination-table"><thead><tr>']
            for _uh in ["Rank", "Category", "Fund Name", "RS Rating", "Risk (Max DD)", "Volatility", "Momentum Score"]:
                _unified_html.append(f"<th>{_uh}</th>")
            _unified_html.append("</tr></thead><tbody>")
            
            _df_unified = df_core_t.sort_values(by="Rank", ascending=True)
            for _, _ur in _df_unified.iterrows():
                _usym = str(_ur.get("Symbol", "-"))
                _uname = _global_mf_n.get(_usym, str(_ur.get("Name", _usym)))
                _ucat = _normalize_cat(_ur.get("Category", "Uncategorized"))
                _urs = float(_ur.get("RS_Rating", 0.0))
                _urs_color = "#10b981" if _urs > 50 else ("#f59e0b" if _urs > 20 else "#ef4444")
                _udd = float(_ur.get("Drawdown_252", 0.0))
                _udd_color = "#ef4444" if _udd > 0.03 else "#34d399"
                _uvol = float(_ur.get("Volatility", 0.0))
                _umom = float(_ur.get("Score", 0.0))
                _umom_color = "#10b981" if _umom > 70 else ("#60a5fa" if _umom > 40 else "#94a3b8")
                # BUG FIX 2: Show name as primary + AMFI code as small secondary label
                _name_cell = f"<b>{_uname}</b><br><span style='font-size:0.7rem;color:#475569;'>AMFI: {_usym}</span>" if _usym.isdigit() else f"<b>{_uname}</b>"
                _unified_html.append(f"<tr><td style='text-align:center;'>{int(_ur.get('Rank', 999))}</td>"
                                     f"<td><span style='font-size:0.75rem;color:#94a3b8;'>{_ucat}</span></td>"
                                     f"<td><span style='font-size:0.85rem;'>{_name_cell}</span></td>"
                                     f"<td style='text-align:right;color:{_urs_color}; font-weight:600;'>{_urs:.1f}</td>"
                                     f"<td style='text-align:right;color:{_udd_color};'>{_udd*100:.1f}%</td>"
                                     f"<td style='text-align:right;'>{_uvol:.1f}%</td>"
                                     f"<td style='text-align:right;color:{_umom_color};font-weight:700;'>{_umom:.1f}</td></tr>")
            _unified_html.append("</tbody></table></div>")
            st.markdown("".join(_unified_html), unsafe_allow_html=True)
            
            st.markdown("##### 📂 Category-Based Analysis")
            if "Category" in df_core_t.columns:
                df_core_t["Category"] = df_core_t["Category"].fillna("Uncategorized")
                df_core_t = df_core_t.sort_values(by=["Category", "Rank"], ascending=[True, True])
            _core_html_t = ['<div style="overflow-x:auto; margin-bottom:25px;"><table class="elimination-table"><thead><tr>']
            for _h_t in ["Rank", "Fund / ETF", "Selection Status", "Quality Score", "Regime Stage", "Drawdown (Today vs 52W High)", "Volatility", "Allocated Weight", "Current Price"]:
                _core_html_t.append(f"<th>{_h_t}</th>")
            _core_html_t.append("</tr></thead><tbody>")
            _cur_cat_t = None
            for _, _cr_t in df_core_t.iterrows():
                # BUG FIX 3: use normalize_cat function for consistent display
                _ccat_t = _normalize_cat(_cr_t.get("Category", "Uncategorized"))
                if _ccat_t != _cur_cat_t:
                    _cur_cat_t = _ccat_t
                _core_html_t.append(f'''<tr style="background:rgba(59,130,246,0.15); border-bottom: 2px solid rgba(59,130,246,0.3);">
                            <td colspan="10" style="padding:10px 15px; font-weight:800; font-size:0.9rem; color:#60a5fa; letter-spacing:1px; text-transform:uppercase;">
                            📁 {_ccat_t.upper()}
                            </td></tr>''')
                _crank_t = int(_cr_t.get("Rank", 999)); _csym_t = str(_cr_t.get("Symbol", "-"))
                _cname_t = _global_mf_n.get(_csym_t, str(_cr_t.get("Name", "")))
                    # BUG FIX 2: Show clean fund name in category table (not raw AMFI code)
                _disp_sym_t = f"<b>{_cname_t}</b><br><span style='font-size:0.7rem;color:#475569;'>AMFI: {_csym_t}</span>" if _csym_t.isdigit() and _cname_t else (f"<b>{_cname_t}</b>" if _cname_t else f"<b>{_csym_t}</b>")
                _cweight_t = float(_alloc_weights_t.get(_csym_t, 0.0))
                    # BUG FIX: Use Score (actual col name) instead of Quality_Score (doesn't exist in CSV)
                _cscore_t = float(_cr_t.get("Score", _cr_t.get("Quality_Score", 0.0)))
                    # BUG FIX: Use Is_Trending + Is_Buy_Eligible (actual cols) instead of Regime_Stage (not in CSV)
                _is_trending_t = bool(_cr_t.get("Is_Trending", False))
                _is_buy_elig_t = bool(_cr_t.get("Is_Buy_Eligible", False))
                _cdrawdown_t = float(_cr_t.get("Drawdown_252", 0.0))
                _cvol_t = float(_cr_t.get("Volatility", 0.0))
                _cprice_t = float(_cr_t.get("Close", 0.0))
                    # Regime badge: built from Is_Trending + Is_Buy_Eligible
                if _is_trending_t and _is_buy_elig_t:
                    _rbadge_t = '<span style="background:rgba(16,185,129,0.15);color:#34d399;border:1px solid rgba(16,185,129,0.3);padding:2px 7px;border-radius:10px;font-size:0.67rem;font-weight:700;">✅ TRENDING · BUY-ELIGIBLE</span>'
                elif _is_trending_t:
                    _rbadge_t = '<span style="background:rgba(245,158,11,0.15);color:#fbbf24;border:1px solid rgba(245,158,11,0.3);padding:2px 7px;border-radius:10px;font-size:0.67rem;font-weight:700;">📈 TRENDING</span>'
                else:
                    _rbadge_t = '<span style="background:rgba(148,163,184,0.12);color:#94a3b8;border:1px solid rgba(148,163,184,0.25);padding:2px 7px;border-radius:10px;font-size:0.67rem;">⏸ CONSOLIDATING</span>'
                if _csym_t in _alloc_syms_t:
                    _sbadge_t = '<span style="background:rgba(16,185,129,0.18);color:#34d399;border:1px solid rgba(16,185,129,0.35);padding:2px 7px;border-radius:10px;font-size:0.67rem;font-weight:700;">✅ SELECTED</span>'
                    _rstyle_t = "background: rgba(16,185,129,0.04);"
                    _score_str_t = f'<b style="color:#60a5fa;">{_cscore_t:.1f}</b>'
                    _wt_str_t = f'<b style="color:#34d399;">{_cweight_t*100:.1f}%</b>'
                elif _is_buy_elig_t:
                    _sbadge_t = '<span style="background:rgba(59,130,246,0.15);color:#60a5fa;border:1px solid rgba(59,130,246,0.3);padding:2px 7px;border-radius:10px;font-size:0.67rem;font-weight:700;">🔵 ELIGIBLE</span>'
                    _rstyle_t = "background: rgba(59,130,246,0.02);"
                    _score_str_t = f'<span style="color:#60a5fa;">{_cscore_t:.1f}</span>'
                    _wt_str_t = '<span style="color:#475569;">-</span>'
                else:
                    _sbadge_t = '<span style="background:rgba(148,163,184,0.15);color:#94a3b8;border:1px solid rgba(148,163,184,0.35);padding:2px 7px;border-radius:10px;font-size:0.67rem;font-weight:700;">⏳ PENDING</span>'
                    _rstyle_t = "background: rgba(148,163,184,0.02); opacity: 0.85;"
                    _score_str_t = f'<span style="color:#64748b;">{_cscore_t:.1f}</span>'
                    _wt_str_t = '<span style="color:#475569;">-</span>'
                _dd_color_t = "#ef4444" if _cdrawdown_t > 0.05 else ("#f59e0b" if _cdrawdown_t > 0.02 else "#34d399")
                _dd_str_t = f'<span style="color:{_dd_color_t};">{_cdrawdown_t*100:.1f}%</span>'
                _core_html_t.append(f'''<tr style="{_rstyle_t}">
                    <td style="text-align:center; font-family:monospace; color:#94a3b8;">{_crank_t}</td>
                    <td>{_disp_sym_t}</td>
                    <td style="text-align:center;">{_sbadge_t}</td>
                    <td style="text-align:right;">{_score_str_t}</td>
                    <td style="text-align:center;">{_rbadge_t}</td>
                    <td style="text-align:right;">{_dd_str_t}</td>
                    <td style="text-align:right; color:#94a3b8;">{_cvol_t:.1f}%</td>
                    <td style="text-align:right;">{_wt_str_t}</td>
                    <td style="text-align:right; font-family:monospace; color:#94a3b8;">₹{_cprice_t:,.2f}</td>
                    </tr>'''
                    )
            _core_html_t.append("</tbody></table></div>")
            st.markdown("".join(_core_html_t), unsafe_allow_html=True)
    else:
        st.info("Core Portfolio Universe file not found for this date. Run the pipeline to generate L1_Core_Universe.csv.")
# ──────────────────────────────────────────────────────────────────────────────
# TAB 4: VAMS HOLDINGS — VOLATILITY ADJUSTED MOMENTUM
# ──────────────────────────────────────────────────────────────────────────────
with tab_vams:
    st.markdown("""
    <div style="background:linear-gradient(135deg,rgba(245,158,11,0.12),rgba(234,179,8,0.06));
                border:1px solid rgba(245,158,11,0.2); border-radius:14px;
                padding:16px 22px; margin-bottom:22px;">
    <div style="font-family:'Outfit';font-size:1.05rem;font-weight:700;color:#fbbf24;">
        ⚡ VAM-B (Volatility Adjusted Momentum — Blended) — Screening & Rankings
    </div>
    <div style="font-size:0.82rem;color:#94a3b8;margin-top:4px;">
        Z-Return minus Z-Volatility score over 63 days · Raw Chartink universe, no quality gates · Highest risk-adjusted momentum ranked first.
        Top 20 feed into <b style="color:#a78bfa;">Satellite Holdings</b> in the Master Portfolio.
    </div>
    </div>
    """, unsafe_allow_html=True)
    
    _render_research_links(compact=True)
    render_unified_veto_ui("tab_vams")
    st.caption("🔬 **Role:** Screening Lab — Pure Momentum (No Quality Gates) · VAM-B Rankings")
    # T4-12: Handle Empty MAAC Data Gracefully
    if df_maac.empty:
        st.warning("⚠️ **VAM universe unavailable:** Main pipeline hasn't populated MAAC for the selected date yet.")
        st.stop()
    # Use the entire non-CORE MAAC universe as VAM-B raw universe (no quality gates)
    _df_vams_all = df_maac.copy()
    if "Tier" in _df_vams_all.columns:
        _df_vams_all = _df_vams_all[~_df_vams_all["Tier"].astype(str).str.contains("CORE", case=False)]
    if "Symbol" in _df_vams_all.columns:
        _df_vams_all = _df_vams_all[~_df_vams_all["Symbol"].astype(str).str.match(r"^(NIFTY_|STRATEGY_|INDIA_VIX|MCX_)", case=False)]
    # ── VAM-B Quality Filter: ROE ≥ 3%, ROCE > 0, CFO/PAT > 0 ──
    # Matches main.py VAM-B injection gate thresholds. ROE floor aligned with stock_selector.
    # "Positives not negatives" — reject only confirmed negatives; missing data = pass
    _vamb_n_pre = len(_df_vams_all)
    _vamb_qc = []
    for _idx_q, _row_q in _df_vams_all.iterrows():
        _roe_q  = float(_row_q.get("ROE", 0) or 0)
        _roce_q = float(_row_q.get("ROCE", _row_q.get("roce_3yr", _row_q.get("ROE", 0))) or 0)
        _cfo_q  = float(_row_q.get("CFO_to_PAT", _row_q.get("cfo_pat_3yr", 0.5)) or 0)
        _vamb_qc.append(_roe_q >= 3.0 and _roce_q > 0 and _cfo_q > 0)
    _df_vams_all = _df_vams_all[pd.Series(_vamb_qc, index=_df_vams_all.index)].copy()
    _vamb_n_post = len(_df_vams_all)
    _vamb_filtered = _vamb_n_pre - _vamb_n_post
    if _vamb_filtered > 0:
        st.info(f"🟢 **VAM-B Quality Gate:** {_vamb_filtered} stock(s) removed (ROE < 3% or ROCE/CFO-PAT ≤ 0). {_vamb_n_post} remain for VAM scoring.")
    _vam_symbols_t = _df_vams_all["Symbol"].tolist() if not _df_vams_all.empty else []
    _df_vam_t = pd.DataFrame()
    if _vam_symbols_t:
        _cache_dir_vt = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
        _vam_data_t = {}
        _corrupt_vams = []
        for _sym_vt in _vam_symbols_t:
            _cfile_vt = os.path.join(_cache_dir_vt, f"{_sym_vt}_history.csv")
            if os.path.exists(_cfile_vt):
                try:
                    _dfc_vt = read_data_smart(_cfile_vt)
                    _dfc_vt["Date"] = pd.to_datetime(_dfc_vt["Date"])
                    _s_vt = _dfc_vt.dropna(subset=["Close"]).set_index("Date")["Close"]
                    if not _s_vt.empty:
                        _vam_data_t[_sym_vt] = _s_vt.tail(100)
                except Exception:
                    _corrupt_vams.append(_sym_vt)
        if _corrupt_vams:
            st.warning(f"⚠️ **Corrupted Cache Files:** Failed to load {len(_corrupt_vams)} symbols (e.g., {', '.join(_corrupt_vams[:5])}). Excluded from ranking.")
        if _vam_data_t:
            _selected_dt = pd.to_datetime(selected_date)
            _pm_vam_t = pd.DataFrame(_vam_data_t).sort_index()
            _pm_vam_t = _pm_vam_t[_pm_vam_t.index <= _selected_dt]
            if len(_pm_vam_t) > 0:
                _pm_vam_t = _pm_vam_t.tail(63)
                _valid_vt = []; _ret_list = []; _vol_list = []
                for _sym_vt in _pm_vam_t.columns:
                    _sym_s = _pm_vam_t[_sym_vt].dropna()
                    if len(_sym_s) >= 45 and _sym_s.index[-1] >= _selected_dt - pd.Timedelta(days=5):
                        _ret63 = (_sym_s.iloc[-1] / _sym_s.iloc[0]) - 1.0
                        # ── Velocity Rejection Gate (same as stock_selector.py) ──
                        _vel_reject = False
                        if _ret63 > 1.50:
                            _vel_reject = True
                        elif len(_sym_s) >= 21:
                            _ret21 = (_sym_s.iloc[-1] / _sym_s.iloc[-21]) - 1.0
                            if _ret21 > 0.40 and _ret63 > 0.50:
                                _steep = _ret21 / (_ret63 / 3.0)
                                if _steep > 2.5:
                                    _vel_reject = True
                        if _vel_reject:
                            continue
                        _vol63 = _sym_s.pct_change().std() * np.sqrt(252)
                        if _vol63 > 0:
                            _valid_vt.append(_sym_vt); _ret_list.append(_ret63); _vol_list.append(_vol63)
                if _valid_vt:
                    _ret_v_t = pd.Series(_ret_list, index=_valid_vt)
                    _vol_v_t = pd.Series(_vol_list, index=_valid_vt)
                    _std_ret_t = max(_ret_v_t.std(), 1.0) if pd.notna(_ret_v_t.std()) and _ret_v_t.std() > 0 else 1.0
                    _std_vol_t = max(_vol_v_t.std(), 1.0) if pd.notna(_vol_v_t.std()) and _vol_v_t.std() > 0 else 1.0
                    _z_ret_t = (_ret_v_t - _ret_v_t.mean()) / _std_ret_t
                    _z_vol_t = (_vol_v_t - _vol_v_t.mean()) / _std_vol_t
                    _df_vam_t = pd.DataFrame({
                        "Symbol": _valid_vt, 
                        "Return (63d)": [r * 100.0 for r in _ret_list], 
                        "Volatility (63d)": [v * 100.0 for v in _vol_list],
                        "Z-Return": _z_ret_t.values, 
                        "Z-Volatility": _z_vol_t.values,
                        "VAM Score": (_z_ret_t - _z_vol_t).values
                    }).sort_values("VAM Score", ascending=False).reset_index(drop=True)
                    _df_vam_t.index = _df_vam_t.index + 1  # 1-based rank
    # ── Extension from SMA-200 (Rubber Band) Helpers ─────────────────────
    _cache_dir_ext = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
    _sel_date_ext = pd.to_datetime(selected_date)

    def _calc_extension_raw(sym):
        """Returns (Close / SMA_200 - 1) as decimal (0.40 = 40% above SMA-200)."""
        _cf = os.path.join(_cache_dir_ext, f"{sym}_history.csv")
        if not os.path.exists(_cf):
            return 0.0
        try:
            _d = read_data_smart(_cf)
            _d["Date"] = pd.to_datetime(_d["Date"])
            _d = _d.dropna(subset=["Close"]).set_index("Date")["Close"]
            _d = _d[_d.index <= _sel_date_ext]
            if len(_d) < 200:
                return 0.0
            _c = float(_d.iloc[-1])
            _s = float(_d.rolling(200).mean().iloc[-1])
            if pd.notna(_s) and _s > 0:
                return (_c / _s) - 1.0
        except Exception:
            pass
        return 0.0

    def _extension_weight_mult(ext_dec):
        """Returns position-sizing multiplier based on extension severity."""
        if ext_dec > 2.0:    # >200% above SMA-200 → skip
            return 0.0
        elif ext_dec > 1.0:  # >100% above SMA-200 → reject (parabolic)
            return 0.0
        elif ext_dec > 0.40: # >40% → 50% weight
            return 0.50
        return 1.0           # Normal weight

    def _ext_badge(ext_dec):
        """Returns (badge_icon, badge_color) for visual display."""
        if ext_dec > 2.0:
            return "⛔", "#ef4444"
        elif ext_dec > 1.0:
            return "🔴", "#f97316"
        elif ext_dec > 0.40:
            return "⚠️", "#f59e0b"
        return "✅", "#10b981"

    # ════════════════════════════════════════════════════════════════════════════
    # 🚀 TA 4.0 SATELLITE SELECTION — Top 20 VAM-B + Top 20 VAM-GQ (Pure Momentum)
    # ════════════════════════════════════════════════════════════════════════════
    st.markdown("""
    <div style="background:linear-gradient(135deg,rgba(139,92,246,0.12),rgba(16,185,129,0.06));
                border:1px solid rgba(139,92,246,0.25); border-radius:14px;
                padding:16px 22px; margin-bottom:18px;">
    <div style="font-family:'Outfit';font-size:1.05rem;font-weight:700;color:#a78bfa;">
        🚀 TA 4.0 Satellite Selection — Top 20 VAM-B + Top 20 VAM-GQ
    </div>
    <div style="font-size:0.82rem;color:#94a3b8;margin-top:4px;">
        Candidates scored by <b>Return + Risk + Momentum + Volatility</b> only (Z-Return − Z-Volatility over 63D).
        No quality gates. Top 20 from each track merged, deduplicated, and re-ranked by combined momentum.
        Top-ranked stocks become <b style="color:#a78bfa;">Satellite Holdings</b>.
    </div>
    </div>
    """, unsafe_allow_html=True)
    try:
        # Step 1: Get VAM-B Top 20 (already computed in tab)
        _vamb_top = []
        if '_df_vam_t' in locals() and not _df_vam_t.empty:
            for _, _vr in _df_vam_t.head(20).iterrows():
                _vamb_top.append({
                    "symbol": _vr["Symbol"], "track": "VAM-B",
                    "ret_3m": float(_vr.get("Return (63d)", 0)),
                    "vol": float(_vr.get("Volatility (63d)", 0)),
                    "z_ret": float(_vr.get("Z-Return", 0)),
                    "z_vol": float(_vr.get("Z-Volatility", 0)),
                    "vam_score": float(_vr.get("VAM Score", 0)),
                })
        # Step 2: VAM-GQ Top 20 — use existing pipeline scores (already quality-gate verified)
        _vamgq_top = []
        if not df_maac.empty and "Symbol" in df_maac.columns and "Entry_Eligible" in df_maac.columns:
            _df_vg = df_maac.copy()
            if "Tier" in _df_vg.columns:
                _df_vg = _df_vg[~_df_vg["Tier"].astype(str).str.contains("CORE", case=False)]
            # Only Entry_Eligible stocks (top 50 from pipeline, already quality-verified)
            _df_vg_eligible = _df_vg[_df_vg["Entry_Eligible"] == True].copy()
            # Sort by Factor_Score descending (the 8-factor composite)
            if "Factor_Score" in _df_vg_eligible.columns:
                _df_vg_eligible = _df_vg_eligible.sort_values("Factor_Score", ascending=False).head(20)
            elif "Final_Composite_Score" in _df_vg_eligible.columns:
                _df_vg_eligible = _df_vg_eligible.sort_values("Final_Composite_Score", ascending=False).head(20)
            else:
                _df_vg_eligible = _df_vg_eligible.head(20)
            # Parse factor details for display
            for _, _vr in _df_vg_eligible.iterrows():
                _sym_vg = str(_vr["Symbol"])
                _fs = float(_vr.get("Factor_Score", _vr.get("Final_Composite_Score", 0)) or 0)
                _mom = 0; _sec = 0; _del = 0; _gro = 0; _pea = 0; _fii = 0
                try:
                    _fd = json.loads(str(_vr.get("Factor_Details", "{}")))
                    _mom = int(_fd.get("F3_MOMENTUM", {}).get("score", 0) or 0)
                    _sec = int((_fd.get("F1_SECTORAL_TREND", {}).get("score", 0) or 0) * 0.6 + (_fd.get("F2_THEMATIC_TREND", {}).get("score", 0) or 0) * 0.4)
                    _del = int(_fd.get("F6_DELIVERY_CONFIRMATION", {}).get("score", 0) or 0)
                    _gro = int(_fd.get("F4_GROWTH", {}).get("score", 0) or 0)
                    _pea = int(_fd.get("F7_PEAD", {}).get("score", 0) or 0)
                    _fii = int(_fd.get("F8_FII_DII_CONVICTION", {}).get("score", 0) or 0)
                except:
                    pass
                _vamgq_top.append({
                    "symbol": _sym_vg, "track": "VAM-GQ",
                    "fs": _fs, "mom": _mom, "sec": _sec,
                    "del": _del, "gro": _gro, "pea": _pea, "fii": _fii,
                })
        # Step 3: Merge, deduplicate, re-rank by unified score
        _merged = {}
        for _s in _vamb_top:
            _s["unified_score"] = _s["vam_score"]
            _merged[_s["symbol"]] = _s
        for _s in _vamgq_top:
            _s["unified_score"] = _s["fs"]
            if _s["symbol"] in _merged:
                _existing = _merged[_s["symbol"]]
                _existing["track"] = "VAM-B + VAM-GQ"
                _existing["unified_score"] = _existing["vam_score"] * 0.5 + _s["fs"] * 2.0
                _existing["fs"] = _s["fs"]
                _existing["mom"] = _s["mom"]; _existing["sec"] = _s["sec"]
                _existing["del"] = _s["del"]; _existing["gro"] = _s["gro"]
                _existing["pea"] = _s["pea"]; _existing["fii"] = _s["fii"]
            else:
                _merged[_s["symbol"]] = _s
        _satellite_selection = sorted(_merged.values(), key=lambda x: x["unified_score"], reverse=True)
        # ── Rubber Band (Extension) Penalty: downsize over-extended stocks ──
        for _s in _satellite_selection:
            _s["ext_raw"] = _calc_extension_raw(_s["symbol"])
            _s["ext_pct"] = round(_s["ext_raw"] * 100.0, 1)  # display %
            _s["ext_mult"] = _extension_weight_mult(_s["ext_raw"])
            _s["unified_score"] *= _s["ext_mult"]
        # Re-rank after extension penalty
        _satellite_selection.sort(key=lambda x: x["unified_score"], reverse=True)
        if _satellite_selection:
            _all_scores = [s["unified_score"] for s in _satellite_selection]
            _min_s = min(_all_scores); _max_s = max(_all_scores)
            _range_s = max(_max_s - _min_s, 0.01)
            for _s in _satellite_selection:
                _s["composite"] = max(0, min(100, (_s["unified_score"] - _min_s) / _range_s * 100))
        if _satellite_selection:
            _vamb_count = sum(1 for s in _satellite_selection if s["track"] == "VAM-B")
            _vamgq_count = sum(1 for s in _satellite_selection if s["track"] == "VAM-GQ")
            _total_distinct = len(_satellite_selection)
            _overlap = _vamb_count + _vamgq_count - _total_distinct
            _ks1, _ks2, _ks3, _ks4 = st.columns(4)
            _ks1.markdown(f'<div class="glass-card" style="text-align:center;padding:10px;"><div style="font-size:0.65rem;color:#64748b;">Total Distinct</div><div style="font-family:Outfit;font-weight:800;color:#a78bfa;font-size:1.3rem;">{_total_distinct}</div><div style="font-size:0.6rem;color:#475569;">VAM-B {_vamb_count} + VAM-GQ {_vamgq_count}</div></div>', unsafe_allow_html=True)
            _ks2.markdown(f'<div class="glass-card" style="text-align:center;padding:10px;"><div style="font-size:0.65rem;color:#64748b;">Overlap</div><div style="font-family:Outfit;font-weight:800;color:#fbbf24;font-size:1.3rem;">{_overlap}</div><div style="font-size:0.6rem;color:#475569;">In both VAM-B & VAM-GQ</div></div>', unsafe_allow_html=True)
            _top5_avg = np.mean([s["composite"] for s in _satellite_selection[:5]]) if len(_satellite_selection) >= 5 else 0
            _ks3.markdown(f'<div class="glass-card" style="text-align:center;padding:10px;"><div style="font-size:0.65rem;color:#64748b;">Top 5 Avg Score</div><div style="font-family:Outfit;font-weight:800;color:#10b981;font-size:1.3rem;">{_top5_avg:.1f}</div><div style="font-size:0.6rem;color:#475569;">Composite momentum rank</div></div>', unsafe_allow_html=True)
            _ks4.markdown(f'<div style="text-align:center;padding:10px;background:rgba(16,185,129,0.08);border-radius:10px;"><div style="font-size:0.65rem;color:#64748b;">Satellite Holdings</div><div style="font-family:Outfit;font-weight:800;color:#34d399;font-size:1.1rem;">Up to {_total_distinct}</div><div style="font-size:0.6rem;color:#475569;">Pure momentum selection</div></div>', unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            _al_l = "left"
            _sat_html = ['<div class="glass-card" style="padding:0;overflow-x:auto;border-radius:12px;"><table style="width:100%;border-collapse:collapse;font-size:0.82rem;"><thead><tr style="background:rgba(15,23,42,0.6);border-bottom:1px solid rgba(255,255,255,0.08);">']
            for _h in ["#", "Symbol", "Track", "Score", "Mom 🚀", "Sec+Thm 🔄", "Delivery 📦", "Growth 📈", "PEAD ⚡", "FII/DII 🏦", "Ext 🏏", "Composite"]:
                _sat_html.append(f'<th style="padding:8px;text-align:{_al_l if _h in ("Symbol","Track") else "right"};color:#94a3b8;font-weight:600;font-size:0.72rem;">{_h}</th>')
            _sat_html.append("</tr></thead><tbody>")
            for _si, _s in enumerate(_satellite_selection):
                _track_c = "#a78bfa" if _s["track"] == "VAM-B" else ("#34d399" if _s["track"] == "VAM-GQ" else "#fbbf24")
                _comp_c = "#10b981" if _s["composite"] >= 70 else ("#fbbf24" if _s["composite"] >= 40 else "#f87171")
                if _s["track"] == "VAM-B":
                    _score_val = f'<span style="font-weight:700;color:{"#10b981" if _s["vam_score"]>0 else "#f87171"};">{_s["vam_score"]:+.2f}</span>'
                    _mom_v = "—"; _sec_v = "—"; _del_v = "—"; _gro_v = "—"; _pea_v = "—"; _fii_v = "—"
                else:
                    _score_val = f'<span style="font-weight:700;color:#818cf8;">{_s["fs"]:.1f}</span>'
                    _mom_v = f'<span style="color:#fbbf24;">{_s["mom"]}</span>'
                    _sec_v = f'<span style="color:#3b82f6;">{_s["sec"]}</span>'
                    _del_v = f'<span style="color:#60a5fa;">{_s["del"]}</span>'
                    _gro_v = f'<span style="color:#a855f7;">{_s["gro"]}</span>'
                    _pea_v = f'<span style="color:#34d399;">{_s["pea"]}</span>'
                    _fii_v = f'<span style="color:#ec4899;">{_s["fii"]}</span>'
                # Extension badge
                _ext_ico, _ext_col = _ext_badge(_s.get("ext_raw", 0.0))
                _ext_display = f'{_ext_ico} <span style="color:{_ext_col};font-size:0.72rem;">{_s.get("ext_pct", 0.0):+.0f}%</span>'
                _sat_html.append(f'<tr style="border-bottom:1px solid rgba(255,255,255,0.03);">'
                    f'<td style="padding:8px;text-align:center;color:#64748b;">{_si+1}</td>'
                    f'<td style="padding:8px;font-weight:700;color:#f1f5f9;">{_s["symbol"]}</td>'
                    f'<td style="padding:8px;text-align:left;color:{_track_c};font-weight:600;font-size:0.75rem;">{_s["track"]}</td>'
                    f'<td style="padding:8px;text-align:right;">{_score_val}</td>'
                    f'<td style="padding:8px;text-align:right;">{_mom_v}</td>'
                    f'<td style="padding:8px;text-align:right;">{_sec_v}</td>'
                    f'<td style="padding:8px;text-align:right;">{_del_v}</td>'
                    f'<td style="padding:8px;text-align:right;">{_gro_v}</td>'
                    f'<td style="padding:8px;text-align:right;">{_pea_v}</td>'
                    f'<td style="padding:8px;text-align:right;">{_fii_v}</td>'
                    f'<td style="padding:8px;text-align:right;">{_ext_display}</td>'
                    f'<td style="padding:8px;text-align:right;font-weight:700;color:{_comp_c};">{_s["composite"]:.1f}</td>'
                    f'</tr>')
            _sat_html.append("</tbody></table></div>")
            st.markdown("".join(_sat_html), unsafe_allow_html=True)
            
            # ── CSV Download for Satellite Ranking Table ──
            if _satellite_selection:
                _sat_csv_rows = []
                for _s in _satellite_selection:
                    _sat_csv_rows.append({
                        "Rank": _sat_csv_rows[-1]["Rank"] + 1 if _sat_csv_rows else 1,
                        "Symbol": _s["symbol"],
                        "Track": _s["track"],
                        "Score": round(_s.get("vam_score", _s.get("fs", 0)), 2) if _s["track"] in ("VAM-B",) else round(_s.get("fs", 0), 1),
                        "Mom": _s.get("mom", ""),
                        "Sec+Thm": _s.get("sec", ""),
                        "Delivery": _s.get("del", ""),
                        "Growth": _s.get("gro", ""),
                        "PEAD": _s.get("pea", ""),
                        "FII/DII": _s.get("fii", ""),
                        "Extension_%": _s.get("ext_pct", 0.0),
                        "Ext_Mult": _s.get("ext_mult", 1.0),
                        "Composite": round(_s.get("composite", 0), 1),
                    })
                _sat_csv_df = pd.DataFrame(_sat_csv_rows)
                st.markdown("<div style='margin-bottom: 5px;'></div>", unsafe_allow_html=True)
                download_csv_button(_sat_csv_df, f"Satellite_Ranking_{selected_date}.csv",
                                    label="📥 Download Satellite Ranking as CSV", key="sat_csv_dl")
            _common_symbols = set()
            for _s in _satellite_selection:
                if _s["track"] == "VAM-B + VAM-GQ":
                    _common_symbols.add(_s["symbol"])
            if _common_symbols:
                _common_chips = " ".join(f'<span style="display:inline-block;background:rgba(245,158,11,0.12);color:#fbbf24;padding:2px 8px;border-radius:8px;font-size:0.7rem;font-weight:600;margin:2px;">{s}</span>' for s in sorted(_common_symbols)[:10])
                st.markdown(f'<div style="margin-top:8px;font-size:0.8rem;color:#94a3b8;">Overlapping (in both VAM-B & VAM-GQ top 20): {_common_chips}</div>', unsafe_allow_html=True)
        else:
            st.info("No satellite stocks scored. Run the pipeline to populate price data.")
    except Exception as e:
        st.warning(f"Satellite selection engine issue: {e}")
        pass
    # ── VAM Rankings ─────────────────────────────────────────────────────────
    st.markdown("""
    <div style="display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-bottom:10px;">
    <span style="color:#f1f5f9;font-weight:700;font-size:1rem;">🏆 VAM-B Rankings — Full Universe</span>
    <span style="color:#334155;">·</span>
    <a href="#tab_ta" style="color:#34d399;font-size:0.78rem;text-decoration:none;">→ VAM-GQ Screening Lab</a>
    <span style="color:#334155;">·</span>
    <a href="#tab_active" style="color:#6366f1;font-size:0.78rem;text-decoration:none;">→ Master Portfolio</a>
    </div>
    """, unsafe_allow_html=True)
    # _df_vam_t was computed above in the KPI section — reuse it here for display
    if _df_vam_t is None or _df_vam_t.empty:
        _df_vam_t = pd.DataFrame()
    if not _df_vam_t.empty:
        # Score legend
        _search_vam = st.text_input("🔍 Search Symbol...", key="search_vam_rank").upper()
        _df_vam_disp = _df_vam_t.copy()
        if _search_vam:
            _df_vam_disp = _df_vam_disp[_df_vam_disp["Symbol"].str.contains(_search_vam)]
        st.markdown("<span style='color:#94a3b8;font-size:0.85rem;'>Ranked by highest relative return and lowest relative volatility (Z-Ret - Z-Vol) over the last 3 months. <br>⚠️ Scores are relative to the current qualified universe, not absolute.</span>", unsafe_allow_html=True)
        _u_ret_mean = _df_vam_t['Return (63d)'].mean()
        _u_vol_mean = _df_vam_t['Volatility (63d)'].mean()
        st.markdown(f"<span style='color:#64748b;font-size:0.8rem;'>Universe Mean Return: <b>{_u_ret_mean:.1f}%</b> | Universe Mean Volatility: <b>{_u_vol_mean:.1f}%</b> | 🏆 Top 20 in <b style='color:#fbbf24;'>gold</b></span>", unsafe_allow_html=True)
        st.markdown("<span style='color:#64748b;font-size:0.8rem;'>Score Legend: <span style='color:#10b981;font-weight:bold;'>Top 25%</span> | <span style='color:#fbbf24;font-weight:bold;'>Middle 25%</span> | <span style='color:#ef4444;font-weight:bold;'>Bottom 50%</span></span>", unsafe_allow_html=True)
        _p75 = _df_vam_t['VAM Score'].quantile(0.75) if not _df_vam_t.empty else 1.0
        _p50 = _df_vam_t['VAM Score'].quantile(0.50) if not _df_vam_t.empty else 0.0
        _vam_tbl_t = ["<div style='overflow-x:auto; margin-top:10px;'><table class='screening-table'><thead><tr>"]
        for _c_vt in ["Rank", "Symbol", "VAM Score", "Z-Return", "Z-Volatility", "Return (3M)", "Volatility (Ann)"]:
            _vam_tbl_t.append(f"<th>{_c_vt}</th>")
        _vam_tbl_t.append("</tr></thead><tbody>")
        for _i_vt, _row_vt in _df_vam_disp.iterrows():
            _vs_t = _row_vt['VAM Score']
            _sc_t = "#10b981" if _vs_t >= _p75 else ("#fbbf24" if _vs_t >= _p50 else "#ef4444")
            _zrc_t = "#10b981" if _row_vt['Z-Return'] > 0 else "#ef4444"
            _zvc_t = "#ef4444" if _row_vt['Z-Volatility'] > 0 else "#10b981"
            # Gold highlight for top 20 rows
            _row_style = ""
            if _i_vt < 20:
                _row_style = ' style="border-left:3px solid #fbbf24;background:rgba(251,191,36,0.04);"'
            _vam_tbl_t.append(f"<tr{_row_style}>"
                f"<td style='text-align:center;color:{'#fbbf24;font-weight:800' if _i_vt < 20 else '#94a3b8'};'>{_i_vt+1}</td>"
                f"<td style='font-weight:700;color:#f1f5f9;'>{_row_vt['Symbol']}</td>"
                f"<td style='text-align:center;font-weight:700;color:{_sc_t};'>{_vs_t:.2f}</td>"
                f"<td style='text-align:right;color:{_zrc_t};'>{_row_vt['Z-Return']:.2f}</td>"
                f"<td style='text-align:right;color:{_zvc_t};'>{_row_vt['Z-Volatility']:.2f}</td>"
                f"<td style='text-align:right;'>{_row_vt['Return (63d)']:.1f}%</td>"
                f"<td style='text-align:right;'>{_row_vt['Volatility (63d)']:.1f}%</td>"
                f"</tr>")
        _vam_tbl_t.append("</tbody></table></div>")
        st.markdown("".join(_vam_tbl_t), unsafe_allow_html=True)
    else:
        st.info("Not enough valid volatility data to calculate VAM.")
    # ── Full Scored Universe ──────────────────────────────────────────────────
    st.markdown('<div style="border-top:1px solid rgba(255,255,255,0.05);margin:20px 0 16px;"></div>', unsafe_allow_html=True)
    # T4-09: Add mtime of MAAC to the header
    _maac_mtime_str = "Unknown"
    _maac_fpath = os.path.join(OUTPUT_DIR, f"L7_MAAC_Allocations.csv")
    if os.path.exists(_maac_fpath):
        _maac_mtime_str = datetime.datetime.fromtimestamp(os.path.getmtime(_maac_fpath)).strftime('%Y-%m-%d %H:%M:%S')
    st.markdown(f"#### 📊 Full Scored Universe — Signal Indicators <span style='font-size:0.8rem;color:#94a3b8;font-weight:normal;'>(As of Pipeline Run: {_maac_mtime_str})</span>", unsafe_allow_html=True)
    
    _search_vams = st.text_input("Search Symbol...", key="search_vams_tab").upper()
    _df_vams_disp = _df_vams_all.copy()
    if _search_vams:
        _df_vams_disp = _df_vams_disp[_df_vams_disp["Symbol"].str.contains(_search_vams)]
    _df_vams_disp = _df_vams_disp.sort_values("Final_Rank")
    
    # Use _p75/_p50 from VAM Score Rankings section above
    
    # Build VAM score lookup from _df_vam_t
    _vam_score_dict_local = {}
    if '_df_vam_t' in locals() and not _df_vam_t.empty:
        _vam_score_dict_local = dict(zip(_df_vam_t['Symbol'], _df_vam_t['VAM Score']))
        
    _vams_rows = []
    for _, _row_vd in _df_vams_disp.iterrows():
        _sym = _row_vd['Symbol']
        _v_vd = _row_vd.get("CIO_Verdict", "AVOID")
        _vb_vd = ('<span style="background:rgba(59,130,246,0.15);color:#60a5fa;border:1px solid rgba(59,130,246,0.3);padding:2px 6px;border-radius:10px;font-size:0.72rem;">ALLOC</span>' if _v_vd == "BUY" else
                   '<span style="background:rgba(245,158,11,0.15);color:#fbbf24;border:1px solid rgba(245,158,11,0.3);padding:2px 6px;border-radius:10px;font-size:0.72rem;">WATCH</span>' if _v_vd == "HOLD" else
                   '<span style="background:rgba(239,68,68,0.15);color:#f87171;border:1px solid rgba(239,68,68,0.3);padding:2px 6px;border-radius:10px;font-size:0.72rem;">EXCL</span>')
        _natr_vd = _row_vd.get("NATR_14", 0.0); _natr_t_vd = _row_vd.get("NATR_Trend", "")
        _adx_vd = _row_vd.get("ADX_14", 0.0)
        _obv_vd = "📈" if _row_vd.get("OBV_Rising", False) else "📉"
        _rs_vd = _row_vd.get("RS_vs_Nifty50", 0.0)
        _ia_vd = "✅" if _row_vd.get("Independent_Alpha_Pass", False) else "❌"
        _dv_vd = _row_vd.get("Delivery_Pct", 0.0)
        
        # T4-06: Unified Badge System / Missing Cache Handling
        if _sym in _vam_score_dict_local:
            _score_val = _vam_score_dict_local[_sym]
            _sv_color = "#10b981" if _score_val >= _p75 else ("#fbbf24" if _score_val >= _p50 else "#ef4444")
            _score_str = f"<span style='color:{_sv_color};font-weight:700;'>{_score_val:.2f}</span>"
        else:
            _score_str = "<span style='color:#64748b;font-size:0.75rem;'>N/A (No Cache)</span>"
            
        _vams_rows.append(f"<tr><td>{_sym}</td><td style='text-align:center;'>{int(_row_vd.get('Final_Rank', 999))}</td><td style='text-align:center;'>{_vb_vd}</td><td style='text-align:center;'>{_score_str}</td><td>{_natr_vd:.1f} {str(_natr_t_vd)[:3]}</td><td style='text-align:center;'>{_adx_vd:.1f}</td><td style='text-align:center;'>{_obv_vd}</td><td style='text-align:center;'>{_rs_vd:.4f}</td><td style='text-align:center;'>{_ia_vd}</td><td style='text-align:right;'>{_dv_vd:.1f}%</td><td style='text-align:right;'>{_row_vd.get('Allocation_%', 0):.1f}%</td></tr>")
        
    if _vams_rows:
        st.markdown(f"""<div style='overflow-x:auto;'><table class='screening-table'><thead><tr><th>Symbol</th><th>Rank</th><th>Verdict</th><th>VAM Score</th><th>NATR</th><th>ADX</th><th>OBV</th><th>RS</th><th>Alpha</th><th>Deliv%</th><th>Alloc%</th></tr></thead><tbody>{"".join(_vams_rows)}</tbody></table></div>""", unsafe_allow_html=True)
    else:
        st.info("No matches.")
# ──────────────────────────────────────────────────────────────────────────────
# TAB 5: RANKING, SCREENING & ELIMINATION (Stocks Only)
# ──────────────────────────────────────────────────────────────────────────────
with tab_bt:
    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    
    _render_research_links(compact=True)
    render_unified_veto_ui("tab_bt")
    st.caption("📈 **Role:** Strategy Backtest · CAGR/Sharpe/DD · Benchmark Comparison · Leverage Simulation")
    
    nifty50_path = os.path.join(cache_dir, "NIFTY_50_history.csv")
    if not os.path.exists(nifty50_path):
        st.error("Nifty 50 history not found in cache. Cannot run backtest.")
    else:
        df_nifty = read_data_smart(nifty50_path)
        df_nifty["Date"] = pd.to_datetime(df_nifty["Date"])
        df_nifty = df_nifty.sort_values("Date").reset_index(drop=True)
        
        end_date = pd.to_datetime(selected_date)
        
        # T5-11: Validate selected_date against df_nifty upfront
        if end_date not in df_nifty["Date"].values:
            end_date = df_nifty["Date"].max()
            st.info(f"⚠️ Selected date is not a trading day. Auto-adjusted to latest trading day: {end_date.strftime('%Y-%m-%d')}")
            
        run_dates = []
        for d in sorted(os.listdir(output_dir)):
            if os.path.isdir(os.path.join(output_dir, d)) and re.match(r"^\d{4}-\d{2}-\d{2}$", d):
                # T5-01: Filter to ensure folder contains MAAC file
                if os.path.exists(os.path.join(output_dir, d, "L7_MAAC_Allocations.csv")):
                    run_dates.append(d)
                    
        if run_dates:
            inception_date = pd.to_datetime(run_dates[0]) - pd.Timedelta(days=1)
            # Guard: inception must not be earlier than Nifty 50 cache start
            _nifty_cache_min = df_nifty["Date"].min() if not df_nifty.empty else None
            if _nifty_cache_min is not None and inception_date < _nifty_cache_min:
                inception_date = _nifty_cache_min
        else:
            # Fallback to earliest Nifty 50 cache date
            inception_date = df_nifty["Date"].min() if not df_nifty.empty else pd.Timestamp("2026-05-27")
        
        _bt_inception_label = inception_date.strftime("%b %d, %Y")
        
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,rgba(139,92,246,0.12),rgba(59,130,246,0.06));
                    border:1px solid rgba(139,92,246,0.2); border-radius:14px;
                    padding:18px 24px; margin-bottom:22px;">
        <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px;">
            <div style="font-family:'Outfit';font-size:1.1rem;font-weight:700;color:#818cf8;">
              📊 Performance Analyzer — Dynamic Strategy vs Index
            </div>
            <span style="background:rgba(16,185,129,0.12);color:#34d399;border:1px solid rgba(16,185,129,0.3);padding:4px 12px;border-radius:20px;font-size:0.75rem;font-weight:700;display:inline-flex;align-items:center;gap:4px;">
              🟢 BACKTEST READY
            </span>
        </div>
        <div style="font-size:0.85rem;color:#94a3b8;margin-top:6px;">
            Daily cumulative returns since <b style="color:#818cf8;">{_bt_inception_label}</b> — VAM-GQ (Growth+Quality) · VAM-B (Blended Momentum) ·
            <b style="color:#a78bfa;">Trend Alfa 4.0</b> (Core + Satellite blending) · 5 benchmark indices · VAM-GQ quality-gated momentum
        </div>
        </div>
        """, unsafe_allow_html=True)
            
        st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
        st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
        bt_mode = "Historical Strategy (Output Logs)"
        timeframe = st.radio("Select Evaluation Timeframe", 
                             ["1W", "1M", "3M", "6M", "9M", "12M", "Since Inception"], 
                             horizontal=True, index=6)
        
        # ── Mechanics Toggles / Sliders ─────────────────────────────────────────────
        _tcol1, _tcol2, _tcol3, _tcol4, _tcol5 = st.columns(5)
        with _tcol1:
            leverage_val = st.slider("Leverage (0x to 4x)", min_value=0.0, max_value=4.0, value=1.0, step=0.1)
        with _tcol2:
            leverage_interest_pct = st.number_input("Leverage Interest (% p.a.)", min_value=0.0, max_value=100.0, value=12.0, step=0.5)
        with _tcol3:
            tax_rate_pct = st.number_input("Tax Rate (%)", min_value=0.0, max_value=100.0, value=20.8, step=0.5)
        with _tcol4:
            slippage_bps = st.number_input("Charges & Slippage (bps)", min_value=0, max_value=1000, value=300, step=10)
        with _tcol5:
            core_sat_split = st.slider("Core Allocation %", min_value=0, max_value=100, value=65, step=5, help="Satellite allocation will be 100 - Core")
        
        # T5-10: Calendar-anchored months vs trading-day windows
        bday_mapping = {"1W": 5, "1M": 21, "3M": 63, "6M": 126, "9M": 189, "12M": 252}
        if timeframe in bday_mapping:
            start_date = end_date - pd.offsets.BDay(bday_mapping[timeframe])
            st.markdown(f"<span style='font-size:0.8rem;color:#94a3b8;'>{timeframe} = {bday_mapping[timeframe]} trading days</span>", unsafe_allow_html=True)
        else:
            start_date = inception_date
            
        if start_date > end_date:
            start_date = end_date - pd.Timedelta(days=1)
            
        nifty_min_date = df_nifty["Date"].min()
        if start_date < nifty_min_date:
            start_date = nifty_min_date
            
        df_nifty_range = df_nifty[(df_nifty["Date"] >= start_date) & (df_nifty["Date"] <= end_date)]
        trading_dates = df_nifty_range["Date"].tolist()
        
        if len(trading_dates) < 2:
            st.info("Insufficient trading days in the selected timeframe to compute returns.")
        else:
            def get_close(symbol, date):
                if symbol in price_matrix.columns:
                    s = price_matrix.loc[:date, symbol]
                    if not s.empty and pd.notna(s.iloc[-1]):
                        return float(s.iloc[-1])
                return None
            def load_allocations(date_str):
                for filename in ["L7_MAAC_Allocations.csv"]:
                    path = os.path.join(output_dir, date_str, filename)
                    if os.path.exists(path):
                        df = read_data_smart(path)
                        col = None
                        for c in ["Allocation_%", "Allocation_Pct", "Alloc_%"]:
                            if c in df.columns:
                                col = c
                                break
                        if col:
                            df_active = df[df[col] > 0]
                            return dict(zip(df_active["Symbol"], df_active[col] / 100.0))
                return None
            def load_vamb_gq_top20_allocations(date_str):
                for filename in ["L7_MAAC_Allocations.csv"]:
                    path = os.path.join(output_dir, date_str, filename)
                    if os.path.exists(path):
                        df = read_data_smart(path)
                        score_col = next((c for c in ["Factor_Score", "Final_Composite_Score", "composite_score"] if c in df.columns), None)
                        if score_col:
                            df_sorted = df.sort_values(score_col, ascending=False)
                    else:
                            df_sorted = df
                        
                    eligible_col = next((c for c in ["Entry_Eligible", "Eligible"] if c in df.columns), None)
                    if eligible_col:
                            df_sorted = df_sorted[df_sorted[eligible_col] == True]
                            
                    top20_symbols = df_sorted.head(20)["Symbol"].tolist()
                    if top20_symbols:
                            weight = 1.0 / len(top20_symbols)
                            return {sym: weight for sym in top20_symbols}
                return None
            def load_core_allocations(date_str):
                path = os.path.join(output_dir, date_str, "L1_Core_Allocations.csv")
                if os.path.exists(path):
                    df = read_data_smart(path)
                    if "Symbol" in df.columns and "Core_Weight" in df.columns:
                        df_active = df[df["Core_Weight"] > 0]
                        total_wt = df_active["Core_Weight"].sum()
                        if total_wt > 0:
                            return dict(zip(df_active["Symbol"].astype(str), df_active["Core_Weight"] / total_wt))
                return None
            # T5-12: Read valid symbols from run folders to avoid globbing random sectoral CSVs
            _valid_syms_set = set()
            for _d in run_dates:
                # Include VAM-GQ Top 20 symbols
                _alloc = load_vamb_gq_top20_allocations(_d)
                if _alloc: _valid_syms_set.update(_alloc.keys())
                # Include Core allocation symbols (MFs, ETFs, Index funds)
                _core_alloc = load_core_allocations(_d)
                if _core_alloc: _valid_syms_set.update(_core_alloc.keys())
            
            @st.cache_data(ttl=300)
            def build_price_matrix(_cache_dir, valid_syms=None):
                import os, pandas as pd
                dfs = []
                if not os.path.exists(_cache_dir): return pd.DataFrame()
                for fn in os.listdir(_cache_dir):
                    if fn.endswith("_history.csv"):
                        sym = fn.replace("_history.csv", "")
                        # Skip index files — they're loaded separately as benchmarks
                        if sym in ["^NSEI", "^BSESN", "INDIA_VIX"]: continue
                        if valid_syms is not None and sym not in valid_syms: continue
                        fp = os.path.join(_cache_dir, fn)
                        try:
                            d = pd.read_csv(fp, parse_dates=["Date"])
                            d = d.set_index("Date")[["Close"]].rename(columns={"Close": sym})
                            dfs.append(d)
                        except: pass
                if not dfs: return pd.DataFrame()
                pm = pd.concat(dfs, axis=1).sort_index()
                pm = pm.ffill().bfill()
                # Sort columns for deterministic tie-breaking in VAM nlargest
                pm = pm[sorted(pm.columns)]
                return pm
                
            if 'state_data' in locals() and isinstance(state_data, dict) and "holdings" in state_data:
                _valid_syms_set.update(state_data["holdings"].keys())
                
            _use_syms = tuple(_valid_syms_set) if len(_valid_syms_set) > 0 else None
            price_matrix = build_price_matrix(cache_dir, _use_syms)
            portfolio_value = 100.0
            alfa_portfolio_value = 100.0
            master_portfolio_value = 100.0
            core_portfolio_value = 100.0
            
            portfolio_history = [{
                "Date": trading_dates[0].strftime("%Y-%m-%d"), 
                "Value": 100.0, 
                "Return_%": 0.0,
                "Alfa_Value": 100.0,
                "Alfa_Return_%": 0.0,
                "Master_Value": 100.0,
                "Master_Return_%": 0.0,
                "Core_Value": 100.0,
                "Core_Return_%": 0.0
            }]
            
            # ── Historical Simulation Helper ──
            def _simulate_allocations(as_of_date, pm):
                """Simulate satellite holdings using VAM-B momentum when no pipeline data exists.
                Returns dict of sym->weight for top ~25% of stocks by Z-Ret - Z-Vol over 63d."""
                _lookback = as_of_date - pd.Timedelta(days=66)
                _window = pm[pm.index <= as_of_date]
                if len(_window) < 45:
                    return None
                _window = _window[_window.index >= _lookback].tail(63)
                if len(_window) < 30:
                    return None
                _scores = {}
                for _sym in sorted(_window.columns):
                    _s = _window[_sym].dropna()
                    if len(_s) < 30:
                        continue
                    _ret = (_s.iloc[-1] / _s.iloc[0]) - 1.0
                    _vol = _s.pct_change().std() * np.sqrt(252)
                    if _vol > 0:
                        _scores[_sym] = (_ret, _vol)
                if len(_scores) < 10:
                    return None
                _rets = pd.Series({k: v[0] for k, v in _scores.items()})
                _vols = pd.Series({k: v[1] for k, v in _scores.items()})
                _zr = (_rets - _rets.mean()) / max(_rets.std(), 1e-6)
                _zv = (_vols - _vols.mean()) / max(_vols.std(), 1e-6)
                _vam = _zr - _zv
                _top_n = min(30, max(5, len(_vam)))  # up to 30 satellite holdings
                _best = _vam.nlargest(_top_n)
                _total = _best.sum()
                if _total <= 0:
                    return None
                return {sym: wt / _total * 0.35 for sym, wt in _best.items()}
            
            def _simulate_vamb_allocations(as_of_date, pm):
                """Simulate pure VAM-B Top 20 equally-weighted allocations (no blending)."""
                _lookback = as_of_date - pd.Timedelta(days=66)
                _window = pm[pm.index <= as_of_date]
                if len(_window) < 45:
                    return None
                _window = _window[_window.index >= _lookback].tail(63)
                if len(_window) < 30:
                    return None
                _scores = {}
                for _sym in sorted(_window.columns):
                    if _sym in ["Date", "^NSEI", "^BSESN", "INDIA_VIX"]:
                        continue
                    _s = _window[_sym].dropna()
                    if len(_s) < 30:
                        continue
                    _ret = (_s.iloc[-1] / _s.iloc[0]) - 1.0
                    _vol = _s.pct_change().std() * np.sqrt(252)
                    if _vol > 0:
                        _scores[_sym] = (_ret, _vol)
                if len(_scores) < 10:
                    return None
                _rets = pd.Series({k: v[0] for k, v in _scores.items()})
                _vols = pd.Series({k: v[1] for k, v in _scores.items()})
                _zr = (_rets - _rets.mean()) / max(_rets.std(), 1e-6)
                _zv = (_vols - _vols.mean()) / max(_vols.std(), 1e-6)
                _vam = _zr - _zv
                _top_n = min(20, len(_vam))
                _best = _vam.nlargest(_top_n)
                if _best.empty:
                    return None
                weight = 1.0 / len(_best)
                return {sym: weight for sym in _best.index}
            
            active_allocations = None
            alfa_allocations = None
            pending_active_allocations = None
            pending_alfa_allocations = None
            
            if bt_mode.startswith("Live Portfolio"):
                state_holdings = state_data.get("holdings", {}) if state_loaded else {}
                active_allocations = {sym: float(pos.get("Allocation_Pct", 0.0))/100.0 for sym, pos in state_holdings.items() if float(pos.get("Allocation_Pct", 0.0)) > 0}
                if not active_allocations:
                    st.warning("⚠️ No active holdings found in Live Portfolio. Backtest will be flat.")
                valid_past_runs = []
            else:
                # T5-02: Strictly before the first trading date
                valid_past_runs = [d for d in run_dates if pd.to_datetime(d) < trading_dates[0]]
                if valid_past_runs:
                    active_allocations = load_vamb_gq_top20_allocations(valid_past_runs[-1])
                elif run_dates:
                    active_allocations = load_vamb_gq_top20_allocations(run_dates[0])
                    if timeframe != "Since Inception":
                        st.warning(f"⚠️ Strategy allocation for {trading_dates[0].strftime('%Y-%m-%d')} not found. Falling back to earliest available run ({run_dates[0]}).")
                else:
                    st.warning("⚠️ No valid run folders found. Will simulate satellite holdings using VAM-B momentum.")
            
            # If no pipeline allocations exist, simulate from price data for full deep backtest
            if not bt_mode.startswith("Live Portfolio") and active_allocations is None:
                _sim = _simulate_vamb_allocations(trading_dates[0], price_matrix)
                if _sim:
                    active_allocations = _sim
                    st.info(f"📊 Simulating satellite holdings from {trading_dates[0].strftime('%Y-%m-%d')} using VAM-GQ composite scoring (top 20 equally weighted)")
            
            # Initialize VAM-B Top 20 equally weighted allocations
            alfa_allocations = _simulate_vamb_allocations(trading_dates[0], price_matrix)
            
            if not bt_mode.startswith("Live Portfolio"):
                st.markdown("<span style='font-size:0.75rem;color:#64748b;'><i>Disclaimer: Allocations are applied end-of-day on the run date; returns are calculated on prior-day weights.</i></span>", unsafe_allow_html=True)
            
            last_alfa_rebalance_date = None
            # Calculate daily leverage interest cost (applied only if leverage is greater than 1.0)
            daily_interest_cost = 0.0
            if leverage_val > 1.0:
                daily_interest_cost = (leverage_val - 1.0) * (leverage_interest_pct / 100.0) / 252.0
            
            for i in range(1, len(trading_dates)):
                active_slippage = 0.0
                if pending_active_allocations is not None:
                    # Calculate active turnover for slippage
                    prev_w = active_allocations if active_allocations else {}
                    new_w = pending_active_allocations
                    turnover = sum(abs(new_w.get(sym, 0.0) - prev_w.get(sym, 0.0)) for sym in set(prev_w.keys()).union(new_w.keys()))
                    active_slippage = turnover * (slippage_bps / 10000.0)
                    
                    active_allocations = pending_active_allocations
                    pending_active_allocations = None
                    
                alfa_slippage = 0.0
                if pending_alfa_allocations is not None:
                    # Calculate alfa turnover for slippage
                    prev_w = alfa_allocations if alfa_allocations else {}
                    new_w = pending_alfa_allocations
                    turnover = sum(abs(new_w.get(sym, 0.0) - prev_w.get(sym, 0.0)) for sym in set(prev_w.keys()).union(new_w.keys()))
                    alfa_slippage = turnover * (slippage_bps / 10000.0)
                    
                    alfa_allocations = pending_alfa_allocations
                    pending_alfa_allocations = None
                    
                prev_date = trading_dates[i-1]
                curr_date = trading_dates[i]
                prev_date_str = prev_date.strftime("%Y-%m-%d")
                curr_date_str = curr_date.strftime("%Y-%m-%d")
                
                daily_ret = 0.0
                allocated_weight = 0.0
                alfa_daily_ret = 0.0
                
                if active_allocations:
                    for sym, weight in active_allocations.items():
                        p_prev = get_close(sym, prev_date)
                        p_curr = get_close(sym, curr_date)
                        if p_prev is not None and p_curr is not None and p_prev > 0 and not pd.isna(p_prev) and not pd.isna(p_curr):
                            ret = (p_curr / p_prev) - 1.0
                            daily_ret += weight * ret
                            allocated_weight += weight
                            
                    # T5-03: True Portfolio Returns on Missing Prices (Frozen capital)
                    pass
                        
                # --- Alfa Strategy Allocation Logic (calculated on prev_date) ---
                pm_slice = price_matrix.loc[:prev_date]
                
                # T5-04: (Removed Alfa Overlay Alignment logic to prevent blending)
                pass
                    
                # T5-05: Dynamic VAM-B Top 20 Rebalancing (Pure raw momentum, no blending/overlay)
                if len(pm_slice) >= 63 and (last_alfa_rebalance_date is None or (prev_date - last_alfa_rebalance_date).days >= 10):
                    p_current = pm_slice.iloc[-1]
                    p_63d = pm_slice.iloc[-63]
                    
                    # Calculate 63d returns & volatility for the entire available universe
                    ret_3m = (p_current / p_63d) - 1.0
                    vol_3m = pm_slice.iloc[-63:].pct_change().std(skipna=True) * np.sqrt(252)
                    
                    # Universe consists of all valid stock symbols in price matrix
                    universe_symbols = sorted([col for col in price_matrix.columns if col not in ["Date", "^NSEI", "^BSESN", "INDIA_VIX"]])
                    
                    if universe_symbols:
                        elig_ret = ret_3m.reindex(universe_symbols).dropna()
                        elig_vol = vol_3m.reindex(universe_symbols).dropna()
                        
                        valid_syms = elig_ret.index.intersection(elig_vol.index)
                        valid_syms = sorted([s for s in valid_syms if elig_vol[s] > 0])
                        
                        if len(valid_syms) > 0:
                            std_ret = elig_ret[valid_syms].std()
                            std_vol = elig_vol[valid_syms].std()
                            
                            std_ret = std_ret if pd.notna(std_ret) and std_ret > 0 else 1.0
                            std_vol = std_vol if pd.notna(std_vol) and std_vol > 0 else 1.0
                            z_ret = (elig_ret[valid_syms] - elig_ret[valid_syms].mean()) / std_ret
                            z_vol = (elig_vol[valid_syms] - elig_vol[valid_syms].mean()) / std_vol
                            
                            # VAM Score calculation (matching the VAM-B tab scoring formula)
                            vam_scores = z_ret - z_vol
                            
                            # Select Top 20 ranked VAM-B stocks
                            top20 = vam_scores.nlargest(20)
                            
                            if not top20.empty:
                                # Assign equal weights (each stock in Top 20 gets 1.0 / N weight)
                                pending_alfa_allocations = {sym: 1.0 / len(top20) for sym in top20.index}
                                last_alfa_rebalance_date = prev_date
                                
                if alfa_allocations:
                    a_alloc_weight = 0.0
                    for sym, weight in alfa_allocations.items():
                        p_prev = get_close(sym, prev_date)
                        p_curr = get_close(sym, curr_date)
                        if p_prev is not None and p_curr is not None and p_prev > 0 and not pd.isna(p_prev) and not pd.isna(p_curr):
                            ret = (p_curr / p_prev) - 1.0
                            alfa_daily_ret += weight * ret
                            a_alloc_weight += weight
                    if a_alloc_weight > 0:
                        pass # Removed scaling for frozen capital
                # ----------------------------------------------------------------
                
                # Apply leverage to daily returns
                daily_ret = daily_ret * leverage_val
                alfa_daily_ret = alfa_daily_ret * leverage_val
                
                # Apply daily interest cost of leverage
                daily_ret = max(-1.0, daily_ret - daily_interest_cost)
                alfa_daily_ret = max(-1.0, alfa_daily_ret - daily_interest_cost)
                
                # Apply slippage friction
                daily_ret = max(-1.0, daily_ret - active_slippage)
                alfa_daily_ret = max(-1.0, alfa_daily_ret - alfa_slippage)
                
                # Tax applied at Master portfolio level only (line 6406) — not here.
                # Individual component tax removed to prevent double-taxation.
                            
                # ── Trend Alpha 4.0 — Optimal Dynamic Blending Engine ──
                # Computes master_portfolio_value from core (ETFs/MFs) + satellite (VAM-GQ + VAM-B)
                # with regime-aware, volatility-parity, drawdown-protected weighting.
                
                # Full universe satellite with dynamic GQ/B blend
                # Weight: trailing 20-day GQ vs B relative performance (default 50/50)
                if i >= 20 and alfa_allocations and len(portfolio_history) >= 20:
                    _gq_20 = sum(h["Return_%"] for h in portfolio_history[-20:])
                    _b_20 = sum(h["Alfa_Return_%"] for h in portfolio_history[-20:])
                    _gq_wt = 0.3 + 0.4 * (max(0, _gq_20) / max(0.01, max(0, _gq_20) + max(0, _b_20)))
                    _b_wt = 1.0 - _gq_wt
                else:
                    _gq_wt = 0.5
                    _b_wt = 0.5
                _sat_return = _gq_wt * daily_ret + _b_wt * (alfa_daily_ret if alfa_allocations else daily_ret)
                
                # ── Core Return from actual core allocations (MFs, ETFs, Index funds) ──
                _core_return = 0.0
                _core_wt_count = 0.0
                current_core_allocs = None
                
                # Check if there is a core allocation for the current run date
                if curr_date_str in run_dates:
                    current_core_allocs = load_core_allocations(curr_date_str)
                    
                # If no run-date core allocations exist, find the most recent previous run date's core allocations
                if current_core_allocs is None:
                    last_run_dates = [d for d in run_dates if pd.to_datetime(d) < curr_date]
                    if last_run_dates:
                        current_core_allocs = load_core_allocations(last_run_dates[-1])
                        
                if current_core_allocs:
                    for sym, weight in current_core_allocs.items():
                        p_prev = get_close(sym, prev_date)
                        p_curr = get_close(sym, curr_date)
                        if p_prev is not None and p_curr is not None and p_prev > 0 and not pd.isna(p_prev) and not pd.isna(p_curr):
                            _core_return += weight * ((p_curr / p_prev) - 1.0)
                            _core_wt_count += weight
                    if _core_wt_count > 0:
                        _core_return /= _core_wt_count
                else:
                    # Fallback to Nifty 50 return if no core allocations can be loaded
                    try:
                        _nifty_prev_s = df_nifty[df_nifty["Date"] == prev_date]["Close"]
                        _nifty_curr_s = df_nifty[df_nifty["Date"] == curr_date]["Close"]
                        _nifty_prev = float(_nifty_prev_s.iloc[0]) if not _nifty_prev_s.empty else None
                        _nifty_curr = float(_nifty_curr_s.iloc[0]) if not _nifty_curr_s.empty else None
                        if _nifty_prev and _nifty_curr and _nifty_prev > 0:
                            _core_return = (_nifty_curr / _nifty_prev) - 1.0
                    except:
                        pass
                
                if bt_mode.startswith("Live Portfolio"):
                    # Live mode: use actual state holdings = daily_ret
                    master_daily_ret = daily_ret
                else:
                    # ── Core / Satellite Allocation Weighting ──
                    # Direct weighting based on the user's Core Allocation % slider input.
                    _core_wt = core_sat_split / 100.0
                    _sat_wt = 1.0 - _core_wt
                    master_daily_ret = _core_wt * _core_return + _sat_wt * _sat_return
                    
                    # Apply tax friction to Master Portfolio if positive
                    if tax_rate_pct > 0.0 and master_daily_ret > 0:
                        master_daily_ret = master_daily_ret * (1.0 - tax_rate_pct / 100.0)
                            
                portfolio_value = portfolio_value * (1.0 + daily_ret)
                alfa_portfolio_value = alfa_portfolio_value * (1.0 + alfa_daily_ret)
                master_portfolio_value = master_portfolio_value * (1.0 + master_daily_ret)
                core_portfolio_value = core_portfolio_value * (1.0 + _core_return)
                
                portfolio_history.append({
                    "Date": curr_date_str,
                    "Value": portfolio_value,
                    "Return_%": daily_ret * 100.0,
                    "Alfa_Value": alfa_portfolio_value,
                    "Alfa_Return_%": alfa_daily_ret * 100.0,
                    "Master_Value": master_portfolio_value,
                    "Master_Return_%": master_daily_ret * 100.0,
                    "Core_Value": core_portfolio_value,
                    "Core_Return_%": _core_return * 100.0
                })
                
                # T5-06: new_allocs applied for the NEXT day
                if not bt_mode.startswith("Live Portfolio"):
                    if curr_date_str in run_dates:
                        new_allocs = load_vamb_gq_top20_allocations(curr_date_str)
                        if new_allocs is not None:
                            pending_active_allocations = new_allocs
                            _last_sim_date_g = None  # reset sim counter on real allocation
                    # Hold last known pipeline allocation between run dates.
                    # No simulation — avoids look-ahead bias from price_matrix data.
            df_port = pd.DataFrame(portfolio_history)
            
            indices = {
                "Nifty 50": "NIFTY_50_history.csv",
                "Nifty Next 50": "NIFTY_NEXT_50_history.csv",
                "Nifty Midcap 150": "NIFTY_MIDCAP_150_history.csv",
                "Nifty Smallcap 250": "NIFTY_SMALLCAP_250_history.csv",
                "Nifty Microcap 250": "NIFTY_MICROCAP_250_history.csv"
            }
            
            index_dfs = {}
            missing_benchmarks = []
            for name, filename in indices.items():
                path = os.path.join(cache_dir, filename)
                try:
                    df = read_data_smart(path)
                    df["Date"] = pd.to_datetime(df["Date"])
                    df = df[df["Date"].isin(trading_dates)].sort_values("Date").reset_index(drop=True)
                    df["Date_Str"] = df["Date"].dt.strftime("%Y-%m-%d")
                    index_dfs[name] = df
                except Exception as e:
                    missing_benchmarks.append(name)
                    continue
                
            base_date_str = trading_dates[0].strftime("%Y-%m-%d")
            end_date_str = trading_dates[-1].strftime("%Y-%m-%d")
            
            # --- Dynamically inject outperforming Strategy/Sector/Thematic Indices ---
            # Use Nifty 50 as the benchmark threshold for outperformance
            nifty50_benchmark_return = -999.0
            for name, df in index_dfs.items():
                if name == "Nifty 50":
                    base_rows = df[df["Date_Str"] <= base_date_str]
                    curr_rows = df[df["Date_Str"] <= end_date_str]
                    if not base_rows.empty and not curr_rows.empty:
                        base_close = float(base_rows["Close"].iloc[-1])
                        curr_close = float(curr_rows["Close"].iloc[-1])
                        if base_close > 0:
                            nifty50_benchmark_return = (curr_close / base_close - 1.0) * 100.0
                            break
            if nifty50_benchmark_return == -999.0:
                nifty50_benchmark_return = 0.0  # fallback if Nifty 50 data missing
                        
            outperforming_indices = []
            
            # Evaluate Sectors and Themes (cached in session state to avoid re-fetch)
            _sec_cache_key = f"_sector_cache_{selected_date}"
            if _sec_cache_key not in st.session_state:
                st.session_state[_sec_cache_key] = _load_sectoral_thematic_data()
            sec_df, _ = st.session_state[_sec_cache_key]
            if not sec_df.empty:
                for name, ticker, color in sectoral_indices + thematic_indices:
                    if ticker in sec_df.columns:
                        series = sec_df[ticker].dropna()
                        series.index = pd.to_datetime(series.index)
                        series_reindexed = series.reindex(trading_dates, method='ffill')
                        if pd.notna(series_reindexed.iloc[-1]):
                            base_close = float(series_reindexed.iloc[0]) if pd.notna(series_reindexed.iloc[0]) else float(series_reindexed.dropna().iloc[0])
                            curr_close = float(series_reindexed.iloc[-1])
                            ret = (curr_close / base_close - 1.0) * 100.0
                            if ret > nifty50_benchmark_return:
                                df_temp = pd.DataFrame({"Date": trading_dates, "Close": series_reindexed.ffill().bfill().values})
                                df_temp["Date_Str"] = df_temp["Date"].dt.strftime("%Y-%m-%d")
                                index_dfs[name] = df_temp
                                outperforming_indices.append((name, color, ret))
                                
            # Evaluate Strategy Indices
            for name, color in strategy_tickers:
                cid = STRATEGY_SCREENER_IDS.get(name)
                cache_file = os.path.join(cache_dir, f"strategy_{cid}_history.json")
                if os.path.exists(cache_file):
                    try:
                        with open(cache_file, "r") as f:
                            prices = json.load(f)
                        if prices and len(prices) > 0:
                            pts = {pd.Timestamp(d): float(v) for d,v in prices}
                            series = pd.Series(pts).sort_index()
                            series_reindexed = series.reindex(trading_dates, method='ffill')
                            if pd.notna(series_reindexed.iloc[-1]):
                                base_close = float(series_reindexed.iloc[0]) if pd.notna(series_reindexed.iloc[0]) else float(series_reindexed.dropna().iloc[0])
                                curr_close = float(series_reindexed.iloc[-1])
                                ret = (curr_close / base_close - 1.0) * 100.0
                                if ret > nifty50_benchmark_return:
                                    df_temp = pd.DataFrame({"Date": trading_dates, "Close": series_reindexed.ffill().bfill().values})
                                    df_temp["Date_Str"] = df_temp["Date"].dt.strftime("%Y-%m-%d")
                                    index_dfs[name] = df_temp
                                    outperforming_indices.append((name, color, ret))
                    except Exception:
                        pass
                        
            # Evaluate Mutual Funds
            mf_df = _load_mf_data()
            if not mf_df.empty:
                for name, ticker, color in mutual_funds:
                    if ticker in mf_df.columns:
                        series = mf_df[ticker].dropna()
                        series.index = pd.to_datetime(series.index)
                        series_reindexed = series.reindex(trading_dates, method='ffill')
                        if pd.notna(series_reindexed.iloc[-1]):
                            base_close = float(series_reindexed.iloc[0]) if pd.notna(series_reindexed.iloc[0]) else float(series_reindexed.dropna().iloc[0])
                            curr_close = float(series_reindexed.iloc[-1])
                            ret = (curr_close / base_close - 1.0) * 100.0
                            if ret > nifty50_benchmark_return:
                                df_temp = pd.DataFrame({"Date": trading_dates, "Close": series_reindexed.ffill().bfill().values})
                                df_temp["Date_Str"] = df_temp["Date"].dt.strftime("%Y-%m-%d")
                                index_dfs[name] = df_temp
                                outperforming_indices.append((name, color, ret))
                                
            # T5-08: Cap outperforming_indices at top 4
            outperforming_indices.sort(key=lambda x: x[2], reverse=True)
            outperforming_indices = outperforming_indices[:4]
            outperforming_indices = [(x[0], x[1]) for x in outperforming_indices]
            
            # Build Strategy column safely: use forward-fill for missing dates
            df_port_indexed = df_port.set_index("Date")
            comparison = []
            for date in trading_dates:
                date_str = date.strftime("%Y-%m-%d")
                row = {"Date": date_str}
                if date_str in df_port_indexed.index:
                    p_val = float(df_port_indexed.loc[date_str, "Value"])
                    a_val = float(df_port_indexed.loc[date_str, "Alfa_Value"])
                    m_val = float(df_port_indexed.loc[date_str, "Master_Value"])
                    c_val = float(df_port_indexed.loc[date_str, "Core_Value"])
                else:
                    available = df_port_indexed[df_port_indexed.index <= date_str]
                    p_val = float(available["Value"].iloc[-1]) if not available.empty else 100.0
                    a_val = float(available["Alfa_Value"].iloc[-1]) if not available.empty else 100.0
                    m_val = float(available["Master_Value"].iloc[-1]) if not available.empty else 100.0
                    c_val = float(available["Core_Value"].iloc[-1]) if not available.empty else 100.0
                row["VAM-GQ"] = (p_val / 100.0 - 1.0) * 100.0
                row["VAM-B"] = (a_val / 100.0 - 1.0) * 100.0
                row["Trend Alpha 4.0"] = (m_val / 100.0 - 1.0) * 100.0
                row["Trend Alpha (Core Only - MF/ETF/IF)"] = (c_val / 100.0 - 1.0) * 100.0
                for name, df in index_dfs.items():
                    base_rows = df[df["Date_Str"] == base_date_str]
                    curr_rows = df[df["Date_Str"] == date_str]
                    if not base_rows.empty and not curr_rows.empty:
                        base_close = float(base_rows["Close"].iloc[0])
                        curr_close = float(curr_rows["Close"].iloc[0])
                        row[name] = (curr_close / base_close - 1.0) * 100.0
                    else:
                        prev_rows = df[df["Date_Str"] <= date_str]
                        base_close_rows = df[df["Date_Str"] <= base_date_str]
                        
                        # T5-10: Graceful fallback for benchmark base close
                        if not base_close_rows.empty:
                            bc = float(base_close_rows["Close"].iloc[-1])
                        else:
                            bc = float(df["Close"].iloc[0]) if not df.empty else 0.0
                            
                        if not prev_rows.empty and bc > 0:
                            cc = float(prev_rows["Close"].iloc[-1])
                            row[name] = (cc / bc - 1.0) * 100.0
                        else:
                            row[name] = 0.0
                comparison.append(row)
            df_comp = pd.DataFrame(comparison)
            
            # ── 0. Calculate Performance Statistics ──
            def get_drawdown(series):
                """NaN-safe max drawdown calculation."""
                arr = np.array(series, dtype=float)
                arr = arr[~np.isnan(arr)]  # strip NaNs
                if len(arr) < 2:
                    return 0.0
                cum_max = np.maximum.accumulate(arr)
                dd = (arr - cum_max) / cum_max * 100.0
                return float(np.min(dd))
            stats = []
            # Deduplicate: base list + outperforming extras (skip if already in base)
            _base_cols = [
                "VAM-GQ",
                "VAM-B",
                "Trend Alpha 4.0",
                "Trend Alpha (Core Only - MF/ETF/IF)",
                "Nifty 50",
                "Nifty Next 50",
                "Nifty Midcap 150",
                "Nifty Smallcap 250",
                "Nifty Microcap 250",
            ]
            _extra_cols = [name for name, _ in outperforming_indices if name not in _base_cols]
            cols_to_evaluate = _base_cols + _extra_cols
            # Only keep cols that actually exist in df_comp
            cols_to_evaluate = [c for c in cols_to_evaluate if c in df_comp.columns]
            for col in cols_to_evaluate:
                raw = pd.to_numeric(df_comp[col], errors="coerce")
                series_pct = raw.ffill().fillna(0.0)
                series_lvl = series_pct.values + 100.0  # price-level series from base=100
                total_ret = float(series_pct.iloc[-1]) if not series_pct.empty else 0.0
                max_dd    = get_drawdown(series_lvl)
                daily_rets = pd.Series(series_lvl).pct_change().dropna()
                daily_rets = daily_rets.replace([np.inf, -np.inf], np.nan).dropna()
                if len(daily_rets) >= 5 and daily_rets.std() > 0:
                    std_dev = daily_rets.std() * np.sqrt(252) * 100.0
                    sharpe  = (daily_rets.mean() / daily_rets.std()) * np.sqrt(252)
                    # CAGR
                    n_trading_days = len(daily_rets)
                    years = n_trading_days / 252.0
                    tot_ret_dec = total_ret / 100.0
                    cagr = ((1 + tot_ret_dec) ** (1.0 / years) - 1.0) * 100.0 if years > 0 else 0.0
                    # Sortino (downside deviation)
                    downside = daily_rets[daily_rets < 0]
                    sortino = (daily_rets.mean() / downside.std()) * np.sqrt(252) if len(downside) > 0 and downside.std() > 0 else 0.0
                else:
                    std_dev = 0.0
                    sharpe  = 0.0
                    cagr = 0.0
                    sortino = 0.0
                # Benchmark-relative metrics (vs Nifty 50)
                beta_val = 0.0
                alpha_val = 0.0
                info_ratio = 0.0
                if col == "Nifty 50":
                    # Benchmark vs itself: Beta=1.0, Alpha=0.0, Info Ratio=0.0
                    beta_val = 1.0
                    alpha_val = 0.0
                    info_ratio = 0.0
                elif "Nifty 50" in cols_to_evaluate:
                    nifty_raw = pd.to_numeric(df_comp["Nifty 50"], errors="coerce").ffill().fillna(0.0)
                    nifty_lvl = nifty_raw.values + 100.0
                    nifty_ret = pd.Series(nifty_lvl).pct_change().dropna().replace([np.inf, -np.inf], np.nan).dropna()
                    # Align lengths via strict DatetimeIndex
                    if len(daily_rets) > 5 and len(nifty_ret) > 5:
                        df_align = pd.concat([daily_rets, nifty_ret], axis=1).dropna()
                        if len(df_align) > 5:
                            s_ret = df_align.iloc[:, 0]
                            b_ret = df_align.iloc[:, 1]
                            cov_mat = np.cov(s_ret, b_ret)
                            if cov_mat[1][1] > 0:
                                beta_val = cov_mat[0][1] / cov_mat[1][1]
                                rf_daily = 0.07 / 252.0
                                alpha_daily = (s_ret.mean() - rf_daily) - beta_val * (b_ret.mean() - rf_daily)
                                alpha_val = alpha_daily * 252.0 * 100.0
                                te = (s_ret - b_ret).std()
                                info_ratio = ((s_ret.mean() - b_ret.mean()) / te) * np.sqrt(252) if te > 0 else 0.0
                # Suppress CAGR for short backtest periods (< 63 trading days = ~3 months)
                MIN_CAGR_TRADING_DAYS = 63
                if n_trading_days < MIN_CAGR_TRADING_DAYS:
                    cagr_str = "—  <span style='font-size:0.65rem;color:#64748b;'>(<3mo)</span>"
                else:
                    cagr_str = f"{cagr:.2f}%"
                stats.append({
                    "Asset / Index": col,
                    "Total Return":   f"{total_ret:+.2f}%",
                    "CAGR":           cagr_str,
                    "Max Drawdown":   f"{max_dd:.2f}%",
                    "Annlzd Vol":     f"{std_dev:.2f}%",
                    "Sharpe":         f"{sharpe:.2f}",
                    "Sortino":        f"{sortino:.2f}",
                    "Alpha":          f"{alpha_val:+.2f}%",
                    "Beta":           f"{beta_val:.2f}",
                    "Info Ratio":     f"{info_ratio:.2f}"
                })
            
            # ── Sync TA 4.0 metrics to session state for Master Portfolio KPI cards ──
            for _s in stats:
                if _s["Asset / Index"] == "Trend Alpha 4.0":
                    st.session_state.ta4_ret = _s["Total Return"]
                    st.session_state.ta4_dd = _s["Max Drawdown"]
                    st.session_state.ta4_sharpe = _s["Sharpe"]
                    st.session_state.ta4_vol = _s["Annlzd Vol"]
                    break
            
            # ── 1. Create Glassmorphic KPI Cards for the Strategy ──
            
            
            # ── 2. Create the Cleaned Plotly Chart ──
            fig_bt = go.Figure()
            # Plot strategy in clean emerald green
            # VAM-GQ line
            if "VAM-GQ" in df_comp.columns:
                fig_bt.add_trace(go.Scatter(
                    x=df_comp["Date"], y=df_comp["VAM-GQ"],
                    name="VAM-GQ (Growth & Quality)",
                    line=dict(color="#10b981", width=3, shape='spline'),
                    hovertemplate="VAM-GQ: <b>%{y:.2f}%</b><extra></extra>"
                ))
            
            # VAM-B line
            if "VAM-B" in df_comp.columns:
                fig_bt.add_trace(go.Scatter(
                    x=df_comp["Date"], y=df_comp["VAM-B"],
                    name="VAM-B (Blended)",
                    line=dict(color="#8b5cf6", width=3, shape='spline'),
                    hovertemplate="VAM-B: <b>%{y:.2f}%</b><extra></extra>"
                ))
            
            # Trend Alpha 4.0 line
            if "Trend Alpha 4.0" in df_comp.columns:
                fig_bt.add_trace(go.Scatter(
                    x=df_comp["Date"], y=df_comp["Trend Alpha 4.0"], 
                    name="Trend Alpha 4.0 (Core + Satellite)", 
                    line=dict(color="#f59e0b", width=4.5, shape='spline', dash='dash'),
                    hovertemplate="Trend Alpha 4.0: <b>%{y:.2f}%</b><extra></extra>"
                ))
            # Trend Alpha (Core Only) line
            if "Trend Alpha (Core Only - MF/ETF/IF)" in df_comp.columns:
                fig_bt.add_trace(go.Scatter(
                    x=df_comp["Date"], y=df_comp["Trend Alpha (Core Only - MF/ETF/IF)"], 
                    name="Trend Alpha (Core Only - MF/ETF/IF)", 
                    line=dict(color="#38bdf8", width=3, shape='spline', dash='dashdot'),
                    hovertemplate="Core Only: <b>%{y:.2f}%</b><extra></extra>"
                ))
            # Plot indices with matching theme colors and shapes (Vibrant, distinct premium colors)
            fig_bt.add_trace(go.Scatter(
                x=df_comp["Date"], y=df_comp["Nifty 50"], 
                name="Nifty 50", 
                line=dict(color="#94a3b8", width=1.5, dash="solid", shape='spline'),
                hovertemplate="Nifty 50: %{y:.2f}%<extra></extra>"
            ))
            fig_bt.add_trace(go.Scatter(
                x=df_comp["Date"], y=df_comp["Nifty Next 50"], 
                name="Nifty Next 50", 
                line=dict(color="#06b6d4", width=1.5, dash="dash", shape='spline'),
                hovertemplate="Nifty Next 50: %{y:.2f}%<extra></extra>"
            ))
            fig_bt.add_trace(go.Scatter(
                x=df_comp["Date"], y=df_comp["Nifty Midcap 150"], 
                name="Nifty Midcap 150", 
                line=dict(color="#3b82f6", width=1.5, dash="dot", shape='spline'),
                hovertemplate="Nifty Midcap 150: %{y:.2f}%<extra></extra>"
            ))
            fig_bt.add_trace(go.Scatter(
                x=df_comp["Date"], y=df_comp["Nifty Smallcap 250"], 
                name="Nifty Smallcap 250", 
                line=dict(color="#ec4899", width=1.5, dash="dashdot", shape='spline'),
                hovertemplate="Nifty Smallcap 250: %{y:.2f}%<extra></extra>"
            ))
            fig_bt.add_trace(go.Scatter(
                x=df_comp["Date"], y=df_comp["Nifty Microcap 250"], 
                name="Nifty Microcap 250", 
                line=dict(color="#ef4444", width=1.5, dash="longdash", shape='spline'),
                hovertemplate="Nifty Microcap 250: %{y:.2f}%<extra></extra>"
            ))
            
            # Plot dynamic outperforming indices (T5-08)
            for name, color in outperforming_indices:
                fig_bt.add_trace(go.Scatter(
                    x=df_comp["Date"], y=df_comp[name], 
                    name=f"{name} ⭐", 
                    line=dict(color=color, width=1.5, dash="dashdot", shape='spline'),
                    hovertemplate=name + ": <b>%{y:.2f}%</b><extra></extra>",
                    opacity=0.75
                ))
                
            # T5-09: Visualize Rebalance Events
            _tds = pd.to_datetime(trading_dates)
            for d_str in list(valid_past_runs) + list(run_dates):
                rd = pd.to_datetime(d_str)
                if rd >= _tds.min() and rd <= _tds.max():
                    fig_bt.add_vline(x=rd, line_width=1, line_dash="dash", line_color="rgba(255,255,255,0.1)")
            
            fig_bt.update_layout(
                hovermode="x unified",
                template="plotly_dark",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=50, r=20, t=20, b=120),
                height=600,
                legend=dict(
                    orientation="h",
                    y=-0.12,
                    x=0.5,
                    xanchor="center",
                    yanchor="top",
                    bgcolor="rgba(0,0,0,0)",
                    font=dict(size=11, color="#94a3b8")
                ),
                xaxis=dict(
                    showgrid=True,
                    gridcolor="rgba(255, 255, 255, 0.05)",
                    zeroline=False,
                    tickfont=dict(color="#64748b")
                ),
                yaxis=dict(
                    showgrid=True,
                    gridcolor="rgba(255, 255, 255, 0.05)",
                    zeroline=True,
                    zerolinecolor="rgba(255, 255, 255, 0.1)",
                    ticksuffix="%",
                    tickfont=dict(color="#64748b")
                )
            )
            st.plotly_chart(fig_bt, use_container_width=True)
            
            # --- PHASE 3: UNDERWATER CHART & MONTHLY HEATMAP ---
            try:
                _ta4_lvl = df_comp.get("Trend Alpha 4.0", df_comp["Strategy"]) + 100.0
                _ta4_cummax = _ta4_lvl.cummax()
                _ta4_dd = (_ta4_lvl - _ta4_cummax) / _ta4_cummax * 100.0
                
                fig_dd = go.Figure()
                fig_dd.add_trace(go.Scatter(
                    x=df_comp["Date"], y=_ta4_dd,
                    mode="lines", name="Drawdown",
                    fill="tozeroy", fillcolor="rgba(239,68,68,0.2)",
                    line=dict(color="#ef4444", width=1.5),
                    hovertemplate="Drawdown: %{y:.2f}%<extra></extra>"
                ))
                fig_dd.update_layout(
                    height=200, template="plotly_dark",
                    margin=dict(l=40, r=20, t=10, b=10),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(showgrid=True, gridcolor="rgba(255, 255, 255, 0.05)", zeroline=False),
                    yaxis=dict(showgrid=True, gridcolor="rgba(255, 255, 255, 0.05)", zeroline=True, ticksuffix="%")
                )
                st.plotly_chart(fig_dd, use_container_width=True, config={"displayModeBar": False})
                
                _ta4_lvl.index = pd.to_datetime(df_comp["Date"])
                _monthly_prices = _ta4_lvl.resample("ME").last()
                _monthly_rets = _monthly_prices.pct_change() * 100.0
                if len(_monthly_prices) > 0 and pd.isna(_monthly_rets.iloc[0]):
                    _monthly_rets.iloc[0] = (_monthly_prices.iloc[0] / 100.0 - 1.0) * 100.0
                
                _hm_df = pd.DataFrame({
                    "Year": _monthly_rets.index.year,
                    "Month": _monthly_rets.index.strftime("%b"),
                    "Return": _monthly_rets.values
                })
                _hm_pivot = _hm_df.pivot_table(index="Year", columns="Month", values="Return", aggfunc="sum")
                _month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
                _hm_pivot = _hm_pivot.reindex(columns=[m for m in _month_order if m in _hm_pivot.columns])
                
                _hm_df["Ret_1"] = _hm_df["Return"] / 100.0 + 1.0
                _ytd = _hm_df.groupby("Year")["Ret_1"].prod() - 1.0
                _hm_pivot["YTD"] = _ytd * 100.0
                
                _text_vals = _hm_pivot.applymap(lambda x: f"{x:.1f}%" if pd.notna(x) else "")
                
                fig_hm2 = px.imshow(
                    _hm_pivot, 
                    text_auto=False, 
                    aspect="auto", 
                    color_continuous_scale="RdYlGn", 
                    color_continuous_midpoint=0
                )
                fig_hm2.update_traces(text=_text_vals, texttemplate="%{text}")
                fig_hm2.update_layout(
                    height=max(180, 100 + 40 * len(_hm_pivot)),
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    title=dict(text="<b>Month-by-Month Returns (%)</b>", font=dict(color="#94a3b8", size=13)),
                    xaxis=dict(title="", tickfont=dict(color="#cbd5e1")),
                    yaxis=dict(title="", tickfont=dict(color="#cbd5e1"), dtick=1)
                )
                st.plotly_chart(fig_hm2, use_container_width=True, config={"displayModeBar": False})
            except Exception as e:
                pass
            # --- END PHASE 3 ---
            
            # T5-07: Display missing benchmarks
            if missing_benchmarks:
                st.warning(f"⚠️ {len(missing_benchmarks)} benchmark(s) unavailable for comparison: {', '.join(missing_benchmarks)}")
            # ── 3. Performance Statistics Summary Card Table ──
            st.markdown("<div style='margin-top: 25px;'></div>", unsafe_allow_html=True)
            st.subheader("📊 Performance Statistics Comparison")
            
            stats_html = []
            stats_html.append("""
            <div class="glass-card" style="padding: 0; overflow: hidden; border-radius: 12px; border: 1px solid rgba(255, 255, 255, 0.08);">
            <table style="width: 100%; border-collapse: collapse; font-size: 0.92rem; font-family: 'Inter', sans-serif;">
            <thead>
            <tr style="background: rgba(30, 41, 59, 0.7); border-bottom: 2px solid rgba(255, 255, 255, 0.08);">
            <th style="padding: 14px 16px; text-align: left; color: #94a3b8; font-family: 'Outfit'; font-weight: 600; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.5px;">Asset / Index</th>
            <th style="padding: 14px 16px; text-align: right; color: #94a3b8; font-family: 'Outfit'; font-weight: 600; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.5px;">Total Return</th>
            <th style="padding: 14px 16px; text-align: right; color: #94a3b8; font-family: 'Outfit'; font-weight: 600; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.5px;">CAGR</th>
            <th style="padding: 14px 16px; text-align: right; color: #94a3b8; font-family: 'Outfit'; font-weight: 600; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.5px;">Max DD</th>
            <th style="padding: 14px 16px; text-align: right; color: #94a3b8; font-family: 'Outfit'; font-weight: 600; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.5px;">Vol</th>
            <th style="padding: 14px 16px; text-align: right; color: #94a3b8; font-family: 'Outfit'; font-weight: 600; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.5px;">Sharpe</th>
            <th style="padding: 14px 16px; text-align: right; color: #94a3b8; font-family: 'Outfit'; font-weight: 600; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.5px;">Sortino</th>
            <th style="padding: 14px 16px; text-align: right; color: #94a3b8; font-family: 'Outfit'; font-weight: 600; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.5px;">Alpha</th>
            <th style="padding: 14px 16px; text-align: right; color: #94a3b8; font-family: 'Outfit'; font-weight: 600; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.5px;">Beta</th>
            <th style="padding: 14px 16px; text-align: right; color: #94a3b8; font-family: 'Outfit'; font-weight: 600; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.5px;">Info Ratio</th>
            </tr>
            </thead>
            <tbody>""")
            
            for s_row in stats:
                s_name = s_row["Asset / Index"]
                s_clean = s_name.replace(" ⭐", "")
                is_vam_gq = s_clean == "VAM-GQ"
                is_vam_b = s_clean == "VAM-B"
                is_ta4 = s_clean == "Trend Alpha 4.0"
                is_core = s_clean == "Trend Alpha (Core Only - MF/ETF/IF)"
                is_strat = is_vam_gq or is_vam_b or is_ta4 or is_core
                
                if is_vam_gq:
                    name_display = "VAM-GQ (Volatility Adjusted Momentum — Growth and Quality)"
                    row_style = "background: rgba(16, 185, 129, 0.08); border-bottom: 1px solid rgba(16, 185, 129, 0.15);"
                    label_style = "font-weight: 700; color: #34d399;"
                    ret_style = "font-weight: 700; color: #34d399; font-family: monospace; font-size: 1rem;"
                    cagr_style = "font-weight: 600; color: #6ee7b7; font-family: monospace;"
                    dd_style = "font-weight: 600; color: #f87171; font-family: monospace;"
                    vol_style = "color: #38bdf8; font-family: monospace;"
                    sharpe_style = "font-weight: 700; color: #fbbf24; font-family: monospace;"
                    sortino_style = "color: #fcd34d; font-family: monospace;"
                    alpha_style = "font-weight: 600; color: #a78bfa; font-family: monospace;"
                    beta_style = "color: #94a3b8; font-family: monospace;"
                    ir_style = "font-weight: 600; color: #c084fc; font-family: monospace;"
                elif is_vam_b:
                    name_display = "VAM-B (Volatility Adjusted Momentum — Blended)"
                    row_style = "background: rgba(139, 92, 246, 0.08); border-bottom: 1px solid rgba(139, 92, 246, 0.15);"
                    label_style = "font-weight: 700; color: #a78bfa;"
                    ret_style = "color: #c4b5fd; font-family: monospace;"
                    cagr_style = "color: #c4b5fd; font-family: monospace;"
                    dd_style = "color: #fca5a5; font-family: monospace;"
                    vol_style = "color: #38bdf8; font-family: monospace;"
                    sharpe_style = "color: #fbbf24; font-family: monospace;"
                    sortino_style = "color: #fcd34d; font-family: monospace;"
                    alpha_style = "color: #a78bfa; font-family: monospace;"
                    beta_style = "color: #94a3b8; font-family: monospace;"
                    ir_style = "color: #c084fc; font-family: monospace;"
                elif is_ta4:
                    name_display = "Trend Alpha 4.0 (Core + Satellite) ⭐"
                    row_style = "background: rgba(245, 158, 11, 0.08); border-bottom: 1px solid rgba(245, 158, 11, 0.15);"
                    label_style = "font-weight: 700; color: #fbbf24;"
                    ret_style = "color: #fcd34d; font-family: monospace;"
                    cagr_style = "color: #fcd34d; font-family: monospace;"
                    dd_style = "color: #fca5a5; font-family: monospace;"
                    vol_style = "color: #38bdf8; font-family: monospace;"
                    sharpe_style = "color: #fde047; font-family: monospace;"
                    sortino_style = "color: #fde047; font-family: monospace;"
                    alpha_style = "color: #a78bfa; font-family: monospace;"
                    beta_style = "color: #94a3b8; font-family: monospace;"
                    ir_style = "color: #c084fc; font-family: monospace;"
                elif is_core:
                    name_display = "Trend Alpha (Core Only - MF/ETF/IF)"
                    row_style = "background: rgba(96, 165, 250, 0.08); border-bottom: 1px solid rgba(96, 165, 250, 0.15);"
                    label_style = "font-weight: 600; color: #60a5fa;"
                    ret_style = "color: #93c5fd; font-family: monospace;"
                    cagr_style = "color: #93c5fd; font-family: monospace;"
                    dd_style = "color: #fca5a5; font-family: monospace;"
                    vol_style = "color: #38bdf8; font-family: monospace;"
                    sharpe_style = "color: #fde047; font-family: monospace;"
                    sortino_style = "color: #fde047; font-family: monospace;"
                    alpha_style = "color: #a78bfa; font-family: monospace;"
                    beta_style = "color: #94a3b8; font-family: monospace;"
                    ir_style = "color: #c084fc; font-family: monospace;"
                else:
                    row_style = "border-bottom: 1px solid rgba(255, 255, 255, 0.04);"
                    label_style = "color: #e2e8f0; font-weight: 500;"
                    ret_style = "color: #cbd5e1; font-family: monospace;"
                    cagr_style = "color: #94a3b8; font-family: monospace;"
                    dd_style = "color: #fca5a5; font-family: monospace;"
                    vol_style = "color: #94a3b8; font-family: monospace;"
                    sharpe_style = "color: #fde047; font-family: monospace;"
                    sortino_style = "color: #fde047; font-family: monospace;"
                    alpha_style = "color: #c4b5fd; font-family: monospace;"
                    beta_style = "color: #94a3b8; font-family: monospace;"
                    ir_style = "color: #c084fc; font-family: monospace;"
                    name_display = s_name.replace(" ⭐", "")
                
                stats_html.append(f"""<tr style="{row_style}">
                <td style="padding: 12px 16px; text-align: left; {label_style}">{name_display}</td>
                <td style="padding: 12px 16px; text-align: right; {ret_style}">{s_row['Total Return']}</td>
                <td style="padding: 12px 16px; text-align: right; {cagr_style}">{s_row['CAGR']}</td>
                <td style="padding: 12px 16px; text-align: right; {dd_style}">{s_row['Max Drawdown']}</td>
                <td style="padding: 12px 16px; text-align: right; {vol_style}">{s_row['Annlzd Vol']}</td>
                <td style="padding: 12px 16px; text-align: right; {sharpe_style}">{s_row['Sharpe']}</td>
                <td style="padding: 12px 16px; text-align: right; {sortino_style}">{s_row['Sortino']}</td>
                <td style="padding: 12px 16px; text-align: right; {alpha_style}">{s_row['Alpha']}</td>
                <td style="padding: 12px 16px; text-align: right; {beta_style}">{s_row['Beta']}</td>
                <td style="padding: 12px 16px; text-align: right; {ir_style}">{s_row['Info Ratio']}</td>
                </tr>""")
                
            stats_html.append("</tbody></table></div>")
            st.markdown("".join(stats_html).replace("\n", ""), unsafe_allow_html=True)
            
            # ── 4. Collapsible Detailed Daily Log ──
            st.markdown("<div style='margin-top: 25px;'></div>", unsafe_allow_html=True)
            with st.expander("📅 Show Detailed Daily Cumulative Performance Log"):
                df_comp_formatted = df_comp.copy()
                for col in cols_to_evaluate:
                    df_comp_formatted[col] = df_comp_formatted[col].map("{:,.2f}%".format)
                st.dataframe(df_comp_formatted.sort_values(by="Date", ascending=False), use_container_width=True)
st.markdown("""
<div style="text-align: center; color: #64748b; font-size: 0.85rem; padding: 20px; border-top: 1px solid rgba(255, 255, 255, 0.05);">
    Trend Alpha 4.0 • Institutional Portfolio OS Terminal • 100% Systematic Execution
</div>
""", unsafe_allow_html=True)
# ═══════════════════════════════════════════════════════════════════════════════
# TAB 7 – STOCK RS MONITOR (merged from Trend Alfa _ Hermes)
# ═══════════════════════════════════════════════════════════════════════════════
with tab_orch_rs_5d:
    _render_research_links(compact=True)
    # ── CSV Download — top row ──
    _dl_top_c1, _dl_top_c2, _dl_top_c3 = st.columns([1, 1, 4])
    with _dl_top_c1:
        if 'OUTPUT_DIR' in globals() and os.path.exists(OUTPUT_DIR):
            _zip_data = create_zip_of_folder(OUTPUT_DIR)
            st.download_button(
                label="📦 Download All CSVs (ZIP)",
                data=_zip_data,
                file_name=f"TrendAlfa_Data_{selected_date}.zip",
                mime="application/zip",
                use_container_width=True,
                type="primary"
            )
    render_unified_veto_ui("tab_orch_rs_5d")
    st.caption("🚀 **Role:** Unified Analysis · RS Monitor · Combined VAM-GQ + VAM-B View · CSV Export")
    
    import subprocess
    
    # Dynamically resolve Hermes location if possible
    ORCH_SCRIPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Hermes", "scripts", "master_orchestrator_v3.py")
    if not os.path.exists(ORCH_SCRIPT):
        st.warning(f"⚠️ Orchestrator script not found at: {ORCH_SCRIPT}")
        
    REPORT_PATH = os.path.join(
        os.path.dirname(os.path.dirname(ORCH_SCRIPT)), "logs", "master_orch_v3_latest.json"
    )
    
    # Load Orchestrator Report
    report = {}
    if os.path.exists(REPORT_PATH):
        try:
            with open(REPORT_PATH) as f:
                report = json.load(f)
        except:
            pass
            
    # Load Dynamic RS Data with cache-busting refresh key
    _rs_key = st.session_state.get("_rs_refresh_key", 0)
    rs_df = _h_get_rs_df(_rs_key)
    _h_portfolio = _h_get_portfolio()
    # Fallback: use MAAC RS data if pipeline RS unavailable
    if rs_df.empty and not df_maac.empty and "RS_vs_Nifty50" in df_maac.columns:
        _fallback_rows = []
        _maac_stocks = df_maac[~df_maac["Symbol"].astype(str).str.match(r"^\d+$")]
        for _, _mr in _maac_stocks.iterrows():
            _sym_fb = str(_mr["Symbol"])
            if _sym_fb:
                # Use Entry_Price from MAAC if Close column isn't available
                _fb_price = float(_mr.get("Entry_Price", 0) or _mr.get("Close", 0) or 0)
                # Stock_123d% uses Return_63d if available, otherwise compute from Final_Composite_Score proxy
                _fb_ret = float(_mr.get("Return_63d", 0) or 0)
                _fallback_rows.append({
                    "Stock": _sym_fb,
                    "RS": float(_mr.get("RS_vs_Nifty50", 0) or 0) / 100.0,
                    "Price": _fb_price,
                    "Stock_123d%": _fb_ret,
                    "Nifty_123d%": 0.0,
                })
        if _fallback_rows:
            rs_df = pd.DataFrame(_fallback_rows)
            st.warning("📡 **Pipeline RS data unavailable** — using MAAC-cached RS as secondary fallback.")
    elif rs_df.empty:
        st.caption("⚠️ RS data unavailable. Run pipeline or check connectivity.")
    
    st.subheader("📋 Unified Master Analysis Table")
    
    # Manual refresh button to bust cache
    _ref_col1, _ref_col2 = st.columns([3, 1])
    with _ref_col2:
        if st.button("🔄 Refresh RS Data", key="master_refresh_rs", use_container_width=True):
            st.cache_data.clear()
            st.session_state["_rs_refresh_key"] = st.session_state.get("_rs_refresh_key", 0) + 1
            st.rerun()
    
    if not rs_df.empty:
        with _ref_col1:
            st.markdown(f"*Combined universe: **{len(_h_portfolio)}** stocks (VAM-GQ top 20 + VAM-B top 20 + Core)*")
        report_stocks = {s.get("ticker", "").replace(".NS", ""): s for s in report.get("stocks", [])}
        
        # Build name map from L1_Core_Universe.csv and _mf_name_map_global so MF scheme codes render with human-readable names
        _universe_name_map_master = _mf_name_map_global.copy()
        _core_universe_path_m = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", selected_date, "L1_Core_Universe.csv")
        if not os.path.exists(_core_universe_path_m):
            _mf_path, _ = _resolve_pipeline_file("L1_Core_Universe.csv")
            if _mf_path:
                _core_universe_path_m = _mf_path
        if os.path.exists(_core_universe_path_m):
            try:
                _df_u_m = pd.read_csv(_core_universe_path_m, usecols=["Symbol", "Name"]).dropna(subset=["Name"])
                _universe_name_map_master = dict(zip(_df_u_m["Symbol"].astype(str), _df_u_m["Name"].astype(str)))
            except Exception:
                pass
        
        # BUG FIX 8: Pre-load MAAC for ADX/OBV/CIO signals to enrich table without re-fetching yfinance
        _maac_path_m = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", selected_date, "L7_MAAC_Allocations.csv")
        _maac_sig_map = {}  # sym -> {adx, adx_bull, obv_rise, cio_verdict, cio_score, factor_score}
        if os.path.exists(_maac_path_m):
            try:
                _df_maac_m = pd.read_csv(_maac_path_m)
                _df_maac_m_stocks = _df_maac_m[~_df_maac_m["Symbol"].astype(str).str.match(r'^\d+$')]
                for _, _mr in _df_maac_m_stocks.iterrows():
                    _maac_sig_map[str(_mr["Symbol"])] = {
                        "adx": float(_mr.get("ADX_14", 0) or 0),
                        "adx_bull": bool(_mr.get("ADX_Bullish", False)),
                        "obv": bool(_mr.get("OBV_Rising", False)),
                        "cio": str(_mr.get("CIO_Verdict", "—")),
                        "cio_score": float(_mr.get("CIO_Score", 0) or 0),
                        "factor": float(_mr.get("Factor_Score", 0) or 0),
                        "sector": str(_mr.get("Sector", "")),
                    }
            except Exception:
                pass
        # Build RS ranking from MAAC universe (Nifty 500 proxy)
        _rs_rank_map = {}
        if not df_maac.empty and "RS_vs_Nifty50" in df_maac.columns and "Symbol" in df_maac.columns:
            try:
                _rs_univ = df_maac[~df_maac["Symbol"].astype(str).str.match(r"^\d+$")][["Symbol", "RS_vs_Nifty50"]].copy()
                _rs_univ["RS_vs_Nifty50"] = pd.to_numeric(_rs_univ["RS_vs_Nifty50"], errors="coerce").fillna(0)
                _rs_univ = _rs_univ.sort_values("RS_vs_Nifty50", ascending=False).reset_index(drop=True)
                _rs_univ["Rank"] = range(1, len(_rs_univ) + 1)
                _rs_rank_map = dict(zip(_rs_univ["Symbol"].astype(str), _rs_univ["Rank"]))
            except:
                pass
        # Build unified table data
        unified_data = []
        import math
        _has_orch_report = bool(report_stocks)
        for idx, row in rs_df.iterrows():
            sym = row["Stock"]
            orch_data = report_stocks.get(sym, {})
            maac_sig = _maac_sig_map.get(sym, {})
            rs = row["RS"]
            price = row["Price"]
            s123 = row["Stock_123d%"]
            n123 = row["Nifty_123d%"]
            # Prefer MAAC sector over orch sector (MAAC is from the pipeline, more reliable)
            sector = maac_sig.get("sector") or orch_data.get("sector", "—") or "—"
            mom = orch_data.get("mom", None)
            comp = orch_data.get("composite", None)
            action = str(orch_data.get("action", "—")).upper()
            # BUG FIX 4 + 5: Normalise RS to 0-100 scale so it has equal weight with composite.
            # BUG FIX 5: When no orch report, sort purely by live RS descending.
            rs_safe = rs if (rs is not None and not math.isnan(rs)) else -1
            rs_norm = rs_safe * 100.0  # normalise 0-1 → 0-100
            if _has_orch_report:
                comp_safe = comp if (comp is not None and not math.isnan(comp)) else -999
                sort_val = comp_safe * 0.6 + rs_norm * 0.4  # weighted blend (60% orch, 40% RS)
            else:
                sort_val = rs_norm  # BUG FIX 5: pure RS sort when no report
            unified_data.append({
                "sym": sym, "rs": rs, "price": price, "s123": s123, "n123": n123,
                "sector": sector, "mom": mom, "comp": comp, "action": action,
                "sort_val": sort_val,
                "adx": maac_sig.get("adx", 0), "adx_bull": maac_sig.get("adx_bull", False),
                "obv": maac_sig.get("obv", False), "cio": maac_sig.get("cio", "—"),
                "factor": maac_sig.get("factor", 0),
                "has_rs": rs is not None and not (isinstance(rs, float) and math.isnan(rs)),
                "rs_line": "-" if (rs is not None and not (isinstance(rs, float) and math.isnan(rs)) and rs < 0.10) else ("0" if (rs is not None and not (isinstance(rs, float) and math.isnan(rs)) and rs < 0.20) else ("+" if (rs is not None and not (isinstance(rs, float) and math.isnan(rs))) else "—")),
                "rs_rank": _rs_rank_map.get(sym, None),
            })
        unified_data.sort(key=lambda x: x["sort_val"], reverse=True)
        
        # S3: Search/filter bar above the table
        _search_term = st.text_input("🔍 Master Analyzer Search (Enter exact Symbol for Tear Sheet, or Sector to filter)", placeholder="e.g. APARINDS or Technology", key="master_search", label_visibility="collapsed")
        
        if _search_term.strip():
            _st = _search_term.strip().upper()
            _exact_stock = next((d for d in unified_data if d["sym"].upper() == _st), None)
            
            # --- PHASE 3: MASTER ANALYZER TEAR SHEET ---
            if _exact_stock:
                st.markdown(f"## 📄 Tear Sheet: {_exact_stock['sym']} <span style='font-size:1.0rem;color:#94a3b8;font-weight:normal;'>| {_exact_stock['sector']} | ₹{_exact_stock['price']:,.1f}</span>", unsafe_allow_html=True)
                with st.spinner(f"Generating Tear Sheet for {_exact_stock['sym']}..."):
                    try:
                        d_ts = yf.download(_exact_stock['sym'] + ".NS", period="1y", progress=False, auto_adjust=True)
                        if isinstance(d_ts.columns, pd.MultiIndex):
                            c_ts = d_ts.xs("Close", axis=1, level=0).squeeze()
                        else:
                            c_ts = d_ts["Close"].squeeze()
                        c_ts = pd.Series(c_ts).dropna()
                        _g_c1, _g_c2, _g_c3 = st.columns(3)
                        _rs_val = _exact_stock.get("rs")
                        fig_g1 = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=_rs_val * 100,
                        title={'text': "Relative Strength", 'font': {'size': 14, 'color': '#cbd5e1'}},
                        number={'font': {'color': '#f1f5f9', 'size':26}},
                        gauge={'axis': {'range': [None, 50], 'tickwidth': 1, 'tickcolor': "rgba(255,255,255,0.1)"},
                                   'bar': {'color': "#34d399" if _rs_val >= 0.2 else "#fbbf24" if _rs_val >= 0.1 else "#f87171"},
                                   'bgcolor': "rgba(255,255,255,0.05)",
                                   'steps': [{'range': [0, 10], 'color': "rgba(239,68,68,0.15)"},
                                             {'range': [10, 20], 'color': "rgba(245,158,11,0.15)"},
                                             {'range': [20, 50], 'color': "rgba(16,185,129,0.15)"}]}
                        ))
                        fig_g1.update_layout(height=180, margin=dict(l=10, r=10, t=30, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                        _g_c1.plotly_chart(fig_g1, use_container_width=True, config={"displayModeBar": False})
                        
                        _adx_val = _exact_stock.get("adx", 0)
                        fig_g2 = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=_adx_val,
                            title={'text': "Trend Strength (ADX)", 'font': {'size': 14, 'color': '#cbd5e1'}},
                            number={'font': {'color': '#f1f5f9', 'size':26}},
                            gauge={'axis': {'range': [None, 60], 'tickwidth': 1, 'tickcolor': "rgba(255,255,255,0.1)"},
                                   'bar': {'color': "#60a5fa" if _adx_val >= 25 else "#94a3b8"},
                                   'bgcolor': "rgba(255,255,255,0.05)"}
                        ))
                        fig_g2.update_layout(height=180, margin=dict(l=10, r=10, t=30, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                        _g_c2.plotly_chart(fig_g2, use_container_width=True, config={"displayModeBar": False})
                        
                        _comp_val = _exact_stock.get("comp")
                        _comp_val = float(_comp_val) if _comp_val is not None and not math.isnan(_comp_val) else 0.0
                        fig_g3 = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=_comp_val,
                            title={'text': "Composite Score", 'font': {'size': 14, 'color': '#cbd5e1'}},
                            number={'font': {'color': '#f1f5f9', 'size':26}},
                            gauge={'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "rgba(255,255,255,0.1)"},
                                   'bar': {'color': "#a855f7" if _comp_val >= 70 else "#f472b6" if _comp_val >= 40 else "#ef4444"},
                                   'bgcolor': "rgba(255,255,255,0.05)"}
                        ))
                        fig_g3.update_layout(height=180, margin=dict(l=10, r=10, t=30, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                        _g_c3.plotly_chart(fig_g3, use_container_width=True, config={"displayModeBar": False})
                        
                        fig_ts = go.Figure()
                        fig_ts.add_trace(go.Scatter(x=c_ts.index, y=c_ts.values, mode="lines", name="Price", line=dict(color="#00d4aa", width=2)))
                        fig_ts.add_trace(go.Scatter(x=c_ts.index, y=c_ts.ewm(span=50).mean().values, mode="lines", name="EMA50", line=dict(color="#fbbf24", width=1.5, dash="dash")))
                        if len(c_ts) >= 200:
                            fig_ts.add_trace(go.Scatter(x=c_ts.index, y=c_ts.rolling(200).mean().values, mode="lines", name="SMA200", line=dict(color="#ef4444", width=1.5, dash="dot")))
                        fig_ts.update_layout(height=380, template="plotly_dark", title=dict(text=f"Technical Profile (1 Year) — {_exact_stock['sym']}", font=dict(color="#94a3b8")), 
                                          margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.4)")
                        st.plotly_chart(fig_ts, use_container_width=True, config={"displayModeBar": False})
                        st.markdown("---")
                    except Exception as e:
                        pass
            # --- END PHASE 3 ---
            
            unified_data = [d for d in unified_data if _st in d["sym"].upper() or _st in d["sector"].upper()]
            st.caption(f"🔍 Showing {len(unified_data)} results for **{_search_term}**")
        
        # Toggle: show only stocks with live RS data vs all portfolio holdings
        _show_all_holdings = st.toggle("Show all portfolio holdings (incl. MFs without RS data)", value=False, key="master_show_all")
        if not _show_all_holdings:
            _before = len(unified_data)
            unified_data = [d for d in unified_data if d["has_rs"]]
            if len(unified_data) < _before:
                st.caption(f"📊 Showing **{len(unified_data)}** stocks with live RS data ({_before - len(unified_data)} portfolio-only holdings hidden)")
        
        # Limit to top 50 RS stocks for focused analysis
        unified_data = unified_data[:50]
        # Data coverage summary
        _total_stocks = sum(1 for d in unified_data if d.get("has_rs", False))
        _total_portfolio = len(unified_data)
        _with_adx = sum(1 for d in unified_data if d.get("adx", 0) > 0 and d.get("has_rs", False))
        _with_cio = sum(1 for d in unified_data if d.get("cio", "—") != "—" and d.get("has_rs", False))
        _with_orch = sum(1 for d in unified_data if d.get("comp", None) is not None and not (isinstance(d.get("comp"), float) and math.isnan(d.get("comp"))))
        st.markdown(f"""<div style="display:flex;gap:8px;margin:6px 0 14px 0;flex-wrap:wrap;font-size:0.75rem;">
            <span style="color:#94a3b8;background:rgba(15,23,42,0.5);padding:4px 10px;border-radius:8px;">
                📈 <b style="color:#e2e8f0;">{_total_stocks}</b> stocks w/ RS
            </span>
            <span style="color:#94a3b8;background:rgba(15,23,42,0.5);padding:4px 10px;border-radius:8px;">
                📁 <b style="color:#e2e8f0;">{_total_portfolio - _total_stocks}</b> portfolio holdings
            </span>
            <span style="color:#94a3b8;background:rgba(15,23,42,0.5);padding:4px 10px;border-radius:8px;">
                📊 <b style="color:#e2e8f0;">{_with_adx}</b> w/ ADX
            </span>
            <span style="color:#94a3b8;background:rgba(15,23,42,0.5);padding:4px 10px;border-radius:8px;">
                🔮 <b style="color:#e2e8f0;">{_with_cio}</b> w/ CIO
            </span>
            <span style="color:#94a3b8;background:rgba(15,23,42,0.5);padding:4px 10px;border-radius:8px;">
                🧠 <b style="color:#e2e8f0;">{_with_orch}</b> w/ Orch Score
            </span>
        </div>""", unsafe_allow_html=True)
        
        html = ["<div class='glass-card' style='padding:0; overflow-x:auto; border-radius:12px; margin-top:10px;'>"]
        html.append("<table style='width:100%; border-collapse:collapse; font-size:0.85rem; font-family:\"Inter\", sans-serif;'>")
        html.append("<thead><tr style='background:rgba(15,23,42,0.6); border-bottom:1px solid rgba(255,255,255,0.08);'>")
        headers = ["#", "STOCK", "SECTOR", "PRICE", "RS RANK", "RS LINE", "123d% RET", "ADX", "OBV", "CIO", "ALIGN", "ORCH COMP", "ORCH ACTION", "LIVE SIGNAL"]
        for h in headers:
            align = "left" if h in ["STOCK", "SECTOR"] else "center" if h in ["#", "ORCH ACTION", "LIVE SIGNAL", "OBV", "CIO", "ALIGN"] else "right"
            html.append(f"<th style='padding:10px; text-align:{align}; color:#94a3b8; font-weight:600; white-space:nowrap;'>{h}</th>")
        html.append("</tr></thead><tbody>")
        
        for idx, d in enumerate(unified_data):
            # Portfolio-only holding (MF, numeric code) — no RS data available
            if not d.get("has_rs", False):
                _name_display_mf = _universe_name_map_master.get(d['sym'], d['sym'])
                html.append(f"<tr style='border-bottom:1px solid rgba(255,255,255,0.02); opacity:0.75;'>")
                html.append(f"<td style='padding:10px; text-align:center; font-weight:600; color:#64748b;'>{idx+1}</td>")
                html.append(f"<td style='padding:10px; font-weight:600; color:#94a3b8; font-family:\"Outfit\";'>{_name_display_mf}</td>")
                html.append(f"<td style='padding:10px; color:#64748b; font-size:0.78rem; font-style:italic;'>Portfolio Only</td>")
                html.append(f"<td style='padding:10px; text-align:right; font-family:monospace; color:#475569;'>—</td>")
                html.append(f"<td style='padding:10px; text-align:center; font-family:monospace; color:#475569;'>—</td>")
                html.append(f"<td style='padding:10px; text-align:right; font-family:monospace; color:#475569;'>—</td>")
                html.append(f"<td style='padding:10px; text-align:right; color:#475569;'>—</td>")
                html.append(f"<td style='padding:10px; text-align:center; color:#475569;'>—</td>")
                html.append(f"<td style='padding:10px; text-align:center; color:#475569;'>—</td>")
                html.append(f"<td style='padding:10px; text-align:center; color:#475569;'>—</td>")  # ALIGN
                html.append(f"<td style='padding:10px; text-align:right; font-family:monospace; color:#475569;'>—</td>")
                html.append(f"<td style='padding:10px; text-align:center;'><span style='color:#64748b;font-size:0.72rem;'>—</span></td>")
                html.append(f"<td style='padding:10px; text-align:center;'><span style='color:#6366f1;font-weight:600;font-size:0.78rem;'>📁 HOLDING</span></td>")
                html.append("</tr>")
                continue
            # RS Rank display (within Nifty 500 proxy universe)
            _rs_rank = d.get("rs_rank", None)
            if _rs_rank is not None:
                _rank_clr = "#10b981" if _rs_rank <= 20 else ("#fbbf24" if _rs_rank <= 50 else "#f87171")
                _rs_rank_str = f"<span style='color:{_rank_clr};font-weight:700;font-family:monospace;'>{_rs_rank}</span>"
            else:
                _rs_rank_str = '<span style="color:#475569;">—</span>'
            
            # Live Signal Formatting
            rs = d["rs"]
            if pd.isna(rs):
                sig, rs_html = '<span style="color:#64748b;font-weight:600;">⚪ N/A</span>', "N/A"
            elif rs <= 0.10:
                sig, rs_html = '<span style="background:rgba(239,68,68,0.15);color:#f87171;padding:2px 8px;border-radius:6px;font-weight:700;">🔴 EXIT</span>', f"<span style='color:#f87171;font-weight:700;'>{_rs_rank_str}</span>"
            elif rs < 0.20:
                sig, rs_html = '<span style="background:rgba(245,158,11,0.15);color:#fbbf24;padding:2px 8px;border-radius:6px;font-weight:700;">🟡 WATCH</span>', f"<span style='color:#fbbf24;font-weight:700;'>{_rs_rank_str}</span>"
            else:
                sig, rs_html = '<span style="background:rgba(16,185,129,0.15);color:#34d399;padding:2px 8px;border-radius:6px;font-weight:700;">🟢 HOLD</span>', f"<span style='color:#10b981;font-weight:700;'>{_rs_rank_str}</span>"
                
            # Orch Action Formatting
            action = d["action"]
            if "ADD" in action: bg, clr = "rgba(16,185,129,0.15)", "#34d399"
            elif "HOLD" in action: bg, clr = "rgba(59,130,246,0.15)", "#93c5fd"
            elif "REDUCE" in action: bg, clr = "rgba(245,158,11,0.15)", "#fbbf24"
            elif "EXIT" in action: bg, clr = "rgba(239,68,68,0.15)", "#fca5a5"
            else: bg, clr = "rgba(255,255,255,0.05)", "#94a3b8"
            action_badge = f'<span style="background:{bg}; color:{clr}; padding:4px 8px; border-radius:6px; font-weight:700;">{action}</span>'
            
            s123 = f"{d['s123']:.1f}%" if pd.notna(d['s123']) and d['s123'] != 0 else "—"
            prc = f"\u20b9{d['price']:,.1f}" if pd.notna(d['price']) and d['price'] != 0 else "—"
            comp_str = f"{d['comp']:.1f}" if pd.notna(d['comp']) else "—"
            # ADX badge
            _adx_v = d.get("adx", 0)
            _adx_bull = d.get("adx_bull", False)
            _adx_clr = "#10b981" if _adx_v > 25 and _adx_bull else ("#fbbf24" if _adx_v > 20 else "#64748b")
            _adx_str = f"<span style='color:{_adx_clr};font-weight:700;font-family:monospace;'>{_adx_v:.0f}</span>" if _adx_v else "<span style='color:#475569;'>—</span>"
            # OBV badge
            _obv_v = d.get("obv", False)
            _obv_str = "<span style='color:#34d399;font-weight:700;'>↑</span>" if _obv_v else "<span style='color:#f87171;'>↓</span>"
            # CIO verdict badge
            _cio = d.get("cio", "—")
            _cio_bg = {"BUY": "rgba(16,185,129,0.15)", "HOLD": "rgba(59,130,246,0.12)", "REDUCE": "rgba(245,158,11,0.12)", "SELL": "rgba(239,68,68,0.12)", "EXIT": "rgba(239,68,68,0.12)"}
            _cio_clr = {"BUY": "#34d399", "HOLD": "#93c5fd", "REDUCE": "#fbbf24", "SELL": "#f87171", "EXIT": "#f87171"}
            _cbg = _cio_bg.get(_cio.upper(), "rgba(255,255,255,0.04)")
            _cclr = _cio_clr.get(_cio.upper(), "#94a3b8")
            _cio_badge = f"<span style='background:{_cbg};color:{_cclr};padding:2px 7px;border-radius:6px;font-size:0.72rem;font-weight:700;'>{_cio}</span>"
            
            # ── RS-CIO Alignment Check ──
            _rs_bull = not pd.isna(rs) and rs >= 0.20
            _rs_bear = not pd.isna(rs) and rs < 0.10
            _cio_upper = _cio.upper()
            _cio_bull = _cio_upper in ("BUY", "HOLD")
            _cio_bear = _cio_upper in ("REDUCE", "SELL", "EXIT")
            if pd.isna(rs) or _cio_upper not in ("BUY", "HOLD", "REDUCE", "SELL", "EXIT"):
                _align_badge = '<span style="color:#475569;">—</span>'
            elif (_rs_bull and _cio_bull) or (_rs_bear and _cio_bear):
                _align_badge = '<span style="background:rgba(16,185,129,0.15);color:#34d399;padding:2px 8px;border-radius:6px;font-weight:700;font-size:0.72rem;">✅ ALIGNED</span>'
            elif (_rs_bull and _cio_bear) or (_rs_bear and _cio_bull):
                _align_badge = '<span style="background:rgba(239,68,68,0.15);color:#f87171;padding:2px 8px;border-radius:6px;font-weight:700;font-size:0.72rem;">⚠️ CONFLICT</span>'
            else:
                _align_badge = '<span style="background:rgba(245,158,11,0.12);color:#fbbf24;padding:2px 8px;border-radius:6px;font-weight:700;font-size:0.72rem;">🔶 PARTIAL</span>'
            html.append(f"<tr style='border-bottom:1px solid rgba(255,255,255,0.03);'>")
            html.append(f"<td style='padding:10px; text-align:center; font-weight:600; color:#64748b;'>{idx+1}</td>")
            html.append(f"<td style='padding:10px; font-weight:700; color:#f1f5f9; font-family:\"Outfit\";'>{_universe_name_map_master.get(d['sym'], d['sym'])}</td>")
            html.append(f"<td style='padding:10px; color:#94a3b8; font-size:0.8rem;'>{d['sector']}</td>")
            html.append(f"<td style='padding:10px; text-align:right; font-family:monospace; color:#e2e8f0;'>{prc}</td>")
            html.append(f"<td style='padding:10px; text-align:right; font-family:monospace;'>{rs_html}</td>")
            # RS Line — actual raw RS number (color-coded by threshold)
            _rs_raw_val = d.get("rs", None)
            if _rs_raw_val is not None and pd.notna(_rs_raw_val) and not (isinstance(_rs_raw_val, float) and math.isnan(_rs_raw_val)):
                _rs_line_clr = "#10b981" if _rs_raw_val >= 0.20 else ("#fbbf24" if _rs_raw_val >= 0.10 else "#f87171")
                _rs_line_html = f'<span style="color:{_rs_line_clr};font-weight:700;font-family:monospace;">{_rs_raw_val:.2f}</span>'
            else:
                _rs_line_html = '<span style="color:#475569;">—</span>'
            html.append(f"<td style='padding:10px; text-align:right;'>{_rs_line_html}</td>")
            html.append(f"<td style='padding:10px; text-align:right; font-family:monospace; color:#e2e8f0;'>{s123}</td>")
            html.append(f"<td style='padding:10px; text-align:right;'>{_adx_str}</td>")
            html.append(f"<td style='padding:10px; text-align:center;'>{_obv_str}</td>")
            html.append(f"<td style='padding:10px; text-align:center;'>{_cio_badge}</td>")
            html.append(f"<td style='padding:10px; text-align:center;'>{_align_badge}</td>")
            html.append(f"<td style='padding:10px; text-align:right; font-family:monospace; color:#c084fc; font-weight:700;'>{comp_str}</td>")
            html.append(f"<td style='padding:10px; text-align:center;'>{action_badge}</td>")
            html.append(f"<td style='padding:10px; text-align:center;'>{sig}</td>")
            html.append("</tr>")
            
        html.append("</tbody></table></div>")
        st.markdown("".join(html), unsafe_allow_html=True)
        
        # ── CSV Download for Master Analyzer Table ──
        if unified_data:
            _ma_csv_rows = []
            for _d in unified_data:
                _rs_line_v = _d.get("rs", None)
                _rs_line_v = round(_rs_line_v, 2) if _rs_line_v is not None and not (isinstance(_rs_line_v, float) and math.isnan(_rs_line_v)) else ""
                _s123_v = _d.get("s123", None)
                _s123_v = f"{_s123_v:.1f}%" if pd.notna(_s123_v) and _s123_v != 0 else ""
                _prc_v = _d.get("price", None)
                _prc_v = round(_prc_v, 1) if pd.notna(_prc_v) and _prc_v != 0 else ""
                _rs_rank_v = _d.get("rs_rank", "")
                _adx_v = _d.get("adx", "")
                _cio_v = _d.get("cio", "")
                _obv_v = "↑" if _d.get("obv", False) else "↓"
                _comp_v = round(_d.get("comp", 0), 1) if pd.notna(_d.get("comp")) else ""
                _action_v = _d.get("action", "")
                _rs_live = round(_rs_line_v * 100, 2) if isinstance(_rs_line_v, (int, float)) else ""
                _ma_csv_rows.append({
                    "RS_RANK": _rs_rank_v,
                    "STOCK": _d.get("sym", ""),
                    "SECTOR": _d.get("sector", ""),
                    "PRICE": _prc_v,
                    "LIVE_RS": _rs_live,
                    "RS_LINE": _rs_line_v,
                    "123d%_RET": _s123_v,
                    "ADX": _adx_v,
                    "OBV": _obv_v,
                    "CIO": _cio_v,
                    "ORCH_COMP": _comp_v,
                    "ORCH_ACTION": _action_v,
                })
            _ma_csv_df = pd.DataFrame(_ma_csv_rows)
            download_csv_button(_ma_csv_df, f"Master_Analyzer_Table_{selected_date}.csv",
                                label="📥 Download Table as CSV", key="ma_csv_dl")
        
        # ── Charts ──
        st.markdown("### 📈 Visual Analysis")
        # ─────────────────────────────────────────────────────────────────────
        # BUG FIX 9: RS Distribution — sorted by RS value, top-25 by default
        # ─────────────────────────────────────────────────────────────────────
        st.markdown("#### 📊 RS Distribution")
        rs_vals_all = rs_df[["Stock", "RS"]].dropna(subset=["RS"]).sort_values("RS", ascending=False)
        _show_all_rs = st.toggle("Show all stocks", value=False, key="rs_dist_show_all")
        _rs_display = rs_vals_all.head(50) if _show_all_rs else rs_vals_all.head(25)
        if len(_rs_display) > 0:
            _rs_colors = ["#ef4444" if v <= 0.10 else "#fbbf24" if v < 0.20 else "#10b981" for v in _rs_display["RS"]]
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=_rs_display["Stock"],
                y=_rs_display["RS"],
                marker_color=_rs_colors,
                text=[f"{v:.3f}" for v in _rs_display["RS"]],
                textposition="outside",
                textfont=dict(size=9, color="#cbd5e1"),
            ))
            fig.add_hline(y=0.10, line_dash="dash", line_color="#ef4444",
                          annotation_text="EXIT ≤ 0.10", annotation_font_color="#f87171")
            fig.add_hline(y=0.20, line_dash="dot", line_color="#fbbf24",
                          annotation_text="WATCH", annotation_font_color="#fbbf24")
            fig.update_layout(
                height=500,
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15,23,42,0.4)",
                margin=dict(l=20, r=20, t=40, b=140),
                xaxis=dict(tickangle=-45, tickfont=dict(size=10, color="#cbd5e1"), title=""),
                yaxis=dict(title="Relative Strength (RS)", title_font=dict(size=12, color="#cbd5e1"),
                           gridcolor="rgba(255,255,255,0.04)"),
                title=dict(
                    text=f"RS Rankings — Top {len(_rs_display)} of {len(rs_vals_all)} stocks (sorted by RS)",
                    font=dict(size=13, color="#94a3b8"), x=0.5, xanchor="center",
                ),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No RS data available.")
        
        st.markdown("#### 📈 RS Lines — Selectable Stocks vs Nifty 500")
        try:
            _n500 = yf.download("^CRSLDX", period="6mo", progress=False, auto_adjust=True)
            if isinstance(_n500.columns, pd.MultiIndex):
                _n500_close = _n500.xs("Close", axis=1, level=0).squeeze()
            else:
                _n500_close = _n500["Close"].squeeze()
            _n500_close = _n500_close.dropna().tail(123)
            if len(_n500_close) >= 20:
                _rs_available = [d["sym"] for d in unified_data[:50] if d.get("has_rs", False)]
                if _rs_available:
                    _default_sel = _rs_available[:5]
                    _sel_stocks = st.multiselect(
                        "Select stocks to compare (max 10 for clarity)",
                        options=_rs_available, default=_default_sel,
                        key="rs_line_selector", max_selections=10
                    )
                    if _sel_stocks:
                        with st.spinner("Loading RS data..."):
                            _sel_tickers = [s + ".NS" for s in _sel_stocks]
                            _all_data = yf.download(_sel_tickers, period="6mo", progress=False, auto_adjust=True)
                            _rs_lines = {}
                            for _s in _sel_stocks:
                                try:
                                    if isinstance(_all_data.columns, pd.MultiIndex):
                                        _sc = _all_data.xs(_s+".NS", axis=1, level=1)["Close"].squeeze()
                                    else:
                                        _sc = _all_data["Close"].squeeze()
                                    _sc = pd.Series(_sc).dropna().tail(123)
                                    if len(_sc) >= 20:
                                        _na = _n500_close.reindex(_sc.index).dropna()
                                        _sa = _sc.reindex(_na.index).dropna()
                                        if len(_sa) >= 20:
                                            _rs_lines[_s] = (_sa / _na) / (_sa.iloc[0] / _na.iloc[0]) * 100
                                except:
                                    pass
                            if _rs_lines:
                                _palette = ["#f72585","#7209b7","#3a0ca3","#4361ee","#4cc9f0",
                                            "#06d6a0","#ffd166","#ef476f","#118ab2","#073b4c"]
                                fig_rs = go.Figure()
                                _n500_norm = _n500_close / _n500_close.iloc[0] * 100
                                for _si, (_sym, _line) in enumerate(sorted(_rs_lines.items())):
                                    _c = _palette[_si % len(_palette)]
                                    _fr = _line.iloc[-1]
                                    fig_rs.add_trace(go.Scatter(
                                        x=_line.index, y=_line.values, mode="lines", name=_sym,
                                        line=dict(color=_c, width=2.5 if _fr >= 110 else 1.5,
                                                  dash="solid" if _fr >= 100 else "dash"),
                                        opacity=1.0 if _fr >= 100 else 0.7,
                                    ))
                                fig_rs.add_trace(go.Scatter(
                                    x=_n500_norm.index, y=_n500_norm.values,
                                    mode="lines", name="Nifty 500",
                                    line=dict(color="#64748b", width=2, dash="dot"),
                                ))
                                fig_rs.add_hline(y=100, line_color="#475569", line_width=1,
                                                  annotation_text="Index = 100", annotation_font_color="#475569")
                                fig_rs.update_layout(
                                    height=450, template="plotly_dark", xaxis_rangeslider_visible=False,
                                    yaxis=dict(title="RS (Rebased)", side="left", gridcolor="rgba(255,255,255,0.04)",
                                               range=[70, max(250, max(max(v.values) for v in _rs_lines.values() if len(v) > 0) + 20)]),
                                    margin=dict(l=20, r=20, t=10, b=20),
                                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.5)",
                                    legend=dict(font=dict(color="#94a3b8", size=10), orientation="h", y=1.05, x=0),
                                    hovermode="x unified",
                                )
                                st.plotly_chart(fig_rs, use_container_width=True, config={"displayModeBar": False})
                            else:
                                st.info("Could not compute RS lines for selected stocks.")
                else:
                    st.info("Select at least one stock to display its RS line.")
            else:
                st.info("Nifty 500 data unavailable (need ≥ 20 trading days).")
        except Exception as _e_rs:
            st.warning(f"RS Line chart: {_e_rs}")
        # ─────────────────────────────────────────────────────────────────────
        # RS Trend Analysis — MOVED HERE (directly below RS Distribution)
        # BUG FIX 3: guard ni < 123 to prevent index wrap-around crash
        # Extra fix: filter AMFI codes (numeric) from the stock selectbox
        # ─────────────────────────────────────────────────────────────────────
        st.markdown("#### 📈 RS Trend Analysis")
        _stock_only_portfolio = [s for s in _h_portfolio if not str(s).isdigit()]
        if not _stock_only_portfolio:
            _stock_only_portfolio = [d["sym"] for d in unified_data[:50] if d.get("has_rs", False)]
        col_t1, col_t2 = st.columns([1, 3])
        with col_t1:
            selected = st.selectbox(
                "Select Stock",
                _stock_only_portfolio if _stock_only_portfolio else ["N/A"],
                key="rs_trend_stock"
            )
            days = st.slider("Lookback", 60, 500, 252, key="rs_trend_lookback")
        with col_t2:
            if selected and selected != "N/A":
                with st.spinner(f"Loading {selected}..."):
                    try:
                        nifty_data = _h_get_nifty_2y()
                        if isinstance(nifty_data.columns, pd.MultiIndex):
                            nifty_c = nifty_data.xs("Close", axis=1, level=0).squeeze()
                        else:
                            nifty_c = nifty_data["Close"].squeeze()
                        nifty_c = pd.Series(nifty_c).dropna()
                        s = yf.download(selected + ".NS", period="2y", progress=False, auto_adjust=True)
                        if isinstance(s.columns, pd.MultiIndex):
                            sc = s.xs("Close", axis=1, level=0).squeeze()
                        else:
                            sc = s["Close"].squeeze()
                        sc = pd.Series(sc).dropna()
                        rs_series, dates = [], []
                        start_idx = max(123, len(sc) - days)
                        for i in range(start_idx, len(sc)):
                            cp, pp = float(sc.iloc[i]), float(sc.iloc[i - 123])
                            sr = cp / pp
                            d_date = sc.index[i]
                            ni = nifty_c.index.get_indexer([d_date], method="ffill")[0]
                            # BUG FIX 3: Guard against ni < 123 to prevent negative-index wrap-around
                            if ni < 123 or ni >= len(nifty_c):
                                continue
                            nr = float(nifty_c.iloc[ni]) / float(nifty_c.iloc[ni - 123])
                            if nr == 0:
                                continue
                            rs_series.append((sr / nr) - 1.0)
                            dates.append(d_date)
                        if rs_series:
                            fig_trend = go.Figure()
                            fig_trend.add_trace(go.Scatter(
                                x=dates, y=rs_series, mode="lines",
                                name=f"{selected} RS",
                                line=dict(color="#00d4aa", width=2),
                                fill="tozeroy", fillcolor="rgba(0,212,170,0.08)"
                            ))
                            fig_trend.add_hline(y=0.10, line_dash="dash", line_color="#ef4444",
                                                annotation_text="EXIT ≤ 0.10", annotation_font_color="#f87171")
                            fig_trend.add_hline(y=0.20, line_dash="dot", line_color="#fbbf24",
                                                annotation_text="WATCH", annotation_font_color="#fbbf24")
                            fig_trend.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.15)")
                            fig_trend.update_layout(
                                height=460, template="plotly_dark",
                                paper_bgcolor="rgba(0,0,0,0)",
                                plot_bgcolor="rgba(15,23,42,0.4)",
                                title=dict(text=f"{selected} — RS Line vs Nifty 50 ({days}d)",
                                           font=dict(size=13, color="#94a3b8"), x=0.5, xanchor="center"),
                                margin=dict(l=20, r=20, t=50, b=80),
                                xaxis=dict(tickangle=-45, tickfont=dict(size=10, color="#cbd5e1")),
                                yaxis=dict(title="RS", title_font=dict(size=11, color="#cbd5e1"),
                                           tickfont=dict(size=10, color="#cbd5e1"),
                                           gridcolor="rgba(255,255,255,0.04)"),
                            )
                            st.plotly_chart(fig_trend, use_container_width=True)
                            cur = rs_series[-1]
                            if cur <= 0.10: st.error(f"🚨 RS = {cur:.4f} — EXIT SIGNAL")
                            elif cur < 0.20: st.warning(f"⚠️ RS = {cur:.4f} — WATCH")
                        else: st.success(f"✅ RS = {cur:.4f} — HOLD")
                    except Exception as e:
                        st.error(f"Error loading RS trend: {e}")
            else:
                st.info("Select a stock to view its RS trend chart.")


# ──────────────────────────────────────────────────────────────────────────────
# TAB 8: GLOBAL ASSETS & THEMATIC SCREENING
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=86400)
def _load_excel_funds_mapping():
    import pandas as pd
    import os
    from mf_fetcher import resolve_mf_code
    excel_path = os.path.join("Core allocations", "global and inetrnational funds.xlsx")
    if not os.path.exists(excel_path):
        return {}
    try:
        df = pd.read_excel(excel_path)
        mapping = {}
        for _, row in df.iterrows():
            name = str(row["Name"]).strip()
            name_upper = name.upper()
            
            # Manual overrides
            if "NASDAQ 100 FOF" in name_upper or "NASDAQ 100 FUND OF FUND" in name_upper:
                mapping["145552"] = "Motilal Oswal Nasdaq 100 FoF"
            elif "KOTAK NASDAQ 100" in name_upper:
                mapping["148602"] = "Kotak Nasdaq 100 FoF"
            elif "US SPECIFIC EQUITY PASSIVE FOF" in name_upper:
                mapping["148662"] = "Kotak US Specific Equity Passive FoF"
            elif "TAIWAN EQUITY" in name_upper:
                mapping["149329"] = "Nippon India Taiwan Equity Fund"
            elif "JAPAN EQUITY" in name_upper:
                mapping["130860"] = "Nippon India Japan Equity Fund"
            elif "US EQUITY OPP" in name_upper:
                mapping["134923"] = "Nippon India US Equity Opportunities Fund"
            elif "GLOBAL X ARTIFICIAL" in name_upper or "ARTIFICIAL INTELL" in name_upper:
                mapping["150597"] = "Mirae Asset Global X AI & Tech ETF FoF"
            elif "GLOBAL CONSUMER TRENDS" in name_upper:
                mapping["148614"] = "Invesco India - Invesco Global Consumer Trends FoF"
            elif "EQQQ NASDAQ" in name_upper or "EQQQ" in name_upper:
                mapping["149236"] = "Invesco India - Invesco EQQQ NASDAQ 100 FoF"
            elif "US TREASURY BOND 0-1 YEAR" in name_upper:
                mapping["151838"] = "Bandhan US Treasury Bond 0-1 Year FoF"
            elif "US TREASURY 3-10 YEAR" in name_upper:
                mapping["151842"] = "Aditya Birla SL US Treasury 3-10 Year FoF"
            elif "EUROPE DYNAMIC EQUITY" in name_upper:
                mapping["140237"] = "Edelweiss Europe Dynamic Equity Offshore Fund"
            else:
                code = resolve_mf_code(name)
                if code:
                    mapping[str(code)] = name
        return mapping
    except Exception:
        return {}

with tab_global:
    st.markdown('<div style="border-top:2px solid rgba(255,255,255,0.08);margin:24px 0 16px;"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="background:linear-gradient(135deg,rgba(99,102,241,0.12),rgba(16,185,129,0.08));
                border:1px solid rgba(99,102,241,0.2); border-radius:14px;
                padding:14px 20px; margin-bottom:16px;">
      <div style="font-family:'Outfit';font-size:1.0rem;font-weight:700;color:#818cf8;">
        🌐 Global Assets & Thematic Screening Lab
      </div>
      <div style="font-size:0.78rem;color:#94a3b8;margin-top:3px;">
        Multi-window momentum ranking across 11 key international themes, commodities, and digital assets.
      </div>
    </div>
    """, unsafe_allow_html=True)
    
    _render_research_links(compact=True)
    
    # Predefined assets with metadata (Name, Category, Theme)
    predefined_assets = {
        # Global ETFs listed on NSE
        "MON100.NS": {"Name": "Motilal Oswal Nasdaq 100 ETF", "Cat": "Intl ETF", "Theme": "Broad Market US"},
        "MAFANG.NS": {"Name": "Mirae Asset NYSE FANG+ ETF", "Cat": "Intl ETF", "Theme": "AI & Big Tech"},
        "HNGSNGBEES.NS": {"Name": "Nippon India ETF Hang Seng BeES", "Cat": "Intl ETF", "Theme": "China & HK"},
        "MAHKTECH.NS": {"Name": "Mirae Asset Hang Seng TECH ETF", "Cat": "Intl ETF", "Theme": "China & HK Tech"},
        "MASPTOP50.NS": {"Name": "Mirae Asset S&P 500 Top 50 ETF", "Cat": "Intl ETF", "Theme": "Broad Market US"},
        # Commodities
        "GOLDBEES.NS": {"Name": "Gold BeES", "Cat": "Commodities", "Theme": "Precious Metals"},
        "SILVERBEES.NS": {"Name": "Silver BeES", "Cat": "Commodities", "Theme": "Precious Metals"},
        "HG=F": {"Name": "Copper Futures", "Cat": "Commodities", "Theme": "Industrial Metals"},
        "ALI=F": {"Name": "Aluminum Futures", "Cat": "Commodities", "Theme": "Industrial Metals"},
        "DBB": {"Name": "Invesco DB Base Metals (Zinc/Copper/Alu)", "Cat": "Commodities", "Theme": "Industrial Metals"},
        # Cryptocurrency
        "BTC-USD": {"Name": "Bitcoin USD", "Cat": "Crypto", "Theme": "Cryptocurrency"},
        # US Thematic ETFs
        "SMH": {"Name": "VanEck Semiconductor ETF", "Cat": "US Thematic ETF", "Theme": "Semiconductors"},
        "SOXX": {"Name": "iShares Semiconductor ETF", "Cat": "US Thematic ETF", "Theme": "Semiconductors"},
        "AIQ": {"Name": "Global X Artificial Intelligence & Tech", "Cat": "US Thematic ETF", "Theme": "AI & Robotics"},
        "GRID": {"Name": "First Trust Clean Edge Smart Grid", "Cat": "US Thematic ETF", "Theme": "Power & Cooling"},
        "SRVR": {"Name": "Pacer Benchmark Data Center Real Estate", "Cat": "US Thematic ETF", "Theme": "Data Centers"},
        "XLU": {"Name": "Utilities Select Sector SPDR", "Cat": "US Thematic ETF", "Theme": "Power & Cooling"},
        "BOTZ": {"Name": "Global X Robotics & Artificial Intelligence", "Cat": "US Thematic ETF", "Theme": "AI & Robotics"},
        "ROBO": {"Name": "ROBO Global Robotics & Automation", "Cat": "US Thematic ETF", "Theme": "AI & Robotics"},
        "OZEM": {"Name": "Roundhill GLP-1 & Weight Loss ETF", "Cat": "US Thematic ETF", "Theme": "Biotech & Healthcare"},
        "XBI": {"Name": "SPDR S&P Biotech ETF", "Cat": "US Thematic ETF", "Theme": "Biotech & Healthcare"},
        "IBB": {"Name": "iShares Biotechnology ETF", "Cat": "US Thematic ETF", "Theme": "Biotech & Healthcare"},
        "ARKG": {"Name": "ARK Genomic Revolution ETF", "Cat": "US Thematic ETF", "Theme": "Biotech & Healthcare"},
        "WGMI": {"Name": "Valkyrie Bitcoin Miners ETF", "Cat": "US Thematic ETF", "Theme": "Crypto Equities"},
        "BLOK": {"Name": "Amplify Transformational Data Sharing", "Cat": "US Thematic ETF", "Theme": "Crypto Equities"},
        "LIT": {"Name": "Global X Lithium & Battery Tech", "Cat": "US Thematic ETF", "Theme": "Critical Materials"},
        "REMX": {"Name": "VanEck Rare Earth/Strategic Metals", "Cat": "US Thematic ETF", "Theme": "Critical Materials"},
        "URA": {"Name": "Global X Uranium ETF", "Cat": "US Thematic ETF", "Theme": "Nuclear & Uranium"},
        "BUG": {"Name": "Global X Cybersecurity ETF", "Cat": "US Thematic ETF", "Theme": "Cybersecurity"},
        "CIBR": {"Name": "First Trust Nasdaq Cybersecurity ETF", "Cat": "US Thematic ETF", "Theme": "Cybersecurity"},
        "ITA": {"Name": "iShares US Aerospace & Defense ETF", "Cat": "US Thematic ETF", "Theme": "Defense & Aerospace"},
        "WCLD": {"Name": "WisdomTree Cloud Computing Fund", "Cat": "US Thematic ETF", "Theme": "Enterprise SaaS"},
    }
    
    with st.spinner("Synchronizing global assets and Excel funds..."):
        # Load Excel funds
        excel_mapping = _load_excel_funds_mapping()
        
        # Combine all assets
        all_assets = {}
        for ticker, meta in predefined_assets.items():
            all_assets[ticker] = meta
            
        for code, name in excel_mapping.items():
            if code not in all_assets:
                all_assets[code] = {"Name": name, "Cat": "Excel Mutual Fund", "Theme": "Global Allocation"}
                
        # Process data
        processed_data = []
        from cache_manager import get_historical_data
        
        for ticker, meta in all_assets.items():
            try:
                df = get_historical_data(ticker, days=365)
                if df is not None and not df.empty and len(df) >= 20:
                    df = df.sort_index()
                    close_s = df["Close"].squeeze()
                    if isinstance(close_s, pd.DataFrame):
                        close_s = close_s.iloc[:, 0]
                    curr_price = float(close_s.iloc[-1])
                    
                    # SMAs
                    sma_50 = float(close_s.rolling(50, min_periods=10).mean().iloc[-1])
                    sma_200 = float(close_s.rolling(200, min_periods=30).mean().iloc[-1])
                    
                    # Returns helper
                    def compute_ret(periods):
                        if len(close_s) > periods:
                            p_start = float(close_s.iloc[-periods - 1])
                            return (curr_price - p_start) / p_start * 100.0
                        return 0.0
                        
                    ret_15d = compute_ret(10)
                    ret_1m = compute_ret(21)
                    ret_2m = compute_ret(42)
                    ret_3m = compute_ret(63)
                    ret_6m = compute_ret(126)
                    ret_12m = compute_ret(252)
                    
                    # Volatility
                    daily_ret = close_s.pct_change().dropna()
                    vol_12m = float(daily_ret.std() * np.sqrt(252) * 100.0) if len(daily_ret) > 30 else 0.0
                    vol_1m = float(daily_ret.tail(21).std() * np.sqrt(252) * 100.0) if len(daily_ret) > 5 else vol_12m
                    
                    # Score
                    weighted_ret = (0.20 * (ret_15d / 100.0)) + (0.30 * (ret_1m / 100.0)) + (0.25 * (ret_2m / 100.0)) + (0.15 * (ret_3m / 100.0)) + (0.10 * (ret_6m / 100.0))
                    max_risk = max(vol_12m, vol_1m)
                    composite_score = (weighted_ret / max(0.1, max_risk / 100.0)) * 100.0
                    
                    # Stage
                    if curr_price > sma_50 and sma_50 > sma_200:
                        stage = "Stage 2 (Advancing)"
                        stage_color = "#10b981"
                    elif curr_price < sma_50 and sma_50 < sma_200:
                        stage = "Stage 4 (Declining)"
                        stage_color = "#ef4444"
                    elif curr_price > sma_50 and sma_50 < sma_200:
                        stage = "Stage 1 (Basing)"
                        stage_color = "#3b82f6"
                    else:
                        stage = "Stage 3 (Topping)"
                        stage_color = "#fbbf24"
                        
                    processed_data.append({
                        "Symbol": ticker,
                        "Name": meta["Name"],
                        "Category": meta["Cat"],
                        "Theme": meta["Theme"],
                        "Price": curr_price,
                        "1M%": ret_1m,
                        "3M%": ret_3m,
                        "6M%": ret_6m,
                        "12M%": ret_12m,
                        "Volatility%": vol_12m,
                        "Score": composite_score,
                        "Stage": stage,
                        "StageColor": stage_color,
                        "df": df
                    })
            except Exception as e_proc:
                pass
                
        # Rank by Score descending
        processed_data = sorted(processed_data, key=lambda x: x["Score"], reverse=True)
        for i, item in enumerate(processed_data):
            item["Rank"] = i + 1
            
    if processed_data:
        df_display_all = pd.DataFrame(processed_data)
        
        # ── KPI Summary Cards ──
        kc1, kc2, kc3, kc4 = st.columns(4)
        with kc1:
            top_momentum = processed_data[0]
            st.markdown(f"""
            <div class='kpi-card-v4'>
              <div class='kpi-title-v4'>🔥 Top Momentum Asset</div>
              <div class='kpi-value-v4' style='color:#a78bfa;'>{top_momentum['Symbol']}</div>
              <div class='kpi-desc-v4'>{top_momentum['Name'][:28]}<br>Score: <b>{top_momentum['Score']:.1f}</b></div>
            </div>
            """, unsafe_allow_html=True)
        with kc2:
            btc_data = next((x for x in processed_data if x["Symbol"] == "BTC-USD"), None)
            if btc_data:
                st.markdown(f"""
                <div class='kpi-card-v4'>
                  <div class='kpi-title-v4'>🪙 Digital Reserve</div>
                  <div class='kpi-value-v4' style='color:#f59e0b;'>BTC-USD</div>
                  <div class='kpi-desc-v4'>Price: <b>${btc_data['Price']:,.0f}</b><br>Stage: <b style='color:{btc_data['StageColor']}'>{btc_data['Stage'].split()[0]}</b></div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""<div class='kpi-card-v4'><div class='kpi-title-v4'>🪙 Digital Reserve</div><div class='kpi-value-v4'>N/A</div></div>""", unsafe_allow_html=True)
        with kc3:
            gold_data = next((x for x in processed_data if x["Symbol"] == "GOLDBEES.NS"), None)
            if gold_data:
                st.markdown(f"""
                <div class='kpi-card-v4'>
                  <div class='kpi-title-v4'>🟡 Precious Metals</div>
                  <div class='kpi-value-v4' style='color:#fbbf24;'>Gold BeES</div>
                  <div class='kpi-desc-v4'>Price: <b>₹{gold_data['Price']:.2f}</b><br>1M: <b style='color:#10b981;'>+{gold_data['1M%']:.1f}%</b></div>
                </div>
                """, unsafe_allow_html=True)
            else:
                 st.markdown("""<div class='kpi-card-v4'><div class='kpi-title-v4'>🟡 Precious Metals</div><div class='kpi-value-v4'>N/A</div></div>""", unsafe_allow_html=True)
        with kc4:
            semis = [x for x in processed_data if x["Theme"] == "Semiconductors"]
            if semis:
                top_semi = semis[0]
                st.markdown(f"""
                <div class='kpi-card-v4'>
                  <div class='kpi-title-v4'>⚡ Top Semiconductor</div>
                  <div class='kpi-value-v4' style='color:#3b82f6;'>{top_semi['Symbol']}</div>
                  <div class='kpi-desc-v4'>Score: <b>{top_semi['Score']:.1f}</b><br>3M: <b style='color:#10b981;'>+{top_semi['3M%']:.1f}%</b></div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""<div class='kpi-card-v4'><div class='kpi-title-v4'>⚡ Top Semiconductor</div><div class='kpi-value-v4'>N/A</div></div>""", unsafe_allow_html=True)
                
        # ── Filters & Category Sub-tabs ──
        sub_tab_names = [
            "🏆 All Ranked Assets",
            "📊 International ETFs (NSE)",
            "🏛️ Indian Mutual Funds (Excel)",
            "🇺🇸 US Thematic ETFs",
            "🔩 Commodities & Crypto"
        ]
        stab_all, stab_etf, stab_mf, stab_us, stab_comm = st.tabs(sub_tab_names)
        
        def render_thematic_table(df_subset, key_suffix):
            search_query = st.text_input("🔍 Filter by Name/Symbol/Theme...", key=f"global_search_{key_suffix}").upper()
            if search_query:
                df_subset = df_subset[
                    df_subset["Symbol"].astype(str).str.upper().str.contains(search_query) |
                    df_subset["Name"].astype(str).str.upper().str.contains(search_query) |
                    df_subset["Theme"].astype(str).str.upper().str.contains(search_query)
                ]
            
            if df_subset.empty:
                st.info("No matching assets found.")
                return
                
            # Render custom HTML table
            html_rows = []
            for _, r in df_subset.iterrows():
                stage_badge = f"<span style='background:rgba(255,255,255,0.04);color:{r['StageColor']};border:1px solid {r['StageColor']}40;border-radius:6px;padding:2px 8px;font-size:0.72rem;font-weight:700;'>{r['Stage']}</span>"
                
                # Format returns with plus signs and green/red colors
                def fmt_ret(val):
                    c = "#10b981" if val >= 0 else "#ef4444"
                    sign = "+" if val >= 0 else ""
                    return f"<span style='color:{c};font-weight:600;'>{sign}{val:.1f}%</span>"
                
                score_color = "#10b981" if r["Score"] >= 50 else ("#fbbf24" if r["Score"] >= 20 else "#ef4444")
                score_badge = f"<span style='color:{score_color};font-weight:700;'>{r['Score']:.1f}</span>"
                
                html_rows.append(
                    f"<tr>"
                    f"<td style='text-align:center;font-weight:700;color:#94a3b8;'>{r['Rank']}</td>"
                    f"<td style='font-weight:700;color:#e2e8f0;'>{r['Symbol']}</td>"
                    f"<td style='color:#94a3b8;font-size:0.76rem;'>{r['Name']}</td>"
                    f"<td style='color:#a78bfa;font-size:0.73rem;font-weight:600;'>{r['Theme']}</td>"
                    f"<td style='text-align:center;'>{stage_badge}</td>"
                    f"<td style='text-align:right;font-weight:700;'>{score_badge}</td>"
                    f"<td style='text-align:right;'>{fmt_ret(r['1M%'])}</td>"
                    f"<td style='text-align:right;'>{fmt_ret(r['3M%'])}</td>"
                    f"<td style='text-align:right;'>{fmt_ret(r['6M%'])}</td>"
                    f"<td style='text-align:right;'>{fmt_ret(r['12M%'])}</td>"
                    f"<td style='text-align:right;color:#cbd5e1;'>{r['Volatility%']:.1f}%</td>"
                    f"</tr>"
                )
                
            table_html = f"""
            <div style='overflow-x:auto;'>
              <table class='screening-table' style='width:100%;border-collapse:collapse;'>
                <thead>
                  <tr style='border-bottom:1px solid rgba(255,255,255,0.08);'>
                    <th style='text-align:center;'>Rank</th>
                    <th>Symbol</th>
                    <th>Asset Name</th>
                    <th>Theme</th>
                    <th style='text-align:center;'>Trend Stage</th>
                    <th style='text-align:right;'>Score</th>
                    <th style='text-align:right;'>1M%</th>
                    <th style='text-align:right;'>3M%</th>
                    <th style='text-align:right;'>6M%</th>
                    <th style='text-align:right;'>12M%</th>
                    <th style='text-align:right;'>Vol 12M</th>
                  </tr>
                </thead>
                <tbody>
                  {"".join(html_rows)}
                </tbody>
              </table>
            </div>
            """
            st.markdown(table_html, unsafe_allow_html=True)
            
        with stab_all:
            st.markdown("#### 🏆 Top Global & Thematic Assets by Momentum Score")
            render_thematic_table(df_display_all, "all")
            
        with stab_etf:
            st.markdown("#### 📊 International ETFs Listed on NSE")
            df_sub = df_display_all[df_display_all["Category"] == "Intl ETF"]
            render_thematic_table(df_sub, "etf")
            
        with stab_mf:
            st.markdown("#### 🏛️ Indian Mutual Funds (Resolved from Excel Sheet)")
            df_sub = df_display_all[df_display_all["Category"] == "Excel Mutual Fund"]
            render_thematic_table(df_sub, "mf")
            
        with stab_us:
            st.markdown("#### 🇺🇸 US Thematic ETFs (AI, Power Infrastructure, Biotech, Robotics, SaaS)")
            df_sub = df_display_all[df_display_all["Category"] == "US Thematic ETF"]
            render_thematic_table(df_sub, "us")
            
        with stab_comm:
            st.markdown("#### 🔩 Global Commodities (Gold, Silver, Base Metals) & Cryptocurrency")
            df_sub = df_display_all[df_display_all["Category"].isin(["Commodities", "Crypto"])]
            render_thematic_table(df_sub, "comm")
            
        # ── Comparative Performance Chart ──
        st.markdown('<div style="margin:24px 0 16px;"></div>', unsafe_allow_html=True)
        st.markdown("### 📊 Relative Performance & Strength Chart")
        
        c_col1, c_col2 = st.columns([2, 1])
        with c_col1:
            chart_tickers = st.multiselect(
                "Select assets to plot",
                options=df_display_all["Symbol"].tolist(),
                default=df_display_all["Symbol"].tolist()[:5],
                key="global_chart_select"
            )
        with c_col2:
            chart_mode = st.selectbox(
                "Chart Mode",
                [
                    "Price Performance (Smoothed 30d SMA)",
                    "Relative Strength vs Nifty 50 (Smoothed 30d SMA)",
                    "Relative Strength vs S&P 500 (Smoothed 30d SMA)"
                ],
                index=0,
                key="global_chart_mode"
            )
        
        if chart_tickers:
            import plotly.graph_objects as go
            fig_perf = go.Figure()
            
            # Load benchmark if needed
            bench_series = None
            if "Nifty 50" in chart_mode:
                bench_df = get_historical_data("^NSEI", days=365)
                if bench_df is not None and not bench_df.empty:
                    bench_series = bench_df["Close"].squeeze()
            elif "S&P 500" in chart_mode:
                bench_df = get_historical_data("^GSPC", days=365)
                if bench_df is not None and not bench_df.empty:
                    bench_series = bench_df["Close"].squeeze()
            
            if bench_series is not None and isinstance(bench_series, pd.DataFrame):
                bench_series = bench_series.iloc[:, 0]
                
            asset_series = {}
            for t in chart_tickers:
                row_asset = next((x for x in processed_data if x["Symbol"] == t), None)
                if row_asset and row_asset["df"] is not None and not row_asset["df"].empty:
                    close_c = row_asset["df"]["Close"].squeeze()
                    if isinstance(close_c, pd.DataFrame):
                        close_c = close_c.iloc[:, 0]
                    
                    if bench_series is not None:
                        # Align dates with benchmark
                        df_align = pd.DataFrame({"Asset": close_c, "Bench": bench_series}).dropna()
                        if not df_align.empty:
                            rs_ratio = df_align["Asset"] / df_align["Bench"]
                            smoothed = rs_ratio.rolling(window=30, min_periods=5).mean()
                            asset_series[t] = smoothed
                    else:
                        smoothed = close_c.rolling(window=30, min_periods=5).mean()
                        asset_series[t] = smoothed
            
            if asset_series:
                df_combined = pd.DataFrame(asset_series).dropna(how="all").sort_index()
                # Find date where >= 75% of assets are available
                min_available = max(1, int(0.75 * len(asset_series)))
                non_null_counts = df_combined.notnull().sum(axis=1)
                valid_dates = df_combined.index[non_null_counts >= min_available]
                
                if len(valid_dates) > 0:
                    start_date = valid_dates[0]
                    df_combined = df_combined.loc[start_date:]
                
                for t in chart_tickers:
                    if t in df_combined.columns:
                        series = df_combined[t].dropna()
                        if not series.empty:
                            # Rebase to 100 on the first available aligned date
                            rebased = (series / float(series.iloc[0])) * 100.0
                            
                            # Get name
                            row_asset = next((x for x in processed_data if x["Symbol"] == t), None)
                            asset_name = row_asset["Name"] if row_asset else t
                            if len(asset_name) > 30:
                                asset_name = asset_name[:27] + "..."
                            display_label = f"{t} ({asset_name})"
                            
                            fig_perf.add_trace(go.Scatter(
                                x=series.index,
                                y=rebased,
                                mode="lines",
                                name=display_label,
                                line=dict(width=2)
                             ))
            
            y_title = "Rebased RS (Inception = 100)" if bench_series is not None else "Rebased Price (Inception = 100)"
            fig_perf.update_layout(
                height=480,
                template="plotly_dark",
                xaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
                yaxis=dict(title=y_title, gridcolor="rgba(255,255,255,0.04)"),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15,23,42,0.4)",
                legend=dict(font=dict(color="#94a3b8", size=9), orientation="h", y=1.12, x=0),
                margin=dict(l=20, r=20, t=10, b=20),
                hovermode="x unified"
            )
            st.plotly_chart(fig_perf, use_container_width=True)
        else:
            st.info("Select one or more assets to plot their performance comparison.")
    else:
        st.info("No global asset data successfully synchronized. Verify internet connection or yfinance cache.")