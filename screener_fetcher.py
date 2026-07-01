import os
from functools import lru_cache
import sys
import json
import time
import requests
import datetime
import pandas as pd
import re
import yfinance as yf
from bs4 import BeautifulSoup
from utils import log_info, log_warning, log_success

# Base Workspace directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Default fundamental profile matching Indian market metrics (fallback database)
# Quality Gate fields added: Net_NPA_%, CAR_%, ROA_% (BFSI), Promoter_Pledge_%, TTM_CFO (all)
FUNDAMENTAL_DB = {
    "ACUTAAS":   {"ROE_%": 24.5, "Profit_Growth_%": 35.2, "Sales_Growth_%": 28.6, "Debt_to_Equity": 0.15, "PE_Ratio": 32.4, "Smart_Score": 9, "ROCE_3Yr_Avg": 25.0, "CFO_PAT_3Yr_Avg": 0.85, "ROE_3Yr_Avg": 24.5, "Market_Cap_Cr": 25500.0,  "Promoter_Pledge_%": 0.0,  "TTM_CFO": 180.0,  "Net_NPA_%": 0.0,  "CAR_%": 0.0,  "ROA_%": 0.0},
    "SKYGOLD":   {"ROE_%": 22.1, "Profit_Growth_%": 45.8, "Sales_Growth_%": 38.4, "Debt_to_Equity": 0.45, "PE_Ratio": 28.2, "Smart_Score": 8, "ROCE_3Yr_Avg": 21.0, "CFO_PAT_3Yr_Avg": 0.78, "ROE_3Yr_Avg": 22.1, "Market_Cap_Cr": 2500.0,   "Promoter_Pledge_%": 2.5,  "TTM_CFO": 45.0,   "Net_NPA_%": 0.0,  "CAR_%": 0.0,  "ROA_%": 0.0},
    "BSE":       {"ROE_%": 38.6, "Profit_Growth_%": 55.4, "Sales_Growth_%": 42.1, "Debt_to_Equity": 0.00, "PE_Ratio": 48.6, "Smart_Score": 9, "ROCE_3Yr_Avg": 35.0, "CFO_PAT_3Yr_Avg": 1.10, "ROE_3Yr_Avg": 38.6, "Market_Cap_Cr": 35000.0,  "Promoter_Pledge_%": 0.0,  "TTM_CFO": 850.0,  "Net_NPA_%": 0.0,  "CAR_%": 42.0, "ROA_%": 3.5},
    "SHILPAMED": {"ROE_%": 14.8, "Profit_Growth_%": 22.6, "Sales_Growth_%": 18.2, "Debt_to_Equity": 0.38, "PE_Ratio": 38.4, "Smart_Score": 7, "ROCE_3Yr_Avg": 12.0, "CFO_PAT_3Yr_Avg": 0.65, "ROE_3Yr_Avg": 14.8, "Market_Cap_Cr": 6000.0,   "Promoter_Pledge_%": 0.0,  "TTM_CFO": 52.0,   "Net_NPA_%": 0.0,  "CAR_%": 0.0,  "ROA_%": 0.0},
    "SOLARINDS": {"ROE_%": 28.4, "Profit_Growth_%": 29.1, "Sales_Growth_%": 24.5, "Debt_to_Equity": 0.12, "PE_Ratio": 52.1, "Smart_Score": 8, "ROCE_3Yr_Avg": 30.0, "CFO_PAT_3Yr_Avg": 0.90, "ROE_3Yr_Avg": 28.4, "Market_Cap_Cr": 85000.0,  "Promoter_Pledge_%": 0.0,  "TTM_CFO": 620.0,  "Net_NPA_%": 0.0,  "CAR_%": 0.0,  "ROA_%": 0.0},
    "LLOYDSME":  {"ROE_%": 19.5, "Profit_Growth_%": 20.4, "Sales_Growth_%": 16.8, "Debt_to_Equity": 0.22, "PE_Ratio": 22.4, "Smart_Score": 6, "ROCE_3Yr_Avg": 18.0, "CFO_PAT_3Yr_Avg": 0.75, "ROE_3Yr_Avg": 19.5, "Market_Cap_Cr": 40000.0,  "Promoter_Pledge_%": 0.0,  "TTM_CFO": 410.0,  "Net_NPA_%": 0.0,  "CAR_%": 0.0,  "ROA_%": 0.0},
    "FINCABLES": {"ROE_%": 16.2, "Profit_Growth_%": 18.5, "Sales_Growth_%": 14.2, "Debt_to_Equity": 0.05, "PE_Ratio": 25.6, "Smart_Score": 7, "ROCE_3Yr_Avg": 17.5, "CFO_PAT_3Yr_Avg": 0.82, "ROE_3Yr_Avg": 16.2, "Market_Cap_Cr": 17000.0,  "Promoter_Pledge_%": 0.0,  "TTM_CFO": 310.0,  "Net_NPA_%": 0.0,  "CAR_%": 0.0,  "ROA_%": 0.0},
    "CGPOWER":   {"ROE_%": 26.8, "Profit_Growth_%": 24.2, "Sales_Growth_%": 21.0, "Debt_to_Equity": 0.02, "PE_Ratio": 68.2, "Smart_Score": 8, "ROCE_3Yr_Avg": 28.0, "CFO_PAT_3Yr_Avg": 0.95, "ROE_3Yr_Avg": 26.8, "Market_Cap_Cr": 110000.0, "Promoter_Pledge_%": 0.0,  "TTM_CFO": 1200.0, "Net_NPA_%": 0.0,  "CAR_%": 0.0,  "ROA_%": 0.0},
    "J&KBANK":   {"ROE_%": 15.4, "Profit_Growth_%": 28.9, "Sales_Growth_%": 12.5, "Debt_to_Equity": 0.85, "PE_Ratio": 6.8,  "Smart_Score": 7, "ROCE_3Yr_Avg": 10.0, "CFO_PAT_3Yr_Avg": 0.88, "ROE_3Yr_Avg": 15.4, "Market_Cap_Cr": 13000.0,  "Promoter_Pledge_%": 0.0,  "TTM_CFO": 1500.0, "Net_NPA_%": 0.68, "CAR_%": 14.2, "ROA_%": 1.10},
    "APARINDS":  {"ROE_%": 25.6, "Profit_Growth_%": 32.1, "Sales_Growth_%": 26.4, "Debt_to_Equity": 0.35, "PE_Ratio": 35.2, "Smart_Score": 8, "ROCE_3Yr_Avg": 27.0, "CFO_PAT_3Yr_Avg": 0.84, "ROE_3Yr_Avg": 25.6, "Market_Cap_Cr": 35000.0,  "Promoter_Pledge_%": 0.0,  "TTM_CFO": 420.0,  "Net_NPA_%": 0.0,  "CAR_%": 0.0,  "ROA_%": 0.0},
    "SAIL":      {"ROE_%": 8.4,  "Profit_Growth_%": -12.4,"Sales_Growth_%": 4.2,  "Debt_to_Equity": 0.95, "PE_Ratio": 11.2, "Smart_Score": 5, "ROCE_3Yr_Avg": 9.0,  "CFO_PAT_3Yr_Avg": 0.60, "ROE_3Yr_Avg": 8.4,  "Market_Cap_Cr": 55000.0,  "Promoter_Pledge_%": 0.0,  "TTM_CFO": 3800.0, "Net_NPA_%": 0.0,  "CAR_%": 0.0,  "ROA_%": 0.0},
    "GRANULES":  {"ROE_%": 12.6, "Profit_Growth_%": 14.5, "Sales_Growth_%": 11.8, "Debt_to_Equity": 0.28, "PE_Ratio": 21.4, "Smart_Score": 6, "ROCE_3Yr_Avg": 14.0, "CFO_PAT_3Yr_Avg": 0.72, "ROE_3Yr_Avg": 12.6, "Market_Cap_Cr": 11000.0,  "Promoter_Pledge_%": 8.2,  "TTM_CFO": 280.0,  "Net_NPA_%": 0.0,  "CAR_%": 0.0,  "ROA_%": 0.0},
    "HINDALCO":  {"ROE_%": 11.8, "Profit_Growth_%": 8.2,  "Sales_Growth_%": 6.4,  "Debt_to_Equity": 0.58, "PE_Ratio": 14.8, "Smart_Score": 6, "ROCE_3Yr_Avg": 12.0, "CFO_PAT_3Yr_Avg": 0.76, "ROE_3Yr_Avg": 11.8, "Market_Cap_Cr": 140000.0, "Promoter_Pledge_%": 0.0,  "TTM_CFO": 8500.0, "Net_NPA_%": 0.0,  "CAR_%": 0.0,  "ROA_%": 0.0},
    "HINDCOPPER":{"ROE_%": 10.2, "Profit_Growth_%": 12.1, "Sales_Growth_%": 8.5,  "Debt_to_Equity": 0.42, "PE_Ratio": 42.1, "Smart_Score": 5, "ROCE_3Yr_Avg": 11.0, "CFO_PAT_3Yr_Avg": 0.70, "ROE_3Yr_Avg": 10.2, "Market_Cap_Cr": 30000.0,  "Promoter_Pledge_%": 0.0,  "TTM_CFO": 680.0,  "Net_NPA_%": 0.0,  "CAR_%": 0.0,  "ROA_%": 0.0},
    "KIRLOSENG": {"ROE_%": 16.5, "Profit_Growth_%": 19.8, "Sales_Growth_%": 15.2, "Debt_to_Equity": 0.18, "PE_Ratio": 24.6, "Smart_Score": 7, "ROCE_3Yr_Avg": 17.0, "CFO_PAT_3Yr_Avg": 0.80, "ROE_3Yr_Avg": 16.5, "Market_Cap_Cr": 13000.0,  "Promoter_Pledge_%": 0.0,  "TTM_CFO": 195.0,  "Net_NPA_%": 0.0,  "CAR_%": 0.0,  "ROA_%": 0.0},
    "NLCINDIA":  {"ROE_%": 9.8,  "Profit_Growth_%": 6.4,  "Sales_Growth_%": 5.2,  "Debt_to_Equity": 1.25, "PE_Ratio": 16.4, "Smart_Score": 5, "ROCE_3Yr_Avg": 10.0, "CFO_PAT_3Yr_Avg": 0.68, "ROE_3Yr_Avg": 9.8,  "Market_Cap_Cr": 35000.0,  "Promoter_Pledge_%": 0.0,  "TTM_CFO": 2100.0, "Net_NPA_%": 0.0,  "CAR_%": 0.0,  "ROA_%": 0.0},
    "SCHNEIDER": {"ROE_%": 15.0, "Profit_Growth_%": 15.0, "Sales_Growth_%": 10.5, "Debt_to_Equity": 0.82, "PE_Ratio": 95.0, "Smart_Score": 4, "ROCE_3Yr_Avg": 14.5, "CFO_PAT_3Yr_Avg": 0.75, "ROE_3Yr_Avg": 15.0, "Market_Cap_Cr": 20000.0,  "Promoter_Pledge_%": 0.0,  "TTM_CFO": 210.0,  "Net_NPA_%": 0.0,  "CAR_%": 0.0,  "ROA_%": 0.0},
    "ANGELONE":  {"ROE_%": 34.2, "Profit_Growth_%": 26.5, "Sales_Growth_%": 31.4, "Debt_to_Equity": 0.48, "PE_Ratio": 19.8, "Smart_Score": 8, "ROCE_3Yr_Avg": 32.0, "CFO_PAT_3Yr_Avg": 0.92, "ROE_3Yr_Avg": 34.2, "Market_Cap_Cr": 22000.0,  "Promoter_Pledge_%": 0.0,  "TTM_CFO": 920.0,  "Net_NPA_%": 0.0,  "CAR_%": 38.0, "ROA_%": 4.2},
    "SANSERA":   {"ROE_%": 13.8, "Profit_Growth_%": 16.4, "Sales_Growth_%": 12.8, "Debt_to_Equity": 0.52, "PE_Ratio": 29.5, "Smart_Score": 6, "ROCE_3Yr_Avg": 14.0, "CFO_PAT_3Yr_Avg": 0.74, "ROE_3Yr_Avg": 13.8, "Market_Cap_Cr": 6000.0,   "Promoter_Pledge_%": 0.0,  "TTM_CFO": 185.0,  "Net_NPA_%": 0.0,  "CAR_%": 0.0,  "ROA_%": 0.0},
    "KTKBANK":   {"ROE_%": 12.4, "Profit_Growth_%": 18.2, "Sales_Growth_%": 9.8,  "Debt_to_Equity": 0.90, "PE_Ratio": 5.2,  "Smart_Score": 6, "ROCE_3Yr_Avg": 8.0,  "CFO_PAT_3Yr_Avg": 0.80, "ROE_3Yr_Avg": 12.4, "Market_Cap_Cr": 9000.0,   "Promoter_Pledge_%": 0.0,  "TTM_CFO": 1200.0, "Net_NPA_%": 0.82, "CAR_%": 17.5, "ROA_%": 1.05},
    "LAURUSLABS":{"ROE_%": 9.4,  "Profit_Growth_%": -15.2,"Sales_Growth_%": 2.1,  "Debt_to_Equity": 0.68, "PE_Ratio": 58.4, "Smart_Score": 5, "ROCE_3Yr_Avg": 10.0, "CFO_PAT_3Yr_Avg": 0.50, "ROE_3Yr_Avg": 9.4,  "Market_Cap_Cr": 22000.0,  "Promoter_Pledge_%": 5.8,  "TTM_CFO": 320.0,  "Net_NPA_%": 0.0,  "CAR_%": 0.0,  "ROA_%": 0.0},
    "JINDALSAW": {"ROE_%": 18.2, "Profit_Growth_%": 32.4, "Sales_Growth_%": 19.5, "Debt_to_Equity": 0.72, "PE_Ratio": 12.4, "Smart_Score": 7, "ROCE_3Yr_Avg": 16.5, "CFO_PAT_3Yr_Avg": 0.72, "ROE_3Yr_Avg": 18.2, "Market_Cap_Cr": 17000.0,  "Promoter_Pledge_%": 0.0,  "TTM_CFO": 680.0,  "Net_NPA_%": 0.0,  "CAR_%": 0.0,  "ROA_%": 0.0},
    "CUMMINSIND":{"ROE_%": 22.4, "Profit_Growth_%": 21.5, "Sales_Growth_%": 16.4, "Debt_to_Equity": 0.05, "PE_Ratio": 42.8, "Smart_Score": 8, "ROCE_3Yr_Avg": 24.0, "CFO_PAT_3Yr_Avg": 0.86, "ROE_3Yr_Avg": 22.4, "Market_Cap_Cr": 90000.0,  "Promoter_Pledge_%": 0.0,  "TTM_CFO": 1150.0, "Net_NPA_%": 0.0,  "CAR_%": 0.0,  "ROA_%": 0.0},
    "GVT&D":     {"ROE_%": 14.2, "Profit_Growth_%": 26.4, "Sales_Growth_%": 18.2, "Debt_to_Equity": 0.42, "PE_Ratio": 38.6, "Smart_Score": 7, "ROCE_3Yr_Avg": 15.5, "CFO_PAT_3Yr_Avg": 0.78, "ROE_3Yr_Avg": 14.2, "Market_Cap_Cr": 30000.0,  "Promoter_Pledge_%": 0.0,  "TTM_CFO": 380.0,  "Net_NPA_%": 0.0,  "CAR_%": 0.0,  "ROA_%": 0.0},
    "ABB":       {"ROE_%": 21.8, "Profit_Growth_%": 28.4, "Sales_Growth_%": 22.1, "Debt_to_Equity": 0.01, "PE_Ratio": 82.4, "Smart_Score": 8, "ROCE_3Yr_Avg": 23.0, "CFO_PAT_3Yr_Avg": 0.88, "ROE_3Yr_Avg": 21.8, "Market_Cap_Cr": 180000.0, "Promoter_Pledge_%": 0.0,  "TTM_CFO": 2200.0, "Net_NPA_%": 0.0,  "CAR_%": 0.0,  "ROA_%": 0.0},
    "BANDHANBNK":{"ROE_%": 11.2, "Profit_Growth_%": 5.2,  "Sales_Growth_%": 6.8,  "Debt_to_Equity": 1.48, "PE_Ratio": 12.5, "Smart_Score": 4, "ROCE_3Yr_Avg": 7.0,  "CFO_PAT_3Yr_Avg": 0.65, "ROE_3Yr_Avg": 11.2, "Market_Cap_Cr": 35000.0,  "Promoter_Pledge_%": 0.0,  "TTM_CFO": 5500.0, "Net_NPA_%": 1.10, "CAR_%": 20.2, "ROA_%": 0.92},
    "DATAPATTNS":{"ROE_%": 18.6, "Profit_Growth_%": 29.5, "Sales_Growth_%": 24.1, "Debt_to_Equity": 0.08, "PE_Ratio": 65.2, "Smart_Score": 7, "ROCE_3Yr_Avg": 19.5, "CFO_PAT_3Yr_Avg": 0.81, "ROE_3Yr_Avg": 18.6, "Market_Cap_Cr": 12000.0,  "Promoter_Pledge_%": 0.0,  "TTM_CFO": 185.0,  "Net_NPA_%": 0.0,  "CAR_%": 0.0,  "ROA_%": 0.0},
    "HSCL":      {"ROE_%": 14.5, "Profit_Growth_%": 12.4, "Sales_Growth_%": 11.2, "Debt_to_Equity": 0.38, "PE_Ratio": 34.6, "Smart_Score": 6, "ROCE_3Yr_Avg": 15.2, "CFO_PAT_3Yr_Avg": 0.76, "ROE_3Yr_Avg": 14.5, "Market_Cap_Cr": 22000.0,  "Promoter_Pledge_%": 0.0,  "TTM_CFO": 280.0,  "Net_NPA_%": 0.0,  "CAR_%": 0.0,  "ROA_%": 0.0},
    "GMDCLTD":   {"ROE_%": 12.1, "Profit_Growth_%": -8.4, "Sales_Growth_%": 3.8,  "Debt_to_Equity": 0.12, "PE_Ratio": 9.2,  "Smart_Score": 5, "ROCE_3Yr_Avg": 11.5, "CFO_PAT_3Yr_Avg": 0.55, "ROE_3Yr_Avg": 12.1, "Market_Cap_Cr": 12000.0,  "Promoter_Pledge_%": 0.0,  "TTM_CFO": 280.0,  "Net_NPA_%": 0.0,  "CAR_%": 0.0,  "ROA_%": 0.0},
}

# Dynamic database built from scanning local Screener.in CSV files
DYNAMIC_FUNDAMENTAL_DB = {}
DYNAMIC_SECTOR_DB = {}
DYNAMIC_THEME_DB = {}

def scan_screener_csvs():
    """Scans the workspace directory for any screener*.csv files to load fundamental and sector metrics dynamically."""
    global DYNAMIC_FUNDAMENTAL_DB, DYNAMIC_SECTOR_DB, DYNAMIC_THEME_DB
    DYNAMIC_FUNDAMENTAL_DB = {}
    DYNAMIC_SECTOR_DB = {}
    DYNAMIC_THEME_DB = {}
    
    csv_files = [f for f in os.listdir(BASE_DIR) if f.lower().startswith("screener") and f.lower().endswith(".csv")]
    
    if not csv_files:
        return
        
    for file_name in csv_files:
        path = os.path.join(BASE_DIR, file_name)
        log_info(f"Ingesting fundamentals/sectors from Screener export: {file_name}...")
        try:
            df = pd.read_csv(path)
            
            # Map columns
            col_map = {}
            for col in df.columns:
                lower_col = col.lower()
                if "symbol" in lower_col or "ticker" in lower_col or "name" in lower_col:
                    col_map[col] = "Symbol"
                elif "roe" in lower_col:
                    col_map[col] = "ROE_%"
                elif "roce" in lower_col:
                    col_map[col] = "ROCE_%"
                elif "sales" in lower_col or "revenue" in lower_col:
                    col_map[col] = "Sales_Growth_%"
                elif "profit" in lower_col or "net profit" in lower_col:
                    col_map[col] = "Profit_Growth_%"
                elif "debt" in lower_col:
                    col_map[col] = "Debt_to_Equity"
                elif "pe" in lower_col:
                    col_map[col] = "PE_Ratio"
                elif "sector" in lower_col:
                    col_map[col] = "Sector"
                elif "industry" in lower_col or "theme" in lower_col:
                    col_map[col] = "Theme"
                    
            df = df.rename(columns=col_map)
            
            if "Symbol" not in df.columns:
                log_warning(f"Screener CSV {file_name} does not contain Symbol/Ticker columns. Skipping.")
                continue
                
            for idx, row in df.iterrows():
                symbol = str(row["Symbol"]).strip().upper()
                if symbol.endswith(".NS"):
                    symbol = symbol[:-3]
                    
                DYNAMIC_FUNDAMENTAL_DB[symbol] = {
                    "ROE_%": row.get("ROE_%", 15.0),
                    "Profit_Growth_%": row.get("Profit_Growth_%", 15.0),
                    "Sales_Growth_%": row.get("Sales_Growth_%", 12.0),
                    "Debt_to_Equity": row.get("Debt_to_Equity", 0.30),
                    "PE_Ratio": row.get("PE_Ratio", 25.0),
                    "Smart_Score": 6
                }
                
                if "Sector" in row and pd.notna(row["Sector"]):
                    DYNAMIC_SECTOR_DB[symbol] = str(row["Sector"]).strip()
                if "Theme" in row and pd.notna(row["Theme"]):
                    DYNAMIC_THEME_DB[symbol] = str(row["Theme"]).strip()
                    
            log_success(f"Loaded {len(df)} stock profiles dynamically from {file_name}.")
        except Exception as e:
            log_warning(f"Error parsing Screener CSV {file_name}: {e}")

# Scan on load
scan_screener_csvs()

# Cache file path
CACHE_FILE = os.path.join(CACHE_DIR if 'CACHE_DIR' in globals() else os.path.join(BASE_DIR, "cache"), "screener_fundamentals_cache.json")

def load_fundamentals_cache():
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def save_fundamentals_cache(cache):
    try:
        os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
        with open(CACHE_FILE, "w") as f:
            json.dump(cache, f, indent=4)
    except Exception:
        pass

def clean_val(val_str):
    # Remove commas, percentage signs, spaces
    val_str = re.sub(r'[^\d\.\-]', '', val_str)
    try:
        return float(val_str)
    except ValueError:
        return 0.0

def adjust_historical_eps(eps_series, sales_series):
    """Retrospectively split/bonus adjusts historical quarterly EPS.
    If EPS drops by >= 50% in a single quarter while quarterly sales remain stable (>= 80%),
    we assume a corporate action occurred and adjust all preceding historical EPS data points."""
    if not eps_series or not sales_series or len(eps_series) < 2:
        return eps_series
    
    n = len(eps_series)
    adjusted_eps = list(eps_series)
    for i in range(1, n):
        # Detect sudden drop in EPS (>= 50%) but normal sales (>= 80% of previous quarter)
        if adjusted_eps[i-1] > 0 and adjusted_eps[i] < 0.5 * adjusted_eps[i-1]:
            sales_ratio = sales_series[i] / sales_series[i-1] if (i < len(sales_series) and sales_series[i-1] > 0) else 1.0
            if sales_ratio >= 0.8:
                factor = adjusted_eps[i] / adjusted_eps[i-1]
                # Adjust all historical EPS before quarter i
                for j in range(i):
                    adjusted_eps[j] *= factor
    return adjusted_eps

# ── MARKET CAP CLASSIFICATION ────────────────────────────────────────────

MCAP_CACHE_FILE = os.path.join(CACHE_DIR if 'CACHE_DIR' in globals() else os.path.join(BASE_DIR, "cache"), "market_cap_cache.json")

def _load_json_cache(path):
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def _save_json_cache(path, data):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=4)
    except Exception:
        pass

def classify_cap_category(market_cap_cr):
    """Classifies a market cap value (in ₹ Crores) into a cap category."""
    if market_cap_cr < 1000:
        return "BELOW_MIN"
    elif market_cap_cr < 5000:
        return "SMALL_CAP"
    elif market_cap_cr < 20000:
        return "MID_CAP"
    elif market_cap_cr < 100000:
        return "LARGE_CAP"
    else:
        return "MEGA_CAP"

def fetch_market_cap(symbol):
    """Fetches market cap in ₹ Crores using caching and robust fallbacks."""
    symbol = symbol.strip().upper()
    if symbol.endswith(".NS"):
        symbol = symbol[:-3]
        
    cache = _load_json_cache(MCAP_CACHE_FILE)
    today_str = datetime.date.today().strftime("%Y-%m-%d")
    
    if symbol in cache:
        try:
            cached_date = datetime.datetime.strptime(cache[symbol].get("date"), "%Y-%m-%d").date()
            if (datetime.date.today() - cached_date).days < 30:
                mcap = cache[symbol]["Market_Cap_Cr"]
                if mcap not in [25000.0, 3000.0]:
                    return mcap, classify_cap_category(mcap)
        except Exception:
            pass
            
    mcap = None
    # Try Screener.in Fundamentals
    try:
        fundamentals = fetch_screener_fundamentals(symbol)
        if fundamentals and "Market_Cap_Cr" in fundamentals:
            mcap = fundamentals["Market_Cap_Cr"]
    except Exception:
        pass
        
    # Try dynamic or hardcoded DB
    if mcap is None or mcap in [25000.0, 3000.0]:
        db_entry = DYNAMIC_FUNDAMENTAL_DB.get(symbol, FUNDAMENTAL_DB.get(symbol))
        if db_entry and db_entry.get("Market_Cap_Cr") not in [None, 25000.0, 3000.0]:
            mcap = db_entry.get("Market_Cap_Cr")
            
    # Try yfinance fast_info if still unresolved/fallback
    if mcap is None or mcap in [25000.0, 3000.0]:
        try:
            ticker_symbol = symbol if symbol in ["MCX_GOLD", "MCX_SILVER", "NIFTY_50", "NIFTY_NEXT_50", "NIFTY_MIDCAP_150", "NIFTY_SMALLCAP_250"] else f"{symbol}.NS"
            ticker = yf.Ticker(ticker_symbol)
            if hasattr(ticker, "fast_info") and ticker.fast_info:
                raw_mcap = ticker.fast_info.get("marketCap") or ticker.fast_info.get("market_cap")
                if raw_mcap:
                    mcap = float(raw_mcap) / 10000000.0
        except:
            pass

    # Default fallback
    if mcap is None or mcap in [25000.0, 3000.0]:
        mcap = 1500.0
        
    cache[symbol] = {
        "date": today_str,
        "Market_Cap_Cr": mcap
    }
    _save_json_cache(MCAP_CACHE_FILE, cache)
    
    return mcap, classify_cap_category(mcap)

# ── DELIVERY DATA FETCHING ───────────────────────────────────────────────

DELIVERY_CACHE_FILE = os.path.join(CACHE_DIR if 'CACHE_DIR' in globals() else os.path.join(BASE_DIR, "cache"), "delivery_data_cache.json")

@lru_cache(maxsize=1024)
def fetch_delivery_data(symbol):
    """Fetches delivery percentage data from NSE API with fallbacks. Returns dict with delivery metrics."""
    symbol = symbol.strip().upper()
    if symbol.endswith(".NS"):
        symbol = symbol[:-3]

    cache = _load_json_cache(DELIVERY_CACHE_FILE)
    today_str = datetime.date.today().strftime("%Y-%m-%d")

    if symbol in cache:
        try:
            cached_date = datetime.datetime.strptime(cache[symbol].get("date"), "%Y-%m-%d").date()
            if (datetime.date.today() - cached_date).days < 5:
                return cache[symbol]["data"]
        except Exception:
            pass

    delivery_pct = 0.0
    delivery_pct_5d_avg = 0.0
    source = "Unknown"

    # 1. Try NSE API
    try:
        nse_url = f"https://www.nseindia.com/api/quote-equity?symbol={symbol}"
        nse_headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "application/json",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.nseindia.com/"
        }
        
        for attempt in range(3):
            try:
                session = requests.Session()
                # Hit the main page to get cookies
                session.get("https://www.nseindia.com/", headers=nse_headers, timeout=15)
                r = session.get(nse_url, headers=nse_headers, timeout=15)
                if r.status_code == 200:
                    data = r.json()
                    sec_info = data.get("securityWiseDP", {})
                    if sec_info:
                        delivery_pct = float(sec_info.get("delToTradedQty", 0.0))
                        source = "NSE API"
                        delivery_pct_5d_avg = delivery_pct
                    break # Success, exit retry loop
            except Exception as e:
                if attempt == 2:
                    log_warning(f"NSE delivery data fetch failed for {symbol}: {e}")
                else:
                    time.sleep(1) # wait before retrying
    except Exception as e:
        log_warning(f"Unexpected error in NSE fetch for {symbol}: {e}")

    # 2. Fallback: Estimate from yfinance volume patterns
    if delivery_pct <= 0:
        try:
            ticker = yf.Ticker(f"{symbol}.NS")
            hist = ticker.history(period="10d")
            if hist is not None and len(hist) >= 5:
                # Approximate delivery: on green days delivery tends to be higher
                # Use a heuristic: average of (close-open)/range as proxy
                closes = hist["Close"].values
                opens = hist["Open"].values
                highs = hist["High"].values
                lows = hist["Low"].values
                ranges = highs - lows
                ranges[ranges == 0] = 1e-5
                body_ratios = abs(closes - opens) / ranges

                # Stocks with larger body ratios tend to have higher delivery
                avg_body = float(body_ratios[-5:].mean())
                delivery_pct = min(40.0 + avg_body * 40.0, 85.0)  # Heuristic range 40-85%
                delivery_pct_5d_avg = delivery_pct
                source = "yfinance heuristic"
        except Exception:
            pass

    # 3. Ultimate fallback
    if delivery_pct <= 0:
        delivery_pct = 45.0  # Conservative average for Indian equities
        delivery_pct_5d_avg = 45.0
        source = "Default fallback"

    result = {
        "delivery_pct": delivery_pct,
        "delivery_pct_5d_avg": delivery_pct_5d_avg,
        "source": source
    }

    cache[symbol] = {"date": today_str, "data": result}
    _save_json_cache(DELIVERY_CACHE_FILE, cache)

    return result



def scrape_screener_in(symbol):
    symbol = symbol.strip().upper()
    if symbol.endswith(".NS"):
        symbol = symbol[:-3]
    url = f"https://www.screener.in/company/{symbol}/"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    try:
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code != 200:
            return None
        soup = BeautifulSoup(r.text, "html.parser")
        
        # 1. Parse Warehouse Ratios (latest values)
        latest_ratios = {}
        for li in soup.find_all("li", class_="flex flex-space-between"):
            name_span = li.find("span", class_="name")
            value_span = li.find("span", class_="number")
            if name_span and value_span:
                name = name_span.text.strip().lower()
                val = clean_val(value_span.text.strip())
                latest_ratios[name] = val
                
        def get_ratio_val(key_sub, default_val):
            for k, v in latest_ratios.items():
                if key_sub in k:
                    return v
            return default_val
                
        # Helper to parse tables
        def parse_table_section(section_id):
            sec = soup.find("section", id=section_id)
            if not sec:
                return None, {}
            table = sec.find("table")
            if not table:
                return None, {}
            headers = [th.text.strip() for th in table.find_all("th") if th.text.strip() != '']
            rows_data = {}
            rows = table.find("tbody").find_all("tr") if table.find("tbody") else table.find_all("tr")
            for r in rows:
                cols = [td.text.strip() for td in r.find_all("td")]
                if not cols:
                    continue
                label = cols[0].replace('+', '').strip().lower()
                vals = [clean_val(c) for c in cols[1:]]
                rows_data[label] = vals
            return headers, rows_data

        q_headers, q_data = parse_table_section("quarters")
        pl_headers, pl_data = parse_table_section("profit-loss")
        bs_headers, bs_data = parse_table_section("balance-sheet")
        sh_headers, sh_data = parse_table_section("shareholding")
        
        # QUALITY (3-5 Year average ROE & Debt-to-Equity)
        roe_list = []
        de_list = []
        roce_list = []
        cfo_pat_list = []
        
        pbt_row = pl_data.get("profit before tax")
        interest_row = pl_data.get("interest")
        net_profit_row = pl_data.get("net profit")
        
        # CFO row from cash-flow
        cf_headers, cf_data = parse_table_section("cash-flow")
        cfo_row = cf_data.get("cash from operating activity", cf_data.get("operating activity", cf_data.get("cash from operating activities")))
        
        if pl_headers and bs_headers:
            for idx_pl, year in enumerate(pl_headers):
                if year in bs_headers:
                    idx_bs = bs_headers.index(year)
                    cap_row = bs_data.get("equity capital", bs_data.get("share capital"))
                    res_row = bs_data.get("reserves")
                    borrowings_row = bs_data.get("borrowings")
                    
                    if cap_row and res_row:
                        try:
                            equity = cap_row[idx_bs] + res_row[idx_bs]
                            borrowings = borrowings_row[idx_bs] if borrowings_row else 0.0
                            capital_employed = equity + borrowings
                            
                            profit = net_profit_row[idx_pl] if net_profit_row else 0.0
                            
                            if equity > 0:
                                roe_list.append(profit / equity * 100.0)
                            if borrowings_row and equity > 0:
                                de_list.append(borrowings_row[idx_bs] / equity)
                                
                            pbt = pbt_row[idx_pl] if pbt_row else (profit * 1.3)
                            interest = interest_row[idx_pl] if interest_row else 0.0
                            ebit = pbt + interest
                            
                            if capital_employed > 0:
                                roce_list.append(ebit / capital_employed * 100.0)
                        except IndexError:
                            pass
                            
                # CFO / PAT matching year
                if cf_headers and cfo_row and net_profit_row and year in cf_headers:
                    idx_cf = cf_headers.index(year)
                    try:
                        cfo = cfo_row[idx_cf]
                        pat = net_profit_row[idx_pl]
                        if pat > 0:
                            cfo_pat_list.append(cfo / pat)
                    except IndexError:
                        pass
        
        avg_roe = sum(roe_list[-5:]) / len(roe_list[-5:]) if roe_list else get_ratio_val("roe", 15.0)
        avg_de = sum(de_list[-5:]) / len(de_list[-5:]) if de_list else get_ratio_val("debt to equity", get_ratio_val("debt/equity", 0.30))
        avg_roce = sum(roce_list[-3:]) / len(roce_list[-3:]) if roce_list else get_ratio_val("roce", 15.0)
        avg_cfo_pat = sum(cfo_pat_list[-3:]) / len(cfo_pat_list[-3:]) if cfo_pat_list else 0.80
        avg_roe_3yr = sum(roe_list[-3:]) / len(roe_list[-3:]) if roe_list else avg_roe
        
        # GROWTH (8 Quarters Average of YoY Sales, EPS, and OPM)
        sales_growth_8q = []
        eps_growth_8q = []
        opm_8q = []
        
        if q_data:
            sales_row = q_data.get("sales")
            op_row = q_data.get("operating profit")
            eps_row = q_data.get("eps in rs", q_data.get("eps", q_data.get("net profit")))
            if sales_row and eps_row:
                eps_row = adjust_historical_eps(eps_row, sales_row)
            
            n_quarters = len(sales_row) if sales_row else 0
            
            # We want to check up to 8 quarters (from the latest back to -8)
            # To compute YoY growth (Q_t vs Q_{t-4}), we need Q_{t-4} to exist.
            start_idx = max(4, n_quarters - 8)
            
            for i in range(start_idx, n_quarters):
                # YoY Sales Growth
                if sales_row and sales_row[i-4] > 0:
                    sales_growth_8q.append((sales_row[i] - sales_row[i-4]) / sales_row[i-4] * 100.0)
                # YoY EPS / Profit Growth
                if eps_row and eps_row[i-4] > 0:
                    eps_growth_8q.append((eps_row[i] - eps_row[i-4]) / eps_row[i-4] * 100.0)
                # OPM
                if op_row and sales_row and sales_row[i] > 0:
                    opm_8q.append(op_row[i] / sales_row[i] * 100.0)
                    
        avg_sales_growth = sum(sales_growth_8q) / len(sales_growth_8q) if sales_growth_8q else 15.0
        avg_eps_growth = sum(eps_growth_8q) / len(eps_growth_8q) if eps_growth_8q else 15.0
        avg_opm = sum(opm_8q) / len(opm_8q) if opm_8q else 12.0
        
        # PEAD: Surprise %
        earnings_surprise = 0.0
        if q_data:
            eps_row = q_data.get("eps in rs", q_data.get("eps", q_data.get("net profit")))
            if eps_row and len(eps_row) >= 5:
                actual = eps_row[-1]
                expected = sum(eps_row[-5:-1]) / 4.0
                if expected > 0:
                    earnings_surprise = (actual - expected) / expected * 100.0
                    
        # SMART MONEY: FII/DII Shareholding Change
        fii_dii_net_change = 0.0
        total_fii_dii_holding_pct = 0.0
        promoter_pledge_pct = 0.0
        if sh_data:
            fii_row = sh_data.get("fiis")
            dii_row = sh_data.get("diis")
            if fii_row and dii_row and len(fii_row) >= 2 and len(dii_row) >= 2:
                fii_dii_net_change = (fii_row[-1] - fii_row[-2]) + (dii_row[-1] - dii_row[-2])
                total_fii_dii_holding_pct = float(fii_row[-1]) + float(dii_row[-1])
            # Promoter pledge: look for "pledged" row in shareholding or ratio section
            pledge_row = sh_data.get("pledged", sh_data.get("promoter pledged", sh_data.get("pledge", None)))
            if pledge_row and pledge_row:
                try:
                    promoter_pledge_pct = float(pledge_row[-1])
                except (ValueError, TypeError):
                    promoter_pledge_pct = 0.0

        # If pledge not in shareholding table, try ratio section
        if promoter_pledge_pct == 0.0:
            promoter_pledge_pct = get_ratio_val("pledged", get_ratio_val("pledge", 0.0))

        # BFSI-SPECIFIC METRICS: Net NPA, Capital Adequacy Ratio (CAR), Return on Assets (ROA), PCR, CASA
        # Screener.in shows these in the ratio section for banking/NBFC companies.
        net_npa_pct = get_ratio_val("net npa", get_ratio_val("npa", 0.0))
        car_pct = get_ratio_val("capital adequacy", get_ratio_val("car", 16.0))  # default > 15% threshold
        roa_pct = get_ratio_val("return on asset", get_ratio_val("roa", 1.0))   # default > 0.80% threshold
        pcr_pct = get_ratio_val("provision coverage", get_ratio_val("pcr", 0.0)) # Add PCR
        casa_pct = get_ratio_val("casa", 0.0) # Add CASA

        # TTM Cash Flow from Operations (latest column from cash-flow table)
        ttm_cfo = 0.0
        if cf_headers and cfo_row and cfo_row:
            try:
                ttm_cfo = float(cfo_row[-1])  # Most recent period
            except (ValueError, TypeError, IndexError):
                ttm_cfo = 1.0  # Safe default (positive)
        elif avg_cfo_pat > 0 and net_profit_row:
            # Approximate TTM CFO from CFO/PAT ratio x latest profit if table missing
            try:
                ttm_cfo = avg_cfo_pat * float(net_profit_row[-1])
            except Exception:
                ttm_cfo = 1.0
                
        # Return structured data
        return {
            "ROE_%": avg_roe,
            "Debt_to_Equity": avg_de,
            "Sales_Growth_%": avg_sales_growth,
            "Profit_Growth_%": avg_eps_growth,
            "Earnings_Surprise_Pct": earnings_surprise,
            "FII_DII_Net_Change": fii_dii_net_change,
            "Total_FII_DII_Holding_%": total_fii_dii_holding_pct,
            "OPM_%": avg_opm,
            "PE_Ratio": get_ratio_val("stock p/e", get_ratio_val("p/e", 25.0)),
            "Smart_Score": 8.0,
            "Data_Source": "Screener.in Scraper",
            "ROCE_3Yr_Avg": avg_roce,
            "CFO_PAT_3Yr_Avg": avg_cfo_pat,
            "ROE_3Yr_Avg": avg_roe_3yr,
            "Market_Cap_Cr": get_ratio_val("market cap", None),
            "PCR_%": pcr_pct,
            "CASA_%": casa_pct,
            # ── Quality Gate fields ──────────────────────────────────────────────────────────────
            "Net_NPA_%": net_npa_pct,          # BFSI gate: must be < 1.75%
            "CAR_%": car_pct,                  # BFSI gate: must be > 15%
            "ROA_%": roa_pct,                  # BFSI gate: must be > 0.80%
            "Promoter_Pledge_%": promoter_pledge_pct,  # Both gates: BFSI < 15%, non-BFSI < 20%
            "TTM_CFO": ttm_cfo,                # Standard gate: must be > 0
        }
    except Exception as e:
        log_warning(f"Screener.in scraping failed for {symbol}: {e}")
        return None

def fetch_yfinance_fundamentals(symbol):
    symbol = symbol.strip().upper()
    ticker_symbol = symbol if symbol in ["MCX_GOLD", "MCX_SILVER", "NIFTY_50", "NIFTY_NEXT_50", "NIFTY_MIDCAP_150", "NIFTY_SMALLCAP_250"] else f"{symbol}.NS"
    try:
        ticker = yf.Ticker(ticker_symbol)
        fin = ticker.financials
        bs = ticker.balance_sheet
        q_fin = ticker.quarterly_financials
        info = ticker.info or {}
        
        # 1. QUALITY (Average ROE & Debt to Equity over last 3-4 years)
        roe_list = []
        de_list = []
        roce_val_list = []
        cfo_pat_val_list = []
        roe_3yr_list = []
        
        net_income = None
        for key in ['Net Income', 'NetIncome']:
            if fin is not None and key in fin.index:
                net_income = fin.loc[key]
                break
                
        equity = None
        for key in ['Stockholders Equity', 'StockholdersEquity', 'Total Stockholder Equity']:
            if bs is not None and key in bs.index:
                equity = bs.loc[key]
                break
                
        debt = None
        for key in ['Total Debt', 'TotalDebt']:
            if bs is not None and key in bs.index:
                debt = bs.loc[key]
                break
                
        ebit_series = None
        for key in ['EBIT', 'Operating Income', 'OperatingIncome']:
            if fin is not None and key in fin.index:
                ebit_series = fin.loc[key]
                break
                
        tot_assets = None
        for key in ['Total Assets', 'TotalAssets']:
            if bs is not None and key in bs.index:
                tot_assets = bs.loc[key]
                break
                
        curr_liab = None
        for key in ['Total Current Liabilities', 'TotalCurrentLiabilities']:
            if bs is not None and key in bs.index:
                curr_liab = bs.loc[key]
                break
                
        cfo_series = None
        cf = getattr(ticker, 'cashflow', None)
        if cf is None:
            try:
                cf = ticker.cashflow
            except Exception:
                pass
        for key in ['Operating Cash Flow', 'Cash Flow From Continuing Operating Activities', 'CashFlowFromContinuingOperatingActivities']:
            if cf is not None and key in cf.index:
                cfo_series = cf.loc[key]
                break
                
        if net_income is not None and equity is not None:
            common_idx = net_income.index.intersection(equity.index)
            for idx in common_idx:
                inc_val = net_income.loc[idx]
                eq_val = equity.loc[idx]
                if eq_val > 0:
                    roe_list.append(inc_val / eq_val * 100.0)
                    if debt is not None and idx in debt.index:
                        de_list.append(debt.loc[idx] / eq_val)
                        
                # ROCE
                if ebit_series is not None and idx in ebit_series.index and tot_assets is not None and idx in tot_assets.index:
                    ta = tot_assets.loc[idx]
                    cl = curr_liab.loc[idx] if (curr_liab is not None and idx in curr_liab.index) else 0.0
                    cap_emp = ta - cl
                    eb = ebit_series.loc[idx]
                    if cap_emp > 0:
                        roce_val_list.append(eb / cap_emp * 100.0)
                        
                # CFO / PAT
                if cfo_series is not None and idx in cfo_series.index:
                    cfo = cfo_series.loc[idx]
                    if inc_val > 0:
                        cfo_pat_val_list.append(cfo / inc_val)
                        
                if eq_val > 0:
                    roe_3yr_list.append(inc_val / eq_val * 100.0)
                        
        avg_roe = sum(roe_list) / len(roe_list) if roe_list else (info.get("returnOnEquity", 0.15) * 100.0)
        avg_de = sum(de_list) / len(de_list) if de_list else info.get("debtToEquity", 0.3)
        
        if avg_roe < 1.0:
            avg_roe = avg_roe * 100.0
        if avg_de > 5.0 and symbol not in ["J&KBANK", "KTKBANK", "BANDHANBNK"]:
            avg_de = avg_de / 100.0
            
        avg_roce = sum(roce_val_list[:3]) / len(roce_val_list[:3]) if roce_val_list else (info.get("returnOnAssets", 0.10) * 1.5 * 100.0)
        avg_cfo_pat = sum(cfo_pat_val_list[:3]) / len(cfo_pat_val_list[:3]) if cfo_pat_val_list else 0.80
        avg_roe_3yr = sum(roe_3yr_list[:3]) / len(roe_3yr_list[:3]) if roe_3yr_list else avg_roe
            
        # 2. GROWTH & PEAD (Last 8 Quarters Average of YoY Sales, EPS, and OPM)
        sales_growth_8q = []
        eps_growth_8q = []
        opm_8q = []
        earnings_surprise = 0.0
        
        if q_fin is not None:
            q_revenue = None
            for key in ['Total Revenue', 'TotalRevenue']:
                if key in q_fin.index:
                    q_revenue = q_fin.loc[key]
                    break
            q_net_income = None
            for key in ['Net Income', 'NetIncome']:
                if key in q_fin.index:
                    q_net_income = q_fin.loc[key]
                    break
            q_op_income = None
            for key in ['Operating Income', 'OperatingIncome']:
                if key in q_fin.index:
                    q_op_income = q_fin.loc[key]
                    break
                    
            if q_revenue is not None and len(q_revenue) >= 5:
                q_rev_ts = q_revenue.iloc[::-1]
                q_ni_ts = q_net_income.iloc[::-1] if q_net_income is not None else None
                if q_ni_ts is not None:
                    adjusted_ni = adjust_historical_eps(list(q_ni_ts.values), list(q_rev_ts.values))
                    q_ni_ts = pd.Series(adjusted_ni, index=q_ni_ts.index)
                q_op_ts = q_op_income.iloc[::-1] if q_op_income is not None else None
                
                n_q = len(q_rev_ts)
                start_i = max(4, n_q - 8)
                for i in range(start_i, n_q):
                    if q_rev_ts.iloc[i-4] > 0:
                        sales_growth_8q.append((q_rev_ts.iloc[i] - q_rev_ts.iloc[i-4]) / q_rev_ts.iloc[i-4] * 100.0)
                    if q_ni_ts is not None and q_ni_ts.iloc[i-4] > 0:
                        eps_growth_8q.append((q_ni_ts.iloc[i] - q_ni_ts.iloc[i-4]) / q_ni_ts.iloc[i-4] * 100.0)
                    if q_op_ts is not None and q_rev_ts.iloc[i] > 0:
                        opm_8q.append(q_op_ts.iloc[i] / q_rev_ts.iloc[i] * 100.0)
                        
                if q_ni_ts is not None and len(q_ni_ts) >= 5:
                    actual = q_ni_ts.iloc[-1]
                    expected = sum(q_ni_ts.iloc[-5:-1]) / 4.0
                    if expected > 0:
                        earnings_surprise = (actual - expected) / expected * 100.0
                        
        avg_sales_growth = sum(sales_growth_8q) / len(sales_growth_8q) if sales_growth_8q else (info.get("revenueGrowth", 0.15) * 100.0)
        avg_eps_growth = sum(eps_growth_8q) / len(eps_growth_8q) if eps_growth_8q else (info.get("earningsGrowth", 0.15) * 100.0)
        avg_opm = sum(opm_8q) / len(opm_8q) if opm_8q else (info.get("operatingMargins", 0.12) * 100.0)
        
        if avg_sales_growth < 1.0: avg_sales_growth *= 100.0
        if avg_eps_growth < 1.0: avg_eps_growth *= 100.0
        if avg_opm < 1.0: avg_opm *= 100.0
        
        fii_dii_net_change = 0.0   # FIXED: yfinance cannot provide real FII/DII data.
        # Do NOT default to 0.5 (that would silently pass Track 5 for every stock).
        
        mcap_val = None
        try:
            if hasattr(ticker, "fast_info") and ticker.fast_info:
                raw_mcap = ticker.fast_info.get("marketCap") or ticker.fast_info.get("market_cap")
                if raw_mcap:
                    mcap_val = float(raw_mcap) / 10000000.0
        except Exception as fe:
            log_warning(f"Fast info failed for {symbol}: {fe}")
            
        if mcap_val is None and info:
            raw_mcap = info.get("marketCap") or info.get("market_cap")
            if raw_mcap:
                mcap_val = float(raw_mcap) / 10000000.0
                
        return {
            "ROE_%": avg_roe,
            "Debt_to_Equity": avg_de,
            "Sales_Growth_%": avg_sales_growth,
            "Profit_Growth_%": avg_eps_growth,
            "Earnings_Surprise_Pct": earnings_surprise,
            "FII_DII_Net_Change": fii_dii_net_change,
            "OPM_%": avg_opm,
            "PE_Ratio": info.get("trailingPE", 25.0),
            "Smart_Score": 7.0,
            "Data_Source": "yfinance API Fallback",
            "ROCE_3Yr_Avg": avg_roce,
            "CFO_PAT_3Yr_Avg": avg_cfo_pat,
            "ROE_3Yr_Avg": avg_roe_3yr,
            "Market_Cap_Cr": mcap_val
        }
    except Exception as e:
        log_warning(f"yfinance fallback failed for {symbol}: {e}")
        return None

def scrape_screener_index_fundamentals(symbol):
    """Scrapes index-specific fundamental valuation metrics from Screener.in."""
    symbol = symbol.strip().upper()
    url = f"https://www.screener.in/company/{symbol}/"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    try:
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code != 200:
            return None
        soup = BeautifulSoup(r.text, "html.parser")
        
        latest_ratios = {}
        for li in soup.find_all("li", class_="flex flex-space-between"):
            name_span = li.find("span", class_="name")
            value_span = li.find("span", class_="number")
            if name_span and value_span:
                name = name_span.text.strip().lower()
                val = clean_val(value_span.text.strip())
                latest_ratios[name] = val
                
        def get_ratio_val(key_sub, default_val):
            for k, v in latest_ratios.items():
                if key_sub in k:
                    if v is None or v == 0.0:
                        return default_val
                    return v
            return default_val
            
        default_prices = {
            "NIFTY": 24000.0,
            "NIFTYJR": 75500.0,
            "CNXMIDCAP": 22600.0,
            "SMALLCA250": 17000.0,
            "NFMICRO250": 24000.0,
            "CNX500": 22600.0
        }
        default_pes = {
            "NIFTY": 20.5,
            "NIFTYJR": 26.0,
            "CNXMIDCAP": 30.0,
            "SMALLCA250": 28.1,
            "NFMICRO250": 27.1,
            "CNX500": 22.4
        }
        default_pbs = {
            "NIFTY": 3.2,
            "NIFTYJR": 4.5,
            "CNXMIDCAP": 4.0,
            "SMALLCA250": 3.82,
            "NFMICRO250": 3.43,
            "CNX500": 3.45
        }
        default_divs = {
            "NIFTY": 1.3,
            "NIFTYJR": 0.9,
            "CNXMIDCAP": 0.7,
            "SMALLCA250": 0.78,
            "NFMICRO250": 0.65,
            "CNX500": 1.08
        }
        default_mcaps = {
            "NIFTY": 180000000.0,
            "NIFTYJR": 25000000.0,
            "CNXMIDCAP": 35000000.0,
            "SMALLCA250": 10000000.0,
            "NFMICRO250": 2017417.0,
            "CNX500": 41141029.0
        }

        return {
            "Market_Cap_Cr": get_ratio_val("market cap", default_mcaps.get(symbol, 10000000.0)),
            "Current_Price": get_ratio_val("current price", default_prices.get(symbol, 17000.0)),
            "PE_Ratio": get_ratio_val("p/e", get_ratio_val("pe", default_pes.get(symbol, 28.0))),
            "Price_to_Book": get_ratio_val("price to book", default_pbs.get(symbol, 3.8)),
            "Dividend_Yield": get_ratio_val("dividend yield", default_divs.get(symbol, 0.8)),
            "CAGR_1Yr": get_ratio_val("cagr 1yr", get_ratio_val("cagr 1 yr", 0.0)),
            "CAGR_5Yr": get_ratio_val("cagr 5yr", get_ratio_val("cagr 5 yr", 11.0 if symbol in ["CNX500", "JUNIORBEES"] else 15.0)),
            "CAGR_10Yr": get_ratio_val("cagr 10yr", get_ratio_val("cagr 10 yr", 13.0 if symbol in ["CNX500", "JUNIORBEES"] else 15.5)),
            "ROE_%": 15.0,
            "Debt_to_Equity": 0.0,
            "Sales_Growth_%": 12.0,
            "Profit_Growth_%": 12.0,
            "Data_Source": "Screener.in Index Scraper"
        }
    except Exception as e:
        log_warning(f"Screener.in index scraping failed for {symbol}: {e}")
        return None

def fetch_screener_index_history(symbol, days=350):
    """Fetches historical daily close prices for indices from Screener.in chart API and models OHLC."""
    symbol = symbol.strip().upper()
    if symbol.endswith(".NS"):
        symbol = symbol[:-3]
        
    INDEX_CODELIST = {
        "NIFTY_500": ("1272615", "CNX500"),
        "CNX500": ("1272615", "CNX500"),
        "NIFTY_50": ("1272594", "NIFTY"),
        "CNX50": ("1272594", "NIFTY"),
        "NIFTY_SMALLCAP_250": ("1275142", "SMALLCA250"),
        "SMALLCA250": ("1275142", "SMALLCA250"),
        "NIFTY_NEXT_50": ("1272613", "NIFTYJR"),
        "JUNIORBEES": ("1272708", "JUNIORBEES"),
        "NIFTY_MIDCAP_150": ("1272674", "CNXMIDCAP"),
        "MID150": ("1285436", "MID150"),
        "NIFTY_MICROCAP_250": ("1284386", "NFMICRO250"),
        "NFMICRO250": ("1284386", "NFMICRO250")
    }
    
    if symbol not in INDEX_CODELIST:
        return None
        
    company_id, name = INDEX_CODELIST[symbol]
    
    valid_days = [30, 90, 180, 365, 1095, 1825, 3652, 10000]
    chart_days = 365
    for d in valid_days:
        if d >= days:
            chart_days = d
            break
            
    url = f"https://www.screener.in/api/company/{company_id}/chart/?q=Price-DMA50-DMA200&days={chart_days}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Referer": f"https://www.screener.in/company/{name}/"
    }
    try:
        log_info(f"Fetching chart price history for {symbol} ({name}) from Screener API (days={chart_days})...")
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code != 200:
            log_warning(f"Screener Chart API returned status {r.status_code} for {symbol}")
            return None
            
        data = r.json()
        price_values = None
        
        if "datasets" in data:
            for ds in data["datasets"]:
                if ds.get("metric") == "Price":
                    price_values = ds.get("values", [])
                    break
        if not price_values and "chart" in data:
            price_values = [[pt[0], pt[1]] for pt in data["chart"] if len(pt) >= 2]
            
        if not price_values:
            log_warning(f"No price values found in Screener Chart response for {symbol}")
            return None
            
        records = []
        import numpy as np
        
        np.random.seed(hash(symbol) % (2**32 - 1))
        
        for pt in price_values:
            date_str = pt[0]
            price_val = float(pt[1]) if pt[1] is not None else 0.0
            if price_val == 0.0:
                continue
                
            open_var = np.random.normal(0, 0.002)
            open_val = price_val * (1 + open_var)
            high_val = max(open_val, price_val) * (1 + abs(np.random.normal(0, 0.004)))
            low_val = min(open_val, price_val) * (1 - abs(np.random.normal(0, 0.004)))
            vol_val = float(np.random.randint(100000, 1000000))
            
            records.append({
                "Date": pd.to_datetime(date_str),
                "Open": open_val,
                "High": high_val,
                "Low": low_val,
                "Close": price_val,
                "Volume": vol_val
            })
            
        df = pd.DataFrame(records)
        df = df.sort_values("Date").reset_index(drop=True)
        df.set_index("Date", inplace=True)
        
        log_success(f"Successfully loaded {len(df)} historical price records for index {symbol} from Screener API.")
        return df
    except Exception as e:
        log_warning(f"Error fetching Screener index history for {symbol}: {e}")
        return None

@lru_cache(maxsize=1024)
def fetch_screener_fundamentals(symbol, use_jina_fallback=True):
    """Loads fundamental metrics for a stock/index from cache, Screener.in, or yfinance fallback."""
    symbol = symbol.strip().upper()
    if symbol.endswith(".NS"):
        symbol = symbol[:-3]
        
    INDEX_MAP = {
        "NIFTY_500": "CNX500",
        "NIFTY_SMALLCAP_250": "SMALLCA250",
        "CNX500": "CNX500",
        "SMALLCA250": "SMALLCA250",
        "NIFTY_NEXT_50": "NIFTYJR",
        "JUNIORBEES": "NIFTYJR",
        "NIFTY_MIDCAP_150": "CNXMIDCAP",
        "MID150": "CNXMIDCAP",
        "NIFTY_50": "NIFTY",
        "CNX50": "NIFTY",
        "NIFTY_MICROCAP_250": "NFMICRO250",
        "NFMICRO250": "NFMICRO250",
        "MICROCAP250": "NFMICRO250"
    }
    
    # Priority 1: Check Dynamic DB (fastest lookup, no cache required)
    if symbol in DYNAMIC_FUNDAMENTAL_DB:
        data = DYNAMIC_FUNDAMENTAL_DB[symbol].copy()
        data["Data_Source"] = "DYNAMIC_CSV"
        return data
        
    # Priority 2: Check Hardcoded FUNDAMENTAL_DB
    if symbol in FUNDAMENTAL_DB:
        data = FUNDAMENTAL_DB[symbol].copy()
        data["Data_Source"] = "FUNDAMENTAL_DB"
        return data

    # Priority 3: Check Cache
    cache = load_fundamentals_cache()
    today_str = datetime.date.today().strftime("%Y-%m-%d")
    if symbol in cache:
        cached_entry = cache[symbol]
        try:
            cached_date = datetime.datetime.strptime(cached_entry.get("date"), "%Y-%m-%d").date()
            if (datetime.date.today() - cached_date).days < 30:
                cached_data = cached_entry["data"]
                # Bypass if it has the bugged fallback market cap
                cached_mcap = cached_data.get("Market_Cap_Cr")
                if cached_mcap not in [25000.0, 3000.0]:
                    return cached_data
        except Exception:
            pass

    # Priority 4: Try Index-specific parsing if applicable
    if symbol in INDEX_MAP:
        log_info(f"Fetching index fundamentals for {symbol} ({INDEX_MAP[symbol]}) from Screener.in...")
        data = scrape_screener_index_fundamentals(INDEX_MAP[symbol])
        if data:
            cache[symbol] = {
                "date": today_str,
                "data": data
            }
            save_fundamentals_cache(cache)
            return data

    # Priority 5: Try Screener.in Scraper
    log_info(f"Fetching fundamentals for {symbol} from Screener.in...")
    data = scrape_screener_in(symbol)

    # Priority 6: Jina Reader Fallback (only if use_jina_fallback and not an index)
    if data is None and use_jina_fallback and symbol not in INDEX_MAP:
        log_info(f"Trying Jina Reader fallback for {symbol}...")
        try:
            import urllib.request, json, re
            from bs4 import BeautifulSoup
            
            jina_url = f"https://r.jina.ai/https://www.screener.in/company/{symbol}/"
            req = urllib.request.Request(jina_url, headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/125.0.0.0",
                "Accept": "text/plain,text/markdown",
            })
            resp = urllib.request.urlopen(req, timeout=25)
            content = resp.read().decode("utf-8")
            
            jina_data = {"Data_Source": "JINA_SCREENER"}
            
            # Parse PE ratio
            pe_match = re.search(r'P/E\s*[:\s]*([\d,.]+)', content, re.I)
            if pe_match:
                jina_data["PE_Ratio"] = float(pe_match.group(1).replace(",", ""))
            
            # Parse ROE
            roe_match = re.search(r'ROE\s*[:\s%]*([\d,.]+)', content, re.I)
            if roe_match:
                jina_data["ROE_%"] = float(roe_match.group(1).replace(",", ""))
            
            # Parse Sales Growth
            sales_match = re.search(r'Sales\s*(?:Growth)?\s*[:\s%]*([\d,.]+)', content, re.I)
            if sales_match:
                jina_data["Sales_Growth_%"] = float(sales_match.group(1).replace(",", ""))
            
            # Parse Profit Growth  
            profit_match = re.search(r'Profit\s*(?:Growth)?\s*[:\s%]*([\d,.]+)', content, re.I)
            if profit_match:
                jina_data["Profit_Growth_%"] = float(profit_match.group(1).replace(",", ""))
            
            # Default values for missing fields
            jina_data.setdefault("ROE_%", 15.0)
            jina_data.setdefault("Profit_Growth_%", 15.0)
            jina_data.setdefault("Sales_Growth_%", 12.0)
            jina_data.setdefault("Debt_to_Equity", 0.5)
            jina_data.setdefault("PE_Ratio", 25.0)
            jina_data.setdefault("ROCE_3Yr_Avg", 15.0)
            jina_data.setdefault("CFO_PAT_3Yr_Avg", 0.75)
            jina_data.setdefault("ROE_3Yr_Avg", jina_data.get("ROE_%", 15.0))
            jina_data.setdefault("Smart_Score", 6)
            jina_data.setdefault("Market_Cap_Cr", 5000.0)
            jina_data["Data_Flags"] = "JINA_FALLBACK"
            
            log_info(f"Fetched fundamentals for {symbol} via Jina Reader")
            data = jina_data
        except Exception as e:
            log_warning(f"Jina Reader fallback failed for {symbol}: {e}")

    # Priority 7: Try yfinance Fallback
    if data is None:
        log_info(f"Falling back to yfinance for {symbol}...")
        data = fetch_yfinance_fundamentals(symbol)
        
    # Priority 8: Try hardcoded database fallback
    if data is None:
        log_warning(f"Both Screener.in and yfinance failed for {symbol}. Using local database fallback.")
        db_entry = DYNAMIC_FUNDAMENTAL_DB.get(symbol, FUNDAMENTAL_DB.get(symbol))
        if db_entry:
            data = {
                "ROE_%": db_entry.get("ROE_%", 15.0),
                "Debt_to_Equity": db_entry.get("Debt_to_Equity", 0.30),
                "Sales_Growth_%": db_entry.get("Sales_Growth_%", 15.0),
                "Profit_Growth_%": db_entry.get("Profit_Growth_%", 15.0),
                "Earnings_Surprise_Pct": 0.0,
                "FII_DII_Net_Change": 0.0,
                "OPM_%": 15.0,
                "PE_Ratio": db_entry.get("PE_Ratio", 25.0),
                "Smart_Score": db_entry.get("Smart_Score", 6),
                "Data_Source": "Hardcoded fallback",
                "Data_Flags": "SURPRISE_UNAVAILABLE|FII_UNAVAILABLE",
                "ROCE_3Yr_Avg": db_entry.get("ROCE_3Yr_Avg", 15.0),
                "CFO_PAT_3Yr_Avg": db_entry.get("CFO_PAT_3Yr_Avg", 0.70),
                "ROE_3Yr_Avg": db_entry.get("ROE_3Yr_Avg", 15.0),
                "Market_Cap_Cr": db_entry.get("Market_Cap_Cr", 1500.0)
            }
        else:
            # Ultimate default — prevents system crashes but marks data as unavailable
            data = {
                "ROE_%": 15.0,
                "Debt_to_Equity": 0.30,
                "Sales_Growth_%": 15.0,
                "Profit_Growth_%": 15.0,
                "Earnings_Surprise_Pct": 0.0,
                "FII_DII_Net_Change": 0.0,
                "OPM_%": 15.0,
                "PE_Ratio": 25.0,
                "Smart_Score": 6,
                "Data_Source": "Default fallback",
                "Data_Flags": "ALL_DATA_UNAVAILABLE",
                "ROCE_3Yr_Avg": 15.0,
                "CFO_PAT_3Yr_Avg": 0.70,
                "ROE_3Yr_Avg": 15.0,
                "Market_Cap_Cr": 1500.0
            }
            
    # Ensure Market_Cap_Cr is valid and not a bugged fallback value
    if data:
        mcap_resolved = data.get("Market_Cap_Cr")
        if mcap_resolved is None or mcap_resolved in [25000.0, 3000.0]:
            # Try database lookup
            db_entry = DYNAMIC_FUNDAMENTAL_DB.get(symbol, FUNDAMENTAL_DB.get(symbol))
            if db_entry and db_entry.get("Market_Cap_Cr") not in [None, 25000.0, 3000.0]:
                mcap_resolved = db_entry["Market_Cap_Cr"]
            else:
                # Check if yfinance fast_info can provide it if data didn't have it
                try:
                    import yfinance as yf
                    ticker_symbol = symbol if symbol in ["MCX_GOLD", "MCX_SILVER", "NIFTY_50", "NIFTY_NEXT_50", "NIFTY_MIDCAP_150", "NIFTY_SMALLCAP_250"] else f"{symbol}.NS"
                    ticker = yf.Ticker(ticker_symbol)
                    if hasattr(ticker, "fast_info") and ticker.fast_info:
                        raw_mcap = ticker.fast_info.get("marketCap") or ticker.fast_info.get("market_cap")
                        if raw_mcap:
                            mcap_resolved = float(raw_mcap) / 10000000.0
                except:
                    pass
            
            # Final safe fallback: 1500.0 (Small Cap risk category)
            if mcap_resolved is None or mcap_resolved in [25000.0, 3000.0]:
                mcap_resolved = 1500.0
            
            data["Market_Cap_Cr"] = mcap_resolved

    # Save to Cache
    cache[symbol] = {
        "date": today_str,
        "data": data
    }
    save_fundamentals_cache(cache)
    
    return data

def get_dynamic_sector(symbol):
    """Fetches sector for a symbol with auto-discovery for unknown stocks.
    Checks: Dynamic DB → config SECTORS → yfinance live → sector pattern match.
    Never returns 'Diversified' without a real attempt to classify."""
    symbol = symbol.strip().upper()
    if symbol.endswith(".NS"):
        symbol = symbol[:-3]
    
    # 1. Dynamic DB (from CSV exports)
    if symbol in DYNAMIC_SECTOR_DB:
        return DYNAMIC_SECTOR_DB[symbol]
    
    # 2. Config SECTORS mapping (29 benchmark + extended coverage)
    try:
        from config import SECTORS
        if symbol in SECTORS:
            return SECTORS[symbol]
    except ImportError:
        pass
    
    # 3. Try yfinance for live sector data
    try:
        import yfinance as yf
        ticker = yf.Ticker(f"{symbol}.NS")
        info = ticker.info or {}
        sector = info.get("sector", "")
        industry = info.get("industry", "")
        if sector:
            # Normalize sector name to match existing classification
            sector_map = {
                "Technology": "Technology",
                "Healthcare": "Healthcare",
                "Financial Services": "Financial Services",
                "Industrials": "Industrials",
                "Consumer Cyclical": "Consumer Cyclical",
                "Consumer Defensive": "Consumer Defensive",
                "Basic Materials": "Basic Materials",
                "Energy": "Basic Materials",
                "Utilities": "Utilities",
                "Real Estate": "Real Estate",
                "Communication Services": "Technology",
            }
            mapped = sector_map.get(sector, sector)
            # Cache the result
            DYNAMIC_SECTOR_DB[symbol] = mapped
            return mapped
    except Exception:
        pass
    
    # 4. Sector pattern match from symbol name patterns
    # Common Indian stocks have recognizable naming patterns
    sector_patterns = [
        ("BANK", "Financial Services"),
        ("BNK", "Financial Services"),
        ("FIN", "Financial Services"),
        ("INS", "Financial Services"),
        ("MF", "Financial Services"),
        ("HOUSING", "Financial Services"),
        ("NBFC", "Financial Services"),
        ("MUTHOOT", "Financial Services"),
        ("CHOLA", "Financial Services"),
        ("LIC", "Financial Services"),
        ("SBIN", "Financial Services"),
        ("AXIS", "Financial Services"),
        ("HDFC", "Financial Services"),
        ("ICICI", "Financial Services"),
        ("KOTAK", "Financial Services"),
        ("INDUSIND", "Financial Services"),
        ("FED", "Financial Services"),
        ("BANDHAN", "Financial Services"),
        ("YESBANK", "Financial Services"),
        ("RBL", "Financial Services"),
        ("SOUTH", "Financial Services"),
        ("KARUR", "Financial Services"),
        ("CITY", "Financial Services"),
        ("DCB", "Financial Services"),
        ("TECH", "Technology"),
        ("SOFT", "Technology"),
        ("INFY", "Technology"),
        ("TCS", "Technology"),
        ("WIPRO", "Technology"),
        ("HCL", "Technology"),
        ("LTTS", "Technology"),
        ("PERSIST", "Technology"),
        ("CYIENT", "Technology"),
        ("MINDTREE", "Technology"),
        ("COFORGE", "Technology"),
        ("MPHASIS", "Technology"),
        ("HEXAWARE", "Technology"),
        ("ZENSAR", "Technology"),
        ("BSOFT", "Technology"),
        ("TATAELXSI", "Technology"),
        ("LTI", "Technology"),
        ("TECHM", "Technology"),
        ("INFIBEAM", "Technology"),
        ("NEWGEN", "Technology"),
        ("TANLA", "Technology"),
        ("INTELLECT", "Technology"),
        ("KPITTECH", "Technology"),
        ("STEEL", "Basic Materials"),
        ("METAL", "Basic Materials"),
        ("ALUM", "Basic Materials"),
        ("COPPER", "Basic Materials"),
        ("MINING", "Basic Materials"),
        ("TATASTEEL", "Basic Materials"),
        ("JSW", "Basic Materials"),
        ("HINDALCO", "Basic Materials"),
        ("JINDAL", "Basic Materials"),
        ("NATIONALUM", "Basic Materials"),
        ("MOIL", "Basic Materials"),
        ("GRAPHITE", "Basic Materials"),
        ("CHEM", "Basic Materials"),
        ("FERT", "Basic Materials"),
        ("UPL", "Basic Materials"),
        ("PIIND", "Basic Materials"),
        ("SRF", "Basic Materials"),
        ("NAVIN", "Basic Materials"),
        ("TATACHEM", "Basic Materials"),
        ("AARTI", "Basic Materials"),
        ("DEEPAK", "Basic Materials"),
        ("GSK", "Healthcare"),
        ("SUNPHARMA", "Healthcare"),
        ("DRREDDY", "Healthcare"),
        ("CIPLA", "Healthcare"),
        ("DIVIS", "Healthcare"),
        ("BIOCON", "Healthcare"),
        ("LUPIN", "Healthcare"),
        ("APOLLOHOSP", "Healthcare"),
        ("FORTIS", "Healthcare"),
        ("MAXHEALTH", "Healthcare"),
        ("MEDANTA", "Healthcare"),
        ("HEAL", "Healthcare"),
        ("HOSP", "Healthcare"),
        ("NURSE", "Healthcare"),
        ("DIAG", "Healthcare"),
        ("HEALTH", "Healthcare"),
        ("PHARMA", "Healthcare"),
        ("MEDI", "Healthcare"),
        ("GLAXO", "Healthcare"),
        ("ABBOT", "Healthcare"),
        ("PFIZER", "Healthcare"),
        ("NOVARTIS", "Healthcare"),
        ("SANOFI", "Healthcare"),
        ("Laurus", "Healthcare"),
        ("GRANULES", "Healthcare"),
        ("AUROPHARMA", "Healthcare"),
        ("ALEMBIC", "Healthcare"),
        ("NATCO", "Healthcare"),
        ("JUBL", "Consumer Cyclical"),
        ("TITAN", "Consumer Cyclical"),
        ("TRENT", "Consumer Cyclical"),
        ("TRENT", "Consumer Cyclical"),
        ("ZOMATO", "Technology"),
        ("SWIGGY", "Technology"),
        ("NYKAA", "Consumer Cyclical"),
        ("MCDOWELL", "Consumer Defensive"),
        ("DABUR", "Consumer Defensive"),
        ("HINDUNILVR", "Consumer Defensive"),
        ("NESTLE", "Consumer Defensive"),
        ("BRITANNIA", "Consumer Defensive"),
        ("MARICO", "Consumer Defensive"),
        ("COLPAL", "Consumer Defensive"),
        ("GODREJ", "Consumer Defensive"),
        ("PGHH", "Consumer Defensive"),
        ("EMAMI", "Consumer Defensive"),
        ("CABLE", "Industrials"),
        ("ELECTR", "Industrials"),
        ("ENGINEER", "Industrials"),
        ("LARSEN", "Industrials"),
        ("LTIM", "Technology"),
        ("SIEMENS", "Industrials"),
        ("BHEL", "Industrials"),
        ("ABB", "Industrials"),
        ("CGPOWER", "Industrials"),
        ("POLYCAB", "Industrials"),
        ("HAVELLS", "Industrials"),
        ("VOLTAS", "Industrials"),
        ("BLUESTAR", "Industrials"),
        ("KIRLOS", "Industrials"),
        ("CUMMINS", "Industrials"),
        ("THERMAX", "Industrials"),
        ("DEFENCE", "Industrials"),
        ("AEROSPACE", "Industrials"),
        ("HAL", "Industrials"),
        ("BEL", "Industrials"),
        ("BDL", "Industrials"),
        ("SOLARA", "Industrials"),
        ("MISHRA", "Industrials"),
        ("COCHIN", "Industrials"),
        ("POWER", "Utilities"),
        ("ENERGY", "Utilities"),
        ("NTPC", "Utilities"),
        ("ADANI", "Utilities"),
        ("TATA POWER", "Utilities"),
        ("TORNTPOWER", "Utilities"),
        ("NHPC", "Utilities"),
        ("SJVN", "Utilities"),
        ("CESC", "Utilities"),
        ("AUTO", "Consumer Cyclical"),
        ("MOTOR", "Consumer Cyclical"),
        ("MARUTI", "Consumer Cyclical"),
        ("M&M", "Consumer Cyclical"),
        ("TATAMOTORS", "Consumer Cyclical"),
        ("BAJAJ-AUTO", "Consumer Cyclical"),
        ("HERO", "Consumer Cyclical"),
        ("EICHER", "Consumer Cyclical"),
        ("ASHOK", "Consumer Cyclical"),
        ("BALKRIS", "Consumer Cyclical"),
        ("MRF", "Consumer Cyclical"),
        ("APOLLOTYRE", "Consumer Cyclical"),
        ("JEWEL", "Consumer Cyclical"),
        ("TITAN", "Consumer Cyclical"),
        ("THANGAMAYL", "Consumer Cyclical"),
        ("KALYANK", "Consumer Cyclical"),
        ("RETAIL", "Consumer Cyclical"),
        ("AVIATION", "Consumer Cyclical"),
        ("INTERGLOBE", "Consumer Cyclical"),
        ("INDIGO", "Consumer Cyclical"),
    ]
    
    for keyword, sector_name in sector_patterns:
        if keyword in symbol:
            DYNAMIC_SECTOR_DB[symbol] = sector_name
            return sector_name
    
    # 5. Ultimate fallback — try to classify by yfinance industry keywords
    try:
        ticker = yf.Ticker(f"{symbol}.NS")
        info = ticker.info or {}
        industry = (info.get("industry") or "").lower()
        sector2 = (info.get("sector") or "").lower()
        
        if "bank" in industry or "bank" in sector2:
            return "Financial Services"
        if "pharma" in industry or "health" in industry:
            return "Healthcare"
        if "tech" in industry or "software" in industry:
            return "Technology"
        if "steel" in industry or "metal" in industry or "mining" in industry:
            return "Basic Materials"
        if "chemical" in industry:
            return "Basic Materials"
        if "auto" in industry or "tire" in industry:
            return "Consumer Cyclical"
        if "food" in industry or "beverage" in industry:
            return "Consumer Defensive"
        if "power" in industry or "utility" in industry:
            return "Utilities"
        if "industrial" in industry or "machinery" in industry:
            return "Industrials"
        if "retail" in industry:
            return "Consumer Cyclical"
        if "real" in industry:
            return "Financial Services"
    except:
        pass
    
    return "Diversified"


def get_dynamic_theme(symbol):
    """Fetches theme for a symbol with auto-discovery.
    Checks: Dynamic DB → config THEMES → sector-to-theme mapping → industry pattern."""
    symbol = symbol.strip().upper()
    if symbol.endswith(".NS"):
        symbol = symbol[:-3]
    
    # 1. Dynamic DB (from CSV exports)
    if symbol in DYNAMIC_THEME_DB:
        return DYNAMIC_THEME_DB[symbol]
    
    # 2. Config THEMES mapping
    try:
        from config import THEMES
        if symbol in THEMES:
            return THEMES[symbol]
    except ImportError:
        pass
    
    # 3. Map sector to theme
    sector = get_dynamic_sector(symbol)
    
    if sector == "Financial Services":
        try:
            import yfinance as yf
            info = yf.Ticker(f"{symbol}.NS").info or {}
            industry = (info.get("industry") or "").lower()
            if "capital market" in industry or "asset management" in industry or "financial data" in industry or "exchange" in industry:
                DYNAMIC_THEME_DB[symbol] = "Capital Markets"
                return "Capital Markets"
        except:
            pass

    sector_to_theme = {
        "Financial Services": "Financials",
        "Technology": "Electronics Manufacturing",
        "Healthcare": "Pharma & Lifesciences",
        "Basic Materials": "Metals & Mining",
        "Consumer Cyclical": "Consumer & FMCG",
        "Consumer Defensive": "Consumer & FMCG",
        "Utilities": "Power & Electrical Infrastructure",
        "Industrials": "Defense & Capital Goods",
    }
    theme = sector_to_theme.get(sector)
    if theme:
        DYNAMIC_THEME_DB[symbol] = theme
        return theme
    
    # 4. Try yfinance for more precise theme
    try:
        import yfinance as yf
        ticker = yf.Ticker(f"{symbol}.NS")
        info = ticker.info or {}
        industry = (info.get("industry") or "").lower()
        
        industry_theme_map = {
            "defense": "Defense & Capital Goods",
            "aerospace": "Defense & Capital Goods",
            "capital markets": "Financials",
            "bank": "Financials" if sector == "Financial Services" else None,
            "regional": "Regional Banking",
            "jewellery": "Jewellery & Retail",
            "retail": "Jewellery & Retail",
            "pharmaceutical": "Pharma & Lifesciences",
            "biotechnology": "Pharma & Lifesciences",
            "electrical": "Power & Electrical Infrastructure",
            "power": "Power & Electrical Infrastructure",
            "metal": "Metals & Mining",
            "steel": "Metals & Mining",
            "mining": "Metals & Mining",
            "chemical": "Metals & Mining",
            "aluminum": "Metals & Mining",
            "software": "Electronics Manufacturing",
            "semiconductor": "Electronics Manufacturing",
            "electronics": "Electronics Manufacturing",
            "auto": "Consumer & FMCG",
            "household": "Consumer & FMCG",
            "personal": "Consumer & FMCG",
            "food": "Consumer & FMCG",
        }
        
        for keyword, mapped_theme in industry_theme_map.items():
            if mapped_theme and keyword in industry:
                DYNAMIC_THEME_DB[symbol] = mapped_theme
                return mapped_theme
    except:
        pass
    
    return "General Momentum"

def parse_screener_csv_file(csv_path):
    """Helper to parse a downloaded Chartink screener CSV file and extract its stock rows."""
    try:
        df = pd.read_csv(csv_path)
        
        # Normalize column names case-insensitively
        symbol_col = None
        close_col = None
        vol_col = None
        name_col = None
        
        for col in df.columns:
            lcol = col.lower()
            if "symbol" in lcol or "ticker" in lcol or "nsecode" in lcol:
                symbol_col = col
            elif "close" in lcol or "price" in lcol:
                close_col = col
            elif "volume" in lcol or "vol" in lcol:
                vol_col = col
            elif "name" in lcol:
                name_col = col
                
        if not symbol_col:
            # Fallback to column index
            symbol_col = df.columns[2] if len(df.columns) > 2 else df.columns[0]
            
        stocks_list = []
        for idx, row in df.iterrows():
            sym = str(row[symbol_col]).strip().upper()
            if not sym or sym == "NAN" or sym == "NONE":
                continue
            if sym.endswith(".NS"):
                sym = sym[:-3]
                
            # Clean commas and parse numbers (Indian formats)
            price = 0.0
            if close_col and close_col in row and pd.notna(row[close_col]):
                p_str = str(row[close_col]).replace(",", "").strip()
                try: price = float(p_str)
                except ValueError: pass
                
            vol = 0.0
            if vol_col and vol_col in row and pd.notna(row[vol_col]):
                v_str = str(row[vol_col]).replace(",", "").strip()
                try: vol = float(v_str)
                except ValueError: pass
                
            name = str(row[name_col]).strip() if (name_col and name_col in row and pd.notna(row[name_col])) else sym
            
            stocks_list.append({
                "Symbol": sym,
                "Close": price,
                "Volume": vol,
                "Name": name
            })
        return stocks_list
    except Exception as e:
        log_warning(f"Error parsing CSV file {csv_path}: {e}")
        return []

def use_mcp_chartink_universe(date_str, export_dir, urls):
    """Fetches the stock universe visibly using Chrome/Selenium."""
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    
    chrome_options = Options()
    prefs = {
        "download.default_directory": export_dir,
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "safebrowsing.enabled": True
    }
    chrome_options.add_experimental_option("prefs", prefs)
    
    # Hide the "Chrome is being controlled by automated test software" infobar/banner and optimize speed
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option('useAutomationExtension', False)
    chrome_options.add_argument("--disable-infobars")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--disable-extensions")
    chrome_options.page_load_strategy = 'eager' # Don't wait for trackers and ads to load, only the main DOM
    
    log_info("Initializing Chrome browser visibly...")
    driver = webdriver.Chrome(options=chrome_options)
    driver.maximize_window()
    
    universe = {}
    
    try:
        for url in urls:
            screener_name = url.split("/")[-1]
            log_info(f"Navigating visibly to: {url}...")
            driver.get(url)
            
            before_files = set(os.listdir(export_dir))
            
            # Wait for CSV button
            wait = WebDriverWait(driver, 10)
            csv_btn = wait.until(EC.element_to_be_clickable((
                By.XPATH, 
                "//*[contains(@class, 'buttons-csv')] | //button[contains(text(), 'CSV')] | //*[text()='CSV']"
            )))
            
            log_info(f"Clicking CSV download button on browser for: {screener_name}...")
            driver.execute_script("arguments[0].click();", csv_btn)
            
            # Wait for the download (frequent polling at 0.1s to be quick)
            downloaded_file = None
            start_time = time.time()
            while time.time() - start_time < 8:
                after_files = set(os.listdir(export_dir))
                new_files = after_files - before_files
                csv_files = [f for f in new_files if f.lower().endswith('.csv') and not f.lower().endswith('.crdownload') and not f.lower().endswith('.tmp')]
                if csv_files:
                    downloaded_file = os.path.join(export_dir, csv_files[0])
                    break
                time.sleep(0.1)
                
            if downloaded_file:
                target_path = os.path.join(export_dir, f"{screener_name}.csv")
                # Retry loop to handle Windows file locking from Chrome downloads
                rename_success = False
                for attempt in range(8):
                    try:
                        if os.path.exists(target_path):
                            os.remove(target_path)
                        os.rename(downloaded_file, target_path)
                        rename_success = True
                        break
                    except (PermissionError, OSError):
                        time.sleep(0.1)
                
                if rename_success:
                    log_success(f"CSV downloaded & saved to: {target_path}")
                else:
                    log_warning(f"Could not rename {downloaded_file} due to file lock. Trying direct parse...")
                    target_path = downloaded_file
                    
                # Parse stocks
                stocks = parse_screener_csv_file(target_path)
                for stock in stocks:
                    sym = stock["Symbol"]
                    if sym not in universe:
                        universe[sym] = {
                            "Symbol": sym,
                            "Close": stock["Close"],
                            "Volume": stock["Volume"],
                            "Name": stock["Name"],
                            "Screeners": [screener_name]
                        }
                    else:
                        if screener_name not in universe[sym]["Screeners"]:
                            universe[sym]["Screeners"].append(screener_name)
            else:
                log_warning(f"Download timed out on browser for screener: {screener_name}")
            time.sleep(0.1)
            
    finally:
        log_info("Closing Chrome browser...")
        driver.quit()
        
    return universe

def fetch_chartink_universe_rest(date_str, export_dir, urls):
    """Fallback method using requests REST API to scrape screeners."""
    process_url = "https://chartink.com/screener/process"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    universe = {}
    
    for url in urls:
        screener_name = url.split("/")[-1]
        max_retries = 3
        for attempt in range(max_retries):
            try:
                with requests.Session() as s:
                    r = s.get(url, headers=headers, timeout=30)
                    if r.status_code != 200:
                        break # Skip to next url
                    
                    soup = BeautifulSoup(r.text, "html.parser")
                    csrf_meta = soup.find("meta", attrs={"name": "csrf-token"})
                    scanner_meta = soup.find("scanner")
                    if not csrf_meta or not scanner_meta:
                        break # Skip to next url
                        
                    csrf_token = csrf_meta["content"]
                    scan_json = json.loads(scanner_meta.get(":scan-json"))
                    atlas_query = scan_json["atlas_query"]
                    
                    post_headers = {
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, Gecko) Chrome/120.0.0.0 Safari/537.36",
                        "Referer": url,
                        "x-csrf-token": csrf_token,
                        "Content-Type": "application/x-www-form-urlencoded"
                    }
                    
                    payload = {"scan_clause": atlas_query}
                    post_res = s.post(process_url, headers=post_headers, data=payload, timeout=30)
                    
                    if post_res.status_code == 200:
                        stocks = post_res.json().get("data", [])
                        if stocks:
                            df_export = pd.DataFrame(stocks)
                            csv_path = os.path.join(export_dir, f"{screener_name}.csv")
                            df_export.to_csv(csv_path, index=False)
                            
                            for stock in stocks:
                                sym = str(stock.get("nsecode")).strip().upper()
                                if not sym or sym == "NONE":
                                    continue
                                if sym.endswith(".NS"):
                                    sym = sym[:-3]
                                price = float(stock.get("close", 0.0))
                                vol = float(stock.get("volume", 0.0))
                                name = str(stock.get("name", sym))
                                
                                if sym not in universe:
                                    universe[sym] = {
                                        "Symbol": sym,
                                        "Close": price,
                                        "Volume": vol,
                                        "Name": name,
                                        "Screeners": [screener_name]
                                    }
                                else:
                                    if screener_name not in universe[sym]["Screeners"]:
                                        universe[sym]["Screeners"].append(screener_name)
                break # Success, exit retry loop
            except Exception as e:
                if attempt == max_retries - 1:
                    log_warning(f"REST fallback error for {screener_name}: {e}")
                else:
                    time.sleep(2)
        time.sleep(0.5)
        
    return universe

def fetch_chartink_universe(date_str=None):
    """
    Fetches the active stock universe from the 11 Chartink screeners.
    Uses fast REST API scraper to query the Chartink screeners directly.
    """
    try:
        from config import CHARTINK_URLS
    except ImportError:
        CHARTINK_URLS = [
            "https://chartink.com/screener/upside-mean-reversion-base-building-breakouts",
            "https://chartink.com/screener/cum-and-handle-pattern",
            "https://chartink.com/screener/double-bottom-pattren",
            "https://chartink.com/screener/rounding-up-bottom-and-bear-flag",
            "https://chartink.com/screener/early-recovery-3",
            "https://chartink.com/screener/strong-long-term-trend-2",
            "https://chartink.com/screener/52-w-h-in-last-3-days",
            "https://chartink.com/screener/200dma-scan-6",
            "https://chartink.com/screener/52w-h-ath",
            "https://chartink.com/screener/the-techno-funda-leader-screener",
            "https://chartink.com/screener/trend-alfa"
        ]

    if not date_str:
        date_str = datetime.date.today().strftime("%Y-%m-%d")

    export_dir = os.path.abspath(os.path.join(BASE_DIR, "output", date_str, "chartink_exports"))
    os.makedirs(export_dir, exist_ok=True)

    # Check if we already have the exports for this date in the dedicated exports folder
    all_files_exist = True
    for url in CHARTINK_URLS:
        screener_name = url.split("/")[-1]
        csv_path = os.path.join(export_dir, f"{screener_name}.csv")
        if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
            all_files_exist = False
            break

    if all_files_exist:
        log_info(f"Detected existing Chartink exports in dedicated folder for {date_str}. Parsing local files...")
        universe = {}
        for url in CHARTINK_URLS:
            screener_name = url.split("/")[-1]
            csv_path = os.path.join(export_dir, f"{screener_name}.csv")
            stocks = parse_screener_csv_file(csv_path)
            for stock in stocks:
                sym = stock["Symbol"]
                if sym not in universe:
                    universe[sym] = {
                        "Symbol": sym,
                        "Close": stock["Close"],
                        "Volume": stock["Volume"],
                        "Name": stock["Name"],
                        "Screeners": [screener_name]
                    }
                else:
                    if screener_name not in universe[sym]["Screeners"]:
                        universe[sym]["Screeners"].append(screener_name)
        log_success(f"Loaded {len(universe)} unique tickers from local Chartink cache.")
        return universe

    log_info("Fetching active stock list from 11 Chartink screeners via REST fallback...")
    universe = fetch_chartink_universe_rest(date_str, export_dir, CHARTINK_URLS)
    return universe
