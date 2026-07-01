import os
import time
import requests
import json
import difflib
import pandas as pd
from datetime import datetime
from utils import log_info, log_warning, log_success

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MF_CACHE_DIR = os.path.join(BASE_DIR, "cache", "mfapi")
os.makedirs(MF_CACHE_DIR, exist_ok=True)

MASTER_LIST_CACHE_FILE = os.path.join(MF_CACHE_DIR, "master_list.json")

def get_master_mf_list():
    """Fetches and caches the master list of all Indian Mutual Funds from mfapi.in"""
    # Check cache first (Valid for 7 days since new funds don't appear daily)
    if os.path.exists(MASTER_LIST_CACHE_FILE):
        file_age_days = (time.time() - os.path.getmtime(MASTER_LIST_CACHE_FILE)) / (24 * 3600)
        if file_age_days < 7.0:
            try:
                with open(MASTER_LIST_CACHE_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                log_warning(f"Failed to load cached master MF list: {e}")

    log_info("Downloading master Mutual Fund index from api.mfapi.in...")
    try:
        response = requests.get("https://api.mfapi.in/mf", timeout=30)
        response.raise_for_status()
        data = response.json()
        
        # Save to cache
        with open(MASTER_LIST_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f)
            
        log_success(f"Downloaded {len(data)} Mutual Fund codes.")
        return data
    except Exception as e:
        log_warning(f"Failed to fetch master MF list from mfapi.in: {e}")
        return []

def resolve_mf_code(query_name):
    """
    Given a fuzzy query name (e.g. 'Quant Small Cap'), returns the best matching 6-digit AMFI code.
    Returns None if no reasonable match is found.
    """
    master_list = get_master_mf_list()
    if not master_list:
        return None
        
    query_name_lower = str(query_name).lower().strip()
    
    # Fast exact substring match prioritizing Growth/Direct plans
    candidates = []
    for mf in master_list:
        name_lower = mf.get("schemeName", "").lower()
        if query_name_lower in name_lower:
            candidates.append(mf)
            
    if candidates:
        # Prioritize Direct plans, then Growth plans over Regular/IDCW
        best_candidate = candidates[0]
        max_score = -1
        for c in candidates:
            score = 0
            name_lower = c.get("schemeName", "").lower()
            if "direct" in name_lower:
                score += 10
            if "growth" in name_lower:
                score += 5
            if "regular" in name_lower:
                score -= 10
            if "idcw" in name_lower or "dividend" in name_lower:
                score -= 10
            if score > max_score:
                max_score = score
                best_candidate = c
                
        return str(best_candidate.get("schemeCode"))
        
    # Slower fuzzy match if exact substring fails
    names = [mf.get("schemeName", "") for mf in master_list]
    close_matches = difflib.get_close_matches(query_name, names, n=5, cutoff=0.6)
    
    if close_matches:
        # Same prioritization logic on fuzzy matches
        best_match = close_matches[0]
        max_score = -1
        best_code = None
        
        for match in close_matches:
            # Find the dict
            c = next((item for item in master_list if item["schemeName"] == match), None)
            if c:
                score = 0
                name_lower = c.get("schemeName", "").lower()
                if "direct" in name_lower:
                    score += 10
                if "growth" in name_lower:
                    score += 5
                if "regular" in name_lower:
                    score -= 10
                
                if score > max_score:
                    max_score = score
                    best_match = match
                    best_code = str(c.get("schemeCode"))
                    
        return best_code
        
    return None

def fetch_mf_history(scheme_code):
    """
    Downloads historical NAV data for a given AMFI scheme_code from mfapi.in
    and formats it exactly like a yfinance DataFrame.
    """
    url = f"https://api.mfapi.in/mf/{scheme_code}"
    
    data = None
    for attempt in range(5):
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()
            break
        except Exception as req_err:
            if attempt == 4:
                log_warning(f"Failed to fetch NAV history for {scheme_code} after 5 attempts: {req_err}")
                return None
            # Exponential backoff to handle rate limits / concurrent connection limits
            time.sleep(1.5 + (attempt * 1.5))
            
    try:
        if not data or "data" not in data or not data["data"]:
            log_warning(f"No historical data returned for MF Code {scheme_code}")
            return None
            
        # Parse into DataFrame
        df = pd.DataFrame(data["data"])
        
        # 'date' comes in 'dd-mm-yyyy'
        df["Date"] = pd.to_datetime(df["date"], format="%d-%m-%Y")
        df.set_index("Date", inplace=True)
        df.sort_index(ascending=True, inplace=True)
        
        df["Close"] = pd.to_numeric(df["nav"], errors="coerce")
        df.dropna(subset=["Close"], inplace=True)
        
        # Synthesize Open, High, Low from Close to prevent pipeline crashes
        df["Open"] = df["Close"]
        df["High"] = df["Close"]
        df["Low"] = df["Close"]
        
        # Add arbitrary volume since mutual funds don't have secondary market volume
        df["Volume"] = 100000
        
        df = df[["Open", "High", "Low", "Close", "Volume"]]
        
        # Ensure it works nicely with standard tools
        df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
            
        return df
        
    except Exception as e:
        log_warning(f"Failed to fetch NAV history for {scheme_code}: {e}")
        return None

def fetch_mf_data_auto(symbol):
    """
    Master entry point. Attempts to see if the symbol is an AMFI code.
    If not, strips Yahoo extensions and fuzzy matches it to an AMFI code.
    Downloads and returns the formatted DataFrame.
    """
    # Clean symbol
    clean_sym = str(symbol).replace(".NS", "").replace(".BO", "").strip()
    
    # Check if it's already a 6-digit code
    if clean_sym.isdigit() and len(clean_sym) == 6:
        scheme_code = clean_sym
        log_info(f"Symbol {symbol} recognized as AMFI code {scheme_code}.")
    else:
        log_info(f"Symbol {symbol} is not a 6-digit code. Attempting to fuzzy-match mutual fund names...")
        scheme_code = resolve_mf_code(clean_sym)
        if not scheme_code:
            log_warning(f"Could not resolve {symbol} to any Mutual Fund.")
            return None
        log_info(f"Resolved '{symbol}' to Mutual Fund AMFI Code: {scheme_code}")
        
    return fetch_mf_history(scheme_code)
