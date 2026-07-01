import os
import sys

# Windows terminal UTF-8 encoding configuration
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

def log_info(msg):
    try:
        print(f"🔵 [INFO] {msg}")
    except Exception:
        try:
            print(f"[INFO] {msg}")
        except Exception:
            pass

def log_success(msg):
    try:
        print(f"🟢 [SUCCESS] {msg}")
    except Exception:
        try:
            print(f"[SUCCESS] {msg}")
        except Exception:
            pass

def log_warning(msg):
    try:
        print(f"🟡 [WARN] {msg}")
    except Exception:
        try:
            print(f"[WARN] {msg}")
        except Exception:
            pass

def log_error(msg):
    try:
        print(f"🔴 [ERROR] {msg}")
    except Exception:
        try:
            print(f"[ERROR] {msg}")
        except Exception:
            pass

def log_system(msg):
    try:
        print(f"⚡ [SYSTEM] {msg}")
    except Exception:
        try:
            print(f"[SYSTEM] {msg}")
        except Exception:
            pass

def ensure_output_dir(date_str=None):
    from config import OUTPUT_DIR
    if date_str:
        path = os.path.join(OUTPUT_DIR, date_str)
    else:
        path = OUTPUT_DIR
    os.makedirs(path, exist_ok=True)
    return path
def extract_fund_keywords(name: str) -> set:
    """Extracts core index/thematic keywords from a fund name to prevent duplication."""
    if not name:
        return set()
    name = name.lower()
    keywords = set()
    
    # Master list of identifying keywords for core ETFs/MFs
    targets = [
        "nifty 50", "nifty next 50", "midcap 150", "smallcap 250", "nifty 500", "sensex",
        "midcap", "smallcap", "largecap", "multicap", "flexicap", "large & midcap",
        "nasdaq", "s&p 500", "fang", "us equity", "emerging market", "global",
        "it", "technology", "pharma", "healthcare", "bank", "psu bank", "financial",
        "auto", "fmcg", "consumption", "infrastructure", "infra", "defense", "manufacturing",
        "gold", "silver", "metal", "commodity",
        "value", "quality", "momentum", "alpha", "low volatility", "dividend", "quant", "factor"
    ]
    
    for t in targets:
        # For very short keywords, check with word boundaries
        if len(t) <= 2:
            import re
            if re.search(rf'\b{t}\b', name):
                keywords.add(t)
        else:
            if t in name:
                keywords.add(t)
                
    return keywords

def categorize_fund_by_name(name: str) -> str:
    """
    Categorizes a mutual fund or ETF based on its name to ensure diversification.
    """
    if not name:
        return "BROAD_MARKET"
        
    name = name.lower()
    
    # 1. Global / US Equities
    if any(k in name for k in ["nasdaq", "s&p", "us ", "us equity", "emerging market", "global", "world", "hang seng", "taiwan"]):
        return "GLOBAL_EQUITIES"
        
    # 2. Tech / IT Sector
    if any(k in name for k in ["it ", "technology", "tech ", "digital", "software", "eqqq"]):
        return "SECTOR_TECH"
        
    # 3. Small / Micro Cap
    if any(k in name for k in ["smallcap", "small cap", "microcap", "micro cap"]):
        return "SMALL_CAP"
        
    # 4. Mid Cap / MidSmall
    if any(k in name for k in ["midcap", "mid cap", "midsmall", "mid small"]):
        return "MID_CAP"
        
    # 5. Financials / Banking
    if any(k in name for k in ["bank", "financial", "psu bank", "bfsi"]):
        return "SECTOR_FINANCIAL"
        
    # 6. Healthcare / Pharma
    if any(k in name for k in ["pharma", "health", "healthcare"]):
        return "SECTOR_HEALTHCARE"
        
    # 7. Defense / Infra / Manufacturing
    if any(k in name for k in ["defense", "defence", "infra", "manufacturing", "capital goods"]):
        return "SECTOR_INFRA_DEFENSE"
        
    # 8. Consumption / Auto / FMCG
    if any(k in name for k in ["auto", "fmcg", "consumption", "consumer"]):
        return "SECTOR_CONSUMPTION"
        
    # 9. Commodities
    if any(k in name for k in ["gold", "silver", "commodity", "commodities"]):
        return "COMMODITY"
        
    # 10. Cash / Liquid / Debt (though these should be filtered out)
    if any(k in name for k in ["liquid", "cash", "debt", "money market"]):
        return "CASH_LIQUID"
        
    # 11. Thematic / Smart Beta (Value, Quality, Alpha, High Beta)
    if any(k in name for k in ["value", "quality", "alpha", "beta", "momentum", "quant", "smart beta", "factor"]):
        return "SMART_BETA_THEMATIC"

    # Default to Broad Market / Large Cap (Nifty 50, Sensex, Multicap, Flexicap)
    return "BROAD_MARKET"

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
    except Exception:
        pass
    return mf_name_map
