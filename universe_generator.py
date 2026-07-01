import os
import pandas as pd
from datetime import datetime
import config
from utils import log_info, log_success
from mf_fetcher import get_master_mf_list

EXCLUDE_WORDS = [
    "regular", "idcw", "dividend", "debt", "liquid", "overnight", "gilt",
    "money market", "fmp", "fixed maturity", "arbitrage", "segregated", "suspended",
    "tax saver", "elss", "bond", "target maturity", "income"
]

# We want high quality equity funds that match our momentum/alpha/trend themes
THEMATIC_WORDS = [
    "small cap", "micro cap", "mid cap", "momentum", "alpha", "value", 
    "flexi cap", "nifty 50", "next 50", "sensex", "large cap",
    "it ", "technology", "pharma", "defense", "psu", "infrastructure", "auto", "nasdaq"
]

def generate_curated_universe():
    """
    Returns exactly the user-defined CORE_ETF_UNIVERSE (which includes the resolved items
    from the Excel core allocations).
    """
    log_info("Running Dynamic Universe Screener (Restricted to Core Allocations)...")
    
    user_etfs = config.CORE_ETF_UNIVERSE
    final_universe = []
    
    for sym, name in user_etfs.items():
        final_universe.append({"Symbol": sym, "Name": name})
            
    df_universe = pd.DataFrame(final_universe)
    
    # Save the universe mapping for reference
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", datetime.now().strftime("%Y-%m-%d"))
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "Stage1_Curated_Universe.csv")
    df_universe.to_csv(out_file, index=False)
    
    log_success(f"Final Universe created strictly from Core Allocations with {len(df_universe)} assets. Saved to {out_file}")
    
    return [item["Symbol"] for item in final_universe]

if __name__ == "__main__":
    generate_curated_universe()
