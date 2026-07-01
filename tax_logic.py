import pandas as pd
from datetime import datetime
from utils import log_info, log_warning

# Current Indian Equity Tax Rates (including surcharges)
STCG_RATE = 0.208  # 20% + 0.80% surcharge for < 365 days
LTCG_RATE = 0.130  # 12.5% + 0.50% surcharge for >= 365 days
LTCG_THRESHOLD_DAYS = 365

def evaluate_tax_friction(entry_price, current_price, entry_date_str, current_date_str, estimated_daily_decay_pct=0.005, qty=1):
    """
    Evaluates whether an exit should be delayed due to tax friction.
    (The 'CA' Tax-Aware Exit Logic)
    
    If the stock is in STCG but close to LTCG, we calculate if holding for the remaining days 
    costs less in momentum decay than it saves in taxes.
    
    Returns:
        delay_exit (bool): True if we should hold for tax savings, False to exit now.
        reason (str): Explanation of the decision.
    """
    if entry_price >= current_price:
        return False, "No profit, no tax friction. Exit immediately."
        
    try:
        entry_date = pd.to_datetime(entry_date_str)
        current_date = pd.to_datetime(current_date_str)
        days_held = (current_date - entry_date).days
    except Exception as e:
        log_warning(f"Error parsing dates in tax logic: {e}")
        return False, "Date parse error, exit immediately."
        
    if days_held >= LTCG_THRESHOLD_DAYS:
        return False, f"Already LTCG (held {days_held} days). Exit immediately."
        
    days_remaining = LTCG_THRESHOLD_DAYS - days_held
    
    # We only consider delaying if we are close to the LTCG threshold (e.g., <= 45 days)
    if days_remaining > 45:
        return False, f"Too far from LTCG ({days_remaining} days remaining). Exit immediately."
        
    profit = (current_price - entry_price) * qty
    if profit <= 0:
        return False, "No profit, no tax friction."
        
    # Tax under STCG (if sold today)
    stcg_tax = profit * STCG_RATE
    
    # Tax under LTCG (if sold after remaining days, assuming price doesn't change)
    ltcg_tax = profit * LTCG_RATE
    
    # Absolute tax savings
    tax_savings = stcg_tax - ltcg_tax
    
    # Expected loss due to price decay
    # estimated_daily_decay_pct is the expected % drop per day. 
    # expected_loss = current_price * daily_decay * days_remaining * qty
    expected_loss = (current_price * estimated_daily_decay_pct) * days_remaining * qty
    
    if expected_loss < tax_savings:
        # It costs less to hold than to pay the STCG tax!
        net_benefit = tax_savings - expected_loss
        reason = f"TAX GUARD: Delaying exit. Days to LTCG: {days_remaining}. Tax savings: ₹{tax_savings:.2f}, Est. decay loss: ₹{expected_loss:.2f} (Net benefit: ₹{net_benefit:.2f})"
        log_info(reason)
        return True, reason
    else:
        reason = f"TAX GUARD CLEARED: Exit now. Decay loss (₹{expected_loss:.2f}) exceeds tax savings (₹{tax_savings:.2f})."
        return False, reason

