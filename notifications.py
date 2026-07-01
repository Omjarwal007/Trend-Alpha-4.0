"""
Notifications Module — Exit Alerts + Telegram Integration
=========================================================
"""
import os, sys, datetime
import pandas as pd
from utils import log_warning

def send_exit_notifications(date_str=None):
    """Read Execution_Orders.csv and dispatch exit alerts.
    
    Currently logs to console (Telegram/email/webhook ready to add).
    To add Telegram: set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env
    """
    if date_str is None:
        date_str = datetime.date.today().isoformat()
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(base_dir, "output", date_str)
    orders_path = os.path.join(out_dir, "Execution_Orders.csv")
    
    if not os.path.exists(orders_path):
        return 0  # No orders file — nothing to notify
    
    df = pd.read_csv(orders_path)
    if df.empty or "Action" not in df.columns:
        return 0
    
    # Filter for EXIT orders only
    exits = df[df["Action"] == "EXIT"]
    if exits.empty:
        return 0
    
    alerts = []
    for _, row in exits.iterrows():
        symbol = row.get("Symbol", "?")
        reason = row.get("Reason", "No reason")
        qty = row.get("Quantity", 0)
        exit_pct = row.get("Allocation_%", 0)
        
        alert = f"🚨 EXIT: {symbol}"
        if exit_pct > 0:
            alert += f" ({exit_pct:.1f}%)"
        alert += f" | {reason}"
        if qty > 0:
            alert += f" | Qty: {qty}"
        
        alerts.append(alert)
        print(f"[NOTIFICATION] {alert}")
    
    # Also check for critical exits (score ≥ 80)
    critical = [a for a in alerts if "CRITICAL" in a or "🔴" in a]
    
    if critical:
        print("\n" + "=" * 60)
        print("🔴 CRITICAL EXIT ALERTS REQUIRING IMMEDIATE ATTENTION:")
        for c in critical:
            print(f"  {c}")
        print("=" * 60)
    
    print(f"\n[NOTIFICATIONS] {len(alerts)} exit alerts sent")
    
    # ── Future: Telegram Integration ──
    # To enable, set these in your .env file:
    # TELEGRAM_BOT_TOKEN=your_bot_token
    # TELEGRAM_CHAT_ID=your_chat_id
    # 
    # Then uncomment below:
    #
    # bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    # chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
    # if bot_token and chat_id:
    #     import requests
    #     for alert in alerts:
    #         url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    #         data = {"chat_id": chat_id, "text": alert, "parse_mode": "HTML"}
    #         try:
    #             requests.post(url, data=data, timeout=10)
    #         except Exception as e:
    #             log_warning(f"Telegram send failed: {e}")
    
    return len(alerts)
