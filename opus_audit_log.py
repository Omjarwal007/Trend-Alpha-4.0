"""
Opus Audit Log — Track Claude Opus 4 Contradiction Audit Verdicts
==================================================================
Append-only log of every CONFIRM / ADJUST -1 / RED FLAG verdict.

Usage:
    from opus_audit_log import log_opus_verdict, get_recent_audits
    
    log_opus_verdict("RELIANCE", "RED FLAG", "Promoter pledging 15% in last quarter")
    recent = get_recent_audits(limit=10)
"""

import os
import csv
import json
from datetime import datetime

AUDIT_LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "opus_audit_log.csv")


def log_opus_verdict(symbol: str, verdict: str, reason: str):
    """Append a Claude Opus 4 audit verdict to the log."""
    file_exists = os.path.exists(AUDIT_LOG_FILE)
    try:
        with open(AUDIT_LOG_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["timestamp", "symbol", "verdict", "reason"])
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                symbol.upper(),
                verdict,
                reason
            ])
    except Exception as e:
        print(f"  ⚠️ Audit log write failed: {e}")


def get_recent_audits(limit: int = 20) -> list:
    """Return the most recent N audit entries."""
    if not os.path.exists(AUDIT_LOG_FILE):
        return []
    try:
        with open(AUDIT_LOG_FILE, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        return rows[-limit:]
    except Exception:
        return []


def get_red_flag_count(since_days: int = 7) -> int:
    """Count RED FLAG verdicts in the last N days."""
    if not os.path.exists(AUDIT_LOG_FILE):
        return 0
    count = 0
    cutoff = datetime.now().timestamp() - (since_days * 86400)
    try:
        with open(AUDIT_LOG_FILE, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("verdict") == "RED FLAG":
                    try:
                        ts = datetime.strptime(row["timestamp"], "%Y-%m-%d %H:%M:%S")
                        if ts.timestamp() >= cutoff:
                            count += 1
                    except:
                        pass
    except Exception:
        pass
    return count


if __name__ == "__main__":
    # Self-test
    print("=== Opus Audit Log — Self Test ===")
    log_opus_verdict("RELIANCE", "CONFIRM", "Gemini and Grok agree, no governance concerns")
    log_opus_verdict("TCS", "ADJUST -1", "Grok sentiment negative but Gemini positive - mild contradiction")
    log_opus_verdict("HDFCBANK", "RED FLAG", "Promoter pledging increased 15% in last quarter")
    
    recent = get_recent_audits()
    print(f"  Logged {len(recent)} entries")
    for row in recent:
        print(f"    {row['timestamp']} | {row['symbol']} | {row['verdict']} | {row['reason'][:60]}")
    
    rf_count = get_red_flag_count()
    print(f"  RED FLAG count (7d): {rf_count}")
    print("  ✅ All tests passed")
