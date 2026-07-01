"""
Pipeline Health Monitor — Data Source Quality + Pipeline Status
================================================================
Tracks data source availability and pipeline execution health.
"""
import os, json, datetime, subprocess, sys

PIPELINE_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
HERMES_CACHE = r"C:\Vs code Automation\Hermes\bhavcopy_cache"

class HealthReport:
    """Pipeline data source health tracker."""
    
    def __init__(self):
        self.checks = {}
        
    def check_chartink(self):
        """Check if Chartink screeners returned results in latest run."""
        try:
            dates = sorted([d for d in os.listdir(PIPELINE_BASE) if os.path.isdir(os.path.join(PIPELINE_BASE, d))], reverse=True)
            if not dates:
                return {"status": "❌", "detail": "No pipeline output found"}
            
            latest = dates[0]
            export_dir = os.path.join(PIPELINE_BASE, latest, "chartink_exports")
            if os.path.isdir(export_dir):
                files = [f for f in os.listdir(export_dir) if f.endswith(".csv")]
                return {"status": "✅", "detail": f"{len(files)} screeners ran on {latest}"}
            return {"status": "⚠️", "detail": f"No chartink_exports in {latest}"}
        except Exception as e:
            return {"status": "❌", "detail": str(e)}
    
    def check_bhavcopy(self):
        """Check bhavcopy cache freshness."""
        try:
            if not os.path.isdir(HERMES_CACHE):
                return {"status": "❌", "detail": "bhavcopy_cache dir not found"}
            files = [f for f in os.listdir(HERMES_CACHE) if f.endswith(".parquet")]
            if not files:
                return {"status": "❌", "detail": "No parquet files"}
            
            latest = max(files)
            # Extract date from filename: bhav_30122025.parquet
            date_str = latest.replace("bhav_", "").replace(".parquet", "")
            return {"status": "✅", "detail": f"{len(files)} files, latest: {date_str}"}
        except Exception as e:
            return {"status": "❌", "detail": str(e)}
    

    def check_latest_pipeline(self):
        """Check when pipeline last ran successfully."""
        try:
            dates = sorted([d for d in os.listdir(PIPELINE_BASE) if os.path.isdir(os.path.join(PIPELINE_BASE, d))], reverse=True)
            if not dates:
                return {"status": "❌", "detail": "Never ran"}
            
            latest = dates[0]
            state_path = os.path.join(PIPELINE_BASE, latest, "state_3_0.json")
            if os.path.exists(state_path):
                with open(state_path) as f:
                    state = json.load(f)
                regime = state.get("regime", {}).get("market_regime", "?")
                return {"status": "✅", "detail": f"Last run: {latest}, Regime: {regime}"}
            return {"status": "⚠️", "detail": f"Last run: {latest}, no state.json"}
        except Exception as e:
            return {"status": "❌", "detail": str(e)}
    
    def generate_report(self):
        """Run all checks and return structured report."""
        self.checks = {
            "pipeline": self.check_latest_pipeline(),
            "chartink": self.check_chartink(),
            "bhavcopy": self.check_bhavcopy(),
        }
        
        # Overall health
        statuses = [v["status"] for v in self.checks.values()]
        all_ok = all(s == "✅" for s in statuses)
        any_fail = any(s == "❌" for s in statuses)
        
        if all_ok:
            self.overall = "✅ ALL SYSTEMS OPERATIONAL"
        elif any_fail:
            self.overall = "🔴 ISSUES DETECTED — Check details"
        else:
            self.overall = "🟡 DEGRADED — Some systems warning"
        
        return self
    
    def print_report(self):
        """Print formatted health report."""
        print(f"\n{'='*60}")
        print(f"  🩺 PIPELINE HEALTH CHECK")
        print(f"  {datetime.datetime.now().strftime('%d-%b-%Y %H:%M IST')}")
        print(f"{'='*60}")
        print(f"\n  Overall: {self.overall}\n")
        
        for name, check in self.checks.items():
            print(f"  {check['status']}  {name.upper():<15} {check['detail']}")
        
        print(f"\n{'='*60}\n")


def check_and_report():
    """Quick one-liner to run health check."""
    hr = HealthReport()
    hr.generate_report()
    hr.print_report()
    return hr


def aggregate_pipeline_alerts(regime_results=None):
    """Collects pipeline health warnings into a structured alert summary.
    
    Consumed by the dashboard and cron jobs to surface data quality issues.
    Returns a dict with:
        - critical: list of CRITICAL issues requiring immediate attention
        - warnings: list of WARNING-level concerns
        - data_sources: dict of source freshness status
        - overall: "OK" / "DEGRADED" / "CRITICAL"
    """
    alerts = {"critical": [], "warnings": [], "data_sources": {}, "overall": "OK"}
    
    # 1. Check screener.in staleness
    try:
        from cache_manager import is_screener_cache_stale
        if is_screener_cache_stale(max_days=7):
            alerts["critical"].append("Screener.in fundamentals >7 days stale — PE/PB valuations are hardcoded defaults")
            alerts["data_sources"]["screener_in"] = "STALE_7D"
        elif is_screener_cache_stale(max_days=2):
            alerts["warnings"].append("Screener.in not refreshed in 2+ days")
            alerts["data_sources"]["screener_in"] = "STALE_2D"
        else:
            alerts["data_sources"]["screener_in"] = "FRESH"
    except Exception:
        alerts["data_sources"]["screener_in"] = "UNKNOWN"
    
    # 2. Check from regime_results if provided
    if regime_results:
        if regime_results.get("screener_data_stale"):
            alerts["critical"].append("Live pipeline run used stale screener.in data")
        if regime_results.get("stop_new_buys"):
            alerts["warnings"].append(f"Market off-switch active: new buys blocked (breadth < 30% or Nifty 500 < 200 SMA)")
    
    # 3. Check bhavcopy freshness
    try:
        import glob
        bhav_files = sorted(glob.glob(os.path.join(HERMES_CACHE, "bhav_*.parquet")))
        if bhav_files:
            latest_file = os.path.basename(bhav_files[-1])
            date_str = latest_file.replace("bhav_", "").replace(".parquet", "")
            try:
                file_date = datetime.datetime.strptime(date_str, "%d%m%Y").date()
                days_old = (datetime.date.today() - file_date).days
                if days_old > 5:
                    alerts["warnings"].append(f"Bhavcopy data {days_old} days old — breadth calculation may be unreliable")
                alerts["data_sources"]["bhavcopy"] = f"{days_old}d old"
            except:
                alerts["data_sources"]["bhavcopy"] = "UNPARSABLE"
        else:
            alerts["warnings"].append("No bhavcopy files found — breadth calculation degraded")
            alerts["data_sources"]["bhavcopy"] = "MISSING"
    except Exception:
        alerts["data_sources"]["bhavcopy"] = "ERROR"
    
    # 4. Check pipeline last run
    try:
        dates = sorted([d for d in os.listdir(PIPELINE_BASE) if os.path.isdir(os.path.join(PIPELINE_BASE, d))], reverse=True)
        if dates:
            last_run = dates[0]
            last_date = datetime.datetime.strptime(last_run, "%Y-%m-%d").date()
            days_since = (datetime.date.today() - last_date).days
            alerts["data_sources"]["last_pipeline_run"] = f"{last_run} ({days_since}d ago)"
            if days_since > 1:
                alerts["warnings"].append(f"Pipeline last ran {days_since} days ago — data may be stale")
        else:
            alerts["critical"].append("Pipeline has never run — no output directory found")
            alerts["data_sources"]["last_pipeline_run"] = "NEVER"
    except Exception:
        pass
    
    # Determine overall status
    if alerts["critical"]:
        alerts["overall"] = "CRITICAL"
    elif alerts["warnings"]:
        alerts["overall"] = "DEGRADED"
    
    return alerts


if __name__ == "__main__":
    check_and_report()
