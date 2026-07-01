import os
import sys

# Ensure UTF-8 output on Windows terminal
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Core Workspace Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
CACHE_DIR = os.path.join(BASE_DIR, "cache")

# Default Portfolio Capitalization
DEFAULT_PORTFOLIO_CAPITAL = 10000000.0  # ₹1 Crore

# Tickers universe (fallback set matching Trend Alpha standard)
SYMBOLS = [
    "ACUTAAS", "SKYGOLD", "BSE", "SHILPAMED", "SOLARINDS", "LLOYDSME", 
    "FINCABLES", "CGPOWER", "J&KBANK", "APARINDS", "SAIL", "GRANULES", 
    "HINDALCO", "HINDCOPPER", "KIRLOSENG", "NLCINDIA", "SCHNEIDER", 
    "ANGELONE", "SANSERA", "KTKBANK", "LAURUSLABS", "JINDALSAW", 
    "CUMMINSIND", "GVT&D", "ABB", "BANDHANBNK", "DATAPATTNS", "HSCL", "GMDCLTD"
]

# Dedicated Chartink screener URLs for dynamic universe extraction
CHARTINK_URLS = [
    "https://chartink.com/screener/upside-mean-reversion-base-building-breakouts",
    "https://chartink.com/screener/cum-and-handle-pattern",
    "https://chartink.com/screener/reversal-scan-new-2",
    "https://chartink.com/screener/rounding-up-bottom-and-bear-flag",
    "https://chartink.com/screener/early-recovery-3",
    "https://chartink.com/screener/strong-long-term-trend-2",
    "https://chartink.com/screener/52-w-h-in-last-3-days",
    "https://chartink.com/screener/200dma-scan-6",
    "https://chartink.com/screener/52w-h-ath",
    "https://chartink.com/screener/the-techno-funda-leader-screener",
    "https://chartink.com/screener/trend-alfa"
]

# Sector Mapping
_SECTORS_FORWARD = {
    "ACUTAAS": "Consumer Cyclical - Apparel Retail",
    "SKYGOLD": "Consumer Cyclical - Luxury Goods",
    "BSE": "Financial Services - Capital Markets",
    "SHILPAMED": "Healthcare - Biotechnology",
    "SOLARINDS": "Industrials - Aerospace & Defense",
    "LLOYDSME": "Industrials - Metal Fabrication",
    "FINCABLES": "Industrials - Electrical Equipment",
    "CGPOWER": "Industrials - Electrical Equipment",
    "J&KBANK": "Financial Services - Banks - Regional",
    "APARINDS": "Industrials - Electrical Equipment",
    "SAIL": "Basic Materials - Steel",
    "GRANULES": "Healthcare - Drug Manufacturers - Specialty",
    "HINDALCO": "Basic Materials - Aluminum",
    "HINDCOPPER": "Basic Materials - Copper",
    "KIRLOSENG": "Industrials - Specialty Industrial Machinery",
    "NLCINDIA": "Utilities - Utilities - Independent Power Producers",
    "SCHNEIDER": "Industrials - Electrical Equipment",
    "ANGELONE": "Financial Services - Capital Markets",
    "SANSERA": "Consumer Cyclical - Auto Parts",
    "KTKBANK": "Financial Services - Banks - Regional",
    "LAURUSLABS": "Healthcare - Drug Manufacturers - Specialty",
    "JINDALSAW": "Basic Materials - Steel",
    "CUMMINSIND": "Industrials - Specialty Industrial Machinery",
    "GVT&D": "Industrials - Electrical Equipment",
    "ABB": "Industrials - Electrical Equipment",
    "BANDHANBNK": "Financial Services - Banks - Regional",
    "DATAPATTNS": "Industrials - Aerospace & Defense",
    "HSCL": "Basic Materials - Specialty Chemicals",
    "GMDCLTD": "Basic Materials - Other Industrial Metals & Mining"
}

# Theme/Industry Group Mapping for Theme Limits (max 35%)
_THEMES_FORWARD = {
    "SOLARINDS": "Defense & Capital Goods",
    "DATAPATTNS": "Defense & Capital Goods",
    "CGPOWER": "Power & Electrical Infrastructure",
    "SCHNEIDER": "Power & Electrical Infrastructure",
    "GVT&D": "Power & Electrical Infrastructure",
    "ABB": "Power & Electrical Infrastructure",
    "APARINDS": "Power & Electrical Infrastructure",
    "FINCABLES": "Power & Electrical Infrastructure",
    "SAIL": "Metals & Mining",
    "HINDALCO": "Metals & Mining",
    "HINDCOPPER": "Metals & Mining",
    "JINDALSAW": "Metals & Mining",
    "GMDCLTD": "Metals & Mining",
    "GRANULES": "Pharma & Lifesciences",
    "SHILPAMED": "Pharma & Lifesciences",
    "LAURUSLABS": "Pharma & Lifesciences",
    "J&KBANK": "Financials",
    "KTKBANK": "Financials",
    "BANDHANBNK": "Financials",
    "BSE": "Financials",
    "ANGELONE": "Financials",
}

# Core Sizing & Risk parameters (SKILL 00, 03)
BASE_RISK_PER_TRADE_PCT = 0.01      # 1% standard risk per trade
MAX_SINGLE_STOCK_CASH_PCT = 0.08    # Max 8% allocation for base cash position
MAX_SINGLE_STOCK_ABS_PCT = 0.12     # Max 12% absolute stock weight (with leverage)
MAX_SECTOR_PCT = 0.25               # Max 25% exposure per sector
MAX_INDUSTRY_PCT = 0.20             # Max 20% exposure per industry
MAX_THEME_PCT = 0.35                # Max 35% exposure per theme
MAX_PORTFOLIO_HEAT = 0.06           # Max portfolio open risk at 6% (caution at 4-6%, block at >6%, reduce at >8%)
MAX_OPEN_POSITIONS = 45             # Limit portfolio to 45 positions (20 VAM-GQ + 20 VAM-B + 5 Core)

# ── CORE ETF ALLOCATION ARCHITECTURE ──
# NOTE: Config values below document the original design intent.
# The actual runtime allocation is driven by portfolio_manager.py which:
#   - Uses dynamic MTF margin (max_core_cash = portfolio_value * 2.00 for MTF overflow)
#   - Allocates Core ~76% / Satellite ~19% of deployed capital
#   - Caps per position by rs_tilt_alloc_pct (1.5%-5% tiers)
CORE_ALLOCATION_PCT = 0.65
ACTIVE_ALLOCATION_PCT = 0.35
ACTIVE_MAIN_ALLOCATION_PCT = 0.28       # Top 8 active stocks command 28% of portfolio capital
ACTIVE_REMAINING_ALLOCATION_PCT = 0.07  # Remaining active stocks share the final 7% of portfolio capital
ACTIVE_MAIN_STOCKS_COUNT = 8            # Number of top conviction active stocks
CORE_FRICTION_PENALTY_PCT = 0.05    # Core ETF must beat current holding by 5% to trigger rotation
MAX_CORE_ETFS = 5                   # TA 4.0: Max 5 core holdings (max 1 per category)

CORE_ETF_UNIVERSE = {}
CORE_ETF_ZONES = {}
_json_path = os.path.join(CACHE_DIR, "core_allocations_universe.json")
if os.path.exists(_json_path):
    try:
        import json
        with open(_json_path, "r", encoding="utf-8") as f:
            _universe_list = json.load(f)
            for item in _universe_list:
                CORE_ETF_UNIVERSE[item["Symbol"]] = item["Name"]
                CORE_ETF_ZONES[item["Symbol"]] = item.get("FileSource", "Unknown")
    except Exception:
        pass

if not CORE_ETF_UNIVERSE:
    CORE_ETF_UNIVERSE = {
        # Broad Market & Smart Beta
        "MID150BEES.NS": "Nifty Midcap 150",
        "MOMOM100.NS": "Nifty Midcap 150 Momentum 50",
        "ALPHABEES.NS": "Nifty Alpha 50",
        "LOWVOLIETF.NS": "Nifty Low Vol 30",
        "SMALLCAP.NS": "Nifty Smallcap 250 Index",
        "NIFTYMICROCAP250.NS": "Nifty Microcap 250",
        "JUNIORBEES.NS": "Nifty Next 50",
        "NIFTYBEES.NS": "Nifty 50",
        "MID150CASE.NS": "Nifty Midcap 150",
        
        # Advanced Strategy, Quality, & Alpha/Beta Funds
        "MOM50.NS": "Nifty 200 Momentum 30",
        "MIDSMALL.NS": "Nifty MidSmall 400",
        "QUAL30IETF.NS": "Nifty 200 Quality 30",
        "DIVOPPBEES.NS": "Dividend Opportunities",
        "NV20BEES.NS": "Nifty Value 20",
        "GROWTH.NS": "Nifty 500 Value 50",
        "BETA.NS": "Nifty High Beta 50",
        
        # Mutual Funds (Active & Passive Strategies)
        "0P0000XW4J.BO": "Quant Small Cap Fund",
        "0P0000XVAA.BO": "HDFC Small Cap Fund",
        "0P0000XW1A.BO": "SBI Small Cap Fund",
        "0P0000XW8F.BO": "HDFC Mid-Cap Opportunities",
        "0P0000PTGR.BO": "Nippon Small Cap Fund",
        "0P00011MAX.BO": "Axis Small Cap Fund",
        "0P0000XV6I.BO": "Kotak Small Cap Fund",
        "0P0001NJAY.BO": "Motilal Oswal Smallcap 250",
        "0P0001LQY5.BO": "Quant Mid Cap Fund",
        "0P0000XW3A.BO": "SBI Mid Cap Fund",
        
        # Sectoral
        "BANKBEES.NS": "Nifty Bank",
        "ITBEES.NS": "Nifty IT",
        "PHARMABEES.NS": "Nifty Pharma",
        "CONSUMBEES.NS": "Nifty FMCG",
        "AUTOBEES.NS": "Nifty Auto",
        "INFRABEES.NS": "Nifty Infra",
        "PSUBNKBEES.NS": "PSU Bank",
        "HEALTHY.NS": "Healthcare Index",
        
        # Thematic
        "CPSEETF.NS": "CPSE PSE",
        "BHARAT22.NS": "Bharat 22",
        "DEFENSE.NS": "Defense",
        "MAKEINDIA.NS": "Make in India",
        "TNIDEF.NS": "Tata Nifty India Digital",
        
        # Commodities / Defensive
        "GOLDBEES.NS": "Gold",
        "SILVERBEES.NS": "Silver",
        "LIQUIDBEES.NS": "Liquid Cash"
    }

# ── DRAWDOWN CIRCUIT BREAKERS (SKILL 09) ──────────────────────────────────
# IMPORTANT: Drawdown is measured from the ROLLING 30-DAY HIGH, not all-time peak.
# peak_portfolio_value passed to run_drawdown_governor() must be the rolling 30d high.
# These levels match Skill 09 circuit breakers:
#   YELLOW (-8% from rolling 30d high): MTF off, halt new entries, reduce metals 50%
# ═══════════════════════════════════════════════════════════
# QUANTITATIVE EXIT SCORING — Hard Rule
# ═══════════════════════════════════════════════════════════
# Composite exit score (0-100) from 5 independent weakness dimensions.
# Components: RS_Level(30%) + RS_Slope(23%) + Price_Trend(17%) +
#             Momentum(17%) + Underperformance(13%) + Accel_Bonus(0-25)
# Weights sum to 100%. Bonus is an override for rapid freefall.
# ≥ EXIT: Stock exited regardless of RS absolute level
# ≥ WATCH: Monitoring zone, approaching exit
EXIT_SCORE_THRESHOLD = 55.0   # Hard exit threshold. Do not change without backtest validation.
EXIT_SCORE_WATCH_ZONE = 40.0  # Warning zone: score between 40-55 triggers WATCH status.
EXIT_RS_SAFE_ZONE = 0.20      # RS ≥ 0.20 → RS_Level score = 0 (no RS weakness contribution)

# Drawdown circuit breaker thresholds (decimal form — do NOT redefine as integers)
DRAWDOWN_MEASUREMENT = "ROLLING_30D_HIGH"
DRAWDOWN_YELLOW_PCT  = -0.08   # -8%  → Caution: halt entries, exit MTF
DRAWDOWN_ORANGE_PCT  = -0.12   # -12% → Alert: exit overlays, trim to top 10
DRAWDOWN_RED_PCT     = -0.18   # -18% → Emergency: full exit, 30-day pause

# Haircuts & Margin constraints (SKILL 04)
EQUITY_HAIRCUT = 0.20               # 20% haircut on VAM-GQ pledged equities
LIQUID_BONDS_HAIRCUT = 0.10          # 10% haircut on liquid bond units
METALS_MARGIN_REQ = 0.10            # 10% margin requirement on MCX F&O futures (typical range is 5-10%, use 10% for conservative calculation)

# Moving Average periods
MA_FAST = 20
MA_MEDIUM = 50
MA_SLOW_150 = 150
MA_SLOW_200 = 200
MA_WEEKLY = 30                      # Weekly trend MA (30-week MA)

# ── MARKET CAP CLASSIFICATION ─────────────────────────────────────────────
MIN_MARKET_CAP_CR = 1000            # Minimum market cap (₹ Crores) to enter pipeline

# ── 11-STEP PIPELINE CONSTANTS ────────────────────────────────────────────
MIN_ADTV_CR = 10.0                  # 30-day Average Daily Turnover must be >= ₹10 Crores (raised from 5 per Claude review — ensures clean exit execution)
MIN_ADX_14 = 20.0                   # ADX-14 must be > 20
MAX_ANNUAL_VOL_PCT = 60.0           # Annualized daily volatility must be < 60%

# ── LEGACY PIPELINE CONSTANTS (superseded by QUALITY_GATE_BFSI / QUALITY_GATE_STANDARD below) ─
# These are retained for reference only and are NO LONGER used as hard gates in stock_selector.py.
# The Quality Gate dicts below are the authoritative source of truth.
MAX_DE_RATIO = 1.5                  # [LEGACY] D/E — now in QUALITY_GATE_STANDARD
MIN_ROCE_PCT = 12.0                 # [LEGACY] ROCE — now in QUALITY_GATE_STANDARD (relaxed to 10%)
MIN_BFSI_ROE_PCT = 8.0              # [LEGACY] BFSI ROE — replaced by BFSI NPA/CAR/ROA/Pledge gates
MIN_CFO_PAT_RATIO = 0.50            # [LEGACY] CFO/PAT — TTM Cash Flow > 0 now used instead

# ═══════════════════════════════════════════════════════════
# QUALITY GATE — HARD ELIMINATION RULES (non-bypassable)
# ═══════════════════════════════════════════════════════════
# Quality is a TWO-TRACK hard elimination gate, not a scoring factor.
# A stock failing ANY applicable condition is HARD REJECTED before ranking.
# These gates CANNOT be bypassed by is_exceptional_bull or any other override.
#
# Track 1 — BFSI (Banks & NBFCs): banking-specific metrics
# Track 2 — Standard (all other sectors): D/E, CFO/PAT, ROCE, Pledge

# BFSI Quality Gate (applies to banks, NBFCs, financial services)
# All conditions must pass (AND logic)
QUALITY_GATE_BFSI = {
    "Net_NPA_max_pct":          1.00,   # Net NPA must be < 1.00% (Strict)
    "CAR_min_pct":              12.0,   # Capital Adequacy Ratio must be > 12% (RBI floor = 11.5%; 12% adds safe buffer)
    "ROA_min_pct":              0.80,   # Return on Assets must be > 0.80%
    "Promoter_Pledge_max_pct":  15.0,   # Promoter Pledge must be < 15%
    "PCR_min_pct":              70.0,   # Provision Coverage Ratio must be > 70%
    "CASA_min_pct":             35.0,   # CASA ratio must be > 35%
}

# Standard Quality Gate (all sectors except banks & NBFCs)
# All conditions must pass (AND logic)
# NOTE: ROCE check is WAIVED for cyclical sectors (defence, metal, infra, power, etc.)
#       because cyclical recovery stocks are bought precisely BEFORE ROCE normalises.
#       The D/E and CFO_PAT checks still apply to cyclicals.
QUALITY_GATE_STANDARD = {
    "DE_max":                   1.5,    # Debt/Equity must be < 1.5
    "CFO_PAT_3Yr_min":          0.0,    # 3-Year Avg CFO/PAT ratio must be > 0 (positive on average;
                                        #   more forgiving than TTM CFO which punishes cyclical troughs)
    "ROCE_min_pct":             8.0,    # ROCE must be > 8% (relaxed from 10%; cyclical sectors exempt)
    "Promoter_Pledge_max_pct":  20.0,   # Promoter Pledge must be < 20%
}

CAP_CATEGORY_LIMITS = {
    "SMALL_CAP": {"min_cr": 1000,  "max_cr": 5000,   "risk_mult": 0.50, "max_single_pct": 0.04, "max_category_pct": 0.20},
    "MID_CAP":   {"min_cr": 5000,  "max_cr": 20000,  "risk_mult": 0.75, "max_single_pct": 0.06, "max_category_pct": 0.35},
    "LARGE_CAP": {"min_cr": 20000, "max_cr": 100000, "risk_mult": 1.00, "max_single_pct": 0.08, "max_category_pct": 1.00},
    "MEGA_CAP":  {"min_cr": 100000,"max_cr": 1e12,   "risk_mult": 1.25, "max_single_pct": 0.10, "max_category_pct": 1.00},
}

# ── CYCLICAL SECTOR KEYWORDS (for Track 1 trigger) ───────────────────────
CYCLICAL_SECTOR_KEYWORDS = [
    "defence", "defense", "railway", "metal", "mining", "infra",
    "capital goods", "industrial", "psu", "power", "auto", "real estate",
    "cement", "chemical", "fertilizer", "steel", "aluminum", "copper",
    "construction", "engineering", "oil", "gas", "energy", "utilities"
]

# ── WEIGHTED RANKING SYSTEM CONFIGURATION ───────────────────────────────
# Strategy: Trend-Following Positional Trading
# Selection Method: Weighted score (0-100) per factor, rank all candidates,
#                   select TOP_N_STOCKS as eligible. No hard factor gates.
# Hard Safety Gates (non-negotiable, pre-scoring):
#   Market Cap >= 2500 Cr, IPO >= 1yr, ADTV >= 10 Cr, ASM/GSM excl, Price > 200 EMA

# Keep for legacy compatibility (dashboard may reference these)
SELECTION_FILTER_MODE = "WEIGHTED_RANK"
MIN_FACTOR_SCORE = 4   # Legacy fallback — not used in WEIGHTED_RANK mode

# Top N stocks to mark as Entry_Eligible = True
TOP_N_STOCKS = 20

# Factor weights for trend-following positional trading strategy
# Total must sum to 1.0
# NOTE: F5_QUALITY has been removed from scoring — it is now a HARD ELIMINATION GATE.
#       Its former 8% weight has been redistributed: Momentum +3%, Growth +3%,
#       Sectoral +1%, Thematic +1%. ROE now scores inside F4_GROWTH.
FACTOR_WEIGHTS = {
    "F3_MOMENTUM":              0.40,  # VAM-GQ 63-Day Risk-Adjusted Momentum
    "F1_SECTORAL_TREND":        0.14,  # Sector leadership amplifies gains
    "F2_THEMATIC_TREND":        0.09,  # Thematic tailwinds add alpha
    "F4_GROWTH":                0.12,  # Earnings growth + ROE
    "F6_DELIVERY_CONFIRMATION": 0.12,  # Real accumulation vs speculative noise
    "F8_FII_DII_CONVICTION":    0.07,  # Institutional flows confirm trends
    "F7_PEAD":                  0.06,  # Short-term catalyst, decays fast
    # F5_QUALITY: 0.00 — Quality is now a non-bypassable hard elimination gate, not a scored factor.
    #             See QUALITY_GATE_BFSI and QUALITY_GATE_STANDARD for the gate conditions.
}

# List of active themes for Factor 2 check (Thematic Trend)
# These represent genuine macro themes, NOT just sector relabeling
# To add a new theme: add name here AND map stocks in _THEMES_FORWARD/_THEMES_REVERSE
ACTIVE_THEMES = [
    "Defense & Capital Goods",
    "Power & Electrical Infrastructure",
    "Metals & Mining",
    "Pharma & Lifesciences",
    "Financials",
    "Electronics Manufacturing",
    "Regional Banking",
    "Jewellery & Retail",
    "Consumer & FMCG"
]

FACTOR_THRESHOLDS = {
    "F1_SECTORAL_TREND": {
        "name": "Sectoral Trend",
        "emoji": "🔄",
        "sector_rank_min": 50.0,
    },
    "F2_THEMATIC_TREND": {
        "name": "Thematic Trend",
        "emoji": "🎨",
        "theme_rank_min": 50.0,
    },
    "F3_MOMENTUM": {
        "name": "Momentum",
        "emoji": "🚀",
        "rs_rank_min": 60.0,
    },
    "F4_GROWTH": {
        "name": "Growth",
        "emoji": "📈",
        "sales_growth_min": 12.0,
        "profit_growth_min": 12.0,
        "roe_min": 15.0,           # ROE moved here from F5_Quality; 15% = full ROE score in 3-way split
    },
    "F5_QUALITY": {
        # F5_QUALITY is now a HARD ELIMINATION GATE — no longer a scored factor.
        # Conditions are defined in QUALITY_GATE_BFSI and QUALITY_GATE_STANDARD.
        # This entry is retained for dashboard display / labeling purposes only.
        "name": "Quality Gate",
        "emoji": "🔒",
        "mode": "HARD_GATE",       # Signals to dashboard: display as gate, not score
        "bfsi_conditions": "Net NPA < 1.75% | CAR > 12% | ROA > 0.80% | Pledge < 15%",
        "standard_conditions": "D/E < 1.5 | 3Yr CFO/PAT > 0 | ROCE > 8% (cyclicals exempt) | Pledge < 20%",
    },
    "F6_DELIVERY_CONFIRMATION": {
        "name": "Delivery Confirmation",
        "emoji": "📦",
        "delivery_pct_min": 30.0,
        "delivery_vol_ratio_min": 1.2,
    },
    "F7_PEAD": {
        "name": "PEAD Catalyst",
        "emoji": "⚡",
        "surprise_min": 10.0,
        "days_since_earnings_max": 30,
    },
    "F8_FII_DII_CONVICTION": {
        "name": "FII/DII Smart Money",
        "emoji": "🏦",
        "fii_change_min": 0.1,
        "dii_change_min": 0.1,
    }
}

# ═══════════════════════════════════════════════════════════
# Thematic & Sector Config (for MCP server compatibility)
# ═══════════════════════════════════════════════════════════

_SECTORS_REVERSE = {
    "Technology": ["ASTRAMICRO", "AVALON", "SYRMA", "NEOGEN", "DATAPATTNS"],
    "Industrials": ["APARINDS", "RRKABEL", "MTARTECH", "ABB", "CGPOWER", "POWERINDIA", "GVT&D", "KIRLOSENG", "THERMAX", "SCHNEIDER"],
    "Financial Services": ["CGCL", "BSE", "ANGELONE", "J&KBANK", "KTKBANK", "BANDHANBNK"],
    "Consumer Cyclical": ["THANGAMAYL", "SKYGOLD"],
    "Consumer Defensive": ["CUPID"],
    "Basic Materials": ["HINDALCO", "HINDCOPPER", "SAIL", "NLCINDIA", "SOLARINDS"]
}

_THEMES_REVERSE = {
    "Power & Infrastructure": ["APARINDS", "RRKABEL", "ABB", "CGPOWER", "POWERINDIA", "GVT&D", "THERMAX", "SCHNEIDER", "POLYCAB", "FINOXCABLES"],
    "Defense & Aerospace": ["ASTRAMICRO", "MTARTECH", "DATAPATTNS"],
    "Electronics Manufacturing": ["SYRMA", "AVALON", "NETWEB"],
    "NBFC & Finance": ["CGCL", "BSE", "ANGELONE"],
    "Regional Banking": ["J&KBANK", "KTKBANK", "BANDHANBNK"],
    "Jewellery & Retail": ["THANGAMAYL"],
    "Commodities & Metals": ["HINDALCO", "HINDCOPPER", "SAIL", "NLCINDIA"],
    "Healthcare & Pharma": ["LAURUSLABS", "SHILPAMED", "WOCKPHARMA", "GRANULES"],
    "Consumer & FMCG": ["CUPID", "HONASA"]
}

class DoubleLookupDict(dict):
    def __init__(self, forward_map, reverse_map):
        super().__init__()
        self.update(reverse_map)
        self._forward = forward_map
        self._reverse = reverse_map
        for category, symbols in reverse_map.items():
            for symbol in symbols:
                symbol_upper = symbol.strip().upper()
                if symbol_upper not in self._forward:
                    self._forward[symbol_upper] = category

    def get(self, key, default=None):
        if not isinstance(key, str):
            return default
        ukey = key.strip().upper()
        if ukey in self._forward:
            return self._forward[ukey]
        if key in self._reverse:
            return self._reverse[key]
        return super().get(key, default)

    def __getitem__(self, key):
        if not isinstance(key, str):
            raise KeyError(key)
        ukey = key.strip().upper()
        if ukey in self._forward:
            return self._forward[ukey]
        if key in self._reverse:
            return self._reverse[key]
        return super().__getitem__(key)

    def __contains__(self, key):
        if not isinstance(key, str):
            return False
        ukey = key.strip().upper()
        if ukey in self._forward:
            return True
        if key in self._reverse:
            return True
        return super().__contains__(key)

SECTORS = DoubleLookupDict(_SECTORS_FORWARD, _SECTORS_REVERSE)
THEMES = DoubleLookupDict(_THEMES_FORWARD, _THEMES_REVERSE)

TRACK_THRESHOLDS = {
    "T1": {"min_score": 30, "name": "Cyclical Recovery"},
    "T2": {"min_score": 30, "name": "High Growth"},
    "T3": {"min_score": 20, "name": "PEAD Catalyst"},
    "T4": {"min_score": 35, "name": "Sustained Momentum"},
    "T5": {"min_score": 20, "name": "Smart Money"},
    "T6": {"min_score": 20, "name": "Delivery Accumulation"},
    "T7": {"min_score": 20, "name": "Emerging Recovery"}
}

# (Drawdown thresholds defined above as decimals: -0.08, -0.12, -0.18 — do NOT redefine as integers)


# New Entry Thresholds
NEW_BUY_MIN_SCORE = 70.0
NEW_BUY_MAX_RANK = 40
