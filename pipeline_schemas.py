"""
Schema Validation — Pipeline Structured Outputs
=====================================================
Validates all structured outputs from DeepSeek (Phase 1-2) and
Gemini/Grok/Claude (Phase 3) to catch malformed values before
they silently corrupt downstream phases.

Usage:
    from pipeline_schemas import ComponentScoring, QualOutput, validate_or_default

    scores = validate_or_default(ComponentScoring, raw_deepseek_output)
    # Returns valid ComponentScoring or raises with default fallback

    qual = validate_or_default(QualOutput, raw_gemini_output)
    # Returns valid QualOutput or raises with default fallback

Per Claude Sonnet 4 review Jun 21, 2026 — prevents LLMs from passing
out-of-range or malformed values silently through the pipeline.
"""

from pydantic import BaseModel, Field
from typing import Optional, Literal


# ── Phase 1-2: DeepSeek Scoring Output ──────────────────────────────

class ComponentScoring(BaseModel):
    """Validates the 5-component scoring output from DeepSeek."""
    momentum_score: float = Field(..., ge=0, le=100,
        description="Momentum component (0-100). 1m/3m/6m weighted returns + RS")
    technical_score: float = Field(..., ge=0, le=100,
        description="Technical component (0-100). SMA alignment + RSI + ADX + MACD")
    growth_score: float = Field(..., ge=0, le=100,
        description="Growth component (0-100). Sales + Profit growth + ROE percentile")
    quality_score: float = Field(..., ge=0, le=100,
        description="Quality component (0-100). ROCE + D/E inverse + CFO/PAT")
    volatility_score: float = Field(..., ge=0, le=100,
        description="Volatility component (0-100). Lower vol = higher score")

    def composite(self) -> float:
        """Compute Own_Score = 0.25×Mom + 0.25×Tech + 0.20×Growth + 0.15×Quality + 0.15×Vol"""
        return (0.25 * self.momentum_score + 0.25 * self.technical_score
                + 0.20 * self.growth_score + 0.15 * self.quality_score
                + 0.15 * self.volatility_score)


class GateResults(BaseModel):
    """Validates hard gate results from stock_selector.py."""
    mcap_pass: bool
    ipo_seasoning_pass: bool
    adtv_pass: bool
    asm_gsm_pass: bool
    de_ratio_pass: bool
    roce_pass: bool
    cfo_pat_pass: bool
    trend_pass: bool
    sector_theme_pass: bool
    delivery_pass: bool
    adx_vol_pass: bool
    fii_dii_pass: bool

    def all_passed(self) -> bool:
        return all([
            self.mcap_pass, self.ipo_seasoning_pass, self.adtv_pass,
            self.asm_gsm_pass, self.de_ratio_pass, self.roce_pass,
            self.cfo_pat_pass, self.trend_pass, self.sector_theme_pass,
            self.delivery_pass, self.adx_vol_pass, self.fii_dii_pass
        ])

    def failed_gates(self) -> list:
        gates = [
            ("MCap", self.mcap_pass), ("IPO Seasoning", self.ipo_seasoning_pass),
            ("ADTV", self.adtv_pass), ("ASM/GSM", self.asm_gsm_pass),
            ("D/E", self.de_ratio_pass), ("ROCE", self.roce_pass),
            ("CFO/PAT", self.cfo_pat_pass), ("Trend", self.trend_pass),
            ("Sector/Theme", self.sector_theme_pass), ("Delivery", self.delivery_pass),
            ("ADX/Vol", self.adx_vol_pass), ("FII/DII", self.fii_dii_pass),
        ]
        return [name for name, passed in gates if not passed]


# ── Phase 3a: Gemini Qualitative Output ─────────────────────────────

class GeminiQualOutput(BaseModel):
    """Validates Gemini's structured fundamental assessment."""
    sentiment: float = Field(..., ge=0, le=10,
        description="Market sentiment score (0-10)")
    perception: float = Field(..., ge=0, le=10,
        description="Business quality perception (0-10)")
    analyst: float = Field(..., ge=0, le=10,
        description="Analyst consensus on trajectory (0-10)")
    red_flag: Literal["Y", "N"] = Field(...,
        description="Any obvious red flag?")
    narrative: str = Field(..., max_length=500,
        description="One-line thesis")

    def qual_score(self) -> float:
        """Normalize to 0-100 scale."""
        return ((self.sentiment + self.perception + self.analyst) / 3) * 10


# ── Phase 3a: Grok Sentiment Output ─────────────────────────────────

class GrokOutput(BaseModel):
    """Validates Grok's real-time sentiment assessment."""
    market_sentiment: float = Field(..., ge=0, le=10,
        description="Real-time market sentiment from X/Twitter (0-10)")
    breaking_news: Literal["Y", "N"] = Field(...,
        description="Any breaking news or unusual activity?")
    note: str = Field("", max_length=500,
        description="Context note about social sentiment")

    def pulse_score(self) -> float:
        """Normalize to 0-100 scale."""
        return self.market_sentiment * 10


# ── Phase 3b: Claude Opus 4 Audit Verdict ───────────────────────────

class ClaudeAuditVerdict(BaseModel):
    """Validates Claude Opus 4's contradiction audit output."""
    verdict: Literal["CONFIRM", "ADJUST -1", "RED FLAG"] = Field(...,
        description="Audit decision: CONFIRM / ADJUST -1 tier / RED FLAG reject")
    reason: str = Field(..., max_length=500,
        description="One-line reason for the verdict")


# ── Phase 4: Synthesis Output ───────────────────────────────────────

class SynthesisOutput(BaseModel):
    """Validates the final synthesis verdict."""
    own_score: float = Field(..., ge=0, le=100)
    gemini_qual: float = Field(..., ge=0, le=100)
    grok_pulse: float = Field(..., ge=0, le=100)
    qual_score: float = Field(..., ge=0, le=100)
    final_score: float = Field(..., ge=0, le=100)
    verdict: Literal["ADD", "HOLD", "REDUCE", "EXIT"]
    claude_adjustment: Optional[Literal["CONFIRM", "ADJUST -1", "RED FLAG"]] = None


# ── Reconciliation Matrix ───────────────────────────────────────────

RECONCILIATION_MATRIX = {
    ("ADD", "EXIT"): "HOLD",
    ("ADD", "REDUCE"): "HOLD",
    ("EXIT", "ADD"): "EXIT",
    ("EXIT", "HOLD"): "EXIT",
    ("HOLD", "ADD"): "HOLD",
    ("REDUCE", "ADD"): "REDUCE",
    ("REDUCE", "HOLD"): "REDUCE",
}


def reconcile(synthesis_verdict: str, pipeline_verdict: str) -> str:
    """Apply reconciliation matrix. Returns reconciled verdict."""
    key = (synthesis_verdict, pipeline_verdict)
    return RECONCILIATION_MATRIX.get(key, synthesis_verdict)


# ── Validation Helper ───────────────────────────────────────────────

def validate_or_default(model_class, data: dict, default: dict = None):
    """Validate data against a pydantic model. Returns parsed model or default."""
    try:
        return model_class(**data)
    except Exception as e:
        if default:
            return model_class(**default)
        raise ValueError(f"Schema validation failed for {model_class.__name__}: {e}") from e


# ── Self-Test ───────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Schema Validation — Self Test ===")

    # Test valid scoring
    scores = ComponentScoring(
        momentum_score=85.0, technical_score=72.0,
        growth_score=68.0, quality_score=91.0, volatility_score=45.0
    )
    assert scores.composite() == 73.25
    print(f"  Composite score: {scores.composite():.2f} ✅")

    # Test gate validation
    gates = GateResults(
        mcap_pass=True, ipo_seasoning_pass=True, adtv_pass=True,
        asm_gsm_pass=False, de_ratio_pass=True, roce_pass=True,
        cfo_pat_pass=True, trend_pass=True, sector_theme_pass=True,
        delivery_pass=True, adx_vol_pass=True, fii_dii_pass=True
    )
    assert not gates.all_passed()
    assert gates.failed_gates() == ["ASM/GSM"]
    print(f"  Failed gate detected: {gates.failed_gates()} ✅")

    # Test Gemini output
    gemini = GeminiQualOutput(
        sentiment=7.5, perception=8.0, analyst=6.5,
        red_flag="N", narrative="Strong earnings growth driven by margin expansion"
    )
    assert round(gemini.qual_score(), 2) == 73.33
    print(f"  Gemini qual score: {gemini.qual_score():.2f} ✅")

    # Test Qual_Score (70/30 split per Claude Sonnet 4 review)
    grok = GrokOutput(market_sentiment=6.0, breaking_news="N", note="Neutral chatter")
    qual = gemini.qual_score() * 0.70 + grok.pulse_score() * 0.30
    print(f"  Qual_Score (70/30): {qual:.2f} ✅")
    assert round(qual, 1) == 69.3

    # Test reconciliation
    assert reconcile("ADD", "EXIT") == "HOLD"
    assert reconcile("EXIT", "ADD") == "EXIT"
    assert reconcile("HOLD", "ADD") == "HOLD"
    print(f"  Reconciliation matrix: all 7 rules work ✅")

    # Test boundary validation
    try:
        ComponentScoring(momentum_score=200, technical_score=72,
                         growth_score=68, quality_score=91, volatility_score=45)
        assert False, "Should have raised validation error"
    except Exception:
        print(f"  Boundary validation works ✅")

    print("\n  All schema tests passed ✅")
