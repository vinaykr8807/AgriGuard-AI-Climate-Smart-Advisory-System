"""
=============================================================
AgriGuard AI — Domain-Specific Prompt Engine
=============================================================
Each domain has:
  1. classify_domain()       — maps user question → domain key
  2. build_domain_context()  — injects live + historical data as pre-computed facts
  3. get_domain_system_prompt() — the system role Groq sees
  4. get_domain_user_prompt()   — the full user message Groq sees
  5. get_domain_visuals()    — list of charts to render in UI
  6. get_score_breakdown()   — dict of scores for the score-breakdown panel

Domains:
  A. soil_nutrient          — Soil & Fertilizer (Mitti aur Khad)
  B. money_market           — Money & Market (Munafa aur Mandi)
  C. pest_disease           — Pests & Sickness (Keeda aur Bimari)
  D. future_strategy        — The Future (Aage ki Soch)
  E. soil_health_mgmt       — Soil Health & Nutrient Management (technical)
  F. climate_resilience     — Climate Resilience & Risk Mitigation
  G. economic_yield         — Economic & Yield Optimization
  H. pest_diagnostic        — Pest, Disease & Diagnostic Analysis (technical)
  I. longterm_sustainability — Long-term Strategy & Sustainability
=============================================================
"""

from __future__ import annotations
from typing import Dict, Optional, List, Any
import pandas as pd
import numpy as np


# ─────────────────────────────────────────────────────────────
# 1.  DOMAIN CLASSIFIER
# ─────────────────────────────────────────────────────────────

_DOMAIN_KEYWORDS: Dict[str, List[str]] = {
    # ── A: Soil & Fertilizer (conversational / farmer-language) ──
    "soil_nutrient": [
        "white fertilizer", "urea", "locked up", "suck more strength",
        "khad", "fertilizer", "manure",
        "soil locked", "nutrient lock", "weak plants", "small plants",
        "natural manure", "melt into ground", "hard land",
        "bone dry", "pump", "wilt", "drip", "irrigation hours",
        "sulfur", "salty", "white soil", "soil health report",
        "soil score",
    ],
    # ── B: Money & Market ──
    "money_market": [
        "msp", "mandi", "price", "profit", "munafa", "savings", "cash",
        "bumper harvest", "money back", "roi", "investment", "sell",
        "market", "rate", "income", "rupees", "$", "500$", "40000",
        "potash roi", "extra dose", "cost", "spend",
    ],
    # ── C: Pest & Disease (farmer language) ──
    "pest_disease": [
        "keeda", "bug", "pest", "worm", "rot", "black spots", "sticky",
        "bimari", "disease", "spray", "cloudy", "humidity warm", "leaves browning",
        "roots", "fungal", "2019 drought farmers", "save crops", "insects",
        "no sun", "cloudy for days",
    ],
    # ── D: Future Strategy (farmer language) ──
    "future_strategy": [
        "children", "10 years", "trees", "long-term plants", "hotter every year",
        "aage", "future", "summer earlier", "father used to", "prakritik kheti",
        "natural farming", "wells empty", "almost no water", "drought-resistant",
        "40000 rupees", "better seeds", "tool that tells", "save my farm",
    ],
    # ── E: Soil Health & Nutrient Management (technical queries) ──
    "soil_health_mgmt": [
        "ndvi declining", "nutrient lock-in", "adjust p and k", "p and k ratio",
        "ph 8.2", "alkaline", "buffer ph", "soil health score", "45/100",
        "organic fertilizers decompose", "soil temperature", "compare nutrient depletion",
        "nitrogen trends", "5 years of nitrogen", "eto", "evapotranspiration",
        "wilting point", "drip irrigation hours", "soil moisture 15%",
        "high nitrogen", "historical ndvi", "ndvi for rice", "is there nutrient lock",
        "npk", "nitrogen is high", "lock-in issue", "p and k", "n is high",
    ],
    # ── F: Climate Resilience & Risk Mitigation ──
    "climate_resilience": [
        "historical average rainfall", "7-day forecast", "delay sowing",
        "short-duration variety", "current ndvi", "vhi", "heat stress",
        "42°c heatwave", "foliar spray", "uv index", "shade-net", "mulching strategy",
        "flood every 3 years", "flood risk score", "precipitation trend",
        "night temperature", "grain-filling", "historical mean",
    ],
    # ── G: Economic & Yield Optimization ──
    "economic_yield": [
        "msp increasing", "erratic rainfall", "pivot to", "drought-resistant crop",
        "calculate roi", "potash roi", "extra dose of potash", "msp trend",
        "k levels medium", "historical yield patterns", "above district average",
        "mixed cropping plan", "financial gain", "high msp", "low water requirement",
        "price-to-weather correlation", "safe investment",
    ],
    # ── H: Pest, Disease & Diagnostic Analysis (technical) ──
    "pest_diagnostic": [
        "humidity 85%", "soil temp 28", "fungal blight", "preventive measures",
        "ndvi dropped", "0.7 to 0.4", "10 days", "root-knot nematode",
        "nutrient deficiency", "most common pest", "paddy", "rainfall below 100mm",
        "control strategy", "solar radiation", "pest lifecycle", "pesticide effectiveness",
        "multi-year expert advisory", "pest management 2019",
    ],
    # ── I: Long-term Strategy & Sustainability ──
    "longterm_sustainability": [
        "10-year temperature trend", "0.2°c/year", "perennial crops", "profitable in 2030",
        "historical ndvi", "brown wave", "cropping calendar", "shift by 15 days",
        "climate-smart score", "zbnf", "zero budget natural farming",
        "groundwater level", "satellite moisture", "8%", "200mm seasonal",
        "synthesize all data", "better seeds vs fertilizer", "solar-powered irrigation",
    ],
}

# Priority order — more specific domains checked first
_DOMAIN_PRIORITY = [
    "soil_health_mgmt",
    "pest_diagnostic",
    "climate_resilience",
    "economic_yield",
    "longterm_sustainability",
    "soil_nutrient",
    "money_market",
    "pest_disease",
    "future_strategy",
]


def classify_domain(question: str) -> str:
    """Return the best-matching domain key for a user question."""
    q = question.lower()
    scores: Dict[str, int] = {d: 0 for d in _DOMAIN_PRIORITY}
    for domain in _DOMAIN_PRIORITY:
        for kw in _DOMAIN_KEYWORDS[domain]:
            if kw in q:
                scores[domain] += 1
    best = max(scores, key=lambda d: scores[d])
    if scores[best] == 0:
        return "general"
    return best


# ─────────────────────────────────────────────────────────────
# 2.  PRE-COMPUTED DATA INJECTOR  (district-filtered)
# ─────────────────────────────────────────────────────────────

def build_domain_context(
    domain: str,
    district: str,
    state: str,
    crop: str,
    nitrogen: float,
    phosphorus: float,
    potassium: float,
    ph: float,
    rainfall: float,
    temperature: float,
    agri_metrics: Dict[str, Any],
    crop_matches_df,          # historical DataFrame for this district+crop
    advisory_df,              # full advisory DataFrame
    season: str = "All",      # ← NEW: "Kharif" | "Rabi" | "Zaid" | "Annual" | "Perennial" | "All"
) -> str:
    """
    Returns a pre-computed facts block (plain text) that is inserted
    verbatim into the Groq prompt. All numbers come from real API/CSV data.
    """
    ndvi = agri_metrics.get("ndvi") or 0.0
    soil_moisture = agri_metrics.get("soil_moisture") or 0.0
    et0 = agri_metrics.get("et0") or 0.0
    precip_7day = agri_metrics.get("precip_7day") or 0.0
    soil_temp = agri_metrics.get("soil_temp") or temperature

    season_label = season if season != "All" else "All seasons"

    lines: List[str] = [
        f"=== DISTRICT-SPECIFIC GROUND TRUTH: {district.upper()}, {state.upper()} ===",
        f"Crop Under Analysis : {crop}",
        f"Planting Season     : {season_label}",    # ← season injected here
        f"Soil Nutrients      : N={nitrogen:.1f} kg/ha | P={phosphorus:.1f} kg/ha | K={potassium:.1f} kg/ha | pH={ph:.2f}",
        f"Live Weather        : Temp={temperature:.1f}°C | Rainfall(10yr avg)={rainfall:.0f}mm",
        f"Satellite (Live)    : NDVI={ndvi:.3f} | Soil Moisture={soil_moisture:.1f}% | Soil Temp={soil_temp:.1f}°C",
        f"Evapotranspiration  : ET₀={et0:.2f} mm/day",
        f"7-Day Forecast Rain : {precip_7day:.1f} mm",
    ]

    # ── Historical trend extraction ──────────────────────────────
    if crop_matches_df is not None and not crop_matches_df.empty:
        df = crop_matches_df.copy()
        lines.append(f"\n--- HISTORICAL DATA ({district}) ---")
        lines.append(f"Records available   : {len(df)} years")

        if "Year" in df.columns:
            ymin = int(pd.to_numeric(df["Year"], errors="coerce").min())
            ymax = int(pd.to_numeric(df["Year"], errors="coerce").max())
            lines.append(f"Data range          : {ymin} – {ymax}")

        # Rainfall trend
        if "Rainfall_IMD_mm" in df.columns:
            r = pd.to_numeric(df["Rainfall_IMD_mm"], errors="coerce").dropna()
            if len(r) >= 2:
                trend = r.iloc[-1] - r.iloc[0]
                lines.append(
                    f"Rainfall Trend      : {r.iloc[0]:.0f}mm → {r.iloc[-1]:.0f}mm "
                    f"({'📈 +' if trend > 0 else '📉 '}{abs(trend):.0f}mm over period)"
                )

        # Temperature trend
        if "Mean_Temp_Historical" in df.columns:
            t = pd.to_numeric(df["Mean_Temp_Historical"], errors="coerce").dropna()
            if len(t) >= 2:
                trend = t.iloc[-1] - t.iloc[0]
                lines.append(
                    f"Temp Trend          : {t.iloc[0]:.1f}°C → {t.iloc[-1]:.1f}°C "
                    f"({'📈 +' if trend > 0 else '📉 '}{abs(trend):.1f}°C)"
                )

        # NDVI trend
        if "NDVI_Vegetation_Index" in df.columns:
            n = pd.to_numeric(df["NDVI_Vegetation_Index"], errors="coerce").dropna()
            if len(n) >= 2:
                trend = n.iloc[-1] - n.iloc[0]
                lines.append(
                    f"NDVI Trend          : {n.iloc[0]:.3f} → {n.iloc[-1]:.3f} "
                    f"({'Improving ✅' if trend > 0 else 'DECLINING ⚠️'})"
                )

        # MSP trend
        if "Historical_MSP_INR" in df.columns:
            m = pd.to_numeric(df["Historical_MSP_INR"], errors="coerce").dropna()
            if len(m) >= 2:
                pct = ((m.iloc[-1] - m.iloc[0]) / m.iloc[0]) * 100
                lines.append(
                    f"MSP Trend           : ₹{m.iloc[0]:.0f} → ₹{m.iloc[-1]:.0f} "
                    f"({'+' if pct > 0 else ''}{pct:.1f}% over period)"
                )

        # NPK mode (last 5 years)
        recent5 = df.tail(5)
        for nutrient in ["Nitrogen", "Phosphorus", "Potassium"]:
            if nutrient in recent5.columns:
                mode_series = recent5[nutrient].mode()
                if not mode_series.empty:
                    lines.append(f"Typical {nutrient[:1]} demand (last 5yr): {mode_series.iloc[0]}")

    # ── Domain-specific computed fact injection ──────────────────
    if domain == "soil_nutrient":
        lines += _soil_nutrient_facts(nitrogen, phosphorus, potassium, ph, soil_moisture, et0, temperature)

    elif domain == "soil_health_mgmt":
        lines += _soil_health_mgmt_facts(nitrogen, phosphorus, potassium, ph, ndvi, soil_moisture, et0, precip_7day, crop_matches_df)

    elif domain == "money_market":
        lines += _money_market_facts(crop_matches_df, rainfall, temperature, crop)

    elif domain == "economic_yield":
        lines += _economic_yield_facts(crop_matches_df, nitrogen, phosphorus, potassium, ndvi, district, crop)

    elif domain == "pest_disease":
        lines += _pest_disease_facts(agri_metrics, temperature, crop)

    elif domain == "pest_diagnostic":
        lines += _pest_diagnostic_facts(ndvi, soil_moisture, temperature, agri_metrics, crop_matches_df, crop)

    elif domain == "climate_resilience":
        lines += _climate_resilience_facts(precip_7day, rainfall, temperature, ndvi, soil_moisture, et0, crop_matches_df)

    elif domain == "future_strategy":
        lines += _future_strategy_facts(crop_matches_df, temperature, rainfall, soil_moisture)

    elif domain == "longterm_sustainability":
        lines += _longterm_sustainability_facts(crop_matches_df, temperature, rainfall, ndvi, soil_moisture, district)

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# 3.  DOMAIN-SPECIFIC FACT BUILDERS  (called by build_domain_context)
# ─────────────────────────────────────────────────────────────

def _soil_nutrient_facts(n, p, k, ph, soil_moisture, et0, temperature) -> List[str]:
    lines = ["\n--- SOIL & NUTRIENT COMPUTED FACTS ---"]

    # Nutrient lock-up detection
    if n > 120 and p < 35:
        lines.append(f"⚠️  NUTRIENT LOCK-UP DETECTED: N={n:.0f} is high but P={p:.0f} is low → phosphorus deficiency blocks nitrogen uptake")
        lines.append(f"    → Actual NPK ratio: {n:.0f}:{p:.0f}:{k:.0f}  |  Target ratio for most crops: 4:2:1")
        lines.append(f"    → Action: Apply 50 kg/ha DAP + 30 kg/ha MOP IMMEDIATELY, STOP all Urea for 21 days")
    elif n < 40:
        lines.append(f"⚠️  LOW NITROGEN: {n:.0f} kg/ha (optimal: 60–120 kg/ha) → apply Urea 50 kg/ha or FYM 5t/ha")
    else:
        lines.append(f"✅  Nitrogen level: {n:.0f} kg/ha (acceptable)")

    # pH advisory
    if ph > 8.0:
        lines.append(f"⚠️  ALKALINE SOIL (pH {ph:.2f}) → nutrient availability blocked. Apply Gypsum 500 kg/ha + Sulfur 50 kg/ha")
    elif ph < 5.5:
        lines.append(f"⚠️  ACIDIC SOIL (pH {ph:.2f}) → Apply Agricultural Lime 2.5 t/ha to raise pH to 6.0–7.0")
    else:
        lines.append(f"✅  Soil pH {ph:.2f} is in acceptable range (5.5–7.5)")

    # Irrigation urgency
    if soil_moisture < 15:
        deficit_mm = (25 - soil_moisture) * 3.0
        pump_hours = deficit_mm / 2.5
        lines.append(f"🚨  CRITICAL DROUGHT STRESS: Soil moisture {soil_moisture:.1f}% < 15% wilting point")
        lines.append(f"    → Water deficit: {deficit_mm:.0f}mm | Run drip pump: {pump_hours:.1f} hours (at 2.5 mm/hr)")
        lines.append(f"    → ET₀ = {et0:.2f} mm/day meaning crop loses this much water daily — irrigate WITHIN 2 HOURS")
    elif soil_moisture < 25:
        lines.append(f"⚡  Soil moisture {soil_moisture:.1f}% — moderate stress; schedule irrigation within 24 hours")

    # Salt / alkaline soil + rain buffering
    if ph > 7.8:
        lines.append(f"    Rain buffering: 40mm rain lowers pH by only 0.10–0.15 units → rain ALONE will NOT fix pH {ph:.2f}")

    return lines


def _soil_health_mgmt_facts(n, p, k, ph, ndvi, soil_moisture, et0, precip_7day, df) -> List[str]:
    lines = ["\n--- SOIL HEALTH MGMT COMPUTED FACTS ---"]

    # N:P:K ratio
    if p > 0 and k > 0:
        ratio = f"{n/p:.1f}:1:{k/p:.1f}"
        lines.append(f"Current NPK ratio   : N:P:K = {ratio}  (ideal ≈ 4:2:1 for most crops)")
    if n > 100 and p < 30:
        lines.append(f"⚠️  NUTRIENT IMBALANCE: Excess N ({n:.0f}) with low P ({p:.0f}) → classic lock-in scenario")
        lines.append(f"    Adjust: Reduce Urea, add DAP 50 kg/ha. Target P: 40–60 kg/ha")

    # NDVI trend vs fertilizer
    if df is not None and not df.empty and "NDVI_Vegetation_Index" in df.columns:
        ndvi_series = pd.to_numeric(df["NDVI_Vegetation_Index"], errors="coerce").dropna()
        if len(ndvi_series) >= 3:
            trend5 = ndvi_series.iloc[-1] - ndvi_series.iloc[-min(5, len(ndvi_series))]
            if trend5 < -0.05:
                lines.append(f"⚠️  NDVI DECLINING TREND: {ndvi_series.iloc[-min(5,len(ndvi_series))]:.3f} → {ndvi_series.iloc[-1]:.3f} ({trend5:+.3f})")
                lines.append(f"    Vegetation declining despite fertilizer = nutrient lock-in or soil compaction")

    # pH buffering with rainfall
    if ph > 7.5:
        ph_drop = (precip_7day / 100) * 0.25
        lines.append(f"pH buffering by rain : {precip_7day:.0f}mm forecast → pH drop ≈ {ph_drop:.2f} units (from {ph:.2f} → {ph - ph_drop:.2f})")
        if ph - ph_drop > 7.5:
            lines.append(f"    VERDICT: Rain insufficient — still need Sulfur 50 kg/ha or Gypsum 500 kg/ha")
        else:
            lines.append(f"    VERDICT: Rain will partially buffer. Reduce Sulfur dose by 30%.")

    # Drip irrigation calculation
    if soil_moisture > 0 and et0 > 0:
        deficit = max(0, 25 - soil_moisture)
        root_depth = 300  # mm assumed (cotton / standard)
        volume_needed = (deficit / 100) * root_depth  # mm of water
        pump_hours = volume_needed / 2.5
        lines.append(f"Irrigation calc     : Soil moisture {soil_moisture:.1f}%, field capacity 25%, root depth 300mm")
        lines.append(f"    Water deficit   : {volume_needed:.0f}mm | At 2.5mm/hr drip → RUN PUMP {pump_hours:.1f} HOURS")
        lines.append(f"    ET₀ loss rate   : {et0:.2f} mm/day — recharge every {max(1, int(volume_needed/et0))} days")

    return lines


def _money_market_facts(df, rainfall, temperature, crop) -> List[str]:
    lines = ["\n--- MONEY & MARKET COMPUTED FACTS ---"]

    if df is None or df.empty:
        lines.append("No historical market data found for this district+crop combination.")
        return lines

    if "Historical_MSP_INR" in df.columns:
        msp = pd.to_numeric(df["Historical_MSP_INR"], errors="coerce").dropna()
        if len(msp) >= 2:
            pct_change = ((msp.iloc[-1] - msp.iloc[0]) / msp.iloc[0]) * 100
            lines.append(f"MSP 5-year change   : ₹{msp.iloc[0]:.0f} → ₹{msp.iloc[-1]:.0f} ({'+' if pct_change>=0 else ''}{pct_change:.1f}%)")
            if pct_change > 15:
                lines.append(f"✅  Strong MSP growth (>{15}%) — good investment signal for {crop}")
            elif pct_change > 0:
                lines.append(f"⚡  Moderate MSP growth — proceed with caution, compare input costs")
            else:
                lines.append(f"⚠️  MSP declining or flat — high financial risk for {crop} this season")

            latest_msp = msp.iloc[-1]
            # Simple ROI estimate
            typical_yield = {"Rice": 35, "Wheat": 40, "Maize": 30, "Cotton": 18, "Soybean": 20,
                             "Sugarcane": 700, "Potato": 200, "Onion": 180}.get(crop, 25)
            typical_input = 25000  # ₹/ha average
            gross = latest_msp * typical_yield
            roi = ((gross - typical_input) / typical_input) * 100
            lines.append(f"Estimated ROI       : MSP ₹{latest_msp:.0f} × Typical yield {typical_yield}q/ha = ₹{gross:.0f}/ha gross")
            lines.append(f"    Input cost ~₹{typical_input}/ha → Estimated ROI ≈ {roi:.0f}%")

    if "Rainfall_IMD_mm" in df.columns:
        r = pd.to_numeric(df["Rainfall_IMD_mm"], errors="coerce").dropna()
        cv = (r.std() / r.mean() * 100) if r.mean() > 0 else 0
        lines.append(f"Rainfall variability: CV = {cv:.1f}% ({'High risk — erratic rains' if cv > 30 else 'Acceptable variability'})")

    return lines


def _economic_yield_facts(df, n, p, k, ndvi, district, crop) -> List[str]:
    lines = ["\n--- ECONOMIC & YIELD OPTIMIZATION COMPUTED FACTS ---"]

    if df is None or df.empty:
        lines.append("No district-specific yield optimization data available.")
        return lines

    # ROI of extra Potash
    if k < 100:
        k_deficit = 100 - k
        k_cost_per_kg = 18  # ₹/kg approximate MOP cost
        k_dose_ha = k_deficit * 0.5  # simplified
        extra_cost = k_dose_ha * k_cost_per_kg
        yield_gain_pct = min(15, k_deficit * 0.1)  # 0.1% yield gain per kg deficit
        lines.append(f"Potash ROI estimate : Current K={k:.0f} (below optimal 100 kg/ha)")
        lines.append(f"    Extra K needed  : {k_dose_ha:.0f} kg/ha MOP → Cost ≈ ₹{extra_cost:.0f}/ha")
        lines.append(f"    Expected yield boost: +{yield_gain_pct:.1f}% → net positive if crop value >₹{extra_cost/yield_gain_pct*100:.0f}/ha")

    # NDVI-based yield prediction
    if ndvi > 0:
        district_avg_ndvi = 0.45  # assumed baseline; override with df if available
        if df is not None and "NDVI_Vegetation_Index" in df.columns:
            hist_ndvi = pd.to_numeric(df["NDVI_Vegetation_Index"], errors="coerce").dropna()
            if not hist_ndvi.empty:
                district_avg_ndvi = hist_ndvi.mean()

        ndvi_pct = (ndvi / district_avg_ndvi - 1) * 100
        if ndvi_pct > 10:
            lines.append(f"Yield prediction    : Current NDVI {ndvi:.3f} vs district avg {district_avg_ndvi:.3f} → ABOVE average ({ndvi_pct:+.1f}%)")
            lines.append(f"    Forecast: ABOVE-AVERAGE or BUMPER harvest likely")
        elif ndvi_pct < -15:
            lines.append(f"Yield prediction    : Current NDVI {ndvi:.3f} BELOW district avg {district_avg_ndvi:.3f} ({ndvi_pct:+.1f}%)")
            lines.append(f"    Forecast: BELOW-AVERAGE harvest — investigate stress causes")
        else:
            lines.append(f"Yield prediction    : NDVI close to district average → Typical harvest expected")

    # Mixed cropping suggestion
    high_msp_crops = ["Wheat", "Cotton", "Soybean", "Onion"]
    low_water_crops = ["Pearl Millet", "Chickpea", "Sorghum", "Sesame"]
    lines.append(f"Mixed cropping hint : High MSP crop = {high_msp_crops[0]} | Water-safe backup = {low_water_crops[0]}")
    lines.append(f"    Suggested split : 60% {high_msp_crops[0]} + 40% {low_water_crops[0]} for balanced risk/reward")

    return lines


def _pest_disease_facts(agri_metrics, temperature, crop) -> List[str]:
    lines = ["\n--- PEST & DISEASE RISK COMPUTED FACTS ---"]

    humidity = agri_metrics.get("humidity") or agri_metrics.get("soil_moisture") or 60.0
    soil_temp = agri_metrics.get("soil_temp") or temperature

    # Fungal blight risk
    if humidity > 75 and 22 < soil_temp < 32:
        lines.append(f"🚨  FUNGAL BLIGHT RISK: HIGH (Humidity={humidity:.0f}%, Soil Temp={soil_temp:.1f}°C)")
        lines.append(f"    Peak conditions for Alternaria, Powdery Mildew, Downy Mildew")
        lines.append(f"    ACTION: Spray Mancozeb 2g/L OR Copper Oxychloride 3g/L PREVENTIVELY (before symptoms)")
        lines.append(f"    Re-spray every 7 days if conditions persist")
    elif humidity > 60:
        lines.append(f"⚡  Moderate fungal risk (Humidity={humidity:.0f}%) — monitor closely, preventive spray advisable")
    else:
        lines.append(f"✅  Low fungal risk at current humidity={humidity:.0f}%")

    # Pest pressure by season / dryness
    precip_7day = agri_metrics.get("precip_7day") or 0.0
    if precip_7day < 20 and temperature > 30:
        lines.append(f"⚠️  DRY + HOT conditions: High risk of Stem Borer (Rice/Wheat), Bollworm (Cotton), Aphids (Mustard)")
        lines.append(f"    Control: Chlorantraniliprole 0.4mL/L spray | Organic: Neem oil 5mL/L")
    if humidity > 80:
        lines.append(f"⚠️  High humidity promotes leaf hoppers and whitefly — apply Imidacloprid 0.5mL/L if detected")

    # Spray timing
    lines.append(f"Spray timing rule   : Spray works best with 4–6 hours dry weather. Avoid if rain forecast <6 hours.")
    if precip_7day > 40:
        lines.append(f"    Rain forecast {precip_7day:.0f}mm — use systemic fungicides (absorbed before rain washes off)")

    return lines


def _pest_diagnostic_facts(ndvi, soil_moisture, temperature, agri_metrics, df, crop) -> List[str]:
    lines = ["\n--- PEST & DISEASE DIAGNOSTIC COMPUTED FACTS ---"]

    humidity = agri_metrics.get("humidity") or agri_metrics.get("soil_moisture") or 60.0
    soil_temp = agri_metrics.get("soil_temp") or temperature

    # Rapid NDVI drop diagnosis
    if ndvi < 0.35:
        lines.append(f"⚠️  NDVI={ndvi:.3f} (POOR vegetation health)")
        if soil_moisture > 25:
            lines.append(f"    Soil moisture is ADEQUATE ({soil_moisture:.1f}%) but NDVI is low → NOT water stress")
            lines.append(f"    DIAGNOSIS: Suspect root-knot nematode OR nutrient deficiency OR soil-borne fungal disease")
            lines.append(f"    → Check roots: Knotted/galled roots = Nematode (treat: Carbofuran 3G, 15 kg/ha)")
            lines.append(f"    → Yellow interveinal leaves = Micronutrient (treat: ZnSO4 25 kg/ha + FeSO4 20 kg/ha)")
        else:
            lines.append(f"    Soil moisture is LOW ({soil_moisture:.1f}%) + low NDVI = Water stress is PRIMARY cause")

    # Fungal blight quantified risk
    if humidity > 80 and 24 < soil_temp < 30:
        lines.append(f"🚨  Fungal Blight Risk Score: VERY HIGH")
        lines.append(f"    Humidity={humidity:.0f}%, Soil Temp={soil_temp:.1f}°C — textbook conditions for Phytophthora/Fusarium")
        lines.append(f"    Preventive spray: Metalaxyl-M 4g/L + Mancozeb 2g/L before symptoms appear")
    
    # District pest history from data
    if df is not None and not df.empty:
        rainfall_years = pd.to_numeric(df.get("Rainfall_IMD_mm", pd.Series()), errors="coerce")
        dry_years = (rainfall_years < 100).sum() if not rainfall_years.empty else 0
        if dry_years > 2:
            lines.append(f"District dry-year pest history: {dry_years} dry years in record — Stem Borer historically dominant in low-rain years")
            lines.append(f"    Control: BPMC 50 EC 2mL/L spray at tillering + boot leaf stage")

    # Pesticide effectiveness under clouds
    precip_7day = agri_metrics.get("precip_7day") or 0.0
    lines.append(f"Cloud/low-radiation impact: Cloudy days reduce UV-based pesticide breakdown → fungicide lasts longer BUT insecticide absorption decreases 15–20%")
    lines.append(f"    For cloudy weather: Use systemic pesticides over contact pesticides")

    return lines


def _climate_resilience_facts(precip_7day, rainfall_hist, temperature, ndvi, soil_moisture, et0, df) -> List[str]:
    lines = ["\n--- CLIMATE RESILIENCE COMPUTED FACTS ---"]

    # Rainfall deficit
    historical_july_avg = rainfall_hist if rainfall_hist > 0 else 200
    deficit_pct = max(0, (historical_july_avg - precip_7day) / historical_july_avg * 100)
    lines.append(f"Rainfall comparison : Historical avg={historical_july_avg:.0f}mm | 7-day forecast={precip_7day:.1f}mm")
    lines.append(f"    Rainfall deficit: {deficit_pct:.0f}%  ({'SEVERE — switch crops' if deficit_pct > 70 else 'MODERATE — delayed sowing acceptable' if deficit_pct > 30 else 'Normal'})")

    # Flood risk
    flood_prob_per_year = 33  # assumed every 3 years = 33%
    if df is not None and not df.empty and "Rainfall_IMD_mm" in df.columns:
        r = pd.to_numeric(df["Rainfall_IMD_mm"], errors="coerce").dropna()
        flood_threshold = r.quantile(0.85) if len(r) >= 5 else historical_july_avg * 1.5
        flood_risk_pct = min(95, (precip_7day / flood_threshold) * flood_prob_per_year * 2)
    else:
        flood_risk_pct = min(85, deficit_pct * 0.3 if precip_7day > 150 else 10)

    lines.append(f"Flood risk score    : {flood_risk_pct:.0f}%  ({'🚨 HIGH' if flood_risk_pct > 60 else '⚡ MODERATE' if flood_risk_pct > 30 else '✅ LOW'})")
    if flood_risk_pct > 60:
        lines.append(f"    EMERGENCY: Harvest immediately if crop is >75% mature, dig drainage channels 30cm deep")

    # Heat stress
    if temperature >= 40:
        lines.append(f"🚨  EXTREME HEAT STRESS: {temperature:.1f}°C — crop tolerance exceeded")
        lines.append(f"    Kaolin clay spray: 50g/L × 500L/ha → reduces leaf temp 5–7°C | Cost ₹1,200/ha")
        lines.append(f"    Apply BEFORE 9 AM tomorrow")
    elif temperature >= 35:
        lines.append(f"⚡  Heat stress warning: {temperature:.1f}°C — apply antitranspirant spray (Vapor Gard 5L/ha)")

    # Night temp wheat grain filling
    night_temp_est = temperature - 8
    if night_temp_est > 15:
        loss_pct = (night_temp_est - 15) * 5
        lines.append(f"Night temp estimate : {night_temp_est:.0f}°C (ideal <15°C for grain filling)")
        lines.append(f"    Expected yield loss from warm nights: {loss_pct:.0f}%")
        lines.append(f"    Mitigation: Salicylic acid 100ppm + late irrigation (8–10 PM) reduces loss by 30–40%")

    # Short-duration crop recommendations for drought
    if deficit_pct > 70:
        lines.append(f"Crop switch options : Pearl Millet (75 days, 350mm) | Green Gram (60 days, 300mm) | Sorghum (90 days, 450mm)")

    return lines


def _future_strategy_facts(df, temperature, rainfall, soil_moisture) -> List[str]:
    lines = ["\n--- FUTURE STRATEGY COMPUTED FACTS ---"]

    # 10-year temperature trend
    if df is not None and not df.empty and "Mean_Temp_Historical" in df.columns:
        t = pd.to_numeric(df["Mean_Temp_Historical"], errors="coerce").dropna()
        if len(t) >= 5:
            annual_increase = (t.iloc[-1] - t.iloc[0]) / max(1, len(t) - 1)
            proj_10yr = t.iloc[-1] + annual_increase * 10
            lines.append(f"Temperature trend   : +{annual_increase:.2f}°C/year → By 2034 = {proj_10yr:.1f}°C")
            if annual_increase > 0.15:
                lines.append(f"⚠️  Significant warming trend — traditional crops will face increasing heat stress")

    # Rainfall trend
    if df is not None and not df.empty and "Rainfall_IMD_mm" in df.columns:
        r = pd.to_numeric(df["Rainfall_IMD_mm"], errors="coerce").dropna()
        if len(r) >= 5:
            annual_change = (r.iloc[-1] - r.iloc[0]) / max(1, len(r) - 1)
            lines.append(f"Rainfall trend      : {'+' if annual_change>=0 else ''}{annual_change:.1f}mm/year")
            if annual_change < -5:
                lines.append(f"⚠️  Declining rainfall trend — water-intensive crops increasingly risky")

    # Cropping calendar shift (NDVI brown wave)
    if df is not None and not df.empty and "NDVI_Vegetation_Index" in df.columns:
        n_series = pd.to_numeric(df["NDVI_Vegetation_Index"], errors="coerce").dropna()
        if len(n_series) >= 5:
            early_avg = n_series.iloc[:3].mean()
            late_avg = n_series.iloc[-3:].mean()
            ndvi_decline = late_avg - early_avg
            if ndvi_decline < -0.05:
                lines.append(f"Brown wave signal   : Early NDVI {early_avg:.3f} → Recent {late_avg:.3f} → Calendar shift of 10–15 days advisable")

    # Ultra-low water crops
    lines.append(f"Ultra-low water crops: Pearl Millet (250mm) | Sorghum (300mm) | Chickpea (350mm) | Sesame (300mm)")
    lines.append(f"    These use 50–70% less water than Rice (1,200mm) or Sugarcane (2,000mm)")

    # Investment priority ₹40,000
    lines.append(f"₹40,000 investment guide:")
    lines.append(f"    1. Soil test       : ₹500  (know exact deficiency before buying fertilizer)")
    lines.append(f"    2. Drip kit        : ₹15,000 (saves 40% water, ROI in 1 season)")
    lines.append(f"    3. Quality seeds   : ₹8,000 (heat-tolerant/drought-resistant varieties)")
    lines.append(f"    4. Balanced khad   : ₹12,000 (DAP+MOP based on soil test)")
    lines.append(f"    5. Soil sensor     : ₹4,500 (prevents over/under-irrigation)")
    lines.append(f"    Expected combined ROI: 150–200% in first season")

    # Natural farming comparison
    lines.append(f"Natural farming vs conventional: ZBNF reduces input cost ₹8,000–12,000/ha but initial yield may drop 20–30%. Soil moisture retention +15–20% under ZBNF after 2 years.")

    return lines


def _longterm_sustainability_facts(df, temperature, rainfall, ndvi, soil_moisture, district) -> List[str]:
    lines = ["\n--- LONG-TERM SUSTAINABILITY COMPUTED FACTS ---"]

    if df is not None and not df.empty:
        if "Mean_Temp_Historical" in df.columns:
            t = pd.to_numeric(df["Mean_Temp_Historical"], errors="coerce").dropna()
            if len(t) >= 5:
                rate = (t.iloc[-1] - t.iloc[0]) / max(1, len(t) - 1)
                temp_2030 = t.iloc[-1] + rate * (2030 - 2024)
                lines.append(f"District warming    : +{rate:.2f}°C/year | Projected 2030 temp: {temp_2030:.1f}°C")
                if rate > 0.2:
                    lines.append(f"⚠️  At this rate, {district} will see +1.2°C by 2030 — MAJOR crop stress threshold")

        if "Rainfall_IMD_mm" in df.columns:
            r = pd.to_numeric(df["Rainfall_IMD_mm"], errors="coerce").dropna()
            if len(r) >= 5:
                rate = (r.iloc[-1] - r.iloc[0]) / max(1, len(r) - 1)
                rain_2030 = r.iloc[-1] + rate * (2030 - 2024)
                lines.append(f"Rainfall projection : {'+' if rate>=0 else ''}{rate:.1f}mm/year | 2030 est: {rain_2030:.0f}mm")

    # Perennial crops for 2030 profitability
    lines.append(f"Perennial crops for 2030 profitability:")
    lines.append(f"    1. Drumstick (Moringa) — drought-tolerant, high demand, 5–7yr lifespan")
    lines.append(f"    2. Amla (Indian Gooseberry) — climate-resilient, pharmaceutical demand growing")
    lines.append(f"    3. Bamboo — carbon credits + construction market, needs minimal water")
    lines.append(f"    4. Teak/Eucalyptus — 10-year timber, good for degraded land")
    lines.append(f"    5. Turmeric/Ginger — high MSP, drought-adaptive, 2-year cycle")

    # ZBNF vs conventional climate-smart score
    lines.append(f"Climate-Smart Score comparison:")
    lines.append(f"    Conventional farming : Input-dependent, moderate water use, good short-term yield")
    lines.append(f"    ZBNF/Natural Farming : Lower input, 15–20% better water retention, 20–30% lower initial yield")
    lines.append(f"    Climate-smart winner after 3 years: ZBNF (soil organic carbon builds resilience)")

    # Groundwater / ultra-low water
    if soil_moisture < 12:
        lines.append(f"🚨  Live soil moisture {soil_moisture:.1f}% — at ALL-TIME LOW level for this location")
    lines.append(f"Crops surviving <200mm seasonal rainfall: Pearl Millet | Sorghum | Chickpea | Sesame | Moth Bean")

    # Investment synthesis
    lines.append(f"$500 investment synthesis:")
    lines.append(f"    Best ROI: Soil test (₹500) → drip kit (₹15k) → quality seeds (₹8k)")
    lines.append(f"    Reason: Drip alone reduces water cost 40% AND reduces fertilizer leaching 25%")
    lines.append(f"    Do NOT buy more fertilizer before soil test — risks over-application and lock-in")

    return lines


# ─────────────────────────────────────────────────────────────
# 4.  SYSTEM PROMPTS  (the "role" Groq is given per domain)
# ─────────────────────────────────────────────────────────────

def get_domain_system_prompt(domain: str, district: str, state: str, crop: str,
                              nitrogen: float, phosphorus: float, potassium: float,
                              ph: float, rainfall: float, temperature: float,
                              season: str = "All") -> str:
    """
    Returns the Groq system prompt tailored to the domain.
    Each domain has a distinct expert persona, output format rules,
    and mandatory data-use instructions.
    """
    season_note = (
        f"The farmer is in the {season} season. All sowing dates, pesticide timing, "
        f"fertiliser schedules, and water requirements MUST reflect {season} conditions. "
        if season != "All"
        else ""
    )
    base_rule = (
        f"You are advising a farmer in {district}, {state}, India growing {crop}. "
        f"Ground truth: pH={ph:.2f}, N={nitrogen:.0f}, P={phosphorus:.0f}, K={potassium:.0f} kg/ha, "
        f"Rainfall={rainfall:.0f}mm, Temp={temperature:.1f}°C. "
        f"{season_note}"
        f"ALWAYS use these exact numbers. NEVER say 'consider' — say 'DO THIS'. "
        f"Give product names, exact kg/ha doses, timing, and costs in Indian Rupees (₹)."
    )

    domain_personas = {

        "soil_nutrient": (
            "You are an Indian SOIL DOCTOR specialising in NPK therapy for field crops. "
            "Your patients are crops with nutrient deficiency or toxicity. "
            "Diagnose exactly what is wrong using the NPK numbers provided. "
            "Write prescriptions like a doctor: product name, dose, application method, timing. "
            "Structure: 🔬 Diagnosis → 💊 Immediate Treatment (7 days) → 🌿 Organic Alternative → 📅 30-day Recovery Plan → ⚠️ Risk Alert."
        ),

        "money_market": (
            "You are an AGRICULTURAL ECONOMIST and commodity market analyst for Indian farmers. "
            "Your job is to calculate financial returns, analyse MSP trends, and compare investment options. "
            "Always state ROI as a percentage. Compare risk vs reward explicitly. "
            "Structure: 💰 MSP & Price Analysis → 📈 5-Year Market Trend → 🧮 ROI Calculation → 🎯 Investment Verdict → ⚠️ Financial Risk Alert."
        ),

        "pest_disease": (
            "You are a PLANT PROTECTION OFFICER and crop pathologist serving Indian villages. "
            "Identify the exact pest or disease from the environmental conditions given. "
            "Give both chemical and organic control options with exact dosages. "
            "Structure: 🐛 Pest/Disease Identification → 🌡️ Risk Conditions → 💉 Chemical Control → 🌿 Organic Control → 📅 Spray Schedule → ⚠️ Spread Risk."
        ),

        "future_strategy": (
            "You are a CLIMATE-ADAPTIVE FARMING STRATEGIST helping Indian farmers plan for the next 10 years. "
            "Use temperature and rainfall trends from the data to make concrete, timeline-bound recommendations. "
            "Structure: 🌡️ Climate Trend (Next 10 Years) → 🌳 Climate-Resilient Crops to Plant NOW → 📅 Cropping Calendar Adjustment → 💧 Water-Saving Technologies → 💵 Investment Priority (₹40,000 budget) → 🌿 Natural vs Conventional Farming Verdict."
        ),

        "soil_health_mgmt": (
            "You are a SENIOR SOIL SCIENTIST with expertise in nutrient cycling, pH buffering, and precision fertiliser management. "
            "All recommendations must be calculation-backed using the exact NPK, NDVI, ET₀, and pH figures provided. "
            "Structure: 🔬 Nutrient Imbalance Diagnosis → 📊 NPK Ratio Analysis → 🧪 pH Buffering Assessment → 💧 Irrigation Calculation (exact hours) → 💊 Corrective Treatment Schedule → 📈 Expected NDVI Recovery."
        ),

        "climate_resilience": (
            "You are a CLIMATE RISK ANALYST and crop resilience advisor for Indian agriculture. "
            "Quantify every risk as a percentage score. Compare forecast data with historical district averages. "
            "Structure: ⛈️ Climate Risk Score (0–100) → 🌧️ Rainfall Deficit/Surplus Analysis → 🌡️ Heat & Drought Stress → 🌊 Flood Risk Assessment → 🛡️ Mitigation Strategy (with exact products and timeline) → 📅 7-Day Action Plan."
        ),

        "economic_yield": (
            "You are an AGRICULTURAL INVESTMENT ADVISOR combining yield science with market intelligence for Indian farmers. "
            "Calculate ROI, yield predictions, and risk-adjusted returns using real data. "
            "Structure: 📈 MSP Trend Analysis → 🌾 NDVI-Based Yield Forecast → 🧮 ROI Calculation (exact ₹) → 🌱 Mixed Cropping Plan → ⚖️ Risk-Reward Verdict → 📅 Seasonal Investment Timeline."
        ),

        "pest_diagnostic": (
            "You are a CROP DIAGNOSTICS SPECIALIST and IPM (Integrated Pest Management) expert. "
            "Diagnose root cause of crop decline using NDVI, soil moisture, humidity, and temperature data. "
            "Structure: 🔍 Symptom Analysis → 🧬 Root Cause Diagnosis (pest vs disease vs nutrient) → 💊 Targeted Treatment Protocol → 🌿 Organic IPM Alternative → 📅 Monitoring Schedule → ⚠️ Secondary Infection Risk."
        ),

        "longterm_sustainability": (
            "You are a SUSTAINABLE FARMING FUTURIST advising Indian farmers on 10-year climate adaptation and soil health. "
            "Use historical temperature/rainfall trends to project 2030 conditions. Compare ZBNF vs conventional farming. "
            "Structure: 🌡️ 2030 Climate Projection for this District → 🌳 Perennial Crops for Long-Term Income → 📅 Cropping Calendar Shift → 🌿 ZBNF vs Conventional Climate-Smart Score → 💧 Water Scarcity Crops → 💵 Optimal $500 Investment Plan."
        ),

        "general": (
            "You are an expert agricultural advisor for Indian farmers. "
            "Give specific, actionable advice using the data provided. "
            "Always mention exact product names, doses in kg/ha, and costs in ₹. "
            "Structure: 🔬 Situation Analysis → 🛠️ Immediate Actions → 🌿 Organic Alternative → 📅 Monthly Calendar → ⚠️ Risk Alert."
        ),
    }

    persona = domain_personas.get(domain, domain_personas["general"])
    return f"{persona}\n\n{base_rule}"


# ─────────────────────────────────────────────────────────────
# 5.  USER PROMPTS  (the full message Groq processes per domain)
# ─────────────────────────────────────────────────────────────

def get_domain_user_prompt(
    domain: str,
    question: str,
    domain_context: str,   # output of build_domain_context()
    district: str,
    state: str,
    crop: str,
    nitrogen: float,
    phosphorus: float,
    potassium: float,
    ph: float,
    rainfall: float,
    temperature: float,
    crop_suitability_score: float,
    climate_risk_score: float,
    agri_metrics: Dict[str, Any],
    expert_advisory: str = "",
    live_weather_context: str = "",
) -> str:
    """
    Assembles the full user-turn message sent to Groq.
    Domain-specific response FORMAT is embedded so Groq MUST follow it.
    """
    ndvi = agri_metrics.get("ndvi") or "N/A"
    soil_moisture = agri_metrics.get("soil_moisture") or "N/A"
    et0 = agri_metrics.get("et0") or "N/A"
    precip_7day = agri_metrics.get("precip_7day") or "N/A"

    header = f"""
╔══════════════════════════════════════════════════════════════╗
  AGRIGUARD AI — DOMAIN: {domain.upper().replace('_',' ')}
  District: {district}, {state}  |  Crop: {crop}
╚══════════════════════════════════════════════════════════════╝

═══ FARMER'S QUESTION ═══
"{question}"

═══ REAL-TIME DATA FOR {district.upper()}, {state.upper()} ═══
• Soil   : N={nitrogen:.1f} | P={phosphorus:.1f} | K={potassium:.1f} kg/ha | pH={ph:.2f}
• Weather: Temp={temperature:.1f}°C | Rainfall(avg)={rainfall:.0f}mm
• Satellite Live: NDVI={ndvi} | Soil Moisture={soil_moisture}% | ET₀={et0}mm/day
• 7-Day Forecast Rain: {precip_7day}mm
• ML Scores: Crop Suitability={crop_suitability_score:.1f}% | Climate Risk={climate_risk_score:.1f}%

═══ PRE-COMPUTED DISTRICT ANALYSIS ═══
{domain_context}

═══ 10-YEAR EXPERT ADVISORY (2015–2024, {district}) ═══
{expert_advisory if expert_advisory else "No multi-year advisory on record — use general best practices for this region."}

═══ LIVE WEATHER CONDITIONS ═══
{live_weather_context if live_weather_context else "Live weather data unavailable."}
"""

    # Domain-specific response FORMAT instructions
    format_blocks = {

        "soil_nutrient": """
═══ MANDATORY RESPONSE FORMAT ═══
You MUST produce all of the following sections. Use exact numbers from the data above.

### 🔬 Diagnosis
- State exact NPK levels: N={N}, P={P}, K={K}, pH={pH}
- Identify the problem clearly (lock-up / deficiency / toxicity / saline soil)
- One-sentence root cause

### 💊 Immediate Treatment (Next 7 Days)
1. [Product name] — [exact kg/ha or L/ha] — [apply WHEN and HOW]
2. [Product name] — [exact kg/ha] — [timing]
3. [Action] — [details]
(Include ₹ cost per hectare for each item)

### 🌿 Organic / Low-Cost Alternative
- [Method 1] — [quantity] — [cost ₹/ha]
- [Method 2] — [quantity] — [cost ₹/ha]

### 💧 Irrigation Prescription (if relevant)
- Soil moisture status: [value]% vs wilting point 15%
- Run drip pump: [X] hours (calculated from ET₀ and deficit)

### 📅 30-Day Recovery Calendar
Week 1: [Action]
Week 2: [Action]
Week 3: [Action]
Week 4: [Monitor + measure]

### ⚠️ Risk Alert for {DISTRICT}
- If untreated: [exact yield loss %] within [days]
- DEADLINE: [Do THIS by DATE/TIME]
""",

        "money_market": """
═══ MANDATORY RESPONSE FORMAT ═══

### 💰 MSP & Price Situation
- Current MSP for {CROP}: ₹[value]/quintal
- 5-year MSP trend: ₹[start] → ₹[end] ([+/-X%])
- Market verdict: [Favourable / Cautious / Avoid]

### 🧮 ROI Calculation for {DISTRICT}
- Expected yield: [X] quintals/ha (based on NDVI and district data)
- Gross revenue: ₹[MSP] × [yield] = ₹[total]/ha
- Input cost estimate: ₹[amount]/ha
- **Net profit: ₹[amount]/ha | ROI: [X]%**

### 📈 5-Year Market Intelligence
[State whether MSP is rising, flat, or falling — use exact numbers from data]
[Compare with rainfall variability risk — a crop is only profitable if you can grow it]

### 🌧️ Climate-Market Risk Overlay
- Rainfall CV for this district: [X]% ([High/Low] variability)
- Verdict on {CROP} in current climate: [Safe / Risky / Switch recommended]

### 🎯 Investment Decision
**DO THIS:**
1. [Specific action]
2. [Specific action]
**AVOID:** [What NOT to do and why]

### ⚠️ Financial Risk Alert
- Worst-case scenario: [what happens if rain fails]
- Hedge: [Suggest backup crop or crop insurance]
""",

        "pest_disease": """
═══ MANDATORY RESPONSE FORMAT ═══

### 🐛 Pest / Disease Identification
- Condition: Humidity=[X]%, Temp=[Y]°C, Rainfall=[Z]mm
- Most likely threat: [Specific pest/disease name]
- Confidence: [High/Medium] based on [conditions]

### 🌡️ Why NOW (Risk Conditions)
[Explain in 2–3 lines why current temp/humidity creates this specific risk]
[Reference district history if available from expert advisory]

### 💉 Chemical Control (Fastest Result)
1. [Product] — [dose mL or g per L water] — Spray [WHEN]
2. [Product] — [dose] — Frequency: every [X] days
Cost: ₹[amount]/ha | Re-spray: [conditions]

### 🌿 Organic Control (Low Cost)
1. [Organic method] — [dosage] — [application]
2. [Organic method] — [dosage] — [application]
Cost: ₹[amount]/ha

### 📅 Spray Schedule
Day 0: [Action — do immediately]
Day 7: [Action]
Day 14: [Check + re-spray if needed]

### ⚠️ Spread Risk Alert
- If untreated in [X] days: [consequence]
- Neighbouring crops at risk: [Yes/No — and why]
""",

        "future_strategy": """
═══ MANDATORY RESPONSE FORMAT ═══

### 🌡️ Climate Trend for Your District (10-Year Outlook)
- Temperature change rate: [+X°C/year] → by 2034: [projected temp]°C
- Rainfall change rate: [+/-Xmm/year] → by 2034: [projected]mm
- Summary: [What this means for farming in practical terms]

### 🌳 Crops to Plant TODAY for 2030+ Profit
| Crop | Water Need | Start Yielding | MSP Trend | Climate Fit |
|------|-----------|---------------|-----------|-------------|
| [Crop 1] | [mm] | [years] | [trend] | [score/10] |
| [Crop 2] | [mm] | [years] | [trend] | [score/10] |
| [Crop 3] | [mm] | [years] | [trend] | [score/10] |

### 📅 Cropping Calendar Adjustment
- Historical sowing: [month]
- Recommended new window: [X days earlier/later]
- Reason: [Data-backed explanation]

### 💵 ₹40,000 Investment Priority
1. [Item] — ₹[amount] — [ROI reason]
2. [Item] — ₹[amount] — [ROI reason]
3. [Item] — ₹[amount] — [ROI reason]
Expected ROI: [X]% in first season

### 🌿 Natural Farming vs Conventional — Verdict for YOUR Conditions
[State which wins in current climate conditions of {DISTRICT} and WHY, with specific numbers]

### ⚠️ Climate Risk Alert
[One critical thing that will happen if no action taken — with timeline]
""",

        "soil_health_mgmt": """
═══ MANDATORY RESPONSE FORMAT ═══

### 🔬 Nutrient Imbalance Diagnosis
- N={N} kg/ha | P={P} kg/ha | K={K} kg/ha | pH={pH}
- NPK ratio: [current] vs ideal [target]
- Problem: [Exactly what is wrong and why]
- NDVI status: [current value] — [interpretation]

### 📊 Calculation-Backed Analysis
[For NUTRIENT LOCK-IN: Show the math — why high N blocks P at this ratio]
[For pH: Calculate exactly how much amendment is needed to reach target pH]
[For IRRIGATION: Show: deficit_mm = (25% - current%) × root_depth; hours = deficit / 2.5]

### 🧪 pH Buffering Assessment
- Current pH: [value] | Target: [value]
- 7-day forecast rain: [X]mm
- Rain will lower pH by: [calculated value] units
- **Verdict: Still need [gypsum/sulfur] because [reason]**

### 💊 Corrective Treatment Schedule
Immediate (Today):
1. [Product] — [kg/ha] — [method]
Week 2:
2. [Product] — [kg/ha] — [method]
Month 2:
3. [Product] — [kg/ha] — [method]

### 📈 Expected Recovery
- NDVI recovery: [X.XXX] → [X.XXX] in [weeks]
- pH correction: [current] → [target] in [days]
- Soil moisture target: [X]% (via [hours] irrigation)
""",

        "climate_resilience": """
═══ MANDATORY RESPONSE FORMAT ═══

### ⛈️ Climate Risk Score for {DISTRICT}
**Overall Risk: [XX]% — [LOW / MODERATE / HIGH / CRITICAL]**
| Risk Factor | Score | Status |
|-------------|-------|--------|
| Flood Risk | [X]% | [status] |
| Drought Risk | [X]% | [status] |
| Heat Stress | [X]% | [status] |
| Night Temp (Grain) | [X]% | [status] |

### 🌧️ Rainfall Deficit/Surplus Analysis
- Historical district avg: [X]mm | 7-day forecast: [Y]mm
- Deficit: [Z]% [above/below] normal
- Implication: [Delay sowing / Switch crop / Proceed / Harvest now]

### 🌡️ Heat & Drought Intervention
[If temp > 38°C: Kaolin clay spray prescription with exact dose]
[If soil moisture < 15%: Exact pump hours calculation]

### 🌊 Flood Risk Assessment
- District flood frequency: every [X] years ([Y]% annual probability)
- Current 7-day forecast vs flood threshold: [mm vs mm]
- **Flood probability: [Z]%**
[If high: Specific drainage/harvesting instructions]

### 🛡️ 7-Day Action Plan
Day 1: [Specific action]
Day 2–3: [Specific action]
Day 4–7: [Specific action]
""",

        "economic_yield": """
═══ MANDATORY RESPONSE FORMAT ═══

### 📈 MSP Trend Analysis for {CROP} in {DISTRICT}
- 5-year MSP: ₹[start] → ₹[end] ([trend %])
- Current MSP: ₹[value]/quintal
- Trend assessment: [Strong buy / Hold / Avoid]

### 🌾 NDVI-Based Yield Forecast
- Current NDVI: [value] | District average NDVI: [value]
- Yield forecast: [X]% [above/below] district average
- Expected harvest: [X] quintals/ha

### 🧮 ROI Calculation
Gross revenue   : ₹[MSP] × [yield q/ha] = ₹[total]/ha
Input costs     : ₹[amount]/ha  
Net profit      : ₹[amount]/ha  
**ROI           : [X]%**
Break-even yield: [X] quintals/ha

### 🌱 Mixed Cropping Plan for 2 Hectares
- Hectare 1 (1.2 ha): [Main crop] — High MSP, [water need]mm
- Hectare 2 (0.8 ha): [Insurance crop] — Drought-safe, [water need]mm
- Risk reduction: 40% lower total loss probability

### ⚖️ Risk-Reward Verdict
[Clear verdict: Proceed with {CROP} / Switch to [alternative] / Hedge with mixed]
[One critical condition that could invalidate this advice]
""",

        "pest_diagnostic": """
═══ MANDATORY RESPONSE FORMAT ═══

### 🔍 Symptom & Data Analysis
- NDVI: [current] — [interpretation: healthy / stressed / declining]
- Soil moisture: [value]% — [status vs crop need]
- Humidity: [value]% | Soil temp: [value]°C
- Key anomaly: [What the data is telling us — e.g., "moisture OK but NDVI dropped 43% in 10 days"]

### 🧬 Root Cause Diagnosis
**Primary suspect: [Pest name / Disease name / Nutrient deficiency]**
- Evidence: [Why these conditions point to this diagnosis]
- Confidence: [High/Medium/Low]

**Rule out:**
- [Alternative 1]: [Why less likely]
- [Alternative 2]: [Why less likely]

### 💊 Targeted Treatment Protocol
Chemical (fastest):
1. [Product] — [dose mL or g/L] — Apply [when/how]
   Cost: ₹[X]/ha | Expected result in [days]

### 🌿 Organic IPM Alternative
1. [Organic treatment] — [dose] — [method]
2. [Biological control] if available: [product/method]

### 📅 Monitoring Schedule
Day 0: Apply treatment + photograph affected area
Day 3: Check for [specific sign of recovery]
Day 7: Re-inspect. If [condition], re-spray.
Day 14: NDVI re-assessment (should show [X] improvement)

### ⚠️ Secondary Infection Risk
[What could spread if primary infection isn't controlled]
[Neighbouring crop risk and buffer zone recommendation]
""",

        "longterm_sustainability": """
═══ MANDATORY RESPONSE FORMAT ═══

### 🌡️ 2030 Climate Projection for {DISTRICT}
- Temperature trajectory: +[X]°C/year → 2030 estimate: [Y]°C
- Rainfall trajectory: [+/-X]mm/year → 2030 estimate: [Y]mm
- Impact: [Which traditional crops will become unviable]

### 🌳 Perennial Crops to Plant This Year for 2030+ Returns
| Crop | Water (mm/yr) | Years to First Yield | Market Outlook | Climate Score |
|------|--------------|----------------------|----------------|---------------|
| [Crop 1] | [X] | [Y] | [outlook] | [/10] |
| [Crop 2] | [X] | [Y] | [outlook] | [/10] |
| [Crop 3] | [X] | [Y] | [outlook] | [/10] |

### 📅 Cropping Calendar Shift Recommendation
- Current calendar vs data-suggested shift: [+/- X days]
- Reason: [Temperature onset data / NDVI brown wave analysis]

### 🌿 ZBNF vs Conventional Farming — Climate-Smart Score
| Criteria | ZBNF | Conventional |
|----------|------|--------------|
| Water retention | +15–20% | Baseline |
| Input cost | ₹[X]/ha less | ₹[Y]/ha |
| Yield (Year 1) | -20–30% | Baseline |
| Soil health (Year 3+) | +40% organic C | Declining |
| Climate resilience | High | Medium |

### 💧 Water Scarcity Crop Shortlist
[3 crops that survive <200mm seasonal water — with yield and MSP data]

### 💵 Optimal $500 / ₹40,000 Investment Plan
[Ranked allocation with ROI justification for each line item]

### ⚠️ 10-Year Risk Alert
[The single biggest threat to this farm's sustainability and the ONE action to take NOW]
""",

        "general": """
═══ MANDATORY RESPONSE FORMAT ═══

### 🔬 Situation Analysis
[2 lines: What is happening and why, using the exact numbers from data]

### 🛠️ Immediate Actions (Do This Now)
1. [Product/Action] — [quantity] — [timing] — ₹[cost]
2. [Product/Action] — [quantity] — [timing] — ₹[cost]
3. [Product/Action] — [quantity] — [timing] — ₹[cost]

### 🌿 Organic / Low-Cost Alternative
[2–3 natural methods with dosages and costs]

### 📅 Monthly Calendar
Month 1: [Actions]
Month 2: [Actions]
Month 3: [Check/adjust]

### ⚠️ Risk Alert for {DISTRICT}
[Consequence of inaction + DEADLINE]
""",
    }

    # Fill in the format placeholders (static substitution — Groq fills [X] dynamic parts)
    fmt = format_blocks.get(domain, format_blocks["general"])
    fmt = fmt.replace("{N}", f"{nitrogen:.1f}").replace("{P}", f"{phosphorus:.1f}") \
             .replace("{K}", f"{potassium:.1f}").replace("{pH}", f"{ph:.2f}") \
             .replace("{DISTRICT}", district).replace("{CROP}", crop) \
             .replace("{STATE}", state)

    return f"{header}\n{fmt}\n\nNOW RESPOND TO THE FARMER'S QUESTION USING EVERY SECTION ABOVE. Use data from '{district}, {state}' only. Be a doctor, not a professor."


# ─────────────────────────────────────────────────────────────
# 6.  VISUAL ANALYTICS SPECIFICATION
# ─────────────────────────────────────────────────────────────

def get_domain_visuals(domain: str) -> List[str]:
    """
    Returns list of chart keys to render for this domain.
    The UI layer maps these keys to actual Plotly figure builders.
    """
    domain_charts = {
        "soil_nutrient":          ["npk_bar", "ph_gauge", "soil_moisture_gauge", "irrigation_deficit"],
        "soil_health_mgmt":       ["npk_radar", "ndvi_trend", "ph_trend", "irrigation_schedule"],
        "money_market":           ["msp_trend", "rainfall_variability", "roi_bar"],
        "economic_yield":         ["msp_trend", "ndvi_vs_yield", "mixed_crop_pie", "roi_waterfall"],
        "pest_disease":           ["humidity_temp_risk", "spray_timing_calendar"],
        "pest_diagnostic":        ["ndvi_decline_trend", "humidity_temp_risk", "soil_moisture_gauge"],
        "climate_resilience":     ["rainfall_forecast_vs_hist", "flood_risk_gauge", "heat_stress_timeline", "7day_precip_bar"],
        "future_strategy":        ["temp_trend_10yr", "rainfall_trend_10yr", "crop_water_comparison", "investment_allocation_pie"],
        "longterm_sustainability": ["temp_trend_10yr", "rainfall_trend_10yr", "ndvi_trend", "zbnf_vs_conventional_radar"],
        "general":                ["npk_bar", "ph_gauge"],
    }
    return domain_charts.get(domain, ["npk_bar", "ph_gauge"])


# ─────────────────────────────────────────────────────────────
# 7.  SCORE BREAKDOWN  (domain-specific scoring for UI panel)
# ─────────────────────────────────────────────────────────────

def get_score_breakdown(
    domain: str,
    nitrogen: float,
    phosphorus: float,
    potassium: float,
    ph: float,
    rainfall: float,
    temperature: float,
    ndvi: float,
    soil_moisture: float,
    et0: float,
    precip_7day: float,
    crop: str,
    crop_matches_df,
) -> Dict[str, Any]:
    """
    Returns a dict of named scores (0–100) relevant to the domain.
    Used by the UI to show a colour-coded breakdown panel.
    """

    def clamp(v):
        return round(min(100, max(0, v)), 1)

    # ── Shared base scores ────────────────────────────────────
    # pH score
    ph_score = 100 - abs(ph - 6.5) * 20
    ph_score = clamp(ph_score)

    # Nitrogen balance
    n_score = clamp(100 - abs(nitrogen - 80) * 0.8)

    # P balance
    p_score = clamp(100 - abs(phosphorus - 40) * 1.2)

    # K balance
    k_score = clamp(100 - abs(potassium - 120) * 0.5)

    # NDVI health
    ndvi_score = clamp(ndvi * 130) if ndvi else 50

    # Soil moisture
    if soil_moisture < 15:
        moisture_score = clamp(soil_moisture * 4)
    elif soil_moisture > 45:
        moisture_score = clamp(100 - (soil_moisture - 45) * 3)
    else:
        moisture_score = clamp(80 + (soil_moisture - 15) * 0.67)

    # Temperature score
    if 20 <= temperature <= 32:
        temp_score = 90
    elif 15 <= temperature <= 38:
        temp_score = 65
    else:
        temp_score = clamp(40 - abs(temperature - 26) * 2)

    # Rainfall vs crop need
    crop_rain_need = {"Rice": 1200, "Wheat": 500, "Cotton": 600, "Maize": 600,
                      "Soybean": 500, "Sugarcane": 1500, "Potato": 500, "Onion": 400}.get(crop, 600)
    rain_score = clamp(100 - abs(rainfall - crop_rain_need) / crop_rain_need * 100)

    # MSP trend score
    msp_score = 50  # neutral default
    if crop_matches_df is not None and not crop_matches_df.empty and "Historical_MSP_INR" in crop_matches_df.columns:
        m = pd.to_numeric(crop_matches_df["Historical_MSP_INR"], errors="coerce").dropna()
        if len(m) >= 2:
            pct = ((m.iloc[-1] - m.iloc[0]) / m.iloc[0]) * 100
            msp_score = clamp(50 + pct * 1.5)

    # Flood / drought risk (inverse)
    if precip_7day > 200:
        flood_score = clamp(100 - (precip_7day - 200) * 0.3)
    elif precip_7day < 10 and rainfall < 400:
        flood_score = clamp(precip_7day * 3)
    else:
        flood_score = 70

    # ── Domain-specific breakdown dicts ──────────────────────
    breakdowns = {
        "soil_nutrient": {
            "Nitrogen Balance": n_score,
            "Phosphorus Balance": p_score,
            "Potassium Balance": k_score,
            "Soil pH": ph_score,
            "Soil Moisture": moisture_score,
            "Overall Soil Health": clamp((n_score + p_score + k_score + ph_score + moisture_score) / 5),
        },
        "soil_health_mgmt": {
            "Nitrogen": n_score,
            "Phosphorus": p_score,
            "Potassium": k_score,
            "pH Suitability": ph_score,
            "NDVI Health": ndvi_score,
            "Moisture Adequacy": moisture_score,
            "Soil Health Score": clamp((n_score + p_score + k_score + ph_score + ndvi_score + moisture_score) / 6),
        },
        "money_market": {
            "MSP Growth Trend": msp_score,
            "Rainfall Reliability": rain_score,
            "Crop Suitability": clamp((n_score + rain_score + temp_score) / 3),
            "Financial Viability": clamp((msp_score + rain_score + ndvi_score) / 3),
            "Investment Risk": clamp(100 - ((abs(rainfall - crop_rain_need) / crop_rain_need) * 50 + abs(ph - 6.5) * 10)),
        },
        "economic_yield": {
            "MSP Trend": msp_score,
            "NDVI Yield Signal": ndvi_score,
            "Soil Fertility": clamp((n_score + p_score + k_score) / 3),
            "Climate Suitability": clamp((rain_score + temp_score) / 2),
            "ROI Potential": clamp((msp_score + ndvi_score + rain_score) / 3),
        },
        "pest_disease": {
            "Fungal Risk": clamp(100 - moisture_score * 0.3 - temp_score * 0.4),
            "Pest Pressure": clamp(100 - rain_score * 0.5 - temp_score * 0.5),
            "Spray Efficacy": clamp(70 + (100 - moisture_score) * 0.3),
            "Crop Vulnerability": clamp(100 - ndvi_score * 0.6),
            "Overall Disease Risk": clamp(100 - (moisture_score + temp_score + ndvi_score) / 3),
        },
        "pest_diagnostic": {
            "NDVI Health": ndvi_score,
            "Root Zone Moisture": moisture_score,
            "Pathogen Conditions": clamp(100 - (temp_score * 0.4 + moisture_score * 0.6)),
            "Nutrient Adequacy": clamp((n_score + p_score + k_score) / 3),
            "Diagnosis Confidence": clamp(70 + ndvi_score * 0.1 + moisture_score * 0.2),
        },
        "climate_resilience": {
            "Flood Risk (inverse)": flood_score,
            "Drought Risk (inverse)": clamp(moisture_score * 0.7 + rain_score * 0.3),
            "Heat Stress (inverse)": temp_score,
            "Soil Buffer Capacity": ph_score,
            "Overall Climate Safety": clamp((flood_score + temp_score + rain_score + moisture_score) / 4),
        },
        "future_strategy": {
            "Current Soil Health": clamp((n_score + p_score + k_score + ph_score) / 4),
            "Water Security": moisture_score,
            "Climate Trend Risk": temp_score,
            "Crop Adaptability": clamp((rain_score + ndvi_score) / 2),
            "Long-term Viability": clamp((ph_score + ndvi_score + rain_score + temp_score) / 4),
        },
        "longterm_sustainability": {
            "Soil Carbon Potential": clamp((ph_score + ndvi_score) / 2),
            "Water Sustainability": clamp((moisture_score + rain_score) / 2),
            "Climate Resilience Score": clamp((temp_score + flood_score) / 2),
            "Biodiversity (NDVI)": ndvi_score,
            "Sustainability Index": clamp((ph_score + ndvi_score + moisture_score + rain_score + temp_score) / 5),
        },
        "general": {
            "Soil Health": clamp((n_score + p_score + k_score + ph_score) / 4),
            "Water Availability": moisture_score,
            "Climate Suitability": clamp((rain_score + temp_score) / 2),
            "Vegetation Health": ndvi_score,
            "Overall Farm Score": clamp((n_score + p_score + k_score + ph_score + moisture_score + ndvi_score + rain_score + temp_score) / 8),
        },
    }

    result = breakdowns.get(domain, breakdowns["general"])

    # Attach colour metadata for UI
    scored = {}
    for label, value in result.items():
        if value >= 75:
            colour = "green"
            status = "Good ✅"
        elif value >= 50:
            colour = "orange"
            status = "Moderate ⚡"
        else:
            colour = "red"
            status = "Poor ⚠️"
        scored[label] = {"score": value, "colour": colour, "status": status}

    return scored


# ─────────────────────────────────────────────────────────────
# 8.  MAIN ENTRY POINT  (called from app.py)
# ─────────────────────────────────────────────────────────────

def build_groq_payload(
    question: str,
    district: str,
    state: str,
    crop: str,
    nitrogen: float,
    phosphorus: float,
    potassium: float,
    ph: float,
    rainfall: float,
    temperature: float,
    agri_metrics: Dict[str, Any],
    crop_matches_df,
    advisory_df,
    crop_suitability_score: float,
    climate_risk_score: float,
    expert_advisory: str = "",
    live_weather_context: str = "",
) -> Dict[str, Any]:
    """
    Single entry-point for app.py.
    Returns:
      {
        "domain"         : str,
        "system_prompt"  : str,
        "user_prompt"    : str,
        "visuals"        : List[str],
        "score_breakdown": Dict[str, Dict],
        "domain_context" : str,
      }
    """
    domain = classify_domain(question)

    domain_context = build_domain_context(
        domain=domain,
        district=district,
        state=state,
        crop=crop,
        nitrogen=nitrogen,
        phosphorus=phosphorus,
        potassium=potassium,
        ph=ph,
        rainfall=rainfall,
        temperature=temperature,
        agri_metrics=agri_metrics,
        crop_matches_df=crop_matches_df,
        advisory_df=advisory_df,
    )

    system_prompt = get_domain_system_prompt(
        domain=domain,
        district=district,
        state=state,
        crop=crop,
        nitrogen=nitrogen,
        phosphorus=phosphorus,
        potassium=potassium,
        ph=ph,
        rainfall=rainfall,
        temperature=temperature,
    )

    user_prompt = get_domain_user_prompt(
        domain=domain,
        question=question,
        domain_context=domain_context,
        district=district,
        state=state,
        crop=crop,
        nitrogen=nitrogen,
        phosphorus=phosphorus,
        potassium=potassium,
        ph=ph,
        rainfall=rainfall,
        temperature=temperature,
        crop_suitability_score=crop_suitability_score,
        climate_risk_score=climate_risk_score,
        agri_metrics=agri_metrics,
        expert_advisory=expert_advisory,
        live_weather_context=live_weather_context,
    )

    visuals = get_domain_visuals(domain)

    score_breakdown = get_score_breakdown(
        domain=domain,
        nitrogen=nitrogen,
        phosphorus=phosphorus,
        potassium=potassium,
        ph=ph,
        rainfall=rainfall,
        temperature=temperature,
        ndvi=agri_metrics.get("ndvi") or 0.4,
        soil_moisture=agri_metrics.get("soil_moisture") or 20.0,
        et0=agri_metrics.get("et0") or 5.0,
        precip_7day=agri_metrics.get("precip_7day") or 0.0,
        crop=crop,
        crop_matches_df=crop_matches_df,
    )

    return {
        "domain": domain,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "visuals": visuals,
        "score_breakdown": score_breakdown,
        "domain_context": domain_context,
    }
