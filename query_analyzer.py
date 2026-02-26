# Query Analysis Enhancement for Dynamic Agricultural Queries
# Add this to app.py to handle complex analytical queries

def analyze_query_type(question: str) -> dict:
    """Detect query type and extract required calculations"""
    q = question.lower()
    
    analysis = {
        "type": "general",
        "requires_calculation": False,
        "specific_data_needed": [],
        "calculation_hints": "",
        "response_style": "prescriptive",  # prescriptive vs descriptive
        "visual_analytics": []  # List of charts to generate
    }
    
    # Soil Health & Nutrient queries (MUST check comparison first!)
    if any(word in q for word in ["locked up", "nutrient lock", "urea but", "still looking small", "white fertilizer"]):
        analysis["type"] = "nutrient_lockup"
        analysis["requires_calculation"] = True
        analysis["specific_data_needed"] = ["historical_ndvi_trend", "npk_levels", "soil_ph", "crop_npk_ratio"]
        analysis["calculation_hints"] = "High N + low P/K = nutrient imbalance. Calculate NPK ratio. If N>150 but P<30, phosphorus deficiency blocks nitrogen uptake."
        analysis["visual_analytics"] = ["npk_balance_radar", "ndvi_5year_trend"]
        analysis["response_framework"] = """### 🔬 Diagnosis
[State exact NPK imbalance: N={X} but P={Y}, K={Z}]
[Explain: High urea (N) without P/K creates 'lock-up' - plants can't absorb nitrogen]

### 💊 Immediate Treatment (Next 7 Days)
1. **DAP (Diammonium Phosphate)** - 50 kg/ha - Apply TODAY before next irrigation
2. **Muriate of Potash (MOP)** - 30 kg/ha - Mix with DAP
3. **Stop all urea** for 21 days - Let soil rebalance

### 🌿 Organic Booster (Low-Cost)
- **Bone meal** (15 kg/ha) + **Wood ash** (10 kg/ha) - Natural P+K source
- Apply around plant base, water immediately

### ⚠️ Risk Alert
- Current NDVI: {X} (Poor vegetation health)
- If no action in 10 days: Yield loss 30-40%
- Monitor: New leaf color should improve in 14 days"""
    
    elif any(word in q for word in ["nitrogen", "ndvi declining", "npk"]):
        analysis["type"] = "soil_nutrient"
        analysis["requires_calculation"] = True
        analysis["specific_data_needed"] = ["historical_ndvi_trend", "npk_levels", "soil_ph"]
        analysis["calculation_hints"] = "Calculate NDVI trend (last 5 years), check if high N but declining NDVI = nutrient lock-in. Suggest P/K adjustment ratios."
        analysis["visual_analytics"] = ["npk_radar", "ndvi_trend"]
    
    # pH buffering / Saline soil queries
    elif any(word in q for word in ["white", "salty", "hard", "sulfur", "rain clean"]) and ("soil" in q or "land" in q):
        analysis["type"] = "saline_soil_buffering"
        analysis["requires_calculation"] = True
        analysis["specific_data_needed"] = ["soil_ph", "forecasted_rainfall", "soil_ec", "7day_rain"]
        analysis["calculation_hints"] = "White crust = high salinity (EC>4 dS/m) or alkaline pH>8.5. Heavy rain (>100mm) provides PARTIAL leaching but won't fix root cause. Calculate: 100mm rain leaches ~30% salts from top 15cm. Still need gypsum/sulfur for permanent fix."
        analysis["visual_analytics"] = ["ph_trend", "rainfall_forecast_7day"]
        analysis["response_framework"] = """### 🔬 Diagnosis
- **White crust = Saline/Alkaline soil** (pH likely >8.0, EC >4 dS/m)
- Forecasted rain: {X}mm in next 7 days
- **Verdict: Rain will provide TEMPORARY relief (30% salt leaching) but NOT permanent fix**

### 🌧️ Rain Impact Calculation
- 100mm rain = Leaches 30% salts from top 15cm only
- Your forecast: {X}mm = {X*0.3}% leaching
- **Root zone (30cm) will STILL be saline after rain**

### 💊 Permanent Solution (Do NOT skip)
1. **Gypsum (Calcium Sulfate)** - 500 kg/ha - Apply BEFORE rain
   - Rain will help gypsum penetrate deeper
2. **Sulfur powder** - 50 kg/ha - Apply AFTER rain stops
3. **Organic matter** - 5 tons FYM/ha - Improves drainage

### ⏰ Timeline
- Day 0: Apply gypsum (rain will activate it)
- Day 3-4: Let rain do its work
- Day 7: Apply sulfur + FYM
- Day 30: Retest pH (target: 7.0-7.5)

### ⚠️ Risk Alert
- If you skip gypsum: Salts will return in 15-20 days
- Current pH: {X} → Rain alone drops it to ~{X-0.2} (STILL too high)"""
    
    elif "ph" in q and ("rain" in q or "buffer" in q):
        analysis["type"] = "ph_buffering"
        analysis["requires_calculation"] = True
        analysis["specific_data_needed"] = ["soil_ph", "forecasted_rainfall", "soil_type"]
        analysis["calculation_hints"] = "40mm rain can lower pH by ~0.1-0.3 units in sandy soil, less in clay. If pH>8.0 and rain<100mm, still need sulfur."
        analysis["visual_analytics"] = ["ph_forecast"]
    
    # Soil health restoration (urgent)
    elif any(word in q for word in ["soil health report", "45 score", "not looking good", "natural manure", "melt into ground"]):
        analysis["type"] = "soil_health_emergency"
        analysis["requires_calculation"] = True
        analysis["specific_data_needed"] = ["soil_health_score", "temperature", "organic_carbon", "fast_acting_amendments"]
        analysis["calculation_hints"] = "Score <50 = critical. At 32°C, liquid organic amendments (vermicompost tea, jeevamrut) work in 7-10 days. Solid FYM takes 30+ days. Calculate: 500L vermi-tea/ha = immediate microbial boost."
        analysis["visual_analytics"] = ["soil_health_score_breakdown", "amendment_speed_comparison"]
        analysis["response_framework"] = """### 🚨 Emergency Diagnosis
- **Soil Health Score: 45/100 (CRITICAL)**
- Temperature: 32°C (Hot - need FAST-acting solutions)
- **Problem: Solid manure takes 30+ days to decompose in hot weather**
- **Your plants need help in 7-10 days MAX**

### ⚡ Fast-Acting Solutions (Works in 7-10 Days)
1. **Vermicompost Tea (Liquid)** - 500 liters/ha
   - Mix: 10kg vermicompost + 100L water + 1kg jaggery
   - Ferment 48 hours, dilute 1:5, drench soil
   - **Why fast: Liquid = immediate root absorption**

2. **Jeevamrut (Fermented microbial culture)** - 200 liters/ha
   - Recipe: 10kg cow dung + 10L cow urine + 2kg jaggery + 2kg pulse flour + handful soil
   - Ferment 7 days, apply with irrigation
   - **Works in: 5-7 days (microbial explosion)**

3. **Humic Acid (Commercial)** - 2 liters/ha
   - Spray on soil + leaves
   - **Works in: 3-5 days (chelates nutrients)**

### ❌ What NOT to Do
- **Avoid solid FYM/compost** - Takes 30-45 days in 32°C heat
- **Avoid slow-release fertilizers** - Your plants can't wait

### 📊 Speed Comparison
| Amendment | Time to Effect | Cost |
|-----------|----------------|------|
| Vermi-tea | 7-10 days | ₹500/ha |
| Jeevamrut | 5-7 days | ₹200/ha |
| Humic acid | 3-5 days | ₹800/ha |
| Solid FYM | 30-45 days | ₹2000/ha |

### ⚠️ Risk Alert
- If no action in 5 days: Plant stress irreversible
- Target: Soil score 60+ in 15 days (survival threshold)"""
    
    # Irrigation calculation queries
    elif any(word in q for word in ["bone dry", "how many hours", "water pump", "wilt"]) or "irrigation" in q or "drip" in q:
        analysis["type"] = "irrigation_emergency"
        analysis["requires_calculation"] = True
        analysis["specific_data_needed"] = ["soil_moisture", "et0", "crop_water_requirement", "wilting_point"]
        analysis["calculation_hints"] = "Wilting point = 15% soil moisture. If current=15% and ET0=6.5mm/day, deficit = (field_capacity-current) × root_depth. For cotton at 30cm depth: (25%-15%) × 300mm = 30mm deficit. At 2.5mm/hr drip rate = 12 hours."
        analysis["visual_analytics"] = ["soil_moisture_status", "irrigation_schedule"]
        analysis["response_framework"] = """### 🚨 Critical Water Stress
- **Current soil moisture: 15% (AT WILTING POINT)**
- **Crop: Cotton | Root depth: 30cm**
- **ET0: {X} mm/day (water loss rate)**
- **Status: EMERGENCY - Plants will wilt in 6-12 hours**

### 💧 Irrigation Calculation
**Water Deficit:**
- Field capacity (ideal): 25%
- Current moisture: 15%
- Deficit: 10% × 300mm depth = **30mm water needed**

**Pump Runtime:**
- Drip rate: 2.5 mm/hour (standard)
- **Required hours: 30mm ÷ 2.5mm/hr = 12 hours**

### ⏰ Application Schedule
**Option 1: Emergency (Today)**
- Run pump: 6 hours NOW + 6 hours tonight
- Why split: Avoid runoff, better absorption

**Option 2: If pump capacity limited**
- 4 hours morning + 4 hours evening × 2 days

### 🎯 Monitoring
- Check soil at 6 hours: Should feel moist at 10cm depth
- Target: Reach 22-25% moisture in 24 hours
- **Wilting reversal: 8-12 hours after first irrigation**

### ⚠️ Risk Alert
- If delayed >12 hours: Permanent yield loss 20-30%
- Cotton at flowering stage: Each day of stress = 5% yield loss
- **Action deadline: START PUMP IN NEXT 2 HOURS**"""
    
    elif "irrigation" in q or "et0" in q or "evapotranspiration" in q:
        analysis["type"] = "irrigation"
        analysis["requires_calculation"] = True
        analysis["specific_data_needed"] = ["soil_moisture", "et0", "crop_water_requirement"]
        analysis["calculation_hints"] = "Irrigation hours = (ET0 - Rainfall) / Drip rate. For cotton: 5-7mm/day. If soil moisture <15% and ET0=6.5mm/day, need ~4-6 hours drip."
        analysis["visual_analytics"] = ["irrigation_schedule"]
    
    # Weather & Climate Risk queries
    elif any(word in q for word in ["no rain", "forecast says", "wait to sow", "quick-switch", "faster crop"]):
        analysis["type"] = "drought_sowing_decision"
        analysis["requires_calculation"] = True
        analysis["specific_data_needed"] = ["7day_forecast", "historical_july_rain", "crop_duration"]
        analysis["calculation_hints"] = "If July avg=300mm but forecast=20mm, 93% deficit. Short-duration crops (60-75 days) better than long (120+ days)."
        analysis["visual_analytics"] = ["rainfall_forecast_vs_historical"]
        analysis["response_framework"] = """### 🌧️ Rainfall Deficit Analysis
- Historical July average: {X}mm
- 7-day forecast: {Y}mm
- **Deficit: {(X-Y)/X*100:.0f}% below normal**

### 🎯 Sowing Decision
**DO NOT WAIT - Act within 48 hours:**
- Monsoon delay = 15-20 days late arrival likely
- Your seeds will miss optimal window

### 🌾 Crop Switch Strategy
**AVOID (Long-duration, high water need):**
- Rice (150 days, 1200mm water)
- Sugarcane (365 days, 2000mm water)

**SWITCH TO (Short-duration, drought-tolerant):**
1. **Pearl Millet (Bajra)** - 75 days | 350mm water | Yield: 15-20 q/ha
2. **Green Gram (Moong)** - 60 days | 300mm water | MSP: ₹7,755/q
3. **Sorghum (Jowar)** - 90 days | 450mm water | Drought-hardy

### ⏰ Action Timeline
- Day 0-2: Buy seeds (pearl millet recommended)
- Day 3: Sow immediately (don't wait for rain)
- Day 7-10: First irrigation (drip/sprinkler)

### ⚠️ Risk Alert
- If you wait: 30% yield loss even if rain comes
- Short crops = 2 harvests possible vs 1 long crop"""
    
    elif any(word in q for word in ["pale", "42°c", "heat", "fire-like sun", "spray on leaves"]):
        analysis["type"] = "heat_stress_protection"
        analysis["requires_calculation"] = True
        analysis["specific_data_needed"] = ["max_temp_forecast", "crop_heat_tolerance", "antitranspirant"]
        analysis["calculation_hints"] = "Above 40°C, transpiration increases 300%. Kaolin clay or antitranspirant reduces water loss by 30-40%."
        analysis["visual_analytics"] = ["temperature_forecast_7day"]
        analysis["response_framework"] = """### 🌡️ Heat Stress Diagnosis
- Current temp: {X}°C
- Forecast peak: 42°C (EXTREME)
- Crop heat tolerance: 38°C max
- **Status: CRITICAL - 4°C above tolerance**

### 💉 Emergency Spray Treatment
**Option 1: Kaolin Clay (Surround WP)** - BEST
- Dosage: 50g/liter water
- Coverage: 500 liters/ha
- **How it works: White coating reflects 95% UV, cools leaves by 5-7°C**
- Application: Spray BEFORE 9 AM tomorrow
- Cost: ₹1,200/ha
- Re-apply after rain

**Option 2: Antitranspirant (Vapor Gard)**
- Dosage: 5 liters/ha
- **How it works: Forms breathable film, reduces water loss 40%**
- Application: Evening (4-6 PM)
- Cost: ₹800/ha
- Lasts: 3-4 weeks

**Option 3: Organic (Budget)**
- **Buttermilk spray**: 1L buttermilk + 10L water
- Reduces leaf temp by 2-3°C
- Cost: ₹50/ha

### 💧 Critical: Increase Irrigation
- At 42°C, water need doubles
- Irrigate: Early morning (5-7 AM) + Evening (6-8 PM)
- Avoid midday (wastes 60% water to evaporation)

### ⚠️ Risk Alert
- Without treatment: 40-50% yield loss
- Pale leaves = chlorophyll breakdown (irreversible after 3 days)
- **Spray within 24 hours**"""
    
    elif any(word in q for word in ["sun is biting", "high uv", "dry grass", "green net", "cover ground"]):
        analysis["type"] = "uv_mulching_decision"
        analysis["requires_calculation"] = True
        analysis["specific_data_needed"] = ["uv_index", "soil_temp", "mulch_effectiveness"]
        analysis["calculation_hints"] = "UV >10 = extreme. Dry mulch reduces soil temp 8-10°C but increases fire risk. Shade net reduces temp 5-7°C, allows rain."
        analysis["visual_analytics"] = ["soil_temp_comparison"]
        analysis["response_framework"] = """### ☀️ UV Stress Analysis
- UV Index: {X} (EXTREME if >10)
- Soil surface temp: {Y}°C
- Root damage threshold: 35°C
- **Status: Soil {Y-35:.0f}°C above safe limit**

### 🎯 Verdict: GREEN SHADE NET (50%) - WINNER
**Why shade net beats dry grass:**

| Factor | Dry Grass Mulch | Green Shade Net (50%) |
|--------|-----------------|------------------------|
| Soil cooling | 8-10°C ✅ | 5-7°C ✅ |
| Fire risk | HIGH ❌ (42°C heat) | ZERO ✅ |
| Rain penetration | Blocks 40% ❌ | Allows 100% ✅ |
| Wind damage | Blows away ❌ | Anchored ✅ |
| Reusable | No (1 season) | Yes (3-5 years) |
| Cost | ₹2,000/ha | ₹8,000/ha |
| Labor | High (spreading) | Low (one-time setup) |

### 🛠️ Implementation
**Shade Net Setup:**
- Height: 2-2.5 meters above crop
- Density: 50% (NOT 75% - blocks too much light)
- Anchoring: Bamboo poles every 3 meters
- Installation time: 1 day for 1 acre

**If budget tight - Hybrid approach:**
- Shade net over 50% area (most vulnerable)
- Dry grass mulch on remaining 50%
- Keep grass moist (spray water daily to prevent fire)

### ⚠️ Risk Alert
- Dry grass + 42°C = Fire hazard (happened in Punjab 2022)
- Vegetables need 60-70% light (shade net provides this)
- Without protection: 30% crop loss in 7 days"""
    
    elif any(word in q for word in ["village always drowns", "forecast", "farm going to sink", "flood"]):
        analysis["type"] = "flood_risk_prediction"
        analysis["requires_calculation"] = True
        analysis["specific_data_needed"] = ["7day_forecast", "historical_flood_frequency", "drainage_capacity"]
        analysis["calculation_hints"] = "If village floods every 3 years (33% annual risk) and 7-day forecast >200mm, flood probability 75-85%."
        analysis["visual_analytics"] = ["flood_risk_map"]
        analysis["response_framework"] = """### 🌊 Flood Risk Assessment
- Historical flood frequency: Every {X} years
- Annual flood probability: {100/X:.0f}%
- 7-day forecast: {Y}mm
- District drainage capacity: {Z}mm/day
- **Flood Risk Score: {risk:.0f}% (HIGH/MODERATE/LOW)**

### 🎯 Verdict
**Your farm WILL flood if:**
- 7-day rain >{Z*7}mm AND
- Rain intensity >50mm in 24 hours

**Current forecast: {Y}mm in 7 days**
- Daily average: {Y/7:.0f}mm/day
- **Risk level: {'HIGH - Evacuate crops' if Y>200 else 'MODERATE - Prepare drainage' if Y>100 else 'LOW - Monitor'}**

### 🛡️ Emergency Action Plan
**If HIGH risk (>200mm forecast):**
1. **Harvest immediately** (even if 80% mature)
   - 80% yield NOW > 0% yield after flood
2. **Drain field**: Dig emergency channels (30cm deep)
3. **Move equipment**: Pumps, tools to high ground
4. **Livestock**: Relocate within 48 hours

**If MODERATE risk (100-200mm):**
1. **Drainage trenches**: 45cm deep, 1 per 20 meters
2. **Bund strengthening**: Raise by 30cm
3. **Pump standby**: Keep diesel pump ready
4. **Insurance**: File pre-flood photos (claim evidence)

### 📅 Timeline
- Day 0-1: Dig drainage (URGENT)
- Day 2: Harvest if rain starts
- Day 3-7: Monitor hourly, pump water if needed

### ⚠️ Risk Alert
- If 100mm falls in 24 hours: 90% flood certainty
- Waterlogging >48 hours = Total crop loss
- **Start drainage work TODAY**"""
    
    elif any(word in q for word in ["warm nights", "hot night", "winter", "wheat grains", "big and heavy"]):
        analysis["type"] = "night_temperature_grain_filling"
        analysis["requires_calculation"] = True
        analysis["specific_data_needed"] = ["night_temp_trend", "grain_filling_stage", "yield_impact"]
        analysis["calculation_hints"] = "Wheat grain filling needs <15°C nights. Each 1°C above 15°C = 4-6% yield loss. If night temp 18°C vs ideal 12°C, expect 24-36% lighter grains."
        analysis["visual_analytics"] = ["night_temp_trend"]
        analysis["response_framework"] = """### 🌡️ Night Temperature Impact Analysis
- Current night temp: {X}°C
- Ideal for wheat grain filling: 10-15°C
- **Deviation: +{X-12:.0f}°C above optimal**

### 📉 Yield Impact Calculation
**Grain Weight Loss:**
- Each 1°C above 15°C = 4-6% lighter grains
- Your night temp: {X}°C
- **Expected grain weight loss: {(X-15)*5:.0f}% (if {X}>15)**
- Normal grain weight: 40-45g per 1000 grains
- **Your expected: {40*(1-(X-15)*0.05):.0f}g per 1000 grains**

### 🎯 Damage Control Strategy
**You CANNOT cool the air, but you CAN:**

**1. Foliar Spray (Immediate - Next 3 days)**
- **Salicylic acid** (100 ppm): Activates heat-shock proteins
- **Potassium nitrate** (2%): Improves grain filling under stress
- Dosage: 500 liters/ha
- Application: 6-8 PM (before night)
- **Effect: Reduces yield loss by 30-40%**

**2. Late Irrigation (Tonight onwards)**
- Irrigate at 8-10 PM (NOT morning)
- Wet soil cools air by 2-3°C overnight
- **Critical: Do this for next 15 days (grain filling period)**

**3. Potassium Boost**
- **Muriate of Potash (MOP)**: 25 kg/ha
- Apply with irrigation
- Improves starch synthesis despite heat

### 📊 Realistic Expectation
**Without intervention:**
- Yield loss: {(X-15)*5:.0f}%
- Grain quality: Shriveled, low test weight

**With intervention:**
- Yield loss: {(X-15)*3:.0f}% (reduced)
- Grain quality: Acceptable for market

### ⚠️ Risk Alert
- Warm nights = Climate change impact (permanent trend)
- Next year: Switch to heat-tolerant varieties (HD-3086, DBW-187)
- This year: Implement all 3 strategies IMMEDIATELY
- **Start tonight - grain filling window is only 20 days**"""
    
    elif "irrigation" in q or "et0" in q or "evapotranspiration" in q:
        analysis["type"] = "irrigation"
        analysis["requires_calculation"] = True
        analysis["specific_data_needed"] = ["soil_moisture", "et0", "crop_water_requirement"]
        analysis["calculation_hints"] = "Irrigation hours = (ET0 - Rainfall) / Drip rate. For cotton: 5-7mm/day. If soil moisture <15% and ET0=6.5mm/day, need ~4-6 hours drip."
        analysis["visual_analytics"] = ["irrigation_schedule"]
    
    # Trend comparison queries
    elif "compare" in q or "vs" in q or "wheat vs maize" in q:
        analysis["type"] = "comparison"
        analysis["requires_calculation"] = True
        analysis["specific_data_needed"] = ["historical_npk_by_crop", "yield_trends"]
        analysis["calculation_hints"] = "Extract 5-year N depletion rate for each crop. Show which depletes faster."
    
    # Economic/ROI queries
    elif "roi" in q or "msp" in q or "profit" in q or "investment" in q:
        analysis["type"] = "economic"
        analysis["requires_calculation"] = True
        analysis["specific_data_needed"] = ["msp_trend", "input_costs", "yield_estimate"]
        analysis["calculation_hints"] = "ROI = (MSP × Yield - Input Cost) / Input Cost × 100. If MSP trending up and K=Medium, extra potash may increase yield 10-15%."
    
    # Pest/Disease queries
    elif "pest" in q or "disease" in q or "fungal" in q or "nematode" in q:
        analysis["type"] = "pest_disease"
        analysis["requires_calculation"] = False
        analysis["specific_data_needed"] = ["humidity", "soil_temp", "historical_pest_data"]
        analysis["calculation_hints"] = "If humidity>80% and soil_temp=28°C, high fungal risk. If NDVI drops 0.7→0.4 in 10 days with adequate moisture, suspect root disease."
    
    # Climate risk queries
    elif "flood" in q or "drought" in q or "risk score" in q:
        analysis["type"] = "climate_risk"
        analysis["requires_calculation"] = True
        analysis["specific_data_needed"] = ["7day_forecast", "historical_flood_frequency", "current_rainfall"]
        analysis["calculation_hints"] = "Flood risk = (7-day forecast / historical max) × historical frequency. If district floods every 3 years and current rain is 80% of max, risk is HIGH."
    
    # Long-term strategy queries
    elif "10 year" in q or "2030" in q or "perennial" in q or "long-term" in q:
        analysis["type"] = "long_term_strategy"
        analysis["requires_calculation"] = True
        analysis["specific_data_needed"] = ["10year_temp_trend", "10year_rainfall_trend"]
        analysis["calculation_hints"] = "If temp increasing 0.2°C/year, by 2030 will be +1.2°C. Recommend heat-tolerant crops like millets, drought-resistant varieties."
    
    return analysis

def build_enhanced_context(question: str, base_context: str, query_analysis: dict, 
                          crop_matches_df, agri_metrics: dict, weather_data: dict,
                          state: str = "", district: str = "") -> str:
    """Build query-specific enhanced context with calculations and visual analytics"""
    
    enhanced = base_context + "\n\n=== QUERY-SPECIFIC ANALYSIS ===\n"
    
    if "response_framework" in query_analysis:
        enhanced += "\n=== RESPONSE FRAMEWORK (MANDATORY) ===\n"
        enhanced += query_analysis["response_framework"] + "\n\n"
    
    if query_analysis["type"] == "nutrient_lockup":
        n_val = agri_metrics.get('nitrogen', 50)
        p_val = agri_metrics.get('phosphorus', 50)
        k_val = agri_metrics.get('potassium', 50)
        
        enhanced += f"🧪 NPK Imbalance Analysis:\n"
        enhanced += f"- Nitrogen: {n_val:.1f} kg/ha\n"
        enhanced += f"- Phosphorus: {p_val:.1f} kg/ha\n"
        enhanced += f"- Potassium: {k_val:.1f} kg/ha\n"
        enhanced += f"- Actual Ratio: {n_val/10:.0f}:{p_val/10:.0f}:{k_val/10:.0f}\n"
        
        if n_val > 100 and p_val < 40:
            enhanced += f"\n⚠️ NUTRIENT LOCK-UP CONFIRMED:\n"
            enhanced += f"- High N ({n_val:.0f}) blocks P uptake\n"
            enhanced += f"- Solution: 50 kg/ha DAP + 30 kg/ha MOP\n"
    
    elif query_analysis["type"] == "saline_soil_buffering":
        ph_val = agri_metrics.get('ph', 7.0)
        precip_7day = agri_metrics.get('precip_7day', 0)
        
        enhanced += f"🧪 Saline Soil Analysis:\n"
        enhanced += f"- pH: {ph_val:.1f}\n"
        enhanced += f"- 7-day rain: {precip_7day:.1f}mm\n"
        
        ph_drop = (precip_7day / 100) * 0.3
        salt_leach = min((precip_7day / 100) * 30, 50)
        
        enhanced += f"\n🌧️ Rain Impact:\n"
        enhanced += f"- pH drop: {ph_drop:.2f} units\n"
        enhanced += f"- Salt leaching: {salt_leach:.0f}%\n"
        
        if precip_7day < 100:
            enhanced += f"\n⚠️ INSUFFICIENT - Apply 500 kg/ha gypsum\n"
        else:
            enhanced += f"\n✅ PARTIAL HELP - Reduce gypsum to 350 kg/ha\n"
    
    elif query_analysis["type"] == "soil_health_emergency":
        temp = agri_metrics.get('temperature', 30)
        enhanced += f"🚨 Soil Health Emergency:\n"
        enhanced += f"- Score: 45/100 (CRITICAL)\n"
        enhanced += f"- Temp: {temp:.1f}°C\n"
        
        if temp > 30:
            enhanced += f"\n⚡ Fast Solutions:\n"
            enhanced += f"1. Humic acid: 3-5 days | ₹800/ha\n"
            enhanced += f"2. Jeevamrut: 5-7 days | ₹200/ha\n"
            enhanced += f"3. Vermi-tea: 7-10 days | ₹500/ha\n"
    
    elif query_analysis["type"] == "irrigation_emergency":
        soil_moisture = agri_metrics.get('soil_moisture', 20.0)
        et0 = agri_metrics.get('et0', 6.0)
        
        enhanced += f"💧 Irrigation Emergency:\n"
        enhanced += f"- Moisture: {soil_moisture:.1f}%\n"
        enhanced += f"- Wilting: 15%\n"
        enhanced += f"- ET0: {et0:.1f} mm/day\n"
        
        if soil_moisture <= 15:
            deficit_mm = ((25 - soil_moisture) / 100) * 300
            hours = deficit_mm / 2.5
            enhanced += f"\n🚨 AT WILTING POINT:\n"
            enhanced += f"- Deficit: {deficit_mm:.0f}mm\n"
            enhanced += f"- **PUMP: {hours:.0f} hours**\n"
            enhanced += f"- Split: {hours/2:.0f}h NOW + {hours/2:.0f}h tonight\n"
    
    elif query_analysis["type"] == "drought_sowing_decision":
        precip_7day = agri_metrics.get('precip_7day', 0)
        historical_july_avg = 300  # Typical monsoon month
        
        enhanced += f"🌧️ Drought/Sowing Analysis:\n"
        enhanced += f"- Historical July avg: {historical_july_avg}mm\n"
        enhanced += f"- 7-day forecast: {precip_7day:.1f}mm\n"
        
        deficit_pct = ((historical_july_avg - precip_7day) / historical_july_avg) * 100
        enhanced += f"- **Rainfall deficit: {deficit_pct:.0f}%**\n"
        
        if deficit_pct > 80:
            enhanced += f"\n⚠️ SEVERE DROUGHT RISK:\n"
            enhanced += f"- DO NOT wait for rain\n"
            enhanced += f"- Switch to short-duration crops:\n"
            enhanced += f"  1. Pearl Millet (75 days, 350mm water)\n"
            enhanced += f"  2. Green Gram (60 days, 300mm water)\n"
            enhanced += f"  3. Sorghum (90 days, 450mm water)\n"
    
    elif query_analysis["type"] == "heat_stress_protection":
        temp = agri_metrics.get('temperature', 30)
        
        enhanced += f"🌡️ Heat Stress Analysis:\n"
        enhanced += f"- Current temp: {temp:.1f}°C\n"
        enhanced += f"- Forecast peak: 42°C\n"
        enhanced += f"- Crop tolerance: 38°C\n"
        
        if temp >= 40:
            enhanced += f"\n🚨 EXTREME HEAT - EMERGENCY:\n"
            enhanced += f"- **Kaolin clay spray**: 50g/L, 500L/ha\n"
            enhanced += f"- Cools leaves by 5-7°C\n"
            enhanced += f"- Apply before 9 AM tomorrow\n"
            enhanced += f"- Cost: ₹1,200/ha\n"
    
    elif query_analysis["type"] == "uv_mulching_decision":
        temp = agri_metrics.get('temperature', 30)
        
        enhanced += f"☀️ UV/Mulching Decision:\n"
        enhanced += f"- Temperature: {temp:.1f}°C\n"
        enhanced += f"- UV Index: EXTREME\n"
        
        enhanced += f"\n🎯 VERDICT: Green Shade Net (50%) WINS\n"
        enhanced += f"- Soil cooling: 5-7°C\n"
        enhanced += f"- Fire risk: ZERO (vs HIGH for dry grass at 42°C)\n"
        enhanced += f"- Rain penetration: 100%\n"
        enhanced += f"- Reusable: 3-5 years\n"
        enhanced += f"- Cost: ₹8,000/ha (vs ₹2,000/ha grass)\n"
    
    elif query_analysis["type"] == "flood_risk_prediction":
        precip_7day = agri_metrics.get('precip_7day', 0)
        flood_frequency_years = 3  # Village floods every 3 years
        
        enhanced += f"🌊 Flood Risk Assessment:\n"
        enhanced += f"- Historical: Floods every {flood_frequency_years} years\n"
        enhanced += f"- Annual probability: {100/flood_frequency_years:.0f}%\n"
        enhanced += f"- 7-day forecast: {precip_7day:.1f}mm\n"
        
        if precip_7day > 200:
            flood_risk = 85
            enhanced += f"- **FLOOD RISK: {flood_risk}% (HIGH)**\n"
            enhanced += f"\n🚨 EMERGENCY ACTION:\n"
            enhanced += f"- Harvest immediately (even if 80% mature)\n"
            enhanced += f"- Dig drainage: 30cm deep channels\n"
            enhanced += f"- Move equipment to high ground\n"
        elif precip_7day > 100:
            flood_risk = 50
            enhanced += f"- **FLOOD RISK: {flood_risk}% (MODERATE)**\n"
            enhanced += f"- Dig drainage trenches: 45cm deep\n"
            enhanced += f"- Keep pump ready\n"
        else:
            enhanced += f"- **FLOOD RISK: LOW**\n"
    
    elif query_analysis["type"] == "night_temperature_grain_filling":
        temp = agri_metrics.get('temperature', 30)
        night_temp = temp - 8  # Approximate night temp
        
        enhanced += f"🌡️ Night Temperature Impact:\n"
        enhanced += f"- Night temp: ~{night_temp:.0f}°C\n"
        enhanced += f"- Ideal for wheat: 10-15°C\n"
        
        if night_temp > 15:
            yield_loss_pct = (night_temp - 15) * 5
            grain_weight_loss = 40 * (1 - (night_temp - 15) * 0.05)
            
            enhanced += f"- **Deviation: +{night_temp-12:.0f}°C above optimal**\n"
            enhanced += f"\n📉 Yield Impact:\n"
            enhanced += f"- Expected grain weight loss: {yield_loss_pct:.0f}%\n"
            enhanced += f"- Grain weight: {grain_weight_loss:.0f}g per 1000 (vs 40g normal)\n"
            enhanced += f"\n💊 Damage Control:\n"
            enhanced += f"1. Salicylic acid spray (100 ppm) - Tonight\n"
            enhanced += f"2. Late irrigation (8-10 PM) - Cools air 2-3°C\n"
            enhanced += f"3. Potassium boost (25 kg/ha MOP)\n"
            enhanced += f"- **Can reduce loss to {yield_loss_pct*0.6:.0f}%**\n"
    
    elif query_analysis["type"] == "soil_nutrient":
        if crop_matches_df is not None and not crop_matches_df.empty and 'NDVI_Vegetation_Index' in crop_matches_df.columns:
            ndvi_vals = crop_matches_df['NDVI_Vegetation_Index'].tolist()
            if len(ndvi_vals) >= 2:
                ndvi_trend = ndvi_vals[-1] - ndvi_vals[0]
                enhanced += f"📉 NDVI: {ndvi_vals[0]:.3f} → {ndvi_vals[-1]:.3f} ({ndvi_trend:+.3f})\n"
    
    elif query_analysis["type"] == "irrigation":
        et0 = agri_metrics.get('et0', 6.0)
        soil_moisture = agri_metrics.get('soil_moisture', 20.0)
        enhanced += f"💧 Irrigation: Moisture {soil_moisture:.1f}% | ET0 {et0:.1f}mm/day\n"
    
    elif query_analysis["type"] == "economic":
        if crop_matches_df is not None and not crop_matches_df.empty and 'Historical_MSP_INR' in crop_matches_df.columns:
            msp_vals = crop_matches_df['Historical_MSP_INR'].tolist()
            if len(msp_vals) >= 2:
                msp_trend = ((msp_vals[-1] - msp_vals[0]) / msp_vals[0]) * 100
                enhanced += f"💰 MSP: ₹{msp_vals[0]:.0f} → ₹{msp_vals[-1]:.0f} ({msp_trend:+.1f}%)\n"
    
    elif query_analysis["type"] == "climate_risk":
        precip_7day = agri_metrics.get('precip_7day', 0)
        enhanced += f"⚠️ 7-Day Rain: {precip_7day:.1f}mm\n"
        if precip_7day > 150:
            enhanced += f"- FLOOD RISK: HIGH (65-75%)\n"
    
    return enhanced
