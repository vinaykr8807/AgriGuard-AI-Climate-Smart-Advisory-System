import streamlit as st
import pandas as pd
import pickle
import torch

# Optional imports with fallbacks
try:
    from transformers import T5ForConditionalGeneration, T5Tokenizer
    from peft import PeftModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

import requests
import json
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
from typing import Dict, Optional, Tuple, List
import time
import io
from gtts import gTTS

# Domain-specific prompt engine
from domain_prompts import build_groq_payload

# Language Mapping for Translation & Voice
LANG_MAP = {
    "Hindi (हिंदी)": {"code": "hi", "tld": "co.in", "name": "Hindi"},
    "Tamil (தமிழ்)": {"code": "ta", "tld": "co.in", "name": "Tamil"},
    "Telugu (తెలుగు)": {"code": "te", "tld": "co.in", "name": "Telugu"},
    "Marathi (मराठी)": {"code": "mr", "tld": "co.in", "name": "Marathi"},
    "Punjabi (ਪੰਜਾਬੀ)": {"code": "pa", "tld": "co.in", "name": "Punjabi"},
    "English": {"code": "en", "tld": "co.in", "name": "English"}
}

# Query Analysis Enhancement
def analyze_query_type(question: str) -> dict:
    """Detect query type and extract required calculations"""
    q = question.lower()
    analysis = {"type": "general", "requires_calculation": False, "specific_data_needed": [], "calculation_hints": ""}
    
    # 1. Comparison queries - detect crop comparison (HIGHEST PRIORITY)
    if any(word in q for word in ["compare", "vs", "versus", "wheat or maize", "which crop", "confused between", "suck more", "depletes"]):
        analysis.update({"type": "comparison", "requires_calculation": True, "specific_data_needed": ["historical_npk_by_crop", "yield_trends", "water_requirement"], "calculation_hints": "Extract 5-year N depletion rate for each crop. Build comparison table: NPK consumption, water needs, duration, MSP trend, climate risk."})
    
    # 2. Economic & Market queries 
    elif any(word in q for word in ["msp", "price", "mandi", "profit", "munafa", "savings", "cost", "money back", "bumper harvest", "cash"]):
        analysis.update({"type": "economic", "requires_calculation": True, "specific_data_needed": ["msp_trend", "input_costs", "yield_estimate"], "calculation_hints": "Calculate ROI: (MSP × Expected Yield - Input Cost) / Input Cost × 100. Check 5-year MSP trend. If increasing >15%, good investment."})
        
    # 3. Future Planning & Strategy
    elif any(word in q for word in ["hotter every year", "children", "10 years", "trees", "long-term plants", "profitable in 2030", "ten years ago", "still safe", "changing weather", "too risky", "father used to"]):
        analysis.update({"type": "long_term_strategy", "requires_calculation": True, "specific_data_needed": ["10year_temp_trend", "10year_rainfall_trend"], "calculation_hints": "Compare 2015-2019 vs 2020-2024 climate data. If temp +1°C and rain -15%, crop viability changed. Recommend climate-resilient alternatives."})

    # 4. Soil & Fertilizer queries
    elif any(word in q for word in ["urea", "fertilizer", "locked", "weak", "small plants", "nutrient lock", "npk", "nitrogen", "white fertilizer", "strength", "strength of my land"]):
        analysis.update({"type": "soil_nutrient", "requires_calculation": True, "specific_data_needed": ["historical_ndvi_trend", "npk_levels", "soil_ph"], "calculation_hints": "High N but poor growth = nutrient lock (pH issue or P/K deficiency). Check NDVI trend. Prescribe DAP/MOP with exact kg/ha."})
    elif any(word in q for word in ["salty", "white soil", "hard land", "sulfur", "ph", "rain clean", "alkaline", "acidic"]):
        analysis.update({"type": "ph_buffering", "requires_calculation": True, "specific_data_needed": ["soil_ph", "forecasted_rainfall", "soil_type"], "calculation_hints": "Saline/alkaline soil (pH>8). Heavy rain (>100mm) provides partial buffering. Calculate if sulfur/gypsum still needed based on 7-day forecast."})
    elif any(word in q for word in ["soil health", "45 score", "infertile", "barren", "natural manure", "recover soil", "melt into the ground", "restore it"]):
        analysis.update({"type": "soil_recovery", "requires_calculation": True, "specific_data_needed": ["soil_health_score", "temperature", "organic_matter"], "calculation_hints": "Low soil score (<50) needs urgent organic intervention. At 32°C, use fast-acting FYM/vermicompost. Give month-by-month recovery plan."})
    elif any(word in q for word in ["dry", "moisture", "pump", "water hours", "irrigation", "wilt", "bone dry", "water pump", "drink"]):
        analysis.update({"type": "irrigation", "requires_calculation": True, "specific_data_needed": ["soil_moisture", "et0", "crop_water_requirement"], "calculation_hints": "Soil moisture <15% = critical. Calculate irrigation hours: (ET0 - Rain) / Drip rate. For cotton at 32°C: 6-8mm/day needed."})
    
    # 5. Weather & Climate Risk
    elif any(word in q for word in ["flood", "drown", "sink", "heavy rain", "forecast rain", "drowning", "sink"]):
        analysis.update({"type": "climate_risk", "requires_calculation": True, "specific_data_needed": ["7day_forecast", "historical_flood_frequency", "current_rainfall"], "calculation_hints": "Check 7-day forecast vs historical flood threshold. If >150mm predicted and district floods every 3 years, risk is HIGH (70%+)."})
    elif any(word in q for word in ["no rain", "drought", "wait to sow", "quick-switch", "faster crop", "short-duration", "delay"]):
        analysis.update({"type": "drought_strategy", "requires_calculation": True, "specific_data_needed": ["7day_forecast", "soil_moisture", "crop_duration"], "calculation_hints": "No rain forecast + low soil moisture = switch to short-duration crops (60-75 days). Recommend millets, pulses over rice/sugarcane."})
    elif any(word in q for word in ["heat", "42°c", "fire sun", "pale field", "hot night", "warm winter", "fire-like sun", "bite the skin"]):
        analysis.update({"type": "heat_stress", "requires_calculation": True, "specific_data_needed": ["temperature", "humidity", "crop_stage"], "calculation_hints": "Temp >38°C = heat stress. Prescribe anti-transpirant spray (Kaolin clay 5%). Hot nights (>20°C) reduce wheat grain filling by 10-15%."})
    elif any(word in q for word in ["uv", "cover ground", "dry grass", "green net", "shade", "mulch"]):
        analysis.update({"type": "heat_protection", "requires_calculation": False, "specific_data_needed": ["uv_index", "crop_type"], "calculation_hints": "High UV: Green shade net (50%) for vegetables. Mulching with dry grass reduces soil temp by 3-5°C."})
    
    # 6. Technical & Monitoring
    elif any(word in q for word in ["satellite view", "green field", "yield prediction", "how much i will get"]):
        analysis.update({"type": "yield_prediction", "requires_calculation": True, "specific_data_needed": ["ndvi", "historical_yield", "weather"], "calculation_hints": "High NDVI (>0.6) + adequate rain = potential bumper harvest. Compare current NDVI with historical average for district."})
    elif any(word in q for word in ["two crops", "intercrop", "2 hectares", "diversify", "mixed cropping", "lose everything"]):
        analysis.update({"type": "crop_diversification", "requires_calculation": False, "specific_data_needed": ["land_size", "risk_tolerance"], "calculation_hints": "Recommend 60-40 split: Main crop (cash) + insurance crop (pulses/millets). Reduces total risk by 40%."})
    
    # 7. Pest & Disease
    elif any(word in q for word in ["sticky", "humidity", "black spots", "rot", "spray now", "fungal", "spots"]):
        analysis.update({"type": "pest_disease", "requires_calculation": True, "specific_data_needed": ["humidity", "temperature", "crop_stage"], "calculation_hints": "Humidity >75% + Temp 25-30°C = fungal outbreak risk. Prescribe preventive fungicide (Mancozeb 2g/L or Copper oxychloride). Spray before symptoms."})
    elif any(word in q for word in ["browning leaves", "worm eating roots", "nutrients missing", "plants dying", "leaves are browning"]):
        analysis.update({"type": "diagnosis", "requires_calculation": True, "specific_data_needed": ["ndvi", "soil_moisture", "npk"], "calculation_hints": "Browning despite water = root disease OR nutrient deficiency. Check NDVI drop rate. If sudden (<7 days), suspect pest. If gradual, check NPK."})
    elif any(word in q for word in ["keeda", "pest", "paddy", "dry year", "kill it", "bugs"]):
        analysis.update({"type": "pest_identification", "requires_calculation": False, "specific_data_needed": ["crop", "season", "district_history"], "calculation_hints": "Dry years: Stem borer (rice), bollworm (cotton) are common. Prescribe: Chlorantraniliprole 0.4ml/L or neem oil 5ml/L (organic)."})
    elif any(word in q for word in ["cloudy", "no sun", "bugs grow", "spray work"]):
        analysis.update({"type": "spray_timing", "requires_calculation": False, "specific_data_needed": ["weather", "pesticide_type"], "calculation_hints": "Cloudy weather: Fungal risk increases. Most sprays need 4-6 hours dry time. Avoid spraying if rain expected within 6 hours."})
    elif any(word in q for word in ["drought 2019", "successful farmers", "save crops"]):
        analysis.update({"type": "historical_learning", "requires_calculation": False, "specific_data_needed": ["historical_events", "district_practices"], "calculation_hints": "Reference 2019 drought strategies: Drip irrigation, mulching, drought-tolerant varieties (e.g., Arjun wheat, Phule Revati sorghum)."})
    
    # 8. Others
    elif any(word in q for word in ["summer earlier", "grass browning", "15 days earlier"]):
        analysis.update({"type": "seasonal_shift", "requires_calculation": True, "specific_data_needed": ["historical_sowing_dates", "temperature_trend"], "calculation_hints": "If summer advancing by 10-15 days, adjust sowing accordingly. Check 10-year temperature onset data for district."})
    elif any(word in q for word in ["natural farming", "prakritik kheti", "no chemicals", "survive heat"]):
        analysis.update({"type": "organic_comparison", "requires_calculation": False, "specific_data_needed": ["organic_practices", "heat_tolerance"], "calculation_hints": "Natural farming: Better soil moisture retention (+15-20%), but 20-30% lower yield initially. Heat tolerance similar if mulching used."})
    elif any(word in q for word in ["wells empty", "no water", "almost no water", "drought-resistant"]):
        analysis.update({"type": "water_scarcity", "requires_calculation": False, "specific_data_needed": ["water_requirement", "drought_crops"], "calculation_hints": "Ultra-low water crops: Pearl millet (250mm), Sorghum (300mm), Chickpea (350mm), Sesame (300mm). 50-70% less water than rice."})
    elif any(word in q for word in ["40,000 rupees", "500$", "better seeds", "more khad", "tool", "save farm"]):
        analysis.update({"type": "investment_priority", "requires_calculation": True, "specific_data_needed": ["budget", "current_constraints"], "calculation_hints": "₹40k priority: 1) Soil test (₹500), 2) Drip kit (₹15k), 3) Quality seeds (₹8k), 4) Balanced fertilizer (₹12k), 5) Soil moisture sensor (₹4k). ROI: 150-200%."})
    
    return analysis

def build_enhanced_context(question: str, base_context: str, query_analysis: dict, crop_matches_df, agri_metrics: dict, weather_data: dict, state: str = "", district: str = "") -> str:
    """Build query-specific enhanced context with calculations"""
    enhanced = base_context + "\n\n=== QUERY-SPECIFIC ANALYSIS ===\n"
    
    if query_analysis["type"] == "soil_nutrient":
        if crop_matches_df is not None and not crop_matches_df.empty and 'NDVI_Vegetation_Index' in crop_matches_df.columns:
            ndvi_vals = crop_matches_df['NDVI_Vegetation_Index'].tolist()
            if len(ndvi_vals) >= 2:
                ndvi_trend = ndvi_vals[-1] - ndvi_vals[0]
                enhanced += f"📉 NDVI Trend (10-year): {ndvi_vals[0]:.3f} → {ndvi_vals[-1]:.3f} (Change: {ndvi_trend:+.3f})\n"
                if ndvi_trend < -0.1:
                    enhanced += "⚠️ DECLINING NDVI detected despite fertilizer use = Possible nutrient lock-in or soil degradation\n"
    
    elif query_analysis["type"] == "comparison":
        # Extract crop names from question
        q_lower = question.lower()
        crops_to_compare = []
        common_crops = ['wheat', 'maize', 'rice', 'cotton', 'soybean', 'sugarcane', 'potato', 'onion', 'tomato', 'chickpea', 'pigeon pea']
        for crop in common_crops:
            if crop in q_lower:
                crops_to_compare.append(crop.title())
        
        if len(crops_to_compare) >= 2 and 'advisory_df' in globals() and not globals()['advisory_df'].empty:
            advisory_data = globals()['advisory_df']
            
            # CRITICAL: Filter global advisory data by CURRENT state and district
            if state and district:
                location_specific_data = advisory_data[
                    (advisory_data['State'].str.strip() == state.strip()) & 
                    (advisory_data['District'].str.strip() == district.strip())
                ]
            else:
                location_specific_data = advisory_data # Fallback to global if location not provided
                
            enhanced += f"\n📊 CROP COMPARISON TABLE for {district}, {state}: {' vs '.join(crops_to_compare)}\n\n"
            enhanced += "| Parameter | " + " | ".join(crops_to_compare) + " |\n"
            enhanced += "|" + "---|" * (len(crops_to_compare) + 1) + "\n"
            
            # Compare NPK depletion with 5-year trend analysis
            for nutrient in ['Nitrogen', 'Phosphorus', 'Potassium']:
                if nutrient in advisory_data.columns:
                    row = f"| {nutrient} Requirement |"
                    for crop in crops_to_compare:
                        crop_data = location_specific_data[location_specific_data['Recommended_Crop'].str.contains(crop, case=False, na=False)]
                        if not crop_data.empty:
                            # Get last 5 years from this location
                            recent_data = crop_data.tail(5)
                            val = recent_data[nutrient].mode()[0] if len(recent_data[nutrient].mode()) > 0 else 'Medium'
                            row += f" {val} |"
                        else:
                            # Try global if location-specific missing for this crop
                            global_crop_data = advisory_data[advisory_data['Recommended_Crop'].str.contains(crop, case=False, na=False)].tail(5)
                            val = global_crop_data[nutrient].mode()[0] if not global_crop_data.empty and len(global_crop_data[nutrient].mode()) > 0 else 'N/A'
                            row += f" {val} (global) |"
                    enhanced += row + "\n"
            
            # Compare water requirement
            if 'Rainfall_IMD_mm' in advisory_data.columns:
                row = "| Water Requirement |"
                for crop in crops_to_compare:
                    crop_data = location_specific_data[location_specific_data['Recommended_Crop'].str.contains(crop, case=False, na=False)]
                    if not crop_data.empty:
                        avg_rain = crop_data.tail(5)['Rainfall_IMD_mm'].mean()
                        row += f" {avg_rain:.0f}mm |"
                    else:
                        row += " N/A |"
                enhanced += row + "\n"
            
            # Compare MSP trend
            if 'Historical_MSP_INR' in advisory_data.columns:
                row = "| Avg MSP (₹/quintal) |"
                for crop in crops_to_compare:
                    crop_data = location_specific_data[location_specific_data['Recommended_Crop'].str.contains(crop, case=False, na=False)]
                    if not crop_data.empty:
                        avg_msp = crop_data.tail(5)['Historical_MSP_INR'].mean()
                        row += f" ₹{avg_msp:.0f} |"
                    else:
                        row += " N/A |"
                enhanced += row + "\n"
            
            # 5-YEAR NITROGEN DEPLETION TREND ANALYSIS
            enhanced += f"\n💡 5-YEAR NITROGEN DEPLETION RISK ANALYSIS for {district}:\n"
            for crop in crops_to_compare:
                crop_data = location_specific_data[location_specific_data['Recommended_Crop'].str.contains(crop, case=False, na=False)]
                if crop_data.empty:
                    crop_data = advisory_data[advisory_data['Recommended_Crop'].str.contains(crop, case=False, na=False)] # Fallback
                
                if not crop_data.empty and 'Nitrogen' in crop_data.columns:
                    recent_5yr = crop_data.tail(5)
                    n_req = recent_5yr['Nitrogen'].mode()[0] if len(recent_5yr['Nitrogen'].mode()) > 0 else 'Medium'
                    
                    # Calculate depletion rate
                    high_count = (recent_5yr['Nitrogen'].isin(['High', 'Very High'])).sum()
                    depletion_risk = (high_count / len(recent_5yr)) * 100
                    
                    enhanced += f"\n**{crop}:**\n"
                    enhanced += f"- Last 5 years ({recent_5yr['Year'].min()}-{recent_5yr['Year'].max()}) N demand: {n_req}\n"
                    enhanced += f"- Depletion risk score: {depletion_risk:.0f}% ({high_count}/5 years showed high N demand in {district})\n"
                    
                    if n_req in ['High', 'Very High']:
                        enhanced += f"- ⚠️ HIGH DEPLETION: Removes 80-120 kg N/ha per season\n"
                        enhanced += f"- Soil recovery needed: Apply 100-150 kg N/ha + 5 tonnes FYM\n"
                    elif n_req == 'Medium':
                        enhanced += f"- MODERATE DEPLETION: Removes 50-80 kg N/ha per season\n"
                        enhanced += f"- Soil recovery needed: Apply 60-80 kg N/ha + 3 tonnes FYM\n"
                    else:
                        enhanced += f"- LOW DEPLETION: Removes 30-50 kg N/ha per season\n"
                        enhanced += f"- Soil recovery needed: Apply 40-50 kg N/ha + 2 tonnes FYM\n"
            
            # VERDICT
            enhanced += "\n🎯 VERDICT:\n"
            if len(crops_to_compare) == 2:
                c1_data = location_specific_data[location_specific_data['Recommended_Crop'].str.contains(crops_to_compare[0], case=False, na=False)].tail(5)
                c2_data = location_specific_data[location_specific_data['Recommended_Crop'].str.contains(crops_to_compare[1], case=False, na=False)].tail(5)
                
                if not c1_data.empty and not c2_data.empty and 'Nitrogen' in c1_data.columns:
                    c1_high = (c1_data['Nitrogen'].isin(['High', 'Very High'])).sum()
                    c2_high = (c2_data['Nitrogen'].isin(['High', 'Very High'])).sum()
                    
                    if c1_high > c2_high:
                        enhanced += f"- {crops_to_compare[0]} depletes soil FASTER than {crops_to_compare[1]} in {district} ({c1_high} vs {c2_high} high-N years)\n"
                        enhanced += f"- Recommendation: Rotate with legumes (chickpea/pigeon pea) after {crops_to_compare[0]} to restore N\n"
                    elif c2_high > c1_high:
                        enhanced += f"- {crops_to_compare[1]} depletes soil FASTER than {crops_to_compare[0]} in {district} ({c2_high} vs {c1_high} high-N years)\n"
                        enhanced += f"- Recommendation: Rotate with legumes (chickpea/pigeon pea) after {crops_to_compare[1]} to restore N\n"
                    else:
                        enhanced += f"- Both crops have SIMILAR nitrogen depletion rates in this location\n"
                        enhanced += f"- Recommendation: Choose based on market price (MSP) and water availability\n"
        else:
            enhanced += f"\n⚠️ Comparison requested but insufficient data for {district}, {state}.\n"
    
    elif query_analysis["type"] == "irrigation":
        et0 = agri_metrics.get('et0', 6.0)
        soil_moisture = agri_metrics.get('soil_moisture', 20.0)
        enhanced += f"💧 Irrigation Calculation:\n- Current Soil Moisture: {soil_moisture:.1f}%\n- ET0 (Evapotranspiration): {et0:.1f} mm/day\n- Wilting Point: ~15% soil moisture\n"
        if soil_moisture < 15:
            deficit = (15 - soil_moisture) * 10
            hours_needed = deficit / 2.5
            enhanced += f"⚠️ URGENT: Soil moisture below wilting point!\n- Estimated water deficit: {deficit:.1f}mm\n- Drip irrigation needed: ~{hours_needed:.1f} hours (at 2.5mm/hr rate)\n"
    
    elif query_analysis["type"] == "ph_buffering":
        precip_7day = agri_metrics.get('precip_7day', 0)
        enhanced += f"🧪 pH Buffering Analysis:\n- Forecasted 7-day rainfall: {precip_7day:.1f}mm\n"
        if precip_7day < 100:
            enhanced += f"- Rain is INSUFFICIENT to naturally buffer alkaline pH\n- 40mm rain typically lowers pH by only 0.1-0.2 units\n- RECOMMENDATION: Still apply sulfur/gypsum as planned\n"
        else:
            enhanced += f"- Heavy rain may provide PARTIAL pH buffering (0.2-0.4 units)\n- RECOMMENDATION: Reduce sulfur dose by 30-40%, retest after rain\n"
    
    elif query_analysis["type"] == "economic":
        if crop_matches_df is not None and not crop_matches_df.empty and 'Historical_MSP_INR' in crop_matches_df.columns:
            msp_vals = crop_matches_df['Historical_MSP_INR'].tolist()
            if len(msp_vals) >= 2:
                msp_trend = ((msp_vals[-1] - msp_vals[0]) / msp_vals[0]) * 100
                enhanced += f"💰 Economic Analysis:\n- MSP Trend (10-year): ₹{msp_vals[0]:.0f} → ₹{msp_vals[-1]:.0f} ({msp_trend:+.1f}%)\n- Average MSP: ₹{sum(msp_vals)/len(msp_vals):.0f}\n"
                if msp_trend > 20:
                    enhanced += f"📈 STRONG UPWARD TREND - Good investment potential\n"
    
    elif query_analysis["type"] == "climate_risk":
        precip_7day = agri_metrics.get('precip_7day', 0)
        enhanced += f"⚠️ Climate Risk Assessment:\n- 7-Day Forecasted Rain: {precip_7day:.1f}mm\n"
        if precip_7day > 150:
            enhanced += f"- FLOOD RISK: HIGH (>150mm in 7 days)\n- Historical flood frequency: Every 3 years (33% annual probability)\n- Current Flood Risk Score: 65-75% (HIGH)\n"
        elif precip_7day > 100:
            enhanced += f"- FLOOD RISK: MODERATE (100-150mm in 7 days)\n- Current Flood Risk Score: 40-50% (MODERATE)\n"
    
    return enhanced

# Try to import transformers and PEFT for LoRA models (optional)
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Detect deployment environment
IS_STREAMLIT_CLOUD = os.environ.get("STREAMLIT_SHARING", "").lower() == "true"
IS_CLOUD_DEPLOYMENT = IS_STREAMLIT_CLOUD

# Page configuration
st.set_page_config(
    page_title="🌾 Climate Resilience Chatbot",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------- Load Models --------------------

# =========================================================
# ML MODELS - DISABLED (Using Heuristic Scoring Instead)
# =========================================================
# The pickle models have been replaced with a heuristic-based scoring system
# that is more responsive to user inputs and doesn't require model files

crop_model = None
scaler = None
risk_model = None


# Load LLM model (simplified without cache for now)
if TRANSFORMERS_AVAILABLE:
    try:
        base_model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")
        llm_model = PeftModel.from_pretrained(base_model, "models/LLM")
        llm_tokenizer = T5Tokenizer.from_pretrained("models/LLM")
    except Exception:
        llm_model = None
        llm_tokenizer = None
else:
    llm_model = None
    llm_tokenizer = None

# Load datasets
def load_csv_data():
    """Load or reload CSV data to catch updates"""
    try:
        features = pd.read_csv("data/merged_feature_store.csv")
    except FileNotFoundError:
        features = pd.DataFrame()
    
    try:
        advisory = pd.read_csv("data/Multilingual_Expert_Advisory.csv", encoding='utf-8-sig')
    except FileNotFoundError:
        advisory = pd.DataFrame()
    except Exception:
        try:
             advisory = pd.read_csv("data/Multilingual_Expert_Advisory.csv", encoding='latin1')
        except:
             advisory = pd.DataFrame()
    
    # Build state-district mapping from advisory CSV
    mapping = {}
    if not advisory.empty and 'State' in advisory.columns and 'District' in advisory.columns:
        mapping = advisory.groupby('State')['District'].unique().apply(sorted).to_dict()
    
    return features, advisory, mapping

# Initial load
features_df, advisory_df, state_district_mapping = load_csv_data()

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E7D32;
        text-align: center;
        margin-bottom: 0.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #2E7D32, #4CAF50);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-header {
        text-align: center;
        color: #555;
        margin-bottom: 2rem;
        font-size: 1.2rem;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 10px;
        padding: 12px 28px;
        font-size: 16px;
        font-weight: bold;
        border: none;
        width: 100%;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .chat-message {
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        animation: fadeIn 0.5s;
        color: #1f1f1f;
        font-size: 16px;
        line-height: 1.6;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196F3;
        color: #1f1f1f;
    }
    .bot-message {
        background-color: #f1f8e9;
        border-left: 4px solid #4CAF50;
        color: #1f1f1f;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .info-box {
        background-color: #e8f5e9;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #4CAF50;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'recommendation_history' not in st.session_state:
    st.session_state.recommendation_history = []
if 'location_data' not in st.session_state:
    st.session_state.location_data = {}
if 'soil_params' not in st.session_state:
    st.session_state.soil_params = {'N': 50, 'P': 50, 'K': 50, 'pH': 6.5}
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'lora_model' not in st.session_state:
    st.session_state.lora_model = None
if 'tokenizer' not in st.session_state:
    st.session_state.tokenizer = None
if 't5_peft_model' not in st.session_state:
    st.session_state.t5_peft_model = None
if 't5_peft_tokenizer' not in st.session_state:
    st.session_state.t5_peft_tokenizer = None
if 'advisory_cache' not in st.session_state:
    st.session_state.advisory_cache = {}

# API Keys - Replace with your own or use Streamlit secrets
DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_MODEL = "gemma3:4b"  # Using 4b for faster responses, you can change to "llama3.2:1b" for even faster
DEFAULT_GROQ_API_KEY = ""  # ENTER YOUR GROQ API KEY HERE

# Initialize API keys in session state
if 'ollama_host' not in st.session_state:
    st.session_state.ollama_host = DEFAULT_OLLAMA_HOST
if 'ollama_model' not in st.session_state:
    st.session_state.ollama_model = DEFAULT_OLLAMA_MODEL
if 'groq_api_key' not in st.session_state:
    st.session_state.groq_api_key = DEFAULT_GROQ_API_KEY

# Title
st.markdown('<h1 class="main-header">🌾 AgriGuard AI – Climate Smart Advisory System</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-Powered Agricultural Advisor with Real-Time Weather, Soil Data & Climate Adaptation Strategies</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("🤖 AI Model Configuration")
    
    # API Key Configuration
    with st.expander("🔑 API Key Settings", expanded=False):
        groq_key = st.text_input("Groq API Key", value=st.session_state.groq_api_key, type="password")
        if groq_key != st.session_state.groq_api_key:
            st.session_state.groq_api_key = groq_key
            st.success("API Key updated for this session!")
    
    st.success("**🔬 Factual Ensemble AI Mode**")
    st.caption("Evidence-based insights from multiple sources!")
    
    st.divider()
    st.caption("**Active Models:**")
    if st.session_state.t5_peft_model:
        st.caption("✅ T5-PEFT (Agriculture Expert)")
    if st.session_state.lora_model:
        st.caption("✅ Climate-LoRA (Local Adaptive)")
    st.caption("✅ Ollama (General Knowledge)")
    st.caption("✅ Groq (Factual Synthesis)")
    
    st.divider()
    st.header("🌐 Multilingual Support")
    target_language = st.selectbox(
        "Select Your Language:",
        ["English", "Hindi (हिंदी)", "Tamil (தமிழ்)", "Telugu (తెలుగు)", "Marathi (मराठी)", "Punjabi (ਪੰਜਾਬੀ)"],
        index=0
    )
    st.session_state.target_language = target_language
    
    enable_voice = st.checkbox("🎙️ Enable Voice Response", value=False)
    st.session_state.enable_voice = enable_voice
    
    st.divider()
    st.info("""
    **How it works:**
    1. Fetches 10-year expert advisory (2015-2024)
    2. Gets real-time weather data
    3. Multi-Model Analysis (T5-PEFT + LoRA + Ollama)
    4. Groq synthesizes into fact-based recommendations
    5. **Result: Evidence-backed, not speculative!**
    """)

# Set default values for removed sidebar configuration
use_local_model = False  # Disabled LoRA for now - using Ollama + Groq instead
model_choice = "gemma3:4b"
st.session_state.model_choice = model_choice
st.session_state.ollama_model = model_choice
temperature = 0.7
max_tokens = 800  # Reduced for faster responses

# Load LoRA Model (if available)
def load_lora_model(model_path="models/climate_advisor_lora"):
    """Load LoRA fine-tuned model for agricultural advice"""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        import torch
        import os
        
        base_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load base model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None
        )
        
        # Load LoRA adapter
        if os.path.exists(model_path):
            model = PeftModel.from_pretrained(base_model, model_path)
            model.eval()
            st.success("✅ LoRA model loaded successfully!")
            return model, tokenizer
        else:
            st.info(f"ℹ️ LoRA model not found at {model_path}")
            return None, None
    except ImportError as e:
        st.warning(f"⚠️ Required libraries not installed for LoRA: {str(e)}")
        return None, None
    except Exception as e:
        st.warning(f"⚠️ LoRA model loading failed: {str(e)}")
        return None, None

def generate_with_lora(prompt: str, model, tokenizer, max_tokens: int = 200) -> Optional[str]:
    """Generate response using LoRA fine-tuned model"""
    if model is None or tokenizer is None:
        return None
        
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Format prompt for TinyLlama
        formatted_prompt = f"<|system|>\nYou are a professional agricultural advisor.<|user|>\n{prompt}<|assistant|>\n"
        
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )
            
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract only the assistant's part
        if "<|assistant|>\n" in full_response:
            response = full_response.split("<|assistant|>\n")[-1].strip()
        else:
            response = full_response.strip()
            
        return response if response else None
    except Exception as e:
        st.warning(f"⚠️ LoRA generation error: {str(e)}")
        return None

def load_t5_peft_model(model_path="models/LLM"):
    """Load T5 PEFT fine-tuned model for agricultural advice"""
    try:
        from transformers import T5ForConditionalGeneration, T5Tokenizer
        from peft import PeftModel
        import torch
        import os
        
        if not os.path.exists(model_path):
            return None, None
        
        # Load base T5 model
        base_model_name = "google/flan-t5-base"
        base_model = T5ForConditionalGeneration.from_pretrained(base_model_name)
        
        # Load PEFT adapter
        model = PeftModel.from_pretrained(base_model, model_path)
        tokenizer = T5Tokenizer.from_pretrained(model_path)
        
        model.eval()
        st.success("✅ T5-PEFT agricultural model loaded successfully!")
        return model, tokenizer
        
    except ImportError as e:
        st.info(f"ℹ️ T5-PEFT requires transformers and peft: pip install transformers peft")
        return None, None
    except Exception as e:
        st.warning(f"⚠️ T5-PEFT model loading failed: {str(e)}")
        return None, None

def generate_with_t5_peft(prompt: str, model, tokenizer, max_tokens: int = 200) -> Optional[str]:
    """Generate response using T5-PEFT model"""
    if model is None or tokenizer is None:
        return None
    
    try:
        import torch
        
        # Format prompt for T5
        formatted_prompt = f"Generate agricultural advisory: {prompt}"
        
        inputs = tokenizer(
            formatted_prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True
        )
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs.input_ids,
                max_length=max_tokens,
                num_beams=4,
                temperature=0.7,
                do_sample=True,
                early_stopping=True
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response if response else None
        
    except Exception as e:
        st.warning(f"⚠️ T5-PEFT generation error: {str(e)}")
        return None


def get_weather_data_by_coords(lat: float, lon: float, location_name: str) -> Optional[Dict]:
    """Fetch real-time weather data using coordinates directly"""
    try:
        # Get weather data using forecast endpoint for current conditions
        weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true&hourly=temperature_2m,relativehumidity_2m,precipitation,windspeed_10m,winddirection_10m,surface_pressure&timezone=auto&forecast_days=1"
        weather_response = requests.get(weather_url, timeout=10)
        
        if weather_response.status_code == 200:
            data = weather_response.json()
            current = data['current_weather']
            hourly = data['hourly']
            return {
                "temperature": current['temperature'],
                "humidity": hourly['relativehumidity_2m'][0] if hourly['relativehumidity_2m'] else 50,
                "condition": "Clear",
                "rainfall": hourly['precipitation'][0] if hourly['precipitation'] else 0,
                "wind_speed": current['windspeed'],
                "wind_direction": current['winddirection'],
                "uv_index": 5,
                "pressure": hourly['surface_pressure'][0] if hourly['surface_pressure'] else 1013,
                "feels_like": current['temperature'],
                "visibility": 10,
                "location": location_name,
                "lat": lat,
                "lon": lon,
                "timezone": data['timezone'],
                "last_updated": current['time']
            }
        else:
            return None
    except Exception as e:
        return None
@st.cache_data(ttl=3600)
def get_keyless_agri_metrics(lat: float, lon: float) -> Dict:
    """Fetch real-time NDVI-equivalent and Soil Properties from public APIs.
    Uses Open-Meteo for reliable soil data + computes Vegetation Health Index (VHI)
    as an NDVI proxy from evapotranspiration, solar radiation, and soil moisture."""
    metrics = {
        "ndvi": None,
        "soil_moisture": None,
        "soil_temp": None,
        "et0": None,           # Evapotranspiration (mm/day)
        "radiation": None,     # Solar radiation (MJ/m²)
        "precip_7day": None,   # 7-day precipitation forecast
        "source": "Satellite (Open-Meteo/ERA5)"
    }
    
    # 1. Fetch comprehensive soil + agri data from Open-Meteo (ALWAYS works, keyless)
    try:
        agri_url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={lon}"
            f"&hourly=soil_temperature_0cm,soil_temperature_6cm,"
            f"soil_moisture_0_to_1cm,soil_moisture_1_to_3cm,soil_moisture_3_to_9cm"
            f"&daily=et0_fao_evapotranspiration,precipitation_sum,shortwave_radiation_sum"
            f"&timezone=auto&forecast_days=7"
        )
        response = requests.get(agri_url, timeout=8)
        if response.status_code == 200:
            data = response.json()
            
            # Hourly soil data (use latest reading)
            if "hourly" in data:
                h = data["hourly"]
                # Surface soil moisture (avg of 0-1cm and 1-3cm layers)
                sm_0 = h.get("soil_moisture_0_to_1cm", [None])[0]
                sm_1 = h.get("soil_moisture_1_to_3cm", [None])[0]
                if sm_0 is not None and sm_1 is not None:
                    metrics["soil_moisture"] = ((sm_0 + sm_1) / 2) * 100  # Convert to %
                elif sm_0 is not None:
                    metrics["soil_moisture"] = sm_0 * 100
                
                # Soil temperature (surface)
                st_0 = h.get("soil_temperature_0cm", [None])[0]
                if st_0 is not None:
                    metrics["soil_temp"] = st_0
            
            # Daily agri data
            if "daily" in data:
                d = data["daily"]
                et0_list = d.get("et0_fao_evapotranspiration", [])
                rad_list = d.get("shortwave_radiation_sum", [])
                precip_list = d.get("precipitation_sum", [])
                
                if et0_list and et0_list[0] is not None:
                    metrics["et0"] = et0_list[0]
                if rad_list and rad_list[0] is not None:
                    metrics["radiation"] = rad_list[0]
                if precip_list:
                    metrics["precip_7day"] = sum(p for p in precip_list if p is not None)
                    
    except Exception:
        pass
    
    # 2. Compute NDVI-equivalent Vegetation Health Index (VHI)
    # Based on FAO methodology: healthy vegetation = high ET0, good soil moisture, adequate radiation
    # Scale: 0.0 (barren/dead) to 1.0 (lush green)
    try:
        et0 = metrics.get("et0")
        sm = metrics.get("soil_moisture")  # Already in % (0-100)
        rad = metrics.get("radiation")
        
        if et0 is not None and sm is not None and rad is not None:
            # Normalize each component to 0-1 range based on Indian agri conditions
            # ET0: typically 1-7 mm/day in India
            et0_norm = min(1.0, max(0.0, (et0 - 0.5) / 6.0))
            
            # Soil moisture: 5-50% typical range  
            sm_norm = min(1.0, max(0.0, (sm - 3.0) / 40.0))
            
            # Solar radiation: 5-30 MJ/m² typical in India
            rad_norm = min(1.0, max(0.0, (rad - 3.0) / 25.0))
            
            # Weighted VHI formula (soil moisture is most important for vegetation)
            # Weight: SM=0.45, ET0=0.35, Radiation=0.20
            vhi = (0.45 * sm_norm) + (0.35 * et0_norm) + (0.20 * rad_norm)
            
            # Clamp to NDVI-like range (0.05 to 0.95)
            metrics["ndvi"] = round(min(0.95, max(0.05, vhi)), 3)
            metrics["source"] = "Satellite (Open-Meteo ERA5 — VHI Proxy)"
        
    except Exception:
        pass
    
    # 3. Fallback: Try ORNL MODIS for actual NDVI (but with short timeout since it often fails)
    if metrics["ndvi"] is None:
        try:
            now = datetime.now()
            doy = now.timetuple().tm_yday
            start_date = f"A{now.year}{str(max(1, doy-32)).zfill(3)}"
            end_date = f"A{now.year}{str(doy).zfill(3)}"
            
            ndvi_url = (
                f"https://modis.ornl.gov/rst/api/v1/MOD13Q1/subset?"
                f"latitude={lat}&longitude={lon}"
                f"&startDate={start_date}&endDate={end_date}"
                f"&kmAboveBelow=0&kmLeftRight=0"
            )
            response = requests.get(ndvi_url, timeout=5)  # Short timeout
            
            if response.status_code == 200:
                data = response.json()
                if "subset" in data:
                    for entry in data["subset"]:
                        if entry.get("band") == "_250m_16_days_NDVI":
                            values = entry.get("data", [])
                            valid_vals = [v for v in values if v > -2000]
                            if valid_vals:
                                metrics["ndvi"] = round(sum(valid_vals) / len(valid_vals) * 0.0001, 3)
                                metrics["source"] = "Satellite (NASA MODIS — Real NDVI)"
        except Exception:
            pass  # VHI proxy already set above, or stays None
        
    return metrics

def get_weather_data(location: str, api_key: str = None) -> Optional[Dict]:
    """Fetch real-time weather data from OpenMeteo API"""
    try:
        # Get coordinates from location name using OpenMeteo geocoding
        geocoding_url = f"https://geocoding-api.open-meteo.com/v1/search?name={location}&count=1&language=en&format=json"
        geo_response = requests.get(geocoding_url, timeout=10)
        
        if geo_response.status_code != 200:
            st.error("Location not found")
            return None
            
        geo_data = geo_response.json()
        if not geo_data.get('results'):
            st.error("Location not found")
            return None
            
        result = geo_data['results'][0]
        lat, lon = result['latitude'], result['longitude']
        location_name = f"{result['name']}, {result.get('admin1', '')}, {result['country']}"
        
        # Get weather data using forecast endpoint for current conditions
        weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true&hourly=temperature_2m,relativehumidity_2m,precipitation,windspeed_10m,winddirection_10m,surface_pressure&timezone=auto&forecast_days=1"
        weather_response = requests.get(weather_url, timeout=10)
        
        if weather_response.status_code == 200:
            data = weather_response.json()
            current = data['current_weather']
            hourly = data['hourly']
            return {
                "temperature": current['temperature'],
                "humidity": hourly['relativehumidity_2m'][0] if hourly['relativehumidity_2m'] else 50,
                "condition": "Clear",
                "rainfall": hourly['precipitation'][0] if hourly['precipitation'] else 0,
                "wind_speed": current['windspeed'],
                "wind_direction": current['winddirection'],
                "uv_index": 5,
                "pressure": hourly['surface_pressure'][0] if hourly['surface_pressure'] else 1013,
                "feels_like": current['temperature'],
                "visibility": 10,
                "location": location_name,
                "lat": lat,
                "lon": lon,
                "timezone": data['timezone'],
                "last_updated": current['time']
            }
        else:
            st.error(f"Weather API Error: {weather_response.status_code}")
            return None
    except Exception as e:
        st.error(f"❌ Error fetching weather: {str(e)}")
        return None

@st.cache_data(ttl=3600)
def get_weather_forecast(location: str, api_key: str = None, days: int = 7) -> Optional[Dict]:
    """Fetch weather forecast from OpenMeteo API"""
    try:
        # Get coordinates from location name
        geocoding_url = f"https://geocoding-api.open-meteo.com/v1/search?name={location}&count=1&language=en&format=json"
        geo_response = requests.get(geocoding_url, timeout=10)
        
        if geo_response.status_code != 200:
            return None
            
        geo_data = geo_response.json()
        if not geo_data.get('results'):
            return None
            
        result = geo_data['results'][0]
        lat, lon = result['latitude'], result['longitude']
        location_name = f"{result['name']}, {result.get('admin1', '')}"
        
        # Get forecast data with simplified parameters
        forecast_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&daily=temperature_2m_max,temperature_2m_min,precipitation_sum&timezone=auto&forecast_days={days}"
        forecast_response = requests.get(forecast_url, timeout=10)
        
        if forecast_response.status_code == 200:
            data = forecast_response.json()
            daily = data['daily']
            
            forecast_days = []
            for i in range(len(daily['time'])):
                forecast_days.append({
                    'date': daily['time'][i],
                    'max_temp': daily['temperature_2m_max'][i],
                    'min_temp': daily['temperature_2m_min'][i],
                    'avg_temp': (daily['temperature_2m_max'][i] + daily['temperature_2m_min'][i]) / 2,
                    'condition': "Clear" if daily['precipitation_sum'][i] == 0 else "Rainy",
                    'rainfall': daily['precipitation_sum'][i],
                    'humidity': 50,
                    'wind_speed': 10,
                    'uv_index': 5
                })
            
            return {
                'location': location_name,
                'forecast': forecast_days
            }
        return None
    except Exception as e:
        return None




# --- Dataset loader and summarizer ---
@st.cache_data(ttl=3600)
def load_datasets(data_dir: str) -> Dict[str, Optional[pd.DataFrame]]:
    paths = {
        'merged': os.path.join(data_dir, 'merged_feature_store.csv'),
        'crop_rec': os.path.join(data_dir, 'Crop_recommendation.csv'),
        'multilingual': os.path.join(data_dir, 'Multilingual_Expert_Advisory.csv'),
        'smart': os.path.join(data_dir, 'Smart_Advisory_Reports_All.csv')
    }
    dfs = {}
    for key, p in paths.items():
        try:
            if os.path.exists(p):
                dfs[key] = pd.read_csv(p)
            else:
                dfs[key] = None
        except Exception:
            dfs[key] = None
    return dfs


def _match_rows_by_location(df: pd.DataFrame, state: Optional[str], district: Optional[str], crop: Optional[str], max_rows: int = 3):
    if df is None or df.empty:
        return pd.DataFrame()
    q = pd.Series([True] * len(df))
    if state and 'state' in df.columns:
        q = q & df['state'].fillna('').str.lower().str.contains(state.lower())
    if district and 'district' in df.columns:
        q = q & df['district'].fillna('').str.lower().str.contains(district.lower())
    if crop:
        # try several possible crop columns
        crop_cols = [c for c in df.columns if 'crop' in c.lower() or 'recommended' in c.lower()]
        if crop_cols:
            crop_q = pd.Series([False] * len(df))
            for ccol in crop_cols:
                crop_q = crop_q | df[ccol].fillna('').str.lower().str.contains(crop.lower())
            q = q & crop_q
    try:
        res = df[q].head(max_rows)
        return res
    except Exception:
        return pd.DataFrame()


def build_dataset_summary(dfs: Dict[str, Optional[pd.DataFrame]], location: str, question: str, max_rows: int = 3) -> str:
    """Build comprehensive dataset summary from all available datasets"""
    summary_lines = []
    
    # Extract location info
    loc_low = (location or '').lower()
    inferred_state = None
    inferred_district = None
    inferred_crop = None
    
    # Check merged dataset for location matching
    merged = dfs.get('merged')
    if merged is not None:
        # Find state
        if 'state' in merged.columns:
            states = merged['state'].dropna().unique()
            for s in states:
                if str(s).lower() in loc_low:
                    inferred_state = str(s)
                    break
        
        # Find district
        if 'district' in merged.columns:
            districts = merged['district'].dropna().unique()
            for d in districts:
                if str(d).lower() in loc_low:
                    inferred_district = str(d)
                    break
    
    # Extract crop from question
    crop_keywords = ['wheat', 'rice', 'maize', 'cotton', 'sugarcane', 'potato', 'onion', 'tomato', 'soybean', 'mustard']
    for crop in crop_keywords:
        if crop in question.lower():
            inferred_crop = crop
            break
    
    # Get relevant data from each dataset
    datasets_info = []
    
    # 1. Merged feature store
    if merged is not None:
        query = pd.Series([True] * len(merged))
        if inferred_state and 'state' in merged.columns:
            query = query & merged['state'].str.contains(inferred_state, case=False, na=False)
        if inferred_district and 'district' in merged.columns:
            query = query & merged['district'].str.contains(inferred_district, case=False, na=False)
        
        matches = merged[query].head(max_rows)
        if not matches.empty:
            datasets_info.append("**Merged Dataset:**")
            for _, row in matches.iterrows():
                info = []
                if 'state' in row: info.append(f"State: {row['state']}")
                if 'district' in row: info.append(f"District: {row['district']}")
                if 'recommended_crop' in row: info.append(f"Recommended: {row['recommended_crop']}")
                if 'soil_ph' in row: info.append(f"Soil pH: {row['soil_ph']}")
                if 'rainfall_imd_mm' in row: info.append(f"Rainfall: {row['rainfall_imd_mm']}mm")
                datasets_info.append(" | ".join(info))
    
    # 2. Crop recommendation dataset
    crop_rec = dfs.get('crop_rec')
    if crop_rec is not None and not crop_rec.empty:
        datasets_info.append("\n**Crop Recommendations:**")
        sample = crop_rec.head(2)
        for _, row in sample.iterrows():
            info = []
            if 'label' in row: info.append(f"Crop: {row['label']}")
            if 'N' in row: info.append(f"N: {row['N']}")
            if 'P' in row: info.append(f"P: {row['P']}")
            if 'K' in row: info.append(f"K: {row['K']}")
            if 'ph' in row: info.append(f"pH: {row['ph']}")
            if 'rainfall' in row: info.append(f"Rainfall: {row['rainfall']}mm")
            datasets_info.append(" | ".join(info))
    
    # 3. Expert advisory dataset
    advisory = dfs.get('multilingual')
    if advisory is not None and not advisory.empty:
        datasets_info.append("\n**Expert Advisory:**")
        # Look for relevant advisories
        if inferred_crop:
            crop_advisories = advisory[advisory.astype(str).apply(lambda x: x.str.contains(inferred_crop, case=False, na=False)).any(axis=1)]
            if not crop_advisories.empty:
                sample = crop_advisories.head(2)
                for _, row in sample.iterrows():
                    if 'advisory_text' in row:
                        datasets_info.append(f"Advisory: {str(row['advisory_text'])[:100]}...")
                    elif 'recommendation' in row:
                        datasets_info.append(f"Recommendation: {str(row['recommendation'])[:100]}...")
    
    # 4. Smart advisory reports
    smart = dfs.get('smart')
    if smart is not None and not smart.empty:
        datasets_info.append("\n**Smart Advisory:**")
        if inferred_state or inferred_district:
            query = pd.Series([True] * len(smart))
            if inferred_state:
                for col in smart.columns:
                    if 'state' in col.lower():
                        query = query & smart[col].str.contains(inferred_state, case=False, na=False)
                        break
            
            matches = smart[query].head(2)
            if not matches.empty:
                for _, row in matches.iterrows():
                    info = []
                    for col in ['crop', 'advisory', 'recommendation', 'season']:
                        if col in row and pd.notna(row[col]):
                            info.append(f"{col.title()}: {str(row[col])[:50]}")
                    if info:
                        datasets_info.append(" | ".join(info))
    
    return "\n".join(datasets_info) if datasets_info else ""

@st.cache_data(ttl=86400)
def get_nasa_power_data(lat: float, lon: float) -> Optional[Dict]:
    """Fetch soil and climate data from NASA POWER API"""
    try:
        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=30)).strftime("%Y%m%d")
        
        url = f"https://power.larc.nasa.gov/api/temporal/daily/point"
        params = {
            "parameters": "T2M,PRECTOTCORR,RH2M,WS2M,ALLSKY_SFC_SW_DWN",
            "community": "AG",
            "longitude": lon,
            "latitude": lat,
            "start": start_date,
            "end": end_date,
            "format": "JSON"
        }
        
        response = requests.get(url, params=params, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            properties = data.get('properties', {}).get('parameter', {})
            
            if properties:
                temp_data = properties.get('T2M', {})
                precip_data = properties.get('PRECTOTCORR', {})
                humidity_data = properties.get('RH2M', {})
                wind_data = properties.get('WS2M', {})
                solar_data = properties.get('ALLSKY_SFC_SW_DWN', {})
                
                recent_dates = sorted(temp_data.keys())[-7:] if temp_data else []
                
                if recent_dates:
                    avg_temp = np.mean([temp_data.get(d, 0) for d in recent_dates])
                    avg_precip = np.mean([precip_data.get(d, 0) for d in recent_dates])
                    avg_humidity = np.mean([humidity_data.get(d, 0) for d in recent_dates])
                    avg_wind = np.mean([wind_data.get(d, 0) for d in recent_dates])
                    avg_solar = np.mean([solar_data.get(d, 0) for d in recent_dates])
                    
                    return {
                        "avg_temperature_30d": avg_temp,
                        "avg_precipitation_30d": avg_precip,
                        "avg_humidity_30d": avg_humidity,
                        "avg_wind_speed_30d": avg_wind,
                        "avg_solar_radiation_30d": avg_solar,
                        "data_points": len(recent_dates),
                        "date_range": f"{recent_dates[0]} to {recent_dates[-1]}"
                    }
        return None
    except Exception as e:
        st.warning(f"⚠️ NASA POWER API unavailable: {str(e)}")
        return None

def load_lora_model(model_path: str):
    """Load LoRA adapter model"""
    if not TRANSFORMERS_AVAILABLE:
        return None, None
    
    try:
        base_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        full_path = os.path.join("models", model_path)
        
        if not os.path.exists(full_path):
            return None, None
        
        # Try loading tokenizer from base model (more reliable)
        try:
            tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        except:
            tokenizer = AutoTokenizer.from_pretrained(full_path)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map="auto" if device == "cuda" else None,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            low_cpu_mem_usage=True
        )
        
        if device == "cpu":
            base_model = base_model.to(device)
        
        # Clean adapter config if needed
        config_path = os.path.join(full_path, "adapter_config.json")
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    adapter_config = json.load(f)
                unsupported_fields = ['corda_config', 'eva_config', 'megatron_config', 'megatron_core']
                cleaned_config = {k: v for k, v in adapter_config.items() 
                                if k not in unsupported_fields and v is not None}
                with open(config_path, 'w', encoding='utf-8') as f:
                    json.dump(cleaned_config, f, indent=2)
            except:
                pass
        
        model = PeftModel.from_pretrained(base_model, full_path)
        model.eval()
        
        return model, tokenizer
    except Exception as e:
        return None, None

def generate_with_lora(prompt: str, model, tokenizer, max_tokens: int = 1000) -> str:
    """Generate response using LoRA model"""
    try:
        formatted_prompt = f"<|system|>\nYou are an expert agricultural advisor.\n<|user|>\n{prompt}\n<|assistant|>\n"
        
        device = next(model.parameters()).device
        inputs = tokenizer(formatted_prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        return response
    except Exception as e:
        return None

def get_ollama_recommendation(prompt: str, model: str, temp: float, max_tokens: int, host: str) -> Optional[str]:
    """Get recommendation from Ollama API"""
    try:
        data = {
            "model": model,
            "prompt": f"You are an expert agricultural advisor specialized in climate-resilient farming. Analyze soil conditions, weather data, and recommend the most suitable crop with climate adaptation strategies. Provide practical, actionable advice for farmers.\n\n{prompt}",
            "stream": False,
            "options": {
                "temperature": temp,
                "num_predict": max_tokens
            }
        }
        
        # Timeout reduced to 45s - Groq will be used as backup if Ollama is too slow
        response = requests.post(
            f"{host}/api/generate",
            json=data,
            timeout=45
        )
        
        if response.status_code == 200:
            result = response.json()
            generated_text = result.get('response', '')
            if generated_text:
                return generated_text
            else:
                st.warning("⚠️ Ollama returned empty response")
                return None
        else:
            st.error(f"❌ Ollama API Error: Status {response.status_code} - {response.text[:200]}")
            return None
    except requests.exceptions.Timeout:
        # Silent timeout - Groq will handle as fallback
        return None
    except requests.exceptions.ConnectionError:
        # Silent connection error - Groq will handle as fallback
        return None
    except Exception as e:
        st.error(f"❌ Error getting recommendation: {str(e)}")
        return None

def get_groq_recommendation(prompt: str, api_key: str = None, system_prompt: str = None) -> Optional[str]:
    """Get recommendation from Groq API as synthesis engine"""
    if not api_key or api_key == "ENTER_YOUR_GROQ_API_KEY":
        return None
    
    try:
        import groq
        client = groq.Groq(api_key=api_key)
        
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": system_prompt if system_prompt else "You are a specialized agricultural synthesis engine for Indian farmers. Your goal is to provide evidence-based, factual advice."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            max_tokens=1500,
        )
        
        return chat_completion.choices[0].message.content
    except Exception as e:
        return None

def translate_text(text: str, target_lang: str, api_key: str) -> Dict[str, str]:
    """Translate text to target language, providing both a full detailed version and a concise voice summary."""
    lang_info = LANG_MAP.get(target_lang, {"name": "English", "code": "en"})
    lang_name = lang_info["name"]
    
    # Custom instructions for tone
    tone_instruction = f"Use a professional yet RELATABLE tone suitable for an Indian farmer."
    if lang_name == "Hindi":
        tone_instruction = "Use HINGLISH (a natural mixture of Hindi and English words like 'aap fertilizer use karein' or 'soil health check karein')."
    elif lang_name == "English":
        tone_instruction = "Use simple, clear English that is easy for a farmer to understand."
    
    try:
        import groq
        client = groq.Groq(api_key=api_key)
        
        prompt = f"""
        TASK: Process the agricultural advisory below into {lang_name}.
        
        STYLE: {tone_instruction}
        
        OUTPUT FORMAT (Use EXACT tags):
        [DETAILED]
        (Provide the FULL, professional translation/adaptation with all headers and details preserved)
        
        [VOICE_SUMMARY]
        (Provide a very CONCISE summary, MAX 120 words, for voice reading in under 1 minute)
        
        TEXT TO PROCESS:
        {text}"""
        
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0.3,
        )
        
        content = chat_completion.choices[0].message.content
        
        # Parse the two versions
        detailed = ""
        summary = ""
        
        if "[DETAILED]" in content and "[VOICE_SUMMARY]" in content:
            parts = content.split("[VOICE_SUMMARY]")
            detailed = parts[0].replace("[DETAILED]", "").strip()
            summary = parts[1].strip()
        else:
            # Fallback if model doesn't follow format strictly
            detailed = content
            summary = content[:500] + "..." if len(content) > 500 else content
            
        return {"detailed": detailed, "summary": summary}
        
    except Exception as e:
        # Fallback for errors
        return {"detailed": text, "summary": text[:300] + "..." if len(text) > 300 else text}

def speak_text(text: str, lang_name: str):
    """Generate and play audio for the translated text with Indian accent"""
    try:
        lang_info = LANG_MAP.get(lang_name, LANG_MAP["English"])
        # Clean text for speech
        import re
        clean_text = re.sub(r'[#\*]', '', text)
        
        tts = gTTS(text=clean_text, lang=lang_info["code"], tld=lang_info["tld"], slow=False)
        
        audio_fp = io.BytesIO()
        tts.write_to_fp(audio_fp)
        audio_fp.seek(0)
        
        st.audio(audio_fp, format='audio/mp3')
    except Exception as e:
        pass

def create_weather_visualization(weather_data: Dict):
    """Create comprehensive weather visualization"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Temperature', 'Humidity', 'Wind Speed', 'UV Index'),
        specs=[[{"type": "indicator"}, {"type": "indicator"}],
               [{"type": "indicator"}, {"type": "indicator"}]]
    )
    
    # Temperature gauge
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=weather_data['temperature'],
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "°C"},
            gauge={
                'axis': {'range': [None, 50]},
                'bar': {'color': "darkgreen"},
                'steps': [
                    {'range': [0, 15], 'color': "lightblue"},
                    {'range': [15, 30], 'color': "yellow"},
                    {'range': [30, 50], 'color': "orange"}
                ]
            }
        ),
        row=1, col=1
    )
    
    # Humidity gauge
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=weather_data['humidity'],
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "%"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"}
            }
        ),
        row=1, col=2
    )
    
    # Wind Speed gauge
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=weather_data['wind_speed'],
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "km/h"},
            gauge={
                'axis': {'range': [None, 50]},
                'bar': {'color': "darkgreen"}
            }
        ),
        row=2, col=1
    )
    
    # UV Index gauge
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=weather_data['uv_index'],
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "UV Index"},
            gauge={
                'axis': {'range': [None, 11]},
                'bar': {'color': "darkred"},
                'steps': [
                    {'range': [0, 3], 'color': "green"},
                    {'range': [3, 6], 'color': "yellow"},
                    {'range': [6, 8], 'color': "orange"},
                    {'range': [8, 11], 'color': "red"}
                ]
            }
        ),
        row=2, col=2
    )
    
    fig.update_layout(height=600, showlegend=False, title_text="Real-Time Weather Metrics")
    return fig

def create_soil_visualization(soil_params: Dict):
    """Create soil nutrients visualization"""
    nutrients = ['Nitrogen (N)', 'Phosphorus (P)', 'Potassium (K)']
    values = [soil_params['N'], soil_params['P'], soil_params['K']]
    
    fig = go.Figure(data=[
        go.Bar(
            x=nutrients,
            y=values,
            marker_color=['#4CAF50', '#2196F3', '#FF9800'],
            text=values,
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Soil Nutrient Levels (kg/hectare)",
        xaxis_title="Nutrients",
        yaxis_title="Amount (kg/hectare)",
        height=400,
        template="plotly_white"
    )
    
    return fig

def create_dataset_visualizations(dfs: Dict[str, Optional[pd.DataFrame]], location: str):
    """Create visualizations from dataset analysis"""
    figs = []
    
    # 1. Crop distribution from merged dataset
    merged = dfs.get('merged')
    if merged is not None and 'recommended_crop' in merged.columns:
        crop_counts = merged['recommended_crop'].value_counts().head(10)
        fig1 = px.bar(x=crop_counts.index, y=crop_counts.values, 
                     title="Top 10 Recommended Crops", 
                     labels={'x': 'Crops', 'y': 'Frequency'})
        figs.append(fig1)
    
    # 2. NPK distribution from crop recommendation dataset
    crop_rec = dfs.get('crop_rec')
    if crop_rec is not None and all(col in crop_rec.columns for col in ['N', 'P', 'K']):
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=crop_rec['N'], y=crop_rec['P'], 
                                 mode='markers', name='N vs P',
                                 marker=dict(size=crop_rec['K']/10, opacity=0.6)))
        fig2.update_layout(title="NPK Distribution in Crops", 
                          xaxis_title="Nitrogen (N)", yaxis_title="Phosphorus (P)")
        figs.append(fig2)
    
    # 3. Rainfall vs pH correlation
    if merged is not None and all(col in merged.columns for col in ['rainfall_imd_mm', 'soil_ph']):
        fig3 = px.scatter(merged, x='rainfall_imd_mm', y='soil_ph', 
                         title="Rainfall vs Soil pH",
                         labels={'rainfall_imd_mm': 'Rainfall (mm)', 'soil_ph': 'Soil pH'})
        figs.append(fig3)
    
    return figs

def create_location_specific_charts(dfs: Dict[str, Optional[pd.DataFrame]], state: str, district: str = None):
    """Create location-specific visualizations"""
    charts = []
    
    merged = dfs.get('merged')
    if merged is not None:
        # Filter by location
        filtered_data = merged.copy()
        if state and 'state' in merged.columns:
            filtered_data = filtered_data[filtered_data['state'].str.contains(state, case=False, na=False)]
        if district and 'district' in merged.columns:
            filtered_data = filtered_data[filtered_data['district'].str.contains(district, case=False, na=False)]
        
        if not filtered_data.empty:
            # Crop suitability chart
            if 'recommended_crop' in filtered_data.columns:
                crop_dist = filtered_data['recommended_crop'].value_counts().head(8)
                fig = px.pie(values=crop_dist.values, names=crop_dist.index,
                           title=f"Crop Distribution in {state}")
                charts.append(fig)
            
            # Soil parameters heatmap
            if all(col in filtered_data.columns for col in ['soil_ph', 'rainfall_imd_mm']):
                fig = px.histogram_2d(filtered_data, x='soil_ph', y='rainfall_imd_mm',
                                    title=f"Soil pH vs Rainfall Distribution - {state}")
                charts.append(fig)
    
    return charts

def create_ph_visualization(ph_value: float):
    """Create pH level visualization"""
    fig = go.Figure()
    
    ph_scale = np.arange(3, 11, 0.1)
    colors = ['red' if x < 5.5 else 'orange' if x < 6.5 else 'green' if x < 7.5 else 'orange' if x < 8.5 else 'red' for x in ph_scale]
    
    fig.add_trace(go.Scatter(
        x=ph_scale,
        y=[1]*len(ph_scale),
        mode='markers',
        marker=dict(size=10, color=colors),
        name='pH Scale'
    ))
    
    fig.add_trace(go.Scatter(
        x=[ph_value],
        y=[1],
        mode='markers',
        marker=dict(size=30, color='darkblue', symbol='diamond'),
        name=f'Current pH: {ph_value}'
    ))
    
    fig.update_layout(
        title="Soil pH Level",
        xaxis_title="pH Value",
        yaxis=dict(showticklabels=False, range=[0.5, 1.5]),
        height=200,
        template="plotly_white"
    )
    
    return fig

def create_location_specific_charts(dfs: Dict[str, Optional[pd.DataFrame]], state: str, district: str = None):
    """Create location-specific visualizations"""
    charts = []
    
    merged = dfs.get('merged')
    if merged is not None:
        # Filter by location
        filtered_data = merged.copy()
        if state and 'state' in merged.columns:
            filtered_data = filtered_data[filtered_data['state'].str.contains(state, case=False, na=False)]
        if district and 'district' in merged.columns:
            filtered_data = filtered_data[filtered_data['district'].str.contains(district, case=False, na=False)]
        
        if not filtered_data.empty:
            # Crop suitability chart
            if 'recommended_crop' in filtered_data.columns:
                crop_dist = filtered_data['recommended_crop'].value_counts().head(8)
                fig = px.pie(values=crop_dist.values, names=crop_dist.index,
                           title=f"Crop Distribution in {state}")
                charts.append(fig)
            
            # Soil parameters scatter plot
            if all(col in filtered_data.columns for col in ['soil_ph', 'rainfall_imd_mm']):
                fig = px.scatter(filtered_data, x='soil_ph', y='rainfall_imd_mm',
                               title=f"Soil pH vs Rainfall Distribution - {state}")
                charts.append(fig)
    
    return charts

def calculate_soil_health_score(soil_params: Dict) -> Dict:
    """Calculate soil health score based on NPK and pH"""
    n, p, k, ph = soil_params['N'], soil_params['P'], soil_params['K'], soil_params['pH']
    
    # Optimal ranges
    n_optimal = (40, 80)
    p_optimal = (20, 50)
    k_optimal = (100, 200)
    ph_optimal = (6.0, 7.5)
    
    # Calculate scores (0-100)
    def score_value(value, optimal_range, weight=1.0):
        min_val, max_val = optimal_range
        if min_val <= value <= max_val:
            return 100 * weight
        elif value < min_val:
            return max(0, (value / min_val) * 100 * weight)
        else:
            return max(0, (max_val / value) * 100 * weight)
    
    n_score = score_value(n, n_optimal, 0.25)
    p_score = score_value(p, p_optimal, 0.25)
    k_score = score_value(k, k_optimal, 0.25)
    ph_score = score_value(ph, ph_optimal, 0.25)
    
    total_score = n_score + p_score + k_score + ph_score
    
    # Determine health level
    if total_score >= 80:
        level = "Excellent"
        color = "green"
    elif total_score >= 60:
        level = "Good"
        color = "blue"
    elif total_score >= 40:
        level = "Fair"
        color = "orange"
    else:
        level = "Poor"
        color = "red"
    
    return {
        'total_score': round(total_score, 1),
        'level': level,
        'color': color,
        'breakdown': {
            'N': round(n_score, 1),
            'P': round(p_score, 1),
            'K': round(k_score, 1),
            'pH': round(ph_score, 1)
        },
        'recommendations': get_soil_recommendations(soil_params, total_score)
    }

def get_soil_recommendations(soil_params: Dict, score: float) -> list:
    """Get recommendations based on soil health score"""
    recommendations = []
    n, p, k, ph = soil_params['N'], soil_params['P'], soil_params['K'], soil_params['pH']
    
    if n < 40:
        recommendations.append("Add nitrogen-rich fertilizers (urea, ammonium sulfate)")
    elif n > 80:
        recommendations.append("Nitrogen levels are high - reduce nitrogen inputs")
    
    if p < 20:
        recommendations.append("Add phosphorus fertilizers (superphosphate, bone meal)")
    elif p > 50:
        recommendations.append("Phosphorus levels are adequate - maintain current levels")
    
    if k < 100:
        recommendations.append("Add potassium fertilizers (potash, wood ash)")
    elif k > 200:
        recommendations.append("Potassium levels are high - reduce potassium inputs")
    
    if ph < 6.0:
        recommendations.append("Soil is acidic - add lime to raise pH")
    elif ph > 7.5:
        recommendations.append("Soil is alkaline - add sulfur or organic matter to lower pH")
    
    if score < 40:
        recommendations.append("Consider soil testing and professional consultation")
        recommendations.append("Add organic matter (compost, manure) to improve overall soil health")
    
    return recommendations


def render_domain_visuals(
    domain_visuals: list,
    crop_matches_df,
    agri_metrics: dict,
    weather_data: dict,
    soil_params: dict,
    district: str,
    state: str,
    crop: str,
):
    """
    Renders domain-specific charts inside the analysis expander.
    domain_visuals: list of chart-key strings from get_domain_visuals()
    """
    import plotly.graph_objects as go
    import plotly.express as px
    import pandas as pd

    df = crop_matches_df

    # Helper: ensure numeric series
    def numeric(series, col):
        if df is not None and not df.empty and col in df.columns:
            return pd.to_numeric(df[col], errors="coerce").dropna()
        return pd.Series([], dtype=float)

    charts_rendered = 0

    for chart_key in domain_visuals:

        # ── NPK Bar ──────────────────────────────────────────
        if chart_key == "npk_bar":
            n = soil_params.get("N", 50)
            p = soil_params.get("P", 30)
            k = soil_params.get("K", 100)
            fig = go.Figure(go.Bar(
                x=["Nitrogen (N)", "Phosphorus (P)", "Potassium (K)"],
                y=[n, p, k],
                marker_color=["#4CAF50", "#2196F3", "#FF9800"],
                text=[f"{n:.0f}", f"{p:.0f}", f"{k:.0f}"],
                textposition="auto",
            ))
            fig.update_layout(title=f"Soil NPK — {district}", yaxis_title="kg/ha",
                              height=320, template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
            charts_rendered += 1

        # ── NPK Radar ────────────────────────────────────────
        elif chart_key == "npk_radar":
            n = min(100, soil_params.get("N", 50) / 1.5)
            p = min(100, soil_params.get("P", 30) / 0.8)
            k = min(100, soil_params.get("K", 100) / 2.5)
            ph_n = min(100, max(0, (soil_params.get("pH", 6.5) - 4) / 5 * 100))
            sm = min(100, agri_metrics.get("soil_moisture", 20) * 2.5)
            cats = ["Nitrogen", "Phosphorus", "Potassium", "pH", "Moisture"]
            vals = [n, p, k, ph_n, sm]
            fig = go.Figure(go.Scatterpolar(
                r=vals + [vals[0]], theta=cats + [cats[0]],
                fill="toself", fillcolor="rgba(34,139,34,0.2)",
                line=dict(color="#228B22"),
            ))
            fig.update_layout(title=f"Soil Profile Radar — {district}",
                              polar=dict(radialaxis=dict(range=[0, 100])),
                              height=350)
            st.plotly_chart(fig, use_container_width=True)
            charts_rendered += 1

        # ── NDVI Trend ───────────────────────────────────────
        elif chart_key in ("ndvi_trend", "ndvi_decline_trend"):
            n_series = numeric(df, "NDVI_Vegetation_Index")
            if not n_series.empty:
                years = numeric(df, "Year").astype(int).values if df is not None and "Year" in df.columns else list(range(len(n_series)))
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=years, y=n_series.values,
                                         mode="lines+markers",
                                         line=dict(color="#4CAF50", width=2),
                                         name="NDVI"))
                fig.add_hline(y=0.4, line_dash="dash", line_color="orange",
                               annotation_text="Fair threshold 0.4")
                fig.add_hline(y=0.6, line_dash="dot", line_color="green",
                               annotation_text="Good threshold 0.6")
                fig.update_layout(title=f"Historical NDVI Trend — {district}",
                                  yaxis_title="NDVI", height=320,
                                  template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)
                charts_rendered += 1

        # ── MSP Trend ────────────────────────────────────────
        elif chart_key == "msp_trend":
            m_series = numeric(df, "Historical_MSP_INR")
            if not m_series.empty:
                years = numeric(df, "Year").astype(int).values if df is not None and "Year" in df.columns else list(range(len(m_series)))
                fig = go.Figure(go.Scatter(
                    x=years, y=m_series.values,
                    mode="lines+markers",
                    line=dict(color="#FF9800", width=2),
                    fill="tozeroy", fillcolor="rgba(255,152,0,0.1)",
                ))
                fig.update_layout(title=f"MSP Trend for {crop} — {district}",
                                  yaxis_title="₹ / quintal", height=320,
                                  template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)
                charts_rendered += 1

        # ── 10-Year Temperature Trend ─────────────────────────
        elif chart_key == "temp_trend_10yr":
            t_series = numeric(df, "Mean_Temp_Historical")
            if not t_series.empty:
                years = numeric(df, "Year").astype(int).values if df is not None and "Year" in df.columns else list(range(len(t_series)))
                import numpy as np
                if len(years) >= 2:
                    z = np.polyfit(list(range(len(t_series))), t_series.values, 1)
                    trend_line = np.poly1d(z)(range(len(t_series)))
                else:
                    trend_line = t_series.values
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=years, y=t_series.values,
                                         mode="lines+markers", name="Actual Temp",
                                         line=dict(color="#FF7043")))
                fig.add_trace(go.Scatter(x=years, y=trend_line,
                                         mode="lines", name="Trend",
                                         line=dict(color="red", dash="dash")))
                fig.add_hline(y=38, line_dash="dot", line_color="darkred",
                               annotation_text="Heat stress 38°C")
                fig.update_layout(title=f"Temperature Trend — {district}",
                                  yaxis_title="°C", height=320, template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)
                charts_rendered += 1

        # ── 10-Year Rainfall Trend ────────────────────────────
        elif chart_key == "rainfall_trend_10yr":
            r_series = numeric(df, "Rainfall_IMD_mm")
            if not r_series.empty:
                years = numeric(df, "Year").astype(int).values if df is not None and "Year" in df.columns else list(range(len(r_series)))
                fig = go.Figure(go.Bar(
                    x=years, y=r_series.values,
                    marker_color=["#4FC3F7" if v >= r_series.mean() else "#FF7043" for v in r_series.values],
                    name="Rainfall",
                ))
                fig.add_hline(y=r_series.mean(), line_dash="dash", line_color="navy",
                               annotation_text=f"Avg {r_series.mean():.0f}mm")
                fig.update_layout(title=f"Rainfall Trend — {district}",
                                  yaxis_title="mm", height=320, template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)
                charts_rendered += 1

        # ── Rainfall Forecast vs Historical ──────────────────
        elif chart_key == "rainfall_forecast_vs_hist":
            r_series = numeric(df, "Rainfall_IMD_mm")
            precip_7day = agri_metrics.get("precip_7day") or 0.0
            hist_avg = r_series.mean() if not r_series.empty else 200
            fig = go.Figure(go.Bar(
                x=["Historical Monthly Avg", "7-Day Forecast"],
                y=[hist_avg, precip_7day],
                marker_color=["#4FC3F7", "#FF7043" if precip_7day < hist_avg * 0.3 else "#4CAF50"],
                text=[f"{hist_avg:.0f}mm", f"{precip_7day:.0f}mm"],
                textposition="auto",
            ))
            fig.update_layout(title=f"Forecast vs Historical Rainfall — {district}",
                               yaxis_title="mm", height=300, template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
            charts_rendered += 1

        # ── Flood Risk Gauge ─────────────────────────────────
        elif chart_key == "flood_risk_gauge":
            precip_7day = agri_metrics.get("precip_7day") or 0.0
            r_series = numeric(df, "Rainfall_IMD_mm")
            flood_thresh = r_series.quantile(0.85) if len(r_series) >= 5 else 300
            risk_pct = min(95, (precip_7day / max(1, flood_thresh)) * 100)
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=risk_pct,
                title={"text": "Flood Risk Score (%)"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#dc3545" if risk_pct > 60 else "#ffc107" if risk_pct > 30 else "#28a745"},
                    "steps": [
                        {"range": [0, 30],  "color": "#d4edda"},
                        {"range": [30, 60], "color": "#fff3cd"},
                        {"range": [60, 100],"color": "#f8d7da"},
                    ],
                    "threshold": {"line": {"color": "red", "width": 3}, "value": 60},
                },
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            charts_rendered += 1

        # ── Humidity/Temp Pest Risk ───────────────────────────
        elif chart_key == "humidity_temp_risk":
            humidity = agri_metrics.get("soil_moisture") or 60.0
            temp = agri_metrics.get("soil_temp") or weather_data.get("temperature", 30) if weather_data else 30
            fungal_risk = min(100, max(0, (humidity - 50) * 1.5 + (temp - 20) * 1.2))
            pest_risk   = min(100, max(0, 100 - humidity * 0.3 + (temp - 25) * 2))
            fig = go.Figure(go.Bar(
                x=["Fungal Disease Risk", "Pest Pressure"],
                y=[fungal_risk, pest_risk],
                marker_color=["#dc3545" if fungal_risk > 60 else "#ffc107",
                               "#dc3545" if pest_risk > 60 else "#ffc107"],
                text=[f"{fungal_risk:.0f}%", f"{pest_risk:.0f}%"],
                textposition="auto",
            ))
            fig.update_layout(title=f"Pest & Disease Risk — {district}",
                               yaxis_title="Risk %", yaxis_range=[0, 100],
                               height=300, template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
            charts_rendered += 1

        # ── Investment Allocation Pie ─────────────────────────
        elif chart_key == "investment_allocation_pie":
            fig = go.Figure(go.Pie(
                labels=["Drip Kit", "Quality Seeds", "Balanced Fertiliser", "Soil Sensor", "Soil Test"],
                values=[15000, 8000, 12000, 4500, 500],
                hole=0.35,
                marker_colors=["#4CAF50","#2196F3","#FF9800","#9C27B0","#009688"],
            ))
            fig.update_layout(title="₹40,000 Optimal Investment Allocation",
                               height=340)
            st.plotly_chart(fig, use_container_width=True)
            charts_rendered += 1

        # ── Mixed Crop Pie ────────────────────────────────────
        elif chart_key == "mixed_crop_pie":
            fig = go.Figure(go.Pie(
                labels=["Main Cash Crop (60%)", "Drought-Safe Backup (40%)"],
                values=[60, 40],
                marker_colors=["#4CAF50", "#FF9800"],
                hole=0.4,
            ))
            fig.update_layout(title="Recommended Land Allocation (2 ha)",
                               height=300)
            st.plotly_chart(fig, use_container_width=True)
            charts_rendered += 1

        # ── Crop Water Comparison Bar ─────────────────────────
        elif chart_key == "crop_water_comparison":
            crops_w = ["Pearl Millet", "Chickpea", "Sorghum", "Sesame", "Maize", "Wheat", "Rice", "Sugarcane"]
            water_w = [250, 350, 300, 300, 500, 500, 1200, 2000]
            colors_w = ["#4CAF50" if w < 400 else "#FF9800" if w < 800 else "#dc3545" for w in water_w]
            fig = go.Figure(go.Bar(
                x=crops_w, y=water_w,
                marker_color=colors_w,
                text=water_w, textposition="auto",
            ))
            fig.update_layout(title="Water Requirement Comparison (mm/season)",
                               yaxis_title="mm", height=340, template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
            charts_rendered += 1

        # ── pH Trend ─────────────────────────────────────────
        elif chart_key == "ph_trend":
            ph_series = numeric(df, "Soil_pH")
            if not ph_series.empty:
                years = numeric(df, "Year").astype(int).values if df is not None and "Year" in df.columns else list(range(len(ph_series)))
                fig = go.Figure(go.Scatter(
                    x=years, y=ph_series.values,
                    mode="lines+markers",
                    line=dict(color="#9C27B0", width=2),
                ))
                fig.add_hrect(y0=6.0, y1=7.5, fillcolor="rgba(76,175,80,0.1)",
                               annotation_text="Optimal pH 6.0–7.5")
                fig.update_layout(title=f"Soil pH Trend — {district}",
                                  yaxis_title="pH", height=300,
                                  template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)
                charts_rendered += 1

        # ── ROI Waterfall ─────────────────────────────────────
        elif chart_key == "roi_waterfall":
            m_series = numeric(df, "Historical_MSP_INR")
            msp_val  = m_series.iloc[-1] if not m_series.empty else 2000
            typical_yield = 30
            gross    = msp_val * typical_yield
            input_c  = 25000
            net      = gross - input_c
            fig = go.Figure(go.Waterfall(
                x=["MSP × Yield", "Input Cost", "Net Profit"],
                measure=["absolute", "relative", "total"],
                y=[gross, -input_c, net],
                text=[f"₹{gross:.0f}", f"-₹{input_c}", f"₹{net:.0f}"],
                connector={"line": {"color": "rgb(63,63,63)"}},
                increasing={"marker": {"color": "#4CAF50"}},
                decreasing={"marker": {"color": "#dc3545"}},
                totals={"marker": {"color": "#2196F3"}},
            ))
            fig.update_layout(title=f"ROI Breakdown — {crop} in {district}",
                               yaxis_title="₹ per hectare", height=320,
                               template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
            charts_rendered += 1

    if charts_rendered == 0:
        st.caption("(No additional domain charts available for this query)")

# Main Interface
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "💬 Chatbot & ML Predictions", 
    "🌍 Location Analysis", 
    "📊 Visualizations", 
    "🌡️ Weather Forecast",
    "🌾 Crop Recommendations",
    "📜 History & Export"
])

with tab1:
    col_h1, col_h2 = st.columns([5, 1])
    with col_h1:
        st.header("🤖 AI Advisor + ML Predictions")
    with col_h2:
        if st.button("🔄 Reload", help="Reload CSV to catch updates"):
            features_df, advisory_df, state_district_mapping = load_csv_data()
            st.success("✅ Reloaded!")
    
    # Input Parameters
    # Selection Row (State, District, Crop)
    sel_col1, sel_col2, sel_col3 = st.columns(3)
    
    with sel_col1:
        # State selection from CSV
        if state_district_mapping:
            available_states = sorted(state_district_mapping.keys())
            state = st.selectbox("🏞️ State", available_states, 
                                index=available_states.index('Tamil Nadu') if 'Tamil Nadu' in available_states else 0)
        else:
            state = "Tamil Nadu"
            st.info("📊 Using fallback state")

    with sel_col2:
        # District selection based on selected state
        if state_district_mapping and state in state_district_mapping:
            available_districts = state_district_mapping[state]
            district = st.selectbox("📍 District", available_districts)
        else:
            district = st.text_input("District", "Chennai")
            
        # Update location based on state and district
        location = f"{district}, {state}, India"
        st.caption(f"📌 Location: {location}")

    with sel_col3:
        # Crop selection
        if not advisory_df.empty and 'Recommended_Crop' in advisory_df.columns:
            # Get crops for selected state and district
            location_crops_df = advisory_df[(advisory_df['State'] == state) & (advisory_df['District'] == district)]
            if not location_crops_df.empty:
                available_crops = location_crops_df['Recommended_Crop'].unique()
                crop = st.selectbox("🌾 Crop", available_crops)
            else:
                common_crops = ["Rice", "Wheat", "Cotton", "Sugarcane", "Maize", "Potato", "Onion", "Tomato", "Soybean"]
                crop = st.selectbox("🌾 Crop", common_crops)
        else:
            common_crops = ["Rice", "Wheat", "Cotton", "Sugarcane", "Maize", "Potato", "Onion", "Tomato", "Soybean"]
            crop = st.selectbox("🌾 Crop", common_crops)
    
    # --- AUTOMATIC ENVIRONMENT DETECTION & WEATHER FETCHING ---
    with st.spinner("🔍 Detecting local environment..."):
        # 1. Get coordinates from CSV for selected location
        location_lat = None
        location_lon = None
        if not advisory_df.empty and 'state' in locals() and 'district' in locals():
            location_data_df = advisory_df[(advisory_df['State'] == state) & (advisory_df['District'] == district)]
            if not location_data_df.empty:
                location_lat = location_data_df.iloc[0].get('Lat', None)
                location_lon = location_data_df.iloc[0].get('Lon', None)
        
        # 2. Get live weather data
        weather_data = None
        if location_lat and location_lon:
            weather_data = get_weather_data_by_coords(location_lat, location_lon, location)
        elif location:
            weather_data = get_weather_data(location)
            
        if weather_data:
            st.session_state.location_data = weather_data
        
        # 3. Get Keyless Satellite Agri Metrics (NDVI & Soil)
        agri_metrics = {"ndvi": None, "soil_moisture": None, "soil_temp": None}
        if location_lat and location_lon:
            agri_metrics = get_keyless_agri_metrics(location_lat, location_lon)
            st.session_state.agri_metrics = agri_metrics
        
        # 4. Calculate optimized environmental parameters
        # Default fallbacks
        r_val, t_val, n_val, p_val, k_val, ph_val = 120.0, 30.0, 50.0, 50.0, 50.0, 6.5
        
        if not advisory_df.empty:
            # Clean district/state names to prevent filtering failures
            clean_state = state.strip()
            clean_district = district.strip()
            
            loc_data = advisory_df[
                (advisory_df['State'].str.strip() == clean_state) & 
                (advisory_df['District'].str.strip() == clean_district)
            ]
            
            if not loc_data.empty:
                # Narrow by crop if possible for better historical context
                crop_specific = loc_data[loc_data['Recommended_Crop'].str.contains(crop, case=False, na=False)]
                source_df = crop_specific if not crop_specific.empty else loc_data
                
                # Fetch averages from dataset with robust numeric conversion
                r_val = pd.to_numeric(source_df['Rainfall_IMD_mm'], errors='coerce').mean()
                t_val = pd.to_numeric(source_df['Mean_Temp_Historical'], errors='coerce').mean()
                ph_val = pd.to_numeric(source_df['Soil_pH'], errors='coerce').mean()
                
                # Handle NPK - Data might be categorical (High/Medium/Low) or Numeric
                def map_npk(series):
                    if series.empty: return 50.0
                    # Try numeric first
                    num_avg = pd.to_numeric(series, errors='coerce').mean()
                    if pd.notna(num_avg): return num_avg
                    # Fallback to categorical mapping
                    cat_map = {"High": 150.0, "Medium": 80.0, "Low": 30.0, "Very High": 250.0}
                    mode_val = series.mode()
                    if not mode_val.empty:
                        return cat_map.get(mode_val[0], 50.0)
                    return 50.0

                n_val = map_npk(source_df['Nitrogen'])
                p_val = map_npk(source_df['Phosphorus'])
                k_val = map_npk(source_df['Potassium'])

        # 5. Overwrite Temp/Rainfall with Live Weather (Higher Priority)
        # Ensure we don't end up with 0.0 rainfall unless it's truly intended
        if weather_data:
            t_val = weather_data.get('temperature', t_val)
            live_rain = weather_data.get('rainfall', 0)
            if live_rain > 1.0: # Only use live rain if significant, otherwise prefer seasonal avg
                r_val = live_rain
            elif pd.isna(r_val) or r_val < 1.0:
                r_val = 120.0 # Emergency fallback to prevents AI hallucination of 0.0mm

        # 5. Final assignment with NaN safety
        rainfall = float(r_val) if pd.notna(r_val) else 120.0
        temperature = float(t_val) if pd.notna(t_val) else 30.0
        nitrogen = float(n_val) if pd.notna(n_val) else 50.0
        phosphorus = float(p_val) if pd.notna(p_val) else 50.0
        potassium = float(k_val) if pd.notna(k_val) else 50.0
        ph = float(ph_val) if pd.notna(ph_val) else 6.5
        
        # Store in session state for cross-tab access (e.g., Location Analysis)
        st.session_state.current_params = {
            "nitrogen": nitrogen,
            "phosphorus": phosphorus,
            "potassium": potassium,
            "ph": ph,
            "rainfall": rainfall,
            "temperature": temperature
        }

    # --- ALTERNATIVE UI: SMART CONTEXT DASHBOARD ---
    st.markdown("---")
    st.subheader("📡 Smart Farm Context (Detected Automatically)")
    
    # Modern Metric Dashboard
    m_col1, m_col2, m_col3, m_col4, m_col5, m_col6 = st.columns(6)
    m_col1.metric("🌧️ Rainfall", f"{rainfall:.0f}mm")
    m_col2.metric("🌡️ Temp", f"{temperature:.1f}°C")
    m_col3.metric("🧪 Nitrogen", f"{nitrogen:.0f}")
    m_col4.metric("🧪 Phosph.", f"{phosphorus:.0f}")
    m_col5.metric("🧪 Potass.", f"{potassium:.0f}")
    m_col6.metric("🧬 pH", f"{ph:.1f}")

    # Location Awareness Banner
    st.info(f"📍 **Analyzing Data for:** {district}, {state} (Detected from your selection)")
    
    # --- NEW: SATELLITE INDICES DASHBOARD ---
    st.markdown("🛰️ **Satellite Vegetation & Soil Indices (Real-Time)**")
    s_col1, s_col2, s_col3, s_col4, s_col5 = st.columns(5)
    
    live_ndvi = agri_metrics.get("ndvi")
    live_sm = agri_metrics.get("soil_moisture")
    live_st = agri_metrics.get("soil_temp")
    live_et0 = agri_metrics.get("et0")
    live_precip7 = agri_metrics.get("precip_7day")
    
    with s_col1:
        if live_ndvi is not None:
            ndvi_status = "🟢 Good" if live_ndvi > 0.5 else "🟡 Fair" if live_ndvi > 0.3 else "🔴 Poor"
            st.metric("🌿 NDVI", f"{live_ndvi:.3f}", delta=ndvi_status,
                     help="Vegetation Health Index (0-1). Higher = healthier crops.")
        else:
            st.caption("🌿 NDVI: N/A")
            
    with s_col2:
        if live_sm is not None:
            st.metric("💧 Soil Moisture", f"{live_sm:.1f}%",
                     help="Volumetric soil water content (0-7cm depth)")
        else:
            st.caption("💧 Moisture: N/A")
            
    with s_col3:
        if live_st is not None:
            st.metric("🌡️ Soil Temp", f"{live_st:.1f}°C")
        else:
            st.caption("🌡️ Soil Temp: N/A")
    
    with s_col4:
        if live_et0 is not None:
            st.metric("💨 ET₀", f"{live_et0:.1f} mm/d",
                     help="Reference Evapotranspiration (FAO Penman-Monteith)")
        else:
            st.caption("💨 ET₀: N/A")
    
    with s_col5:
        if live_precip7 is not None:
            st.metric("🌧️ 7-Day Rain", f"{live_precip7:.1f}mm",
                     help="Forecasted total precipitation for next 7 days")
        else:
            st.caption("🌧️ 7-Day: N/A")

    # Source + Status
    data_source = agri_metrics.get("source", "Unknown")
    status_cols = st.columns(3)
    with status_cols[0]:
        if weather_data:
            st.success(f"✅ Live weather active for {district}")
        else:
            st.warning("⚠️ Using historical weather data")
    with status_cols[1]:
        st.info(f"🛰️ Source: {data_source}")
    with status_cols[2]:
        st.info("📊 Soil nutrients from district health archives")

    # Update session state for visualizations
    st.session_state.soil_params = {'N': nitrogen, 'P': phosphorus, 'K': potassium, 'pH': ph}

    # Optional Overrides for advanced users (Hidden by default)
    with st.expander("⚙️ Manual Override (If you have a soil test report)"):
        o_col1, o_col2, o_col3 = st.columns(3)
        with o_col1:
            rainfall = st.number_input("Custom Rainfall (mm)", 0.0, 5000.0, rainfall)
            temperature = st.number_input("Custom Temp (°C)", 0.0, 60.0, temperature)
        with o_col2:
            nitrogen = st.number_input("Custom N", 0.0, 1000.0, nitrogen)
            phosphorus = st.number_input("Custom P", 0.0, 500.0, phosphorus)
        with o_col3:
            potassium = st.number_input("Custom K", 0.0, 1000.0, potassium)
            ph = st.number_input("Custom pH", 0.0, 14.0, ph)
        
        # Re-update session state if overridden
        st.session_state.soil_params = {'N': nitrogen, 'P': phosphorus, 'K': potassium, 'pH': ph}
    
    # Suggested Questions for Farmers
    with st.expander("💡 Not sure what to ask? Try these:", expanded=False):
        st.markdown("""
        - **Soil Health**: "My land has become infertile and crop yield is low. How can I restore it?"
        - **Crop Selection**: "With the current rainfall and temperature, is there a more profitable crop than Cotton?"
        - **Pest Warning**: "What are the common pests for this crop in this district during this season?"
        - **Climate Advice**: "How can I protect my crops from the increasing summer heat?"
        """)

    # Single Action Button
    user_question = st.text_input("💬 Ask Question", placeholder="What crops should I grow?")
    
    if st.button("🚀 Get Complete Analysis (ML + AI)", use_container_width=True) and user_question:
        # AI Configuration
        max_tokens = 800  # Reduced from 1500 for faster responses
        use_local_model = False  # Set to True if using LoRA model
        
        # Get relevant data from CSV including FULL expert advisory
        relevant_data = ""
        expert_advisory_full = ""
        state_name = state if 'state' in locals() else ""
        csv_lat = None
        csv_lon = None
        
        if not advisory_df.empty and isinstance(district, str):
            # Filter by exact state and district match - GET ALL YEARS
            filtered_data = advisory_df[
                (advisory_df['State'].str.strip() == state_name.strip()) & 
                (advisory_df['District'].str.strip() == district.strip())
            ]
            
            # Get matching crop data - ALL YEARS
            if not filtered_data.empty and 'Recommended_Crop' in filtered_data.columns:
                crop_matches = filtered_data[filtered_data['Recommended_Crop'].str.contains(crop, case=False, na=False)]
                
                if not crop_matches.empty:
                    # Get coordinates from most recent entry
                    sample_row = crop_matches.iloc[0]
                    csv_lat = sample_row.get('Lat', None)
                    csv_lon = sample_row.get('Lon', None)
                    
                    # Aggregate ALL expert advisories across all years
                    if 'Expert_Advisory' in crop_matches.columns:
                        all_advisories = crop_matches['Expert_Advisory'].dropna().tolist()
                    else:
                        all_advisories = []
                    if all_advisories:
                        expert_advisory_full = "\n\n--- MULTI-YEAR EXPERT ADVISORY (2015-2024) ---\n"
                        for idx, advisory in enumerate(all_advisories[:10], 1):  # Limit to 10 most relevant
                            if advisory and len(advisory) > 20:  # Skip empty or very short entries
                                expert_advisory_full += f"\nYear {idx}: {advisory}\n"
                    
                    # Build comprehensive multi-year historical data summary
                    relevant_data = f"\n=== COMPREHENSIVE 10-YEAR DATA for {district}, {state_name} ({crop}) ===\n\n"
                    
                    # Climate trends over years
                    relevant_data += "📊 CLIMATE TRENDS (2015-2024):\n"
                    if 'Rainfall_IMD_mm' in crop_matches.columns:
                        rainfall_avg = crop_matches['Rainfall_IMD_mm'].mean()
                        rainfall_min = crop_matches['Rainfall_IMD_mm'].min()
                        rainfall_max = crop_matches['Rainfall_IMD_mm'].max()
                        relevant_data += f"- Rainfall: Avg {rainfall_avg:.1f}mm (Range: {rainfall_min:.1f}-{rainfall_max:.1f}mm)\n"
                    
                    if 'Mean_Temp_Historical' in crop_matches.columns:
                        temp_avg = crop_matches['Mean_Temp_Historical'].mean()
                        temp_min = crop_matches['Mean_Temp_Historical'].min()
                        temp_max = crop_matches['Mean_Temp_Historical'].max()
                        relevant_data += f"- Temperature: Avg {temp_avg:.1f}°C (Range: {temp_min:.1f}-{temp_max:.1f}°C)\n"
                    
                    if 'Soil_Moisture_Historical' in crop_matches.columns:
                        moisture_avg = crop_matches['Soil_Moisture_Historical'].mean()
                        relevant_data += f"- Soil Moisture: Avg {moisture_avg:.2f}\n"
                    
                    if 'NDVI_Vegetation_Index' in crop_matches.columns:
                        ndvi_avg = crop_matches['NDVI_Vegetation_Index'].mean()
                        relevant_data += f"- Historical NDVI: {ndvi_avg:.3f}\n"
                    
                    # Add Live Satellite Context if available
                    if 'agri_metrics' in st.session_state:
                        m = st.session_state.agri_metrics
                        relevant_data += "\n🛰️ LIVE SATELLITE READINGS:\n"
                        if m.get('ndvi'): relevant_data += f"- Current Live NDVI: {m['ndvi']:.3f}\n"
                        if m.get('soil_moisture'): relevant_data += f"- Current Soil Moisture: {m['soil_moisture']:.1f}%\n"
                        if m.get('soil_temp'): relevant_data += f"- Surface Soil Temp: {m['soil_temp']:.1f}°C\n"

                    
                    # Soil health profile
                    relevant_data += "\n🧪 SOIL HEALTH PROFILE:\n"
                    if 'Soil_Type' in crop_matches.columns:
                        soil_types = crop_matches['Soil_Type'].mode()
                        if len(soil_types) > 0:
                            relevant_data += f"- Soil Type: {soil_types[0]}\n"
                    
                    if 'Soil_pH' in crop_matches.columns:
                        try:
                            ph_avg = pd.to_numeric(crop_matches['Soil_pH'], errors='coerce').mean()
                            if not pd.isna(ph_avg):
                                relevant_data += f"- Average pH: {ph_avg:.2f}\n"
                        except:
                            pass
                    
                    # Handle NPK - may be text like "Medium" or numeric
                    if 'Nitrogen' in crop_matches.columns and 'Phosphorus' in crop_matches.columns and 'Potassium' in crop_matches.columns:
                        try:
                            # Try to convert to numeric, coerce errors to NaN
                            n_series = pd.to_numeric(crop_matches['Nitrogen'], errors='coerce')
                            p_series = pd.to_numeric(crop_matches['Phosphorus'], errors='coerce')
                            k_series = pd.to_numeric(crop_matches['Potassium'], errors='coerce')
                            
                            # If we have numeric values, show averages
                            if n_series.notna().any() and p_series.notna().any() and k_series.notna().any():
                                n_avg = n_series.mean()
                                p_avg = p_series.mean()
                                k_avg = k_series.mean()
                                relevant_data += f"- NPK Levels: N={n_avg:.1f}, P={p_avg:.1f}, K={k_avg:.1f}\n"
                            else:
                                # If text values, show mode (most common)
                                n_mode = crop_matches['Nitrogen'].mode()
                                p_mode = crop_matches['Phosphorus'].mode()  
                                k_mode = crop_matches['Potassium'].mode()
                                if len(n_mode) > 0 and len(p_mode) > 0 and len(k_mode) > 0:
                                    relevant_data += f"- NPK Levels: N={n_mode[0]}, P={p_mode[0]}, K={k_mode[0]}\n"
                        except Exception as e:
                            pass  # Skip if unable to process
                    
                    # Agricultural patterns
                    relevant_data += "\n🌾 AGRICULTURAL PATTERNS:\n"
                    if 'NDVI_Vegetation_Index' in crop_matches.columns:
                        ndvi_avg = crop_matches['NDVI_Vegetation_Index'].mean()
                        relevant_data += f"- Average NDVI: {ndvi_avg:.3f} (vegetation health indicator)\n"
                    
                    if 'Historical_MSP_INR' in crop_matches.columns:
                        msp_avg = crop_matches['Historical_MSP_INR'].mean()
                        msp_trend = crop_matches['Historical_MSP_INR'].iloc[-1] - crop_matches['Historical_MSP_INR'].iloc[0] if len(crop_matches) > 1 else 0
                        relevant_data += f"- Average MSP: ₹{msp_avg:.2f}\n"
                        relevant_data += f"- MSP Trend: {'📈 Increasing' if msp_trend > 0 else '📉 Decreasing'} (₹{abs(msp_trend):.2f})\n"
                    
                    relevant_data += f"\n📍 Location: {csv_lat}, {csv_lon}\n"
                    relevant_data += f"📅 Data Points: {len(crop_matches)} years of records\n"
                    relevant_data += f"\n=== MULTI-YEAR EXPERT ADVISORY ===\n{expert_advisory_full}\n"
                else:
                    st.warning(f"⚠️ No expert data found for {crop} in {district}, {state_name}")
            else:
                st.warning(f"⚠️ No data found for {district}, {state_name}")
        
        # =========================================================
        # HEURISTIC-BASED SCORING SYSTEM
        # Dynamic, responsive predictions based on agricultural science
        # =========================================================
        
        def calculate_heuristic_score(nitrogen, phosphorus, potassium, ph, rainfall, temperature, crop):
            """
            Calculate crop suitability score based on agricultural heuristics
            Returns a score from 0-100
            """
            score = 100.0  # Start with perfect score
            penalties = []
            
            # 1. NPK Balance Score (30 points)
            npk_score = 30.0
            
            # Nitrogen scoring
            if crop.lower() in ['rice', 'wheat', 'maize', 'sugarcane']:
                # High N requirement crops
                if nitrogen < 60:
                    npk_score -= 10
                    penalties.append("Low nitrogen for this crop (-10)")
                elif nitrogen > 200:
                    npk_score -= 5
                    penalties.append("Excess nitrogen may cause lodging (-5)")
            else:
                # Medium N requirement crops
                if nitrogen < 40:
                    npk_score -= 8
                    penalties.append("Low nitrogen (-8)")
                elif nitrogen > 150:
                    npk_score -= 3
                    penalties.append("Excess nitrogen (-3)")
            
            # Phosphorus scoring
            if phosphorus < 20:
                npk_score -= 8
                penalties.append("Low phosphorus (-8)")
            elif phosphorus > 80:
                npk_score -= 2
                penalties.append("Excess phosphorus (-2)")
            
            # Potassium scoring
            if potassium < 40:
                npk_score -= 7
                penalties.append("Low potassium (-7)")
            elif potassium > 180:
                npk_score -= 2
                penalties.append("Excess potassium (-2)")
            
            # 2. pH Score (20 points)
            ph_score = 20.0
            ideal_ph = 6.5
            
            if crop.lower() in ['rice']:
                ideal_ph = 6.0
            elif crop.lower() in ['potato', 'tea']:
                ideal_ph = 5.5
            
            ph_deviation = abs(ph - ideal_ph)
            if ph_deviation > 2.0:
                ph_score -= 15
                penalties.append(f"pH far from ideal {ideal_ph} (-15)")
            elif ph_deviation > 1.0:
                ph_score -= 8
                penalties.append(f"pH deviation from ideal (-8)")
            elif ph_deviation > 0.5:
                ph_score -= 3
                penalties.append(f"Slight pH deviation (-3)")
            
            # 3. Rainfall Score (25 points)
            rainfall_score = 25.0
            
            if crop.lower() in ['rice', 'sugarcane']:
                # High water requirement
                if rainfall < 600:
                    rainfall_score -= 15
                    penalties.append("Insufficient rainfall for this crop (-15)")
                elif rainfall > 2000:
                    rainfall_score -= 5
                    penalties.append("Excess rainfall may cause waterlogging (-5)")
            elif crop.lower() in ['wheat', 'maize']:
                # Medium water requirement
                if rainfall < 400:
                    rainfall_score -= 12
                    penalties.append("Low rainfall (-12)")
                elif rainfall > 1500:
                    rainfall_score -= 4
                    penalties.append("High rainfall (-4)")
            else:
                # General crops
                if rainfall < 300:
                    rainfall_score -= 10
                    penalties.append("Very low rainfall (-10)")
                elif rainfall > 1800:
                    rainfall_score -= 5
                    penalties.append("Very high rainfall (-5)")
            
            # 4. Temperature Score (25 points)
            temp_score = 25.0
            
            if crop.lower() in ['rice', 'cotton', 'sugarcane']:
                # Warm season crops
                if temperature < 20:
                    temp_score -= 15
                    penalties.append("Too cold for this crop (-15)")
                elif temperature > 38:
                    temp_score -= 10
                    penalties.append("Too hot, heat stress (-10)")
                elif temperature < 25:
                    temp_score -= 5
                    penalties.append("Suboptimal temperature (-5)")
            elif crop.lower() in ['wheat', 'potato']:
                # Cool season crops
                if temperature > 30:
                    temp_score -= 15
                    penalties.append("Too hot for this crop (-15)")
                elif temperature < 15:
                    temp_score -= 10
                    penalties.append("Too cold (-10)")
            else:
                # General crops
                if temperature < 15 or temperature > 35:
                    temp_score -= 12
                    penalties.append("Extreme temperature (-12)")
                elif temperature < 18 or temperature > 32:
                    temp_score -= 5
                    penalties.append("Suboptimal temperature (-5)")
            
            # Calculate total score
            total_score = npk_score + ph_score + rainfall_score + temp_score
            
            return max(0, min(100, total_score)), penalties
        
        def calculate_risk_score(nitrogen, phosphorus, potassium, ph, rainfall, temperature, agri_metrics=None):
            """
            Calculate a granular climate and environmental risk score (0-100)
            higher = more dangerous for farming
            """
            risk = 5.0  # Base natural risk
            
            # 🧪 Nutrient Imbalance Risk (Granular bands)
            if nitrogen < 30 or nitrogen > 300: risk += 15
            elif nitrogen < 60 or nitrogen > 200: risk += 8
            
            if phosphorus < 15 or phosphorus > 120: risk += 12
            elif phosphorus < 30 or phosphorus > 80: risk += 6
            
            if potassium < 40 or potassium > 250: risk += 12
            elif potassium < 70 or potassium > 180: risk += 6
            
            # 🧬 pH Risk (High sensitivity for Indian soils)
            if ph < 4.5 or ph > 9.0: risk += 30
            elif ph < 5.5 or ph > 8.0: risk += 15
            elif ph < 6.0 or ph > 7.5: risk += 7
            
            # 🌧️ Rainfall Risk (Drought & Flood bands)
            if rainfall < 300: risk += 35 # High drought risk
            elif rainfall < 600: risk += 18 # Moderate water scarcity
            elif rainfall > 2200: risk += 25 # High flood/leaching risk
            elif rainfall > 1600: risk += 12 # Moderate drainage risk
            
            # 🌡️ Temperature Stress Risk
            if temperature < 10 or temperature > 45: risk += 30 # Extreme thermal stress
            elif temperature < 18 or temperature > 38: risk += 15 # Suboptimal heat/cold
            elif temperature < 22 or temperature > 34: risk += 5
            
            # 🛰️ SATELLITE RISK OVERLAY (Live Ground Evidence)
            if agri_metrics:
                live_ndvi = agri_metrics.get("ndvi")
                live_sm = agri_metrics.get("soil_moisture")
                
                # Low NDVI indicates existing vegetation stress or poor land productivity
                if live_ndvi is not None:
                    if live_ndvi < 0.2: risk += 25
                    elif live_ndvi < 0.35: risk += 12
                    elif live_ndvi > 0.8: risk -= 5 # Bonus for extremely healthy green cover
                
                # Low Soil Moisture indicates immediate water stress regardless of historical rain
                if live_sm is not None:
                    if live_sm < 8: risk += 20
                    elif live_sm < 15: risk += 10
                    elif live_sm > 45: risk += 15 # Waterlogging risk
            
            return min(100.0, max(0.0, risk))
        
        # Calculate scores using heuristics
        crop_pred, penalties = calculate_heuristic_score(
            nitrogen, phosphorus, potassium, ph, rainfall, temperature, crop
        )
        
        # Get agri_metrics from session state for risk calculation
        current_agri_metrics = st.session_state.get('agri_metrics', {})
        risk_pred = calculate_risk_score(
            nitrogen, phosphorus, potassium, ph, rainfall, temperature, current_agri_metrics
        )
        
        # Display results
        suitability_label = "Excellent" if crop_pred > 80 else "Good" if crop_pred > 60 else "Moderate" if crop_pred > 40 else "Poor"
        risk_label = "Low" if risk_pred < 20 else "Moderate" if risk_pred < 40 else "High"
        
        st.success(f"🌾 **Crop Suitability: {crop_pred:.1f}%** ({suitability_label}) | **Climate Risk: {risk_pred:.1f}%** ({risk_label})")
        
        # ── Domain-specific Score Breakdown ──────────────────────────────
        with st.expander("🔍 Domain Score Breakdown", expanded=False):
            domain_scores = st.session_state.get("domain_score_breakdown", {})
            detected_domain_label = st.session_state.get(
                "detected_domain", "general"
            ).replace("_", " ").title()

            if domain_scores:
                st.markdown(f"**Domain: {detected_domain_label}**")

                # Show parameter table
                col_s1, col_s2 = st.columns(2)
                items = list(domain_scores.items())
                mid   = len(items) // 2

                for col, chunk in zip([col_s1, col_s2], [items[:mid], items[mid:]]):
                    with col:
                        for label, info in chunk:
                            score   = info["score"]
                            colour  = info["colour"]
                            status  = info["status"]
                            bar_col = "#28a745" if colour == "green" else "#ffc107" if colour == "orange" else "#dc3545"
                            st.markdown(
                                f"""<div style="margin-bottom:8px">
                                    <span style="font-size:0.85em">{label}</span><br>
                                    <div style="background:#eee;border-radius:4px;height:14px;width:100%">
                                      <div style="background:{bar_col};border-radius:4px;height:14px;width:{score}%"></div>
                                    </div>
                                    <span style="font-size:0.78em;color:{bar_col}">{score:.0f}/100 — {status}</span>
                                </div>""",
                                unsafe_allow_html=True,
                            )

                # Overall score
                all_scores = [v["score"] for v in domain_scores.values()]
                overall    = round(sum(all_scores) / len(all_scores), 1)
                ov_colour  = "#28a745" if overall >= 70 else "#ffc107" if overall >= 45 else "#dc3545"
                st.markdown(
                    f"""<div style="text-align:center;margin-top:12px;padding:10px;
                         border:2px solid {ov_colour};border-radius:8px">
                         <strong style="font-size:1.2em;color:{ov_colour}">
                           Overall {detected_domain_label} Score: {overall}/100
                         </strong>
                    </div>""",
                    unsafe_allow_html=True,
                )

                # Detected parameters table
                st.markdown("---")
                st.markdown("**📥 Parameters Used for This Analysis**")
                st.markdown(f"""
| Parameter | Value | Source |
|---|---|---|
| 🌱 N | `{nitrogen:.1f} kg/ha` | Historical/Default |
| 🌱 P | `{phosphorus:.1f} kg/ha` | Historical/Default |
| 🌱 K | `{potassium:.1f} kg/ha` | Historical/Default |
| 🧪 pH | `{ph:.2f}` | Historical/Default |
| 🌧️ Rainfall | `{rainfall:.0f} mm` | Live + Historical |
| 🌡️ Temperature | `{temperature:.1f} °C` | Live Weather |
| 🛰️ NDVI | `{st.session_state.get("agri_metrics", {}).get("ndvi", "N/A")}` | Satellite |
| 💧 Soil Moisture | `{st.session_state.get("agri_metrics", {}).get("soil_moisture", "N/A")}%` | Satellite |
| 💦 ET₀ | `{st.session_state.get("agri_metrics", {}).get("et0", "N/A")} mm/day` | Satellite |
| ⛈️ 7-Day Forecast | `{st.session_state.get("agri_metrics", {}).get("precip_7day", "N/A")} mm` | Forecast API |
""")
            else:
                st.info("Score breakdown will appear here after your first question.")


        
        
        
        # Get live weather data for augmentation
        live_weather_context = ""
        if csv_lat and csv_lon:
            # Use coordinates from CSV (more reliable for districts)
            current_weather = get_weather_data_by_coords(csv_lat, csv_lon, f"{district}, {state_name}")
            if current_weather:
                live_weather_context = f"\n=== LIVE WEATHER DATA ===\n"
                live_weather_context += f"Location: {current_weather.get('location', location)}\n"
                live_weather_context += f"Coordinates: {csv_lat}, {csv_lon}\n"
                live_weather_context += f"Current Temperature: {current_weather.get('temperature', 'N/A')}°C\n"
                live_weather_context += f"Humidity: {current_weather.get('humidity', 'N/A')}%\n"
                live_weather_context += f"Wind Speed: {current_weather.get('wind_speed', 'N/A')} km/h\n"
                live_weather_context += f"Rainfall: {current_weather.get('rainfall', 'N/A')}mm\n"
                live_weather_context += f"Pressure: {current_weather.get('pressure', 'N/A')} hPa\n"
                live_weather_context += f"Last Updated: {current_weather.get('last_updated', 'N/A')}\n"
                st.success(f"✅ Live weather fetched for {district}, {state_name} ({csv_lat}, {csv_lon})")
            else:
                st.warning(f"⚠️ Could not fetch weather data using CSV coordinates")
        elif location:
            # Fallback to geocoding if CSV coordinates not available
            current_weather = get_weather_data(location)
            if current_weather:
                live_weather_context = f"\n=== LIVE WEATHER DATA ===\n"
                live_weather_context += f"Location: {current_weather.get('location', location)}\n"
                live_weather_context += f"Current Temperature: {current_weather.get('temperature', 'N/A')}°C\n"
                live_weather_context += f"Humidity: {current_weather.get('humidity', 'N/A')}%\n"
                live_weather_context += f"Wind Speed: {current_weather.get('wind_speed', 'N/A')} km/h\n"
                live_weather_context += f"Rainfall: {current_weather.get('rainfall', 'N/A')}mm\n"
                live_weather_context += f"Pressure: {current_weather.get('pressure', 'N/A')} hPa\n"
                live_weather_context += f"Last Updated: {current_weather.get('last_updated', 'N/A')}\n"
            else:
                st.warning(f"⚠️ Could not fetch weather data for {location}")
        
        # AI Response with FULL expert advisory, live weather, and dataset context
        # Analyze query type for enhanced context
        query_analysis = analyze_query_type(user_question)
        
        # Build satellite context string
        sat_context = ""
        if 'agri_metrics' in st.session_state:
            am = st.session_state.agri_metrics
            if am.get('ndvi'): sat_context += f"Live NDVI (Vegetation Index): {am['ndvi']:.3f} ({am.get('source', 'Unknown source')})\n"
            if am.get('soil_moisture'): sat_context += f"Live Soil Moisture: {am['soil_moisture']:.1f}%\n"
            if am.get('soil_temp'): sat_context += f"Live Soil Temperature: {am['soil_temp']:.1f}°C\n"
            if am.get('et0'): sat_context += f"Live Evapotranspiration (ET0): {am['et0']:.1f} mm/day\n"
            if am.get('precip_7day'): sat_context += f"7-Day Forecasted Rainfall (Satellite/ERA5): {am['precip_7day']:.1f} mm\n"
            if am.get('radiation'): sat_context += f"Solar Radiation: {am['radiation']:.1f} MJ/m²\n"
        
        # Build enhanced context based on query type
        if query_analysis["requires_calculation"]:
            enhanced_context = build_enhanced_context(
                user_question, 
                relevant_data, 
                query_analysis, 
                crop_matches if 'crop_matches' in locals() else None,
                {
                    'nitrogen': nitrogen,
                    'phosphorus': phosphorus,
                    'potassium': potassium,
                    'ph': ph,
                    'temperature': temperature,
                    'et0': st.session_state.get('agri_metrics', {}).get('et0'),
                    'soil_moisture': st.session_state.get('agri_metrics', {}).get('soil_moisture'),
                    'precip_7day': st.session_state.get('agri_metrics', {}).get('precip_7day')
                },
                weather_data if 'weather_data' in locals() else {},
                state=state if 'state' in locals() else "",
                district=district if 'district' in locals() else ""
            )
            relevant_data = enhanced_context
            st.info(f"🔍 Detected {query_analysis['type'].replace('_', ' ').title()} query - performing specialized analysis...")

        context = f"""=== CRITICAL INSTRUCTION (MUST FOLLOW) ===
You are responding to a {query_analysis['type'].upper()} question.

YOU MUST START YOUR RESPONSE WITH THIS EXACT HEADING:
### 🔬 Diagnosis

DO NOT write:
- "Ariyalur, Tamil Nadu has..."
- "The soil in {district}..."
- Any location description

START IMMEDIATELY with "### 🔬 Diagnosis" and the NPK analysis.

=== YOUR ROLE ===
You are an EXPERT AGRICULTURAL ADVISOR with 20+ years of experience serving farmers in {district}, {state_name}.

You are a DATA-DRIVEN agricultural consultant who provides SPECIFIC, ACTIONABLE recommendations.

=== PATIENT PROFILE ===
Location: {district}, {state_name}, India
Crop: {crop}
Current Conditions:
- Rainfall: {rainfall:.0f}mm (10-year avg) | Temperature: {temperature:.1f}°C (current)
- Soil: N={nitrogen:.1f} kg/ha, P={phosphorus:.1f} kg/ha, K={potassium:.1f} kg/ha, pH={ph:.2f}
- Live NDVI: {st.session_state.get('agri_metrics', {}).get('ndvi', 'N/A')}
- Soil Moisture: {st.session_state.get('agri_metrics', {}).get('soil_moisture', 'N/A')}%
- ET0: {st.session_state.get('agri_metrics', {}).get('et0', 'N/A')} mm/day
- 7-Day Forecast Rain: {st.session_state.get('agri_metrics', {}).get('precip_7day', 'N/A')}mm

=== ML ANALYSIS ===
- Crop Suitability Score: {crop_pred:.1f}% ({suitability_label})
- Climate Risk Score: {risk_pred:.1f}% ({risk_label})

=== HISTORICAL DATA ===
{relevant_data}

=== LIVE WEATHER ===
{live_weather_context}

=== FARMER'S QUESTION ===
"{user_question}"

=== MANDATORY RESPONSE FORMAT ===
For {query_analysis['type'].upper()} questions, you MUST use this structure:

### 🔬 Diagnosis
- Nitrogen (N): {nitrogen:.1f} kg/ha
- Phosphorus (P): {phosphorus:.1f} kg/ha
- Potassium (K): {potassium:.1f} kg/ha
- pH: {ph:.2f}
[Explain the problem using these exact numbers]

### 💊 Immediate Treatment (Next 7 Days)
1. [Product name] - [kg/ha] - [When to apply]
2. [Product name] - [kg/ha] - [When to apply]
3. [Action] - [Details]

### 🌿 Organic Alternative (Low-Cost)
- [Product] ([kg/ha]) + [Product] ([kg/ha])
- [Application method]

### ⚠️ Risk Alert
- [Timeline and consequences]
- [Monitoring instructions]

**For COMPARISON questions:**
### 📊 5-Year Comparison Table
[Copy exact table from HISTORICAL DATA]

### 🎯 Verdict
[State which crop depletes MORE with exact percentages]

### 💡 Soil Recovery Strategy
[Specific fertilizer amounts needed to restore soil after each crop]
[Crop rotation recommendation with legumes]

**For IRRIGATION questions:**
### 💧 Water Deficit Calculation
- Current soil moisture: {st.session_state.get('agri_metrics', {}).get('soil_moisture', 'N/A')}%
- Wilting point: 15%
- ET0: {st.session_state.get('agri_metrics', {}).get('et0', 'N/A')} mm/day
- **Irrigation needed: [Calculate exact hours]**

### 🚰 Application Method
[Drip/sprinkler specifics with flow rate]

**For CLIMATE RISK questions:**
### ⛈️ Risk Assessment
- Historical flood frequency: [From data]
- 7-day forecast: {st.session_state.get('agri_metrics', {}).get('precip_7day', 'N/A')}mm
- **Flood Risk Score: [Calculate %]**

### 🛡️ Mitigation Strategy
[Specific actions with timeline]

**For ECONOMIC questions:**
### 💰 ROI Analysis
[Use MSP trend data to calculate expected returns]

### 📈 Market Intelligence
[5-year price trend with recommendation]

**For PEST/DISEASE questions:**
### 🐛 Pest Identification
[Specific pest name based on conditions]

### 💉 Treatment Protocol
[Chemical: Product + dosage | Organic: Alternative]

### 📅 Application Schedule
[Day 1, Day 7, Day 14 actions]

**For LONG-TERM STRATEGY questions:**
### 🌡️ Climate Trend Analysis
[10-year temperature/rainfall change rate]

### 🌳 Future-Proof Crops
[3-5 climate-resilient options with rationale]

### 💵 Investment Priority
[Rank: Seeds vs Fertilizer vs Technology]

=== CRITICAL RULES ===
1. **MANDATORY**: If HISTORICAL DATA contains a "RESPONSE FRAMEWORK" section, you MUST copy and follow that exact structure
2. **MANDATORY**: Use EXACT numbers from Current Conditions (N={nitrogen:.1f}, P={phosphorus:.1f}, K={potassium:.1f}, pH={ph:.2f})
3. NEVER say "consider" or "you may want to" - give DIRECT instructions with product names and kg/ha doses
4. For NUTRIENT_LOCKUP: Start response with "### 🔬 Diagnosis" then state exact NPK imbalance
5. For COMPARISON: Copy the comparison table from HISTORICAL DATA, don't create your own
6. For IRRIGATION_EMERGENCY: Calculate exact pump hours using formula in HISTORICAL DATA
7. For HEAT_STRESS: Recommend Kaolin clay (50g/L, ₹1,200/ha) as first option
8. For FLOOD_RISK: State probability percentage and risk level (HIGH/MODERATE/LOW)
9. Always include costs in Indian Rupees (₹) and dosages in kg/ha or liters/ha
10. End every response with a timeline or deadline ("Apply TODAY", "Start within 2 hours", etc.)

NOW RESPOND TO THE FARMER'S QUESTION USING THE FORMAT ABOVE."""
        
        # ── Build domain-aware Groq payload ──────────────────────────────
        st.info("🔬 **Domain AI Mode**: Routing to domain-specific expert prompt...")

        groq_payload = build_groq_payload(
            question        = user_question,
            district        = district,
            state           = state_name if "state_name" in locals() else state,
            crop            = crop,
            nitrogen        = nitrogen,
            phosphorus      = phosphorus,
            potassium       = potassium,
            ph              = ph,
            rainfall        = rainfall,
            temperature     = temperature,
            agri_metrics    = st.session_state.get("agri_metrics", {}),
            crop_matches_df = crop_matches if "crop_matches" in locals() and not crop_matches.empty else None,
            advisory_df     = advisory_df,
            crop_suitability_score = crop_pred,
            climate_risk_score     = risk_pred,
            expert_advisory        = expert_advisory_full if "expert_advisory_full" in locals() else "",
            live_weather_context   = live_weather_context if "live_weather_context" in locals() else "",
        )

        detected_domain = groq_payload["domain"]
        st.caption(f"🎯 Domain detected: **{detected_domain.replace('_', ' ').title()}**")

        # ── Store domain score breakdown in session state ─────────────────
        st.session_state["domain_score_breakdown"] = groq_payload["score_breakdown"]
        st.session_state["domain_visuals"]         = groq_payload["visuals"]

        # ── Call Groq with domain system + user prompt ────────────────────
        raw_response = None
        ai_backend_used = "Groq Domain Expert"

        # Try local models first as supplementary context
        ensemble_responses = {}

        if st.session_state.t5_peft_model:
            with st.spinner("🌾 T5-PEFT analysing..."):
                t5_resp = generate_with_t5_peft(
                    groq_payload["user_prompt"],
                    st.session_state.t5_peft_model,
                    st.session_state.t5_peft_tokenizer,
                    max_tokens,
                )
                if t5_resp:
                    ensemble_responses["T5-PEFT"] = t5_resp

        if st.session_state.lora_model:
            with st.spinner("🌡️ Climate-LoRA analysing..."):
                lora_resp = generate_with_lora(
                    groq_payload["user_prompt"],
                    st.session_state.lora_model,
                    st.session_state.tokenizer,
                    max_tokens,
                )
                if lora_resp:
                    ensemble_responses["Climate-LoRA"] = lora_resp

        # Build final user message (append local model insights if available)
        final_user_msg = groq_payload["user_prompt"]
        if ensemble_responses:
            final_user_msg += "\n\n═══ LOCAL MODEL INSIGHTS (use as supporting context) ═══"
            for model_name, resp in ensemble_responses.items():
                final_user_msg += f"\n\n--- {model_name} ---\n{resp[:600]}"

        # ── Groq API call ─────────────────────────────────────────────────
        with st.spinner(f"🤖 Groq [{detected_domain}] expert generating response..."):
            try:
                import groq as groq_lib
                client = groq_lib.Groq(api_key=st.session_state.groq_api_key)

                chat_completion = client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": groq_payload["system_prompt"]},
                        {"role": "user",   "content": final_user_msg},
                    ],
                    model="llama-3.3-70b-versatile",
                    temperature=0.3,
                    max_tokens=1800,
                    top_p=0.85,
                )
                raw_response = chat_completion.choices[0].message.content
                ai_backend_used = (
                    f"Groq [{detected_domain.replace('_',' ').title()}]"
                    + (f" + {', '.join(ensemble_responses.keys())}" if ensemble_responses else "")
                )

            except Exception as e:
                st.warning(f"⚠️ Groq error: {e}")
                raw_response = None

        # ── Translation / voice ───────────────────────────────────────────
        if raw_response:
            with st.spinner(f"🌐 Processing {st.session_state.target_language} summary..."):
                translation_result = translate_text(
                    raw_response,
                    st.session_state.target_language,
                    st.session_state.groq_api_key,
                )
                response    = translation_result["detailed"]
                voice_summary = translation_result["summary"]

            if st.session_state.enable_voice:
                with st.spinner("🎙️ Generating voice response..."):
                    speak_text(voice_summary, st.session_state.target_language)
        else:
            response = None
        
        # Error handling - when ALL backends fail
        if not response:
            st.error("❌ **Unable to generate AI response**")
            st.warning("""
            **All AI backends failed. Please check:**
            
            **Groq API (Your configured backend):**
            - Verify your API key is valid at: https://console.groq.com
            - Check if you've hit the free tier rate limit (wait 60 seconds and retry)
            
            **Alternative: Install Ollama (Free & Local)**
            1. Download from: https://ollama.ai
            2. Run: `ollama pull llama3.2:1b`
            3. Restart the app
            """)
        
        # Display successful response
        if response:
            st.success(f"✅ **Response generated using:** {ai_backend_used}")
        if response:
            with st.expander("📊 View Data Used for Analysis", expanded=False):
                # Show input parameters
                st.subheader("📥 Your Input Parameters")
                input_debug = f"""
**Location**: {district}, {state_name if 'state_name' in locals() else state}  
**Crop**: {crop}  
**Rainfall**: {rainfall}mm  
**Temperature**: {temperature}°C  
**Soil**: N={nitrogen}, P={phosphorus}, K={potassium}, pH={ph}  
**Question**: "{user_question}"
                """
                st.info(input_debug)
                
                # Show ML predictions
                st.subheader("🤖 ML Model Predictions")
                
                # Calculate interpretations outside f-string
                suitability_label = "Excellent" if crop_pred > 70 else "Good" if crop_pred > 50 else "Moderate" if crop_pred > 30 else "Poor"
                risk_label = "Low" if risk_pred < 20 else "Moderate" if risk_pred < 40 else "High"
                model_name = "Heuristic-Based Scoring (XGBoost-Inspired Agricultural Science)"
                
                ml_debug = f"""
**Crop Suitability Score**: {crop_pred:.2f}% ({suitability_label})  
**Climate Risk Score**: {risk_pred:.2f}% ({risk_label})  
**Model Used**: {model_name}
                """
                st.warning(ml_debug)
                
                if expert_advisory_full:
                    st.subheader("🎓 Expert Advisory (from CSV)")
                    st.success(expert_advisory_full)
                if live_weather_context:
                    st.subheader("🌤️ Live Weather Data")
                    st.info(live_weather_context)
                if relevant_data:
                    st.subheader("📈 Historical Agricultural Data")
                    st.warning(relevant_data)
            
            # Show individual model responses (if ensemble was used)
            if len(ensemble_responses) > 1:
                with st.expander("🔬 View Individual Model Responses (Before Synthesis)"):
                    for model_name, model_response in ensemble_responses.items():
                        st.markdown(f"**{model_name} Response:**")
                        st.info(model_response)
                        st.markdown("---")
            
            # Display the final synthesized response with professional formatting
            st.markdown("---")
            st.header("📋 Professional Agricultural Advisory Report")
            
            # --- NEW: PROMINENT DATA VISUALIZATIONS (Moved higher for better visibility) ---
            if 'crop_matches' in locals() and not crop_matches.empty:
                with st.expander("📊 Analyze Historical 10-Year Trends", expanded=True):
                    # Sort by year for consistent plotting
                    plot_df = crop_matches.copy()
                    if 'Year' in plot_df.columns:
                        plot_df['Year'] = pd.to_numeric(plot_df['Year'], errors='coerce')
                        plot_df = plot_df.sort_values('Year')
                    
                    # Create two columns for charts
                    plot_col1, plot_col2 = st.columns(2)
                    
                    with plot_col1:
                        # 1. Climate Trends Chart
                        fig_climate = go.Figure()
                        years = plot_df['Year'] if 'Year' in plot_df.columns else list(range(1, len(plot_df) + 1))
                        
                        if 'Rainfall_IMD_mm' in plot_df.columns:
                            fig_climate.add_trace(go.Bar(
                                x=years, y=plot_df['Rainfall_IMD_mm'],
                                name="Rainfall (mm)", marker_color='#4FC3F7', opacity=0.6
                            ))
                        
                        if 'Mean_Temp_Historical' in plot_df.columns:
                            fig_climate.add_trace(go.Scatter(
                                x=years, y=plot_df['Mean_Temp_Historical'],
                                name="Temp (°C)", mode='lines+markers',
                                line=dict(color='#FF7043', width=3), yaxis="y2"
                            ))
                        
                        fig_climate.update_layout(
                            title_text="Climate Trends (Rainfall vs Temp)",
                            xaxis_title_text="Year",
                            yaxis=dict(
                                title=dict(text="Rainfall (mm)", font=dict(color="#4FC3F7")),
                                tickfont=dict(color="#4FC3F7")
                            ),
                            yaxis2=dict(
                                title=dict(text="Temp (°C)", font=dict(color="#FF7043")),
                                tickfont=dict(color="#FF7043"),
                                overlaying="y",
                                side="right"
                            ),
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            font=dict(color="white"),
                            height=350,
                            margin=dict(l=10, r=10, t=50, b=10),
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                        )
                        st.plotly_chart(fig_climate, use_container_width=True)

                    with plot_col2:
                        # 2. Soil Nutrient Profile
                        n_vals = pd.to_numeric(plot_df['Nitrogen'], errors='coerce').dropna()
                        p_vals = pd.to_numeric(plot_df['Phosphorus'], errors='coerce').dropna()
                        k_vals = pd.to_numeric(plot_df['Potassium'], errors='coerce').dropna()
                        
                        if not n_vals.empty:
                            categories = ['Nitrogen', 'Phosphorus', 'Potassium']
                            values = [n_vals.mean(), p_vals.mean() if not p_vals.empty else 0, k_vals.mean() if not k_vals.empty else 0]
                            
                            fig_soil = go.Figure(go.Bar(
                                x=categories, y=values,
                                marker_color=['#66BB6A', '#9CCC65', '#D4E157'],
                                text=[f"{v:.1f}" for v in values], textposition='auto',
                            ))
                            fig_soil.update_layout(
                                title="Average Soil Nutrient Profile",
                                yaxis_title="Quantity (kg/ha)",
                                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                font=dict(color="white"), height=350,
                                margin=dict(l=10, r=10, t=50, b=10)
                            )
                            st.plotly_chart(fig_soil, use_container_width=True)
                        else:
                            st.info("ℹ️ Nutrient data is available in text format (Medium/High) in the detailed cards below.")

                    st.markdown("---")
                    st.markdown("#### 📊 Domain-Specific Analytics")
                    render_domain_visuals(
                        domain_visuals = st.session_state.get("domain_visuals", []),
                        crop_matches_df= crop_matches if "crop_matches" in locals() and not crop_matches.empty else None,
                        agri_metrics   = st.session_state.get("agri_metrics", {}),
                        weather_data   = weather_data,
                        soil_params    = {"N": nitrogen, "P": phosphorus, "K": potassium, "pH": ph},
                        district       = district,
                        state          = state_name if "state_name" in locals() else state,
                        crop           = crop,
                    )

            # Helper to create professional cards matching the app theme
            def insight_card(title, content):
                import re
                processed_content = content.replace('\n', '<br>')
                # Convert markdown bold **text** to HTML bold <b>text</b>
                processed_content = re.sub(r'\*\*(.*?)\*\*', r'<b style="color: #ffffff;">\1</b>', processed_content)
                
                card_html = """
                <div style="background-color: rgba(255, 255, 255, 0.05); padding: 20px; border-radius: 10px; border: 1px solid rgba(255, 255, 255, 0.1); margin-bottom: 20px; backdrop-filter: blur(5px);">
                    <div style="color: #4CAF50; font-weight: bold; font-size: 1.1em; margin-bottom: 10px; border-bottom: 1px solid rgba(255, 255, 255, 0.1); padding-bottom: 5px; display: flex; align-items: center; gap: 10px;">
                        {t}
                    </div>
                    <div style="color: #e0e0e0; line-height: 1.6; font-family: 'Segoe UI', sans-serif; font-size: 1rem;">
                        {c}
                    </div>
                </div>
                """.format(t=title, c=processed_content)
                st.markdown(card_html, unsafe_allow_html=True)

            import re
            # Split the response into major sections based on ### header
            sections = re.split(r'### ', response)
            
            # If the response was properly split
            if len(sections) > 1:
                # First section is usually intro text, loop through others
                for section in sections[1:]:
                    lines = section.strip().split('\n', 1)
                    if len(lines) == 2:
                        header = lines[0].strip()
                        body = lines[1].strip()
                        insight_card(header, body)
                    else:
                        st.markdown(section)
            else:
                # Fallback to the previous styled single card if split fails
                html_response = response.replace('\n', '<br>')
                st.markdown(f"""
                <div style="background-color: white; padding: 25px; border-radius: 12px; border: 1px solid #e1e8ed; border-left: 5px solid #2e7d32;">
                    <div style="line-height: 1.7; color: #333;">{html_response}</div>
                    <div style="margin-top: 15px; font-size: 0.8em; color: #888;">✨ {ai_backend_used}</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Bottom signature
            st.caption(f"🛡️ This report is anchor-factual based on 10-year research data for {district}. Produced via {ai_backend_used}.")            
            # Save to chat history
            st.session_state.chat_history.append({'role': 'user', 'content': user_question})
            st.session_state.chat_history.append({'role': 'assistant', 'content': response})
            
            # Save to recommendation history for Export tab
            from datetime import datetime
            recommendation_entry = {
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'location': f"{district}, {state_name if 'state_name' in locals() else state}",
                'crop': crop,
                'question': user_question,
                'recommendation': response,
                'ai_backend': ai_backend_used,
                'parameters': {
                    'rainfall': float(rainfall),
                    'temperature': float(temperature),
                    'soil': {
                        'N': float(nitrogen), 
                        'P': float(phosphorus), 
                        'K': float(potassium), 
                        'pH': float(ph)
                    }
                },
                'ml_prediction': {
                    'crop_suitability': float(crop_pred),
                    'climate_risk': float(risk_pred)
                }
            }
            st.session_state.recommendation_history.append(recommendation_entry)

with tab2:
    st.header("🌍 Location-Based Analysis")
    
    # Allow district selection directly in this tab
    st.subheader("📍 Select Location for Analysis")
    col_select1, col_select2 = st.columns(2)
    
    with col_select1:
        if state_district_mapping:
            available_states = sorted(state_district_mapping.keys())
            selected_state = st.selectbox("🏞️ Select State", available_states, key="tab2_state", 
                                         index=available_states.index('Tamil Nadu') if 'Tamil Nadu' in available_states else 0)
        else:
            selected_state = st.text_input("State", "Tamil Nadu", key="tab2_state")
    
    with col_select2:
        if state_district_mapping and selected_state in state_district_mapping:
            available_districts = state_district_mapping[selected_state]
            selected_district = st.selectbox("📍 Select District", available_districts, key="tab2_district")
        else:
            selected_district = st.text_input("District", "Chennai", key="tab2_district")
    
    st.markdown("---")
    
    # Fetch district-specific data from CSV
    if not advisory_df.empty and selected_state and selected_district:
        district_data = advisory_df[
            (advisory_df['State'] == selected_state) & 
            (advisory_df['District'] == selected_district)
        ]
        
        if not district_data.empty:
            st.success(f"✅ Found {len(district_data)} records for {selected_district}, {selected_state}")
            
            # Get first record for location details
            district_info = district_data.iloc[0]
            
            # Location Overview
            st.subheader("📍 District Overview")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("📍 Coordinates", f"{district_info.get('Lat', 'N/A'):.2f}, {district_info.get('Lon', 'N/A'):.2f}")
            with col2:
                st.metric("🌍 Region", district_info.get('Region', 'N/A'))
            with col3:
                st.metric("🌾 Crops Available", len(district_data['Recommended_Crop'].unique()))
            with col4:
                st.metric("📊 Total Records", len(district_data))
            
            # Climate Analysis
            st.markdown("---")
            st.subheader("�️ Climate Profile")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                avg_rainfall = district_data['Rainfall_IMD_mm'].mean()
                min_rainfall = district_data['Rainfall_IMD_mm'].min()
                max_rainfall = district_data['Rainfall_IMD_mm'].max()
                st.metric("Avg Rainfall", f"{avg_rainfall:.1f}mm", f"Range: {min_rainfall:.0f}-{max_rainfall:.0f}mm")
            
            with col2:
                avg_temp = district_data['Mean_Temp_Historical'].mean()
                min_temp = district_data['Mean_Temp_Historical'].min()
                max_temp = district_data['Mean_Temp_Historical'].max()
                st.metric("Avg Temperature", f"{avg_temp:.1f}°C", f"Range: {min_temp:.1f}-{max_temp:.1f}°C")
            
            with col3:
                avg_moisture = district_data['Soil_Moisture_Historical'].mean()
                st.metric("Avg Soil Moisture", f"{avg_moisture:.3f}")
            
            with col4:
                avg_ndvi = district_data['NDVI_Vegetation_Index'].mean()
                st.metric("Avg NDVI", f"{avg_ndvi:.3f}")
            
            # Soil Analysis
            st.markdown("---")
            st.subheader("🧪 Soil Profile")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                avg_ph = district_data['Soil_pH'].mean()
                st.metric("Avg Soil pH", f"{avg_ph:.2f}")
            
            with col2:
                # Count soil types
                soil_types = district_data['Soil_Type'].value_counts()
                most_common_soil = soil_types.index[0] if len(soil_types) > 0 else "N/A"
                st.markdown("**Primary Soil Type:**")
                st.info(most_common_soil)
            
            with col3:
                # Show Predominant Nitrogen Status
                n_status = district_data['Nitrogen'].value_counts()
                most_common_n = n_status.index[0] if not n_status.empty else "N/A"
                n_high_count = district_data['Nitrogen'].astype(str).str.strip().str.lower().eq('high').sum()
                
                st.metric("Primary Nitrogen", most_common_n, delta=f"{n_high_count} High" if n_high_count > 0 else None)
                st.caption(f"Count: {n_status.get(most_common_n, 0)} records")
            
            with col4:
                # Show Predominant Phosphorus Status
                p_status = district_data['Phosphorus'].value_counts()
                most_common_p = p_status.index[0] if not p_status.empty else "N/A"
                p_high_count = district_data['Phosphorus'].astype(str).str.strip().str.lower().eq('high').sum()
                
                st.metric("Primary Phosphorus", most_common_p, delta=f"{p_high_count} High" if p_high_count > 0 else None)
                st.caption(f"Count: {p_status.get(most_common_p, 0)} records")
            
            # Crop Recommendations
            st.markdown("---")
            st.subheader("🌾 Crop Recommendations")
            
            crop_counts = district_data['Recommended_Crop'].value_counts()
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Pie chart of crops
                fig_crops = px.pie(
                    values=crop_counts.values, 
                    names=crop_counts.index,
                    title=f"Crop Distribution in {selected_district}"
                )
                st.plotly_chart(fig_crops, use_container_width=True)
            
            with col2:
                st.write("**Recommended Crops:**")
                for idx, (crop, count) in enumerate(crop_counts.items(), 1):
                    st.write(f"{idx}. **{crop}** ({count} records)")
            
            # Expert Advisories
            st.markdown("---")
            st.subheader("📜 Expert Agricultural Advisories")
            
            # Show advisories for each crop
            for crop in crop_counts.head(5).index:
                crop_specific = district_data[district_data['Recommended_Crop'] == crop].iloc[0]
                
                with st.expander(f"🌾 {crop} - Expert Advisory", expanded=False):
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.write("**Crop Details:**")
                        st.write(f"- **Historical MSP:** ₹{crop_specific.get('Historical_MSP_INR', 'N/A')}")
                        st.write(f"- **Soil Type:** {crop_specific.get('Soil_Type', 'N/A')}")
                        st.write(f"- **Soil pH:** {crop_specific.get('Soil_pH', 'N/A')}")
                        st.write(f"- **Rainfall:** {crop_specific.get('Rainfall_IMD_mm', 'N/A')}mm")
                        st.write(f"- **Temperature:** {crop_specific.get('Mean_Temp_Historical', 'N/A')}°C")
                    
                    with col2:
                        st.markdown("#### ✨ 2026 AI Climate-Smart Advisor")
                        st.caption(f"Language: {target_language}")
                        
                        # Gather variables for prompt
                        avg_rain = district_data['Rainfall_IMD_mm'].mean()
                        avg_tmp  = district_data['Mean_Temp_Historical'].mean()
                        msp_val  = crop_specific.get('Historical_MSP_INR', '0')
                        
                        # Gather latest session parameters or use dataset averages
                        params = st.session_state.get('current_params', {
                            "nitrogen": 50.0,
                            "phosphorus": 50.0,
                            "potassium": 50.0,
                            "ph": 6.5,
                            "temperature": avg_tmp,
                            "rainfall": avg_rain,
                        })

                        # Local Language Selector for this advisory
                        st.markdown("🌐 **Translate Strategy:**")
                        advisory_lang = st.selectbox(
                            "Select Advisory Language:",
                            ["English", "Hindi (हिंदी)", "Tamil (தமிழ்)", "Telugu (తెలుగు)", "Marathi (मराठी)", "Punjabi (ਪੰਜਾਬੀ)"],
                            index=["English", "Hindi (हिंदी)", "Tamil (தமிழ்)", "Telugu (తెలుగు)", "Marathi (मराठी)", "Punjabi (ਪੰਜਾਬੀ)"].index(target_language) if target_language in ["English", "Hindi (हिंदी)", "Tamil (தமிழ்)", "Telugu (తెలుగు)", "Marathi (मराठी)", "Punjabi (ਪੰਜਾਬੀ)"] else 0,
                            key=f"lang_sel_{crop}"
                        )

                        # Automatic Generation Logic
                        report_key = f"auto_{selected_district}_{crop}_{advisory_lang}"
                        
                        if report_key not in st.session_state.advisory_cache:
                            with st.spinner(f"📡 AI Scientist analyzing live data for 2026 in {advisory_lang}..."):
                                auto_prompt = f"""
                                ROLE: Senior Agricultural Scientist & Expert Advisor
                                LANGUAGE: {advisory_lang}
                                LOCATION: {selected_district}, {selected_state}
                                TARGET YEAR: 2026
                                
                                DATA:
                                1. STATS: 10-Year Rain: {avg_rain:.1f}mm, Temp: {avg_tmp:.1f}°C, MSP: ₹{msp_val}
                                2. SOIL: N={params['nitrogen']:.1f}, P={params['phosphorus']:.1f}, K={params['potassium']:.1f}, pH={params['ph']:.2f}
                                3. WEATHER: Live {params['temperature']:.1f}°C
                                
                                TASK: Provide a professional 2026 ADVISORY for {crop} in {advisory_lang}.
                                
                                STRUCTURE YOUR RESPONSE EXACTLY LIKE THIS:
                                ### 🌡️ 2026 Climate Forecast
                                [Content here]
                                ### 🧪 Specific Soil Fixes
                                [Content here]
                                ### 💰 Market & ROI Outlook
                                [Content here]
                                ### 🛠️ 2026 Action Plan
                                [Content here]
                                ---SUMMARY---
                                [Very short 1-sentence summary for voice]
                                """
                                raw_response = get_groq_recommendation(auto_prompt, st.session_state.groq_api_key)
                                
                                if raw_response:
                                    if "---SUMMARY---" in raw_response:
                                        parts = raw_response.split("---SUMMARY---")
                                        st.session_state.advisory_cache[report_key] = {
                                            "detailed": parts[0].strip(),
                                            "summary": parts[1].strip()
                                        }
                                    else:
                                        st.session_state.advisory_cache[report_key] = {
                                            "detailed": raw_response,
                                            "summary": "Full strategy ready."
                                        }

                        # Display the cached report beautifully
                        report = st.session_state.advisory_cache.get(report_key, {})
                        st.markdown("---")
                        st.subheader(f"✨ 2026 Expert Strategy: {crop}")
                        
                        detailed_text = report.get("detailed", "Strategy loading...")
                        st.markdown(detailed_text)
                        
            
            # Historical Trends
            st.markdown("---")
            st.subheader("📈 Historical Trends Analysis")
            
            if 'Year' in district_data.columns and len(district_data) > 1:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Rainfall trend
                    fig_rainfall = px.line(
                        district_data.sort_values('Year'),
                        x='Year',
                        y='Rainfall_IMD_mm',
                        color='Recommended_Crop',
                        title=f"Rainfall Trends in {selected_district}",
                        labels={'Rainfall_IMD_mm': 'Rainfall (mm)', 'Year': 'Year'}
                    )
                    st.plotly_chart(fig_rainfall, use_container_width=True)
                
                with col2:
                    # Temperature trend
                    fig_temp = px.line(
                        district_data.sort_values('Year'),
                        x='Year',
                        y='Mean_Temp_Historical',
                        color='Recommended_Crop',
                        title=f"Temperature Trends in {selected_district}",
                        labels={'Mean_Temp_Historical': 'Temperature (°C)', 'Year': 'Year'}
                    )
                    st.plotly_chart(fig_temp, use_container_width=True)
        
        else:
            st.warning(f"⚠️ No data found for {selected_district}, {selected_state}. Please check if the district name is correct.")
            st.info("💡 Try selecting a different district from the dropdown.")
    
    else:
        st.info("📊 Select a state and district above to view detailed analysis from the dataset.")

with tab3:
    st.header("📊 Dataset Visualizations")
    
    if advisory_df.empty:
        st.info("Dataset not available for visualizations")
    else:
        # District selection for visualizations
        st.subheader("� Select Location for Visualization")
        col_viz1, col_viz2 = st.columns(2)
        
        with col_viz1:
            if state_district_mapping:
                available_states = sorted(state_district_mapping.keys())
                viz_state = st.selectbox("🏞️ Select State", available_states, key="tab3_state",
                                        index=available_states.index('Tamil Nadu') if 'Tamil Nadu' in available_states else 0)
            else:
                viz_state = st.text_input("State", "Tamil Nadu", key="tab3_state")
        
        with col_viz2:
            if state_district_mapping and viz_state in state_district_mapping:
                available_districts = ["All Districts"] + state_district_mapping[viz_state]
                viz_district = st.selectbox("📍 Select District", available_districts, key="tab3_district")
            else:
                viz_district = st.text_input("District (or 'All Districts')", "All Districts", key="tab3_district")
        
        st.markdown("---")
        
        # Filter data based on selection
        if viz_district == "All Districts":
            filtered_data = advisory_df[advisory_df['State'] == viz_state]
            location_label = viz_state
        else:
            filtered_data = advisory_df[
                (advisory_df['State'] == viz_state) & 
                (advisory_df['District'] == viz_district)
            ]
            location_label = f"{viz_district}, {viz_state}"
        
        if not filtered_data.empty:
            st.success(f"✅ Showing visualizations for {location_label} ({len(filtered_data)} records)")
            
            # Crop Distribution
            st.subheader(f"🌾 Crop Distribution in {location_label}")
            crop_counts = filtered_data['Recommended_Crop'].value_counts().head(10)
            
            col1, col2 = st.columns([2, 1])
            with col1:
                fig1 = px.bar(
                    x=crop_counts.index, 
                    y=crop_counts.values, 
                    title=f"Top Crops in {location_label}",
                    labels={'x': 'Crops', 'y': 'Number of Records'},
                    color=crop_counts.values,
                    color_continuous_scale='Greens'
                )
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                fig1_pie = px.pie(
                    values=crop_counts.values, 
                    names=crop_counts.index,
                    title="Crop Distribution %"
                )
                st.plotly_chart(fig1_pie, use_container_width=True)
            
            # Climate Analysis Visualizations
            st.markdown("---")
            st.subheader(f"�️ Climate Patterns in {location_label}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Rainfall vs Temperature scatter
                fig2 = px.scatter(
                    filtered_data, 
                    x='Rainfall_IMD_mm', 
                    y='Mean_Temp_Historical',
                    color='Recommended_Crop', 
                    title=f"Climate Patterns: Rainfall vs Temperature",
                    labels={'Rainfall_IMD_mm': 'Rainfall (mm)', 'Mean_Temp_Historical': 'Temperature (°C)'},
                    hover_data=['District', 'Soil_Type']
                )
                st.plotly_chart(fig2, use_container_width=True)
            
            with col2:
                # NDVI vs Soil Moisture
                fig2b = px.scatter(
                    filtered_data,
                    x='Soil_Moisture_Historical',
                    y='NDVI_Vegetation_Index',
                    color='Recommended_Crop',
                    title="Vegetation Index vs Soil Moisture",
                    labels={'Soil_Moisture_Historical': 'Soil Moisture', 'NDVI_Vegetation_Index': 'NDVI'},
                    hover_data=['District', 'Rainfall_IMD_mm']
                )
                st.plotly_chart(fig2b, use_container_width=True)
            
            # Soil Analysis
            st.markdown("---")
            st.subheader(f"🧪 Soil Profile Analysis for {location_label}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Soil pH distribution
                fig3 = px.histogram(
                    filtered_data, 
                    x='Soil_pH', 
                    title=f"Soil pH Distribution",
                    labels={'Soil_pH': 'Soil pH'},
                    nbins=20,
                    color_discrete_sequence=['#2ecc71']
                )
                st.plotly_chart(fig3, use_container_width=True)
            
            with col2:
                # Soil Type distribution
                soil_type_counts = filtered_data['Soil_Type'].value_counts()
                fig3b = px.bar(
                    x=soil_type_counts.index,
                    y=soil_type_counts.values,
                    title="Soil Type Distribution",
                    labels={'x': 'Soil Type', 'y': 'Count'},
                    color=soil_type_counts.values,
                    color_continuous_scale='brwnyl'
                )
                st.plotly_chart(fig3b, use_container_width=True)
            
            # NPK Analysis
            st.markdown("---")
            st.subheader("🧪 Soil Nutrient Analysis (NPK)")
            
            if all(col in filtered_data.columns for col in ['Nitrogen', 'Phosphorus', 'Potassium']):
                # Create nutrient comparison
                nutrient_summary = pd.DataFrame({
                    'Nutrient': ['Nitrogen', 'Phosphorus', 'Potassium'],
                    'High': [
                        (filtered_data['Nitrogen'] == 'High').sum(),
                        (filtered_data['Phosphorus'] == 'High').sum(),
                        (filtered_data['Potassium'] == 'High').sum()
                    ],
                    'Medium': [
                        (filtered_data['Nitrogen'] == 'Medium').sum(),
                        (filtered_data['Phosphorus'] == 'Medium').sum(),
                        (filtered_data['Potassium'] == 'Medium').sum()
                    ],
                    'Low': [
                        (filtered_data['Nitrogen'] == 'Low').sum(),
                        (filtered_data['Phosphorus'] == 'Low').sum(),
                        (filtered_data['Potassium'] == 'Low').sum()
                    ]
                })
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig4 = px.bar(
                        nutrient_summary, 
                        x='Nutrient', 
                        y=['High', 'Medium', 'Low'],
                        title="Soil Nutrient Levels Distribution", 
                        barmode='group',
                        color_discrete_map={'High': '#27ae60', 'Medium': '#f39c12', 'Low': '#e74c3c'}
                    )
                    st.plotly_chart(fig4, use_container_width=True)
                
                with col2:
                    # Stacked percentage
                    nutrient_summary['Total'] = nutrient_summary['High'] + nutrient_summary['Medium'] + nutrient_summary['Low']
                    nutrient_summary['High_%'] = (nutrient_summary['High'] / nutrient_summary['Total'] * 100).round(1)
                    nutrient_summary['Medium_%'] = (nutrient_summary['Medium'] / nutrient_summary['Total'] * 100).round(1)
                    nutrient_summary['Low_%'] = (nutrient_summary['Low'] / nutrient_summary['Total'] * 100).round(1)
                    
                    fig4b = px.bar(
                        nutrient_summary,
                        x='Nutrient',
                        y=['High_%', 'Medium_%', 'Low_%'],
                        title="Nutrient Levels (% Distribution)",
                        barmode='stack',
                        labels={'value': 'Percentage', 'variable': 'Level'},
                        color_discrete_map={'High_%': '#27ae60', 'Medium_%': '#f39c12', 'Low_%': '#e74c3c'}
                    )
                    st.plotly_chart(fig4b, use_container_width=True)
            
            # Historical Trends (if Year column exists)
            if 'Year' in filtered_data.columns and len(filtered_data['Year'].unique()) > 1:
                st.markdown("---")
                st.subheader(f"📈 Historical Trends for {location_label}")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Rainfall trend over years
                    yearly_rainfall = filtered_data.groupby('Year')['Rainfall_IMD_mm'].mean().reset_index()
                    fig5 = px.line(
                        yearly_rainfall,
                        x='Year',
                        y='Rainfall_IMD_mm',
                        title="Average Rainfall Trend Over Years",
                        labels={'Rainfall_IMD_mm': 'Avg Rainfall (mm)', 'Year': 'Year'},
                        markers=True
                    )
                    st.plotly_chart(fig5, use_container_width=True)
                
                with col2:
                    # Temperature trend over years
                    yearly_temp = filtered_data.groupby('Year')['Mean_Temp_Historical'].mean().reset_index()
                    fig6 = px.line(
                        yearly_temp,
                        x='Year',
                        y='Mean_Temp_Historical',
                        title="Average Temperature Trend Over Years",
                        labels={'Mean_Temp_Historical': 'Avg Temperature (°C)', 'Year': 'Year'},
                        markers=True,
                        color_discrete_sequence=['#e74c3c']
                    )
                    st.plotly_chart(fig6, use_container_width=True)
        
        else:
            st.warning(f"⚠️ No data found for {location_label}")
            st.info("Try selecting a different location.")

with tab4:
    st.header("🌡️ 7-Day Weather Forecast")
    
    if not st.session_state.location_data:
        st.info("👈 Enter location in Chatbot tab to see forecast")
    else:
        location = st.session_state.location_data.get('location', 'Delhi, India')
        lat = st.session_state.location_data.get('lat')
        lon = st.session_state.location_data.get('lon')
        
        if lat and lon:
            forecast_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&daily=temperature_2m_max,temperature_2m_min,precipitation_sum&timezone=auto&forecast_days=7"
            try:
                forecast_response = requests.get(forecast_url, timeout=10)
                if forecast_response.status_code == 200:
                    data = forecast_response.json()
                    daily = data['daily']
                    forecast_days = []
                    for i in range(len(daily['time'])):
                        forecast_days.append({
                            'date': daily['time'][i],
                            'max_temp': daily['temperature_2m_max'][i],
                            'min_temp': daily['temperature_2m_min'][i],
                            'avg_temp': (daily['temperature_2m_max'][i] + daily['temperature_2m_min'][i]) / 2,
                            'condition': "Clear" if daily['precipitation_sum'][i] == 0 else "Rainy",
                            'rainfall': daily['precipitation_sum'][i],
                            'humidity': 50,
                            'wind_speed': 10,
                            'uv_index': 5
                        })
                    forecast_data = {'location': location, 'forecast': forecast_days}
                else:
                    forecast_data = None
            except:
                forecast_data = None
        else:
            forecast_data = None
        
        if forecast_data:
            st.subheader(f"📍 {forecast_data['location']}")
            
            forecast_df = pd.DataFrame(forecast_data['forecast'])
            forecast_df['date'] = pd.to_datetime(forecast_df['date'])
            forecast_df['day_name'] = forecast_df['date'].dt.strftime('%a, %b %d')
            
            # Temperature chart
            fig_temp = go.Figure()
            fig_temp.add_trace(go.Scatter(
                x=forecast_df['day_name'],
                y=forecast_df['max_temp'],
                name='Max Temp',
                line=dict(color='#ff6b6b', width=3),
                mode='lines+markers'
            ))
            fig_temp.add_trace(go.Scatter(
                x=forecast_df['day_name'],
                y=forecast_df['min_temp'],
                name='Min Temp',
                line=dict(color='#4ecdc4', width=3),
                mode='lines+markers',
                fill='tonexty',
                fillcolor='rgba(78, 205, 196, 0.2)'
            ))
            fig_temp.update_layout(
                title="Temperature Forecast (7 Days)",
                xaxis_title="Date",
                yaxis_title="Temperature (°C)",
                height=400,
                template="plotly_white"
            )
            st.plotly_chart(fig_temp, use_container_width=True)
            
            # Daily forecast cards
            st.subheader("📅 Daily Forecast Details")
            cols = st.columns(7)
            for idx, day in enumerate(forecast_data['forecast']):
                with cols[idx]:
                    st.metric(
                        day['date'].split('-')[2],
                        f"{day['max_temp']}°C",
                        f"Min: {day['min_temp']}°C"
                    )
                    st.caption(f"🌧️ {day['rainfall']}mm")
                    st.caption(f"💨 {day['wind_speed']} km/h")
                    st.caption(day['condition'])
        else:
            st.warning("Weather forecast data unavailable")

with tab5:
    st.header("🌾 Crop Recommendations & Analysis")
    
    if not st.session_state.location_data:
        st.info("👈 Enter location and parameters in Chatbot tab")
    else:
        # Dataset-based Crop Recommendations
        st.subheader("📊 Dataset-Based Crop Analysis")
        
        if not advisory_df.empty:
            location_parts = st.session_state.location_data.get('location', '').split(', ')
            state_name = location_parts[1] if len(location_parts) > 1 else location_parts[0]
            
            # Filter by state
            state_data = advisory_df[advisory_df['State'].str.contains(state_name, case=False, na=False)]
            
            if not state_data.empty:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader(f"🌾 Top Crops in {state_name}")
                    top_crops = state_data['Recommended_Crop'].value_counts().head(8)
                    fig = px.pie(values=top_crops.values, names=top_crops.index,
                               title=f"Crop Distribution in {state_name}")
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.subheader("🌡️ Climate Suitability")
                    # Show climate ranges for top crops
                    for crop in top_crops.head(5).index:
                        crop_data = state_data[state_data['Recommended_Crop'] == crop]
                        avg_temp = crop_data['Mean_Temp_Historical'].mean()
                        avg_rainfall = crop_data['Rainfall_IMD_mm'].mean()
                        st.write(f"**{crop}:** {avg_temp:.1f}°C, {avg_rainfall:.0f}mm")
                
                # Expert Advisory from Dataset
                st.markdown("---")
                st.subheader("📜 Expert Advisory from Dataset")
                
                # Get a sample advisory
                sample_advisory = state_data.sample(1).iloc[0] if len(state_data) > 0 else None
                if sample_advisory is not None:
                    with st.expander(f"Sample Advisory for {sample_advisory['Recommended_Crop']} in {sample_advisory['District']}"):
                        st.write(f"**Crop:** {sample_advisory['Recommended_Crop']}")
                        st.write(f"**District:** {sample_advisory['District']}")
                        st.write(f"**Soil Type:** {sample_advisory['Soil_Type']}")
                        st.write(f"**Advisory:** {sample_advisory['Expert_Advisory'][:300]}...")
        
        # Soil Health Analysis
        st.markdown("---")
        st.subheader("🧪 Soil Health Analysis")
        soil_score = calculate_soil_health_score(st.session_state.soil_params)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Overall Score", f"{soil_score['total_score']}/100")
        with col2:
            st.metric("Health Level", soil_score['level'])
        with col3:
            st.metric("Status", "✅ Good" if soil_score['total_score'] >= 60 else "⚠️ Needs Improvement")
        
        # Recommendations from AI History
        if st.session_state.recommendation_history:
            st.markdown("---")
            st.subheader("🤖 Latest AI Recommendation")
            latest = st.session_state.recommendation_history[-1]
            with st.expander("View Full AI Recommendation", expanded=True):
                st.markdown(latest.get('recommendation', 'No recommendation available'))
        else:
            st.info("💬 Ask questions in the Chatbot tab to get AI recommendations!")

with tab6:
    st.header("📜 Recommendation History & Export")
    
    if st.session_state.recommendation_history:
        st.success(f"✅ {len(st.session_state.recommendation_history)} recommendations saved")
        
        # Export options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Export as JSON with custom serializer for numpy types
            import numpy as np
            
            def convert_to_json_serializable(obj):
                """Convert numpy types to Python native types"""
                if isinstance(obj, (np.integer, np.floating)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_to_json_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_to_json_serializable(item) for item in obj]
                return obj
            
            # Convert all data to JSON-serializable format
            serializable_history = convert_to_json_serializable(st.session_state.recommendation_history)
            json_data = json.dumps(serializable_history, indent=2)
            
            st.download_button(
                label="📥 Download JSON",
                data=json_data,
                file_name=f"recommendations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        with col2:
            # Export as CSV
            df_export = pd.DataFrame([
                {
                    'Timestamp': rec.get('timestamp', ''),
                    'Location': rec.get('location', ''),
                    'Question': rec.get('question', ''),
                    'Temperature': rec.get('weather', {}).get('temperature', ''),
                    'Humidity': rec.get('weather', {}).get('humidity', ''),
                    'N': rec.get('soil', {}).get('N', ''),
                    'P': rec.get('soil', {}).get('P', ''),
                    'K': rec.get('soil', {}).get('K', ''),
                    'pH': rec.get('soil', {}).get('pH', ''),
                    'Recommendation': rec.get('recommendation', '')[:200] + '...' if len(rec.get('recommendation', '')) > 200 else rec.get('recommendation', '')
                }
                for rec in st.session_state.recommendation_history
            ])
            csv_data = df_export.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv_data,
                file_name=f"recommendations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with col3:
            # Clear history with confirmation
            st.write("**🗑️ Clear Data:**")
            if st.button("🗑️ Clear All History", type="secondary"):
                st.session_state.recommendation_history = []
                st.success("✅ History cleared!")
                st.rerun()
        
        st.markdown("---")
        
        # Display history
        st.subheader("📋 All Recommendations")
        for idx, rec in enumerate(reversed(st.session_state.recommendation_history), 1):
            with st.expander(f"#{len(st.session_state.recommendation_history) - idx + 1} - {rec.get('timestamp', 'N/A')} | {rec.get('location', 'N/A')}", expanded=False):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**📍 Location:**", rec.get('location', 'N/A'))
                    st.write("**🌾 Crop:**", rec.get('crop', 'N/A'))
                    st.write("**❓ Question:**", rec.get('question', 'N/A'))
                    st.write("**📅 Time:**", rec.get('timestamp', 'N/A'))
                    st.write("**🤖 AI Backend:**", rec.get('ai_backend', 'N/A'))
                
                with col2:
                    params = rec.get('parameters', {})
                    st.write("**🌤️ Parameters:**")
                    st.write(f"- Rainfall: {params.get('rainfall', 'N/A')}mm")
                    st.write(f"- Temperature: {params.get('temperature', 'N/A')}°C")
                    
                    soil = params.get('soil', {})
                    st.write("**🧪 Soil:**")
                    st.write(f"- N: {soil.get('N', 'N/A')}, P: {soil.get('P', 'N/A')}, K: {soil.get('K', 'N/A')}")
                    st.write(f"- pH: {soil.get('pH', 'N/A')}")
                    
                    ml_pred = rec.get('ml_prediction', {})
                    if ml_pred:
                        st.write("**🔮 ML Predictions:**")
                        st.write(f"- Crop Suitability: {ml_pred.get('crop_suitability', 0):.2f}")
                        st.write(f"- Climate Risk: {ml_pred.get('climate_risk', 0):.2f}")
                
                st.markdown("**💡 AI Recommendation:**")
                st.success(rec.get('recommendation', 'N/A'))
                st.markdown("---")
    else:
        st.info("📝 No recommendations yet. Start chatting in the Chatbot tab to generate recommendations!")
        st.markdown("""
        **How to generate recommendations:**
        1. Go to the **💬 Chatbot** tab
        2. Enter a location
        3. Set soil parameters
        4. Ask questions like:
           - "What crops should I grow?"
           - "What's the best crop for my soil?"
           - "Recommend suitable crops"
        
        Recommendations will be automatically saved here!
        """)

# Load LoRA model if requested
if use_local_model and TRANSFORMERS_AVAILABLE and not st.session_state.model_loaded:
    current_model = st.session_state.get('model_choice', 'climate_advisor_lora')
    with st.spinner("Loading LoRA model... This may take a few minutes..."):
        model, tokenizer = load_lora_model(current_model)
        if model and tokenizer:
            st.session_state.lora_model = model
            st.session_state.tokenizer = tokenizer
            st.session_state.model_loaded = True
            st.success("✅ LoRA model loaded!")
        else:
            st.warning("⚠️ LoRA model loading failed. Using Groq API instead.")
            use_local_model = False

# Auto-load T5-PEFT Model (Agriculture-Specific)
if st.session_state.t5_peft_model is None:
    try:
        model, tokenizer = load_t5_peft_model()
        if model is not None:
            st.session_state.t5_peft_model = model
            st.session_state.t5_peft_tokenizer = tokenizer
    except Exception as e:
        pass

# Auto-load LoRA Model (Climate-Adaptive)
if st.session_state.lora_model is None:
    try:
        model, tokenizer = load_lora_model()
        if model is not None:
            st.session_state.lora_model = model
            st.session_state.tokenizer = tokenizer
    except Exception as e:
        pass

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 20px; color: #666;">
    <p><strong>🌾 Climate Resilience Chatbot for Farmers</strong></p>
</div>
""", unsafe_allow_html=True)
