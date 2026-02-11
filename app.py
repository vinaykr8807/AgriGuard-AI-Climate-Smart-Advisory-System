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

# Language Mapping for Translation & Voice
LANG_MAP = {
    "Hindi (हिंदी)": {"code": "hi", "tld": "co.in", "name": "Hindi"},
    "Tamil (தமிழ்)": {"code": "ta", "tld": "co.in", "name": "Tamil"},
    "Telugu (తెలుగు)": {"code": "te", "tld": "co.in", "name": "Telugu"},
    "Marathi (मराठी)": {"code": "mr", "tld": "co.in", "name": "Marathi"},
    "Punjabi (ਪੰਜਾਬੀ)": {"code": "pa", "tld": "co.in", "name": "Punjabi"},
    "English": {"code": "en", "tld": "co.in", "name": "English"}
}

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
        advisory = pd.read_csv("data/Multilingual_Expert_Advisory.csv", encoding='utf-8')
    except FileNotFoundError:
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

# API Keys - Replace with your own or use Streamlit secrets
DEFAULT_WEATHERAPI_KEY = "ENTER_YOUR_WEATHER_API_KEY"
DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_MODEL = "gemma3:4b"  # Using 4b for faster responses, you can change to "llama3.2:1b" for even faster
DEFAULT_GROQ_API_KEY = "ENTER_YOUR_GROQ_API_KEY"  # Get from https://console.groq.com

# Initialize API keys in session state
if 'weather_api_key' not in st.session_state:
    st.session_state.weather_api_key = DEFAULT_WEATHERAPI_KEY
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

def get_groq_recommendation(prompt: str, api_key: str = None) -> Optional[str]:
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
                    "content": "You are a specialized agricultural synthesis engine for Indian farmers. Your goal is to provide evidence-based, factual advice."
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
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # State selection from CSV
        if state_district_mapping:
            available_states = sorted(state_district_mapping.keys())
            state = st.selectbox("🏞️ State", available_states, index=available_states.index('Tamil Nadu') if 'Tamil Nadu' in available_states else 0)
            
            # District selection based on selected state
            if state in state_district_mapping:
                available_districts = state_district_mapping[state]
                district = st.selectbox("📍 District", available_districts)
            else:
                district = st.text_input("District", "Chennai")
            
            # Update location based on state and district
            location = f"{district}, {state}, India"
            st.caption(f"📌 Location: {location}")
        else:
            # Fallback when CSV data is not available
            location = st.text_input("📍 Location", "Chennai, India", help="Enter city name (e.g., Chennai, Mumbai, Delhi)")
            district = st.text_input("District", "Chennai")
            state = "Tamil Nadu"
            st.info("📊 Using fallback options (CSV data not available)")
        
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
    
    with col2:
        rainfall = st.slider("Rainfall (mm)", 0, 500, 120)
        temperature = st.slider("Temperature (°C)", 15, 50, 30)
    
    with col3:
        nitrogen = st.slider("N (kg/ha)", 0, 300, 50)
        phosphorus = st.slider("P (kg/ha)", 0, 100, 50)
        potassium = st.slider("K (kg/ha)", 0, 200, 50)
        ph = st.slider("pH", 3.0, 10.0, 6.5)
    
    # Update session state
    st.session_state.soil_params = {'N': nitrogen, 'P': phosphorus, 'K': potassium, 'pH': ph}
    
    # Get coordinates from CSV for selected location
    location_lat = None
    location_lon = None
    if not advisory_df.empty and 'state' in locals() and 'district' in locals():
        location_data_df = advisory_df[(advisory_df['State'] == state) & (advisory_df['District'] == district)]
        if not location_data_df.empty:
            location_lat = location_data_df.iloc[0].get('Lat', None)
            location_lon = location_data_df.iloc[0].get('Lon', None)
    
    # Get weather data for the entered location
    if location_lat and location_lon:
        # Use CSV coordinates for more reliable weather fetching
        weather_data = get_weather_data_by_coords(location_lat, location_lon, location)
        if weather_data:
            st.session_state.location_data = weather_data
            # Show current weather in a compact format
            col_w1, col_w2 = st.columns(2)
            with col_w1:
                st.success(f"✅ {district}, {state}")
            with col_w2:
                st.info(f"Current: {weather_data.get('temperature', 'N/A')}°C, {weather_data.get('humidity', 'N/A')}%")
        else:
            st.warning(f"⚠️ Could not fetch weather data for {location}")
    elif location:
        # Fallback to geocoding
        weather_data = get_weather_data(location)
        if weather_data:
            st.session_state.location_data = weather_data
            # Show current weather in a compact format
            col_w1, col_w2 = st.columns(2)
            with col_w1:
                st.success(f"✅ {weather_data.get('location', location)}")
            with col_w2:
                st.info(f"Current: {weather_data.get('temperature', 'N/A')}°C, {weather_data.get('humidity', 'N/A')}%")
        else:
            st.warning(f"⚠️ Could not fetch weather data for {location}")
    
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
                (advisory_df['State'] == state_name) & 
                (advisory_df['District'] == district)
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
                    all_advisories = crop_matches['Expert_Advisory'].dropna().tolist()
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
        
        def calculate_risk_score(nitrogen, phosphorus, potassium, ph, rainfall, temperature):
            """
            Calculate climate risk score (0-100, lower is better)
            """
            risk = 0.0
            
            # Nutrient imbalance risk
            if nitrogen < 40 or nitrogen > 200:
                risk += 15
            if phosphorus < 20 or phosphorus > 80:
                risk += 10
            if potassium < 40 or potassium > 180:
                risk += 10
            
            # pH risk
            if ph < 5.0 or ph > 8.0:
                risk += 20
            elif ph < 5.5 or ph > 7.5:
                risk += 10
            
            # Rainfall risk
            if rainfall < 200:
                risk += 25  # Drought risk
            elif rainfall > 2000:
                risk += 15  # Flood risk
            
            # Temperature risk
            if temperature < 10 or temperature > 40:
                risk += 20  # Extreme temperature
            
            return min(100, risk)
        
        # Calculate scores using heuristics
        crop_pred, penalties = calculate_heuristic_score(
            nitrogen, phosphorus, potassium, ph, rainfall, temperature, crop
        )
        risk_pred = calculate_risk_score(
            nitrogen, phosphorus, potassium, ph, rainfall, temperature
        )
        
        # Display results
        suitability_label = "Excellent" if crop_pred > 80 else "Good" if crop_pred > 60 else "Moderate" if crop_pred > 40 else "Poor"
        risk_label = "Low" if risk_pred < 20 else "Moderate" if risk_pred < 40 else "High"
        
        st.success(f"🌾 **Crop Suitability: {crop_pred:.1f}%** ({suitability_label}) | **Climate Risk: {risk_pred:.1f}%** ({risk_label})")
        
        # Show breakdown
        with st.expander("🔍 Score Breakdown & Analysis", expanded=False):
            st.write(f"""
**Your Input Parameters:**
- 🌱 Nitrogen: {nitrogen} kg/ha
- 🌱 Phosphorus: {phosphorus} kg/ha
- 🌱 Potassium: {potassium} kg/ha
- 🧪 pH: {ph}
- 🌧️ Rainfall: {rainfall}mm
- 🌡️ Temperature: {temperature}°C
- 🌾 Crop: {crop}

**Suitability Score: {crop_pred:.1f}%**
- Rating: {suitability_label}
- Based on NPK balance, pH, rainfall, and temperature requirements for {crop}

**Risk Score: {risk_pred:.1f}%**
- Risk Level: {risk_label}
- Considers nutrient imbalances, pH extremes, drought/flood, and temperature stress
            """)
            
            if penalties:
                st.warning("**⚠️ Factors Reducing Your Score:**")
                for penalty in penalties:
                    st.write(f"• {penalty}")
                    
                st.info("**💡 Recommendations:**")
                if any("nitrogen" in p.lower() for p in penalties):
                    st.write("• Consider soil testing and nitrogen fertilizer application")
                if any("phosphorus" in p.lower() for p in penalties):
                    st.write("• Add phosphate fertilizers or organic compost")
                if any("potassium" in p.lower() for p in penalties):
                    st.write("• Apply potash or wood ash to increase potassium")
                if any("ph" in p.lower() for p in penalties):
                    st.write("• Adjust soil pH using lime (to increase) or sulfur (to decrease)")
                if any("rainfall" in p.lower() for p in penalties):
                    st.write("• Consider irrigation systems or water conservation methods")
                if any("temperature" in p.lower() or "cold" in p.lower() or "hot" in p.lower() for p in penalties):
                    st.write("• Consider crop timing or protected cultivation methods")

        
        
        
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
        context = f"""You are an expert agricultural advisor. Analyze the following comprehensive data and provide detailed, actionable advice.

=== LOCATION ===
State: {state_name}
District: {district}
Crop: {crop}

=== CURRENT INPUT PARAMETERS ===
Rainfall Input: {rainfall}mm
Temperature Input: {temperature}°C
Soil Nutrients - N: {nitrogen}, P: {phosphorus}, K: {potassium}
Soil pH: {ph}

=== ML PREDICTION ===
Crop Suitability Score: {crop_pred:.2f}
Climate Risk Score: {risk_pred:.2f}
{live_weather_context}
{relevant_data}

=== FARMER'S QUESTION ===
{user_question}

=== INSTRUCTIONS ===
Provide comprehensive agricultural advice by:
1. Analyzing the expert advisory data provided above (these are important historical insights)
2. Augmenting it with the current live weather conditions
3. Considering the ML predictions and soil parameters
4. Giving specific, actionable recommendations for the farmer
5. Mentioning any climate adaptation strategies needed

Your response:
"""
        
        # 🎯 ENSEMBLE APPROACH: Use multiple models and merge responses with Groq
        st.info("🔬 **Ensemble AI Mode**: Gathering insights from multiple models...")
        
        ensemble_responses = {}
        
        # Try T5-PEFT (Agriculture-Specific Model)
        if st.session_state.t5_peft_model:
            with st.spinner("🌾 T5-PEFT analyzing..."):
                t5_response = generate_with_t5_peft(context, st.session_state.t5_peft_model, st.session_state.t5_peft_tokenizer, max_tokens)
                if t5_response:
                    ensemble_responses['T5-PEFT'] = t5_response
                    st.success("✅ T5-PEFT response received")
        
        # Try Climate-LoRA (TinyLlama-based Adaptive Model)
        if st.session_state.lora_model:
            with st.spinner("🌡️ Climate-LoRA analyzing..."):
                lora_response = generate_with_lora(context, st.session_state.lora_model, st.session_state.tokenizer, max_tokens)
                if lora_response:
                    ensemble_responses['Climate-LoRA'] = lora_response
                    st.success("✅ Climate-LoRA response received")
        
        # Try Ollama (General Knowledge)
        with st.spinner("🤖 Ollama analyzing..."):
            ollama_response = get_ollama_recommendation(context, st.session_state.ollama_model, 0.7, max_tokens, st.session_state.ollama_host)
            if ollama_response:
                ensemble_responses['Ollama'] = ollama_response
                st.success("✅ Ollama response received")
        
        # Now merge all responses using Groq
        if len(ensemble_responses) > 0:
            with st.spinner("🔄 Groq synthesizing all responses..."):
                # Create FACTUAL synthesis prompt with all historical and real-time data
                synthesis_prompt = f"""You are a DATA-DRIVEN agricultural synthesis engine. Your role is to combine insights from multiple AI models into ONE factual, evidence-based recommendation.

CRITICAL RULES:
1. ONLY use information from the provided data sources below
2. DO NOT make up or assume any information
3. PRIORITIZE the 10-year expert advisory data (proven historical advice)
4. AUGMENT with real-time weather data for current conditions
5. If data conflicts, prefer expert advisory over AI speculation

=== FARMER'S QUESTION ===
{user_question}

=== LOCATION & CROP ===
Location: {district}, {state_name if 'state_name' in locals() else state}
Crop: {crop}

=== 10-YEAR HISTORICAL EXPERT ADVISORY (2015-2024) ===
{expert_advisory_full if expert_advisory_full else "No historical advisory available"}

=== COMPREHENSIVE HISTORICAL DATA ===
{relevant_data if relevant_data else "Limited historical data"}

=== REAL-TIME WEATHER DATA (CURRENT) ===
{live_weather_context if live_weather_context else "No live weather available"}

=== ML PREDICTIONS (CURRENT) ===
Crop Suitability Score: {crop_pred:.2f}
Climate Risk Score: {risk_pred:.2f}

=== AI MODEL ANALYSIS ===
"""
                for model_name, response in ensemble_responses.items():
                    synthesis_prompt += f"\n{model_name} Analysis:\n{response}\n---\n"
                
                synthesis_prompt += f"""
=== YOUR SYNTHESIS TASK ===
You are a DYNAMIC and FACTUAL agricultural advisor. Your primary goal is to DIRECTLY ANSWER THE FARMER'S QUESTION based on the data provided.

🎯 **THE FARMER ASKED**: "{user_question}"

**CRITICAL**: Your response MUST be UNIQUE and SPECIFIC to this question. Analyze what they're REALLY asking:
- If they ask about "dry land in winter" → Focus on drought management, alternative crops, irrigation
- If they ask about "pests" → Focus on pest control, prevention strategies
- If they ask about "fertilizer" → Focus on NPK recommendations, soil health

1. **FIRST PRIORITY**: Directly address the "FARMER'S QUESTION" ({user_question}). Do not give a generic response; make it specific to what they asked.
2. **EVIDENCE-BASED**: Use the 10-year expert advisory and comprehensive data to justify your answer.
3. **WEATHER-AWARE**: Current conditions: {current_weather.get('temperature', 'N/A') if 'current_weather' in locals() and current_weather else 'N/A'}°C, {current_weather.get('humidity', 'N/A') if 'current_weather' in locals() and current_weather else 'N/A'}% humidity, {current_weather.get('rainfall', 'N/A') if 'current_weather' in locals() and current_weather else 'N/A'}mm rainfall
4. **ML-GROUNDED**: Crop Suitability: {crop_pred:.1f}% (interpret: {"Excellent" if crop_pred > 70 else "Good" if crop_pred > 50 else "Moderate" if crop_pred > 30 else "Poor"}), Risk: {risk_pred:.1f}% (interpret: {"Low" if risk_pred < 20 else "Moderate" if risk_pred < 40 else "High"})

=== STRUCTURE YOUR RESPONSE ===
Start with a direct answer to the question, then use these sections:

### 📋 IMMEDIATE ACTIONS
[Provide 3-5 specific, high-priority actions for the NEXT 24-48 HOURS based on current weather + the user's question]

### 🌾 EXPERT INSIGHTS ({crop})
[Crucial best practices from the 10-year dataset that are relevant to the user's question]

### 🌡️ CLIMATE & SOIL ADJUSTMENTS
[How current conditions compare to historical ranges and what the farmer should change right now]

### ⚠️ CRITICAL RISKS
[Evidence-based threats for {district} and how to mitigate them according to the data]

6. **QUOTE actual data points** (e.g., "Avg Rainfall: 700mm") and use simple, relatable language.

YOUR FACTUAL RESPONSE:"""
                
                # Use Groq to synthesize with higher temperature for better dynamic responses
                try:
                    import groq
                    client = groq.Groq(api_key=st.session_state.groq_api_key)
                    
                    chat_completion = client.chat.completions.create(
                        messages=[
                            {"role": "system", "content": f"You are an expert agricultural advisor for {district}, {state_name}. You MUST provide SPECIFIC answers to the farmer's question: '{user_question}'. Use the provided data to give UNIQUE, CONTEXTUAL advice. DO NOT give generic responses."},
                            {"role": "user", "content": synthesis_prompt}
                        ],
                        model="llama-3.3-70b-versatile",
                        temperature=1.0, # Increased to 1.0 for maximum variation and creativity
                        max_tokens=1500,
                        top_p=0.95, # Add nucleus sampling for more diverse outputs
                    )
                    final_response = chat_completion.choices[0].message.content
                except:
                    final_response = get_groq_recommendation(synthesis_prompt, st.session_state.groq_api_key)
                
                if final_response:
                    # Always process through translation/summary engine for consistency
                    with st.spinner(f"🌐 Processing {st.session_state.target_language} summary..."):
                        translation_result = translate_text(final_response, st.session_state.target_language, st.session_state.groq_api_key)
                        response = translation_result["detailed"]
                        voice_summary = translation_result["summary"]
                    
                    ai_backend_used = f"Ensemble ({', '.join(ensemble_responses.keys())} → Groq Factual Synthesis)"
                    st.success(f"✅ **Factual Synthesis & {st.session_state.target_language} Summary Complete!**")
                    
                    # Voice Output
                    if st.session_state.enable_voice:
                        with st.spinner("🎙️ Generating voice response..."):
                            speak_text(voice_summary, st.session_state.target_language)
                else:
                    # If Groq fails, use the best available response BUT enhance it with Groq
                    raw_response = ensemble_responses.get('T5-PEFT') or ensemble_responses.get('Ollama') or list(ensemble_responses.values())[0]
                    
                    # IMPORTANT: Expand brief response into comprehensive advisory using Groq
                    expansion_prompt = f"""Based on the brief agricultural insight below, create a COMPREHENSIVE, DETAILED advisory report.

BRIEF INSIGHT:
{raw_response}

CONTEXT:
Location: {district}, {state_name if 'state_name' in locals() else state}
Crop: {crop}
Question: {user_question}

Expert Advisory Data: {expert_advisory_full[:500] if expert_advisory_full else 'N/A'}

CREATE A DETAILED REPORT with these sections:
### 📋 IMMEDIATE ACTIONS
### 🌾 EXPERT INSIGHTS
### 🌡️ CLIMATE & SOIL ADJUSTMENTS
### ⚠️ CRITICAL RISKS

Make it specific, actionable, and comprehensive (at least 300 words)."""

                    expanded_response = get_groq_recommendation(expansion_prompt, st.session_state.groq_api_key)
                    
                    if expanded_response:
                        # Use expanded version
                        translation_result = translate_text(expanded_response, st.session_state.target_language, st.session_state.groq_api_key)
                        response = translation_result["detailed"]
                        voice_summary = translation_result["summary"]
                        ai_backend_used = f"Ensemble ({', '.join(ensemble_responses.keys())}) → Groq Expansion"
                    else:
                        # Last resort: use raw response
                        translation_result = translate_text(raw_response, st.session_state.target_language, st.session_state.groq_api_key)
                        response = translation_result["detailed"]
                        voice_summary = translation_result["summary"]
                        ai_backend_used = f"Ensemble ({', '.join(ensemble_responses.keys())})"
                    
                    if st.session_state.enable_voice:
                        speak_text(voice_summary, st.session_state.target_language)
        else:
            # No ensemble responses - try Groq directly
            with st.spinner("☁️ Trying Groq API directly..."):
                raw_response = get_groq_recommendation(context, st.session_state.groq_api_key)
                if raw_response:
                    with st.spinner(f"🌐 Processing {st.session_state.target_language} summary..."):
                        translation_result = translate_text(raw_response, st.session_state.target_language, st.session_state.groq_api_key)
                        response = translation_result["detailed"]
                        voice_summary = translation_result["summary"]
                    
                    if st.session_state.enable_voice:
                        with st.spinner("🎙️ Generating voice response..."):
                            speak_text(voice_summary, st.session_state.target_language)
                    ai_backend_used = "Groq API (Cloud)"
                else:
                    response = None
                    ai_backend_used = None
        
        # Error handling
        if not response:
            st.error("❌ **Unable to generate AI response**")
            st.warning("""
            **Please configure at least one AI backend:**
            
            **Option 1: T5-PEFT Model (Best for Agriculture)**
            - Already available in `models/LLM` directory
            - Install dependencies: `pip install transformers peft`
            - Reload the app to auto-load
            
            **Option 2: Ollama (Recommended - Free & Local)**
            1. Download from: https://ollama.ai
            2. Install and run: `ollama pull llama3.2:1b`
            3. Ollama will run on http://localhost:11434
            
            **Option 3: Groq API (Free Cloud Alternative)**
            1. Get free API key from: https://console.groq.com
            2. Already configured with your key!
            
            **Option 4: Use LoRA Model (Advanced)**
            - Requires transformers and model files in models/ directory
            """)
        else:
            # Display successful response
            st.success(f"✅ **Response generated using:** {ai_backend_used}")
        if response:
            # Display what data was used for analysis
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
                            title="Climate Trends (Rainfall vs Temp)",
                            xaxis_title="Year",
                            yaxis=dict(title="Rainfall (mm)", titlefont=dict(color="#4FC3F7")),
                            yaxis2=dict(title="Temp (°C)", titlefont=dict(color="#FF7043"), overlaying="y", side="right"),
                            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                            font=dict(color="white"), height=350,
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
                st.metric("Primary Soil Type", most_common_soil)
            
            with col3:
                # NPK analysis (these might be categorical)
                nitrogen_high = (district_data['Nitrogen'] == 'High').sum()
                st.metric("High Nitrogen Records", nitrogen_high)
            
            with col4:
                phosphorus_high = (district_data['Phosphorus'] == 'High').sum()
                st.metric("High Phosphorus Records", phosphorus_high)
            
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
                        st.write("**Expert Advisory (English):**")
                        st.info(crop_specific.get('Expert_Advisory', 'N/A'))
                        
                        # Multilingual support - using checkbox instead of nested expander
                        if pd.notna(crop_specific.get('Advisory_Hindi')):
                            st.markdown("---")
                            show_translations = st.checkbox(f"🇮🇳 Show in Other Languages", key=f"translate_{crop}")
                            if show_translations:
                                st.write("**Hindi (हिन्दी):**")
                                st.success(crop_specific.get('Advisory_Hindi', 'N/A'))
                                st.write("**Tamil (தமிழ்):**")
                                st.success(crop_specific.get('Advisory_Tamil', 'N/A'))
            
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
