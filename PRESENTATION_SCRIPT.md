# 🌾 AgriGuard AI - 10-Minute Presentation Script

## **SLIDE 1: Title & Introduction (30 seconds)**

**[Screen: Show app.py in VS Code]**

"Good morning/afternoon everyone. Today I'll present **AgriGuard AI** - a Climate-Smart Agricultural Advisory System that combines AI, real-time weather data, and 10 years of historical agricultural data to help Indian farmers make informed decisions.

In the next 10 minutes, I'll walk you through:
1. Project architecture
2. Key code components
3. Frontend features
4. Live demonstration"

---

## **SLIDE 2: Project Structure Overview (1 minute)**

**[Screen: Show folder structure in VS Code Explorer]**

"Let me start with the project structure:

```
AgriGuard-AI/
├── app.py                    # Main Streamlit application (2,800+ lines)
├── domain_prompts.py         # Domain-specific AI prompt engine
├── load_model.py             # LoRA model loader
├── requirements.txt          # Dependencies
├── data/
│   ├── merged_feature_store.csv           # 10-year climate data
│   ├── Multilingual_Expert_Advisory.csv   # Expert advisories
│   └── Unified_Decadal_Master_2015_2024.csv
└── models/
    ├── LLM/                  # T5-PEFT model (850MB)
    └── climate_advisor_lora/ # LoRA adapter (120MB)
```

**Key highlight**: This is a production-ready system with 6,690+ expert advisories covering 36 states and 800+ districts."

---

## **SLIDE 3: Core Dependencies (45 seconds)**

**[Screen: Open requirements.txt]**

"Let's look at our technology stack in `requirements.txt`:

```python
streamlit>=1.28.0      # Web framework
requests>=2.31.0       # API calls
pandas>=2.0.0          # Data processing
plotly>=5.15.0         # Interactive visualizations
transformers           # Hugging Face models
peft                   # LoRA adapters
torch                  # Deep learning
gtts                   # Voice generation
```

**Why these?**
- Streamlit: Rapid UI development
- Plotly: Professional interactive charts
- Transformers + PEFT: Fine-tuned agricultural AI models
- gTTS: Voice accessibility for low-literacy farmers"

---

## **SLIDE 4: Application Entry Point (1 minute)**

**[Screen: app.py lines 1-50]**

"Now let's dive into `app.py`. Starting with imports and configuration:

```python
# Lines 1-25: Core imports
import streamlit as st
import pandas as pd
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from peft import PeftModel
import requests
import plotly.graph_objects as go
from gtts import gTTS

# Lines 27-35: Language mapping for 6 Indic languages
LANG_MAP = {
    "Hindi (हिंदी)": {"code": "hi", "tld": "co.in"},
    "Tamil (தமிழ்)": {"code": "ta", "tld": "co.in"},
    "Telugu (తెలుగు)": {"code": "te", "tld": "co.in"},
    # ... 3 more languages
}
```

**Key Point**: We support 6 Indian languages with proper regional TLDs for accurate pronunciation."

---

## **SLIDE 5: Page Configuration & Session State (1 minute)**

**[Screen: app.py lines 300-350]**

"Setting up the Streamlit application:

```python
# Line 305: Page configuration
st.set_page_config(
    page_title="🌾 Climate Resilience Chatbot",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Lines 380-395: Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'recommendation_history' not in st.session_state:
    st.session_state.recommendation_history = []
if 'advisory_cache' not in st.session_state:
    st.session_state.advisory_cache = {}
```

**Why session state?**
- Persist chat history across interactions
- Cache AI responses to avoid redundant API calls
- Store user preferences (language, location)"

---

## **SLIDE 6: Weather Data Integration (1.5 minutes)**

**[Screen: app.py lines 600-680 - get_weather_data function]**

"One of our core features is real-time weather integration using OpenMeteo API:

```python
# Line 615: Weather data fetching
def get_weather_data(location: str) -> Optional[Dict]:
    # Step 1: Geocode location
    geocoding_url = f"https://geocoding-api.open-meteo.com/v1/search?name={location}"
    geo_response = requests.get(geocoding_url, timeout=10)
    
    result = geo_data['results'][0]
    lat, lon = result['latitude'], result['longitude']
    
    # Step 2: Fetch weather
    weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
    weather_response = requests.get(weather_url, timeout=10)
    
    # Step 3: Return structured data
    return {
        "temperature": current['temperature'],
        "humidity": hourly['relativehumidity_2m'][0],
        "rainfall": hourly['precipitation'][0],
        "wind_speed": current['windspeed'],
        "pressure": hourly['surface_pressure'][0]
    }
```

**Why OpenMeteo?**
- Free, unlimited API
- No authentication required
- Reliable global coverage"

---

## **SLIDE 7: Satellite Data Integration (1.5 minutes)**

**[Screen: app.py lines 700-800 - get_keyless_agri_metrics function]**

"We fetch real-time satellite data for vegetation health:

```python
# Line 710: Satellite metrics (NDVI, Soil Moisture, ET0)
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_keyless_agri_metrics(lat: float, lon: float) -> Dict:
    # Fetch soil + agri data from Open-Meteo
    agri_url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}"
        f"&hourly=soil_temperature_0cm,soil_moisture_0_to_1cm"
        f"&daily=et0_fao_evapotranspiration,precipitation_sum"
    )
    
    # Compute NDVI-equivalent Vegetation Health Index (VHI)
    et0_norm = min(1.0, max(0.0, (et0 - 0.5) / 6.0))
    sm_norm = min(1.0, max(0.0, (sm - 3.0) / 40.0))
    rad_norm = min(1.0, max(0.0, (rad - 3.0) / 25.0))
    
    # Weighted VHI formula
    vhi = (0.45 * sm_norm) + (0.35 * et0_norm) + (0.20 * rad_norm)
    
    return {"ndvi": vhi, "soil_moisture": sm, "et0": et0}
```

**Innovation**: We compute a Vegetation Health Index when direct NDVI is unavailable."

---

## **SLIDE 8: Heuristic Scoring Engine (2 minutes)**

**[Screen: app.py lines 1200-1350 - calculate_heuristic_score function]**

"Instead of black-box ML models, we use transparent agricultural science:

```python
# Line 1210: Crop suitability scoring
def calculate_heuristic_score(nitrogen, phosphorus, potassium, ph, rainfall, temperature, crop):
    score = 100.0  # Start with perfect score
    
    # 1. NPK Balance Score (30 points)
    if crop.lower() in ['rice', 'wheat', 'maize']:
        if nitrogen < 60:
            score -= 10  # High N requirement crops
    
    # 2. pH Score (20 points)
    ideal_ph = 6.5
    if crop.lower() == 'rice':
        ideal_ph = 6.0
    elif crop.lower() == 'potato':
        ideal_ph = 5.5
    
    ph_deviation = abs(ph - ideal_ph)
    if ph_deviation > 2.0:
        score -= 15
    
    # 3. Rainfall Score (25 points)
    if crop.lower() in ['rice', 'sugarcane']:
        if rainfall < 600:
            score -= 15  # Insufficient water
    
    # 4. Temperature Score (25 points)
    if crop.lower() in ['rice', 'cotton']:
        if temperature < 20:
            score -= 15  # Too cold
    
    return max(0, min(100, score))
```

**Why heuristics?**
- Explainable to farmers
- Based on proven agricultural science
- No training data required"

---

## **SLIDE 9: AI Ensemble Architecture (1.5 minutes)**

**[Screen: app.py lines 1800-1900 - AI response generation]**

"We use a 3-model ensemble for robust recommendations:

```python
# Line 1820: Ensemble AI pipeline
ensemble_responses = {}

# Model 1: T5-PEFT (Agricultural Specialist)
if st.session_state.t5_peft_model:
    t5_resp = generate_with_t5_peft(prompt, model, tokenizer)
    ensemble_responses["T5-PEFT"] = t5_resp

# Model 2: Climate-LoRA (Climate Adaptation)
if st.session_state.lora_model:
    lora_resp = generate_with_lora(prompt, model, tokenizer)
    ensemble_responses["Climate-LoRA"] = lora_resp

# Model 3: Groq Llama-3.3-70B (Synthesis)
groq_resp = get_groq_recommendation(
    prompt=final_prompt,
    api_key=st.session_state.groq_api_key
)

# Combine all responses
final_response = groq_resp  # Groq synthesizes local model insights
```

**Architecture Benefits**:
- T5-PEFT: Crop-specific recommendations
- LoRA: Climate risk analysis
- Groq: Fact-grounded synthesis"

---

## **SLIDE 10: Frontend - Tab Structure (1 minute)**

**[Screen: app.py lines 2100-2150 - Tab creation]**

"The UI is organized into 6 tabs:

```python
# Line 2110: Tab structure
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "💬 Chatbot & ML Predictions",    # Main interaction
    "🌍 Location Analysis",            # District-level insights
    "📊 Visualizations",               # Charts & graphs
    "🌡️ Weather Forecast",            # 7-day forecast
    "🌾 Crop Recommendations",         # Crop suitability
    "📜 History & Export"              # Download reports
])
```

**User Flow**:
1. User selects location (State → District)
2. System auto-fetches weather + soil data
3. User asks question
4. AI generates response with visualizations"

---

## **SLIDE 11: Frontend - Smart Context Dashboard (1 minute)**

**[Screen: app.py lines 950-1050 - Context detection]**

"We automatically detect and display farm context:

```python
# Line 960: Automatic parameter detection
with st.spinner("🔍 Detecting local environment..."):
    # 1. Get coordinates from CSV
    location_data_df = advisory_df[
        (advisory_df['State'] == state) & 
        (advisory_df['District'] == district)
    ]
    
    # 2. Fetch live weather
    weather_data = get_weather_data_by_coords(lat, lon, location)
    
    # 3. Get satellite metrics
    agri_metrics = get_keyless_agri_metrics(lat, lon)
    
    # 4. Calculate optimized parameters
    rainfall = weather_data.get('rainfall', historical_avg)
    temperature = weather_data.get('temperature', historical_avg)

# Display metrics
col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("🌧️ Rainfall", f"{rainfall:.0f}mm")
col2.metric("🌡️ Temp", f"{temperature:.1f}°C")
col3.metric("🧪 Nitrogen", f"{nitrogen:.0f}")
# ... more metrics
```

**User Experience**: Farmers don't need to manually enter data - it's auto-populated!"

---

## **SLIDE 12: Multilingual Translation (1 minute)**

**[Screen: app.py lines 1450-1520 - translate_text function]**

"We provide professional agricultural translation:

```python
# Line 1460: Translation with Groq
def translate_text(text: str, target_lang: str, api_key: str) -> Dict:
    lang_info = LANG_MAP.get(target_lang)
    
    # Custom tone for Hindi (Hinglish)
    tone_instruction = "Use HINGLISH (mixture of Hindi and English)"
    
    prompt = f'''
    TASK: Translate to {lang_name}
    STYLE: {tone_instruction}
    
    OUTPUT FORMAT:
    [DETAILED] - Full translation
    [VOICE_SUMMARY] - Concise 120-word summary for voice
    
    TEXT: {text}
    '''
    
    # Call Groq API
    response = groq_client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile"
    )
    
    return {"detailed": detailed_text, "summary": voice_summary}
```

**Key Feature**: Separate detailed and voice versions for accessibility."

---

## **SLIDE 13: Voice Generation (45 seconds)**

**[Screen: app.py lines 1530-1560 - speak_text function]**

"Voice output for low-literacy farmers:

```python
# Line 1535: Text-to-speech with gTTS
def speak_text(text: str, lang_name: str):
    lang_info = LANG_MAP.get(lang_name)
    
    # Clean text (remove markdown)
    clean_text = re.sub(r'[#\\*]', '', text)
    
    # Generate audio with Indian accent
    tts = gTTS(
        text=clean_text, 
        lang=lang_info["code"],
        tld=lang_info["tld"],  # co.in for Indian accent
        slow=False
    )
    
    # Stream to Streamlit
    audio_fp = io.BytesIO()
    tts.write_to_fp(audio_fp)
    st.audio(audio_fp, format='audio/mp3')
```

**Impact**: Farmers can listen to recommendations while working in fields."

---

## **SLIDE 14: Data Visualization (1 minute)**

**[Screen: app.py lines 2400-2550 - render_domain_visuals function]**

"Dynamic charts based on query type:

```python
# Line 2420: Domain-specific visualizations
def render_domain_visuals(domain_visuals, crop_matches_df, agri_metrics):
    for chart_key in domain_visuals:
        
        if chart_key == "npk_bar":
            # Soil nutrient bar chart
            fig = go.Figure(go.Bar(
                x=["Nitrogen", "Phosphorus", "Potassium"],
                y=[nitrogen, phosphorus, potassium],
                marker_color=["#4CAF50", "#2196F3", "#FF9800"]
            ))
            st.plotly_chart(fig)
        
        elif chart_key == "ndvi_trend":
            # 10-year vegetation health trend
            fig = go.Figure(go.Scatter(
                x=years, y=ndvi_values,
                mode="lines+markers"
            ))
            st.plotly_chart(fig)
        
        elif chart_key == "msp_trend":
            # Market price trend
            fig = go.Figure(go.Scatter(
                x=years, y=msp_values,
                fill="tozeroy"
            ))
            st.plotly_chart(fig)
```

**Smart Feature**: Charts adapt to the type of question asked!"

---

## **SLIDE 15: Export & History (45 seconds)**

**[Screen: app.py lines 2700-2800 - Tab 6 export functionality]**

"Farmers can export their consultation history:

```python
# Line 2720: Export as JSON
json_data = json.dumps(st.session_state.recommendation_history, indent=2)
st.download_button(
    label="📥 Download JSON",
    data=json_data,
    file_name=f"recommendations_{datetime.now()}.json"
)

# Line 2740: Export as CSV
df_export = pd.DataFrame([{
    'Timestamp': rec['timestamp'],
    'Location': rec['location'],
    'Question': rec['question'],
    'Recommendation': rec['recommendation']
} for rec in st.session_state.recommendation_history])

st.download_button(
    label="📥 Download CSV",
    data=df_export.to_csv(index=False)
)
```

**Use Case**: Farmers can share reports with agricultural extension officers."

---

## **SLIDE 16: Live Demo (30 seconds)**

**[Screen: Run the application]**

"Let me show you a quick demo:

1. **[Select Location]**: Tamil Nadu → Ariyalur
2. **[Auto-detection]**: System fetches weather, soil data, satellite NDVI
3. **[Ask Question]**: 'What is the best crop for my soil?'
4. **[AI Response]**: 
   - Crop Suitability Score: 85% (Good)
   - Climate Risk: 25% (Low)
   - Detailed recommendation with NPK analysis
5. **[Translate]**: Switch to Hindi (हिंदी)
6. **[Voice]**: Play audio recommendation
7. **[Export]**: Download as PDF"

---

## **SLIDE 17: Key Innovations (30 seconds)**

**[Screen: Show README.md highlights]**

"What makes AgriGuard AI unique:

1. **Ensemble AI**: 3 models working together (T5-PEFT + LoRA + Groq)
2. **Heuristic Scoring**: Transparent, explainable predictions
3. **10-Year Data**: 6,690 expert advisories (2015-2024)
4. **Real-Time Integration**: Live weather + satellite NDVI
5. **Multilingual**: 6 Indic languages with voice output
6. **Domain-Aware**: Specialized prompts for 15+ query types
7. **Offline Capable**: Can run with local Ollama models"

---

## **SLIDE 18: Technical Achievements (30 seconds)**

"Technical highlights:

- **2,800+ lines** of production-ready Python code
- **6 interactive tabs** with responsive UI
- **15+ Plotly visualizations** (rainfall trends, NPK radar, MSP charts)
- **Caching strategy**: 1-hour TTL for API calls
- **Error handling**: Graceful fallbacks for all external APIs
- **Session persistence**: Chat history, recommendations, preferences
- **Export formats**: JSON, CSV for data portability"

---

## **SLIDE 19: Impact & Future Scope (30 seconds)**

"Real-world impact:

**Current Capabilities**:
- Serves 800+ districts across 36 Indian states
- Supports 6 major Indian languages
- Provides evidence-based recommendations

**Future Enhancements**:
1. Mobile app (React Native)
2. SMS-based advisory for feature phones
3. Crop disease detection (image upload)
4. Market linkage recommendations
5. Federated learning for privacy-preserving updates"

---

## **SLIDE 20: Conclusion & Q&A (30 seconds)**

"To summarize:

**AgriGuard AI** is a comprehensive climate-smart advisory system that:
- Combines AI, real-time data, and 10 years of agricultural research
- Provides actionable, explainable recommendations
- Accessible in 6 Indian languages with voice support
- Helps farmers adapt to climate change

**Thank you! Questions?**

---

## **BONUS: Code Walkthrough Checklist**

When sharing screen, highlight these specific lines:

### **Core Functions to Explain**:
1. **Lines 600-680**: `get_weather_data()` - Weather API integration
2. **Lines 710-850**: `get_keyless_agri_metrics()` - Satellite data
3. **Lines 1210-1350**: `calculate_heuristic_score()` - Scoring engine
4. **Lines 1460-1520**: `translate_text()` - Multilingual support
5. **Lines 1820-1900**: Ensemble AI pipeline
6. **Lines 2420-2650**: `render_domain_visuals()` - Dynamic charts

### **UI Components to Show**:
1. **Lines 950-1050**: Smart context dashboard
2. **Lines 2110-2150**: Tab structure
3. **Lines 2200-2400**: Chat interface
4. **Lines 2700-2800**: Export functionality

### **Key Variables to Point Out**:
- `st.session_state.chat_history` - Conversation persistence
- `st.session_state.advisory_cache` - Response caching
- `LANG_MAP` - Language configuration
- `advisory_df` - 10-year dataset (6,690 records)

---

## **PRESENTATION TIPS**

1. **Start with live demo** (30 sec) to grab attention
2. **Use VS Code split view**: Code on left, running app on right
3. **Highlight line numbers** when explaining functions
4. **Show actual data**: Open CSV files to prove 10-year dataset
5. **Demonstrate multilingual**: Switch between English and Hindi
6. **Play voice output**: Show accessibility feature
7. **Export a report**: Download JSON to show data portability
8. **End with impact**: Show how it helps real farmers

**Time Management**:
- Introduction: 30s
- Architecture: 2 min
- Core Code: 4 min
- Frontend: 2 min
- Demo: 1 min
- Conclusion: 30s
- **Total: 10 minutes**

Good luck with your presentation! 🌾
