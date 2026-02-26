"""
=============================================================
app.py  — INTEGRATION PATCH for domain_prompts.py
=============================================================

HOW TO USE:
  1. Copy domain_prompts.py into the same folder as app.py
  2. Find the three sections marked PATCH A, PATCH B, PATCH C
     inside app.py and replace them with the code blocks below.
  3. That's it — no other changes needed.

PATCH A  — Import (top of file, after existing imports)
PATCH B  — Replace the Groq "Direct mode" and "Ensemble synthesis" block
           (~lines 2580–2796 in the original)
PATCH C  — Replace the score breakdown expander
           (~lines 2249–2340 in the original)
=============================================================
"""

# ══════════════════════════════════════════════════════════
# PATCH A  — add after "from gtts import gTTS" (line 25)
# ══════════════════════════════════════════════════════════

PATCH_A = """
# Domain-specific prompt engine
from domain_prompts import build_groq_payload
"""


# ══════════════════════════════════════════════════════════
# PATCH B  — replace the Groq call blocks
#            (the section that builds synthesis_prompt /
#             direct_prompt and calls client.chat.completions)
# ══════════════════════════════════════════════════════════
#
# In app.py, locate the comment:
#   "# 🎯 ENSEMBLE APPROACH: Use multiple models and merge responses with Groq"
# Replace everything from that comment down to (and including)
#   "ai_backend_used = 'Groq API (Direct — Cloud)'"
# with the code below.
# ══════════════════════════════════════════════════════════

PATCH_B = '''
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
            final_user_msg += "\\n\\n═══ LOCAL MODEL INSIGHTS (use as supporting context) ═══"
            for model_name, resp in ensemble_responses.items():
                final_user_msg += f"\\n\\n--- {model_name} ---\\n{resp[:600]}"

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
                    f"Groq [{detected_domain.replace(\'_\',\' \').title()}]"
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
'''


# ══════════════════════════════════════════════════════════
# PATCH C  — replace the Score Breakdown expander
#            (the `with st.expander("🔍 Score Breakdown...")` block
#             around lines 2249–2340 in original app.py)
# ══════════════════════════════════════════════════════════

PATCH_C = '''
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
'''


# ══════════════════════════════════════════════════════════
# PATCH D  — Domain-aware visual chart renderer
#   Add this function BEFORE the "Main Interface" section
#   (around line 1661 in original app.py).
#   Then call  render_domain_visuals(...)  inside the
#   "📊 Analyze Historical 10-Year Trends" expander,
#   AFTER the existing climate/NPK charts.
# ══════════════════════════════════════════════════════════

PATCH_D = '''
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
'''

# ══════════════════════════════════════════════════════════
# WHERE TO CALL render_domain_visuals() in app.py
# ══════════════════════════════════════════════════════════
#
# Find the expander titled "📊 Analyze Historical 10-Year Trends" 
# (around line 2869 in original).
# AFTER the existing plot_col1 / plot_col2 charts, add:
#
#     st.markdown("---")
#     st.markdown("#### 📊 Domain-Specific Analytics")
#     render_domain_visuals(
#         domain_visuals = st.session_state.get("domain_visuals", []),
#         crop_matches_df= crop_matches if "crop_matches" in locals() and not crop_matches.empty else None,
#         agri_metrics   = st.session_state.get("agri_metrics", {}),
#         weather_data   = weather_data,
#         soil_params    = {"N": nitrogen, "P": phosphorus, "K": potassium, "pH": ph},
#         district       = district,
#         state          = state_name if "state_name" in locals() else state,
#         crop           = crop,
#     )
#
# ══════════════════════════════════════════════════════════
