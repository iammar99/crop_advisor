# app.py
import streamlit as st
import pandas as pd
import numpy as np
import requests
from streamlit_folium import st_folium
import folium
import joblib

st.set_page_config(page_title="Smart Crop Advisor", page_icon="🌱", layout="wide")

# 1. LOAD THE PRE-TRAINED BINARY FILE INSTANTLY
@st.cache_resource
def load_saved_model():
    # Loads the brain in less than 0.1 seconds, no matter how huge the original dataset was
    return joblib.load('naive_bayes_crop.pkl')

best_nb_model = load_saved_model()

# 2. WEATHER API INTEGRATION (Open-Meteo)
def fetch_60day_weather_averages(lat, lon):
    url = f"https://climate-api.open-meteo.com/v1/climate?latitude={lat}&longitude={lon}&daily=temperature_2m_mean,relative_humidity_2m_mean,precipitation_sum&models=EC_Earth3P_HR&start_date=2026-06-01&end_date=2026-07-30&timezone=auto"
    try:
        response = requests.get(url).json()
        if "daily" in response:
            return (
                round(np.nanmean(response["daily"]["temperature_2m_mean"]), 2),
                round(np.nanmean(response["daily"]["relative_humidity_2m_mean"]), 2),
                round(np.nansum(response["daily"]["precipitation_sum"]), 2)
            )
    except:
        return None, None, None
    return None, None, None

# 3. STREAMLIT LAYOUT INTERFACE
st.title("🌱 AI Agricultural Crop Recommendation Engine")

col_map, col_inputs = st.columns([3, 2])

with col_map:
    st.subheader("1. Locate Your Farm")
    m = folium.Map(location=[24.8607, 67.0011], zoom_start=5)
    m.add_child(folium.LatLngPopup())
    map_data = st_folium(m, height=450, width="100%")
    
    lat, lon = None, None
    if map_data and map_data.get("last_clicked"):
        lat = map_data["last_clicked"]["lat"]
        lon = map_data["last_clicked"]["lng"]
        st.success(f"Coordinates Locked: Latitude {lat:.4f}, Longitude {lon:.4f}")

with col_inputs:
    st.subheader("2. Soil Metrics")
    n_input = st.number_input("Nitrogen (N)", min_value=0, max_value=200, value=50)
    p_input = st.number_input("Phosphorus (P)", min_value=0, max_value=200, value=50)
    k_input = st.number_input("Potassium (K)", min_value=0, max_value=200, value=50)
    ph_input = st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=6.5, step=0.1)

# 3. FINAL RECOMMENDATIONS (UPGRADED WITH BEAUTIFUL CARD & IMAGES)
st.markdown("---")
st.subheader("3. Final Recommendations")

# Dictionary mapping ML string outputs to relevant photography URLs
# Place this right above your main layout block in app.py
CROP_IMAGES = {
    "rice": ["static/rice1.png", "static/rice2.png"],
    "maize": ["static/maize1.png", "static/maize2.png"],

    "chickpea": ["static/chickpea1.png", "static/chickpea2.png"],
    "kidneybeans": ["static/kidneybeans1.png", "static/kidneybeans2.png"],
    "pigeonpeas": ["static/pigeonpeas1.png", "static/pigeonpeas2.png"],
    "mothbeans": ["static/mothbeans1.png", "static/mothbeans2.png"],
    "mungbean": ["static/mungbean1.png", "static/mungbean2.png"],
    "blackgram": ["static/blackgram1.png", "static/blackgram2.png"],
    "lentil": ["static/lentil1.png", "static/lentil2.png"],

    "pomegranate": ["static/pomegranate1.png", "static/pomegranate2.png"],
    "banana": ["static/banana1.png", "static/banana2.png"],
    "mango": ["static/mango1.png", "static/mango2.png"],
    "grapes": ["static/grapes1.png", "static/grapes2.png"],
    "watermelon": ["static/watermelon1.png", "static/watermelon2.png"],
    "muskmelon": ["static/muskmelon1.png", "static/muskmelon2.png"],
    "apple": ["static/apple1.png", "static/apple2.png"],
    "orange": ["static/orange1.png", "static/orange2.png"],
    "papaya": ["static/papaya1.png", "static/papaya2.png"],
    "coconut": ["static/coconut1.png", "static/coconut2.png"],

    "cotton": ["static/cotton1.png", "static/cotton2.png"],
    "jute": ["static/jute1.png", "static/jute2.png"],
    "coffee": ["static/coffee1.png", "static/coffee2.png"]
}

if lat is not None and lon is not None:
    with st.spinner("Analyzing climate data for your coordinates..."):
        temp, humidity, rainfall = fetch_60day_weather_averages(lat, lon)
        
    if temp is not None:
        # Format features and predict using the loaded model file
        input_features = [[n_input, p_input, k_input, temp, humidity, ph_input, rainfall]]
        predicted_crop = best_nb_model.predict(input_features)[0]
        max_prob = np.max(best_nb_model.predict_proba(input_features)[0]) * 100
        
        # Clean up the prediction text for matching
        crop_key = str(predicted_crop).strip().lower()
        
        # Split layout: Card on left side, Image on right side
        card_col, img_col = st.columns([3, 2], gap="large")
        
        with card_col:
            # Custom styled HTML Material Design Card 
            card_html = f"""
            <div style="
                background-color: #f8f9fa;
                padding: 25px;
                border-radius: 15px;
                border-left: 8px solid #2e7d32;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
                margin-top: 10px;
            ">
                <span style="color: #2e7d32; font-size: 14px; font-weight: bold; text-transform: uppercase; letter-spacing: 1px;">
                    Optimal Match Found
                </span>
                <h1 style="color: #1b5e20; margin: 5px 0 15px 0; font-size: 38px; font-weight: 800;">
                    {predicted_crop.upper()}
                </h1>
                <p style="color: #495057; font-size: 16px; margin-bottom: 20px;">
                    Based on your localized soil properties and the 60-day climate forecast, your land shows exceptional conditions for cultivating this crop.
                </p>
                <div style="display: flex; gap: 20px;">
                    <div style="background: white; padding: 10px 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.02);">
                        <small style="color: #6c757d; display: block;">Confidence Score</small>
                        <strong style="color: #2e7d32; font-size: 20px;">{max_prob:.1f}%</strong>
                    </div>
                    <div style="background: white; padding: 10px 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.02);">
                        <small style="color: #6c757d; display: block;">Climate Zone</small>
                        <strong style="color: #495057; font-size: 18px;">{temp}°C Avg</strong>
                    </div>
                </div>
            </div>
            """
            st.markdown(card_html, unsafe_allow_html=True)
            
        with img_col:
          # Get images safely
          images = CROP_IMAGES.get(crop_key, [])
      
          if images:
              # Initialize index in session state
              key_index = f"{crop_key}_index"
              if key_index not in st.session_state:
                  st.session_state[key_index] = 0
      
              # Navigation buttons
              col1, col2, col3 = st.columns([1, 2, 1])
      
              with col1:
                  if st.button("⬅️ Prev", key=f"prev_{crop_key}"):
                      st.session_state[key_index] = (st.session_state[key_index] - 1) % len(images)
      
              with col3:
                  if st.button("Next ➡️", key=f"next_{crop_key}"):
                      st.session_state[key_index] = (st.session_state[key_index] + 1) % len(images)
      
              # Current image
              current_img = images[st.session_state[key_index]]
      
              st.image(
                  current_img,
                  use_container_width=True,
                  caption=f"{crop_key.title()} image {st.session_state[key_index] + 1} of {len(images)}"
              )
      
          else:
              st.info(
                  f"💡 Prediction ready! Add images for '{crop_key}' inside CROP_IMAGES to enable gallery view."
              )
else:
    st.warning("Please click on the map to pin your location.")
