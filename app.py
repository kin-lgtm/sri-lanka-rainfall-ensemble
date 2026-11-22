import os
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Sri Lanka Rainfall Ensemble Dashboard", layout="wide")

st.title("🇱🇰 Sri Lanka Rainfall Prediction — Ensemble Learning Dashboard")
st.markdown(
    "Comparing **Bagging**, **Boosting**, and **Stacking** models trained on meteorological data across Sri Lankan cities."
)

def resolve_file(filename):
    for path in [filename, os.path.join("/content", filename)]:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"Missing required artifact: {filename}")

metrics_df = pd.read_csv(resolve_file("metrics.csv"))
models = joblib.load(resolve_file("trained_models.joblib"))
cities = joblib.load(resolve_file("cities.joblib"))

st.subheader("1. Ensemble Model Comparison")
col1, col2 = st.columns([1.2, 1])

with col1:
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

with col2:
    st.bar_chart(metrics_df.set_index("Model")["R2 Score"])

best_model_name = metrics_df.iloc[0]["Model"]
st.info(f"🏆 **Top Performing Ensemble (Lowest RMSE):** {best_model_name}")

st.divider()

st.subheader("2. Interactive Weather & Rainfall Simulator")

c1, c2, c3 = st.columns(3)
with c1:
    city = st.selectbox(
        "Sri Lankan City",
        cities,
        index=cities.index("Colombo") if "Colombo" in cities else 0,
    )
    month = st.slider("Month (1 = Jan, 12 = Dec)", 1, 12, 5)
    elevation = st.number_input("City Elevation (m)", min_value=0, max_value=2500, value=10)

with c2:
    temp_max = st.slider("Max Temperature (°C)", 15.0, 40.0, 31.0)
    temp_min = st.slider("Min Temperature (°C)", 10.0, 32.0, 25.0)
    wind = st.slider("Max Wind Speed (km/h)", 0.0, 60.0, 18.0)

with c3:
    radiation = st.slider("Shortwave Radiation (MJ/m²)", 5.0, 30.0, 17.5)
    et0 = st.slider("Evapotranspiration ET0 (mm)", 1.0, 8.0, 3.8)
    selected_model = st.selectbox("Select Ensemble Model", list(models.keys()))

input_df = pd.DataFrame([{
    "temperature_2m_max": temp_max,
    "temperature_2m_min": temp_min,
    "shortwave_radiation_sum": radiation,
    "windspeed_10m_max": wind,
    "et0_fao_evapotranspiration": et0,
    "elevation": elevation,
    "month": month,
    "day_of_year": month * 30 - 15,
    "city": city,
}])

if st.button("Predict Daily Rainfall", type="primary"):
    pred = max(0.0, float(models[selected_model].predict(input_df)[0]))
    st.success(f"**Predicted Rainfall in {city} ({selected_model}):** `{pred:.2f} mm`")

    all_preds = {
        m_name: round(max(0.0, float(m.predict(input_df)[0])), 2)
        for m_name, m in models.items()
    }
    st.write("**Comparison across all three ensemble methods (mm):**", all_preds)
