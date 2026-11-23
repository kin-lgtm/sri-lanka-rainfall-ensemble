import datetime
import os
import joblib
import pandas as pd
import streamlit as st

MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

st.set_page_config(page_title="Sri Lanka Rainfall Ensemble Dashboard", layout="wide")

st.title("🇱🇰 Sri Lanka Rainfall Prediction — Ensemble Learning Dashboard")
st.markdown(
    "Comparing **Bagging**, **Boosting**, and **Stacking** models trained on meteorological data across Sri Lankan cities."
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def resolve_file(filename):
    for folder in (BASE_DIR, os.getcwd(), "/content"):
        path = os.path.join(folder, filename)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"Missing required artifact: {filename}")


@st.cache_data
def load_metrics():
    return pd.read_csv(resolve_file("metrics.csv"))


@st.cache_resource(show_spinner="Loading trained models...")
def load_models():
    return joblib.load(resolve_file("trained_models.joblib"))


@st.cache_data
def load_cities():
    return joblib.load(resolve_file("cities.joblib"))


metrics_df = load_metrics()
models = load_models()
cities = load_cities()

st.subheader("1. Ensemble Model Comparison")
col1, col2 = st.columns([1.2, 1])

with col1:
    st.dataframe(metrics_df, width="stretch", hide_index=True)

with col2:
    st.bar_chart(metrics_df.set_index("Model")["R2 Score"])

best_model_name = metrics_df.loc[metrics_df["RMSE (mm)"].idxmin(), "Model"]
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
    use_date = st.checkbox("Use a specific date (overrides month)")
    day_of_year = month * 30 - 15
    if use_date:
        picked = st.date_input("Date", value=datetime.date(2024, month, 15))
        month = picked.month
        day_of_year = picked.timetuple().tm_yday
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
    "day_of_year": day_of_year,
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

    st.markdown("**Model predictions compared (mm)**")
    st.bar_chart(pd.Series(all_preds, name="Predicted rainfall (mm)"))

    st.divider()
    st.subheader(f"3. Seasonal Rainfall Curve — {city}")
    st.caption(
        "Predicted daily rainfall for each month (mid-month), using the weather "
        "values and elevation set above."
    )
    year_df = pd.concat(
        [input_df.assign(month=m, day_of_year=datetime.date(2024, m, 15).timetuple().tm_yday)
         for m in range(1, 13)],
        ignore_index=True,
    )
    curve = pd.DataFrame(
        {
            m_name: [max(0.0, float(v)) for v in m_obj.predict(year_df)]
            for m_name, m_obj in models.items()
        },
        index=pd.CategoricalIndex(MONTH_LABELS, categories=MONTH_LABELS, ordered=True),
    )
    st.line_chart(curve)
