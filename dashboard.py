import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px

# Load models and scaler
bte_model = pickle.load(open("bte_model.pkl", "rb"))
co2_model = pickle.load(open("co2_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.set_page_config(page_title="Biofuel Optimizer", layout="centered")
st.title("🔬 Biofuel Blend Optimizer")
st.markdown("Use machine learning + genetic algorithms to optimize blending ratios for **better energy efficiency** and **lower emissions**.")

# Sidebar inputs
st.sidebar.header("🧪 Input Blend Ratios")
diesel = st.sidebar.slider("Diesel (%)", 0.0, 100.0, 50.0, step=1.0)
biodiesel = st.sidebar.slider("Biodiesel (%)", 0.0, 100.0 - diesel, 30.0, step=1.0)
ethanol = st.sidebar.slider("Ethanol (%)", 0.0, 100.0 - diesel - biodiesel, 20.0, step=1.0)
rpm = st.sidebar.slider("Engine RPM", 1000, 3000, 2000, step=100)

# Validate total
total = diesel + biodiesel + ethanol
if total > 100:
    st.error("⚠️ Total blend ratio exceeds 100%. Please adjust the sliders.")
else:
    X = pd.DataFrame([[diesel, biodiesel, ethanol, rpm]], columns=['Diesel (%)', 'Biodiesel (%)', 'Ethanol (%)', 'RPM'])
    X_scaled = scaler.transform(X)
    bte_pred = bte_model.predict(X_scaled)[0]
    co2_pred = co2_model.predict(X_scaled)[0]

    st.subheader("📈 Predicted Performance")
    st.metric(label="Brake Thermal Efficiency (BTE)", value=f"{bte_pred:.2f} %")
    st.metric(label="CO₂ Emissions", value=f"{co2_pred:.2f} g/kWh")
