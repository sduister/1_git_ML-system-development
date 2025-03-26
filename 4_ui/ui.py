import streamlit as st
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Streamlit UI Title
st.title("Ship Resistance Predictor")

# Fetch feature boundaries from FastAPI
feature_bounds = requests.get("http://127.0.0.1:8000/feature_bounds").json()

# Fetch actual vs. predicted results
results_data = requests.get("http://127.0.0.1:8000/results").json()
results_df = pd.DataFrame(results_data)

# 🎯 Sidebar Inputs
st.sidebar.markdown("## Input Ship Design Features")
user_inputs = {}
for feature, bounds in feature_bounds.items():
    user_inputs[feature] = st.sidebar.slider(
        f"{feature} ({bounds['min']:.2f} to {bounds['max']:.2f})",
        float(bounds["min"]), float(bounds["max"]),
        float((bounds["min"] + bounds["max"]) / 2)
    )

# 🚀 Resistance curve speed range
speed_range = st.sidebar.slider("Ship Speed Range (knots)", 5, 30, (5, 25))
speeds = np.linspace(speed_range[0], speed_range[1], num=20)

# 📊 **Plot 1: Actual vs. Predicted Resistance**
st.markdown("### Actual vs. Predicted Resistance")
fig, ax = plt.subplots()
ax.scatter(results_df["Actual"], results_df["Predicted"], alpha=0.6, label="Predictions")
ax.plot([results_df["Actual"].min(), results_df["Actual"].max()],
        [results_df["Actual"].min(), results_df["Actual"].max()],
        color='black', label="Ideal line")
ax.set_xlabel("Actual $R_{tot}$ (kN)")
ax.set_ylabel("Predicted $R_{tot}$ (kN)")
ax.legend()
ax.grid(True)
st.pyplot(fig)

# 📉 **Plot 2: Resistance Curve**
if st.sidebar.button("Generate Resistance Curve"):
    resistance_values = []
    for speed in speeds:
        payload = {
            "Vs_kn": speed,
            "Lwl_m": user_inputs["Lwl (m)"],
            "Bmld_m": user_inputs["Bmld (m)"],
            "T_m": user_inputs["T (m)"]
        }
        response = requests.post("http://127.0.0.1:8000/predict", json=payload)
        if response.status_code == 200:
            resistance_values.append(response.json()["predicted_resistance"][0])
        else:
            st.error(f"Error in API call: {response.status_code}")
            break

    # Plot the resistance curve
    fig, ax = plt.subplots()
    ax.plot(speeds, resistance_values, marker='o', linestyle='-', label="Predicted Resistance")
    ax.set_xlabel("Ship Speed $V_s$ (knots)")
    ax.set_ylabel("Calm-water Resistance $R_{tot}$ (kN)")
    ax.set_title("Resistance Curve")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

# terminal 1: uvicorn app:app --reload
# terminal 2: streamlit run ui.py