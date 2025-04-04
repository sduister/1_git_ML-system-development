import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import joblib

# 📂 Define file path (Make sure this path is correct!)
file_path = r"C:\Users\sietse.duister\OneDrive - De Voogt Naval Architects\00_specialists group\1_projects\2_ML system development\1_git_ML system development\1_data\cleaned_CFD.xlsx"

# 🛠 Define target and features
target_column = "Rtot [kN]"
feature_columns = ["Vs [kn]", "Lwl [m]", "T [m]"]

# 📊 Load the Excel file
df = pd.read_excel(file_path)

# 🔍 Check if all required columns exist
missing_columns = [col for col in feature_columns + [target_column] if col not in df.columns]
if missing_columns:
    raise ValueError(f"Missing columns in Excel file: {missing_columns}")

# 🎯 Extract features (X) and target (y)
X = df[feature_columns]
y = df[target_column]

# 📏 Compute feature boundaries (min/max values)
feature_bounds = {col: {"min": X[col].min(), "max": X[col].max()} for col in feature_columns}

# 🤖 Train model
model = RandomForestRegressor()
model.fit(X, y)

# 🔍 Predictions (for actual vs. predicted plot)
y_pred = model.predict(X)
results_df = pd.DataFrame({"Actual": y, "Predicted": y_pred})

# 📊 Plot 1: Actual vs. Predicted Scatter Plot
plt.figure(figsize=(6, 6))
plt.scatter(y, y_pred, alpha=0.6, label="Predictions")
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='black', label="Ideal line")
plt.xlabel("Actual $R_{tot}$ (kN)")
plt.ylabel("Predicted $R_{tot}$ (kN)")
plt.legend()
plt.grid(True)
plt.title("Actual vs. Predicted Resistance")
plt.show()

# 📉 Plot 2: Resistance Curve for a Sample Design
sample_design = {
    "Vs [kn]": np.linspace(feature_bounds["Vs [kn]"]["min"], feature_bounds["Vs [kn]"]["max"], num=20),
    "Lwl [m]": [X["Lwl [m]"].median()] * 20,
    "T [m]": [X["T [m]"].median()] * 20,
}

# Convert to DataFrame
sample_df = pd.DataFrame(sample_design)
sample_resistance = model.predict(sample_df)

# Plot Resistance Curve
plt.figure(figsize=(7, 5))
plt.plot(sample_design["Vs [kn]"], sample_resistance, marker='o', linestyle='-', label="Predicted Resistance")
plt.xlabel("Ship Speed $V_s$ (knots)")
plt.ylabel("Calm-water Resistance $R_{tot}$ (kN)")
plt.title("Resistance Curve (Sample Design)")
plt.legend()
plt.grid(True)
plt.show()
