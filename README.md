# Machine Learning API System

This project is a modular machine learning pipeline that:
- Loads data from local Excel files
- Trains and saves an ML model
- Serves predictions via an API (FastAPI)
- Can be connected to Power BI or PowerApps for visualization

## 🗂️ Folder Structure
ml-api-system/
├── data/                  # Your Excel files (input data)
│   └── raw/               # Raw input data (Excel, CSV)
├── src/                   # Core Python logic
│   ├── preprocessing.py   # Data cleaning, feature engineering
│   ├── train.py           # Training the ML model
│   ├── predict.py         # Load model + make predictions
│   └── model/             # Folder to store trained model files (.pkl, etc.)
├── api/                   # API code (FastAPI or Flask)
│   └── main.py            # Main entry point for your API
├── ui/                    # UI integrations or exports for Power BI/PowerApps
├── notebooks/             # Jupyter notebooks for experimentation (optional)
├── .gitignore             # Ignore unnecessary files
├── requirements.txt       # List of dependencies (numpy, pandas, sklearn, etc.)
├── README.md              # Project description, setup, how to run
