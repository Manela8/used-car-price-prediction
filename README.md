# 🚗 Used Car Price Prediction

A machine learning project to predict used car prices based on features like company, model, fuel type, kilometers driven, and more. The project includes a full pipeline from data preprocessing to a Streamlit web application.

---

## 📌 Project Overview

The used car market is highly dynamic, and pricing a car accurately is challenging. This project builds a robust regression model to predict used car prices, helping:

- **Buyers** assess whether a car is fairly priced
- **Sellers** set competitive prices
- **Dealers** optimize inventory and pricing strategies

### Key Features
- End-to-end ML pipeline: EDA → Preprocessing → Training → Deployment
- Comparison of 5 regression algorithms
- Interactive Streamlit web app for single and batch predictions
- Log transformation on Price to handle skewness

---

## 📁 Project Structure

```
used-car-price-prediction/
├── src/
│   ├── __init__.py
│   ├── config.py            # Paths and hyperparameter settings
│   ├── preprocessing.py     # Data loading, splitting, and feature engineering
│   ├── training.py          # Model training and evaluation
│   └── deployment.py        # Prediction utilities
├── data/
│   └── raw_data.csv         # Original dataset
├── models/                  # Saved models (auto-generated after training)
├── notebooks/
│   └── eda.ipynb            # Exploratory Data Analysis notebook
├── app.py                   # Streamlit web application
├── run.bat                  # One-click pipeline runner (Windows)
├── requirements.txt         # Python dependencies
└── README.md
```

---

## 📊 Model Performance

All models were trained on log-transformed Price values and evaluated on a 20% test set.

| Model | R² | MAE | MSE | RMSE |
|---|---|---|---|---|
| **Linear Regression** | **0.8371** | 0.1545 | 0.0485 | 0.2202 |
| Ridge Regression | 0.8295 | 0.1606 | 0.0508 | 0.2253 |
| Gradient Boosting | 0.7922 | 0.1893 | 0.0619 | 0.2487 |
| Random Forest | 0.7663 | 0.1948 | 0.0696 | 0.2638 |
| Decision Tree | 0.6681 | 0.2320 | 0.0988 | 0.3144 |

> ✅ **Best Model: Linear Regression** with R² = 0.8371
> Metrics are computed in log scale. Predictions are converted back to original ₹ scale using `expm1()`.

---


## ▶️ How to Run

### Option 1 — One click (Windows)
```batch
run.bat
```

### Option 2 — Step by step
```bash
# Step 1: Verify preprocessing
python -m src.preprocessing

# Step 2: Train all models (saves best model)
python -m src.training

# Step 3: Test prediction
python -m src.deployment

# Step 4: Launch Streamlit app
streamlit run app.py
```

---

## 🌐 Streamlit App

The app supports two modes:

- **Single Prediction** — Enter car details manually and get an instant price estimate
- **Batch Prediction** — Upload a CSV file and download predictions for all cars

---

## 📦 Dependencies

```
pandas
numpy
scikit-learn
streamlit
joblib
```

Install all with:
```bash
pip install -r requirements.txt
```

## 📄 License

This project is for educational purposes.
