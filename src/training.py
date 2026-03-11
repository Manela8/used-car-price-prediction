"""
Training module for Used Car Price Prediction.

- loads cleaned_data.csv
- splits into train/test
- builds preprocessor (imputer+scaler / imputer+ohe)
- trains multiple REGRESSION models with GridSearchCV
- saves per-model best and overall best model
- saves feature column list to models/feature_columns.json
"""

from pathlib import Path
from typing import Dict, List, Tuple
import json

import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from src.config import DATA_FILE, MODEL_DIR, BEST_MODEL_PATH, FEATURES_PATH, CV_FOLDS, SCORING, N_JOBS
from src.preprocessing import data_load, split_data, build_preprocessor


# ──────────────────────────────────────────────────────────────────
# 1. Model Definitions
# ──────────────────────────────────────────────────────────────────

def get_models_and_params() -> List[Tuple[str, object, Dict[str, List]]]:
    """
    Define all regression models and their hyperparameter grids.

    Returns
    -------
    List of (name, estimator, param_grid) tuples.
    """
    models_and_params = []

    # ── Linear Regression (baseline — no hyperparams to tune)
    lr = LinearRegression()
    lr_params = {}
    models_and_params.append(("linear_regression", lr, lr_params))

    # ── Ridge Regression (regularized linear)
    ridge = Ridge()
    ridge_params = {"clf__alpha": [0.1, 1.0, 10.0, 100.0]}
    models_and_params.append(("ridge_regression", ridge, ridge_params))

    # ── Decision Tree Regressor ✅ required by project guidelines
    dtr = DecisionTreeRegressor(random_state=42)
    dtr_params = {
        "clf__max_depth"        : [None, 5, 10, 15],
        "clf__min_samples_split": [2, 5, 10],
        "clf__criterion"        : ["squared_error", "absolute_error"],
    }
    models_and_params.append(("decision_tree", dtr, dtr_params))

    # ── Random Forest Regressor ✅ required by project guidelines
    rfr = RandomForestRegressor(random_state=42)
    rfr_params = {
        "clf__n_estimators": [100, 200],
        "clf__max_depth"   : [None, 5, 10],
        "clf__max_features": ["sqrt", "log2"],
    }
    models_and_params.append(("random_forest", rfr, rfr_params))

    # ── Gradient Boosting Regressor (bonus — strong performer)
    gbr = GradientBoostingRegressor(random_state=42)
    gbr_params = {
        "clf__n_estimators" : [100, 200],
        "clf__learning_rate": [0.05, 0.1],
        "clf__max_depth"    : [3, 5],
    }
    models_and_params.append(("gradient_boosting", gbr, gbr_params))

    return models_and_params


# ──────────────────────────────────────────────────────────────────
# 2. Evaluation Helper
# ──────────────────────────────────────────────────────────────────

def evaluate_regression(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute regression metrics as required by project guidelines.

    Metrics
    -------
    - MAE  : Mean Absolute Error
    - MSE  : Mean Squared Error
    - RMSE : Root Mean Squared Error
    - R2   : R² Score

    NOTE: y_true and y_pred are in LOG scale.
          Metrics are computed in log scale for model comparison.

    Parameters
    ----------
    y_true : pd.Series   — actual log(Price) values
    y_pred : np.ndarray  — predicted log(Price) values

    Returns
    -------
    dict of metric name -> value
    """
    return {
        "MAE" : round(metrics.mean_absolute_error(y_true, y_pred), 4),
        "MSE" : round(metrics.mean_squared_error(y_true, y_pred), 4),
        "RMSE": round(np.sqrt(metrics.mean_squared_error(y_true, y_pred)), 4),
        "R2"  : round(metrics.r2_score(y_true, y_pred), 4),
    }


# ──────────────────────────────────────────────────────────────────
# 3. Main Training Function
# ──────────────────────────────────────────────────────────────────

def train_and_select_model() -> pd.DataFrame:
    """
    Full training pipeline:
      1. Load data
      2. Log-transform Price (fixes skewness & negative R2 issue)
      3. Train/test split
      4. Build preprocessor
      5. Train each model with GridSearchCV
      6. Evaluate with MAE, MSE, RMSE, R2
      7. Save best model overall

    Returns
    -------
    pd.DataFrame
        Summary of all models with their metrics.
    """
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load data
    print("Loading pre-cleaned data ->", DATA_FILE)
    df = data_load(DATA_FILE)
    print("Data shape  :", df.shape)
    print("Columns     :", df.columns.tolist())

    # ── ✅ Log-transform Price before splitting
    # Fixes skewness and resolves negative R2 scores
    print(f"\nPrice before log → min: {df['Price'].min():.2f} | max: {df['Price'].max():.2f} | mean: {df['Price'].mean():.2f}")
    df['Price'] = np.log1p(df['Price'])
    print(f"Price after  log → min: {df['Price'].min():.4f} | max: {df['Price'].max():.4f} | mean: {df['Price'].mean():.4f}\n")

    # ── Split
    X_train, X_test, y_train, y_test = split_data(df)

    # ── Save feature columns for inference alignment
    feature_cols = X_train.columns.tolist()
    with open(FEATURES_PATH, "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, ensure_ascii=False, indent=2)
    print(f"Saved feature list ({len(feature_cols)} cols) -> {FEATURES_PATH}")
    print(f"Train shape : {X_train.shape} | Test shape: {X_test.shape}")

    # ── Build preprocessor
    preprocessor = build_preprocessor(X_train)
    print("Preprocessor built.\n")

    # ── Train all models
    models_and_params = get_models_and_params()

    best_overall  = None
    best_r2_score = -np.inf
    best_name     = None
    results       = []

    for name, estimator, param_grid in models_and_params:
        print(f"=== Training: {name} ===")

        pipeline = Pipeline([
            ("preprocess", preprocessor),
            ("clf",        estimator),
        ])

        # GridSearchCV — skip if no params (e.g. LinearRegression)
        if param_grid:
            grid = GridSearchCV(
                estimator  = pipeline,
                param_grid = param_grid,
                cv         = CV_FOLDS,
                scoring    = SCORING,
                n_jobs     = N_JOBS,
                verbose    = 1,
            )
            grid.fit(X_train, y_train)
            best     = grid.best_estimator_
            cv_score = grid.best_score_
            print(f"Best params : {grid.best_params_}")
        else:
            # No hyperparams — fit directly
            pipeline.fit(X_train, y_train)
            best     = pipeline
            cv_score = None
            print("No hyperparameters to tune — fitted directly.")

        # ── Evaluate on test set (log scale)
        y_pred       = best.predict(X_test)
        eval_metrics = evaluate_regression(y_test, y_pred)

        print(f"CV Score ({SCORING}) : {cv_score}")
        print(f"MAE  : {eval_metrics['MAE']}")
        print(f"MSE  : {eval_metrics['MSE']}")
        print(f"RMSE : {eval_metrics['RMSE']}")
        print(f"R2   : {eval_metrics['R2']}\n")

        # ── Save individual model
        model_path = MODEL_DIR / f"{name}_best_model.joblib"
        joblib.dump(best, model_path)
        print(f"Saved -> {model_path}\n")

        results.append({
            "model"   : name,
            "cv_score": cv_score,
            **eval_metrics,
        })

        # ── Track best model by R2
        if eval_metrics["R2"] > best_r2_score:
            best_r2_score = eval_metrics["R2"]
            best_overall  = best
            best_name     = name

    # ── Save overall best model
    if best_overall is not None:
        joblib.dump(best_overall, BEST_MODEL_PATH)
        print(f"Overall best model : {best_name} (R2: {best_r2_score:.4f}) -> {BEST_MODEL_PATH}")
    else:
        print("No model trained successfully.")

    return pd.DataFrame(results)


# ──────────────────────────────────────────────────────────────────
# 4. Entry Point
# ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df_results = train_and_select_model()
    print("\n── Training Summary ──")
    print(df_results.to_string(index=False))