"""
Deployment utilities for Used Car Price Prediction.

- load_model                       : lazy-load the best trained pipeline
- _get_feature_columns_from_pipeline: infer feature list from pipeline or load from feature_columns.json
- predict_single                   : predict price for a single car (input as dict)
- predict_batch                    : predict prices for multiple cars (input as DataFrame)
"""

from typing import Any, Dict, List, Optional
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import BEST_MODEL_PATH, FEATURES_PATH, MODEL_DIR


# Module-level cache (avoid reloading model on every prediction)

_model = None
_feature_cols_cache: Optional[List[str]] = None



# 1. Load Model


def load_model():
    """
    Lazy-load the best trained pipeline from disk.
    Caches model in memory so it is only loaded once.

    Returns
    -------
    sklearn Pipeline
        The best trained regression pipeline.
    """
    global _model
    if _model is None:
        _model = joblib.load(BEST_MODEL_PATH)
    return _model



# 2. Feature Column Resolution


def _get_feature_columns_from_pipeline(model) -> Optional[List[str]]:
    """
    Try to infer the expected feature column list from the pipeline.
    Falls back to loading from feature_columns.json if inference fails.

    Attempts (in order):
      1. model.feature_names_in_
      2. preprocessor.feature_names_in_
      3. preprocessor.transformers_ column specs
      4. models/feature_columns.json

    Returns
    -------
    List[str] or None
    """
    # 1) pipeline.feature_names_in_
    try:
        if hasattr(model, "feature_names_in_"):
            return list(model.feature_names_in_)
    except Exception:
        pass

    # 2) preprocessor.feature_names_in_
    try:
        pre = model.named_steps.get("preprocess", None)
        if pre is not None and hasattr(pre, "feature_names_in_"):
            return list(pre.feature_names_in_)
    except Exception:
        pass

    # 3) preprocessor.transformers_ column specs
    try:
        pre = model.named_steps.get("preprocess", None)
        if pre is not None and hasattr(pre, "transformers_"):
            cols = []
            for _, _, cols_spec in pre.transformers_:
                if isinstance(cols_spec, (list, tuple)):
                    cols.extend([c for c in cols_spec if isinstance(c, str)])
            if cols:
                return cols
    except Exception:
        pass

    # 4) feature_columns.json saved during training
    try:
        if Path(FEATURES_PATH).exists():
            with open(FEATURES_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
    except Exception:
        pass

    return None



# 3. Input Alignment Helpers


def _ensure_input_frame(input_obj: Any, feature_cols: List[str]) -> pd.DataFrame:
    """
    Convert input (dict or DataFrame) into a DataFrame aligned to expected features.
    Missing columns are filled with NaN (preprocessor handles imputation).

    Parameters
    ----------
    input_obj   : dict or pd.DataFrame
    feature_cols: List[str] — expected feature columns from the trained model

    Returns
    -------
    pd.DataFrame with exactly the columns the model expects
    """
    if isinstance(input_obj, dict):
        row = {c: input_obj.get(c, np.nan) for c in feature_cols}
        return pd.DataFrame([row])
    elif isinstance(input_obj, pd.DataFrame):
        return input_obj.reindex(columns=feature_cols)
    else:
        raise TypeError("Input must be a dict or pandas DataFrame.")


def _coerce_numeric_columns(df: pd.DataFrame, model) -> pd.DataFrame:
    """
    Ensure numeric columns are correctly typed before passing to the pipeline.
    Coerces strings/objects to numeric, replacing unparseable values with NaN.

    Parameters
    ----------
    df    : pd.DataFrame — aligned input frame
    model : fitted sklearn Pipeline

    Returns
    -------
    pd.DataFrame with numeric columns properly typed
    """
    try:
        pre = model.named_steps.get("preprocess", None)
        if pre is not None and hasattr(pre, "transformers_"):
            for name, _, cols in pre.transformers_:
                if name == "num" and isinstance(cols, (list, tuple)):
                    for c in cols:
                        if c in df.columns:
                            df[c] = pd.to_numeric(df[c], errors="coerce")
    except Exception:
        pass
    return df



# 4. Predict Single Car


def predict_single(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Predict the price for a single car.

    Parameters
    ----------
    input_data : dict
        Car features e.g.
        {
            "Company"     : "Toyota",
            "Model"       : "Innova",
            "FuelType"    : "Petrol",
            "Kilometer"   : 45000,
            "ModelYear"   : 2019,
            "Owner"       : "First",
            "Warranty"    : 1,
            "QualityScore": 8.5,
            "CarAge"      : 5
        }

    Returns
    -------
    dict
        {"predicted_price": <float>}
    """
    model = load_model()

    global _feature_cols_cache
    if _feature_cols_cache is None:
        _feature_cols_cache = _get_feature_columns_from_pipeline(model)

    if _feature_cols_cache is None:
        raise ValueError(
            "Could not determine expected feature columns. "
            "Ensure feature_columns.json exists in the models/ directory."
        )

    df_input = _ensure_input_frame(input_data, _feature_cols_cache)
    df_input = _coerce_numeric_columns(df_input, model)

    try:
        predicted_price = model.predict(df_input)[0]
    except Exception as e:
        raise ValueError("Model prediction failed: " + str(e)) from e

    return {
        "predicted_price": round(float(predicted_price), 2)
    }



# 5. Predict Batch (multiple cars)


def predict_batch(df: pd.DataFrame) -> pd.DataFrame:
    """
    Predict prices for multiple cars at once.

    Parameters
    ----------
    df : pd.DataFrame
        Each row is one car with the same feature columns as training.

    Returns
    -------
    pd.DataFrame
        Original DataFrame with an extra column: 'predicted_price'
    """
    model = load_model()

    global _feature_cols_cache
    if _feature_cols_cache is None:
        _feature_cols_cache = _get_feature_columns_from_pipeline(model)

    if _feature_cols_cache is None:
        raise ValueError(
            "Could not determine expected feature columns. "
            "Ensure feature_columns.json exists in the models/ directory."
        )

    df_input = df.reindex(columns=_feature_cols_cache)
    df_input = _coerce_numeric_columns(df_input, model)

    try:
        predicted_prices = model.predict(df_input)
    except Exception as e:
        raise ValueError("Model prediction failed: " + str(e)) from e

    out = df.copy()
    out["predicted_price"] = np.round(predicted_prices, 2)
    return out

if __name__ == "__main__":
    print("Testing single prediction...")
    test_input = {
        "Company"     : "Toyota",
        "Model"       : "Innova",
        "FuelType"    : "Petrol",
        "Kilometer"   : 45000,
        "ModelYear"   : 2019,
        "Owner"       : "First",
        "Warranty"    : 1,
        "QualityScore": 8.5,
        "CarAge"      : 5,
    }
    result = predict_single(test_input)
    print("Predicted Price: ₹", result["predicted_price"])