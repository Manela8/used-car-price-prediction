"""
Preprocessing utilities for Used Car Price Prediction.

- data_load        : read cleaned CSV and normalize column names.
- build_preprocessor: build ColumnTransformer including:
    * numeric pipeline    : SimpleImputer(mean) -> StandardScaler
    * categorical pipeline: SimpleImputer(most_frequent) -> OneHotEncoder(handle_unknown='ignore')
  Accepts either a DataFrame with the target column or features-only DataFrame.
- split_data       : train_test_split on continuous target Price
                     (returns X_train, X_test, y_train, y_test)
"""

from pathlib import Path
from typing import Tuple, List

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from src.config import DATA_FILE, TARGET_COL, TEST_SIZE, RANDOM_STATE



# 1. Data Loading


def data_load(data_path: Path = DATA_FILE) -> pd.DataFrame:
    """
    Load the pre-cleaned used car dataset and normalize column names.

    Parameters
    ----------
    data_path : Path
        Path to the cleaned CSV file. Defaults to DATA_FILE from config.

    Returns
    -------
    pd.DataFrame
        Cleaned dataset with stripped column names.
    """
    df = pd.read_csv(data_path)
    df.columns = [str(c).strip() for c in df.columns]
    return df


# 2. Helper - infer feature column types


def _infer_feature_columns(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Separate numeric and categorical feature columns from a features-only DataFrame.

    Parameters
    ----------
    X : pd.DataFrame
        Features DataFrame (no target column).

    Returns
    -------
    Tuple[List[str], List[str]]
        (numeric_cols, categorical_cols)
    """
    numeric_cols     = X.select_dtypes(include=["int64", "float64", "number"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    return numeric_cols, categorical_cols



# 3. Preprocessor Builder

def build_preprocessor(df_or_X: pd.DataFrame) -> ColumnTransformer:
    """
    Build a ColumnTransformer for imputation + scaling / encoding.

    Numeric columns  : SimpleImputer(mean)          -> StandardScaler
    Categorical cols : SimpleImputer(most_frequent)  -> OneHotEncoder(handle_unknown='ignore')

    Parameters
    ----------
    df_or_X : pd.DataFrame
        Either the full DataFrame (including Price) or features-only DataFrame.

    Returns
    -------
    ColumnTransformer
        Unfitted preprocessor ready for fit_transform / pipeline use.
    """
    df = df_or_X.copy()

    # Drop target column if present to get features only
    if TARGET_COL in df.columns:
        df = df.dropna(subset=[TARGET_COL])
        X = df.drop(columns=[TARGET_COL])
    else:
        X = df

    num_cols, cat_cols = _infer_feature_columns(X)

    # Numeric pipeline
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler",  StandardScaler()),
    ])

    # Categorical pipeline
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot",  OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    transformers = []
    if num_cols:
        transformers.append(("num", numeric_pipeline, num_cols))
    if cat_cols:
        transformers.append(("cat", categorical_pipeline, cat_cols))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor



# 4. Train-Test Split

def split_data(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split the dataset into train and test sets.

    NOTE: stratify is NOT used because Price is a continuous variable.
          Stratification is only valid for classification (discrete) targets.

    Parameters
    ----------
    df : pd.DataFrame
        Full cleaned DataFrame including the Price (target) column.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
        X_train, X_test, y_train, y_test
    """
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    if TARGET_COL not in df.columns:
        raise KeyError(
            f"Target column '{TARGET_COL}' not found. "
            f"Available columns: {df.columns.tolist()}"
        )

    # Drop rows where target is missing
    df = df.dropna(subset=[TARGET_COL])

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    # ✅ No stratify — Price is continuous, stratify is for classification only
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    df = data_load()
    print("Data loaded successfully.")
    print("Shape   :", df.shape)
    print("Columns :", df.columns.tolist())

    X_train, X_test, y_train, y_test = split_data(df)
    print("Train shape :", X_train.shape)
    print("Test shape  :", X_test.shape)

    preprocessor = build_preprocessor(X_train)
    print("Preprocessor built successfully.")