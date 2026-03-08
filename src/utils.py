from __future__ import annotations
import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from joblib import dump, load


# Constants
TARGET_COL = "is_suspicious"
ID_COL = "id"
RANDOM_STATE = 42

# -----------------------------------
# Loading all data (historical and new)
# -----------------------------------

def load_historical(path:str) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Load historical data (contains target)
    Returns: X, y, df aka featues, target and the full dataframe
    """
    df = pd.read_csv(path)
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in the dataset.")
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    return X, y, df

def load_new(path:str) -> pd.DataFrame:
    """
    Load new_data (no target) when it comes. Safe even if the target column is present, it will be ignored.
    Returns: df
    """
    df = pd.read_csv(path)
    if TARGET_COL in df.columns:
        df = df.drop(columns=[TARGET_COL])
    return df

# -----------------------------------
# Preprocessing/Pipeline building functions
# -----------------------------------

def infer_feature_types(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """
    Infer which columns are categorical and which are numeric based on their dtype.
    - Categorical: dtype == 'object'
    - Numeric: all other dtypes

    """
    num_cols = [col for col in X.columns if is_numeric_dtype(X[col])]
    cat_cols = [col for col in X.columns if col not in num_cols]
    return num_cols, cat_cols

def build_preprocess(X_schema: pd.DataFrame) -> ColumnTransformer:
    """
    Build a preprocessing:
    - Numeric: median imputation
    - Categorical: most frequent imputation + one-hot encoding
    X_schema is used to "lock" which columns are treated as numeric/categorical, so that if the new data has different dtypes, the pipeline will still work.
    """
    num_cols, cat_cols = infer_feature_types(X_schema)

    preprocess = ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy='median'), num_cols),
            ('cat', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore')),
            ]), cat_cols),
        ],
        remainder='drop',  # Drop any columns not specified in transformers
        verbose_feature_names_out=False,  # Keep original column names for easier interpretation
    )
    return preprocess

def make_pipeline(model, X_schema: pd.DataFrame) -> Pipeline:
    """
    Create a pipeline with preprocessing and the given model.
    """
    preprocess = build_preprocess(X_schema)
    pipe = Pipeline(steps=[
        ('prep', preprocess),
        ('model', model)
    ])
    return pipe

# -----------------------------------
# Model saving/loading functions
# -----------------------------------

def save_pipeline(pipe: Pipeline, path: str) -> None:
    """
    Save a trained pipeline (preprocessing + model).
    """
    dump(pipe, path)

def load_pipeline(path: str) -> Pipeline:
    """
    Load the saved pipeline.
    """
    return load(path)

# -----------------------------------
# Evaluation functions
# -----------------------------------

def topx_report(y_true: pd.Series, y_proba: np.ndarray, top_frac: float) -> dict:
    """
    Evalute Top-X strategy.
    Example: top_frac=0.05 means flag the 5% highest risk cases.
    Returns:
    - top_frac
    - k_flagged
    - precision
    - recall
    """
    if not (0 < top_frac <= 1):
        raise ValueError("top_frac must be between 0 and 1.")
    
    y_true = pd.Series(y_true).reset_index(drop=True)
    y_proba = np.asarray(y_proba)

    n = len(y_true)
    k = max(1, int(round(top_frac * n)))  # Ensure at least 1 case is flagged

    idx = np.argsort(-y_proba)[:k]  # Indices of the top k predicted probabilities
    flagged_true = y_true.iloc[idx]

    precision = float((flagged_true.mean()))  # Proportion of flagged cases that are actually positive
    recall = float(flagged_true.sum() / max(1, y_true.sum()))  # Proportion of all positive cases that are flagged

    return {
        'top_frac': float(top_frac),
        'k_flagged': int(k),
        'precision': precision,
        'recall': recall
    }

def compare_topx_levels(y_true: pd.Series, y_proba: np.ndarray, top_fracs: tuple[float, ...] = (0.03, 0.05, 0.10)) -> pd.DataFrame:
    """
    Compare several Top-X levels at once, returning a dataframe with the results for each level.
    """
    rows = []
    for frac in top_fracs:
        rows.append(topx_report(y_true, y_proba, frac))
    return pd.DataFrame(rows)

def threshold_report(y_true: pd.Series, y_proba: np.ndarray, threshold: float) -> dict:
    """
    Evaluate threshold-based strategy.
    Returns:
    - TP, FP, TN, FN, precision, recall and flagged counts at the given threshold.
    """
    if not(0 <= threshold <= 1):
        raise ValueError("threshold must be between 0 and 1.")
    
    y_true = pd.Series(y_true).astype(int).reset_index(drop=True)
    y_proba = np.asarray(y_proba)

    y_pred = (y_proba >= threshold).astype(int)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) else 0.0
    return {
        'threshold': float(threshold),
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn,
        'precision': precision,
        'recall': recall,
        'flagged': int((y_pred == 1).sum())
    }

# -----------------------------------
# Deployment helper functions
# -----------------------------------

def prioritize_new_data(
        pipe: Pipeline,
        new_df: pd.DataFrame,
        top_frac: float = 0.02,
        id_col: str = ID_COL
) -> pd.DataFrame:
    """
    Given a new dataframe, predict probabilities and return the top X% highest risk cases, sorted by risk score.
     - pipe: the trained pipeline to use for predictions
     - new_df: the new data to prioritize (must contain id_col)
     - top_frac: the fraction of cases to flag (e.g. 0.02 for top 2%)
     - id_col: the name of the ID column in new_df
     Returns a dataframe with columns [id_col, risk_score], sorted by risk_score descending, containing only the top X% cases.
     Note: this function assumes that the pipeline's preprocessing can handle the new_df as is (e.g. it has the same columns as the training data).
    """
    if id_col not in new_df.columns:
        raise ValueError(f"ID column '{id_col}' not found in the new data.")
    
    if not (0 < top_frac <= 1):
        raise ValueError("top_frac must be between 0 and 1.")
    
    proba = pipe.predict_proba(new_df)[:, 1]  # Get probability of the positive class

    out = new_df[[id_col]].copy()
    out['risk_score'] = proba
    out = out.sort_values(by='risk_score', ascending=False)

    k = max(1, int(round(top_frac * len(out))))  # Ensure at least 1 case is flagged

    return out.head(k)

