# app.py
"""
Predictive Risk Modeling Framework for Microplastic Pollution (Streamlit UI, auto-run)

New features added:
1) Data Cleaning Dashboard (missing values, outliers, duplicates)
2) Automatic Outlier Detection & Removal (IQR capping, IsolationForest)
3) Interactive Visualizations (heatmap, histograms, boxplots, categorical counts)
4) Model Comparison (LogisticRegression, RandomForest, GradientBoosting, SVM, XGBoost if present)
5) SHAP Explainability (if shap package available)
6) Download Cleaned Dataset
7) Full sklearn Pipeline (preprocessing -> scaler -> estimator)
8) Correlation Matrix + Feature Importance plotting
9) Merge multiple CSV files automatically (uploader supports multiple files)
10) Interactive map for GPS coordinates (if Latitude & Longitude present)

Notes:
- This script is defensive: missing optional packages (xgboost, shap) will not break the app.
- Outputs (plots, CSVs) continue to be saved to ./outputs/
"""
import os
import sys
import warnings
from typing import Dict, Tuple, List, Optional, Any

import io
import glob
import zipfile
import time

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, IsolationForest
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

warnings.filterwarnings("ignore")
RANDOM_STATE = 42

# Optional packages
try:
    import xgboost as xgb  # type: ignore
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

try:
    import shap  # type: ignore
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

# Directories
OUTPUT_DIR = "outputs"
INPUT_DIR = "inputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(INPUT_DIR, exist_ok=True)
sns.set(style="whitegrid")

# Try Streamlit import
try:
    import streamlit as st  # type: ignore

    STREAMLIT_AVAILABLE = True
except Exception:
    STREAMLIT_AVAILABLE = False

# ----------------------------
# Utility helpers
# ----------------------------
def save_and_show(fig, filename: str, dpi: int = 150, tight: bool = True) -> str:
    path = os.path.join(OUTPUT_DIR, filename)
    if tight:
        fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def read_csv_preview(path: str, nrows: int = 10) -> Tuple[pd.DataFrame, str]:
    encodings_to_try = [None, "utf-8", "latin1", "cp1252"]
    last_exception = None

    for enc in encodings_to_try:
        try:
            if enc is None:
                df = pd.read_csv(path, nrows=nrows)
            else:
                df = pd.read_csv(path, encoding=enc, nrows=nrows)
            return df, enc or "default"
        except UnicodeDecodeError as e:
            last_exception = e
            continue
        except Exception as e:
            last_exception = e
            continue

    # Final fallback: open with errors='replace'
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
        df = pd.read_csv(io.StringIO(text), nrows=nrows)
        return df, "replaced-invalid-bytes"
    except Exception as e:
        raise RuntimeError(f"Failed to read CSV for preview. Last error: {last_exception}; final error: {e}")


def robust_read_csv(path: str) -> Tuple[pd.DataFrame, str]:
    """Robust CSV loader used for full pipeline — tries several encodings and fallback."""
    encodings_to_try = [None, "utf-8", "latin1", "cp1252"]
    last_exception = None
    df = None
    used_encoding = None

    for enc in encodings_to_try:
        try:
            if enc is None:
                df = pd.read_csv(path)
                used_encoding = "default"
            else:
                df = pd.read_csv(path, encoding=enc)
                used_encoding = enc
            break
        except Exception as e:
            last_exception = e
            continue

    if df is None:
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()
            df = pd.read_csv(io.StringIO(text))
            used_encoding = "fallback-replace-invalid-bytes"
        except Exception as e:
            raise RuntimeError(
                f"Failed to read CSV from '{path}'. Last: {last_exception}; final: {e}"
            )
    return df, used_encoding


# ----------------------------
# Plot helpers
# ----------------------------
def plot_missing_values_heatmap(df: pd.DataFrame, fname: str = "missing_heatmap.png"):
    fig, ax = plt.subplots(figsize=(10, max(4, len(df.columns) * 0.2)))
    sns.heatmap(df.isnull(), cbar=False, yticklabels=False, ax=ax)
    ax.set_title("Missing Values Heatmap (True = missing)")
    return save_and_show(fig, fname)


def plot_boxplot(df: pd.DataFrame, col: str, fname: str):
    fig, ax = plt.subplots(figsize=(6, 3))
    sns.boxplot(x=df[col], ax=ax)
    ax.set_title(f"Boxplot: {col}")
    return save_and_show(fig, fname)


def plot_hist_and_kde(df: pd.DataFrame, col: str, fname: str):
    fig, ax = plt.subplots(figsize=(6, 3))
    sns.histplot(df[col].dropna(), kde=True, ax=ax)
    ax.set_title(f"Histogram: {col}")
    return save_and_show(fig, fname)


def plot_categorical_counts(df: pd.DataFrame, col: str, top_n: int = 10, fname: str = None):
    vc = df[col].astype(str).value_counts().nlargest(top_n)
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x=vc.values, y=vc.index, ax=ax)
    ax.set_xlabel("Count")
    ax.set_ylabel(col)
    ax.set_title(f"Top {len(vc)} value counts for {col}")
    return save_and_show(fig, fname or f"cat_counts_{col}.png")


def plot_correlation_matrix(df: pd.DataFrame, fname: str = "correlation_heatmap.png"):
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=False, cmap="coolwarm", ax=ax)
    ax.set_title("Correlation matrix (numeric features)")
    return save_and_show(fig, fname)


def plot_feature_importances(model, feature_names: List[str], name: str = "model", top_n: int = 15):
    importances = None
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        coef = model.coef_
        if coef.ndim == 1:
            importances = np.abs(coef)
        else:
            importances = np.mean(np.abs(coef), axis=0)
    else:
        return None

    fi = pd.Series(importances, index=feature_names).sort_values(ascending=False).head(top_n)
    fig, ax = plt.subplots(figsize=(8, max(4, len(fi) * 0.35)))
    sns.barplot(x=fi.values, y=fi.index, ax=ax)
    ax.set_title(f"{name} feature importances")
    ax.set_xlabel("Importance")
    return save_and_show(fig, f"feature_importances_{name}.png")


def plot_confusion_matrix_heatmap(y_true, y_pred, classes: List[str], name: str):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix: {name}")
    if classes is not None:
        ax.set_xticklabels(classes, rotation=45, ha="right")
        ax.set_yticklabels(classes, rotation=0)
    return save_and_show(fig, f"confusion_matrix_{name}.png")


# ----------------------------
# Feature ranking / selection
# ----------------------------
def rank_and_select_features(X: pd.DataFrame, y: pd.Series, top_n: int = 30) -> Tuple[List[str], Dict]:
    meta: Dict[str, Any] = {}
    if X.shape[0] < 2 or y.nunique() < 2:
        return X.columns.tolist(), meta
    rf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X, y)
    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    top_features = importances.head(min(top_n, len(importances))).index.tolist()
    selector = SelectFromModel(rf, threshold="mean", prefit=True)
    selected_mask = selector.get_support()
    selected = list(X.columns[selected_mask])
    if not selected:
        selected = top_features[: min(10, len(top_features))]
    meta["feature_importances"] = importances.to_dict()
    meta["top_features"] = top_features
    meta["selected_features"] = selected
    return selected, meta


# ----------------------------
# Preprocessing & pipeline helpers
# ----------------------------
def fill_missing_values(df: pd.DataFrame, strategy: str = "median", custom_values: Optional[Dict[str, Any]] = None):
    """Fill missing values: strategy='median'|'mean'|'mode'|'custom' """
    df = df.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    obj_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if strategy == "median":
        for c in num_cols:
            df[c] = df[c].fillna(df[c].median())
        for c in obj_cols:
            df[c] = df[c].fillna(df[c].mode().iloc[0] if not df[c].mode().empty else "Missing")
    elif strategy == "mean":
        for c in num_cols:
            df[c] = df[c].fillna(df[c].mean())
        for c in obj_cols:
            df[c] = df[c].fillna(df[c].mode().iloc[0] if not df[c].mode().empty else "Missing")
    elif strategy == "mode":
        for c in num_cols:
            df[c] = df[c].fillna(df[c].mode().iloc[0] if not df[c].mode().empty else 0)
        for c in obj_cols:
            df[c] = df[c].fillna(df[c].mode().iloc[0] if not df[c].mode().empty else "Missing")
    elif strategy == "custom" and custom_values:
        for c, v in custom_values.items():
            if c in df.columns:
                df[c] = df[c].fillna(v)
    else:
        # default fallback
        df = df.fillna(0)
    return df


def iqr_cap_series(s: pd.Series):
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return s.clip(lower=lower, upper=upper)


def detect_outliers_isolationforest(df: pd.DataFrame, numeric_cols: List[str], contamination: float = 0.05):
    if len(numeric_cols) == 0 or df.shape[0] < 10:
        return np.array([], dtype=int)
    iso = IsolationForest(contamination=contamination, random_state=RANDOM_STATE)
    try:
        preds = iso.fit_predict(df[numeric_cols].fillna(0))
        outlier_idx = np.where(preds == -1)[0]
        return outlier_idx
    except Exception:
        return np.array([], dtype=int)


def encode_dataframe(df: pd.DataFrame, cols_to_encode: Optional[List[str]] = None, max_onehot=12):
    """Encode all object columns: one-hot for low cardinality, label-encode otherwise.
    Returns encoded_df, encoders."""
    df = df.copy()
    encoders: Dict[str, Any] = {}
    if cols_to_encode is None:
        cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    else:
        cols = [c for c in cols_to_encode if c in df.columns]
    for col in cols:
        df[col] = df[col].astype(str).fillna("Missing")
        nunique = df[col].nunique()
        if nunique <= max_onehot:
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
            df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
        else:
            le = LabelEncoder()
            df[col + "_LE"] = le.fit_transform(df[col])
            encoders[col] = le
            df = df.drop(columns=[col])
    return df, encoders


# ----------------------------
# Core pipeline
# ----------------------------
def preprocess_and_split(df: pd.DataFrame,
                         target_col: Optional[str] = None,
                         fill_strategy: str = "median",
                         cap_outliers: bool = True,
                         remove_outliers: bool = False,
                         outlier_method: str = "iqr",
                         isolation_contamination: float = 0.05,
                         onehot_max: int = 12,
                         select_features: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Dict]:
    """
    Preprocess DataFrame and return X_train, X_test, y_train, y_test, meta
    meta includes encoders, scaler, selected features, transforms
    """
    meta: Dict[str, Any] = {}
    df = df.copy()
    # detect target column
    if target_col is None:
        if "Risk_Level" in df.columns:
            target_col = "Risk_Level"
        elif "Risk_Type" in df.columns:
            target_col = "Risk_Type"
        else:
            target_col = df.columns[-1]

    # fill missing
    df = fill_missing_values(df, strategy=fill_strategy)

    # Outlier detection options
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)

    if cap_outliers:
        # cap numeric columns using IQR
        for c in numeric_cols:
            try:
                df[c] = iqr_cap_series(df[c])
            except Exception:
                pass

    outlier_indices = np.array([], dtype=int)
    if remove_outliers and outlier_method == "isolationforest":
        outlier_indices = detect_outliers_isolationforest(df, numeric_cols, contamination=isolation_contamination)
        if outlier_indices.size:
            df = df.drop(df.index[outlier_indices]).reset_index(drop=True)

    # encode categorical columns
    df_encoded, encoders = encode_dataframe(df, None, max_onehot=onehot_max)

    # Prepare y
    if target_col not in df_encoded.columns and target_col in df.columns:
        # Use original target column for label encoding decision
        y_raw = df[target_col]
    elif target_col in df_encoded.columns:
        y_raw = df_encoded[target_col]
    else:
        raise RuntimeError(f"Target column '{target_col}' not found in dataset after encoding.")

    if y_raw.dtype == "O" or not pd.api.types.is_numeric_dtype(y_raw):
        le_target = LabelEncoder()
        y = pd.Series(le_target.fit_transform(y_raw.astype(str)), name=target_col)
        meta["label_encoder_target"] = le_target
    else:
        y = pd.Series(y_raw, name=target_col)

    X = df_encoded.drop(columns=[target_col]) if target_col in df_encoded.columns else df_encoded.copy()
    # Ensure all numeric
    non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
    for c in non_numeric:
        X[c] = pd.to_numeric(X[c].astype(str).fillna("0"), errors="coerce").fillna(0.0)

    # scaler
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    scaler = StandardScaler() if numeric_features else None
    if numeric_features:
        X[numeric_features] = scaler.fit_transform(X[numeric_features])

    # feature selection
    selected_features = X.columns.tolist()
    fs_meta = {}
    if select_features and X.shape[0] >= 10 and y.nunique() >= 2:
        selected, fs_meta = rank_and_select_features(X, y, top_n=40)
        if selected:
            selected_features = selected
            X = X[selected_features]

    # train-test split
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    except Exception:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=None)

    meta.update({
        "encoders": encoders,
        "scaler": scaler,
        "selected_features": selected_features,
        "feature_selection_meta": fs_meta,
        "numeric_features": numeric_features,
        "target_col": target_col,
        "outlier_indices": outlier_indices.tolist(),
    })
    return X_train, X_test, y_train, y_test, meta


# ----------------------------
# Training / evaluation / CV
# ----------------------------
def get_models_dict(include_xgboost: bool = True):
    models = {
        "LogisticRegression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=200, random_state=RANDOM_STATE),
        "SVM": SVC(probability=True, random_state=RANDOM_STATE),
    }
    if include_xgboost and XGBOOST_AVAILABLE:
        models["XGBoost"] = xgb.XGBClassifier(n_estimators=200, use_label_encoder=False, eval_metric="logloss", random_state=RANDOM_STATE)
    return models


def train_models(X_train: pd.DataFrame, y_train: pd.Series, chosen_models: Optional[List[str]] = None) -> Dict[str, Any]:
    all_models = get_models_dict(include_xgboost=True)
    if chosen_models:
        models_to_train = {k: v for k, v in all_models.items() if k in chosen_models}
    else:
        models_to_train = all_models
    trained = {}
    if y_train.nunique() < 2:
        raise RuntimeError("Target y contains only one class; cannot train classifiers.")
    for name, model in models_to_train.items():
        model.fit(X_train, y_train)
        trained[name] = model
        try:
            plot_feature_importances(model, X_train.columns.tolist(), name=name, top_n=min(20, X_train.shape[1]))
        except Exception:
            pass
    return trained


def evaluate_models(trained_models: Dict[str, Any], X_test: pd.DataFrame, y_test: pd.Series, meta: Dict):
    results: Dict[str, Dict[str, float]] = {}
    le_target = meta.get("label_encoder_target", None)
    class_names = list(le_target.classes_) if le_target is not None else None
    for name, model in trained_models.items():
        y_pred = model.predict(X_test)
        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
        rec = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
        f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
        results[name] = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}
        try:
            plot_confusion_matrix_heatmap(y_test, y_pred, classes=class_names, name=name)
        except Exception:
            pass
    try:
        pd.DataFrame(results).T.to_csv(os.path.join(OUTPUT_DIR, "test_results_summary.csv"))
    except Exception:
        pass
    try:
        plot_metrics = plot_metrics_comparison(results)  # uses helper defined later
    except Exception:
        plot_metrics = None
    return results


def cross_validate_models(trained_models: Dict[str, Any], X: pd.DataFrame, y: pd.Series, k: int = 5):
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=RANDOM_STATE)
    cv_scores: Dict[str, np.ndarray] = {}
    if y.nunique() < 2:
        return {k: np.array([]) for k in trained_models.keys()}
    for name, model in trained_models.items():
        try:
            scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
            cv_scores[name] = scores
        except Exception:
            cv_scores[name] = np.array([])
    try:
        plot_cv_boxplot(cv_scores, prefix="cv")
    except Exception:
        pass
    try:
        cv_summary = {k: (float(np.mean(v)) if v.size else np.nan) for k, v in cv_scores.items()}
        pd.Series(cv_summary).to_csv(os.path.join(OUTPUT_DIR, "cv_mean_accuracy_summary.csv"))
    except Exception:
        pass
    return cv_scores


# small wrapper used above - placed here to avoid forward ref issues
def plot_metrics_comparison(metrics_dict: Dict[str, Dict[str, float]], prefix: str = "metrics"):
    df = pd.DataFrame(metrics_dict).T  # models x metrics
    metrics = ["accuracy", "precision", "recall", "f1"]
    df = df.reindex(columns=[m for m in metrics if m in df.columns])
    fig, ax = plt.subplots(figsize=(8, 5))
    df.plot(kind="bar", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_title("Model comparison on test set")
    ax.set_ylabel("Score")
    return save_and_show(fig, f"{prefix}_comparison.png")


def plot_cv_boxplot(cv_scores: Dict[str, np.ndarray], prefix: str = "cv"):
    data = []
    labels = []
    for name, scores in cv_scores.items():
        data.append(scores)
        labels.append(name)
    if not any(len(arr) for arr in data):
        return None
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(data, labels=labels)
    ax.set_title("Cross-validation accuracy distribution")
    ax.set_ylabel("Accuracy")
    return save_and_show(fig, f"{prefix}_boxplot.png")


# ----------------------------
# Streamlit UI
# ----------------------------
def run_streamlit_app():
    st.set_page_config(page_title="Microplastic Risk Modeling", layout="wide")
    st.sidebar.title("Navigation")
    nav = st.sidebar.radio("Go to", ("Upload & Merge", "Data Cleaning Dashboard", "Visualizations", "Modeling & Results", "Download Outputs"))

    st.title("Predictive Risk Modeling for Microplastic Pollution — Enhanced")
    st.write("Upload CSV(s), clean data, visualize, train & compare models, and download results.")

    # ---- Upload & Merge ----
    if nav == "Upload & Merge":
        st.header("Upload CSV files (you may upload multiple files to merge)")
        uploaded_files = st.file_uploader("Upload one or more CSV files", type=["csv"], accept_multiple_files=True)
        if uploaded_files:
            merged_df = None
            encodings_info = {}
            for f in uploaded_files:
                save_path = os.path.join(INPUT_DIR, f.name)
                with open(save_path, "wb") as out:
                    out.write(f.getbuffer())
                try:
                    df_tmp, encoding_used = read_csv_preview(save_path, nrows=50)
                    encodings_info[f.name] = encoding_used
                except Exception:
                    encodings_info[f.name] = "preview-failed"
                # robust full read
                try:
                    df_full, used_enc = robust_read_csv(save_path)
                except Exception as e:
                    st.error(f"Failed to read {f.name}: {e}")
                    continue
                if merged_df is None:
                    merged_df = df_full
                else:
                    # align columns (union)
                    merged_df = pd.concat([merged_df, df_full], axis=0, ignore_index=True, sort=False)
            if merged_df is not None:
                st.success(f"Merged {len(uploaded_files)} files — resulting shape: {merged_df.shape}")
                st.write("Encodings detected (preview):")
                st.write(encodings_info)
                st.session_state["raw_merged_df"] = merged_df
                st.dataframe(merged_df.head(50))
                # save merged
                merged_save = os.path.join(INPUT_DIR, "merged_uploaded.csv")
                try:
                    merged_df.to_csv(merged_save, index=False)
                    st.write(f"Merged file saved to {merged_save}")
                except Exception:
                    pass
        else:
            st.info("Upload one or more CSV files to merge and start the pipeline.")

    # ---- Data Cleaning Dashboard ----
    if nav == "Data Cleaning Dashboard":
        st.header("Data Cleaning Dashboard")
        df = st.session_state.get("raw_merged_df", None)
        if df is None:
            st.info("No dataset loaded. Go to 'Upload & Merge' and upload files.")
        else:
            st.subheader("Dataset overview")
            st.write(f"Shape: {df.shape}")
            st.write("Columns and dtypes:")
            st.write(pd.DataFrame({"col": df.columns, "dtype": df.dtypes.astype(str)}))

            st.subheader("Missing values & duplicates")
            miss = df.isnull().sum().sort_values(ascending=False)
            st.write(miss[miss > 0])

            if st.checkbox("Show rows with any missing values (sample 50)"):
                st.dataframe(df[df.isnull().any(axis=1)].head(50))

            if st.checkbox("Remove duplicate rows (keep first)"):
                before = df.shape[0]
                df = df.drop_duplicates(keep="first").reset_index(drop=True)
                after = df.shape[0]
                st.success(f"Removed {before-after} duplicate rows.")
            st.write("Duplicate rows: ", df.duplicated().sum())

            st.subheader("Missing value filling options")
            fill_opt = st.selectbox("Fill strategy", ["median", "mean", "mode", "custom"])
            custom_map = {}
            if fill_opt == "custom":
                st.write("Enter custom fill values per column (optional). Leave blank to skip.")
                for c in df.columns:
                    v = st.text_input(f"Fill value for {c} (string interpreted as-is)", key=f"fill_{c}")
                    if v != "":
                        # try to coerce numeric
                        try:
                            vv = float(v)
                            custom_map[c] = vv
                        except Exception:
                            custom_map[c] = v
            if st.button("Apply fill missing values"):
                df = fill_missing_values(df, strategy=fill_opt, custom_values=custom_map if fill_opt == "custom" else None)
                st.session_state["cleaned_df"] = df
                st.success("Missing values filled as requested.")
                st.dataframe(df.head(20))

            st.subheader("Outlier handling")
            cap_out = st.checkbox("Cap outliers (IQR capping) — apply now", value=True)
            rm_out = st.checkbox("Remove detected outliers using IsolationForest", value=False)
            contamination = st.slider("IsolationForest contamination (if removing)", 0.01, 0.3, 0.05, 0.01)
            if st.button("Apply outlier handling"):
                # use current df (prefer cleaned_df if exists)
                df_work = st.session_state.get("cleaned_df", df).copy()
                numeric_cols = df_work.select_dtypes(include=[np.number]).columns.tolist()
                if cap_out:
                    for c in numeric_cols:
                        try:
                            df_work[c] = iqr_cap_series(df_work[c])
                        except Exception:
                            pass
                if rm_out:
                    idxs = detect_outliers_isolationforest(df_work, numeric_cols, contamination=contamination)
                    nrm = len(idxs)
                    if nrm:
                        df_work = df_work.drop(df_work.index[idxs]).reset_index(drop=True)
                        st.success(f"Removed {nrm} rows detected as outliers by IsolationForest.")
                    else:
                        st.info("No outliers detected by IsolationForest (or detection failed).")
                st.session_state["cleaned_df"] = df_work
                st.write(df_work.head(20))

            st.subheader("Download cleaned dataset")
            cleaned = st.session_state.get("cleaned_df", df)
            if st.button("Save cleaned dataset to outputs/cleaned_dataset.csv"):
                outpath = os.path.join(OUTPUT_DIR, "cleaned_dataset.csv")
                cleaned.to_csv(outpath, index=False)
                st.success(f"Saved cleaned dataset: {outpath}")
                with open(outpath, "rb") as f:
                    st.download_button("Download cleaned CSV", data=f, file_name="cleaned_dataset.csv")

    # ---- Visualizations ----
    if nav == "Visualizations":
        st.header("Interactive Visualizations")
        df = st.session_state.get("cleaned_df", st.session_state.get("raw_merged_df", None))
        if df is None:
            st.info("No dataset available. Upload files in 'Upload & Merge' first.")
        else:
            st.subheader("Missing values heatmap")
            try:
                mh = plot_missing_values_heatmap(df)
                st.image(mh, caption="Missing values heatmap", use_column_width=True)
            except Exception as e:
                st.write("Could not create missing heatmap:", e)

            st.subheader("Choose a column to inspect")
            col = st.selectbox("Column", df.columns.tolist())
            if pd.api.types.is_numeric_dtype(df[col]):
                st.write("Numeric column plots")
                p1 = plot_hist_and_kde(df, col, f"hist_{col}.png")
                p2 = plot_boxplot(df, col, f"box_{col}.png")
                st.image(p1, caption=f"Histogram of {col}")
                st.image(p2, caption=f"Boxplot of {col}")
            else:
                st.write("Categorical column plots")
                p = plot_categorical_counts(df, col, top_n=20, fname=f"cat_{col}.png")
                st.image(p, caption=f"Counts for {col}")

            st.subheader("Correlation matrix (numeric)")
            try:
                corrp = plot_correlation_matrix(df.select_dtypes(include=[np.number]))
                st.image(corrp, caption="Correlation matrix")
            except Exception as e:
                st.write("Could not create correlation matrix:", e)

            st.subheader("Map (if Latitude & Longitude present)")
            lat_cols = [c for c in df.columns if c.lower() in ("latitude", "lat")]
            lon_cols = [c for c in df.columns if c.lower() in ("longitude", "lon", "lng")]
            if lat_cols and lon_cols:
                latc = lat_cols[0]
                lonc = lon_cols[0]
                sub = df[[latc, lonc]].dropna()
                sub = sub.rename(columns={latc: "lat", lonc: "lon"})
                st.map(sub.rename(columns={"lat": "latitude", "lon": "longitude"}))
            else:
                st.info("No Latitude/Longitude columns detected for map visualization.")

    # ---- Modeling & Results ----
    if nav == "Modeling & Results":
        st.header("Modeling & Results")
        df = st.session_state.get("cleaned_df", st.session_state.get("raw_merged_df", None))
        if df is None:
            st.info("No dataset available. Upload files in 'Upload & Merge' first.")
        else:
            st.subheader("Modeling options")
            target_col = st.selectbox("Select target column", df.columns.tolist(), index=min(0, max(0, list(df.columns).index("Risk_Level")) ) )
            fill_strategy = st.selectbox("Missing fill strategy (applied before modeling)", ["median", "mean", "mode"])
            cap_outliers = st.checkbox("Cap outliers (IQR) before modeling", value=True)
            rm_outliers = st.checkbox("Remove outliers via IsolationForest before modeling", value=False)
            isolation_cont = st.slider("IsolationForest contamination if removing", 0.01, 0.3, 0.05, 0.01)
            onehot_max = st.slider("Max cardinality for one-hot (otherwise label-encode)", 2, 50, 12)

            st.write("Choose models to train:")
            available_models = list(get_models_dict(include_xgboost=True).keys())
            chosen = st.multiselect("Models", available_models, default=available_models)

            if st.button("Run preprocessing + train models"):
                with st.spinner("Preprocessing & training..."):
                    try:
                        X_train, X_test, y_train, y_test, meta = preprocess_and_split(
                            df,
                            target_col=target_col,
                            fill_strategy=fill_strategy,
                            cap_outliers=cap_outliers,
                            remove_outliers=rm_outliers,
                            outlier_method="isolationforest" if rm_outliers else "iqr",
                            isolation_contamination=isolation_cont,
                            onehot_max=onehot_max,
                            select_features=True,
                        )
                        st.session_state["modeling"] = {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test, "meta": meta}
                        trained = train_models(X_train, y_train, chosen_models=chosen)
                        st.session_state["trained_models"] = trained
                        results = evaluate_models(trained, X_test, y_test, meta)
                        st.session_state["test_results"] = results
                        cv_scores = cross_validate_models(trained, pd.concat([X_train, X_test], axis=0), pd.concat([y_train, y_test], axis=0), k=5)
                        st.session_state["cv_scores"] = {k: v.tolist() for k, v in cv_scores.items()}
                        st.success("Training & evaluation finished.")
                    except Exception as e:
                        st.error(f"Modeling failed: {e}")

            if "test_results" in st.session_state:
                st.subheader("Test set results")
                tr = st.session_state["test_results"]
                st.dataframe(pd.DataFrame(tr).T)

            if "cv_scores" in st.session_state:
                st.subheader("Cross-validation (accuracy per fold)")
                cs = st.session_state["cv_scores"]
                for m, arr in cs.items():
                    st.write(f"{m}: {arr}")

            st.subheader("Feature selection summary (if used)")
            meta = st.session_state.get("modeling", {}).get("meta", {})
            if meta:
                fs = meta.get("feature_selection_meta", {})
                st.write("Selected features:", meta.get("selected_features", []))
                st.write("Top features (if available):")
                st.write(fs.get("top_features", []) if isinstance(fs, dict) else fs)

            st.subheader("SHAP explainability")
            if SHAP_AVAILABLE:
                if st.button("Compute SHAP for trained RandomForest (slow)"):
                    trained = st.session_state.get("trained_models", {})
                    rf = trained.get("RandomForest", None)
                    if rf is None:
                        st.info("RandomForest not trained; train it or include it in models.")
                    else:
                        try:
                            explainer = shap.TreeExplainer(rf)
                            X_sample = st.session_state["modeling"]["X_test"].sample(min(100, st.session_state["modeling"]["X_test"].shape[0]), random_state=RANDOM_STATE)
                            shap_values = explainer.shap_values(X_sample)
                            # show summary plot
                            shap.summary_plot(shap_values, X_sample, show=False)
                            fig = plt.gcf()
                            path = save_and_show(fig, "shap_summary.png")
                            st.image(path, caption="SHAP summary (sample)")
                        except Exception as e:
                            st.write("SHAP failed:", e)
            else:
                st.info("SHAP package not available. Install `shap` to enable explainability.")

    # ---- Download Outputs ----
    if nav == "Download Outputs":
        st.header("Download Outputs")
        st.write(f"All pipeline outputs are saved to: {os.path.abspath(OUTPUT_DIR)}")
        # list files
        files = glob.glob(os.path.join(OUTPUT_DIR, "*"))
        if files:
            st.write("Saved files:")
            for f in files:
                st.write("-", os.path.basename(f))
            if st.button("Create zip of outputs"):
                zip_path = os.path.join(OUTPUT_DIR, "outputs_bundle.zip")
                with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                    for file in files:
                        zf.write(file, arcname=os.path.basename(file))
                with open(zip_path, "rb") as fh:
                    st.download_button("Download outputs zip", data=fh, file_name="outputs_bundle.zip")
        else:
            st.write("No outputs present yet.")

# ----------------------------
# Entrypoint
# ----------------------------
if __name__ == "__main__":
    if STREAMLIT_AVAILABLE:
        run_streamlit_app()
    else:
        print("Streamlit not available. Run this script with `streamlit run app.py` to use the full UI.")
