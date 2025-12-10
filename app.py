# app.py
# Streamlit Dashboard: Predictive Risk Modeling Framework for Microplastic Pollution
# - Objective 1: EDA + preprocessing (load/apply preprocessor)
# - Objective 2: modeling + feature relevance (load trained artifacts)
# - Objective 3: validation (display CV results; optional run-CV if enabled)

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import streamlit as st

# Optional plotting libs
import matplotlib.pyplot as plt

# Optional ML libs (only needed if you enable "Train/CV inside app")
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

try:
    import joblib  # recommended for sklearn artifacts
except Exception:
    joblib = None


# ----------------------------
# Config (EDIT THESE)
# ----------------------------
@dataclass(frozen=True)
class AppConfig:
    # Column names in your dataset (EDIT to match your CSV headers)
    target_risk_level: str = "risk_level"      # classification target (Objective #1)
    target_risk_type: str = "Risk_Type"        # classification target (Objective #2)
    risk_score_col: str = "risk_score"         # numeric
    mp_count_col: str = "mp count per l"       # numeric
    polymer_type_col: str = "Polymer Type"     # categorical (optional)

    # Artifact paths (produced by your offline training pipeline)
    artifacts_dir: str = "artifacts"
    preprocessor_path: str = "preprocessor.pkl"
    model_path: str = "model.pkl"
    risk_type_model_path: str = "risk_type_model.pkl"
    selected_features_path: str = "selected_features.json"
    feature_relevance_path: str = "feature_relevance.csv"
    cv_results_path: str = "cv_results.csv"
    risk_type_cv_results_path: str = "risk_type_cv_results.csv"


CFG = AppConfig()


# ----------------------------
# Streamlit Page Setup
# ----------------------------
st.set_page_config(
    page_title="Microplastic Risk Modeling Dashboard",
    page_icon="🧪",
    layout="wide",
)

st.title("🧪 Predictive Risk Modeling Framework for Microplastic Pollution")
st.caption(
    "EDA • Preprocessing • Feature Selection • Modeling • Cross-Validation • Prediction"
)


# ----------------------------
# Helpers
# ----------------------------
def _safe_read_csv(uploaded_file) -> pd.DataFrame:
    # Try utf-8, fallback latin1
    try:
        return pd.read_csv(uploaded_file)
    except UnicodeDecodeError:
        return pd.read_csv(uploaded_file, encoding="latin1")


def _artifact(p: str) -> Path:
    return Path(CFG.artifacts_dir) / p


@st.cache_data(show_spinner=False)
def load_dataset_from_upload(uploaded_file) -> pd.DataFrame:
    df = _safe_read_csv(uploaded_file)
    # Normalize column names lightly (keep original too)
    df.columns = [c.strip() for c in df.columns]
    return df


@st.cache_resource(show_spinner=False)
def load_joblib_artifact(path: Path):
    if not path.exists():
        return None
    if joblib is None:
        st.warning("joblib is not available. Install: pip install joblib")
        return None
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"Failed to load artifact: {path}\n\n{e}")
        return None


@st.cache_data(show_spinner=False)
def load_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_csv_artifact(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def make_hist(series: pd.Series, title: str, bins: int = 30):
    fig, ax = plt.subplots()
    s = series.dropna()
    ax.hist(s, bins=bins)
    ax.set_title(title)
    ax.set_xlabel(series.name or "")
    ax.set_ylabel("Count")
    st.pyplot(fig, clear_figure=True)


def make_box_by_group(df: pd.DataFrame, value_col: str, group_col: str, title: str):
    dd = df[[value_col, group_col]].dropna()
    if dd.empty:
        st.info("No data available for this plot (after dropping NA).")
        return

    groups = []
    labels = []
    for g, sub in dd.groupby(group_col):
        groups.append(sub[value_col].values)
        labels.append(str(g))

    fig, ax = plt.subplots()
    ax.boxplot(groups, labels=labels, vert=True)
    ax.set_title(title)
    ax.set_xlabel(group_col)
    ax.set_ylabel(value_col)
    plt.xticks(rotation=30, ha="right")
    st.pyplot(fig, clear_figure=True)


def make_scatter(df: pd.DataFrame, x: str, y: str, title: str):
    dd = df[[x, y]].dropna()
    if dd.empty:
        st.info("No data available for this plot (after dropping NA).")
        return
    fig, ax = plt.subplots()
    ax.scatter(dd[x], dd[y], s=10)
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    st.pyplot(fig, clear_figure=True)


def metric_card_row(metrics: Dict[str, float]):
    cols = st.columns(len(metrics))
    for i, (k, v) in enumerate(metrics.items()):
        cols[i].metric(k, f"{v:.4f}")


def split_features_targets(
    df: pd.DataFrame,
    target_col: str,
    drop_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    drop_cols = drop_cols or []
    y = df[target_col]
    X = df.drop(columns=[target_col] + drop_cols, errors="ignore")
    return X, y


def evaluate_classifier_basic(y_true, y_pred, y_proba=None) -> Dict[str, float]:
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
    # For binary only ROC AUC
    if y_proba is not None:
        try:
            if len(np.unique(y_true)) == 2:
                out["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        except Exception:
            pass
    return out


# ----------------------------
# Sidebar: Data + Settings
# ----------------------------
with st.sidebar:
    st.header("📦 Data Input")
    uploaded = st.file_uploader("Upload CSV dataset", type=["csv"])

    st.divider()
    st.header("🧩 Column Settings (edit if needed)")
    target_risk_level = st.text_input("Target (Risk Level) column", CFG.target_risk_level)
    target_risk_type = st.text_input("Target (Risk_Type) column", CFG.target_risk_type)
    risk_score_col = st.text_input("Risk score column", CFG.risk_score_col)
    mp_count_col = st.text_input("MP count column", CFG.mp_count_col)
    polymer_type_col = st.text_input("Polymer type column", CFG.polymer_type_col)

    st.divider()
    st.header("⚙️ App Mode")
    enable_train_inside_app = st.toggle(
        "Enable train/CV inside app (slower)", value=False,
        help="Recommended OFF. Prefer loading trained artifacts from /artifacts."
    )

    st.caption("Artifacts folder expected: ./artifacts/")
    show_debug = st.toggle("Show debug info", value=False)


if uploaded is None:
    st.info("Upload your CSV to begin.")
    st.stop()

df = load_dataset_from_upload(uploaded)

# ----------------------------
# Artifact Loading
# ----------------------------
preprocessor = load_joblib_artifact(_artifact(CFG.preprocessor_path))
model_obj1 = load_joblib_artifact(_artifact(CFG.model_path))
model_obj2 = load_joblib_artifact(_artifact(CFG.risk_type_model_path))
selected_features = load_json(_artifact(CFG.selected_features_path))
feature_relevance_df = load_csv_artifact(_artifact(CFG.feature_relevance_path))
cv_results_df = load_csv_artifact(_artifact(CFG.cv_results_path))
risk_type_cv_results_df = load_csv_artifact(_artifact(CFG.risk_type_cv_results_path))

if show_debug:
    st.subheader("Debug: Dataset Preview")
    st.write(df.head())
    st.write("Columns:", list(df.columns))
    st.write("Artifacts loaded:", {
        "preprocessor": preprocessor is not None,
        "model_obj1": model_obj1 is not None,
        "model_obj2": model_obj2 is not None,
        "selected_features": selected_features is not None,
        "feature_relevance_df": feature_relevance_df is not None,
        "cv_results_df": cv_results_df is not None,
        "risk_type_cv_results_df": risk_type_cv_results_df is not None,
    })


# ----------------------------
# Tabs
# ----------------------------
tabs = st.tabs([
    "1) Overview",
    "2) EDA (Objective 1)",
    "3) Preprocessing (Objective 1)",
    "4) Feature Selection & Relevance (Objective 2)",
    "5) Modeling Results (Objective 2)",
    "6) Validation / Cross-Validation (Objective 3)",
    "7) Predict",
])

# ----------------------------
# Tab 1: Overview
# ----------------------------
with tabs[0]:
    st.subheader("Dataset Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{df.shape[0]:,}")
    c2.metric("Columns", f"{df.shape[1]:,}")
    c3.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")
    c4.metric("Duplicate rows", f"{int(df.duplicated().sum()):,}")

    with st.expander("Show column types"):
        st.write(pd.DataFrame({
            "column": df.columns,
            "dtype": [str(t) for t in df.dtypes],
            "missing": df.isna().sum().values
        }))

    st.markdown("### Targets availability")
    tcols = st.columns(2)
    if target_risk_level in df.columns:
        tcols[0].success(f"Found target column (Risk Level): `{target_risk_level}`")
        tcols[0].write(df[target_risk_level].value_counts(dropna=False).head(20))
    else:
        tcols[0].warning(f"Missing `{target_risk_level}`")

    if target_risk_type in df.columns:
        tcols[1].success(f"Found target column (Risk_Type): `{target_risk_type}`")
        tcols[1].write(df[target_risk_type].value_counts(dropna=False).head(20))
    else:
        tcols[1].warning(f"Missing `{target_risk_type}`")

# ----------------------------
# Tab 2: EDA (Objective 1)
# ----------------------------
with tabs[1]:
    st.subheader("EDA: Risk Score, Risk Level, MP Count, Polymer Type")

    left, right = st.columns(2)

    with left:
        if risk_score_col in df.columns:
            make_hist(df[risk_score_col], f"Distribution of {risk_score_col}")
        else:
            st.warning(f"Column not found: `{risk_score_col}`")

        if mp_count_col in df.columns and risk_score_col in df.columns:
            make_scatter(df, mp_count_col, risk_score_col, f"{risk_score_col} vs {mp_count_col}")
        else:
            st.info("Scatter plot needs both risk score and mp count columns.")

    with right:
        if risk_score_col in df.columns and target_risk_level in df.columns:
            make_box_by_group(df, risk_score_col, target_risk_level, f"{risk_score_col} by {target_risk_level}")
        else:
            st.info("Box plot needs risk score + risk level columns.")

        if polymer_type_col in df.columns:
            counts = df[polymer_type_col].value_counts(dropna=False).head(30)
            fig, ax = plt.subplots()
            ax.bar(counts.index.astype(str), counts.values)
            ax.set_title(f"Top Polymer Types: {polymer_type_col}")
            ax.set_xlabel(polymer_type_col)
            ax.set_ylabel("Count")
            plt.xticks(rotation=45, ha="right")
            st.pyplot(fig, clear_figure=True)
        else:
            st.info("Polymer type column not found (optional).")

    st.divider()
    st.subheader("Outlier quick check (IQR) — Risk Score")
    if risk_score_col in df.columns:
        s = df[risk_score_col].dropna()
        if len(s) > 10:
            q1, q3 = np.percentile(s, [25, 75])
            iqr = q3 - q1
            low = q1 - 1.5 * iqr
            high = q3 + 1.5 * iqr
            outliers = ((df[risk_score_col] < low) | (df[risk_score_col] > high)).sum()
            st.write(f"IQR bounds: [{low:.4f}, {high:.4f}]")
            st.write(f"Flagged outliers: {int(outliers):,}")
        else:
            st.info("Not enough values to compute IQR.")
    else:
        st.warning(f"Column not found: `{risk_score_col}`")

# ----------------------------
# Tab 3: Preprocessing (Objective 1)
# ----------------------------
with tabs[2]:
    st.subheader("Preprocessing")
    st.markdown(
        "Recommended: apply the same preprocessing used during training via a saved `preprocessor.pkl`."
    )

    if preprocessor is None:
        st.warning(
            "No `preprocessor.pkl` found in ./artifacts. "
            "You can still explore EDA, but prediction and consistent transforms require the preprocessor."
        )
    else:
        st.success("Loaded preprocessor artifact ✅")

    with st.expander("Preview: Preprocessed feature matrix (first 5 rows)"):
        if preprocessor is None:
            st.info("Upload a preprocessor artifact to see transformed features.")
        else:
            # Try to build X by dropping known targets if present
            drop_cols = [c for c in [target_risk_level, target_risk_type] if c in df.columns]
            X_raw = df.drop(columns=drop_cols, errors="ignore")
            try:
                Xp = preprocessor.transform(X_raw)
                # If it's sparse, convert small preview only
                if hasattr(Xp, "toarray"):
                    Xp_preview = Xp[:5].toarray()
                else:
                    Xp_preview = np.asarray(Xp[:5])
                st.write(pd.DataFrame(Xp_preview))
            except Exception as e:
                st.error(f"Preprocessing failed. Ensure your uploaded dataset matches training schema.\n\n{e}")

# ----------------------------
# Tab 4: Feature Selection & Relevance (Objective 2)
# ----------------------------
with tabs[3]:
    st.subheader("Feature Selection & Feature Relevance")
    if selected_features is not None:
        st.success("Loaded selected features ✅")
        st.write(selected_features)
    else:
        st.info("No selected_features.json found in ./artifacts (optional).")

    if feature_relevance_df is not None and not feature_relevance_df.empty:
        st.success("Loaded feature relevance ✅")
        st.dataframe(feature_relevance_df.head(50), use_container_width=True)

        # Expecting columns like: feature, importance
        cols = [c.lower() for c in feature_relevance_df.columns]
        if "feature" in cols and ("importance" in cols or "relevance" in cols):
            fcol = feature_relevance_df.columns[cols.index("feature")]
            icol = feature_relevance_df.columns[cols.index("importance")] if "importance" in cols else feature_relevance_df.columns[cols.index("relevance")]
            topn = st.slider("Top N features to plot", 5, 30, 15)
            top = feature_relevance_df.sort_values(icol, ascending=False).head(topn)

            fig, ax = plt.subplots()
            ax.barh(top[fcol].astype(str), top[icol].values)
            ax.set_title("Top Feature Relevance")
            ax.set_xlabel(icol)
            ax.invert_yaxis()
            st.pyplot(fig, clear_figure=True)
        else:
            st.caption("To auto-plot, provide columns named `feature` and `importance` (or `relevance`).")
    else:
        st.info("No feature_relevance.csv found in ./artifacts (optional).")

# ----------------------------
# Tab 5: Modeling Results (Objective 2)
# ----------------------------
with tabs[4]:
    st.subheader("Modeling Results (Artifacts)")
    st.markdown(
        "This section assumes you already trained models offline and saved them in `./artifacts`."
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Objective #1 Model (Risk Level)")
        st.write("Loaded:", bool(model_obj1))
        if model_obj1 is None:
            st.info("Add `model.pkl` to ./artifacts to enable predictions/evaluation.")
    with c2:
        st.markdown("#### Objective #2 Model (Risk_Type)")
        st.write("Loaded:", bool(model_obj2))
        if model_obj2 is None:
            st.info("Add `risk_type_model.pkl` to ./artifacts to enable predictions/evaluation.")

    st.divider()

    if enable_train_inside_app:
        st.warning("Train inside app is ON. This is slower and mainly for demos.")
        st.markdown("#### Quick train/evaluate (holdout split)")
        target_choice = st.selectbox("Choose target to train", [target_risk_level, target_risk_type])
        test_size = st.slider("Test size", 0.1, 0.4, 0.2, 0.05)
        random_state = st.number_input("Random state", value=42, step=1)

        if st.button("Train & Evaluate (Holdout)"):
            if target_choice not in df.columns:
                st.error(f"Target column not found: {target_choice}")
            elif preprocessor is None:
                st.error("Need preprocessor.pkl to train inside app consistently.")
            else:
                # Basic pipeline assumption: you trained a *full pipeline* offline.
                # Here we only demonstrate evaluating a loaded model if it supports fit.
                # For proper in-app training, you'd need a model pipeline builder.
                st.error(
                    "In-app training requires you to define model pipelines (preprocess + model). "
                    "Recommended: train offline and load artifacts."
                )
    else:
        st.info("Tip: keep training/tuning offline; display results here + predictions in the next tab.")

# ----------------------------
# Tab 6: Validation / Cross-Validation (Objective 3)
# ----------------------------
with tabs[5]:
    st.subheader("Validation / Cross-Validation (Objective 3)")
    st.markdown(
        "Best practice: run CV on TRAIN data only during development, then save results as CSV for reporting."
    )

    left, right = st.columns(2)
    with left:
        st.markdown("#### CV Results — Objective #1 (Risk Level)")
        if cv_results_df is not None and not cv_results_df.empty:
            st.dataframe(cv_results_df, use_container_width=True)
        else:
            st.info("No cv_results.csv found in ./artifacts.")

    with right:
        st.markdown("#### CV Results — Objective #2 (Risk_Type)")
        if risk_type_cv_results_df is not None and not risk_type_cv_results_df.empty:
            st.dataframe(risk_type_cv_results_df, use_container_width=True)
        else:
            st.info("No risk_type_cv_results.csv found in ./artifacts.")

    st.divider()
    st.markdown("#### Optional: Run quick CV inside app (Objective 3)")

    if not enable_train_inside_app:
        st.caption("Turn ON 'Enable train/CV inside app' in the sidebar to run CV here (slower).")
    else:
        st.warning("Make sure your pipeline includes preprocessing (and SMOTE inside folds if used).")

        # NOTE: This is only a template for CV evaluation of an already-built pipeline.
        st.info(
            "Template placeholder: to run CV inside the app, you must load/build a full sklearn Pipeline "
            "(preprocessor + model). This app currently expects you to compute CV offline and save CSVs."
        )

# ----------------------------
# Tab 7: Predict
# ----------------------------
with tabs[6]:
    st.subheader("Predict")
    st.markdown(
        "Use trained artifacts to generate predictions. "
        "Ensure the uploaded dataset has the same schema used during training."
    )

    if preprocessor is None:
        st.error("Missing preprocessor.pkl — required for consistent prediction.")
        st.stop()

    pred_mode = st.radio("Prediction mode", ["Batch (whole dataset)", "Single row (form)"], horizontal=True)

    # Decide which model to use
    model_choice = st.selectbox(
        "Choose prediction target/model",
        [
            ("Objective #1: Risk Level", "obj1"),
            ("Objective #2: Risk_Type", "obj2"),
        ],
        format_func=lambda x: x[0]
    )[1]

    model = model_obj1 if model_choice == "obj1" else model_obj2
    target_name = target_risk_level if model_choice == "obj1" else target_risk_type

    if model is None:
        st.error("Selected model artifact not found. Please add the model file to ./artifacts.")
        st.stop()

    # Build X raw by dropping target if present
    drop_cols = [c for c in [target_risk_level, target_risk_type] if c in df.columns]
    X_raw_all = df.drop(columns=drop_cols, errors="ignore")

    if pred_mode == "Batch (whole dataset)":
        if st.button("Run Batch Prediction"):
            try:
                Xp = preprocessor.transform(X_raw_all)

                # Feature selection (optional)
                if selected_features and isinstance(selected_features, dict) and "columns" in selected_features:
                    # This only works if your preprocessor outputs a DataFrame with feature names.
                    # If it outputs numpy/sparse matrix, feature selection must be embedded in pipeline offline.
                    st.warning(
                        "selected_features.json detected, but feature selection should be embedded in the saved pipeline "
                        "for reliable use with matrix outputs."
                    )

                # Predict
                y_pred = model.predict(Xp)

                out = df.copy()
                out[f"pred_{target_name}"] = y_pred

                st.success("Prediction complete ✅")
                st.dataframe(out.head(50), use_container_width=True)

                # Download
                csv = out.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download predictions as CSV",
                    data=csv,
                    file_name="predictions.csv",
                    mime="text/csv",
                )
            except Exception as e:
                st.error(f"Prediction failed.\n\n{e}")

    else:
        st.markdown("Fill in the feature values below (for columns in your dataset).")

        # Build a form using dataframe columns
        feature_cols = list(X_raw_all.columns)

        if not feature_cols:
            st.error("No feature columns found after dropping targets.")
            st.stop()

        with st.form("single_row_form"):
            inputs = {}
            # Make it manageable: show first N then expand
            N = min(20, len(feature_cols))
            c1, c2 = st.columns(2)
            for i, col in enumerate(feature_cols[:N]):
                col_series = X_raw_all[col]
                is_num = pd.api.types.is_numeric_dtype(col_series)
                container = c1 if i % 2 == 0 else c2

                with container:
                    if is_num:
                        # Use median as default
                        default = float(np.nanmedian(col_series.values)) if np.isfinite(np.nanmedian(col_series.values)) else 0.0
                        inputs[col] = st.number_input(col, value=default)
                    else:
                        options = [str(x) for x in col_series.dropna().unique()[:200]]
                        default = options[0] if options else ""
                        inputs[col] = st.selectbox(col, options=options if options else [""], index=0)

            with st.expander("More columns (if any)"):
                for col in feature_cols[N:]:
                    col_series = X_raw_all[col]
                    is_num = pd.api.types.is_numeric_dtype(col_series)
                    if is_num:
                        default = float(np.nanmedian(col_series.values)) if np.isfinite(np.nanmedian(col_series.values)) else 0.0
                        inputs[col] = st.number_input(col, value=default)
                    else:
                        options = [str(x) for x in col_series.dropna().unique()[:200]]
                        inputs[col] = st.selectbox(col, options=options if options else [""], index=0)

            submitted = st.form_submit_button("Predict")

        if submitted:
            try:
                row = pd.DataFrame([inputs])
                Xp = preprocessor.transform(row)
                pred = model.predict(Xp)[0]

                st.success(f"Predicted `{target_name}`: **{pred}**")

                # If probability exists
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(Xp)[0]
                    st.write("Class probabilities:")
                    st.write(pd.DataFrame({"class": model.classes_, "probability": proba}).sort_values("probability", ascending=False))
            except Exception as e:
                st.error(f"Single-row prediction failed.\n\n{e}")


# ----------------------------
# Footer
# ----------------------------
st.divider()
st.caption(
    "Tip: For best results, train offline and save artifacts to ./artifacts: "
    "preprocessor.pkl, model.pkl, risk_type_model.pkl, selected_features.json, feature_relevance.csv, cv_results.csv."
)
