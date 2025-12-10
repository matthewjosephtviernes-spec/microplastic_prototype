# app.py
# Streamlit Dashboard: Predictive Risk Modeling Framework for Microplastic Pollution
# Fixes:
# - Robust CSV loader (empty file + separator/encoding retries)
# - Column resolver (case/space/symbol insensitive) so Risk_Score + Mp_Count always match

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from difflib import get_close_matches
from pathlib import Path
from typing import Optional, Dict

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from pandas.errors import EmptyDataError, ParserError

try:
    import joblib
except Exception:
    joblib = None


# ----------------------------
# Config (EDIT THESE IF NEEDED)
# ----------------------------
@dataclass(frozen=True)
class AppConfig:
    target_risk_level: str = "Risk_Level"
    risk_score_col: str = "Risk_Score"
    mp_count_col: str = "Mp_Count"
    polymer_type_col: str = "Polymer_Type"
    target_risk_type: str = "Risk_Type"  # optional

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
# Page setup
# ----------------------------
st.set_page_config(page_title="Microplastic Risk Modeling Dashboard", page_icon="🧪", layout="wide")
st.title("🧪 Predictive Risk Modeling Framework for Microplastic Pollution")
st.caption("EDA • Preprocessing • Feature Selection • Modeling • Cross-Validation • Prediction")


# ----------------------------
# Helpers
# ----------------------------
def _artifact(p: str) -> Path:
    return Path(CFG.artifacts_dir) / p


def _safe_read_csv(uploaded_file) -> pd.DataFrame:
    try:
        raw = uploaded_file.getvalue()
        if raw is None or len(raw) == 0:
            raise EmptyDataError("Uploaded file is empty (0 bytes).")
    except Exception:
        pass

    def _rewind():
        try:
            uploaded_file.seek(0)
        except Exception:
            pass

    encodings = ["utf-8", "utf-8-sig", "latin1"]
    seps = [",", ";", "\t", "|"]
    last_err: Optional[Exception] = None

    for enc in encodings:
        for sep in seps:
            try:
                _rewind()
                return pd.read_csv(uploaded_file, encoding=enc, sep=sep)
            except (EmptyDataError, ParserError, UnicodeDecodeError) as e:
                last_err = e
                continue

    try:
        _rewind()
        return pd.read_csv(uploaded_file, engine="python", encoding="utf-8", sep=None)
    except Exception as e:
        last_err = e

    raise last_err if last_err else EmptyDataError("Unable to read uploaded CSV.")


@st.cache_data(show_spinner=False)
def load_dataset_from_upload(uploaded_file) -> pd.DataFrame:
    df = _safe_read_csv(uploaded_file)
    # Strip column names + normalize internal whitespace
    df.columns = [" ".join(str(c).strip().split()) for c in df.columns]
    if df.shape[1] == 0:
        raise EmptyDataError("Parsed file has no columns. Check delimiter/format.")
    return df


def _norm(s: str) -> str:
    # case/space/symbol insensitive
    return re.sub(r"[^a-z0-9]+", "", str(s).strip().lower())


def resolve_column(df: pd.DataFrame, desired: str) -> Optional[str]:
    """Return the actual column name in df that best matches desired."""
    if desired in df.columns:
        return desired

    norm_map: Dict[str, str] = {c: _norm(c) for c in df.columns}
    desired_norm = _norm(desired)

    # exact normalized match
    for c, n in norm_map.items():
        if n == desired_norm:
            return c

    # fuzzy match on normalized values
    close = get_close_matches(desired_norm, list(norm_map.values()), n=1, cutoff=0.70)
    if close:
        best_norm = close[0]
        for c, n in norm_map.items():
            if n == best_norm:
                return c

    return None


@st.cache_resource(show_spinner=False)
def load_joblib_artifact(path: Path):
    if not path.exists():
        return None
    if joblib is None:
        st.warning("joblib not installed. Add it to requirements.txt: joblib")
        return None
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"Failed to load artifact: {path}\n\n{e}")
        return None


@st.cache_data(show_spinner=False)
def load_json(path: Path):
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
    groups, labels = [], []
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


# ----------------------------
# Sidebar
# ----------------------------
with st.sidebar:
    st.header("📦 Data Input")
    uploaded = st.file_uploader("Upload CSV dataset", type=["csv"])

    st.divider()
    st.header("🧩 Column Settings (you can type your expected names)")
    target_risk_level_in = st.text_input("Target (Risk Level)", CFG.target_risk_level)
    risk_score_in = st.text_input("Risk score", CFG.risk_score_col)
    mp_count_in = st.text_input("MP count", CFG.mp_count_col)
    polymer_type_in = st.text_input("Polymer type", CFG.polymer_type_col)
    target_risk_type_in = st.text_input("Target (Risk_Type) (optional)", CFG.target_risk_type)

    st.divider()
    show_debug = st.toggle("Show debug info", value=False)


if uploaded is None:
    st.info("Upload your CSV to begin.")
    st.stop()

try:
    df = load_dataset_from_upload(uploaded)
except EmptyDataError:
    st.error("Your uploaded file looks empty or unreadable as CSV. Please re-upload a valid CSV export.")
    st.stop()
except Exception as e:
    st.error(f"Failed to load CSV: {e}")
    st.stop()

# Resolve actual column names in the uploaded file
target_risk_level = resolve_column(df, target_risk_level_in)
risk_score_col = resolve_column(df, risk_score_in)
mp_count_col = resolve_column(df, mp_count_in)
polymer_type_col = resolve_column(df, polymer_type_in)
target_risk_type = resolve_column(df, target_risk_type_in)

if show_debug:
    st.subheader("Debug: Column Resolution")
    st.write("Available columns:", list(df.columns))
    st.write({
        "Risk_Level (typed)": target_risk_level_in,
        "Risk_Level (resolved)": target_risk_level,
        "Risk_Score (typed)": risk_score_in,
        "Risk_Score (resolved)": risk_score_col,
        "Mp_Count (typed)": mp_count_in,
        "Mp_Count (resolved)": mp_count_col,
        "Polymer_Type (typed)": polymer_type_in,
        "Polymer_Type (resolved)": polymer_type_col,
        "Risk_Type (typed)": target_risk_type_in,
        "Risk_Type (resolved)": target_risk_type,
    })

# ----------------------------
# Artifacts
# ----------------------------
preprocessor = load_joblib_artifact(_artifact(CFG.preprocessor_path))
model_obj1 = load_joblib_artifact(_artifact(CFG.model_path))
model_obj2 = load_joblib_artifact(_artifact(CFG.risk_type_model_path))
selected_features = load_json(_artifact(CFG.selected_features_path))
feature_relevance_df = load_csv_artifact(_artifact(CFG.feature_relevance_path))
cv_results_df = load_csv_artifact(_artifact(CFG.cv_results_path))
risk_type_cv_results_df = load_csv_artifact(_artifact(CFG.risk_type_cv_results_path))

# ----------------------------
# Tabs
# ----------------------------
tabs = st.tabs([
    "1) Overview",
    "2) EDA (Objective 1)",
    "3) Preprocessing (Objective 1)",
    "4) Feature Selection & Relevance (Objective 2)",
    "5) Validation / Cross-Validation (Objective 3)",
    "6) Predict",
])

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

with tabs[1]:
    st.subheader("EDA: Risk Score, Risk Level, MP Count, Polymer Type")

    left, right = st.columns(2)

    with left:
        if risk_score_col:
            make_hist(df[risk_score_col], f"Distribution of {risk_score_col}")
        else:
            st.warning(f"Risk score column not found. You typed: `{risk_score_in}`")

        if risk_score_col and mp_count_col:
            make_scatter(df, mp_count_col, risk_score_col, f"{risk_score_col} vs {mp_count_col}")
        else:
            st.info(
                f"Scatter needs both columns. "
                f"Resolved risk score: `{risk_score_col}` | resolved MP count: `{mp_count_col}`"
            )

    with right:
        if risk_score_col and target_risk_level:
            make_box_by_group(df, risk_score_col, target_risk_level, f"{risk_score_col} by {target_risk_level}")
        else:
            st.info(
                f"Box plot needs risk score + risk level. "
                f"Resolved risk score: `{risk_score_col}` | resolved risk level: `{target_risk_level}`"
            )

        if polymer_type_col:
            counts = df[polymer_type_col].value_counts(dropna=False).head(30)
            fig, ax = plt.subplots()
            ax.bar(counts.index.astype(str), counts.values)
            ax.set_title(f"Top Polymer Types: {polymer_type_col}")
            ax.set_xlabel(polymer_type_col)
            ax.set_ylabel("Count")
            plt.xticks(rotation=45, ha="right")
            st.pyplot(fig, clear_figure=True)
        else:
            st.info(f"Polymer type column not found (typed: `{polymer_type_in}`).")

with tabs[2]:
    st.subheader("Preprocessing")
    st.markdown("Recommended: apply the same preprocessing used during training via saved `preprocessor.pkl`.")

    if preprocessor is None:
        st.warning("No `preprocessor.pkl` found in ./artifacts.")
    else:
        st.success("Loaded preprocessor ✅")

    with st.expander("Preview: Preprocessed feature matrix (first 5 rows)"):
        if preprocessor is None:
            st.info("Upload a preprocessor artifact to see transformed features.")
        else:
            drop_cols = [c for c in [target_risk_level, target_risk_type] if c and c in df.columns]
            X_raw = df.drop(columns=drop_cols, errors="ignore")
            try:
                Xp = preprocessor.transform(X_raw)
                if hasattr(Xp, "toarray"):
                    Xp_preview = Xp[:5].toarray()
                else:
                    Xp_preview = np.asarray(Xp[:5])
                st.write(pd.DataFrame(Xp_preview))
            except Exception as e:
                st.error(f"Preprocessing failed.\n\n{e}")

with tabs[3]:
    st.subheader("Feature Selection & Feature Relevance")

    if selected_features is not None:
        st.success("Loaded selected_features.json ✅")
        st.write(selected_features)
    else:
        st.info("No selected_features.json found (optional).")

    if feature_relevance_df is not None and not feature_relevance_df.empty:
        st.success("Loaded feature_relevance.csv ✅")
        st.dataframe(feature_relevance_df.head(50), use_container_width=True)
    else:
        st.info("No feature_relevance.csv found (optional).")

with tabs[4]:
    st.subheader("Validation / Cross-Validation (Objective 3)")

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

with tabs[5]:
    st.subheader("Predict")

    if preprocessor is None:
        st.error("Missing preprocessor.pkl — required for prediction.")
        st.stop()

    model_choice = st.selectbox(
        "Choose prediction target/model",
        [("Objective #1: Risk Level", "obj1"), ("Objective #2: Risk_Type", "obj2")],
        format_func=lambda x: x[0],
    )[1]

    model = model_obj1 if model_choice == "obj1" else model_obj2
    target_name = target_risk_level if model_choice == "obj1" else target_risk_type

    if model is None:
        st.error("Selected model artifact not found in ./artifacts.")
        st.stop()

    drop_cols = [c for c in [target_risk_level, target_risk_type] if c and c in df.columns]
    X_raw_all = df.drop(columns=drop_cols, errors="ignore")

    if st.button("Run Batch Prediction"):
        try:
            Xp = preprocessor.transform(X_raw_all)
            y_pred = model.predict(Xp)
            out = df.copy()
            out[f"pred_{target_name or 'target'}"] = y_pred

            st.success("Prediction complete ✅")
            st.dataframe(out.head(50), use_container_width=True)

            csv = out.to_csv(index=False).encode("utf-8")
            st.download_button("Download predictions as CSV", data=csv, file_name="predictions.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Prediction failed.\n\n{e}")

st.divider()
st.caption("If plots still don’t show, toggle 'Show debug info' to see the resolved column names.")
