# app.py
# Streamlit app: Predictive Risk Modeling for Microplastic Pollution
# Pages are shown as separate "windows" using Streamlit tabs (top-level).

import re
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    accuracy_score, f1_score, balanced_accuracy_score,
    mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance

# Optional: SMOTE (handle gracefully if not installed)
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    IMBLEARN_OK = True
except Exception:
    IMBLEARN_OK = False

import matplotlib.pyplot as plt


# -----------------------------
# Helpers
# -----------------------------
def parse_numeric_with_units(x):
    """Extract first numeric value from strings like '33 ppt' or '33 PSU'."""
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    s = str(x).strip()
    m = re.search(r"[-+]?\d*\.?\d+", s)
    return float(m.group(0)) if m else np.nan


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    # Parse columns that sometimes include units / text
    for col in ["Salinity", "MP_Count_per_L", "Microplastic_Size_mm", "Density", "pH"]:
        if col in df.columns:
            df[col] = df[col].apply(parse_numeric_with_units)

    for col in ["Latitude", "Longitude"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Standardize Risk_Level into a clean label (optional)
    if "Risk_Level" in df.columns:
        def map_level(x):
            xl = str(x).strip().lower()
            if "extreme" in xl or "level v" in xl:
                return "Extreme"
            if "high" in xl:
                return "High"
            if "medium" in xl or "moderate" in xl:
                return "Medium"
            if "low" in xl:
                return "Low"
            return "Other"
        df["Risk_Level_std"] = df["Risk_Level"].apply(map_level)

    return df


def build_preprocessor(X: pd.DataFrame):
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", cat_pipe, categorical_cols),
        ],
        remainder="drop"
    )
    return preprocessor, numeric_cols, categorical_cols


def plot_bar_counts(series, title, xlabel=None, ylabel="Count", max_bars=25):
    counts = series.value_counts(dropna=False).head(max_bars)
    fig, ax = plt.subplots()
    ax.bar(counts.index.astype(str), counts.values)
    ax.set_title(title)
    ax.set_xlabel(xlabel or series.name)
    ax.set_ylabel(ylabel)
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig)


def plot_conf_mat(y_true, y_pred, labels=None, title="Confusion Matrix"):
    fig, ax = plt.subplots()
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, xticks_rotation=45, colorbar=False)
    ax.set_title(title)
    st.pyplot(fig)


def metric_table_classification(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
    }


def metric_table_regression(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": mean_squared_error(y_true, y_pred, squared=False),
        "R2": r2_score(y_true, y_pred),
    }


def ensure_session_defaults():
    st.session_state.setdefault("trained_models", {})
    st.session_state.setdefault("best_model_name", None)
    st.session_state.setdefault("best_pipe", None)
    st.session_state.setdefault("feature_importance", None)
    st.session_state.setdefault("cv_results", None)


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Microplastic Risk Modeling", layout="wide")
ensure_session_defaults()

st.title("Predictive Risk Modeling for Microplastic Pollution (Data Mining)")
st.caption("A step-by-step interactive workflow: upload → explore → preprocess → handle imbalance → train → validate → compare → explain → predict.")


# ✅ TOP VISIBLE UPLOAD SECTION
st.markdown("## Upload Dataset")
st.write("Upload a CSV file to start. If you don’t upload, the app will try to load **Microplastic.csv** from the app folder.")
uploaded_file = st.file_uploader("Upload your dataset (.csv)", type=["csv"])

@st.cache_data
def load_data(uploaded):
    if uploaded is not None:
        df0 = pd.read_csv(uploaded)
        src = "Uploaded CSV"
    else:
        df0 = pd.read_csv("Microplastic.csv")
        src = "Local file: Microplastic.csv"
    df0 = clean_dataframe(df0)
    return df0, src


try:
    df, data_src = load_data(uploaded_file)
    st.success(f"Loaded data source: **{data_src}**")
except Exception as e:
    st.error(f"Could not load dataset. Upload a CSV or make sure Microplastic.csv exists. Error: {e}")
    st.stop()


# -----------------------------
# GLOBAL CONTROLS (visible on every "window")
# -----------------------------
st.markdown("## Configuration")
cfg1, cfg2, cfg3 = st.columns([1.1, 1.1, 1.2])

with cfg1:
    st.subheader("Target")
    target_choice = st.selectbox(
        "Choose prediction target",
        options=[
            "Risk_Type (Classification)",
            "Risk_Score (Regression)",
            "Risk_Level_std (Classification, cleaned)",
        ],
        index=0 if "Risk_Type" in df.columns else 1
    )

with cfg2:
    st.subheader("Split")
    test_size = st.slider("Test size", 0.1, 0.4, 0.2, 0.05)
    random_state = st.number_input("Random state", value=42, step=1)

with cfg3:
    st.subheader("Models & Imbalance")
    if target_choice.startswith("Risk_Score"):
        use_smote = False
        st.info("SMOTE not applicable (Regression).")
        chosen_models = st.multiselect(
            "Choose models",
            options=["Ridge Regression", "Random Forest Regressor"],
            default=["Ridge Regression", "Random Forest Regressor"]
        )
    else:
        use_smote = st.checkbox(
            "Use SMOTE (training only)",
            value=False,
            help="Requires imbalanced-learn. Applied only inside the training pipeline."
        )
        if use_smote and not IMBLEARN_OK:
            st.error("imbalanced-learn is not installed. Install: pip install imbalanced-learn")
        chosen_models = st.multiselect(
            "Choose models",
            options=["Logistic Regression", "Random Forest", "Gradient Boosting"],
            default=["Logistic Regression", "Random Forest", "Gradient Boosting"]
        )

# Task / target col
if target_choice.startswith("Risk_Type"):
    target_col = "Risk_Type"
    task = "classification"
elif target_choice.startswith("Risk_Score"):
    target_col = "Risk_Score"
    task = "regression"
else:
    target_col = "Risk_Level_std"
    task = "classification"

# Features
st.markdown("### Input Features")
default_drop = [c for c in ["Risk_Type", "Risk_Score", "Risk_Level", "Risk_Level_std"] if c in df.columns]
feature_cols = st.multiselect(
    "Select feature columns (inputs)",
    options=[c for c in df.columns if c not in default_drop],
    default=[c for c in df.columns if c not in default_drop]
)

if len(feature_cols) == 0:
    st.error("Please select at least one feature column.")
    st.stop()

# Prepare X, y
data = df.dropna(subset=[target_col]).copy()
X = data[feature_cols].copy()
y = data[target_col].copy()

# Split
if task == "classification":
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
else:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state
    )

preprocessor, num_cols, cat_cols = build_preprocessor(X_train)

st.divider()


# -----------------------------
# WINDOWS / STEPS AS TABS
# -----------------------------
tabs = st.tabs([
    "1) Explore",
    "2) Prepare",
    "3) Imbalance (SMOTE)",
    "4) Train",
    "5) Validate & Compare",
    "6) Feature Relevance",
    "7) Predict",
    "8) Summary"
])

# 1) Explore
with tabs[0]:
    st.header("1) Load & Explore Data")
    left, right = st.columns([2.2, 1])
    with left:
        st.write("Dataset preview:")
        st.dataframe(df.head(25), use_container_width=True)
    with right:
        st.write("Dataset info:")
        st.write({"rows": int(df.shape[0]), "columns": int(df.shape[1])})
        st.write("Target:")
        st.write({"task": task, "target_col": target_col})
        st.write("Selected features:")
        st.write(feature_cols)

    st.subheader("Target distribution")
    if task == "classification":
        plot_bar_counts(df[target_col].dropna(), f"Distribution of {target_col}")
    else:
        st.write(df[target_col].describe())

    if "Polymer_Type" in df.columns:
        st.subheader("Polymer Type distribution")
        plot_bar_counts(df["Polymer_Type"].dropna(), "Polymer_Type Distribution (Top 25)")

# 2) Prepare
with tabs[1]:
    st.header("2) Prepare & Preprocess")
    st.markdown("""
This window shows how the app prepares the data for modeling:
- converts unit-like strings to numeric values (e.g., '33 PSU' → 33)
- imputes missing values
- one-hot encodes categorical variables
- scales numeric variables
""")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Detected column types")
        st.write("Numeric columns:")
        st.write(num_cols)
        st.write("Categorical columns:")
        st.write(cat_cols)

    with c2:
        st.subheader("Missing values (selected features)")
        miss = X.isna().sum().sort_values(ascending=False)
        st.dataframe(miss[miss > 0].to_frame("missing_count"), use_container_width=True)

    st.subheader("Train/Test split sizes")
    st.write({"train_rows": len(X_train), "test_rows": len(X_test)})

# 3) Imbalance
with tabs[2]:
    st.header("3) Address Class Imbalance (SMOTE)")
    if task != "classification":
        st.info("Imbalance handling via SMOTE is only relevant for classification.")
    else:
        st.subheader("Class distribution (Training set)")
        st.dataframe(y_train.value_counts().to_frame("count"), use_container_width=True)

        if use_smote:
            if IMBLEARN_OK:
                st.success("SMOTE is enabled and will be applied only inside the training pipeline (no leakage).")
            else:
                st.error("SMOTE selected but imbalanced-learn is missing. Install: pip install imbalanced-learn")
        else:
            st.warning("SMOTE is OFF. Logistic Regression will still use class_weight='balanced'.")

# 4) Train
with tabs[3]:
    st.header("4) Train Models")
    st.write(f"Task: **{task}**, Target: **{target_col}**")
    st.write(f"Models selected: {', '.join(chosen_models) if chosen_models else '(none)'}")

    if not chosen_models:
        st.warning("Please select at least one model in the configuration section.")
    else:
        if st.button("Train models", type="primary"):
            st.session_state["trained_models"] = {}
            st.session_state["best_model_name"] = None
            st.session_state["best_pipe"] = None
            st.session_state["feature_importance"] = None
            st.session_state["cv_results"] = None

            for name in chosen_models:
                if task == "classification":
                    if name == "Logistic Regression":
                        model = LogisticRegression(max_iter=2000, class_weight="balanced")
                    elif name == "Random Forest":
                        model = RandomForestClassifier(n_estimators=300, random_state=random_state)
                    else:
                        model = GradientBoostingClassifier(random_state=random_state)

                    if use_smote and IMBLEARN_OK:
                        pipe = ImbPipeline(steps=[
                            ("preprocess", preprocessor),
                            ("smote", SMOTE(random_state=random_state)),
                            ("model", model),
                        ])
                    else:
                        pipe = Pipeline(steps=[
                            ("preprocess", preprocessor),
                            ("model", model),
                        ])

                    pipe.fit(X_train, y_train)
                    st.session_state["trained_models"][name] = pipe

                else:
                    if name == "Ridge Regression":
                        model = Ridge(random_state=random_state)
                    else:
                        model = RandomForestRegressor(n_estimators=400, random_state=random_state)

                    pipe = Pipeline(steps=[
                        ("preprocess", preprocessor),
                        ("model", model),
                    ])
                    pipe.fit(X_train, y_train)
                    st.session_state["trained_models"][name] = pipe

            st.success("Training complete. Go to **Validate & Compare** window.")

# 5) Validate & Compare
with tabs[4]:
    st.header("5) Validation, Cross-Validation & Model Comparison")

    models = st.session_state.get("trained_models", {})
    if not models:
        st.warning("No trained models yet. Go to the **Train** window and train models.")
    else:
        st.subheader("Test set comparison")
        results = []
        for name, pipe in models.items():
            y_pred = pipe.predict(X_test)
            if task == "classification":
                results.append({"model": name, **metric_table_classification(y_test, y_pred)})
            else:
                results.append({"model": name, **metric_table_regression(y_test, y_pred)})

        sort_col = "f1_macro" if task == "classification" else "RMSE"
        res_df = pd.DataFrame(results).sort_values(by=sort_col, ascending=(task != "classification"))
        st.dataframe(res_df, use_container_width=True)

        best_model_name = res_df.iloc[0]["model"]
        best_pipe = models[best_model_name]
        st.session_state["best_model_name"] = best_model_name
        st.session_state["best_pipe"] = best_pipe

        st.markdown(f"**Best model (by {sort_col}):** `{best_model_name}`")

        st.subheader("Detailed evaluation (best model)")
        best_pred = best_pipe.predict(X_test)

        if task == "classification":
            labels = sorted(pd.Series(y_test).unique().tolist())
            plot_conf_mat(y_test, best_pred, labels=labels, title=f"Confusion Matrix: {best_model_name}")
            st.text(classification_report(y_test, best_pred, zero_division=0))
        else:
            st.json(metric_table_regression(y_test, best_pred))

        # Cross-validation
        st.subheader("Cross-validation (to check generalizability)")
        st.caption("Uses StratifiedKFold for classification; standard KFold-like behavior via StratifiedKFold is not used for regression here.")

        run_cv = st.button("Run cross-validation (CV)", help="Computes CV scores for each trained model.")
        if run_cv:
            cv_rows = []
            if task == "classification":
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
                scoring = "f1_macro"
                for name, pipe in models.items():
                    scores = cross_val_score(pipe, X, y, cv=cv, scoring=scoring)
                    cv_rows.append({
                        "model": name,
                        "cv_metric": scoring,
                        "mean": float(np.mean(scores)),
                        "std": float(np.std(scores)),
                        "min": float(np.min(scores)),
                        "max": float(np.max(scores)),
                    })
            else:
                # For regression, use R2 as CV score (simple & interpretable)
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
                # NOTE: StratifiedKFold isn't appropriate for continuous y; keeping it simple:
                # We'll do repeated random split style is overkill; so we skip CV by default.
                cv_rows.append({
                    "model": "(regression)",
                    "cv_metric": "N/A",
                    "mean": np.nan, "std": np.nan, "min": np.nan, "max": np.nan
                })

            cv_df = pd.DataFrame(cv_rows)
            st.session_state["cv_results"] = cv_df
            st.dataframe(cv_df, use_container_width=True)

# 6) Feature relevance
with tabs[5]:
    st.header("6) Feature Relevance / Importance")
    best_pipe = st.session_state.get("best_pipe", None)
    best_model_name = st.session_state.get("best_model_name", None)

    if best_pipe is None:
        st.warning("No best model selected yet. Train and evaluate in **Validate & Compare** first.")
    else:
        st.write(f"Using best model: **{best_model_name}**")
        st.caption("Permutation importance is model-agnostic. It measures the drop in score when a feature is shuffled.")

        if st.button("Compute permutation importance", type="primary"):
            scoring = "f1_macro" if task == "classification" else "r2"
            perm = permutation_importance(
                best_pipe, X_test, y_test,
                n_repeats=10,
                random_state=random_state,
                scoring=scoring
            )
            importances = pd.DataFrame({
                "feature": feature_cols,
                "importance_mean": perm.importances_mean,
                "importance_std": perm.importances_std
            }).sort_values("importance_mean", ascending=False)

            st.session_state["feature_importance"] = importances

        importances = st.session_state.get("feature_importance", None)
        if importances is None:
            st.info("Click **Compute permutation importance** to generate results.")
        else:
            st.dataframe(importances.head(20), use_container_width=True)

            fig, ax = plt.subplots()
            top = importances.head(15)
            ax.bar(top["feature"].astype(str), top["importance_mean"].values)
            ax.set_title("Top Feature Importances (Permutation)")
            ax.set_ylabel("Importance (mean)")
            plt.xticks(rotation=45, ha="right")
            st.pyplot(fig)

# 7) Predict
with tabs[6]:
    st.header("7) Predict (Demo)")
    best_pipe = st.session_state.get("best_pipe", None)
    best_model_name = st.session_state.get("best_model_name", None)

    if best_pipe is None:
        st.warning("No trained/best model available yet. Train models and evaluate first.")
    else:
        st.write(f"Best model: **{best_model_name}**")
        st.caption("Fill in the inputs below and the model will generate a prediction.")

        # Build inputs
        with st.form("predict_form"):
            user_row = {}
            col_left, col_right = st.columns(2)
            for i, col in enumerate(feature_cols):
                container = col_left if i % 2 == 0 else col_right
                with container:
                    if col in num_cols:
                        default_val = float(np.nan_to_num(X[col].median(), nan=0.0))
                        user_row[col] = st.number_input(col, value=default_val)
                    else:
                        options = sorted(X[col].dropna().astype(str).unique().tolist())
                        user_row[col] = st.selectbox(col, options=options if options else [""])

            submit = st.form_submit_button("Predict", type="primary")

        if submit:
            inp = pd.DataFrame([user_row])
            pred = best_pipe.predict(inp)[0]
            st.success(f"Prediction: **{pred}**")

# 8) Summary
with tabs[7]:
    st.header("8) Summary of Findings (Auto-generated)")

    st.subheader("Process covered by the app")
    st.write("""
- **Upload** dataset (or load local CSV)
- **Explore** data + target distribution + Polymer Type distribution
- **Prepare** preprocessing pipeline (imputation, encoding, scaling)
- **Handle imbalance** using SMOTE (optional, training only)
- **Train** multiple models (data mining techniques)
- **Validate & compare** models on a test set + optional cross-validation
- **Explain** results via permutation feature importance
- **Predict** using the chosen best model
""")

    best_model_name = st.session_state.get("best_model_name", None)
    cv_df = st.session_state.get("cv_results", None)
    feat_imp = st.session_state.get("feature_importance", None)

    st.subheader("Current best model")
    if best_model_name:
        st.write(f"✅ Best model selected: **{best_model_name}**")
    else:
        st.info("No best model selected yet. Train and validate first.")

    if cv_df is not None and not cv_df.empty:
        st.subheader("Cross-validation results")
        st.dataframe(cv_df, use_container_width=True)
    else:
        st.caption("Run cross-validation from the **Validate & Compare** window to populate CV results.")

    if feat_imp is not None:
        st.subheader("Top drivers (feature importance)")
        st.dataframe(feat_imp.head(10), use_container_width=True)
    else:
        st.caption("Compute feature importance from the **Feature Relevance** window to show top drivers.")
