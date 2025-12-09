# app.py — Defense-ready Streamlit “Windows per Process”
# Predictive Risk Modeling for Microplastic Pollution using Data Mining Techniques

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

# Optional: SMOTE
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


def plot_bar_counts(series, title, max_bars=25):
    counts = series.value_counts(dropna=False).head(max_bars)
    fig, ax = plt.subplots()
    ax.bar(counts.index.astype(str), counts.values)
    ax.set_title(title)
    ax.set_xlabel(series.name)
    ax.set_ylabel("Count")
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


# -----------------------------
# Robust CSV loader (encoding + delimiter auto-detect)
# -----------------------------
@st.cache_data
def load_data(uploaded):
    def _read_csv(file_obj):
        encodings = ["utf-8", "utf-8-sig", "cp1252", "latin1"]
        last_err = None
        for enc in encodings:
            try:
                if hasattr(file_obj, "seek"):
                    file_obj.seek(0)
                df_ = pd.read_csv(file_obj, encoding=enc, sep=None, engine="python")
                return df_, enc
            except Exception as e:
                last_err = e
        raise last_err

    if uploaded is not None:
        df0, enc = _read_csv(uploaded)
        src = f"Uploaded CSV (encoding: {enc})"
    else:
        df0, enc = _read_csv("Microplastic.csv")
        src = f"Local file: Microplastic.csv (encoding: {enc})"

    df0 = clean_dataframe(df0)
    return df0, src


def reset_model_state():
    st.session_state["trained_models"] = {}
    st.session_state["best_model_name"] = None
    st.session_state["best_pipe"] = None
    st.session_state["feature_importance"] = None
    st.session_state["cv_results"] = None


def goto(step_idx: int):
    st.session_state["step"] = int(step_idx)


def step_nav(total_steps: int):
    """Top progress + Bottom Next/Back buttons."""
    step = st.session_state.get("step", 0)
    st.progress((step + 1) / total_steps)

    b1, b2, b3 = st.columns([1, 6, 1])
    with b1:
        if st.button("⬅ Back", disabled=(step == 0), use_container_width=True):
            goto(step - 1)
            st.rerun()
    with b3:
        if st.button("Next ➡", disabled=(step == total_steps - 1), use_container_width=True):
            goto(step + 1)
            st.rerun()


# -----------------------------
# UI Setup
# -----------------------------
st.set_page_config(page_title="Microplastic Risk Modeling Dashboard", layout="wide")

# Light CSS polish for defense
st.markdown(
    """
    <style>
      .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
      div[data-testid="stMetric"] { background: rgba(0,0,0,0.03); padding: 10px; border-radius: 12px; }
      .title-card { padding: 18px; border-radius: 16px; background: rgba(27, 133, 255, 0.08); border: 1px solid rgba(27,133,255,0.18); }
      .sub-card { padding: 14px; border-radius: 14px; background: rgba(0,0,0,0.02); border: 1px solid rgba(0,0,0,0.06); }
      .small { color: rgba(0,0,0,0.6); font-size: 0.92rem; }
    </style>
    """,
    unsafe_allow_html=True
)

if "step" not in st.session_state:
    st.session_state["step"] = 0
if "trained_models" not in st.session_state:
    reset_model_state()

# Header
st.markdown(
    """
    <div class="title-card">
      <h2 style="margin:0;">Predictive Risk Modeling for Microplastic Pollution</h2>
      <div class="small">Defense-ready dashboard: each process is shown in a separate window (step-by-step).</div>
    </div>
    """,
    unsafe_allow_html=True
)

st.write("")  # spacing


# -----------------------------
# Global: Upload & Config (always visible)
# -----------------------------
top1, top2 = st.columns([1.15, 1.85])

with top1:
    st.markdown("### 1) Upload Dataset")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    try:
        df, data_src = load_data(uploaded_file)
        st.success(f"Loaded: {data_src}")
    except Exception as e:
        st.error(f"Could not load dataset. Upload CSV or ensure Microplastic.csv exists.\n\nError: {e}")
        st.stop()

with top2:
    st.markdown("### 2) Project Configuration")
    c1, c2, c3 = st.columns(3)

    with c1:
        target_choice = st.selectbox(
            "Target",
            options=[
                "Risk_Type (Classification)",
                "Risk_Score (Regression)",
                "Risk_Level_std (Classification, cleaned)",
            ],
            index=0 if "Risk_Type" in df.columns else 1
        )

    with c2:
        test_size = st.slider("Test size", 0.1, 0.4, 0.2, 0.05)
        random_state = st.number_input("Random state", value=42, step=1)

    with c3:
        if target_choice.startswith("Risk_Score"):
            use_smote = False
            st.caption("SMOTE not applicable for regression.")
        else:
            use_smote = st.checkbox("Use SMOTE (training only)", value=False)
            if use_smote and not IMBLEARN_OK:
                st.warning("SMOTE requires: pip install imbalanced-learn")

# Determine task/target_col
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
default_drop = [c for c in ["Risk_Type", "Risk_Score", "Risk_Level", "Risk_Level_std"] if c in df.columns]
feature_cols = st.multiselect(
    "Input Features (X)",
    options=[c for c in df.columns if c not in default_drop],
    default=[c for c in df.columns if c not in default_drop]
)
if not feature_cols:
    st.error("Please select at least one feature column.")
    st.stop()

# Reset models when config changes (simple heuristic)
cfg_signature = (tuple(feature_cols), target_col, test_size, int(random_state), bool(use_smote), task)
if st.session_state.get("cfg_signature") != cfg_signature:
    st.session_state["cfg_signature"] = cfg_signature
    reset_model_state()

# Prepare X, y
data = df.dropna(subset=[target_col]).copy()
X = data[feature_cols].copy()
y = data[target_col].copy()

# Train/Test split
if task == "classification":
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
else:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

preprocessor, num_cols, cat_cols = build_preprocessor(X_train)

st.divider()


# -----------------------------
# Stepper windows
# -----------------------------
STEPS = [
    ("Explore Dataset", "Preview + distributions (Polymer Type, target)."),
    ("Prepare & Preprocess", "Missing values + detected feature types + split."),
    ("Handle Class Imbalance", "Show imbalance + SMOTE info."),
    ("Train Models", "Train multiple models (data mining techniques)."),
    ("Validate & Compare", "Test metrics + best model + cross-validation."),
    ("Feature Relevance", "Permutation importance for interpretability."),
    ("Prediction Demo", "User inputs → predicted risk."),
    ("Summary", "Auto-generated summary for defense.")
]
TOTAL = len(STEPS)

# Left sidebar step list (true “window switching”)
with st.sidebar:
    st.markdown("## Process Windows")
    step_labels = [f"{i+1}. {name}" for i, (name, _) in enumerate(STEPS)]
    selected = st.radio("Select window", step_labels, index=st.session_state["step"])
    goto(step_labels.index(selected))
    st.markdown("---")
    st.caption("Tip: during defense, just click Next/Back or select a window here.")

# Window header
step = st.session_state["step"]
step_title, step_desc = STEPS[step]
st.markdown(f"## {step+1}) {step_title}")
st.caption(step_desc)
step_nav(TOTAL)

st.write("")


# -----------------------------
# Window 1: Explore
# -----------------------------
if step == 0:
    a, b = st.columns([2.2, 1])
    with a:
        st.markdown('<div class="sub-card">', unsafe_allow_html=True)
        st.write("Dataset preview (top rows):")
        st.dataframe(df.head(25), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with b:
        st.markdown('<div class="sub-card">', unsafe_allow_html=True)
        st.metric("Rows", int(df.shape[0]))
        st.metric("Columns", int(df.shape[1]))
        st.metric("Target", target_col)
        st.metric("Task", task)
        st.markdown("</div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Target Distribution")
        if task == "classification":
            plot_bar_counts(df[target_col].dropna(), f"{target_col} distribution")
        else:
            st.write(df[target_col].describe())

    with c2:
        st.markdown("### Polymer Type Distribution")
        if "Polymer_Type" in df.columns:
            plot_bar_counts(df["Polymer_Type"].dropna(), "Polymer_Type distribution (Top 25)")
        else:
            st.info("Polymer_Type column not found in this dataset.")

# -----------------------------
# Window 2: Prepare
# -----------------------------
elif step == 1:
    left, right = st.columns(2)

    with left:
        st.markdown('<div class="sub-card">', unsafe_allow_html=True)
        st.markdown("### Detected Feature Types")
        st.write("Numeric columns:", num_cols)
        st.write("Categorical columns:", cat_cols)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="sub-card">', unsafe_allow_html=True)
        st.markdown("### Missing Values (Selected Features)")
        miss = X.isna().sum().sort_values(ascending=False)
        mv = miss[miss > 0].to_frame("missing_count")
        if mv.empty:
            st.success("No missing values found in selected features.")
        else:
            st.dataframe(mv, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("### Train/Test Split")
    s1, s2, s3 = st.columns(3)
    s1.metric("Train rows", len(X_train))
    s2.metric("Test rows", len(X_test))
    s3.metric("Test size", float(test_size))

# -----------------------------
# Window 3: Imbalance
# -----------------------------
elif step == 2:
    if task != "classification":
        st.info("Class imbalance handling is applicable only for classification targets.")
    else:
        st.markdown('<div class="sub-card">', unsafe_allow_html=True)
        st.markdown("### Class distribution (Training set)")
        st.dataframe(y_train.value_counts().to_frame("count"), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        if use_smote:
            if IMBLEARN_OK:
                st.success("✅ SMOTE is enabled. It will be applied ONLY inside the training pipeline (no data leakage).")
            else:
                st.error("SMOTE is enabled but `imbalanced-learn` is missing. Install: pip install imbalanced-learn")
        else:
            st.warning("SMOTE is OFF. Logistic Regression still uses class_weight='balanced'.")

# -----------------------------
# Window 4: Train
# -----------------------------
elif step == 3:
    st.markdown("### Select Models to Train")
    if task == "classification":
        models_selected = st.multiselect(
            "Classification models",
            ["Logistic Regression", "Random Forest", "Gradient Boosting"],
            default=["Logistic Regression", "Random Forest", "Gradient Boosting"]
        )
    else:
        models_selected = st.multiselect(
            "Regression models",
            ["Ridge Regression", "Random Forest Regressor"],
            default=["Ridge Regression", "Random Forest Regressor"]
        )

    st.session_state["models_selected"] = models_selected

    colA, colB = st.columns([1, 2.2])
    with colA:
        if st.button("🚀 Train now", type="primary", use_container_width=True):
            reset_model_state()
            st.session_state["models_selected"] = models_selected

            trained = {}

            for name in models_selected:
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
                    trained[name] = pipe

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
                    trained[name] = pipe

            st.session_state["trained_models"] = trained
            st.success("Training complete! Proceed to Validate & Compare.")

    with colB:
        st.markdown('<div class="sub-card">', unsafe_allow_html=True)
        st.markdown("### Training Notes (for defense)")
        st.write(
            "- Preprocessing pipeline: imputation → encoding → scaling\n"
            "- SMOTE (if enabled): applied only on training folds to avoid leakage\n"
            "- Multiple models trained for comparison"
        )
        st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# Window 5: Validate & Compare
# -----------------------------
elif step == 4:
    trained = st.session_state.get("trained_models", {})
    if not trained:
        st.warning("Wala pay trained models. Adto sa Train window ug i-train una.")
    else:
        st.markdown("### Test-set Model Comparison")
        results = []
        for name, pipe in trained.items():
            y_pred = pipe.predict(X_test)
            if task == "classification":
                results.append({"model": name, **metric_table_classification(y_test, y_pred)})
            else:
                results.append({"model": name, **metric_table_regression(y_test, y_pred)})

        sort_col = "f1_macro" if task == "classification" else "RMSE"
        res_df = pd.DataFrame(results).sort_values(by=sort_col, ascending=(task != "classification"))
        st.dataframe(res_df, use_container_width=True)

        best_model_name = res_df.iloc[0]["model"]
        best_pipe = trained[best_model_name]
        st.session_state["best_model_name"] = best_model_name
        st.session_state["best_pipe"] = best_pipe

        st.success(f"Best model (by {sort_col}): {best_model_name}")

        st.markdown("### Detailed Evaluation (Best Model)")
        best_pred = best_pipe.predict(X_test)
        if task == "classification":
            labels = sorted(pd.Series(y_test).unique().tolist())
            plot_conf_mat(y_test, best_pred, labels=labels, title=f"Confusion Matrix: {best_model_name}")
            st.text(classification_report(y_test, best_pred, zero_division=0))
        else:
            st.json(metric_table_regression(y_test, best_pred))

        st.markdown("### Cross-Validation (Generalizability)")
        if task != "classification":
            st.info("CV page currently set for classification. If you want regression CV, I can add it.")
        else:
            if st.button("Run 5-fold Stratified CV", use_container_width=True):
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
                scoring = "f1_macro"
                cv_rows = []
                for name, pipe in trained.items():
                    scores = cross_val_score(pipe, X, y, cv=cv, scoring=scoring)
                    cv_rows.append({
                        "model": name,
                        "cv_metric": scoring,
                        "mean": float(np.mean(scores)),
                        "std": float(np.std(scores)),
                        "min": float(np.min(scores)),
                        "max": float(np.max(scores)),
                    })
                cv_df = pd.DataFrame(cv_rows).sort_values("mean", ascending=False)
                st.session_state["cv_results"] = cv_df
                st.dataframe(cv_df, use_container_width=True)

# -----------------------------
# Window 6: Feature Relevance
# -----------------------------
elif step == 5:
    best_pipe = st.session_state.get("best_pipe")
    best_name = st.session_state.get("best_model_name")
    if best_pipe is None:
        st.warning("Wala pa'y best model. Train + validate una.")
    else:
        st.markdown(f"### Best Model: {best_name}")
        st.caption("Permutation importance = unsay pinaka naka-influence sa predictions.")

        if st.button("Compute Feature Importance", type="primary", use_container_width=True):
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

        imp = st.session_state.get("feature_importance")
        if imp is None:
            st.info("I-click ang Compute Feature Importance.")
        else:
            st.dataframe(imp.head(20), use_container_width=True)

            fig, ax = plt.subplots()
            top = imp.head(15)
            ax.bar(top["feature"].astype(str), top["importance_mean"].values)
            ax.set_title("Top Feature Importances (Permutation)")
            ax.set_ylabel("Importance (mean)")
            plt.xticks(rotation=45, ha="right")
            st.pyplot(fig)

# -----------------------------
# Window 7: Predict
# -----------------------------
elif step == 6:
    best_pipe = st.session_state.get("best_pipe")
    best_name = st.session_state.get("best_model_name")
    if best_pipe is None:
        st.warning("Wala pa'y model para prediction. Train + validate una.")
    else:
        st.markdown(f"### Prediction Demo (Best Model: {best_name})")
        st.caption("I-fill ang inputs — human sa app og predict ang Risk.")

        with st.form("predict_form"):
            user_row = {}
            left, right = st.columns(2)
            for i, col in enumerate(feature_cols):
                box = left if i % 2 == 0 else right
                with box:
                    if col in num_cols:
                        default_val = float(np.nan_to_num(X[col].median(), nan=0.0))
                        user_row[col] = st.number_input(col, value=default_val)
                    else:
                        options = sorted(X[col].dropna().astype(str).unique().tolist())
                        user_row[col] = st.selectbox(col, options=options if options else [""])

            submitted = st.form_submit_button("Predict", type="primary")

        if submitted:
            inp = pd.DataFrame([user_row])
            pred = best_pipe.predict(inp)[0]
            st.success(f"Predicted value: **{pred}**")

# -----------------------------
# Window 8: Summary
# -----------------------------
elif step == 7:
    st.markdown("### Summary for Defense Presentation")
    st.write(
        """
**This dashboard demonstrates the complete predictive modeling framework:**
1. Data upload and dataset exploration  
2. Data preparation and preprocessing (imputation, encoding, scaling)  
3. Class imbalance analysis + optional SMOTE  
4. Training multiple data mining models  
5. Validation: test-set evaluation + cross-validation for generalizability  
6. Feature relevance for interpretability  
7. Prediction demo for practical decision support
        """.strip()
    )

    best_name = st.session_state.get("best_model_name")
    cv_df = st.session_state.get("cv_results")
    imp = st.session_state.get("feature_importance")

    a, b, c = st.columns(3)
    a.metric("Best Model", best_name if best_name else "N/A")
    b.metric("Models Trained", len(st.session_state.get("trained_models", {})))
    c.metric("Task Type", task)

    if cv_df is not None and not cv_df.empty:
        st.markdown("#### Cross-validation results")
        st.dataframe(cv_df, use_container_width=True)
    else:
        st.info("Run cross-validation in Validate & Compare to show CV results here.")

    if imp is not None:
        st.markdown("#### Top Drivers (Feature Importance)")
        st.dataframe(imp.head(10), use_container_width=True)
    else:
        st.info("Compute feature importance to show top drivers here.")

st.write("")
step_nav(TOTAL)
