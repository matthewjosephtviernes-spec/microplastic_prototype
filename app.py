# app.py — Clean, Defense-ready Streamlit UI (Windows per Process)
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

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    IMBLEARN_OK = True
except Exception:
    IMBLEARN_OK = False

import matplotlib.pyplot as plt


# -----------------------------
# Styling (clean UI)
# -----------------------------
st.set_page_config(page_title="Microplastic Risk Modeling", layout="wide")

HIDE_DEFAULTS = True  # set False if you want default menu/footer visible

st.markdown(
    """
    <style>
      .block-container { padding-top: 1.1rem; padding-bottom: 2.5rem; max-width: 1150px; }
      h1, h2, h3 { letter-spacing: -0.2px; }
      .hero {
        border-radius: 18px;
        padding: 18px 18px 14px 18px;
        background: linear-gradient(135deg, rgba(0,122,255,0.10), rgba(0,0,0,0.02));
        border: 1px solid rgba(0,122,255,0.18);
      }
      .hero-title { margin: 0; font-size: 1.55rem; }
      .hero-sub { margin-top: 6px; color: rgba(0,0,0,0.65); font-size: 0.95rem; }
      .card {
        border-radius: 16px;
        padding: 14px 14px 10px 14px;
        background: rgba(0,0,0,0.02);
        border: 1px solid rgba(0,0,0,0.06);
      }
      .muted { color: rgba(0,0,0,0.62); font-size: 0.92rem; }
      div[data-testid="stMetric"] {
        background: rgba(0,0,0,0.02);
        padding: 12px;
        border-radius: 14px;
        border: 1px solid rgba(0,0,0,0.06);
      }
      .stTabs [data-baseweb="tab-list"] { gap: 8px; }
      .stButton>button { border-radius: 12px; padding-top: 10px; padding-bottom: 10px; }
      .stDownloadButton>button { border-radius: 12px; padding-top: 10px; padding-bottom: 10px; }
      .small-note { font-size: 0.88rem; color: rgba(0,0,0,0.6); }
    </style>
    """,
    unsafe_allow_html=True
)

if HIDE_DEFAULTS:
    st.markdown(
        """
        <style>
          #MainMenu {visibility: hidden;}
          footer {visibility: hidden;}
          header {visibility: hidden;}
        </style>
        """,
        unsafe_allow_html=True
    )


# -----------------------------
# Helpers
# -----------------------------
def parse_numeric_with_units(x):
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

    for col in ["Salinity", "MP_Count_per_L", "Microplastic_Size_mm", "Density", "pH"]:
        if col in df.columns:
            df[col] = df[col].apply(parse_numeric_with_units)

    for col in ["Latitude", "Longitude"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

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

    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_cols),
            ("cat", cat_pipe, categorical_cols),
        ],
        remainder="drop"
    )
    return pre, numeric_cols, categorical_cols


def plot_bar_counts(series, title, max_bars=20):
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
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced Acc.": balanced_accuracy_score(y_true, y_pred),
        "F1 (Macro)": f1_score(y_true, y_pred, average="macro"),
        "F1 (Weighted)": f1_score(y_true, y_pred, average="weighted"),
    }


def metric_table_regression(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": mean_squared_error(y_true, y_pred, squared=False),
        "R2": r2_score(y_true, y_pred),
    }


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
        src = f"Uploaded CSV ({enc})"
    else:
        df0, enc = _read_csv("Microplastic.csv")
        src = f"Microplastic.csv ({enc})"

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


# -----------------------------
# Session defaults
# -----------------------------
if "step" not in st.session_state:
    st.session_state["step"] = 0
if "trained_models" not in st.session_state:
    reset_model_state()


# -----------------------------
# HERO
# -----------------------------
st.markdown(
    """
    <div class="hero">
      <div class="hero-title"><b>Predictive Risk Modeling for Microplastic Pollution</b></div>
      <div class="hero-sub">
        A defense-ready dashboard showing the complete pipeline:
        Upload → Explore → Preprocess → Imbalance → Train → Validate → Interpret → Predict
      </div>
    </div>
    """,
    unsafe_allow_html=True
)
st.write("")


# -----------------------------
# Sidebar: Steps
# -----------------------------
STEPS = [
    ("📥 Upload", "Upload dataset + basic setup"),
    ("🔎 Explore", "Distributions & preview"),
    ("🧼 Preprocess", "Missing values & feature types"),
    ("⚖️ Imbalance", "Class distribution + SMOTE"),
    ("🧠 Train", "Train multiple models"),
    ("✅ Validate", "Compare + best model + CV"),
    ("🧩 Explain", "Feature importance"),
    ("🔮 Predict", "Prediction demo"),
    ("📌 Summary", "Defense summary"),
]
with st.sidebar:
    st.markdown("## Process Windows")
    labels = [f"{i+1}. {name}" for i, (name, _) in enumerate(STEPS)]
    picked = st.radio("Select a window", labels, index=st.session_state["step"])
    goto(labels.index(picked))
    st.markdown("---")
    st.caption("Tip: During defense, navigate window-by-window for a clean story.")


# -----------------------------
# Window 1: Upload + Config (clean)
# -----------------------------
step = st.session_state["step"]

if step == 0:
    st.subheader("📥 Upload Dataset")
    left, right = st.columns([1.15, 1.85])

    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
        try:
            df, data_src = load_data(uploaded_file)
            st.success(f"Loaded: {data_src}")
        except Exception as e:
            st.error(f"Could not load dataset.\n\nError: {e}")
            st.stop()
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="card" style="margin-top:12px;">', unsafe_allow_html=True)
        st.markdown("**Dataset Quick Info**")
        st.metric("Rows", int(df.shape[0]))
        st.metric("Columns", int(df.shape[1]))
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**Configuration**")

        target_choice = st.selectbox(
            "Target variable",
            options=[
                "Risk_Type (Classification)",
                "Risk_Score (Regression)",
                "Risk_Level_std (Classification, cleaned)",
            ],
            index=0 if "Risk_Type" in df.columns else 1
        )

        test_size = st.slider("Test size", 0.1, 0.4, 0.2, 0.05)
        random_state = st.number_input("Random state", value=42, step=1)

        if target_choice.startswith("Risk_Score"):
            use_smote = False
            st.caption("SMOTE not applicable for regression.")
        else:
            use_smote = st.checkbox("Use SMOTE (training only)", value=False)
            if use_smote and not IMBLEARN_OK:
                st.warning("Install SMOTE dependency: pip install imbalanced-learn")

        # Determine task
        if target_choice.startswith("Risk_Type"):
            target_col = "Risk_Type"
            task = "classification"
        elif target_choice.startswith("Risk_Score"):
            target_col = "Risk_Score"
            task = "regression"
        else:
            target_col = "Risk_Level_std"
            task = "classification"

        default_drop = [c for c in ["Risk_Type", "Risk_Score", "Risk_Level", "Risk_Level_std"] if c in df.columns]
        feature_cols = st.multiselect(
            "Input features (X)",
            options=[c for c in df.columns if c not in default_drop],
            default=[c for c in df.columns if c not in default_drop]
        )
        if not feature_cols:
            st.error("Please select at least one feature column.")
            st.stop()

        # Save config in session for other windows
        st.session_state["cfg"] = {
            "target_choice": target_choice,
            "target_col": target_col,
            "task": task,
            "test_size": float(test_size),
            "random_state": int(random_state),
            "use_smote": bool(use_smote),
            "feature_cols": feature_cols,
        }

        # Reset when config changes
        sig = (
            tuple(feature_cols), target_col, float(test_size), int(random_state),
            bool(use_smote), task
        )
        if st.session_state.get("cfg_sig") != sig:
            st.session_state["cfg_sig"] = sig
            reset_model_state()

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="small-note" style="margin-top:10px;">', unsafe_allow_html=True)
        st.write("Next: Go to **Explore** window to show distributions and dataset preview.")
        st.markdown("</div>", unsafe_allow_html=True)


else:
    # For steps 1..8, ensure we have data and config
    uploaded_file = st.session_state.get("uploaded_file", None)

    # Re-load dataset each run (cache keeps it fast)
    # Use same uploader logic: if none uploaded, it loads Microplastic.csv
    df, _src = load_data(st.session_state.get("last_uploaded", None))

    # If user uploaded during step 0, they might not be here.
    # Fallback: try to load local file; if not present, ask user to go upload.
    cfg = st.session_state.get("cfg")
    if cfg is None:
        st.warning("Please go to **Upload** window first and set your configuration.")
        st.stop()

    target_col = cfg["target_col"]
    task = cfg["task"]
    test_size = cfg["test_size"]
    random_state = cfg["random_state"]
    use_smote = cfg["use_smote"]
    feature_cols = cfg["feature_cols"]

    # Prepare data
    data = df.dropna(subset=[target_col]).copy()
    X = data[feature_cols].copy()
    y = data[target_col].copy()

    if task == "classification":
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

    preprocessor, num_cols, cat_cols = build_preprocessor(X_train)

    # -----------------------------
    # Window 2: Explore
    # -----------------------------
    if step == 1:
        st.subheader("🔎 Explore Dataset")

        a, b = st.columns([2.2, 1])
        with a:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.write("Dataset preview:")
            st.dataframe(df.head(25), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with b:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.metric("Rows", int(df.shape[0]))
            st.metric("Columns", int(df.shape[1]))
            st.metric("Target", target_col)
            st.metric("Task", task)
            st.markdown("</div>", unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("**Target Distribution**")
            if task == "classification":
                plot_bar_counts(df[target_col].dropna(), f"{target_col} distribution")
            else:
                st.write(df[target_col].describe())
            st.markdown("</div>", unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("**Polymer Type Distribution**")
            if "Polymer_Type" in df.columns:
                plot_bar_counts(df["Polymer_Type"].dropna(), "Polymer_Type distribution (Top 20)")
            else:
                st.info("Polymer_Type not found.")
            st.markdown("</div>", unsafe_allow_html=True)

    # -----------------------------
    # Window 3: Preprocess
    # -----------------------------
    elif step == 2:
        st.subheader("🧼 Preprocess")

        left, right = st.columns(2)
        with left:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("**Detected Feature Types**")
            st.write("Numeric:", num_cols)
            st.write("Categorical:", cat_cols)
            st.markdown("</div>", unsafe_allow_html=True)

        with right:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("**Missing Values (Selected Features)**")
            miss = X.isna().sum().sort_values(ascending=False)
            mv = miss[miss > 0].to_frame("missing_count")
            if mv.empty:
                st.success("No missing values for selected features.")
            else:
                st.dataframe(mv, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="card" style="margin-top:12px;">', unsafe_allow_html=True)
        st.markdown("**Train/Test Split**")
        s1, s2, s3 = st.columns(3)
        s1.metric("Train rows", len(X_train))
        s2.metric("Test rows", len(X_test))
        s3.metric("Test size", float(test_size))
        st.markdown("</div>", unsafe_allow_html=True)

    # -----------------------------
    # Window 4: Imbalance
    # -----------------------------
    elif step == 3:
        st.subheader("⚖️ Class Imbalance")

        if task != "classification":
            st.info("Imbalance handling is for classification only.")
        else:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("**Class distribution (Training set)**")
            st.dataframe(y_train.value_counts().to_frame("count"), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

            if use_smote:
                if IMBLEARN_OK:
                    st.success("✅ SMOTE enabled — applied ONLY inside training pipeline (no leakage).")
                else:
                    st.error("SMOTE enabled but missing dependency. Install: pip install imbalanced-learn")
            else:
                st.warning("SMOTE is OFF. Logistic Regression still uses class_weight='balanced'.")

    # -----------------------------
    # Window 5: Train
    # -----------------------------
    elif step == 4:
        st.subheader("🧠 Train Models")

        st.markdown('<div class="card">', unsafe_allow_html=True)
        if task == "classification":
            models_selected = st.multiselect(
                "Choose models",
                ["Logistic Regression", "Random Forest", "Gradient Boosting"],
                default=["Logistic Regression", "Random Forest", "Gradient Boosting"]
            )
        else:
            models_selected = st.multiselect(
                "Choose models",
                ["Ridge Regression", "Random Forest Regressor"],
                default=["Ridge Regression", "Random Forest Regressor"]
            )
        st.markdown("</div>", unsafe_allow_html=True)

        cA, cB = st.columns([1, 1.3])
        with cA:
            if st.button("🚀 Train now", type="primary", use_container_width=True):
                reset_model_state()
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
                st.success("Training complete! Proceed to Validate window.")

        with cB:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("**Notes (for defense)**")
            st.write(
                "- Preprocessing: imputation → encoding → scaling\n"
                "- SMOTE (if enabled): training only\n"
                "- Compare multiple models to select best"
            )
            st.markdown("</div>", unsafe_allow_html=True)

    # -----------------------------
    # Window 6: Validate
    # -----------------------------
    elif step == 5:
        st.subheader("✅ Validate & Compare")

        trained = st.session_state.get("trained_models", {})
        if not trained:
            st.warning("Train models first in the Train window.")
        else:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("**Model comparison (Test set)**")
            results = []
            for name, pipe in trained.items():
                y_pred = pipe.predict(X_test)
                if task == "classification":
                    results.append({"Model": name, **metric_table_classification(y_test, y_pred)})
                else:
                    results.append({"Model": name, **metric_table_regression(y_test, y_pred)})

            sort_col = "F1 (Macro)" if task == "classification" else "RMSE"
            res_df = pd.DataFrame(results).sort_values(by=sort_col, ascending=(task != "classification"))
            st.dataframe(res_df, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

            best_model_name = res_df.iloc[0]["Model"]
            best_pipe = trained[best_model_name]
            st.session_state["best_model_name"] = best_model_name
            st.session_state["best_pipe"] = best_pipe

            st.success(f"Best model: {best_model_name}")

            st.markdown('<div class="card" style="margin-top:12px;">', unsafe_allow_html=True)
            st.markdown("**Detailed evaluation (Best model)**")
            best_pred = best_pipe.predict(X_test)

            if task == "classification":
                labels = sorted(pd.Series(y_test).unique().tolist())
                plot_conf_mat(y_test, best_pred, labels=labels, title=f"Confusion Matrix: {best_model_name}")
                st.text(classification_report(y_test, best_pred, zero_division=0))
            else:
                st.json(metric_table_regression(y_test, best_pred))
            st.markdown("</div>", unsafe_allow_html=True)

            if task == "classification":
                st.markdown('<div class="card" style="margin-top:12px;">', unsafe_allow_html=True)
                st.markdown("**Cross-validation (5-fold Stratified CV)**")
                if st.button("Run CV", use_container_width=True):
                    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
                    scoring = "f1_macro"
                    cv_rows = []
                    for name, pipe in trained.items():
                        scores = cross_val_score(pipe, X, y, cv=cv, scoring=scoring)
                        cv_rows.append({
                            "Model": name,
                            "CV metric": "F1 (Macro)",
                            "Mean": float(np.mean(scores)),
                            "Std": float(np.std(scores)),
                        })
                    cv_df = pd.DataFrame(cv_rows).sort_values("Mean", ascending=False)
                    st.session_state["cv_results"] = cv_df
                    st.dataframe(cv_df, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

    # -----------------------------
    # Window 7: Explain
    # -----------------------------
    elif step == 6:
        st.subheader("🧩 Explain (Feature Relevance)")

        best_pipe = st.session_state.get("best_pipe")
        best_name = st.session_state.get("best_model_name")
        if best_pipe is None:
            st.warning("Select best model first in Validate window.")
        else:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(f"**Best model:** {best_name}")
            st.caption("Permutation importance = impact on model score when feature is shuffled.")

            if st.button("Compute Feature Importance", type="primary", use_container_width=True):
                scoring = "f1_macro" if task == "classification" else "r2"
                perm = permutation_importance(
                    best_pipe, X_test, y_test, n_repeats=10,
                    random_state=random_state, scoring=scoring
                )
                imp = pd.DataFrame({
                    "Feature": feature_cols,
                    "Importance (mean)": perm.importances_mean,
                    "Importance (std)": perm.importances_std
                }).sort_values("Importance (mean)", ascending=False)

                st.session_state["feature_importance"] = imp

            imp = st.session_state.get("feature_importance")
            if imp is None:
                st.info("Click compute to show important features.")
            else:
                st.dataframe(imp.head(20), use_container_width=True)

                fig, ax = plt.subplots()
                top = imp.head(12)
                ax.bar(top["Feature"].astype(str), top["Importance (mean)"].values)
                ax.set_title("Top Feature Importances (Permutation)")
                ax.set_ylabel("Importance")
                plt.xticks(rotation=45, ha="right")
                st.pyplot(fig)

            st.markdown("</div>", unsafe_allow_html=True)

    # -----------------------------
    # Window 8: Predict
    # -----------------------------
    elif step == 7:
        st.subheader("🔮 Predict (Demo)")

        best_pipe = st.session_state.get("best_pipe")
        best_name = st.session_state.get("best_model_name")
        if best_pipe is None:
            st.warning("Train and validate first.")
        else:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(f"**Best model:** {best_name}")

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

            st.markdown("</div>", unsafe_allow_html=True)

    # -----------------------------
    # Window 9: Summary
    # -----------------------------
    elif step == 8:
        st.subheader("📌 Summary (For Defense)")

        best_name = st.session_state.get("best_model_name")
        cv_df = st.session_state.get("cv_results")
        imp = st.session_state.get("feature_importance")

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**Pipeline demonstrated by this dashboard:**")
        st.write(
            "1) Upload Dataset\n"
            "2) Explore Distributions\n"
            "3) Preprocess Data\n"
            "4) Handle Class Imbalance (SMOTE optional)\n"
            "5) Train Multiple Models\n"
            "6) Validate & Compare + Cross-Validation\n"
            "7) Explain via Feature Importance\n"
            "8) Predict Risk from Inputs"
        )
        st.markdown("</div>", unsafe_allow_html=True)

        m1, m2, m3 = st.columns(3)
        m1.metric("Best Model", best_name if best_name else "N/A")
        m2.metric("Models Trained", len(st.session_state.get("trained_models", {})))
        m3.metric("Task", task)

        if cv_df is not None and not cv_df.empty:
            st.markdown("### Cross-validation Results")
            st.dataframe(cv_df, use_container_width=True)
        else:
            st.info("Run CV in Validate window to show CV results here.")

        if imp is not None:
            st.markdown("### Top Drivers (Feature Importance)")
            st.dataframe(imp.head(10), use_container_width=True)
        else:
            st.info("Compute feature importance in Explain window.")

