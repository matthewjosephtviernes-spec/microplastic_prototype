import re
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
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


def plot_bar_counts(series, title):
    counts = series.value_counts(dropna=False).head(25)
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
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Microplastic Risk Modeling", layout="wide")
st.title("Predictive Risk Modeling for Microplastic Pollution (Data Mining)")

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

st.divider()

# Sidebar navigation + controls
st.sidebar.header("Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "1) Load & Explore Data",
        "2) Prepare & Preprocess",
        "3) Class Imbalance (SMOTE)",
        "4) Train Models",
        "5) Evaluate & Compare",
        "6) Feature Relevance",
        "7) Final Model + Prediction",
        "8) Summary of Findings",
    ]
)

# Target selection
st.sidebar.header("Model Target")
target_choice = st.sidebar.selectbox(
    "Choose prediction target",
    options=[
        "Risk_Type (Classification)",
        "Risk_Score (Regression)",
        "Risk_Level_std (Classification, cleaned)",
    ],
    index=0 if "Risk_Type" in df.columns else 1
)

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
st.sidebar.header("Features")
feature_cols = st.sidebar.multiselect(
    "Select feature columns (inputs)",
    options=[c for c in df.columns if c not in default_drop],
    default=[c for c in df.columns if c not in default_drop]
)

# Split settings
st.sidebar.header("Train/Test Split")
test_size = st.sidebar.slider("Test size", 0.1, 0.4, 0.2, 0.05)
random_state = st.sidebar.number_input("Random state", value=42, step=1)

# SMOTE
use_smote = False
if task == "classification":
    st.sidebar.header("Imbalance Handling")
    use_smote = st.sidebar.checkbox(
        "Use SMOTE (training only)",
        value=False,
        help="Requires imbalanced-learn. Applied only inside the training pipeline."
    )

# Models
st.sidebar.header("Models")
if task == "classification":
    chosen_models = st.sidebar.multiselect(
        "Choose models",
        options=["Logistic Regression", "Random Forest", "Gradient Boosting"],
        default=["Logistic Regression", "Random Forest", "Gradient Boosting"]
    )
else:
    chosen_models = st.sidebar.multiselect(
        "Choose models",
        options=["Ridge Regression", "Random Forest Regressor"],
        default=["Ridge Regression", "Random Forest Regressor"]
    )

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

# -----------------------------
# Pages
# -----------------------------
if page.startswith("1)"):
    st.subheader("1) Load & Explore Data")
    c1, c2 = st.columns([2, 1])
    with c1:
        st.write("Preview (first 20 rows):")
        st.dataframe(df.head(20), use_container_width=True)
    with c2:
        st.write("Shape:")
        st.write(df.shape)
        st.write("Target:")
        st.write({"task": task, "target_col": target_col})

    st.markdown("### Target distribution")
    if task == "classification":
        plot_bar_counts(df[target_col].dropna(), f"Distribution of {target_col}")
    else:
        st.write(df[target_col].describe())

    st.markdown("### Polymer Type distribution")
    if "Polymer_Type" in df.columns:
        plot_bar_counts(df["Polymer_Type"].dropna(), "Polymer_Type Distribution (Top 25)")

elif page.startswith("2)"):
    st.subheader("2) Prepare & Preprocess")
    st.markdown("""
This step shows:
- cleaning unit-like fields (e.g., salinity values such as '33 PSU' → numeric)
- imputing missing values
- one-hot encoding categorical features
- scaling numeric features (needed for LR/Ridge)
""")
    st.write("Selected features:")
    st.write(feature_cols)

    st.write("Numeric columns detected:")
    st.write(num_cols)

    st.write("Categorical columns detected:")
    st.write(cat_cols)

    st.markdown("### Missing values in selected features")
    miss = X.isna().sum().sort_values(ascending=False)
    st.dataframe(miss[miss > 0].to_frame("missing_count"), use_container_width=True)

elif page.startswith("3)"):
    st.subheader("3) Address Class Imbalance (SMOTE)")
    if task != "classification":
        st.info("SMOTE is only relevant for classification targets.")
    else:
        st.write("Class distribution in training set:")
        st.dataframe(y_train.value_counts().to_frame("count"), use_container_width=True)

        if use_smote:
            if not IMBLEARN_OK:
                st.error("imbalanced-learn is not installed, so SMOTE cannot run. Install: pip install imbalanced-learn")
            else:
                st.success("SMOTE is enabled. It will be applied ONLY inside the training pipeline (no leakage).")
        else:
            st.warning("SMOTE is OFF. Logistic Regression will still use class_weight='balanced'.")

elif page.startswith("4)"):
    st.subheader("4) Train the Models")
    st.write(f"Task: **{task}**, Target: **{target_col}**")
    st.write(f"Training rows: {len(X_train)}, Test rows: {len(X_test)}")

    st.session_state["trained_models"] = {}

    if st.button("Train now"):
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

        st.success("Training complete. Go to **Evaluate & Compare**.")

elif page.startswith("5)"):
    st.subheader("5) Evaluate & Compare Model Performance")
    models = st.session_state.get("trained_models", {})

    if not models:
        st.warning("No trained models yet. Go to **Train Models** and click Train.")
    else:
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

        st.markdown(f"### Best model (by **{sort_col}**): **{best_model_name}**")

        st.markdown("### Detailed evaluation (best model)")
        best_pred = best_pipe.predict(X_test)

        if task == "classification":
            labels = sorted(pd.Series(y_test).unique().tolist())
            plot_conf_mat(y_test, best_pred, labels=labels, title=f"Confusion Matrix: {best_model_name}")
            st.text(classification_report(y_test, best_pred, zero_division=0))
        else:
            st.json(metric_table_regression(y_test, best_pred))

elif page.startswith("6)"):
    st.subheader("6) Analyze Feature Relevance")
    best_pipe = st.session_state.get("best_pipe", None)
    best_model_name = st.session_state.get("best_model_name", None)

    if best_pipe is None:
        st.warning("Train + evaluate models first so the app can pick a best model.")
    else:
        st.write(f"Using best model: **{best_model_name}**")

        if st.button("Compute permutation importance"):
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

            st.dataframe(importances.head(20), use_container_width=True)

            fig, ax = plt.subplots()
            top = importances.head(15)
            ax.bar(top["feature"].astype(str), top["importance_mean"].values)
            ax.set_title("Top Feature Importances (Permutation)")
            ax.set_ylabel("Importance (mean)")
            plt.xticks(rotation=45, ha="right")
            st.pyplot(fig)

            st.session_state["feature_importance"] = importances

elif page.startswith("7)"):
    st.subheader("7) Final Model + Prediction Demo")
    best_pipe = st.session_state.get("best_pipe", None)
    best_model_name = st.session_state.get("best_model_name", None)

    if best_pipe is None:
        st.warning("Train + evaluate models first so the app can choose a best model.")
    else:
        st.write(f"Best model currently selected: **{best_model_name}**")
        st.markdown("Fill the inputs below to get a prediction.")

        with st.form("predict_form"):
            user_row = {}
            for col in feature_cols:
                if col in num_cols:
                    default_val = float(np.nan_to_num(X[col].median(), nan=0.0))
                    user_row[col] = st.number_input(col, value=default_val)
                else:
                    options = sorted(X[col].dropna().astype(str).unique().tolist())
                    user_row[col] = st.selectbox(col, options=options if options else [""])

            submit = st.form_submit_button("Predict")

        if submit:
            inp = pd.DataFrame([user_row])
            pred = best_pipe.predict(inp)[0]
            st.success(f"Prediction: **{pred}**")

elif page.startswith("8)"):
    st.subheader("8) Summary of Findings")
    best_model_name = st.session_state.get("best_model_name", None)
    feat_imp = st.session_state.get("feature_importance", None)

    st.markdown("""
**This app demonstrates the full process:**
- Upload / load dataset
- Explore distributions (including Polymer Type)
- Preprocess and prepare features
- Optional SMOTE for class imbalance (training only)
- Train multiple models
- Evaluate and compare performance
- Analyze feature relevance
- Predict risk from user inputs
""")

    if best_model_name:
        st.markdown(f"### Current best model: **{best_model_name}**")
    else:
        st.info("No best model selected yet. Train and evaluate first.")

    if feat_imp is not None:
        st.markdown("### Top drivers (from permutation importance)")
        st.dataframe(feat_imp.head(10), use_container_width=True)
    else:
        st.info("Feature relevance not computed yet. Go to Feature Relevance and run it.")
