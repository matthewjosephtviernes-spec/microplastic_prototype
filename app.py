import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import StratifiedKFold, KFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from pandas.errors import EmptyDataError, ParserError

# -------------------------------------------------------
# CONFIG
# -------------------------------------------------------
st.set_page_config(page_title="Microplastic Risk Analysis", layout="wide")

NUMERIC_COLS = [
    "MP_Count_per_L",
    "Risk_Score",
    "Microplastic_Size_mm_midpoint",
    "Density_midpoint",
]

CATEGORICAL_COLS = [
    "Location",
    "Shape",
    "Polymer_Type",
    "pH",
    "Salinity",
    "Industrial_Activity",
    "Population_Density",
    "Author",
]

TARGET_RISK_TYPE = "Risk_Type"
TARGET_RISK_LEVEL = "Risk_Level"

# BIG SPEED WIN:
# Location (~800+ unique) and Author (~90+ unique) explode one-hot features.
# Keep them for EDA pages, but drop them for modeling/CV by default.
DEFAULT_MODEL_DROP_COLS = ["Location", "Author"]


# -------------------------------------------------------
# DATA LOADING (robust for Streamlit uploads)
# -------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data(uploaded_file=None, path: str = "Microplastic.csv"):
    """
    Load CSV from uploaded file or local path.
    Robust for Streamlit UploadedFile pointer resets.
    Tries multiple encodings to avoid UnicodeDecodeError.
    """
    src = uploaded_file if uploaded_file is not None else path
    encodings_to_try = ["latin1", "utf-8", "cp1252"]

    last_err = None
    for enc in encodings_to_try:
        try:
            if uploaded_file is not None:
                try:
                    uploaded_file.seek(0)  # IMPORTANT for reruns/caching
                except Exception:
                    pass
            df = pd.read_csv(src, encoding=enc)
            df.columns = [str(c).strip() for c in df.columns]
            return df
        except (UnicodeDecodeError, EmptyDataError, ParserError) as e:
            last_err = e
            continue
        except FileNotFoundError:
            if uploaded_file is None:
                raise

    if last_err is not None:
        raise last_err
    return None


# -------------------------------------------------------
# PREPROCESSING (EDA-style pages)
# -------------------------------------------------------
def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in NUMERIC_COLS:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            df[col] = s.fillna(s.median())

    for col in CATEGORICAL_COLS:
        if col in df.columns:
            mode_val = df[col].mode(dropna=True)
            if len(mode_val) > 0:
                df[col] = df[col].fillna(mode_val.iloc[0])

    return df


def cap_outliers_iqr(df: pd.DataFrame, cols) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        low = q1 - 1.5 * iqr
        high = q3 + 1.5 * iqr
        clipped = np.where(s < low, low, s)
        clipped = np.where(clipped > high, high, clipped)
        df[col] = clipped
    return df


def transform_skewed(df: pd.DataFrame, cols):
    df = df.copy()
    cols_present = [c for c in cols if c in df.columns]
    if not cols_present:
        return df, pd.Series(dtype=float), []

    for col in cols_present:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    skewness = df[cols_present].skew(numeric_only=True)
    skewed_cols = skewness[skewness.abs() > 1].index.tolist()

    for col in skewed_cols:
        min_val = df[col].min()
        if pd.isna(min_val):
            continue
        shift = (abs(min_val) + 1e-6) if min_val <= 0 else 0
        df[col] = np.log1p(df[col] + shift)

    return df, skewness, skewed_cols


def scale_numeric(df: pd.DataFrame, cols):
    df = df.copy()
    scaler = StandardScaler()
    cols_present = [c for c in cols if c in df.columns]
    if cols_present:
        df[cols_present] = scaler.fit_transform(df[cols_present])
    return df, scaler


# -------------------------------------------------------
# NUMERIC SANITIZER (prevents SimpleImputer crash)
# -------------------------------------------------------
def coerce_numeric_columns(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    """
    Ensures numeric cols are truly numeric (object strings -> float).
    Handles commas like "1,234" by removing commas first.
    """
    df = df.copy()
    for c in numeric_cols:
        if c in df.columns:
            s = df[c].astype(str).str.replace(",", "", regex=False)
            df[c] = pd.to_numeric(s, errors="coerce")
    return df


# -------------------------------------------------------
# SPLIT HELPERS
# -------------------------------------------------------
def merge_rare_classes(y: pd.Series, min_count: int = 2, other_label: str = "Other"):
    y = pd.Series(y).copy()
    counts = y.value_counts(dropna=True)
    rare = counts[counts < min_count].index
    y = y.where(~y.isin(rare), other_label)
    return y


def safe_train_test_split(X, y, test_size=0.2, random_state=42):
    y = pd.Series(y)
    mask = y.notna()
    X = X.loc[mask]
    y = y.loc[mask]

    if y.nunique() < 2:
        raise ValueError("Need at least 2 classes in the target.")

    counts = y.value_counts()
    min_class = int(counts.min())
    n = len(y)
    k = y.nunique()

    if min_class < 2:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=None
        )
        return (X_train, X_test, y_train, y_test), False, float(test_size)

    min_test_size = k / n
    max_test_size = 1 - (k / n)

    ts = float(test_size)
    ts = max(ts, min_test_size)
    if max_test_size > 0:
        ts = min(ts, max_test_size)

    for ts_try in [ts, 0.2, 0.15, 0.1, 0.05]:
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=ts_try, random_state=random_state, stratify=y
            )
            return (X_train, X_test, y_train, y_test), True, float(ts_try)
        except ValueError:
            continue

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=None
    )
    return (X_train, X_test, y_train, y_test), False, float(test_size)


# -------------------------------------------------------
# LEAKAGE-SAFE PIPELINE BUILDERS (for both holdout + CV)
# -------------------------------------------------------
@st.cache_data(show_spinner=False)
def build_preprocess_pipeline_cached(df_raw: pd.DataFrame, drop_cols_for_model: tuple):
    # IMPORTANT: only include numeric cols that have at least one non-null value (median needs data)
    numeric_features = [c for c in NUMERIC_COLS if c in df_raw.columns]
    numeric_features = [c for c in numeric_features if df_raw[c].notna().any()]

    categorical_features = [c for c in CATEGORICAL_COLS if c in df_raw.columns and c not in drop_cols_for_model]

    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_features),
            ("cat", categorical_pipe, categorical_features),
        ],
        remainder="drop"
    )
    return preprocessor


def get_Xy_for_target(df_raw: pd.DataFrame, target_col: str, drop_cols_for_model: tuple):
    """
    Returns raw X (not dummified) and y, with numeric columns coerced to numeric
    to prevent SimpleImputer failures inside pipelines.
    """
    if target_col not in df_raw.columns:
        raise ValueError(f"Target column '{target_col}' not found.")

    df = df_raw.dropna(subset=[target_col]).copy()

    # ✅ Ensure numeric cols are truly numeric (fixes your crash)
    df = coerce_numeric_columns(df, NUMERIC_COLS)

    y = merge_rare_classes(df[target_col], min_count=2, other_label="Other")

    # remove target columns from features
    X = df.drop(columns=[c for c in [TARGET_RISK_TYPE, TARGET_RISK_LEVEL] if c in df.columns], errors="ignore")

    # drop high-cardinality cols for modeling/cv
    if drop_cols_for_model:
        X = X.drop(columns=[c for c in drop_cols_for_model if c in X.columns], errors="ignore")

    if y.nunique() < 2:
        raise ValueError("Need at least 2 classes in the target.")
    return X, y


def build_models_fast(fast_mode: bool):
    if fast_mode:
        return {
            "Logistic Regression": LogisticRegression(max_iter=1500, multi_class="auto", solver="lbfgs"),
            "Random Forest": RandomForestClassifier(
                n_estimators=120, max_depth=12, min_samples_leaf=2, n_jobs=-1, random_state=42
            ),
            "Gradient Boosting": GradientBoostingClassifier(
                n_estimators=120, learning_rate=0.08, max_depth=3, random_state=42
            ),
        }
    return {
        "Logistic Regression": LogisticRegression(max_iter=2500, multi_class="auto", solver="lbfgs"),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
        "Gradient Boosting": Gradient BoostingClassifier(n_estimators=200, random_state=42),
    }


@st.cache_data(show_spinner=False)
def train_holdout_models_cached(
    df_raw: pd.DataFrame,
    target_col: str,
    test_size: float,
    drop_cols_for_model: tuple,
    fast_mode: bool,
    use_smote: bool = False,
):
    """
    Leakage-safe holdout evaluation:
    Split first -> pipeline fit only on train -> test.
    """
    X, y = get_Xy_for_target(df_raw, target_col, drop_cols_for_model)

    (X_train, X_test, y_train, y_test), used_stratify, final_test_size = safe_train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    preprocessor = build_preprocess_pipeline_cached(df_raw, drop_cols_for_model)
    models = build_models_fast(fast_mode)

    metrics_list = []
    fitted_pipes = {}

    for name, model in models.items():
        if use_smote:
            pipe = ImbPipeline(steps=[
                ("prep", preprocessor),
                ("smote", SMOTE(random_state=42)),
                ("model", model),
            ])
        else:
            pipe = Pipeline(steps=[
                ("prep", preprocessor),
                ("model", model),
            ])

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        fitted_pipes[name] = pipe
        metrics_list.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision (weighted)": precision_score(y_test, y_pred, average="weighted", zero_division=0),
            "Recall (weighted)": recall_score(y_test, y_pred, average="weighted", zero_division=0),
            "F1-score (weighted)": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        })

    metrics_df = pd.DataFrame(metrics_list).set_index("Model")

    split_note = (
        f"✅ Stratified split used (test_size={final_test_size:.2f})."
        if used_stratify
        else f"⚠️ Non-stratified split used (test_size={final_test_size:.2f}) because some classes are too small."
    )

    split_info = {
        "X_train_shape": X_train.shape,
        "X_test_shape": X_test.shape,
        "y_train_counts": y_train.value_counts(),
        "y_test_counts": y_test.value_counts(),
        "used_stratify": used_stratify,
        "final_test_size": final_test_size,
    }

    return fitted_pipes, metrics_df, split_info, split_note


def smote_and_tune_logreg_pipeline(
    df_raw: pd.DataFrame,
    target_col: str,
    test_size: float,
    drop_cols_for_model: tuple,
    fast_mode: bool,
):
    """
    Leakage-safe:
    split -> build pipeline (prep -> smote -> logreg) -> GridSearchCV on TRAIN ONLY
    then evaluate on test.
    """
    X, y = get_Xy_for_target(df_raw, target_col, drop_cols_for_model)
    (X_train, X_test, y_train, y_test), used_stratify, final_test_size = safe_train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    preprocessor = build_preprocess_pipeline_cached(df_raw, drop_cols_for_model)

    base_pipe = ImbPipeline(steps=[
        ("prep", preprocessor),
        ("smote", SMOTE(random_state=42)),
        ("model", LogisticRegression(max_iter=2000, multi_class="auto", solver="lbfgs")),
    ])

    param_grid = {
        "model__C": [0.01, 0.1, 1, 10],
    }
    cv_folds = 3 if fast_mode else 5

    grid = GridSearchCV(
        estimator=base_pipe,
        param_grid=param_grid,
        scoring="f1_weighted",
        cv=cv_folds,
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)

    best_pipe = grid.best_estimator_
    y_pred = best_pipe.predict(X_test)

    tuned_metrics = pd.DataFrame([{
        "Model": "LogReg (tuned + SMOTE)",
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision (weighted)": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "Recall (weighted)": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "F1-score (weighted)": f1_score(y_test, y_pred, average="weighted", zero_division=0),
    }]).set_index("Model")

    split_note = (
        f"✅ Stratified split used (test_size={final_test_size:.2f})."
        if used_stratify
        else f"⚠️ Non-stratified split used (test_size={final_test_size:.2f}) because some classes are too small."
    )

    split_info = {
        "X_train_shape": X_train.shape,
        "X_test_shape": X_test.shape,
        "y_train_counts": y_train.value_counts(),
        "y_test_counts": y_test.value_counts(),
        "used_stratify": used_stratify,
        "final_test_size": final_test_size,
    }

    return best_pipe, tuned_metrics, grid.best_params_, split_info, split_note


# -------------------------------------------------------
# CROSS VALIDATION (leakage-safe, cached)
# -------------------------------------------------------
@st.cache_data(show_spinner=False)
def run_cross_validation_cached(
    df_raw: pd.DataFrame,
    target_col: str,
    model_name: str,
    n_splits: int,
    stratified: bool,
    use_smote: bool,
    drop_cols_for_model: tuple,
    fast_mode: bool,
):
    X, y = get_Xy_for_target(df_raw, target_col, drop_cols_for_model)

    models = build_models_fast(fast_mode)
    if model_name not in models:
        raise ValueError("Unknown model selected.")
    model = models[model_name]

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42) if stratified else KFold(
        n_splits=n_splits, shuffle=True, random_state=42
    )

    preprocessor = build_preprocess_pipeline_cached(df_raw, drop_cols_for_model)

    if use_smote:
        pipe = ImbPipeline(steps=[
            ("prep", preprocessor),
            ("smote", SMOTE(random_state=42)),
            ("model", model),
        ])
    else:
        pipe = Pipeline(steps=[
            ("prep", preprocessor),
            ("model", model),
        ])

    scoring = {
        "accuracy": "accuracy",
        "precision_w": "precision_weighted",
        "recall_w": "recall_weighted",
        "f1_w": "f1_weighted",
    }

    scores = cross_validate(pipe, X, y, cv=cv, scoring=scoring, n_jobs=-1, error_score="raise")

    summary = {}
    for k in scoring.keys():
        arr = scores[f"test_{k}"]
        summary[k] = {"mean": float(np.mean(arr)), "std": float(np.std(arr))}
    summary_df = pd.DataFrame(summary).T
    summary_df = summary_df.rename(index={
        "accuracy": "Accuracy",
        "precision_w": "Precision (weighted)",
        "recall_w": "Recall (weighted)",
        "f1_w": "F1-score (weighted)",
    })

    return summary_df, scores


# -------------------------------------------------------
# VISUALS
# -------------------------------------------------------
def plot_hist_box(df, col):
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    if len(s) == 0:
        axes[0].text(0.5, 0.5, f"No numeric data for {col}", ha="center", va="center")
        axes[1].text(0.5, 0.5, f"No numeric data for {col}", ha="center", va="center")
    else:
        sns.histplot(s, kde=True, ax=axes[0])
        axes[0].set_title(f"Histogram of {col}")
        sns.boxplot(x=s, ax=axes[1])
        axes[1].set_title(f"Boxplot of {col}")

    plt.tight_layout()
    return fig


def plot_scatter(df, x_col, y_col):
    x = pd.to_numeric(df[x_col], errors="coerce")
    y = pd.to_numeric(df[y_col], errors="coerce")
    mask = x.notna() & y.notna()

    fig, ax = plt.subplots(figsize=(6, 4))
    if mask.sum() == 0:
        ax.text(0.5, 0.5, f"No numeric data for {x_col} and {y_col}", ha="center", va="center")
    else:
        ax.scatter(x[mask], y[mask], alpha=0.7)
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(f"{y_col} vs {x_col}")
    plt.tight_layout()
    return fig


def plot_metrics_bar(metrics_df, title_suffix=""):
    fig, ax = plt.subplots(figsize=(8, 5))
    metrics_df[["Accuracy", "Precision (weighted)", "Recall (weighted)", "F1-score (weighted)"]].plot(
        kind="bar", ax=ax
    )
    ax.set_title(f"Model Performance {title_suffix}")
    ax.set_ylabel("Score")
    plt.xticks(rotation=0)
    plt.tight_layout()
    return fig


def plot_box_by_category_readable(
    df,
    value_col,
    category_col,
    top_n=8,
    other_label="Other",
    figsize=(12, 5),
    horizontal=True,
):
    val = pd.to_numeric(df[value_col], errors="coerce")
    cat = (
        df[category_col]
        .astype(str)
        .str.strip()
        .replace({"": np.nan, "nan": np.nan, "None": np.nan})
    )

    data = pd.DataFrame({value_col: val, category_col: cat}).dropna(subset=[value_col, category_col])

    fig, ax = plt.subplots(figsize=figsize)
    if data.empty:
        ax.text(0.5, 0.5, f"No usable data for {value_col} by {category_col}", ha="center", va="center")
        plt.tight_layout()
        return fig

    counts = data[category_col].value_counts()
    keep = counts.head(top_n).index
    data[category_col] = np.where(data[category_col].isin(keep), data[category_col], other_label)

    order = data.groupby(category_col)[value_col].median().sort_values().index.tolist()

    if horizontal:
        sns.boxplot(data=data, y=category_col, x=value_col, order=order, ax=ax)
    else:
        sns.boxplot(data=data, x=category_col, y=value_col, order=order, ax=ax)
        ax.tick_params(axis="x", labelrotation=35)
        for label in ax.get_xticklabels():
            label.set_horizontalalignment("right")

    ax.set_title(f"{value_col} by {category_col} (Top {top_n} + {other_label})")
    plt.tight_layout()
    return fig


def plot_categorical_topn_bar(
    series: pd.Series,
    title: str,
    top_n: int = 15,
    other_label: str = "Other",
    figsize=(10, 6),
):
    s = series.dropna().astype(str).str.strip()
    s = s.replace({"": np.nan, "nan": np.nan, "None": np.nan}).dropna()
    counts = s.value_counts()

    if counts.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No category data available", ha="center", va="center")
        plt.tight_layout()
        return fig, counts

    top = counts.head(top_n)
    remainder = counts.iloc[top_n:].sum()

    if remainder > 0:
        top = pd.concat([top, pd.Series({other_label: remainder})])

    fig, ax = plt.subplots(figsize=figsize)
    top.sort_values().plot(kind="barh", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Count")
    ax.set_ylabel(series.name if series.name else "Category")
    plt.tight_layout()
    return fig, counts


# -------------------------------------------------------
# APP
# -------------------------------------------------------
def main():
    st.title("Microplastic Risk Prediction – Streamlit App")
    st.markdown(
        """
        This app demonstrates the analysis and modeling workflow for predicting **Risk_Type**
        and **Risk_Level** using microplastic and environmental features.

        ✅ Modeling + CV are leakage-safe (Pipeline does preprocessing inside train/CV folds).
        ✅ Fix included: numeric coercion prevents SimpleImputer fit errors.
        """
    )

    st.sidebar.header("Navigation")
    page = st.sidebar.radio(
        "Go to",
        [
            "Data Overview & Task 1",
            "Preprocessing (Task 2)",
            "Feature Selection & Relevance (Task 3 & 6)",
            "Classification Modeling (Tasks 4, 5 & 7)",
            "Polymer Type Distribution",
            "SMOTE & Hyperparameter Tuning (Risk_Type)",
            "Cross Validation (K-Fold)",
        ],
    )

    st.sidebar.subheader("Performance")
    fast_mode = st.sidebar.toggle("Fast Mode (recommended)", value=True)
    test_size = st.sidebar.slider("Test size", min_value=0.1, max_value=0.4, value=0.2, step=0.05)

    st.sidebar.subheader("Model Features")
    drop_location_author = st.sidebar.checkbox(
        "Drop Location & Author for modeling/CV (speeds up a lot)",
        value=True,
        help="These columns have many unique values and cause huge one-hot matrices. Keep them for EDA, drop for modeling."
    )
    drop_cols_for_model = tuple(DEFAULT_MODEL_DROP_COLS) if drop_location_author else tuple()

    st.sidebar.subheader("Data source")
    uploaded_file = st.sidebar.file_uploader(
        "Upload Microplastic CSV",
        type=["csv"],
        help="If you don't upload anything, the app will try to use 'Microplastic.csv' from the app folder."
    )

    try:
        df_raw = load_data(uploaded_file=uploaded_file)
    except UnicodeDecodeError:
        st.error("⚠️ Unable to decode the file. Please upload a proper CSV (text).")
        st.stop()
    except EmptyDataError:
        st.error("⚠️ The uploaded file appears empty/unreadable as CSV.")
        st.stop()
    except ParserError:
        st.error("⚠️ The file is not a valid CSV format. Re-export as CSV and try again.")
        st.stop()
    except FileNotFoundError:
        df_raw = None

    if df_raw is None:
        st.error("❌ No dataset found. Upload a CSV or add 'Microplastic.csv' beside app.py.")
        st.stop()

    # -------------------- PAGE 1 --------------------
    if page == "Data Overview & Task 1":
        st.header("Data Overview & Task 1: Risk_Score Analysis")

        tab1, tab2, tab3, tab4 = st.tabs([
            "Raw Data",
            "Risk_Score Distribution",
            "MP_Count vs Risk_Score",
            "Risk_Score by Risk_Level",
        ])

        with tab1:
            st.subheader("Raw Dataset (first 10 rows)")
            st.dataframe(df_raw.head(10))
            st.markdown(f"**Shape:** `{df_raw.shape[0]}` rows × `{df_raw.shape[1]}` columns")

        with tab2:
            if "Risk_Score" in df_raw.columns:
                st.subheader("Distribution of Risk_Score (Histogram & Boxplot)")
                st.pyplot(plot_hist_box(df_raw, "Risk_Score"))
            else:
                st.info("Column 'Risk_Score' not found in the dataset.")

        with tab3:
            if "MP_Count_per_L" in df_raw.columns and "Risk_Score" in df_raw.columns:
                st.subheader("Relationship between Risk_Score and MP_Count_per_L")
                st.pyplot(plot_scatter(df_raw, "MP_Count_per_L", "Risk_Score"))
            else:
                st.info("Columns 'MP_Count_per_L' and/or 'Risk_Score' not found.")

        with tab4:
            if "Risk_Level" in df_raw.columns and "Risk_Score" in df_raw.columns:
                st.subheader("Difference in Risk_Score by Risk_Level (Boxplot)")
                st.pyplot(
                    plot_box_by_category_readable(
                        df_raw,
                        value_col="Risk_Score",
                        category_col="Risk_Level",
                        top_n=8,
                        figsize=(12, 5),
                        horizontal=True,
                    )
                )
            else:
                st.info("Columns 'Risk_Level' and/or 'Risk_Score' not found.")

    # -------------------- PAGE 2 --------------------
    elif page == "Preprocessing (Task 2)":
        st.header("Task 2: Preprocessing (EDA view)")
        df_clean = handle_missing_values(df_raw)
        df_clean = cap_outliers_iqr(df_clean, NUMERIC_COLS)
        df_clean, skewness, skewed_cols = transform_skewed(df_clean, NUMERIC_COLS)
        df_clean, _ = scale_numeric(df_clean, NUMERIC_COLS)

        tab1, tab2 = st.tabs(["Descriptive Stats", "Skewness & Notes"])

        with tab1:
            numeric_present = [c for c in NUMERIC_COLS if c in df_raw.columns]
            if numeric_present:
                st.subheader("Descriptive Stats (Raw)")
                st.write(df_raw[numeric_present].describe())
                st.subheader("Descriptive Stats (Cleaned)")
                st.write(df_clean[numeric_present].describe())
            else:
                st.info("No numeric columns found for descriptive stats.")

        with tab2:
            st.subheader("Skewness (Before Transform)")
            st.write(skewness)
            if len(skewed_cols) > 0:
                st.write("Skewed columns transformed (log1p):")
                st.write(skewed_cols)

            st.info(
                "Note: For modeling/CV, preprocessing is done inside a Pipeline (leakage-safe). "
                "This page is for EDA/interpretation only."
            )

    # -------------------- PAGE 3 --------------------
    elif page == "Feature Selection & Relevance (Task 3 & 6)":
        st.header("Tasks 3 & 6: Feature Selection / Relevance (Model-based)")

        tab_rt, tab_rl = st.tabs(["Risk_Type (RF importance)", "Risk_Level (RF importance)"])

        def rf_importance(target_col: str):
            X, y = get_Xy_for_target(df_raw, target_col, drop_cols_for_model)
            preprocessor = build_preprocess_pipeline_cached(df_raw, drop_cols_for_model)
            rf = RandomForestClassifier(n_estimators=200 if fast_mode else 400, random_state=42, n_jobs=-1)

            pipe = Pipeline(steps=[("prep", preprocessor), ("model", rf)])
            pipe.fit(X, y)

            try:
                feat_names = pipe.named_steps["prep"].get_feature_names_out()
            except Exception:
                feat_names = np.array([f"f{i}" for i in range(pipe.named_steps["model"].feature_importances_.shape[0])])

            importances = pd.Series(pipe.named_steps["model"].feature_importances_, index=feat_names).sort_values(ascending=False)
            return importances

        with tab_rt:
            if TARGET_RISK_TYPE in df_raw.columns:
                st.subheader("Random Forest Feature Importance – Risk_Type")
                importances_rt = rf_importance(TARGET_RISK_TYPE)
                st.dataframe(importances_rt.head(15))

                fig = plt.figure(figsize=(10, 5))
                importances_rt.head(15).sort_values().plot(kind="barh")
                plt.title("Top 15 Feature Importances (Risk_Type)")
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.warning("Risk_Type column not found.")

        with tab_rl:
            if TARGET_RISK_LEVEL in df_raw.columns:
                st.subheader("Random Forest Feature Importance – Risk_Level")
                importances_rl = rf_importance(TARGET_RISK_LEVEL)
                st.dataframe(importances_rl.head(15))

                fig = plt.figure(figsize=(10, 5))
                importances_rl.head(15).sort_values().plot(kind="barh")
                plt.title("Top 15 Feature Importances (Risk_Level)")
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.warning("Risk_Level column not found.")

        st.info("Feature importance is computed after preprocessing (imputation, scaling, one-hot).")

    # -------------------- PAGE 4 --------------------
    elif page == "Classification Modeling (Tasks 4, 5 & 7)":
        st.header("Tasks 4, 5 & 7: Classification Modeling (Leakage-safe holdout)")

        tab1, tab2 = st.tabs(["Risk-Type Models", "Risk-Level Models"])

        with tab1:
            if TARGET_RISK_TYPE not in df_raw.columns:
                st.warning("Risk_Type column not found; cannot train models for Risk-Type.")
            else:
                st.subheader("Models for Risk-Type (Holdout split)")
                with st.spinner("Training models (cached)..."):
                    _, metrics_rt, split_info_rt, split_note_rt = train_holdout_models_cached(
                        df_raw, TARGET_RISK_TYPE, test_size, drop_cols_for_model, fast_mode, use_smote=False
                    )

                st.dataframe(metrics_rt.round(3))
                st.pyplot(plot_metrics_bar(metrics_rt, "(Risk-Type)"))
                st.info(split_note_rt)
                st.write("Class distribution in training set:")
                st.write(split_info_rt["y_train_counts"])
                st.write("Class distribution in test set:")
                st.write(split_info_rt["y_test_counts"])

        with tab2:
            if TARGET_RISK_LEVEL not in df_raw.columns:
                st.warning("Risk_Level column not found; cannot train models for Risk-Level.")
            else:
                st.subheader("Models for Risk-Level (Holdout split)")
                with st.spinner("Training models (cached)..."):
                    _, metrics_rl, split_info_rl, split_note_rl = train_holdout_models_cached(
                        df_raw, TARGET_RISK_LEVEL, test_size, drop_cols_for_model, fast_mode, use_smote=False
                    )

                st.dataframe(metrics_rl.round(3))
                st.pyplot(plot_metrics_bar(metrics_rl, "(Risk-Level)"))
                st.info(split_note_rl)
                st.write("Class distribution in training set:")
                st.write(split_info_rl["y_train_counts"])
                st.write("Class distribution in test set:")
                st.write(split_info_rl["y_test_counts"])

        st.subheader("Overall Notes (Speed)")
        st.markdown(
            f"""
            - Current modeling drop columns: **{', '.join(drop_cols_for_model) if drop_cols_for_model else 'None'}**
            - If the page is slow, keep **Drop Location & Author** ON and keep **Fast Mode** ON.
            """
        )

    # -------------------- PAGE 5 --------------------
    elif page == "Polymer Type Distribution":
        st.header("Polymer Type Distribution")
        df = handle_missing_values(df_raw)

        if "Polymer_Type" in df.columns:
            polymer = df["Polymer_Type"].astype(str).str.strip().replace({"": np.nan, "nan": np.nan, "None": np.nan})
            polymer = polymer.dropna()
            vc = polymer.value_counts()

            tabA, tabB = st.tabs(["Counts Table", "Readable Plot (Top N + Other)"])

            with tabA:
                st.subheader("Value Counts of Polymer_Type")
                st.dataframe(vc.rename("count"))

            with tabB:
                st.subheader("Bar Plot of Polymer_Type Distribution (Readable)")
                top_n = st.slider("Show Top N polymer types", min_value=5, max_value=30, value=15, step=1)
                fig, _ = plot_categorical_topn_bar(
                    polymer,
                    title=f"Distribution of Polymer_Type (Top {top_n} + Other)",
                    top_n=top_n,
                    other_label="Other",
                    figsize=(10, 7),
                )
                st.pyplot(fig)
        else:
            st.warning("Column 'Polymer_Type' not found in the dataset.")

    # -------------------- PAGE 6 --------------------
    elif page == "SMOTE & Hyperparameter Tuning (Risk_Type)":
        st.header("Address Class Imbalance & Tune Logistic Regression (Risk-Type)")

        if TARGET_RISK_TYPE not in df_raw.columns:
            st.warning("Risk_Type column not found; cannot run SMOTE or tuning.")
            return

        tab1, tab2 = st.tabs(["Original Distribution & Base Models", "SMOTE + Tuning & Comparison"])

        with tab1:
            st.subheader("Class Distribution of Risk-Type (after rare-class merge)")
            y_preview = merge_rare_classes(df_raw[TARGET_RISK_TYPE].dropna(), min_count=2, other_label="Other")
            st.write(y_preview.value_counts())

            with st.spinner("Training base models (cached)..."):
                _, base_metrics_rt, split_info_base_rt, split_note_base = train_holdout_models_cached(
                    df_raw, TARGET_RISK_TYPE, test_size, drop_cols_for_model, fast_mode, use_smote=False
                )

            st.subheader("Base Models Performance (Risk-Type)")
            st.dataframe(base_metrics_rt.round(3))
            st.pyplot(plot_metrics_bar(base_metrics_rt, "(Risk-Type – Base)"))
            st.info(split_note_base)

        with tab2:
            st.subheader("SMOTE + Hyperparameter Tuning (LogReg)")
            with st.spinner("Running GridSearchCV on training split (leakage-safe)..."):
                best_pipe, tuned_metrics, best_params, split_info_smote, split_note_smote = smote_and_tune_logreg_pipeline(
                    df_raw, TARGET_RISK_TYPE, test_size, drop_cols_for_model, fast_mode
                )

            st.write("Best Hyperparameters:")
            st.json(best_params)
            st.info(split_note_smote)

            st.subheader("Tuned Logistic Regression Performance")
            st.dataframe(tuned_metrics.round(3))

            combined = pd.concat([base_metrics_rt, tuned_metrics])
            st.subheader("Comparison: Tuned Logistic Regression vs Base Models")
            st.dataframe(combined.round(3))
            st.pyplot(plot_metrics_bar(combined, "(Risk-Type – Base vs Tuned)"))

    # -------------------- PAGE 7 --------------------
    elif page == "Cross Validation (K-Fold)":
        st.header("Cross Validation (K-Fold / Stratified K-Fold)")

        st.markdown(
            """
            This page runs leakage-safe cross-validation using a preprocessing Pipeline.
            - For classification, Stratified K-Fold is recommended.
            - If enabled, SMOTE is applied inside each fold (correct way).
            """
        )

        target = st.selectbox("Select target", [TARGET_RISK_TYPE, TARGET_RISK_LEVEL])
        model_name = st.selectbox("Select model", ["Logistic Regression", "Random Forest", "Gradient Boosting"])
        n_splits = st.slider("Number of folds (k)", min_value=3, max_value=10, value=5, step=1)
        stratified = st.checkbox("Use Stratified K-Fold (recommended for classification)", value=True)

        use_smote = st.checkbox("Use SMOTE (only if classes are imbalanced)", value=False)
        if use_smote and target == TARGET_RISK_LEVEL:
            st.info("SMOTE is usually more relevant for Risk_Type, but you can still try it.")

        st.divider()

        colA, colB = st.columns([1, 2])
        with colA:
            st.subheader("Target distribution (after rare-class merge)")
            if target in df_raw.columns:
                y_preview = merge_rare_classes(df_raw[target].dropna(), min_count=2, other_label="Other")
                st.write(y_preview.value_counts())
            else:
                st.warning(f"Column '{target}' not found.")

        with colB:
            st.subheader("Run CV")
            if st.button("Run Cross-Validation", type="primary"):
                try:
                    with st.spinner("Running cross-validation (cached)..."):
                        summary_df, raw_scores = run_cross_validation_cached(
                            df_raw=df_raw,
                            target_col=target,
                            model_name=model_name,
                            n_splits=n_splits,
                            stratified=stratified,
                            use_smote=use_smote,
                            drop_cols_for_model=drop_cols_for_model,
                            fast_mode=fast_mode
                        )

                    st.success("Done!")
                    st.markdown("### CV Results (mean ± std across folds)")
                    show = summary_df.copy()
                    show["mean±std"] = show.apply(lambda r: f"{r['mean']:.3f} ± {r['std']:.3f}", axis=1)
                    st.dataframe(show[["mean±std"]])

                    with st.expander("Show per-fold scores"):
                        fold_df = pd.DataFrame({
                            "Accuracy": raw_scores["test_accuracy"],
                            "Precision (weighted)": raw_scores["test_precision_w"],
                            "Recall (weighted)": raw_scores["test_recall_w"],
                            "F1-score (weighted)": raw_scores["test_f1_w"],
                        })
                        fold_df.index = [f"Fold {i+1}" for i in range(len(fold_df))]
                        st.dataframe(fold_df.round(3))

                except Exception as e:
                    st.error(f"Cross-validation failed: {e}")


if __name__ == "__main__":
    main()
