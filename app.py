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

DEFAULT_MODEL_DROP_COLS = ["Location", "Author"]


# -------------------------------------------------------
# DATA LOADING
# -------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data(uploaded_file=None, path: str = "Microplastic.csv"):
    src = uploaded_file if uploaded_file is not None else path
    encodings_to_try = ["latin1", "utf-8", "cp1252"]

    last_err = None
    for enc in encodings_to_try:
        try:
            if uploaded_file is not None:
                try:
                    uploaded_file.seek(0)
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
# EDA PREPROCESS (optional pages)
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
# NUMERIC SANITIZER (prevents pipeline errors)
# -------------------------------------------------------
def coerce_numeric_columns(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for c in numeric_cols:
        if c in df.columns:
            s = df[c].astype(str).str.replace(",", "", regex=False)
            df[c] = pd.to_numeric(s, errors="coerce")
    return df


# -------------------------------------------------------
# SPLIT + CLASS HELPERS
# -------------------------------------------------------
def merge_rare_classes(y: pd.Series, min_count: int = 2, other_label: str = "Other"):
    y = pd.Series(y).copy()
    counts = y.value_counts(dropna=True)
    rare = counts[counts < min_count].index
    return y.where(~y.isin(rare), other_label)


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


@st.cache_data(show_spinner=False)
def build_preprocess_pipeline_cached(df_raw: pd.DataFrame, drop_cols_for_model: tuple):
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

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_features),
            ("cat", categorical_pipe, categorical_features),
        ],
        remainder="drop"
    )


def get_Xy_for_target(df_raw: pd.DataFrame, target_col: str, drop_cols_for_model: tuple):
    if target_col not in df_raw.columns:
        raise ValueError(f"Target column '{target_col}' not found.")

    df = df_raw.dropna(subset=[target_col]).copy()
    df = coerce_numeric_columns(df, NUMERIC_COLS)

    y = merge_rare_classes(df[target_col], min_count=2, other_label="Other")

    X = df.drop(columns=[c for c in [TARGET_RISK_TYPE, TARGET_RISK_LEVEL] if c in df.columns], errors="ignore")
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
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, random_state=42),
    }


# -------------------------------------------------------
# ✅ AUTO-SAFE CROSS VALIDATION (the "validation" part)
# -------------------------------------------------------
def auto_safe_n_splits(y: pd.Series, requested: int) -> int:
    """
    For StratifiedKFold, each class needs >= n_splits samples.
    So n_splits <= min_class_count.
    If too small, we lower folds automatically.
    """
    counts = pd.Series(y).value_counts()
    if counts.empty:
        return 2
    min_class = int(counts.min())
    # At least 2 folds; but if min_class < 2, CV isn't meaningful.
    return int(max(2, min(requested, min_class)))


@st.cache_data(show_spinner=False)
def run_cross_validation_auto_cached(
    df_raw: pd.DataFrame,
    target_col: str,
    model_name: str,
    requested_splits: int,
    stratified: bool,
    use_smote: bool,
    drop_cols_for_model: tuple,
    fast_mode: bool,
):
    """
    Leakage-safe CV:
    - preprocessing inside pipeline
    - SMOTE inside fold (if enabled)
    - automatically adjusts k to avoid "all fits failed"
    """
    X, y = get_Xy_for_target(df_raw, target_col, drop_cols_for_model)

    if y.nunique() < 2:
        raise ValueError("Target has < 2 classes; cannot run CV.")

    # Auto-adjust folds if stratified
    final_splits = requested_splits
    if stratified:
        final_splits = auto_safe_n_splits(y, requested_splits)
        # If min class < 2, you can't do stratified CV properly
        if pd.Series(y).value_counts().min() < 2:
            raise ValueError("Some class has only 1 sample; add more data or merge classes.")

    models = build_models_fast(fast_mode)
    if model_name not in models:
        raise ValueError("Unknown model selected.")
    model = models[model_name]

    cv = StratifiedKFold(n_splits=final_splits, shuffle=True, random_state=42) if stratified else KFold(
        n_splits=final_splits, shuffle=True, random_state=42
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

    meta = {
        "requested_splits": int(requested_splits),
        "final_splits": int(final_splits),
        "stratified": bool(stratified),
        "use_smote": bool(use_smote),
        "class_counts": pd.Series(y).value_counts(),
    }
    return summary_df, scores, meta


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
    df, value_col, category_col, top_n=8, other_label="Other", figsize=(12, 5), horizontal=True
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


def plot_categorical_topn_bar(series: pd.Series, title: str, top_n: int = 15, other_label: str = "Other", figsize=(10, 6)):
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
# APP (simple + cross validation page included)
# -------------------------------------------------------
def main():
    st.title("Microplastic Risk Prediction – Streamlit App")
    st.markdown(
        """
        This app includes a **leakage-safe Cross Validation page** (Objective #3).
        - Preprocessing happens inside Pipeline (no leakage)
        - Optional SMOTE happens inside each fold (correct)
        - k is auto-adjusted so CV won't fail on small classes
        """
    )

    st.sidebar.header("Navigation")
    page = st.sidebar.radio(
        "Go to",
        [
            "Data Overview",
            "Polymer Type Distribution",
            "Cross Validation (Objective #3)",
        ],
    )

    st.sidebar.subheader("Performance")
    fast_mode = st.sidebar.toggle("Fast Mode", value=True)

    st.sidebar.subheader("Model Features")
    drop_location_author = st.sidebar.checkbox(
        "Drop Location & Author for modeling/CV (speeds up a lot)",
        value=True,
    )
    drop_cols_for_model = tuple(DEFAULT_MODEL_DROP_COLS) if drop_location_author else tuple()

    st.sidebar.subheader("Data source")
    uploaded_file = st.sidebar.file_uploader("Upload Microplastic CSV", type=["csv"])

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

    if page == "Data Overview":
        st.subheader("Raw Dataset (first 20 rows)")
        st.dataframe(df_raw.head(20))
        st.write("Shape:", df_raw.shape)

        if "Risk_Score" in df_raw.columns:
            st.subheader("Risk_Score Distribution")
            st.pyplot(plot_hist_box(df_raw, "Risk_Score"))

        if "MP_Count_per_L" in df_raw.columns and "Risk_Score" in df_raw.columns:
            st.subheader("MP_Count_per_L vs Risk_Score")
            st.pyplot(plot_scatter(df_raw, "MP_Count_per_L", "Risk_Score"))

        if "Risk_Level" in df_raw.columns and "Risk_Score" in df_raw.columns:
            st.subheader("Risk_Score by Risk_Level")
            st.pyplot(plot_box_by_category_readable(df_raw, "Risk_Score", "Risk_Level"))

    elif page == "Polymer Type Distribution":
        st.subheader("Polymer Type Distribution")
        if "Polymer_Type" not in df_raw.columns:
            st.warning("Column 'Polymer_Type' not found.")
        else:
            polymer = df_raw["Polymer_Type"].astype(str).str.strip().replace({"": np.nan, "nan": np.nan, "None": np.nan}).dropna()
            top_n = st.slider("Top N", 5, 30, 15)
            fig, _ = plot_categorical_topn_bar(polymer, "Distribution of Polymer_Type", top_n=top_n)
            st.pyplot(fig)

    elif page == "Cross Validation (Objective #3)":
        st.header("Objective #3: Validation (Cross-Validation)")

        target = st.selectbox("Select target", [TARGET_RISK_TYPE, TARGET_RISK_LEVEL])
        model_name = st.selectbox("Select model", ["Logistic Regression", "Random Forest", "Gradient Boosting"])

        requested_splits = st.slider("Requested folds (k)", min_value=3, max_value=10, value=5, step=1)
        stratified = st.checkbox("Use StratifiedKFold (recommended)", value=True)
        use_smote = st.checkbox("Use SMOTE inside folds (Risk_Type imbalance)", value=False)

        if st.button("Run Cross-Validation", type="primary"):
            try:
                with st.spinner("Running CV..."):
                    summary_df, raw_scores, meta = run_cross_validation_auto_cached(
                        df_raw=df_raw,
                        target_col=target,
                        model_name=model_name,
                        requested_splits=requested_splits,
                        stratified=stratified,
                        use_smote=use_smote,
                        drop_cols_for_model=drop_cols_for_model,
                        fast_mode=fast_mode,
                    )

                st.success("CV completed!")
                st.subheader("Class distribution (after rare-class merge)")
                st.write(meta["class_counts"])

                if meta["requested_splits"] != meta["final_splits"]:
                    st.info(f"Requested k={meta['requested_splits']} but used k={meta['final_splits']} (auto-safe for small classes).")

                st.subheader("CV Results (mean ± std)")
                show = summary_df.copy()
                show["mean±std"] = show.apply(lambda r: f"{r['mean']:.3f} ± {r['std']:.3f}", axis=1)
                st.dataframe(show[["mean±std"]])

                with st.expander("Per-fold scores"):
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
