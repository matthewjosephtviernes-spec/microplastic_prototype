# Redesigned Streamlit app for Microplastic Risk Prediction
# - Cleaner structure, robust preprocessing, safer modeling fallbacks
# - Improved visuals: aggregated category bars, per-class metric plots, ROC/PR curves for binary
# - Sidebar reorganized as expanders to separate steps and make options discoverable

import io
import re
from typing import Optional, Tuple, List, Dict

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    r2_score, mean_absolute_error, mean_squared_error,
    confusion_matrix, classification_report, roc_auc_score, roc_curve,
    precision_recall_curve
)
from sklearn.metrics import silhouette_score

# ================================================
# MICROPLASTIC POLLUTION DATA CLEANING SCRIPT
# Steps 2–7: Duplicates → Missing Values → Standardization →
# Irrelevant Columns → Outliers → Text Normalization
# (Integrated into the Streamlit app so users can apply these
#  cleaning steps to an uploaded dataset interactively.)
# ================================================

# We'll attempt to import NLTK resources; if unavailable we fallback to lighter text cleaning.
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer

    # Attempt to download required resources if not present.
    try:
        nltk.data.find('corpora/stopwords')
    except Exception:
        nltk.download('stopwords', quiet=True)

    try:
        nltk.data.find('corpora/wordnet')
    except Exception:
        nltk.download('wordnet', quiet=True)
    NLTK_AVAILABLE = True
except Exception:
    NLTK_AVAILABLE = False

# -------------------------
# App configuration & style
# -------------------------
st.set_page_config(page_title="Microplastic Risk Prediction (Professional)", layout="wide")
st.markdown(
    """
    <style>
      .stApp { font-family: "Segoe UI", Roboto, "Helvetica Neue", Arial; }
      .big-header { font-size:22px; font-weight:600; margin-bottom:6px; }
      .muted { color: #6c757d; font-size:12px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="big-header">Microplastic Risk Prediction System</div>', unsafe_allow_html=True)
st.write("A robust Streamlit app for exploring microplastic datasets, preprocessing, modeling, and visualizations.")

# -------------------------
# Helpers
# -------------------------
@st.cache_data
def safe_read_file(uploaded_file) -> Optional[pd.DataFrame]:
    if uploaded_file is None:
        return None
    try:
        name = uploaded_file.name.lower()
        if name.endswith(".csv"):
            return pd.read_csv(uploaded_file, engine="python", encoding="utf-8", on_bad_lines="skip")
        if name.endswith((".xls", ".xlsx")):
            return pd.read_excel(uploaded_file, engine="openpyxl")
    except Exception as e:
        st.error(f"Could not read file: {e}")
        return None

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=lambda c: re.sub(r"[^\w]", "_", str(c).strip()))
    return df

def parse_mp_count(value) -> float:
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    if s.lower() in ("n/a", "na", "-", "", "nd", "no", "no (nd)", "not detected"):
        return np.nan
    s = s.replace("–", "-").replace("—", "-")
    # handle "12 ± 2" etc
    if "±" in s:
        m = re.search(r"([-+]?\d*\.\d+|\d+)\s*±", s)
        if m:
            try:
                return float(m.group(1))
            except:
                pass
    nums = re.findall(r"[-+]?\d*\.\d+|\d+", s)
    if not nums:
        return np.nan
    try:
        nums_f = [float(n) for n in nums]
        return float(np.mean(nums_f))
    except:
        return np.nan

def summarize_cardinality(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in df.columns:
        rows.append({"column": c, "dtype": str(df[c].dtype), "nunique": int(df[c].nunique(dropna=False))})
    return pd.DataFrame(rows).sort_values("nunique")

def aggregate_small_categories(series: pd.Series, top_n: int = 20, other_name: str = "Other") -> pd.Series:
    vc = series.fillna("NaN").astype(str).value_counts()
    if len(vc) <= top_n:
        return vc
    top = vc.iloc[:top_n]
    other_sum = vc.iloc[top_n:].sum()
    top[other_name] = other_sum
    return top

def prepare_features(
    df: pd.DataFrame,
    selected_features: List[str],
    target_col: Optional[str],
    task: str,
    impute_strategy: str = "mean",
) -> Tuple[pd.DataFrame, Optional[np.ndarray], Optional[LabelEncoder]]:
    df_sub = df[selected_features + ([target_col] if target_col else [])].copy()
    # drop columns with single unique value
    drop_cols = [c for c in df_sub.columns if df_sub[c].nunique(dropna=False) <= 1]
    if drop_cols:
        df_sub.drop(columns=drop_cols, inplace=True)
        st.info(f"Dropped non-informative columns: {drop_cols}")

    if target_col and target_col not in df_sub.columns:
        st.error("Selected target column not present after preprocessing.")
        return pd.DataFrame(), None, None

    X = df_sub.drop(columns=[target_col]) if target_col else df_sub.copy()
    y = df_sub[target_col] if target_col else None

    # special handling for MP presence-like columns
    for c in X.columns:
        if "mp_presence" in c.lower() or c.lower() == "mp_presence".lower():
            X[c] = X[c].astype(str).str.lower().map(lambda v: 1 if "yes" in v else (0 if "no" in v or "nd" in v or v.strip() == "" else np.nan))

    # reduce very high cardinality for object columns by frequency mapping
    obj_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    for c in obj_cols:
        if X[c].nunique() > 40:
            freqs = X[c].value_counts(normalize=True)
            X[c] = X[c].map(freqs).fillna(0.0)

    # one-hot encode categoricals (safe)
    X = pd.get_dummies(X, drop_first=True)

    # Encode target if classification
    label_enc = None
    if y is not None and task == "classification":
        if y.dtype == object or str(y.dtype).startswith("category"):
            label_enc = LabelEncoder()
            y = label_enc.fit_transform(y.astype(str))
        else:
            # if numeric but discrete, keep as-is
            y = pd.to_numeric(y, errors="coerce")
    elif y is not None and task == "regression":
        y = pd.to_numeric(y, errors="coerce")

    # Place y as Series and drop rows with NaN target
    if y is not None:
        y = pd.Series(y, index=df_sub.index)
        if y.isna().any():
            n_drop = y.isna().sum()
            st.warning(f"Dropping {n_drop} rows with missing target values.")
            mask = ~y.isna()
            X = X.loc[mask].copy()
            y = y.loc[mask].copy()

    # Impute numeric cols
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        imputer = SimpleImputer(strategy=impute_strategy)
        X[numeric_cols] = imputer.fit_transform(X[numeric_cols])

    # Warn & drop any remaining non-numeric
    non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        st.warning(f"Dropping non-numeric columns after encoding: {non_numeric}")
        X.drop(columns=non_numeric, inplace=True)

    # Scale numeric features
    num_cols = X.columns.tolist()
    if num_cols:
        scaler = StandardScaler()
        X[num_cols] = scaler.fit_transform(X[num_cols])

    # Final drop rows with NaNs (should be none)
    mask_x_na = X.isna().any(axis=1)
    if mask_x_na.any():
        st.warning(f"Dropping {mask_x_na.sum()} rows with NaN after imputation.")
        X = X.loc[~mask_x_na].copy()
        if y is not None:
            y = y.loc[X.index].copy()

    if y is not None:
        return X, np.asarray(y), label_enc
    return X, None, label_enc

def safe_train_test_split(X, y, test_size: float, random_state: int, task: str):
    if y is None:
        return None, None, None, None
    stratify = None
    if task == "classification":
        unique, counts = np.unique(y, return_counts=True)
        if len(unique) >= 2 and np.min(counts) >= 2:
            stratify = y
        else:
            st.warning("Stratify disabled: some classes have fewer than 2 samples.")
            stratify = None
    try:
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify)
    except Exception as e:
        st.warning(f"train_test_split with stratify failed: {e}. Retrying without stratify.")
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=None)

def has_nan_or_inf(arr) -> bool:
    return np.isnan(arr).any() or np.isinf(arr).any()

# -------------------------
# Additional cleaning helpers (from the requested script)
# -------------------------
def _col_map(df: pd.DataFrame) -> Dict[str, str]:
    """Return mapping of lowercase cleaned column name -> actual column name in df"""
    return {c.lower(): c for c in df.columns}

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates().reset_index(drop=True)

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df_out = df.copy()
    cmap = _col_map(df_out)
    # Replace missing categorical values with 'Unknown'
    categorical_cols = ['Location', 'Polymer_Type', 'Risk_Level']
    for col in categorical_cols:
        col_l = col.lower()
        if col_l in cmap:
            df_out[cmap[col_l]] = df_out[cmap[col_l]].fillna('Unknown')
    # Replace missing numerical values with column mean
    numeric_cols = ['MP_Count']
    for col in numeric_cols:
        col_l = col.lower()
        if col_l in cmap:
            try:
                df_out[cmap[col_l]] = pd.to_numeric(df_out[cmap[col_l]], errors='coerce')
                if df_out[cmap[col_l]].isnull().any():
                    mean_val = df_out[cmap[col_l]].mean(skipna=True)
                    df_out[cmap[col_l]].fillna(mean_val, inplace=True)
            except Exception:
                pass
    return df_out

def standardize_units_and_text(df: pd.DataFrame) -> pd.DataFrame:
    df_out = df.copy()
    cmap = _col_map(df_out)
    # Convert risk levels to consistent capitalization
    if 'risk_level' in cmap:
        df_out[cmap['risk_level']] = df_out[cmap['risk_level']].astype(str).str.capitalize()
    # Example of unit standardization (if MP_Count in particles/m3) -> particles/L (divide by 1000)
    if 'mp_count' in cmap:
        try:
            df_out[cmap['mp_count']] = pd.to_numeric(df_out[cmap['mp_count']], errors='coerce')
            df_out[cmap['mp_count']] = df_out[cmap['mp_count']] / 1000.0
        except Exception:
            pass
    return df_out

def remove_irrelevant_columns(df: pd.DataFrame) -> pd.DataFrame:
    df_out = df.copy()
    cols_to_drop = ['Author', 'URL', 'Citation', 'Notes']
    cmap = _col_map(df_out)
    to_drop = [cmap[c.lower()] for c in cols_to_drop if c.lower() in cmap]
    if to_drop:
        df_out = df_out.drop(columns=to_drop, errors='ignore')
    return df_out

def detect_and_remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    df_out = df.copy()
    cmap = _col_map(df_out)
    if 'mp_count' in cmap:
        col = cmap['mp_count']
        try:
            series = pd.to_numeric(df_out[col], errors='coerce').dropna()
            if series.empty:
                return df_out
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            mask = (pd.to_numeric(df_out[col], errors='coerce') >= lower_bound) & (pd.to_numeric(df_out[col], errors='coerce') <= upper_bound)
            df_out = df_out.loc[mask.fillna(False)].reset_index(drop=True)
        except Exception:
            pass
    return df_out

def normalize_and_clean_text(df: pd.DataFrame) -> pd.DataFrame:
    df_out = df.copy()
    cmap = _col_map(df_out)
    text_col = None
    # look for common names for risk description
    for candidate in ['risk_description', 'risk_description_text', 'description', 'notes', 'risk_notes']:
        if candidate in cmap:
            text_col = cmap[candidate]
            break
    # define cleaner
    if text_col is not None:
        if NLTK_AVAILABLE:
            lemmatizer = WordNetLemmatizer()
            stop_words = set(stopwords.words('english'))
            def clean_text_nltk(text):
                if pd.isnull(text):
                    return ""
                text_proc = re.sub(r'[^a-zA-Z\s]', '', str(text).lower())
                words = [lemmatizer.lemmatize(w) for w in text_proc.split() if w not in stop_words]
                return " ".join(words)
            df_out['Cleaned_Text'] = df_out[text_col].apply(clean_text_nltk)
        else:
            # Lightweight fallback
            def clean_text_simple(text):
                if pd.isnull(text):
                    return ""
                text_proc = re.sub(r'[^a-zA-Z\s]', '', str(text).lower())
                words = [w for w in text_proc.split() if len(w) > 2]
                return " ".join(words)
            df_out['Cleaned_Text'] = df_out[text_col].apply(clean_text_simple)
    return df_out

# -------------------------
# Sidebar: structured into expanders for clarity
# -------------------------
with st.sidebar.expander("1) Dataset & Preview", expanded=True):
    st.write("Upload and inspect your dataset. Column names will be cleaned automatically.")
    uploaded_file = st.file_uploader("Upload CSV or Excel (e.g., Data1_Microplastic.csv)", type=["csv", "xls", "xlsx"] )
    if uploaded_file:
        st.caption(f"File: {uploaded_file.name}")

# stop early if missing file
if 'uploaded_file' not in locals() or uploaded_file is None:
    st.info("Please upload your dataset file to proceed.")
    st.stop()

# read file
df_raw = safe_read_file(uploaded_file)
if df_raw is None:
    st.stop()

st.sidebar.success(f"Loaded: {uploaded_file.name} — rows: {df_raw.shape[0]} cols: {df_raw.shape[1]}")

# Clean names and preview
df_raw = clean_column_names(df_raw)
with st.expander("Data preview (first 50 rows)", expanded=True):
    st.dataframe(df_raw.head(50))

# Provide the integrated cleaning script UI (Steps 2-7)
with st.sidebar.expander("DATA CLEANING: Steps 2–7 (Duplicates → Missing → ...)", expanded=False):
    st.write("Apply the standard microplastic cleaning pipeline to the uploaded dataset.")
    st.write("- Removes duplicates\n- Handles missing values\n- Standardizes units/text\n- Drops irrelevant metadata columns\n- Detects & removes outliers (MP count)\n- Normalizes and cleans text columns (requires NLTK)")
    run_cleaning = st.button("Run cleaning steps 2–7 on uploaded data")
    allow_overwrite = st.checkbox("Replace current dataframe with cleaned result after running", value=True)
    save_clean_csv = st.checkbox("Offer cleaned CSV for download", value=True)

if 'cleaned_on_click' not in st.session_state:
    st.session_state['cleaned_on_click'] = False

if run_cleaning or st.session_state['cleaned_on_click']:
    # make sure we only run once per click/refresh unless the user presses again
    st.session_state['cleaned_on_click'] = True
    st.info("Running cleaning steps 2–7 on the uploaded dataset (this will operate on the uploaded file copy).")
    df_clean = df_raw.copy()
    before_shape = df_clean.shape
    # STEP 2: Remove duplicates
    df_clean = remove_duplicates(df_clean)
    st.write(f"Removed duplicates: {before_shape} -> {df_clean.shape}")
    # STEP 3: Handle Missing Values
    df_clean = handle_missing_values(df_clean)
    st.write("Handled missing values (categorical -> 'Unknown', numeric -> mean where applicable).")
    # STEP 4: Standardize Units and Text
    df_clean = standardize_units_and_text(df_clean)
    st.write("Standardized units/text (e.g., risk level capitalization, MP_Count unit transformation if present).")
    # STEP 5: Remove Irrelevant Columns
    df_clean = remove_irrelevant_columns(df_clean)
    st.write("Removed irrelevant metadata columns if present (Author/URL/Citation/Notes).")
    # STEP 6: Detect and Remove Outliers
    before_outlier_shape = df_clean.shape
    df_clean = detect_and_remove_outliers(df_clean)
    st.write(f"Outlier removal (MP_Count IQR method): {before_outlier_shape} -> {df_clean.shape}")
    # STEP 7: Normalize and Clean Text Columns
    df_clean = normalize_and_clean_text(df_clean)
    st.write("Normalized and cleaned text columns; created 'Cleaned_Text' when a text column was found.")
    st.subheader("Cleaning summary")
    st.write(f"Initial shape: {df_raw.shape}, After cleaning: {df_clean.shape}")
    st.dataframe(df_clean.head(50))

    if save_clean_csv:
        try:
            csv_bytes = df_clean.to_csv(index=False).encode('utf-8')
            st.download_button("Download cleaned CSV", data=csv_bytes, file_name="microplastic_cleaned_data.csv", mime="text/csv")
        except Exception as e:
            st.warning(f"Could not prepare download: {e}")

    if allow_overwrite:
        # Overwrite df (the one used later in the app) with cleaned version
        df = df_clean.copy().reset_index(drop=True)
        st.success("Cleaned dataframe is now used for subsequent steps in the app.")
    else:
        # Keep original df variable (defined later) untouched; store cleaned separately
        st.session_state["df_cleaned_preview"] = df_clean.copy()
        df = df_raw.copy()  # default downstream variable remains the original pre-cleaned version
else:
    # If user hasn't run cleaning, proceed with the original df_raw
    df = df_raw.copy().reset_index(drop=True)

# -------------------------
# Preprocessing controls (grouped)
# -------------------------
with st.sidebar.expander("2) Preprocessing", expanded=False):
    st.write("Choose how to handle missing values and imputation.")
    drop_na = st.checkbox("Drop rows with any missing values", value=False)
    impute_strategy = st.selectbox("Numeric imputation strategy", ["mean", "median", "most_frequent"], index=0)

    def impute_numeric_columns(df_in: pd.DataFrame, strategy: str) -> pd.DataFrame:
        df_out = df_in.copy()
        numeric_cols = df_out.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            return df_out
        imputer = SimpleImputer(strategy=strategy)
        df_out[numeric_cols] = imputer.fit_transform(df_out[numeric_cols])
        return df_out

    def impute_categorical_columns(df_in: pd.DataFrame) -> pd.DataFrame:
        df_out = df_in.copy()
        cat_cols = df_out.select_dtypes(include=["object", "category"]).columns.tolist()
        for col in cat_cols:
            if df_out[col].isna().any():
                mode_series = df_out[col].mode()
                fill_value = mode_series.iloc[0] if not mode_series.empty else ""
                df_out[col] = df_out[col].fillna(fill_value)
        return df_out

# Apply preprocessing
if drop_na:
    before_rows = df.shape[0]
    df = df.dropna(axis=0, how="any").reset_index(drop=True)
    after_rows = df.shape[0]
    st.info(f"Dropped rows with missing values: {before_rows} -> {after_rows}")
else:
    df = df.copy().reset_index(drop=True)
    df = impute_numeric_columns(df, impute_strategy)
    df = impute_categorical_columns(df)
    remaining_na = df.isna().sum()
    total_remaining = int(remaining_na.sum())
    if total_remaining == 0:
        st.success("No missing values remain after imputation.")
    else:
        nonzero = remaining_na[remaining_na > 0]
        st.warning(f"There are {total_remaining} remaining missing values. Breakdown:\n{nonzero.to_dict()}")

st.write("✅ Preprocessing finished")
st.write(f"Dataset shape after cleaning: {df.shape}")

# -------------------------
# Modeling selection
# -------------------------
with st.sidebar.expander("3) Modeling & Features", expanded=True):
    task = st.selectbox("Task", ("classification", "regression", "clustering"))

    target_col = ""
    if task != "clustering":
        target_col = st.selectbox("Target column (for supervised tasks)", [""] + df.columns.tolist())
        if target_col == "" or target_col is None:
            st.warning("Please select a target column for supervised tasks.")

    all_cols = df.columns.tolist()
    default_features = [c for c in all_cols if c != target_col]
    selected_features = st.multiselect("Select feature columns (at least one)", all_cols, default=default_features)

# Validate feature/target
if task != "clustering" and (target_col == "" or target_col is None):
    st.stop()
if not selected_features:
    st.error("Select at least one feature column.")
    st.stop()

X, y, label_enc = prepare_features(df, selected_features, target_col if target_col else None, task, impute_strategy=impute_strategy)
st.write(f"Prepared features: X shape {X.shape}" + (f", y shape {y.shape}" if y is not None else ""))

if X.shape[1] == 0:
    st.error("No usable features after preprocessing. Please revise your feature selection.")
    st.stop()

# Train/test split options grouped
with st.sidebar.expander("4) Train/Test split & Randomness", expanded=False):
    test_size = st.slider("Test size", 0.05, 0.5, 0.2, 0.05)
    random_state = int(st.number_input("Random state", value=42, step=1))

if task in ("classification", "regression"):
    if y is None or len(y) == 0:
        st.error("No target values available after preprocessing — cannot train.")
        st.stop()
    X_train, X_test, y_train, y_test = safe_train_test_split(X, y, test_size, random_state, task)
    st.write(f"Train shape: {X_train.shape} — Test shape: {X_test.shape}")
else:
    X_train = X_test = y_train = y_test = None

# -------------------------
# Modeling & evaluation (with improved reporting)
# -------------------------
st.header("Model training & evaluation")

# Utility for pretty classification report and visualizations
def display_classification_results(model, X_test, y_test, label_enc=None, model_name: str = "Model"):
    try:
        y_pred = model.predict(X_test)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return

    # Confusion matrix (normalized and counts)
    cm = confusion_matrix(y_test, y_pred)
    cm_norm = confusion_matrix(y_test, y_pred, normalize='true')

    st.subheader(f"{model_name} — Performance")
    st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")

    # Classification report as dataframe
    cr = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    cr_df = pd.DataFrame(cr).transpose()
    st.dataframe(cr_df.style.format({c: "{:.3f}" for c in cr_df.select_dtypes(include=[float]).columns}))

    # Plot per-class metrics (precision, recall, f1)
    metrics_df = cr_df.loc[[c for c in cr_df.index if c not in ("accuracy", "macro avg", "weighted avg")], ["precision", "recall", "f1-score"]]
    if not metrics_df.empty:
        fig, ax = plt.subplots(figsize=(8, max(3, 0.4 * metrics_df.shape[0])))
        metrics_df.plot(kind='barh', ax=ax)
        ax.set_title(f"Per-class metrics — {model_name}")
        ax.legend(loc='lower right')
        st.pyplot(fig)

    # Confusion matrix heatmap
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2)
    ax2.set_title(f"Confusion Matrix (counts) — {model_name}")
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('Actual')
    st.pyplot(fig2)

    fig3, ax3 = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='viridis', ax=ax3)
    ax3.set_title(f"Confusion Matrix (normalized by true class) — {model_name}")
    ax3.set_xlabel('Predicted')
    ax3.set_ylabel('Actual')
    st.pyplot(fig3)

    # ROC / PR curves for binary classification
    if len(np.unique(y_test)) == 2:
        try:
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X_test)[:, 1]
            elif hasattr(model, "decision_function"):
                probs = model.decision_function(X_test)
            else:
                probs = None

            if probs is not None:
                fpr, tpr, _ = roc_curve(y_test, probs)
                auc_score = roc_auc_score(y_test, probs)
                fig4, ax4 = plt.subplots(figsize=(6, 5))
                ax4.plot(fpr, tpr, label=f"ROC AUC = {auc_score:.3f}")
                ax4.plot([0, 1], [0, 1], 'k--', alpha=0.5)
                ax4.set_xlabel('FPR')
                ax4.set_ylabel('TPR')
                ax4.set_title('ROC Curve')
                ax4.legend()
                st.pyplot(fig4)

                precision, recall, _ = precision_recall_curve(y_test, probs)
                pr_auc = np.trapz(recall, precision)
                fig5, ax5 = plt.subplots(figsize=(6, 5))
                ax5.plot(precision, recall, label=f"PR AUC = {pr_auc:.3f}")
                ax5.set_xlabel('Precision')
                ax5.set_ylabel('Recall')
                ax5.set_title('Precision-Recall Curve')
                ax5.legend()
                st.pyplot(fig5)
        except Exception as e:
            st.warning(f"Could not calculate ROC/PR curves: {e}")

    # Feature importance or coefficients
    try:
        if hasattr(model, 'feature_importances_'):
            fi = pd.Series(model.feature_importances_, index=X_test.columns).sort_values(ascending=False).head(20)
            fig6, ax6 = plt.subplots(figsize=(8, max(3, 0.25 * len(fi))))
            sns.barplot(x=fi.values, y=fi.index, ax=ax6)
            ax6.set_title('Top feature importances')
            st.pyplot(fig6)
        elif hasattr(model, 'coef_'):
            coef = model.coef_
            if coef.ndim == 1:
                coef_s = pd.Series(coef, index=X_test.columns).sort_values(key=abs, ascending=False).head(20)
                fig7, ax7 = plt.subplots(figsize=(8, max(3, 0.25 * len(coef_s))))
                sns.barplot(x=coef_s.values, y=coef_s.index, ax=ax7)
                ax7.set_title('Top coefficients (by magnitude)')
                st.pyplot(fig7)
    except Exception as e:
        st.info(f"Could not compute feature importances/coefficients: {e}")

if task == "classification":
    unique_classes = np.unique(y_train) if y_train is not None else []
    if y_train is None or len(unique_classes) < 2:
        st.error("Classification requires at least 2 classes with enough samples.")
    else:
        model_choices = st.multiselect("Select models to train (order matters for display):", ["Random Forest", "Decision Tree", "Logistic Regression"], default=["Random Forest", "Logistic Regression"])

        classifiers = {
            "Random Forest": RandomForestClassifier(random_state=random_state, n_jobs=-1),
            "Decision Tree": DecisionTreeClassifier(random_state=random_state),
            "Logistic Regression": LogisticRegression(max_iter=2000, solver="lbfgs")
        }

        trained_models = {}
        metrics = {}
        for name in model_choices:
            clf = classifiers.get(name)
            if clf is None:
                continue
            try:
                if has_nan_or_inf(X_train.values) or has_nan_or_inf(np.asarray(y_train)):
                    raise ValueError("Training data contains NaN/inf. Ensure imputation.")
                clf.fit(X_train, y_train)
                trained_models[name] = clf
                y_pred = clf.predict(X_test)
                metrics[name] = {
                    "Accuracy": float(accuracy_score(y_test, y_pred)),
                    "Precision": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
                    "Recall": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
                    "F1": float(f1_score(y_test, y_pred, average="macro", zero_division=0))
                }
            except Exception as e:
                metrics[name] = {"error": str(e)}

        st.subheader("Classification summary (test set)")
        st.table(pd.DataFrame(metrics).T)

        # Show detailed report for each trained model
        for name, model in trained_models.items():
            with st.expander(f"Detailed results — {name}", expanded=False):
                display_classification_results(model, X_test, y_test, label_enc=label_enc, model_name=name)

elif task == "regression":
    if y_train is None:
        st.error("Regression requires a numeric target.")
    else:
        if has_nan_or_inf(X_train.values):
            st.warning("Imputing NaNs in features before training.")
            imputer = SimpleImputer(strategy=impute_strategy)
            X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
            X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns, index=X_test.index)
        if np.isnan(y_train).any():
            st.warning("Dropping rows with NaN in target for training.")
            mask = ~np.isnan(y_train)
            X_train = X_train.loc[mask]
            y_train = y_train[mask]

        regressors = {
            "Random Forest Regressor": RandomForestRegressor(random_state=random_state),
            "Decision Tree Regressor": DecisionTreeRegressor(random_state=random_state),
            "Linear Regression": LinearRegression()
        }
        metrics = {}
        trained = {}
        for name, reg in regressors.items():
            try:
                if X_train.shape[0] < 2:
                    raise ValueError("Not enough training samples for regression.")
                reg.fit(X_train, y_train)
                trained[name] = reg
                y_pred = reg.predict(X_test)
                metrics[name] = {
                    "R2": float(r2_score(y_test, y_pred)),
                    "MAE": float(mean_absolute_error(y_test, y_pred)),
                    "MSE": float(mean_squared_error(y_test, y_pred))
                }
            except Exception as e:
                metrics[name] = {"error": str(e)}
        st.subheader("Regression results (test set)")
        st.table(pd.DataFrame(metrics).T)

        # Visualize residuals for best regressor by R2
        valid = {k: v for k, v in metrics.items() if "R2" in v}
        if valid:
            best = max(valid.items(), key=lambda t: t[1]["R2"])[0]
            st.write(f"Best model by R2: {best}")
            try:
                model = trained[best]
                y_pred = model.predict(X_test)
                fig, ax = plt.subplots(figsize=(7, 5))
                ax.scatter(y_test, y_pred, alpha=0.7)
                ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
                ax.set_xlabel('Actual')
                ax.set_ylabel('Predicted')
                ax.set_title('Actual vs Predicted')
                st.pyplot(fig)

                resid = y_test - y_pred
                fig2, ax2 = plt.subplots(figsize=(7, 4))
                sns.histplot(resid, kde=True, ax=ax2)
                ax2.set_title('Residual distribution')
                st.pyplot(fig2)
            except Exception as e:
                st.warning(f"Could not display regression plots: {e}")

elif task == "clustering":
    st.subheader("Clustering (unsupervised)")
    max_k = max(2, min(10, X.shape[0] - 1)) if X.shape[0] > 2 else 2
    n_clusters = st.sidebar.slider("Number of clusters (k)", min_value=2, max_value=max_k, value=min(3, max_k))
    try:
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(X)
        st.write("Cluster counts:", pd.Series(labels).value_counts().to_dict())
        if X.shape[1] >= 2 and X.shape[0] >= n_clusters:
            try:
                sil = silhouette_score(X, labels)
                st.write(f"Silhouette score: {sil:.4f}")
            except Exception as e:
                st.warning(f"Could not compute silhouette score: {e}")
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.scatter(X.iloc[:, 0], X.iloc[:, 1], c=labels, cmap="tab10", alpha=0.8)
            ax.set_xlabel(X.columns[0])
            ax.set_ylabel(X.columns[1])
            st.pyplot(fig)
        else:
            st.info("Not enough dimensions or samples for scatter plot.")
    except Exception as e:
        st.error(f"Clustering failed: {e}")

# -------------------------
# Cross-validation
# -------------------------
with st.sidebar.expander("5) Cross-validation", expanded=False):
    requested_splits = int(st.number_input("K-Fold splits", min_value=2, max_value=20, value=5))

st.header("Cross-validation")
cv_results = {}

if task in ("classification", "regression") and y is not None:
    if has_nan_or_inf(X.values) or np.isnan(y).any():
        st.warning("Imputing remaining NaNs in features and dropping NaNs in target before CV.")
        imputer = SimpleImputer(strategy=impute_strategy)
        X_cv = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)
        mask = ~np.isnan(y)
        X_cv = X_cv.loc[mask]
        y_cv = np.asarray(y)[mask]
    else:
        X_cv = X.copy()
        y_cv = np.asarray(y)

    n_samples = X_cv.shape[0]
    if n_samples < 2:
        cv_results = {"error": f"Not enough samples ({n_samples}) for cross-validation."}
    else:
        n_splits = min(requested_splits, n_samples)
        if n_splits < 2:
            n_splits = 2

        if task == "classification":
            unique, counts = np.unique(y_cv, return_counts=True)
            n_classes = len(unique)
            min_count = int(np.min(counts)) if len(counts) > 0 else 0

            if n_classes < 2:
                cv_results = {"error": "Cross-validation requires at least 2 classes."}
            else:
                if min_count >= n_splits:
                    cv_strategy = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
                else:
                    if min_count >= 2:
                        n_splits_reduced = min(n_splits, min_count)
                        cv_strategy = StratifiedKFold(n_splits=n_splits_reduced, shuffle=True, random_state=random_state)
                        st.warning(f"Reduced n_splits to {n_splits_reduced} for stratification.")
                    else:
                        n_splits_kfold = min(n_splits, max(2, n_samples // 2))
                        cv_strategy = KFold(n_splits=n_splits_kfold, shuffle=True, random_state=random_state)
                        st.warning("Using KFold (no stratification) due to very small class counts.")

                models_for_cv = {
                    "Random Forest": RandomForestClassifier(random_state=random_state),
                    "Decision Tree": DecisionTreeClassifier(random_state=random_state),
                    "Logistic Regression": LogisticRegression(max_iter=1000, solver="lbfgs")
                }
                for name, model in models_for_cv.items():
                    try:
                        scores = cross_val_score(model, X_cv, y_cv, cv=cv_strategy, scoring="accuracy")
                        cv_results[name] = {"mean_accuracy": float(scores.mean()), "std": float(scores.std())}
                    except Exception as e:
                        cv_results[name] = {"error": str(e)}

        else:  # regression
            n_splits_reg = min(n_splits, max(2, n_samples // 2))
            if n_splits_reg < 2:
                n_splits_reg = 2
            kfold = KFold(n_splits=n_splits_reg, shuffle=True, random_state=random_state)
            models_for_cv = {
                "RF Regressor": RandomForestRegressor(random_state=random_state),
                "DT Regressor": DecisionTreeRegressor(random_state=random_state),
                "LinearReg": LinearRegression()
            }
            for name, model in models_for_cv.items():
                try:
                    scores = cross_val_score(model, X_cv, y_cv, cv=kfold, scoring="r2")
                    cv_results[name] = {"mean_r2": float(scores.mean()), "std": float(scores.std())}
                except Exception as e:
                    cv_results[name] = {"error": str(e)}
else:
    cv_results = {"note": "Cross-validation not applicable for unsupervised task or missing target."}

st.write(cv_results)

# -------------------------
# Visualizations (improved readability and alignment to task)
# -------------------------
st.header("Visualizations")

# Dataset-level insights
with st.expander("Dataset insights", expanded=False):
    st.subheader("Column cardinality summary")
    st.dataframe(summarize_cardinality(df))

# 1) Distribution of a selected categorical column with aggregation
st.subheader("Categorical distribution (top categories aggregated)")
cat_columns = [c for c in df.columns if df[c].dtype == "object" or df[c].dtype.name.startswith("category")]
cat_col = st.selectbox("Select a categorical column to display", options=cat_columns + [""] )
if cat_col:
    top_n = st.slider("Top N categories to show (others -> Other)", min_value=5, max_value=50, value=15)
    series = df[cat_col].fillna("NaN").astype(str)
    agg = aggregate_small_categories(series, top_n=top_n)
    fig, ax = plt.subplots(figsize=(8, min(6, 0.25 * len(agg))))
    agg.sort_values().plot(kind="barh", ax=ax, color=sns.color_palette("tab10", n_colors=len(agg)))
    ax.set_xlabel("Count")
    ax.set_ylabel(cat_col)
    ax.set_title(f"{cat_col} distribution (top {top_n})")
    for i, v in enumerate(agg.sort_values()):
        ax.text(v + max(agg.max()*0.01, 1e-6), i, str(int(v)), va="center")
    st.pyplot(fig)

# 2) Risk_Level if present
if "Risk_Level" in df.columns:
    st.subheader("Risk_Level distribution (improved)")
    rl = df["Risk_Level"].fillna("NaN").astype(str)
    agg_rl = aggregate_small_categories(rl, top_n=20)
    fig, ax = plt.subplots(figsize=(8, max(3, 0.25 * len(agg_rl))))
    agg_rl.sort_values().plot(kind="barh", ax=ax, color="salmon")
    ax.set_xlabel("Count")
    ax.set_ylabel("Risk_Level")
    ax.set_title("Risk_Level distribution")
    st.pyplot(fig)

# 3) Pairplot (numeric features only) capped to avoid clutter
st.subheader("Pairplot (numeric features subset)")
numeric_cols = [c for c in X.columns]
if numeric_cols:
    max_features = st.number_input("Max numeric features to include in pairplot", min_value=2, max_value=8, value=min(6, len(numeric_cols)))
    sel = numeric_cols[:int(max_features)]
    try:
        sample_n = min(len(X), 300)
        df_pair = pd.DataFrame(X[sel]).sample(n=sample_n, random_state=42)
        sns_plot = sns.pairplot(df_pair, corner=True, plot_kws={"s": 20, "alpha": 0.6})
        st.pyplot(sns_plot.fig)
    except Exception as e:
        st.warning(f"Could not create pairplot: {e}")
else:
    st.info("No numeric features available for pairplot.")

st.success("Finished. Visualizations are aggregated and rotated to improve readability for long/crowded labels.")
