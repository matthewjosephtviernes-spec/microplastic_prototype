# streamlit_microplastic_app.py
# Microplastic Risk Analysis Dashboard — Clean & Minimal Theme with Data Cleaning

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, r2_score
import plotly.express as px

# --- Page Config ---
st.set_page_config(layout="wide", page_title="Microplastic Risk Analysis Dashboard", page_icon="🧪")

# --- Minimal, Clear CSS Styling ---
st.markdown("""
<style>
/* App Background */
[data-testid="stAppViewContainer"] {
    background-color: #f9fafb;
    color: #2c2c2c;
    font-family: "Inter", sans-serif;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #f0f2f6;
    color: #333;
}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    color: #111;
}

/* Buttons */
div.stButton > button {
    background-color: #4b8bf5;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.5em 1em;
    transition: 0.3s;
    font-weight: 500;
}
div.stButton > button:hover {
    background-color: #316cd8;
    transform: scale(1.03);
}

/* Expanders */
.streamlit-expanderHeader {
    background-color: #ffffff !important;
    border: 1px solid #e0e0e0;
    border-radius: 6px;
    color: #111 !important;
    font-weight: 600;
}
.streamlit-expanderContent {
    background-color: #fcfcfc;
    border-left: 3px solid #d0d0d0;
    border-radius: 6px;
    padding: 0.5rem 0.5rem;
}

/* Dataframes */
[data-testid="stDataFrame"] {
    background-color: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 6px;
}

/* Messages */
.stSuccess {
    background-color: #e6f4ea !important;
    color: #1b5e20 !important;
    border: 1px solid #c8e6c9;
    border-radius: 6px;
}
.stWarning {
    background-color: #fff3cd !important;
    border: 1px solid #ffeeba;
}
.stInfo {
    background-color: #e3f2fd !important;
    border: 1px solid #bbdefb;
}

/* Titles */
h1, h2, h3 {
    color: #1c1c1c !important;
}

/* Captions */
.stCaption {
    color: #555 !important;
    font-style: italic;
}

/* Footer */
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- Title ---
st.title("🧪 Microplastic Risk Analysis Dashboard")
st.caption("Interactive thesis defense demo — includes data cleaning, clustering, classification, validation, regression, and summary.")

# --- Sidebar ---
st.sidebar.title("Navigation")
st.sidebar.info("Use the buttons below to move through each step of the analysis process.")

st.sidebar.header("1️⃣ Upload or Load Dataset")
uploaded_file = st.sidebar.file_uploader("Upload your .csv dataset", type=["csv"])
use_default = st.sidebar.checkbox("Use default dataset (uploaded earlier)", value=True)

# --- Load Data ---
def load_default(paths):
    for p in paths:
        try:
            return pd.read_csv(p)
        except Exception:
            continue
    return None

default_paths = ["/mnt/data/Cleaned_Microplastic_Dataset.csv", "/mnt/data/Encoded_Microplastic_Dataset.csv"]

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    df = load_default(default_paths) if use_default else None

if df is None:
    st.warning("⚠️ Please upload a dataset or enable 'Use default dataset'.")
    st.stop()

st.success(f"✅ Dataset loaded successfully — {df.shape[0]} rows × {df.shape[1]} columns")

# --- Utility Functions ---
@st.cache_data
def clean_data(df):
    before_rows = df.shape[0]
    df = df.drop_duplicates()
    duplicates_removed = before_rows - df.shape[0]

    missing_summary = df.isnull().sum()
    missing_cols = missing_summary[missing_summary > 0]

    for col in df.select_dtypes(include=[np.number]):
        df[col] = df[col].fillna(df[col].median())
    for col in df.select_dtypes(exclude=[np.number]):
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode().iloc[0])

    return df, duplicates_removed, missing_cols

@st.cache_data
def preprocess_dataframe(in_df):
    df = in_df.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    for c in cat_cols:
        try:
            le = LabelEncoder()
            df[c] = le.fit_transform(df[c].astype(str))
        except Exception:
            pass
    return df, num_cols, cat_cols

@st.cache_data
def compute_pca(X, n_components=2):
    pca = PCA(n_components=n_components)
    Xp = pca.fit_transform(X)
    return Xp, pca

# --- 2️⃣ Data Cleaning Section ---
with st.expander("🧹 Step 2: Data Cleaning", expanded=True):
    if st.button("Run Data Cleaning", key="cleaning"):
        with st.spinner("Cleaning dataset... removing duplicates and handling missing values..."):
            cleaned_df, duplicates_removed, missing_cols = clean_data(df)
            st.session_state['cleaned_df'] = cleaned_df

            st.success(f"✅ Cleaning complete! Removed {duplicates_removed} duplicates.")
            if not missing_cols.empty:
                st.info("Columns with missing values before cleaning:")
                st.dataframe(missing_cols.rename("Missing Count"))
            else:
                st.info("No missing values were found.")
            st.dataframe(cleaned_df.head())
    else:
        cleaned_df = st.session_state.get('cleaned_df', None)

if cleaned_df is None:
    st.info("Please run data cleaning first.")
    st.stop()

# --- 3️⃣ Preprocessing Section ---
with st.expander("⚙️ Step 3: Preprocessing (Encoding)", expanded=False):
    if st.button("Run Preprocessing", key="preprocess"):
        with st.spinner("Encoding categorical variables..."):
            pre_df, num_cols, cat_cols = preprocess_dataframe(cleaned_df)
            st.session_state['pre_df'] = pre_df
            st.session_state['num_cols'] = num_cols
            st.session_state['cat_cols'] = cat_cols
            st.success("✅ Preprocessing complete!")
            st.dataframe(pre_df.head())
    else:
        pre_df = st.session_state.get('pre_df', None)

if pre_df is None:
    st.info("Run preprocessing first.")
    st.stop()

# --- 4️⃣ Clustering ---
with st.expander("🔹 Step 4: K-Means Clustering", expanded=False):
    cluster_cols = st.multiselect("Select features for clustering", options=pre_df.columns.tolist(), default=st.session_state['num_cols'])
    n_clusters = st.slider("Number of clusters", 2, 10, 3)
    if st.button("Run K-Means", key="cluster"):
        X = pre_df[cluster_cols].values
        X_scaled = StandardScaler().fit_transform(X)
        X_pca, _ = compute_pca(X_scaled)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(X_scaled)
        pre_df['Cluster'] = labels
        st.session_state['pre_df'] = pre_df
        st.success(f"✅ K-Means complete — {n_clusters} clusters identified!")

        fig = px.scatter(
            x=X_pca[:,0], y=X_pca[:,1], color=labels.astype(str),
            color_discrete_sequence=px.colors.qualitative.Pastel,
            title="K-Means Clusters (PCA 2D Visualization)",
            labels={'x':'PC1','y':'PC2'}
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(pre_df['Cluster'].value_counts().rename_axis('Cluster').reset_index(name='Count'))

st.markdown("---")
st.caption("🩶 Clean & clear version for thesis defense — focuses on data visibility and simplicity.")
