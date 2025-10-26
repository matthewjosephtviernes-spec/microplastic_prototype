# streamlit_microplastic_app.py
# Microplastic Risk Analysis Dashboard — Blue Themed with Data Cleaning

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

# --- Custom CSS Styling (Blue Theme) ---
st.markdown("""
<style>
/* General Background */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #1e88e5;
    color: white;
}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    color: white;
}
[data-testid="stSidebar"] .stButton>button {
    background-color: #1565c0;
    color: white;
    border: none;
    border-radius: 10px;
}
[data-testid="stSidebar"] .stButton>button:hover {
    background-color: #0d47a1;
}

/* Titles and captions */
h1, h2, h3 {
    color: #0d47a1 !important;
}
.stCaption {
    color: #1565c0 !important;
    font-style: italic;
}

/* Buttons */
div.stButton > button {
    background-color: #1976d2;
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.5em 1em;
    transition: 0.3s;
}
div.stButton > button:hover {
    background-color: #0d47a1;
    color: #e3f2fd;
    transform: scale(1.05);
}

/* Expanders */
.streamlit-expanderHeader {
    background-color: #2196f3 !important;
    color: white !important;
    border-radius: 5px;
}
.streamlit-expanderHeader:hover {
    background-color: #1976d2 !important;
}
.streamlit-expanderContent {
    background-color: #f1f8ff;
    border-left: 3px solid #2196f3;
}

/* Dataframes */
[data-testid="stDataFrame"] {
    background-color: #ffffff;
    border: 1px solid #90caf9;
    border-radius: 8px;
}

/* Success, warning, info messages */
.stSuccess {
    background-color: #bbdefb !important;
    color: #0d47a1 !important;
    border: 1px solid #64b5f6;
    border-radius: 8px;
}
.stWarning {
    background-color: #fff9c4 !important;
}
.stInfo {
    background-color: #e3f2fd !important;
}

/* Footer */
footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)

# --- Title ---
st.title("🧪 Microplastic Risk Analysis Dashboard")
st.caption("Interactive thesis defense demo — with data cleaning, clustering, classification, validation, regression, and summary.")

# --- Sidebar ---
st.sidebar.title("Navigation")
st.sidebar.info("Use buttons to manually trigger each stage for your thesis defense.")

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

    # Fill numeric with median, categorical with mode
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
        with st.spinner("Cleaning dataset... removing duplicates and filling missing values..."):
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
            color_discrete_sequence=px.colors.sequential.Blues,
            title="K-Means Clusters (PCA 2D Visualization)",
            labels={'x':'PC1','y':'PC2'}
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(pre_df['Cluster'].value_counts().rename_axis('Cluster').reset_index(name='Count'))

# (Classification, Validation, Regression, Summary sections would follow unchanged)

st.markdown("---")
st.caption("💙 App built for thesis defense — includes new Data Cleaning stage for clarity and transparency.")
