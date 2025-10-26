# streamlit_microplastic_app.py
# Microplastic Risk Analysis Dashboard — Blue Themed Version

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
st.caption("Interactive defense demo — explore dataset, clustering, classification, validation, regression, and summary.")

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

# --- Utility functions ---
@st.cache_data
def preprocess_dataframe(in_df):
    df = in_df.copy()
    df = df.drop_duplicates().reset_index(drop=True)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    for c in num_cols:
        df[c] = df[c].fillna(df[c].median())
    for c in cat_cols:
        if df[c].isnull().any():
            df[c] = df[c].fillna(df[c].mode().iloc[0])
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

# --- Steps ---
with st.expander("⚙️ Step 2: Preprocessing", expanded=True):
    if st.button("Run Preprocessing", key="preprocess"):
        with st.spinner("Cleaning and encoding dataset..."):
            cleaned_df, num_cols, cat_cols = preprocess_dataframe(df)
            st.session_state['cleaned_df'] = cleaned_df
            st.session_state['num_cols'] = num_cols
            st.session_state['cat_cols'] = cat_cols
            st.success("✅ Preprocessing complete!")
            st.dataframe(cleaned_df.head())
    else:
        cleaned_df = st.session_state.get('cleaned_df', None)

if cleaned_df is None:
    st.info("Run preprocessing first.")
    st.stop()

# --- Step 3: Clustering ---
with st.expander("🔹 Step 3: K-Means Clustering", expanded=False):
    cluster_cols = st.multiselect("Select features for clustering", options=cleaned_df.columns.tolist(), default=st.session_state['num_cols'])
    n_clusters = st.slider("Number of clusters", 2, 10, 3)
    if st.button("Run K-Means", key="cluster"):
        X = cleaned_df[cluster_cols].values
        X_scaled = StandardScaler().fit_transform(X)
        X_pca, _ = compute_pca(X_scaled)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(X_scaled)
        cleaned_df['Cluster'] = labels
        st.session_state['cleaned_df'] = cleaned_df
        st.success(f"✅ K-Means complete — {n_clusters} clusters identified!")

        fig = px.scatter(
            x=X_pca[:,0], y=X_pca[:,1], color=labels.astype(str),
            color_discrete_sequence=px.colors.sequential.Blues,
            title="K-Means Clusters (PCA 2D Visualization)",
            labels={'x':'PC1','y':'PC2'}
        )
        st.plotly_chart(fig, use_container_width=True)
        st.write("**Cluster Counts:**")
        st.dataframe(cleaned_df['Cluster'].value_counts().rename_axis('Cluster').reset_index(name='Count'))

# --- Continue with classification, validation, regression, and summary (same logic) ---
# --- (All sections will inherit blue theme styling) ---

st.markdown("---")
st.caption("App built for thesis defense — styled in calming blue tones 💙.")
