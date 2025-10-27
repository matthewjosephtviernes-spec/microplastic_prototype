import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report,
                             mean_squared_error, r2_score, silhouette_score)
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")
# ---------------------------
# CONFIGURATION
# ---------------------------
st.set_page_config(page_title="🔍 Predictive Risk Modeling", layout="wide")
st.sidebar.title("🔍 Predictive Risk Modeling Dashboard")
page = st.sidebar.selectbox("Navigate", ["📂 Upload Data", "⚙️ Preprocessing", "📊 Classification", "🔢 Regression", "🗂️ Clustering", "✅ Cross-Validation"])
# ---------------------------
# 1. UPLOAD DATA
# ---------------------------
if page == "📂 Upload Data":
    st.header("📂 Upload Dataset")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ **{uploaded_file.name}** loaded!")
        with st.expander("Data Preview"):
            st.write(df.head())
        with st.expander("Data Stats"):
            st.write(f"**Rows:** {df.shape[0]} | **Columns:** {df.shape[1]}")
            st.write(df.describe())
        st.download_button("⬇️ Download Sample", df.head().to_csv(index=False), "sample.csv")
    else:
        st.info("👈 Upload a CSV to start.")
# ---------------------------
# 2. PREPROCESSING
# ---------------------------
elif page == "⚙️ Preprocessing":
    st.header("⚙️ Data Preprocessing")
    if 'df' not in locals():
        st.error("❌ Upload data first!")
    else:
        tab1, tab2, tab3 = st.tabs(["Missing Values", "Encode Categorical", "Cleaned Data"])
        with tab1:
            st.subheader("Handle Missing Values")
            if df.isnull().sum().any():
                cols = df.columns[df.isnull().any()]
                for col in cols:
                    option = st.selectbox(f"Fill **{col}**", ["Mean", "Median", "Drop"], key=col)
                    if st.button("Apply", key=f"btn_{col}"):
                        if option == "Mean":
                            df[col] = df[col].fillna(df[col].mean())
                        elif option == "Median":
                            df[col] = df[col].fillna(df[col].median())
                        else:
                            df = df.dropna(subset=[col])
                        st.success(f"✅ {col} updated!")
            else:
                st.write("✅ No missing values.")
        with tab2:
            st.subheader("Encode Categorical Columns")
            cats = df.select_dtypes(include="object").columns
            if len(cats):
                for col in cats:
                    df[col] = LabelEncoder().fit_transform(df[col])
                st.write("Encoded:", list(cats))
            else:
                st.write("✅ No categorical data.")
        with tab3:
            st.subheader("Cleaned Data")
            st.write(df.head())
            st.download_button("⬇️ Download Cleaned CSV", df.to_csv(index=False), "cleaned_data.csv")
# ---------------------------
# 3. CLASSIFICATION
# ---------------------------
elif page == "📊 Classification":
    st.header("📊 Classification (Random Forest)")
    if 'df' not in locals():
        st.error("❌ Preprocess data first!")
    else:
        target = st.selectbox("Select Target", df.columns)
        X = df.drop(columns=[target])
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📈 Metrics")
            st.text(classification_report(y_test, y_pred))
            st.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.3f}")
        with col2:
            st.subheader("🔹 Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            st.pyplot(fig)
        with st.expander("Feature Importance"):
            imp = pd.DataFrame({"Feature": X.columns, "Importance": model.feature_importances_})
            fig = px.bar(imp, x="Feature", y="Importance", title="Top Features")
            st.plotly_chart(fig)
# ---------------------------
# 4. REGRESSION
# ---------------------------
elif page == "🔢 Regression":
    st.header("🔢 Regression (Linear Model)")
    if 'df' not in locals():
        st.error("❌ Preprocess data first!")
    else:
        target = st.selectbox("Select Target", df.columns)
        X = df.drop(columns=[target])
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📊 Metrics")
            st.metric("RMSE", f"{np.sqrt(mean_squared_error(y_test, y_pred)):.3f}")
            st.metric("R²", f"{r2_score(y_test, y_pred):.3f}")
        with col2:
            st.subheader("🔹 Predicted vs Actual")
            fig = px.scatter(x=y_test, y=y_pred, trendline="ols", title="Fit")
            st.plotly_chart(fig)
# ---------------------------
# 5. CLUSTERING
# ---------------------------
elif page == "🗂️ Clustering":
    st.header("🗂️ KMeans Clustering")
    if 'df' not in locals():
        st.error("❌ Preprocess data first!")
    else:
        k = st.slider("Select K", 2, 10, 3)
        model = KMeans(n_clusters=k, random_state=42)
        labels = model.fit_predict(df)
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📈 Silhouette Score")
            st.metric("Score", f"{silhouette_score(df, labels):.3f}")
        with col2:
            st.subheader("🔹 Clusters")
            fig, ax = plt.subplots()
            sns.scatterplot(x=df.iloc[:, 0], y=df.iloc[:, 1], hue=labels, palette="viridis", ax=ax)
            st.pyplot(fig)
# ---------------------------
# 6. CROSS-VALIDATION
# ---------------------------
elif page == "✅ Cross-Validation":
    st.header("✅ K-Fold Cross-Validation")
    if 'df' not in locals():
        st.error("❌ Preprocess data first!")
    else:
        target = st.selectbox("Select Target", df.columns)
        X = df.drop(columns=[target])
        y = df[target]
        model = RandomForestClassifier()
        scores = cross_val_score(model, X, y, cv=5)
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📊 Results")
            st.write(f"**Mean:** {scores.mean():.3f} | **Std:** {scores.std():.3f}")
        with col2:
            st.subheader("🔹 Fold Distribution")
            fig, ax = plt.subplots()
            ax.boxplot(scores)
            st.pyplot(fig)
# ---------------------------
# FOOTER
# ---------------------------
st.sidebar.markdown("---")
st.sidebar.text("Built with ❤️ using Streamlit")
