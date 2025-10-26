# ==========================================================
# 🌊 Predictive Risk Modeling for Microplastic Pollution
# ==========================================================

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    mean_squared_error, r2_score
)

# -------------------------------
# Streamlit App Setup
# -------------------------------
st.set_page_config(page_title="Microplastic Risk Prediction", layout="wide")
st.title("🌊 Predictive Risk Modeling for Microplastic Pollution")
st.caption("Upload your dataset to preprocess, model, and validate predictive insights on microplastic pollution risks.")

# -------------------------------
# Upload Dataset
# -------------------------------
uploaded_file = st.file_uploader("📂 Upload your CSV dataset", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📋 Raw Dataset Preview")
    st.dataframe(df.head())

    # ==========================================================
    # STEP 1: Data Cleaning
    # ==========================================================
    df = df.drop_duplicates()
    df = df.fillna('')

    # Convert numeric column
    if 'MP_Count (items/individual)' in df.columns:
        df['MP_Count (items/individual)'] = pd.to_numeric(df['MP_Count (items/individual)'], errors='coerce')

    # Encode binary columns
    if 'MP_Presence' in df.columns:
        df['MP_Presence'] = df['MP_Presence'].map({'Yes': 1, 'No': 0})

    # Multi-label feature expansion
    multi_cols = ['MP_Type', 'MP_Structure', 'MP_Color', 'Polymer_Type', 'Dominant_Risk_Type']
    for col in multi_cols:
        if col in df.columns:
            df[col] = df[col].fillna('')
            unique_tokens = set()
            for val in df[col]:
                unique_tokens.update([v.strip() for v in str(val).split(',') if v.strip() != ''])
            for token in unique_tokens:
                df[token] = df[col].apply(lambda x: 1 if token in str(x) else 0)

    # One-hot encode simple categorical columns
    cat_cols = ['Study_Location', 'Site / Sampling_Area', 'Habitat_Type', 'Species_Name']
    cat_cols = [c for c in cat_cols if c in df.columns]
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # Drop non-informative columns
    for col in ['Source_Suspected', 'Author']:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Handle missing numeric values
    df = df.fillna(0)

    st.subheader("✅ Preprocessed Numeric Dataset")
    st.dataframe(df.head())
    st.write("Shape:", df.shape)

    # ==========================================================
    # STEP 2: Feature Selection
    # ==========================================================
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Set target variable (use MP_Count if available)
    if 'MP_Count (items/individual)' in numeric_cols:
        target = 'MP_Count (items/individual)'
    else:
        target = numeric_cols[-1]

    X = df.drop(columns=[target], errors='ignore')
    y = df[target]

    # Keep only numeric columns for scaling
    X_numeric = X.select_dtypes(include=[np.number])
    non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric_cols:
        st.warning(f"⚠️ The following non-numeric columns were excluded from modeling: {non_numeric_cols}")

    # Scale numeric features safely
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_numeric)
    X_scaled = pd.DataFrame(X_scaled, columns=X_numeric.columns)

    # ==========================================================
    # STEP 3: Classification (Predicting MP Presence)
    # ==========================================================
    st.subheader("🧠 Classification: Predicting Microplastic Presence")
    if 'MP_Presence' in df.columns:
        X_class = df.select_dtypes(include=[np.number]).drop(columns=['MP_Presence', target], errors='ignore')
        y_class = df['MP_Presence']

        X_train, X_test, y_train, y_test = train_test_split(X_class, y_class, test_size=0.2, random_state=42)

        clf = RandomForestClassifier(random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        st.write("**Accuracy:**", round(accuracy_score(y_test, y_pred), 3))
        st.text("Classification Report:")
        st.text(classification_report(y_test, y_pred))

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

        # Cross-validation
        cv_score = cross_val_score(clf, X_class, y_class, cv=5).mean()
        st.write("**Cross-Validation Accuracy:**", round(cv_score, 3))

    # ==========================================================
    # STEP 4: Clustering (Group Similar Sites)
    # ==========================================================
    st.subheader("📊 Clustering: Grouping Similar Sampling Areas")
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    df['Cluster'] = clusters

    st.write("Cluster Distribution:")
    st.bar_chart(df['Cluster'].value_counts())

    # 2D Visualization
    fig, ax = plt.subplots()
    ax.scatter(X_scaled.iloc[:, 0], X_scaled.iloc[:, 1], c=clusters, cmap='viridis')
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_title("K-Means Clustering Visualization")
    st.pyplot(fig)

    # ==========================================================
    # STEP 5: Regression (Predicting MP Count)
    # ==========================================================
    st.subheader("📈 Regression: Predicting Microplastic Count")
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.write("**Mean Squared Error:**", round(mean_squared_error(y_test, y_pred), 3))
    st.write("**R² Score:**", round(r2_score(y_test, y_pred), 3))

    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred, alpha=0.7)
    ax.set_xlabel("Actual MP Count")
    ax.set_ylabel("Predicted MP Count")
    ax.set_title("Actual vs Predicted Microplastic Count")
    st.pyplot(fig)

    # ==========================================================
    # STEP 6: Summary of Results
    # ==========================================================
    st.subheader("🧾 Summary of Results")
    st.markdown("""
    - **Preprocessing:** Categorical and text data converted to numeric predictors.
    - **Classification:** Predicted presence of microplastics with measurable accuracy.
    - **Clustering:** Identified patterns among sampling areas.
    - **Regression:** Modeled microplastic count using numeric predictors.
    - **Validation:** Cross-validation and metrics provide interpretability.

    ✅ *Results ready for Chapter 5 discussion and conclusion.*
    """)

else:
    st.info("Please upload your dataset to begin analysis.")
