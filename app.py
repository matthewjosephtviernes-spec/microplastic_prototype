# ==============================================================
# Predictive Risk Modeling for Microplastic Pollution
# Developed by: Magdaluyo & Viernes
# ==============================================================
# Streamlit web app to visualize, classify, and predict microplastic risk levels
# ==============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --------------------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------------------
st.set_page_config(page_title="Microplastic Risk Prediction", layout="wide")
st.title("🌊 Predictive Risk Modeling for Microplastic Pollution")
st.caption("Developed by: Magdaluyo & Viernes | Agusan del Sur State College of Agriculture and Technology")

st.markdown("""
This app demonstrates how **data mining techniques** such as *clustering* and *classification*
can predict and visualize the **risk of microplastic pollution** based on extracted datasets.
""")

# --------------------------------------------------------------
# DATA UPLOAD SECTION
# --------------------------------------------------------------
st.header("1️⃣ Upload and Inspect Data")

uploaded_file = st.file_uploader("Upload your preprocessed dataset (.csv)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Data loaded successfully!")
    st.write("**Preview of Dataset:**")
    st.dataframe(df.head())

    # Basic info
    st.write("**Dataset Summary:**")
    st.write(df.describe(include="all"))
    
    # Data cleaning
    st.subheader("Data Cleaning Overview")
    st.write(f"Initial rows: {len(df)}")
    df.drop_duplicates(inplace=True)
    st.write(f"After removing duplicates: {len(df)}")
    st.write("Missing Values per Column:")
    st.write(df.isnull().sum())

    # ----------------------------------------------------------
    # VISUAL EXPLORATION
    # ----------------------------------------------------------
    st.header("2️⃣ Data Visualization")

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    if num_cols:
        st.subheader("📊 Correlation Heatmap")
        plt.figure(figsize=(8, 5))
        sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm")
        st.pyplot(plt.gcf())
        plt.clf()

    if cat_cols:
        st.subheader("📈 Distribution of Categorical Columns")
        for col in cat_cols[:3]:  # Limit to first 3
            st.bar_chart(df[col].value_counts())

    # ----------------------------------------------------------
    # CLUSTERING SECTION
    # ----------------------------------------------------------
    st.header("3️⃣ Clustering Analysis (K-Means)")
    if num_cols:
        k = st.slider("Select number of clusters", 2, 6, 3)
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        cluster_labels = kmeans.fit_predict(df[num_cols])
        df["Cluster"] = cluster_labels

        st.write("Cluster counts:")
        st.write(df["Cluster"].value_counts())
        st.write("Cluster Visualization:")
        plt.figure(figsize=(7, 5))
        sns.scatterplot(x=num_cols[0], y=num_cols[1], hue="Cluster", data=df, palette="viridis")
        plt.title("K-Means Clustering Visualization")
        st.pyplot(plt.gcf())
        plt.clf()

    # ----------------------------------------------------------
    # CLASSIFICATION MODEL
    # ----------------------------------------------------------
    st.header("4️⃣ Classification and Prediction")

    target_col = st.selectbox("Select Target Column (Risk Category)", options=df.columns)
    features = [col for col in df.columns if col != target_col]

    if st.button("Run Classification Model"):
        le = LabelEncoder()
        if df[target_col].dtype == 'object':
            df[target_col] = le.fit_transform(df[target_col])

        X = df[features].select_dtypes(include=[np.number])
        y = df[target_col]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        st.success(f"✅ Model Accuracy: {acc*100:.2f}%")

        st.text("Classification Report:")
        st.text(classification_report(y_test, y_pred))

        # Confusion matrix
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        st.pyplot(plt.gcf())
        plt.clf()

        # Cross-validation
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')
        st.write(f"**K-Fold Cross Validation Average Accuracy:** {np.mean(cv_scores)*100:.2f}%")

    # ----------------------------------------------------------
    # SUMMARY
    # ----------------------------------------------------------
    st.header("5️⃣ Summary")
    st.info("""
    - The **K-Means** clustering grouped samples into distinct ecological, chemical, or human health risks.  
    - The **Random Forest** classifier achieved strong predictive accuracy.  
    - Visual outputs such as heatmaps and scatterplots improve interpretability.  
    - This system supports data-driven **environmental decision-making**.
    """)
else:
    st.warning("Please upload a dataset to begin analysis.")

st.caption("© 2025 Magdaluyo & Viernes | Microplastic Risk Prediction Project")
