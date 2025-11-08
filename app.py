# app.py – Final Version for "Predictive Risk Modeling for Microplastic Pollution"
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score

# ---------------------------------
# Streamlit Page Configuration
# ---------------------------------
st.set_page_config(page_title="Microplastic Risk Modeling", layout="wide")
st.title("🌊 Predictive Risk Modeling for Microplastic Pollution")
st.markdown("#### Upload your dataset to begin the analysis")

# ---------------------------------
# File Upload Section
# ---------------------------------
uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])

if uploaded_file:
    # Try reading the file safely
    try:
        df = pd.read_csv(uploaded_file, encoding='latin1')
    except UnicodeDecodeError:
        df = pd.read_csv(uploaded_file, encoding='utf-8', errors='ignore')

    # ---------------------------------
    # Data Cleaning & Preparation
    # ---------------------------------
    df.columns = df.columns.str.strip().str.replace(" ", "_")
    df = df.dropna(how="all")

    # Drop unnecessary column
    if "Author" in df.columns:
        df = df.drop(columns=["Author"])

    # Numeric extraction for columns with ranges (e.g., “0.5–1.0”, “8 33 ppt”)
    def extract_numeric(val):
        if isinstance(val, str):
            val = val.replace("ppt", "").replace("–", "-").replace("to", "-")
            parts = val.split("-")
            try:
                nums = [float(p.strip()) for p in parts if p.strip() != ""]
                return np.mean(nums)
            except:
                return np.nan
        return val

    for col in ["Salinity", "Density", "Microplastic_Size"]:
        if col in df.columns:
            df[col] = df[col].apply(extract_numeric)

    if "Risk_Score" in df.columns:
        df["Risk_Score"] = pd.to_numeric(df["Risk_Score"], errors="coerce")

    # ---------------------------------
    # Display Dataset Overview
    # ---------------------------------
    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)
    st.write(f"**Shape:** {df.shape}")
    st.write(f"**Columns:** {list(df.columns)}")

    # ---------------------------------
    # Encode Categorical Columns
    # ---------------------------------
    st.subheader("🧩 Data Preprocessing")
    cat_cols = df.select_dtypes(include=["object"]).columns
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = df[col].astype(str)
        df[col] = le.fit_transform(df[col])

    st.success("Categorical columns encoded successfully!")

    # ---------------------------------
    # Clustering Section
    # ---------------------------------
    st.subheader("🔶 K-Means Clustering")
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    selected_features = st.multiselect("Select features for clustering", num_cols, default=["Latitude", "Longitude", "MP_Count_per_L", "Risk_Score"])

    if len(selected_features) >= 2:
        k = st.slider("Select number of clusters (k)", 2, 10, 3)
        kmeans = KMeans(n_clusters=k, random_state=42)
        df["Cluster"] = kmeans.fit_predict(df[selected_features])

        fig = px.scatter_3d(df, 
                            x=selected_features[0], 
                            y=selected_features[1],
                            z=selected_features[2] if len(selected_features) > 2 else selected_features[1],
                            color="Cluster", 
                            title="3D Cluster Visualization",
                            height=600)
        st.plotly_chart(fig, use_container_width=True)

    # ---------------------------------
    # Classification Section
    # ---------------------------------
    st.subheader("🧠 Risk Level Classification")

    if "Risk_Level" in df.columns:
        target_col = "Risk_Level"
        features = [col for col in df.columns if col not in [target_col, "Risk_Score", "Cluster"]]
        X = df[features]
        y = df[target_col]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        model_choice = st.selectbox("Select Classification Model", ["Random Forest", "Logistic Regression", "Decision Tree"])

        if model_choice == "Random Forest":
            model = RandomForestClassifier(random_state=42)
        elif model_choice == "Logistic Regression":
            model = LogisticRegression(max_iter=500)
        else:
            model = DecisionTreeClassifier(random_state=42)

        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        st.write(f"**Accuracy:** {acc:.2f}")
        st.text("Classification Report:")
        st.text(classification_report(y_test, preds))

        # Feature importance visualization (for tree-based models)
        if model_choice in ["Random Forest", "Decision Tree"]:
            importances = model.feature_importances_
            feat_imp_df = pd.DataFrame({
                "Feature": features,
                "Importance": importances
            }).sort_values(by="Importance", ascending=False)

            st.subheader("📈 Feature Importance (Classification)")
            fig_imp = px.bar(feat_imp_df.head(10), 
                             x="Importance", 
                             y="Feature", 
                             orientation="h", 
                             title="Top 10 Important Features for Risk_Level",
                             height=400)
            st.plotly_chart(fig_imp, use_container_width=True)
    else:
        st.warning("No 'Risk_Level' column found for classification.")

    # ---------------------------------
    # Regression Section
    # ---------------------------------
    st.subheader("📉 Risk Score Regression")

    if "Risk_Score" in df.columns:
        target = "Risk_Score"
        features = [col for col in df.columns if col not in [target, "Risk_Level", "Cluster"]]
        X = df[features]
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        reg = RandomForestRegressor(random_state=42)
        reg.fit(X_train, y_train)
        preds = reg.predict(X_test)

        mse = mean_squared_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        st.write(f"**Mean Squared Error:** {mse:.3f}")
        st.write(f"**R² Score:** {r2:.3f}")

        results_df = pd.DataFrame({"Actual": y_test, "Predicted": preds})
        st.dataframe(results_df.head(10))

        fig2 = px.scatter(results_df, 
                          x="Actual", 
                          y="Predicted", 
                          trendline="ols", 
                          title="Actual vs Predicted Risk Score")
        st.plotly_chart(fig2, use_container_width=True)

        # Featu
