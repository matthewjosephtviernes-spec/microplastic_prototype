# app.py – Streamlit app aligned with "NewData - Copy.csv"
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
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_squared_error, r2_score

# ---------------------
# Streamlit App Header
# ---------------------
st.set_page_config(page_title="Microplastic Risk Modeling", layout="wide")
st.title("🌊 Predictive Risk Modeling for Microplastic Pollution")
st.markdown("### Upload your dataset (.csv) aligned with your microplastic research")

# ---------------------
# File Upload Section
# ---------------------
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, encoding="latin1")

    # ---------------------
    # Data Cleaning
    # ---------------------
    df.columns = df.columns.str.strip().str.replace(" ", "_")
    df = df.dropna(how="all")

    # Drop irrelevant column
    if "Author" in df.columns:
        df = df.drop(columns=["Author"])

    # Convert numeric-like columns
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

    # Apply cleaning
    for col in ["Salinity", "Density", "Microplastic_Size"]:
        if col in df.columns:
            df[col] = df[col].apply(extract_numeric)

    # Convert Risk_Score to float
    if "Risk_Score" in df.columns:
        df["Risk_Score"] = pd.to_numeric(df["Risk_Score"], errors="coerce")

    # ---------------------
    # Display Basic Info
    # ---------------------
    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head(10))
    st.write("Shape:", df.shape)
    st.write("Columns:", list(df.columns))

    # ---------------------
    # Feature Encoding
    # ---------------------
    st.subheader("🧩 Data Preprocessing")
    cat_cols = df.select_dtypes(include=["object"]).columns
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = df[col].astype(str)
        df[col] = le.fit_transform(df[col])

    st.success("Categorical columns encoded successfully!")

    # ---------------------
    # Clustering Section
    # ---------------------
    st.subheader("🔶 K-Means Clustering")
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    selected_features = st.multiselect("Select features for clustering", num_cols, default=["Latitude", "Longitude", "MP_Count_per_L", "Risk_Score"])

    if len(selected_features) >= 2:
        k = st.slider("Select number of clusters (k)", 2, 10, 3)
        kmeans = KMeans(n_clusters=k, random_state=42)
        df["Cluster"] = kmeans.fit_predict(df[selected_features])

        fig = px.scatter_3d(df, x=selected_features[0], y=selected_features[1],
                            z=selected_features[2] if len(selected_features) > 2 else selected_features[1],
                            color="Cluster", title="3D Cluster Visualization", height=600)
        st.plotly_chart(fig, use_container_width=True)

    # ---------------------
    # Classification Section
    # ---------------------
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
    else:
        st.warning("No 'Risk_Level' column found for classification.")

    # ---------------------
    # Regression Section
    # ---------------------
    st.subheader("📈 Risk Score Regression")
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
        fig2 = px.scatter(results_df, x="Actual", y="Predicted", trendline="ols", title="Actual vs Predicted Risk Score")
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.warning("No 'Risk_Score' column found for regression.")

else:
    st.info("Please upload a CSV file to begin.")

# ---------------------
# Footer
# ---------------------
st.markdown("---")
st.caption("Developed for the research project: *Predictive Risk Modeling for Microplastic Pollution Using Data Mining Techniques* 🌍")
