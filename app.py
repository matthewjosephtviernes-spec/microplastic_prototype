# ==============================================================
# Predictive Risk Modeling for Microplastic Pollution
# Developed by: Magdaluyo & Viernes
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
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error reading CSV file: {e}")
        st.stop()

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
        if len(num_cols) < 2:
            st.warning("At least 2 numeric columns are required for the default K-Means scatter visualization. Clustering will still run but visualization will be a distribution plot.")
        k = st.slider("Select number of clusters", 2, 6, 3)

        # Prepare numeric data for clustering: drop rows with missing numeric values
        df_num = df[num_cols].dropna()
        if df_num.empty:
            st.error("No rows with complete numeric data available for clustering. Please clean or impute missing values.")
        else:
            # Use an integer n_init for compatibility across sklearn versions
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(df_num)

            # Assign cluster labels back to the original dataframe aligning by index
            df.loc[df_num.index, "Cluster"] = cluster_labels
            df["Cluster"] = df["Cluster"].astype("Int64")  # allow NA clusters for rows dropped earlier

            st.write("Cluster counts:")
            st.write(df["Cluster"].value_counts(dropna=True))
            st.write("Cluster Visualization:")

            plt.figure(figsize=(7, 5))
            if len(num_cols) >= 2:
                sns.scatterplot(x=num_cols[0], y=num_cols[1], hue="Cluster", data=df.loc[df_num.index], palette="viridis")
                plt.title("K-Means Clustering Visualization")
                st.pyplot(plt.gcf())
                plt.clf()
            else:
                # If only one numeric column is present, show cluster-wise distribution for that column
                single_col = num_cols[0]
                sns.boxplot(x="Cluster", y=single_col, data=df.loc[df_num.index], palette="viridis")
                plt.title(f"Cluster distribution on {single_col}")
                st.pyplot(plt.gcf())
                plt.clf()

    else:
        st.info("No numeric columns found in the dataset. K-Means requires numeric features.")

    # ----------------------------------------------------------
    # CLASSIFICATION MODEL
    # ----------------------------------------------------------
    st.header("4️⃣ Classification and Prediction")

    target_col = st.selectbox("Select Target Column (Risk Category)", options=df.columns)
    features = [col for col in df.columns if col != target_col]

    if st.button("Run Classification Model"):
        # Prepare target
        y = df[target_col]

        # Encode target if it's categorical/object
        le = LabelEncoder()
        if y.dtype == 'object' or y.dtype.name == 'category':
            try:
                y = pd.Series(le.fit_transform(y), index=y.index)
            except Exception as e:
                st.error(f"Error encoding target column: {e}")
                st.stop()

        # Prepare features: select numeric columns only
        X = df[features].select_dtypes(include=[np.number])

        if X.shape[1] == 0:
            st.error("No numeric features available for training. Select a dataset with numeric predictors or preprocess categorical features.")
            st.stop()

        # Drop rows with missing values in features or target, keeping alignment
        combined = pd.concat([X, y], axis=1)
        combined = combined.dropna()
        if combined.empty:
            st.error("No rows remain after dropping missing values. Please clean or impute your data.")
            st.stop()

        X = combined[X.columns]
        y = combined[target_col] if target_col in combined.columns else combined.iloc[:, -1]

        # Ensure y is numeric after encoding (if it came from object)
        if y.dtype == 'object' or y.dtype.name == 'category':
            try:
                y = pd.Series(le.fit_transform(y), index=y.index)
            except Exception:
                st.error("Target column could not be encoded to numeric classes.")
                st.stop()

        # Train/test split
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        except Exception as e:
            st.error(f"Error during train/test split: {e}")
            st.stop()

        # Model training
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        try:
            model.fit(X_train, y_train)
        except Exception as e:
            st.error(f"Error training model: {e}")
            st.stop()

        # Prediction and evaluation
        try:
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
        except Exception as e:
            st.error(f"Error during evaluation: {e}")
            st.stop()

    # ----------------------------------------------------------
    # SUMMARY
    # ----------------------------------------------------------
    st.header("5️⃣ Summary")
    st.info("""
    - The **K-Means** clustering grouped samples into distinct ecological, chemical, or human health risks.  
    - The **Random Forest** classifier achieved strong predictive accuracy (depending on dataset and preprocessing).  
    - Visual outputs such as heatmaps and scatterplots improve interpretability.  
    - This system supports data-driven **environmental decision-making**.
    """)
else:
    st.warning("Please upload a dataset to begin analysis.")

st.caption("© 2025 Magdaluyo & Viernes | Microplastic Risk Prediction Project")
