# app.py (Enhanced with Raw, Cleaned, and Structured Data Views)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
from sklearn.preprocessing import LabelEncoder
import time

# ------------------- APP CONFIG ------------------- #
st.set_page_config(page_title="AI-Powered Risk Prediction System", layout="wide", page_icon="🌊")

# ------------------- CSS Styling ------------------- #
st.markdown("""
    <style>
    body {background: #0f2027; color: white;}
    .main {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        color: white;
    }
    .glow {
        font-size: 42px;
        color: #00f0ff;
        text-align: center;
        font-weight: bold;
        text-shadow: 0 0 5px #00f0ff, 0 0 15px #00f0ff, 0 0 30px #00f0ff;
        margin-bottom: 25px;
    }
    .glass {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 25px;
        margin-bottom: 25px;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        backdrop-filter: blur(8px);
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="glow">🌊 Predictive Risk Modeling System for Microplastic Pollution</h1>', unsafe_allow_html=True)

# ------------------- SIDEBAR ------------------- #
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio("Go to:", [
    "📂 Upload & Preprocess Data",
    "🤖 Train Predictive Model",
    "📊 Visualization",
    "📈 Risk Level Predictions",
    "🔁 K-Fold Cross Validation"
])

# ------------------- SESSION STATE ------------------- #
for key in ['df_raw', 'df_cleaned', 'df_structured', 'model', 'X', 'y', 'predictions', 'le']:
    if key not in st.session_state:
        st.session_state[key] = None

# ------------------- UPLOAD & PREPROCESS DATA ------------------- #
if page == "📂 Upload & Preprocess Data":
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.header("1️⃣ Upload and Preprocess Dataset")

    uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])
    if uploaded_file:
        try:
            df_raw = pd.read_csv(uploaded_file)
            st.session_state.df_raw = df_raw.copy()
            st.success("✅ Dataset uploaded successfully!")

            st.subheader("📘 Raw Data")
            st.dataframe(df_raw.head(), use_container_width=True)

            # ----- Data Cleaning -----
            df_cleaned = df_raw.drop_duplicates()
            df_cleaned = df_cleaned.dropna(how='all')  # remove empty rows
            df_cleaned = df_cleaned.fillna(df_cleaned.mean(numeric_only=True))  # fill numeric NaN with mean

            st.session_state.df_cleaned = df_cleaned

            st.subheader("🧹 Cleaned Data")
            st.write("Removed duplicates, handled missing values.")
            st.dataframe(df_cleaned.head(), use_container_width=True)

            # ----- Structuring -----
            # Encode non-numeric features
            df_structured = df_cleaned.copy()
            for col in df_structured.select_dtypes(include='object').columns:
                df_structured[col] = df_structured[col].astype(str)
            df_structured = pd.get_dummies(df_structured, drop_first=True)

            st.session_state.df_structured = df_structured

            st.subheader("🏗️ Structured Data (Ready for Modeling)")
            st.write("Encoded categorical features and formatted numeric data.")
            st.dataframe(df_structured.head(), use_container_width=True)

            st.success("✅ Data successfully cleaned and structured!")

        except Exception as e:
            st.error(f"Error reading or processing file: {e}")
    else:
        st.info("📁 Please upload a CSV file to continue.")
    st.markdown('</div>', unsafe_allow_html=True)

# ------------------- TRAIN MODEL ------------------- #
elif page == "🤖 Train Predictive Model":
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.header("2️⃣ Train Predictive Model")

    if st.session_state.df_structured is not None:
        df = st.session_state.df_structured
        target_col = st.selectbox("🎯 Select Target Column", df.columns)
        features = st.multiselect("🧩 Select Feature Columns", [c for c in df.columns if c != target_col])
        model_choice = st.selectbox("🤔 Choose Model", ["Random Forest", "Logistic Regression", "Decision Tree"])

        if st.button("🚀 Train Model"):
            if len(features) == 0:
                st.warning("Please select at least one feature column.")
            else:
                with st.spinner("Training model... ⏳"):
                    time.sleep(1)
                    X = df[features].copy()
                    y = df[target_col].copy()

                    le = None
                    if y.dtype == 'object':
                        le = LabelEncoder()
                        y = le.fit_transform(y)
                        st.session_state.le = le

                    st.session_state.X, st.session_state.y = X, y
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                    if model_choice == "Random Forest":
                        model = RandomForestClassifier(random_state=42)
                    elif model_choice == "Logistic Regression":
                        model = LogisticRegression(max_iter=1000)
                    else:
                        model = DecisionTreeClassifier(random_state=42)

                    model.fit(X_train, y_train)
                    st.session_state.model = model

                    preds = model.predict(X_test)
                    acc = accuracy_score(y_test, preds)
                    prec = precision_score(y_test, preds, average='weighted')
                    rec = recall_score(y_test, preds, average='weighted')
                    f1 = f1_score(y_test, preds, average='weighted')

                    st.success("✅ Model trained successfully!")
                    st.progress(int(acc * 100))

                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Accuracy", f"{acc:.2f}")
                    col2.metric("Precision", f"{prec:.2f}")
                    col3.metric("Recall", f"{rec:.2f}")
                    col4.metric("F1 Score", f"{f1:.2f}")

    else:
        st.warning("Please upload and preprocess your dataset first.")
    st.markdown('</div>', unsafe_allow_html=True)

# ------------------- VISUALIZATION ------------------- #
elif page == "📊 Visualization":
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.header("3️⃣ Data Visualization")

    if st.session_state.df_structured is not None and st.session_state.model is not None:
        X = st.session_state.X
        model = st.session_state.model

        # ------------------- Feature Importance ------------------- #
        if hasattr(model, "feature_importances_"):
            st.subheader("📈 Top 15 Most Important Features")
            importances = pd.DataFrame({
                'Feature': X.columns,
                'Importance': model.feature_importances_
            }).sort_values(by='Importance', ascending=False).head(15)

            fig = px.bar(
                importances.sort_values(by='Importance'),
                x='Importance',
                y='Feature',
                orientation='h',
                color='Importance',
                color_continuous_scale='tealgrn',
                title="Top 15 Features Influencing Predictions"
            )

            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white', size=13),
                xaxis_title="Importance Score",
                yaxis_title="Feature",
                height=600
            )

            st.plotly_chart(fig, use_container_width=True)

        # ------------------- Correlation Heatmap ------------------- #
        st.subheader("🔥 Correlation Heatmap")
        numeric_df = st.session_state.df_structured.select_dtypes(include=np.number)
        if not numeric_df.empty:
            corr = numeric_df.corr()
            fig2 = ff.create_annotated_heatmap(
                z=corr.values,
                x=list(corr.columns),
                y=list(corr.index),
                colorscale='Viridis'
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No numeric columns available for correlation heatmap.")

    else:
        st.warning("Train a model first to visualize data.")
    st.markdown('</div>', unsafe_allow_html=True)

# ------------------- PREDICTIONS ------------------- #
elif page == "📈 Risk Level Predictions":
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.header("4️⃣ Risk Level Predictions")

    if st.session_state.model is not None:
        model = st.session_state.model
        preds = model.predict(st.session_state.X)
        y_true = st.session_state.y
        st.session_state.predictions = preds

        if st.session_state.le:
            preds_display = st.session_state.le.inverse_transform(preds)
        else:
            preds_display = preds

        st.subheader("📊 Model Performance Metrics")
        st.write(f"✅ Accuracy: {accuracy_score(y_true, preds):.2f}")
        st.write(f"🎯 Precision: {precision_score(y_true, preds, average='weighted'):.2f}")
        st.write(f"🔁 Recall: {recall_score(y_true, preds, average='weighted'):.2f}")
        st.write(f"⭐ F1 Score: {f1_score(y_true, preds, average='weighted'):.2f}")

        st.subheader("📉 Confusion Matrix")
        cm = confusion_matrix(y_true, preds)
        fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale='Blues', labels=dict(x="Predicted", y="Actual"))
        st.plotly_chart(fig_cm, use_container_width=True)

    else:
        st.warning("Please train a model first.")
    st.markdown('</div>', unsafe_allow_html=True)

# ------------------- K-FOLD CROSS VALIDATION ------------------- #
elif page == "🔁 K-Fold Cross Validation":
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.header("5️⃣ K-Fold Cross Validation")

    if st.session_state.model is not None:
        X = st.session_state.X
        y = np.array(st.session_state.y)
        k = st.slider("Select number of folds", 2, 10, 5)
        kf = KFold(n_splits=k, shuffle=True, random_state=42)

        results = []
        progress = st.progress(0)
        for i, (train_idx, test_idx) in enumerate(kf.split(X), 1):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            model = type(st.session_state.model)(random_state=42)
            model.fit(X_train, y_train)
            acc = model.score(X_test, y_test)
            results.append({"Fold": i, "Accuracy": round(acc, 4)})
            progress.progress(i / k)

        df_results = pd.DataFrame(results)
        st.table(df_results)
        st.success(f"Mean Accuracy: {np.mean(df_results['Accuracy']):.4f}")

    else:
        st.warning("Please train a model first.")
    st.markdown('</div>', unsafe_allow_html=True)
