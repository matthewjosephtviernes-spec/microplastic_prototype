# Streamlit Microplastic Risk Classifier Dashboard
# Single-file Streamlit app implementing:
# - File upload or default dataset
# - Preprocessing (outliers, skewness, encoding, scaling)
# - Multiple model training and comparison (RF, SVM, LR, KNN, GB)
# - Stratified K-Fold cross-validation
# - PCA visualization and feature importance
# - Predict new samples via a form

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, PowerTransformer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import io
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(layout='wide', page_title='Microplastic Risk Classifier')

# ----------------------
# Utility functions
# ----------------------

def load_data(uploaded_file):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    try:
        return pd.read_csv('MicroPlastic.csv')
    except FileNotFoundError:
        st.error('No dataset uploaded and default "MicroPlastic.csv" not found.')
        return None


def handle_outliers(df, num_cols):
    # Winsorize using IQR method (cap at 1.5*IQR)
    df = df.copy()
    for col in num_cols:
        if df[col].isnull().all():
            continue
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        df[col] = np.where(df[col] < lower, lower, df[col])
        df[col] = np.where(df[col] > upper, upper, df[col])
    return df


def transform_skewness(df, num_cols):
    # Use Yeo-Johnson (works with non-positive values)
    df = df.copy()
    pt = PowerTransformer(method='yeo-johnson', standardize=False)
    df[num_cols] = pt.fit_transform(df[num_cols].fillna(0))
    return df, pt


def encode_and_scale(df, cat_cols, num_cols):
    df = df.copy()
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols].astype(float))
    return df, encoders, scaler


def train_models(X_train, y_train):
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42),
        'SVM (RBF)': SVC(probability=True, kernel='rbf', random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=500),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42)
    }
    fitted = {}
    for name, m in models.items():
        m.fit(X_train, y_train)
        fitted[name] = m
    return fitted


def plot_conf_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, cmap=plt.cm.Blues, colorbar=False)
    plt.tight_layout()
    return fig

# ----------------------
# Sidebar - Upload + Options
# ----------------------
st.sidebar.header('Dataset & Options')
uploaded_file = st.sidebar.file_uploader('Upload your MicroPlastic CSV', type=['csv'])
raw_df = load_data(uploaded_file)
if raw_df is None:
    st.stop()

st.sidebar.markdown('---')
if st.sidebar.button('Show raw data'):
    st.write(raw_df.head())

# Auto-detect target if present
default_target = 'Risk_Level' if 'Risk_Level' in raw_df.columns else None
target = st.sidebar.text_input('Target column (classification label)', value=default_target)
if not target:
    st.error('Please specify the target column name in the sidebar.')
    st.stop()

# ----------------------
# Preprocessing options
# ----------------------
st.sidebar.header('Preprocessing')
remove_outliers = st.sidebar.checkbox('Handle outliers (IQR winsorize)', value=True)
apply_skew_transform = st.sidebar.checkbox('Transform skewed numeric features (Yeo-Johnson)', value=True)
scale_features = st.sidebar.checkbox('Scale numerical features (StandardScaler)', value=True)

# ----------------------
# Prepare data
# ----------------------
st.header('Dataset & Preprocessing')

df = raw_df.copy()
# Basic cleaning: drop rows with no target
initial_rows = df.shape[0]
df = df.dropna(subset=[target])
rows_after_drop = df.shape[0]

st.write(f'Rows: {initial_rows} → after dropping missing target: {rows_after_drop}')

# Infer numeric / categorical
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
# If some numeric-looking columns are object typed, try to coerce
for col in df.columns:
    if col not in num_cols and df[col].dtype == 'object':
        try:
            df[col] = pd.to_numeric(df[col].str.replace(',', '').replace('', np.nan))
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        except Exception:
            pass

cat_cols = [c for c in df.columns if c not in num_cols and c != target]
num_cols = [c for c in num_cols if c != target]

st.write('Numeric columns detected:', num_cols)
st.write('Categorical columns detected:', cat_cols)

# Handle outliers
if remove_outliers and len(num_cols) > 0:
    df = handle_outliers(df, num_cols)
    st.write('Outliers handled (IQR winsorize) for numeric columns.')

# Transform skewness
pt = None
if apply_skew_transform and len(num_cols) > 0:
    df[num_cols] = df[num_cols].fillna(0)
    df, pt = transform_skewness(df, num_cols)
    st.write('Applied Yeo-Johnson transform to numeric columns to reduce skewness.')

# Encode and scale
encoders = {}
scaler = None
if len(cat_cols) > 0 or len(num_cols) > 0:
    df_encoded = df.copy()
    if len(cat_cols) > 0:
        for col in cat_cols:
            df_encoded[col] = df_encoded[col].astype(str)
    if scale_features:
        df_encoded, encoders, scaler = encode_and_scale(df_encoded, cat_cols, num_cols)
    else:
        # only encode
        for col in cat_cols:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            encoders[col] = le
    st.write('Data encoded and scaled (if selected).')
else:
    st.warning('No features detected for processing.')

# Final X, y
X = df_encoded.drop(columns=[target])
y = df_encoded[target]

st.write('Processed feature matrix shape:', X.shape)

# ----------------------
# Train-test split & model training
# ----------------------
st.header('Model Training & Comparison')

test_size = st.slider('Test size (%)', 5, 50, 20)
random_state = 42

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size/100.0, stratify=y, random_state=random_state
)

st.write('Train shape:', X_train.shape, 'Test shape:', X_test.shape)

with st.spinner('Training models...'):
    fitted_models = train_models(X_train, y_train)

# Evaluate
scores = {}
for name, m in fitted_models.items():
    preds = m.predict(X_test)
    acc = accuracy_score(y_test, preds)
    scores[name] = acc

st.subheader('Model accuracy on test set')
st.table(pd.DataFrame.from_dict(scores, orient='index', columns=['Accuracy']).sort_values('Accuracy', ascending=False))

# Cross-validation
st.subheader('Stratified K-Fold Cross-Validation (5 folds)')
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
cv_scores = {}
for name, m in fitted_models.items():
    cv = cross_val_score(m, X, y, cv=skf, scoring='accuracy')
    cv_scores[name] = (cv.mean(), cv.std())

cv_df = pd.DataFrame(cv_scores, index=['mean', 'std']).T
cv_df.columns = ['CV Mean Accuracy', 'CV Std']
st.table(cv_df.sort_values('CV Mean Accuracy', ascending=False))

# Show classification report & confusion matrix for selected model
selected_model_name = st.selectbox('Select model for detailed view', list(fitted_models.keys()))
selected_model = fitted_models[selected_model_name]

y_pred = selected_model.predict(X_test)
st.subheader(f'Classification report: {selected_model_name}')
st.text(classification_report(y_test, y_pred))

st.subheader('Confusion Matrix')
labels = np.unique(y)
fig_cm = plot_conf_matrix(y_test, y_pred, labels=labels)
st.pyplot(fig_cm)

# ----------------------
# Feature importance (if RF or GB)
# ----------------------
st.header('Model Explainability')
if hasattr(selected_model, 'feature_importances_'):
    fi = selected_model.feature_importances_
    fi_df = pd.DataFrame({'feature': X.columns, 'importance': fi}).sort_values('importance', ascending=False).head(20)
    st.subheader('Top features by importance')
    st.table(fi_df)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(fi_df['feature'][::-1], fi_df['importance'][::-1])
    ax.set_xlabel('Importance')
    plt.tight_layout()
    st.pyplot(fig)
else:
    st.info('Selected model has no feature_importances_ attribute. Try Random Forest or Gradient Boosting.')

# ----------------------
# PCA Visualization
# ----------------------
st.header('PCA Visualization')
if X.shape[1] >= 2:
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    fig, ax = plt.subplots(figsize=(6, 5))
    unique_labels = np.unique(y)
    for lbl in unique_labels:
        mask = (y == lbl)
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], label=str(lbl), alpha=0.7)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.legend()
    st.pyplot(fig)
else:
    st.info('Not enough features for PCA plot.')

# ----------------------
# Predict new sample
# ----------------------
st.header('Predict New Sample')
with st.form('new_sample_form'):
    st.write('Fill in feature values for a single sample to predict Risk Level')
    sample = {}
    for col in X.columns:
        if col in cat_cols:
            # show a text input but try to list unique values
            opts = raw_df[col].dropna().unique().tolist()[:50]
            if len(opts) > 0:
                sample[col] = st.selectbox(col, options=opts, index=0)
            else:
                sample[col] = st.text_input(col, value='')
        else:
            sample[col] = st.number_input(col, value=float(raw_df[col].dropna().median()) if col in raw_df.columns else 0.0)
    submitted = st.form_submit_button('Predict')

if submitted:
    # Build DataFrame
    new_df = pd.DataFrame([sample])
    # Apply same transforms: encode, skew-transform (if used), scale
    # 1) apply string coercion for categorical
    for col in cat_cols:
        if col in new_df.columns:
            if col in encoders:
                # make sure unseen categories do not break
                val = str(new_df.at[0, col])
                le = encoders[col]
                if val in le.classes_.tolist():
                    new_df[col] = le.transform([val])
                else:
                    # unseen category: append then transform
                    new_classes = list(le.classes_) + [val]
                    le.classes_ = np.array(new_classes)
                    new_df[col] = le.transform([val])
            else:
                new_df[col] = 0
    # 2) numeric ordering and missing
    for col in num_cols:
        if col not in new_df.columns:
            new_df[col] = 0.0
    # 3) skew transform - apply pt if used
    if pt is not None and len(num_cols) > 0:
        new_df[num_cols] = pt.transform(new_df[num_cols].astype(float).fillna(0))
    # 4) scale
    if scaler is not None and len(num_cols) > 0:
        new_df[num_cols] = scaler.transform(new_df[num_cols].astype(float))

    # Predict using selected model
    pred = selected_model.predict(new_df[X.columns])[0]
    probs = None
    if hasattr(selected_model, 'predict_proba'):
        probs = selected_model.predict_proba(new_df[X.columns])[0]
        classes = selected_model.classes_

    st.subheader('Prediction result')
    st.write('Predicted class:', pred)
    if probs is not None:
        prob_df = pd.DataFrame({'class': classes, 'probability': probs}).sort_values('probability', ascending=False)
        st.table(prob_df)

# ----------------------
# Save trained model (optional)
# ----------------------
st.sidebar.markdown('---')
if st.sidebar.button('Download best model (pickle)'):
    import pickle
    # choose best by cv mean
    best_name = max(cv_scores, key=lambda k: cv_scores[k][0])
    best_model = fitted_models[best_name]
    buf = io.BytesIO()
    pickle.dump({'model': best_model, 'encoders': encoders, 'scaler': scaler, 'pt': pt, 'features': X.columns.tolist()}, buf)
    buf.seek(0)
    st.sidebar.download_button('Download model pickle', data=buf, file_name='microplastic_model.pkl')

st.markdown('---')
st.write('Built with ❤️ — Streamlit Microplastic Risk Classifier')
