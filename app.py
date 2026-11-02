import os
import joblib
import pandas as pd
import streamlit as st

from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, recall_score,f1_score, matthews_corrcoef)

TARGET_COLUMN = "Class"

st.set_page_config(page_title="Dry Bean Classification")
st.title("Dry Bean Classification")
st.write("This application evaluates five trained machine learning models for Dry Bean classification.")
st.write("Upload the test CSV file and select a model to view the evaluation results.")


MODEL_FILES = {
  "Logistic Regression": "model/logistic_regression.joblib",
  "Decision Tree": "model/decision_tree.joblib",
  "K-Nearest Neighbors (KNN)": "model/knn.joblib",
  "Naive Bayes": "model/naive_bayes.joblib",
  "Support Vector Machine (SVM)": "model/svm.joblib"
}

def load_model(model_path):
    return joblib.load(model_path)

uploaded_file = st.file_uploader("Upload Test Data",  type=["csv"], help="Upload test_data.csv")
selected_model = st.selectbox("Select ML Model",  options=list(MODEL_FILES.keys()))

if uploaded_file is None:
    st.info("Upload test_data.csv to view the model evaluation results.")
else:
    test_data = pd.read_csv(uploaded_file)


st.subheader("Uploaded Test Data")
st.write("Number of rows:", test_data.shape[0])
st.write("Number of columns:", test_data.shape[1])

#st.dataframe(test_data.head(), use_container_width=True)

if TARGET_COLUMN not in test_data.columns:
    st.error("The uploaded CSV must contain the target column 'Class'.")
    st.stop()

model_path = MODEL_FILES[selected_model]

if not os.path.exists(model_path):
    st.error(f"Model file not found: {model_path}")
    st.stop()

model = load_model(model_path)

X_test = test_data.drop(columns=[TARGET_COLUMN])
y_test = test_data[TARGET_COLUMN]

y_pred = model.predict(X_test)
y_probability = model.predict_proba(X_test)

accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_probability, labels=model.classes_, multi_class="ovr",  average="weighted")
precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
mcc = matthews_corrcoef(y_test, y_pred)


st.subheader(f"Evaluation Results - {selected_model}")
rows = st.columns(6)

rows[0].metric("Accuracy", f"{accuracy:.4f}")
rows[1].metric("AUC", f"{auc:.4f}")
rows[2].metric("Precision", f"{precision:.4f}")
rows[3].metric("Recall", f"{recall:.4f}")
rows[4].metric("F1 Score", f"{f1:.4f}")
rows[5].metric("MCC", f"{mcc:.4f}")

