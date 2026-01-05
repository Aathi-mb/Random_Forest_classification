# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Random Forest Classifier", layout="wide")
st.title("🔥 Sales Data Random Forest Classifier 🔥")

# -----------------------------
# Step 1: Load Dataset & Model
# -----------------------------
# Load dataset (to get feature names and types)
df = pd.read_csv('sales_dataRFC.csv')
st.subheader("Dataset Preview")
st.dataframe(df.head())

# Define target column
target_column = 'Target'  # Change to your actual target column name

X = df.drop(columns=[target_column])
y = df[target_column]

# Detect categorical columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

# Load trained Random Forest model
model = joblib.load("random_forest_model.pkl")

# If you saved label encoders during training
# label_encoders = joblib.load("label_encoders.pkl")
# le_target = joblib.load("le_target.pkl")

# -----------------------------
# Step 2: User Input Form
# -----------------------------
st.header("Enter Input Features")

user_input = {}

for col in X.columns:
    if col in categorical_cols:
        user_input[col] = st.text_input(f"{col} (categorical)")
    else:
        user_input[col] = st.number_input(f"{col} (numeric)", value=float(X[col].mean()))

new_df = pd.DataFrame([user_input])

# -----------------------------
# Step 3: Encode Categorical Features
# -----------------------------
# If you have label encoders, uncomment below
# for col in categorical_cols:
#     try:
#         new_df[col] = label_encoders[col].transform(new_df[col])
#     except ValueError:
#         new_df[col] = -1  # unseen category

# -----------------------------
# Step 4: Make Prediction
# -----------------------------
if st.button("Predict"):
    prediction = model.predict(new_df)
    # If target encoded, decode:
    # pred_class = le_target.inverse_transform(prediction)[0]
    pred_class = prediction[0]

    st.success(f"Predicted Class: {pred_class}")

    # -----------------------------
    # Step 5: Feature Importance Plot
    # -----------------------------
    st.subheader("Feature Importance")
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]

    plt.figure(figsize=(10,5))
    sns.barplot(x=importances[sorted_idx], y=np.array(X.columns)[sorted_idx])
    plt.title("Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    st.pyplot(plt)
