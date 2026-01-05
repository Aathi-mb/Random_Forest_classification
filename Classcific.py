import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -------------------------------
# App Title
# -------------------------------
st.set_page_config(page_title="Sales Data Random Forest Classifier", layout="wide")

st.title("🔥 Sales Data Random Forest Classifier 🔥")

# -------------------------------
# Load Dataset
# -------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("sales_dataRFC.csv")

df = load_data()

st.subheader("Dataset Preview")
st.dataframe(df.head())

# -------------------------------
# Target Column
# -------------------------------
target_column = "Sales_Amount"

if target_column not in df.columns:
    st.error(f"Target column '{target_column}' not found!")
    st.write("Available columns:", df.columns.tolist())
    st.stop()

# -------------------------------
# Features & Target
# -------------------------------
X = df.drop(columns=[target_column])
y = df[target_column]

# -------------------------------
# Handle Categorical Data
# -------------------------------
X = pd.get_dummies(X, drop_first=True)

# -------------------------------
# Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# Train Random Forest Model
# -------------------------------
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# -------------------------------
# Predictions & Evaluation
# -------------------------------
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.subheader("📊 Model Performance")

col1, col2, col3 = st.columns(3)
col1.metric("MAE", f"{mae:.2f}")
col2.metric("MSE", f"{mse:.2f}")
col3.metric("R² Score", f"{r2:.2f}")

# -------------------------------
# Feature Importance Plot
# -------------------------------
st.subheader("🔍 Feature Importance")

importances = model.feature_importances_
feature_names = X.columns

fi_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=False).head(15)

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(data=fi_df, x="Importance", y="Feature", ax=ax)
st.pyplot(fig)

# -------------------------------
# Prediction Section
# -------------------------------
st.subheader("🧮 Predict Sales Amount")

sample_input = X.iloc[0:1]

if st.button("Predict Sample Record"):
    prediction = model.predict(sample_input)
    st.success(f"Predicted Sales Amount: ₹ {prediction[0]:.2f}")
