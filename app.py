# app.py

import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------
# Load the trained model
# ------------------------------------------------
model = joblib.load("model.pkl")
iris = load_iris()

# ------------------------------------------------
# Streamlit Page Config
# ------------------------------------------------
st.set_page_config(
    page_title="Iris Species Predictor",
    page_icon="🌸",
    layout="centered"
)

st.title("🌸 Iris Flower Species Predictor")
st.write(
    """
    Enter the measurements of the iris flower below to predict its species.
    """
)

# ------------------------------------------------
# Input fields for features
# ------------------------------------------------
sepal_length = st.slider("Sepal length (cm)", 4.0, 8.0, 5.8)
sepal_width  = st.slider("Sepal width (cm)", 2.0, 4.5, 3.0)
petal_length = st.slider("Petal length (cm)", 1.0, 7.0, 4.35)
petal_width  = st.slider("Petal width (cm)", 0.1, 2.5, 1.3)

# Combine input into array
input_features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

# ------------------------------------------------
# Make prediction
# ------------------------------------------------
prediction = model.predict(input_features)
proba = model.predict_proba(input_features)

pred_species = iris.target_names[prediction[0]]

st.subheader(f"🌿 Prediction: **{pred_species.capitalize()}**")

st.write("Prediction Probabilities:")
proba_df = pd.DataFrame(proba, columns=iris.target_names)
st.dataframe(proba_df)

# ------------------------------------------------
# Visualization: show input point vs. training data
# ------------------------------------------------
st.subheader("📊 Visualize Input vs. Dataset")

# Create DataFrame for full data
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

# Add user input
input_df = pd.DataFrame(input_features, columns=iris.feature_names)
input_df['species'] = 'Input'

# Combine
plot_df = pd.concat([iris_df, input_df])

# Plot
fig, ax = plt.subplots()
sns.scatterplot(
    data=iris_df, 
    x='petal length (cm)', 
    y='petal width (cm)', 
    hue='species', 
    palette='Set2',
    s=60
)
plt.scatter(
    input_features[0][2], 
    input_features[0][3], 
    color='black', 
    s=200, 
    marker='X', 
    label='Your Input'
)
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.legend()
st.pyplot(fig)
