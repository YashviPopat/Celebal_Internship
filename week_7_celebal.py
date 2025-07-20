# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# Load or Train the model
@st.cache_data
def train_or_load_model():
    iris = load_iris()
    X, y = iris.data, iris.target
    model = RandomForestClassifier()
    model.fit(X, y)
    return model

model = train_or_load_model()
class_names = ['Setosa', 'Versicolour', 'Virginica']

st.title("Iris Flower Prediction App")
st.markdown("Predict the Iris flower type using input features.")

# Input sliders
sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.1)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.5)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 1.4)
petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

input_df = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                        columns=['sepal length (cm)', 'sepal width (cm)',
                                 'petal length (cm)', 'petal width (cm)'])

# Prediction
if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0]
    st.success(f"Predicted class: **{class_names[prediction]}**")

    st.subheader("Prediction Probabilities")
    proba_df = pd.DataFrame({'Class': class_names, 'Probability': proba})
    st.bar_chart(proba_df.set_index("Class"))

    st.subheader("Feature Importances")
    fig, ax = plt.subplots()
    ax.barh(input_df.columns, model.feature_importances_)
    ax.set_xlabel("Importance")
    st.pyplot(fig)
