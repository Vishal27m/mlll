import streamlit as st
import pickle
import json
import numpy as np  # ✅ Ensure NumPy is imported

st.title("Iris Flower Classification")

# ✅ Load stored metrics
with open("metrics.json", "r") as f:
    loaded_metrics = json.load(f)  

# ✅ Load trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# ✅ Take 4 inputs from the user
sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, step=0.1)
sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, step=0.1)
petal_length = st.number_input("Petal Length (cm)", min_value=0.0, step=0.1)
petal_width = st.number_input("Petal Width (cm)", min_value=0.0, step=0.1)

# ✅ Predict button
if st.button("Predict"):
    # ✅ Convert inputs into a NumPy array
    input_features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    # ✅ Make a prediction
    prediction = model.predict(input_features)

    # ✅ Display the prediction
    st.write(f"Predicted Class: {prediction[0]}")

# ✅ Show stored accuracy
st.metric(label="Model Accuracy", value=f"{loaded_metrics['accuracy']:.2%}")
