import os
import pickle
import pandas as pd
import streamlit as st

# Load the saved model
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "titanic_model.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# App title and description
st.title("🚢 Titanic Survival Predictor")
st.write("Enter passenger details below to predict survival likelihood.")

# Input widgets
pclass = st.selectbox("Passenger Class (Pclass)", ["1", "2", "3"])
sex    = st.selectbox("Sex", ["male", "female"])
age    = st.number_input("Age", min_value=0.0, max_value=100.0, value=30.0, step=1.0)
fare   = st.number_input("Fare ($)", min_value=0.0, max_value=600.0, value=32.0, step=1.0)

# Predict button
if st.button("Predict Survival"):
    input_df = pd.DataFrame([{
        "Pclass": pclass,
        "Sex"   : sex,
        "Age"   : age,
        "Fare"  : fare
    }])

    prob = model.predict_proba(input_df)[0][1]
    pred = model.predict(input_df)[0]

    if pred == 1:
        st.success("✅ This passenger **would likely SURVIVE**")
    else:
        st.error("❌ This passenger **would likely NOT survive**")

    st.write(f"**Survival Probability:** {prob:.1%}")
