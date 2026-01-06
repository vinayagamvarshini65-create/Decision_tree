import streamlit as st
import pandas as pd
import joblib
import numpy as np

# -----------------------------
# 1. Load saved model and scalers
# -----------------------------
# Ensure these .pkl files are in the same folder as your app.py
dt_tuned = joblib.load("pet_adoption_decision_tree.pkl")
weight_scaler = joblib.load("weight_scaler.pkl")
fee_scaler = joblib.load("adoption_fee_scaler.pkl")

# -----------------------------
# 2. Streamlit App UI
# -----------------------------
st.set_page_config(page_title="Pet Adoption Predictor", page_icon="🐾")
st.title("🐶 Pet Adoption Likelihood Predictor")
st.write("Enter the pet's details to predict the likelihood of adoption.")

col1, col2 = st.columns(2)

with col1:
    pet_type = st.selectbox("Pet Type", ["Bird", "Cat", "Dog", "Rabbit"])
    breed = st.selectbox("Breed", ["Parakeet", "Rabbit", "Golden Retriever", "Siamese", "Persian", "Poodle", "Labrador"])
    age_months = st.number_input("Age (months)", min_value=0, max_value=240, value=12)
    color = st.selectbox("Color", ["Orange", "White", "Gray", "Brown", "Black"])
    size = st.selectbox("Size", ["Small", "Medium", "Large"])

with col2:
    weight_kg = st.number_input("Weight (kg)", min_value=0.1, max_value=100.0, value=5.0)
    adoption_fee = st.number_input("Adoption Fee ($)", min_value=0.0, max_value=500.0, value=50.0)
    timein_shelter = st.number_input("Days in Shelter", min_value=1, value=30)
    vaccinated = st.selectbox("Vaccinated", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
    health = st.selectbox("Health Condition", [0, 1], format_func=lambda x: "Healthy" if x == 0 else "Medical Condition")
    prev_owner = st.selectbox("Previous Owner", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")

# Map Size manually (Ordinal Encoding used in training)
size_map = {"Small": 2, "Medium": 1, "Large": 0}
size_encoded = size_map[size]

# -----------------------------
# 3. Prediction Logic
# -----------------------------
if st.button("Predict Adoption Likelihood"):
    
    # Define ALL features exactly as seen during model.fit()
    # Numeric/Ordinal features first, then One-Hot columns
    data = {
        "age_months": [age_months],
        "weight_kg": [weight_kg],
        "size": [size_encoded],
        "vaccinated": [vaccinated],
        "health_condition": [health],
        "timein_shelter_days": [timein_shelter],
        "adoption_fee": [adoption_fee],
        "previous_owner": [prev_owner],
        
        # One-Hot Encoding Placeholders (Pet Type)
        "pet_type_Bird": [0], "pet_type_Cat": [0], "pet_type_Dog": [0], "pet_type_Rabbit": [0],
        
        # One-Hot Encoding Placeholders (Color)
        "color_Black": [0], "color_Brown": [0], "color_Gray": [0], "color_Orange": [0], "color_White": [0],
        
        # One-Hot Encoding Placeholders (Breed)
        "breed_Golden Retriever": [0], "breed_Labrador": [0], "breed_Parakeet": [0], 
        "breed_Persian": [0], "breed_Poodle": [0], "breed_Rabbit": [0], "breed_Siamese": [0]
    }

    # Set the selected category to 1
    if f"pet_type_{pet_type}" in data:
        data[f"pet_type_{pet_type}"] = [1]
    if f"color_{color}" in data:
        data[f"color_{color}"] = [1]
    if f"breed_{breed}" in data:
        data[f"breed_{breed}"] = [1]

    # Create DataFrame
    input_df = pd.DataFrame(data)
    
    # Scale numeric features using the loaded scalers
    input_df["weight_kg"] = weight_scaler.transform(input_df[["weight_kg"]])
    input_df["adoption_fee"] = fee_scaler.transform(input_df[["adoption_fee"]])
    
    # Predict
    try:
        # Match the internal feature order of the model
        input_df = input_df[dt_tuned.feature_names_in_]
        
        prediction = dt_tuned.predict(input_df)[0]
        probability = dt_tuned.predict_proba(input_df)[0][prediction]
        
        if prediction == 1:
            st.success(f"### Result: Likely to be Adopted")
        else:
            st.warning(f"### Result: Unlikely to be Adopted")
            
        st.info(f"**Confidence:** {probability*100:.2f}%")
        
    except Exception as e:
        st.error(f"Error during prediction: {e}")