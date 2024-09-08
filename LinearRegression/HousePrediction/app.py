import streamlit as st
import pickle
import numpy as np
import pandas as pd
import gzip

try:
    with gzip.open('./model_new.pklz', 'rb') as file:
        model = pickle.load(file)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")


# Define the Streamlit app
st.title('California House Price Prediction')

# Input features
st.write('Enter the details for prediction:')

# Example feature names based on common features for house price models
median_income = st.number_input('Median Income (in $1000s)', format="%.2f")
house_age = st.number_input('House Age (in years)', format="%.2f")
num_bedrooms = st.number_input('Number of Bedrooms', format="%.2f")
population = st.number_input('Population', format="%.2f")
latitude = st.number_input('Latitude', format="%.6f")
longitude = st.number_input('Longitude', format="%.6f")

if st.button('Predict'):
    # Convert input features to a DataFrame
    input_df = pd.DataFrame({
        'Median Income': [median_income],
        'House Age': [house_age],
        'Number of Bedrooms': [num_bedrooms],
        'Population': [population],
        'Latitude': [latitude],
        'Longitude': [longitude]
    })

    # Ensure the order of features matches the model's training data
    # Assuming the model was trained with features in the same order
    prediction = model.predict(input_df)

    st.write(f'Estimated House Price: ${prediction[0]:,.2f}')
