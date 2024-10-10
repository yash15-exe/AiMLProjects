import streamlit as st # type: ignore
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

# Input features using sliders
st.write('Enter the details for prediction:')

median_income = st.slider('Median Income (in $1000s)', min_value=0.0, max_value=15.0, step=0.1, value=5.0)
house_age = st.slider('House Age (in years)', min_value=0, max_value=100, step=1, value=30)
num_bedrooms = st.slider('Number of Bedrooms', min_value=1, max_value=10, step=1, value=3)
population = st.slider('Population', min_value=0, max_value=50000, step=100, value=1000)
latitude = st.slider('Latitude', min_value=32.0, max_value=42.0, step=0.001, value=37.0)
longitude = st.slider('Longitude', min_value=-125.0, max_value=-114.0, step=0.001, value=-119.0)

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
    prediction = model.predict(input_df)

    st.write(f'Estimated House Price: ${prediction[0]:,.2f}')
