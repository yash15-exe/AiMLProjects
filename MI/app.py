import streamlit as st
import joblib
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk

nltk.download('stopwords')

# Initialize the PorterStemmer
port_stem = PorterStemmer()

# Define the stemming function
def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if word not in stopwords.words("english")]
    stemmed_content = " ".join(stemmed_content)
    return stemmed_content

# Load the vectorizer and model
vectorizer = joblib.load("C:/Users/yashr/OneDrive/Desktop/AiMl/MI/tfidf_vectorizer.pkl")
model = joblib.load("C:/Users/yashr/OneDrive/Desktop/AiMl/MI/fake_predicting_model.pkl")

# Set page config
st.set_page_config(page_title="Fake News Predictor", page_icon="📰", layout="wide")

# Add a title and a subtitle
st.title("📰 Fake News Prediction")
st.subheader("Enter the news article below to predict if it is fake or real.")

# Add a text area for user input with some styling
content = st.text_area("Enter your news article:", height=300, placeholder="Type or paste your news article here...")

# Add a button with some custom styling
if st.button("🔍 Predict", key="predict_button"):
    if content:
        # Preprocess the input
        processed_content = stemming(content)

        # Transform the input using the vectorizer
        processed_content_vectorized = vectorizer.transform([processed_content])

        # Make a prediction
        prediction = model.predict(processed_content_vectorized)

        # Display the result with conditional styling
        if prediction[0] == 1:
            st.success("✅ The above news is possibly a **FAKE NEWS**")
        else:
            st.success("✅ The above news is possibly a **REAL NEWS**")
    else:
        st.warning("⚠️ Please enter some text for classification.")

# Footer
st.markdown("---")
st.markdown("### About this App")
st.write("This app uses a machine learning model to predict whether a news article is fake or real.")
