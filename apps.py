import streamlit as st
import pickle
import numpy as np

# Load the trained model and vectorizer
with open("svc_model1.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Streamlit App
st.title("Hotel Review Sentiment Analysis")
st.write("Enter a hotel review to predict if it's positive or negative.")

# User input
user_review = st.text_area("Enter your review:")

if st.button("Predict Sentiment"):
    if user_review.strip():
        # Transform the input text
        review_vectorized = vectorizer.transform([user_review])
        prediction = model.predict(review_vectorized)[0]
        sentiment = "Positive" if prediction == 1 else "Negative"

        # Display result
        st.write(f"**Predicted Sentiment:** {sentiment}")
    else:
        st.write("Please enter a review before clicking predict.")