import streamlit as st
import joblib
import re

# Load model and vectorizer
model = joblib.load("models/best_sentiment_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

# Clean text function (should match training time)
def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"[^A-Za-z\s]", "", text)
    text = text.lower().strip()
    return text

# Streamlit UI
st.set_page_config(page_title="Sentiment Analysis App", layout="centered")
st.title("🧠 Sentiment Analyzer")
st.subheader("Type a sentence below to detect its sentiment")

# User input
user_input = st.text_area("Enter your text:", placeholder="I love using this app!")

if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text!")
    else:
        clean = clean_text(user_input)
        vectorized = vectorizer.transform([clean])
        prediction = model.predict(vectorized)[0]
        
        # If model supports probabilities (e.g., LogisticRegression, NB)
        try:
            prob = model.predict_proba(vectorized).max()
        except:
            prob = None

        if prediction == 1:
            st.success("🙂 Positive Sentiment")
        else:
            st.error("🙁 Negative Sentiment")

        if prob is not None:
            st.info(f"Confidence: {prob:.2%}")
