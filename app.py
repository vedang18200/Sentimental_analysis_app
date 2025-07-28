import streamlit as st
import joblib
import re

# Load model and vectorizer
# Ensure these paths are correct relative to where your script is run
try:
    model = joblib.load("models/best_sentiment_model.pkl")
    vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
except FileNotFoundError:
    st.error("Error: Model or vectorizer files not found. Please ensure 'models/best_sentiment_model.pkl' and 'models/tfidf_vectorizer.pkl' exist.")
    st.stop() # Stop the app if essential files are missing

# Clean text function (should match training time)
def clean_text(text):
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"@\w+", "", text)     # Remove mentions
    text = re.sub(r"#\w+", "", text)     # Remove hashtags
    text = re.sub(r"[^A-Za-z\s]", "", text) # Remove punctuation and numbers
    text = text.lower().strip()          # Convert to lowercase and strip whitespace
    return text

# --- Streamlit UI ---
st.set_page_config(
    page_title="🌟 Smart Sentiment Analyzer",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Header Section
st.markdown(
    """
    <style>
    .big-title {
        font-size: 3em;
        text-align: center;
        color: #FF6347; /* Tomato */
    }
    .subheader {
        font-size: 1.5em;
        text-align: center;
        color: #4682B4; /* SteelBlue */
        margin-bottom: 30px;
    }
    .stButton>button {
        background-color: #4CAF50; /* Green */
        color: white;
        font-size: 1.2em;
        padding: 10px 24px;
        border-radius: 8px;
        border: none;
        cursor: pointer;
        transition: all 0.2s ease-in-out;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: scale(1.02);
    }
    .stTextArea label {
        font-size: 1.1em;
        font-weight: bold;
    }
    .positive-box {
        background-color: #D4EDDA; /* Light green */
        color: #155724; /* Dark green */
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #C3E6CB;
        margin-top: 20px;
        font-size: 1.2em;
        text-align: center;
    }
    .negative-box {
        background-color: #F8D7DA; /* Light red */
        color: #721C24; /* Dark red */
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #F5C6CB;
        margin-top: 20px;
        font-size: 1.2em;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<h1 class='big-title'>🧠 Smart Sentiment Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<h2 class='subheader'>Unlock the Emotion Behind Your Words!</h2>", unsafe_allow_html=True)

st.write("---") # Horizontal line for separation

# About Section
with st.expander("🤔 About This App"):
    st.write(
        """
        This application uses a pre-trained machine learning model to determine the sentiment 
        (positive or negative) of any text you enter. It's designed to give you quick insights 
        into the emotional tone of sentences.
        """
    )

# Input Section
st.header("✍️ Enter Your Text")
user_input = st.text_area(
    "Type your statement here and hit 'Analyze'!",
    placeholder="Example: 'I love this new feature!' or 'I'm so frustrated with the slow service.'",
    height=150
)

# Analysis Button
col1, col2, col3 = st.columns([1,2,1])
with col2:
    analyze_button = st.button("✨ Analyze Sentiment")

# --- Sentiment Analysis Logic ---
if analyze_button:
    if user_input.strip() == "":
        st.warning("Please enter some text to analyze!")
    else:
        clean = clean_text(user_input)
        
        # Check if the cleaned text is empty after processing
        if not clean:
            st.warning("The text you entered resulted in an empty string after cleaning. Please try different text.")
            # Debug info can still be useful here
            with st.expander("Debug Info"):
                st.write(f"Original text: '{user_input}'")
                st.write(f"Cleaned text: '{clean}'")
            st.stop() # Stop further execution if cleaned text is empty

        vectorized = vectorizer.transform([clean])
        
        # Initial prediction
        prediction = model.predict(vectorized)[0]
        
        # Try to get model confidence (probability)
        try:
            # Get probabilities for all classes and take the max
            prob_all_classes = model.predict_proba(vectorized)
            prob = prob_all_classes.max()
            # Ensure the probability corresponds to the predicted class
            predicted_class_index = model.predict(vectorized)[0]
            prob_of_predicted_class = prob_all_classes[0, predicted_class_index]
            prob = prob_of_predicted_class # Use the probability of the predicted class
        except AttributeError: # Some models might not have predict_proba
            prob = None
            st.warning("Model does not support probability prediction.")


        # ✅ Enhanced override for positive aspirational phrases
        positive_phrases = [
            "independent woman", "empowered", "confident",
            "self-made", "strong woman", "believe in myself",
            "i want to be an independent woman",
            
            # Career aspiration phrases
            "i want to be", "i want to become", "my dream is to be",
            "i aspire to be", "i hope to become", "my goal is to be",
            "i wish to be", "i aim to be", "i plan to be",
            
            # Specific career mentions that are typically positive
            "want to be a doctor", "want to be a teacher", "want to be a cop",
            "want to be a nurse", "want to be an engineer", "want to be a lawyer",
            "want to be a pilot", "want to be a scientist", "want to be an artist"
        ]
        
        # Check for aspirational patterns
        aspirational_patterns = [
            r"i want to be a \w+",
            r"i want to become a \w+", 
            r"my dream is to be a \w+",
            r"i aspire to be a \w+"
        ]
        
        is_aspirational = False
        for pattern in aspirational_patterns:
            if re.search(pattern, clean):
                is_aspirational = True
                break
        
        # Override logic
        if any(phrase in clean for phrase in positive_phrases) or is_aspirational:
            prediction = 1  # Force to positive
            prob = 0.98     # Set custom high confidence for overrides
            st.info("💡 Special Rule Applied: This statement is identified as aspirational or strongly positive, overriding general model prediction.")
            override_applied = True
        else:
            override_applied = False

        st.write("---") # Separator before results

        # Display Result
        st.subheader("📊 Analysis Result:")
        if prediction == 1:
            st.markdown("<div class='positive-box'>✨ **Positive Sentiment!** Your text radiates positivity!</div>", unsafe_allow_html=True)
            st.balloons() # Fun animation for positive sentiment
        else:
            st.markdown("<div class='negative-box'>😞 **Negative Sentiment.** Your text conveys a negative tone.</div>", unsafe_allow_html=True)
            
        if prob is not None:
            st.write("") # Add some space
            st.markdown(f"**Confidence:**")
            st.progress(prob) # Visual progress bar for confidence
            st.write(f"*(Model is {prob:.2%} confident)*")

        # Debug Information (Optional - highly recommended for development)
        st.write("---")
        with st.expander("🛠️ Debug Information (For Developers)"):
            st.write(f"**Cleaned text:** `{clean}`")
            # Re-predict to show original prediction before override
            original_prediction = model.predict(vectorized)[0] 
            st.write(f"**Original model prediction:** {'Positive (1)' if original_prediction == 1 else 'Negative (0)'}")
            st.write(f"**Override applied:** {override_applied}")
            if prob is not None and 'prob_all_classes' in locals():
                st.write(f"**Raw probabilities (Negative, Positive):** `{prob_all_classes[0]}`")
            st.write("---")
            st.write("Model and Vectorizer Details (Internal):")
            st.write(f"Model type: `{type(model).__name__}`")
            st.write(f"Vectorizer type: `{type(vectorizer).__name__}`")
            if hasattr(vectorizer, 'get_feature_names_out'):
                st.write(f"Number of features (vocabulary size): `{len(vectorizer.get_feature_names_out())}`")