import pickle
import string

import nltk
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


# -------------------------
# Configuration
# -------------------------

VECTORIZER_PATH = "vectorizer.pkl"
MODEL_PATH = "model.pkl"


# -------------------------
# NLTK Setup
# -------------------------

nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))


# -------------------------
# Text Preprocessing
# -------------------------

def transform_text(text: str) -> str:
    text = text.lower()

    tokens = nltk.word_tokenize(text)

    tokens = [
        token
        for token in tokens
        if token.isalnum()
        and token not in stop_words
        and token not in string.punctuation
    ]

    tokens = [stemmer.stem(token) for token in tokens]

    return " ".join(tokens)


# -------------------------
# Load Model and Vectorizer
# -------------------------

@st.cache_resource
def load_model():
    with open(VECTORIZER_PATH, "rb") as file:
        vectorizer = pickle.load(file)

    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)

    return vectorizer, model


tfidf, model = load_model()


# -------------------------
# Streamlit Configuration
# -------------------------

st.set_page_config(
    page_title="SMS Spam Classifier",
    page_icon=None,
    layout="centered"
)

st.title("SMS Spam Classifier")
st.write("Enter an SMS message to determine whether it is spam or not spam.")


# -------------------------
# User Input
# -------------------------

input_sms = st.text_area(
    "Enter your message",
    placeholder="Enter an SMS message here...",
    height=150
)


# -------------------------
# Prediction
# -------------------------

if st.button("Predict", use_container_width=True):

    if not input_sms.strip():
        st.warning("Please enter a message.")
    else:
        transformed_sms = transform_text(input_sms)

        vector_input = tfidf.transform([transformed_sms])

        prediction = model.predict(vector_input)[0]

        if prediction == 1:
            st.error("Spam")
        else:
            st.success("Not Spam")
