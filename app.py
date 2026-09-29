import streamlit as st
import pickle
import string
import nltk

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()
stop_words = set(stopwords.words("english"))


# -------------------------
# Text Preprocessing
# -------------------------

def transform_text(text):

    # Lowercase
    text = text.lower()

    # Tokenization
    text = nltk.word_tokenize(text)

    # Remove special characters
    y = []

    for i in text:
        if i.isalnum():
            y.append(i)

    # Remove stopwords
    text = y[:]
    y.clear()

    for i in text:
        if i not in stop_words and i not in string.punctuation:
            y.append(i)

    # Stemming
    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


# -------------------------
# Load Model & Vectorizer
# -------------------------

tfidf = pickle.load(open("vectorizer.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))


# -------------------------
# Streamlit UI
# -------------------------

st.title("📱 Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")


if st.button("Predict"):

    # 1. Preprocess
    transformed_sms = transform_text(input_sms)

    # 2. Vectorize
    vector_input = tfidf.transform([transformed_sms])

    # 3. Predict
    result = model.predict(vector_input)[0]

    # 4. Display
    if result == 1:
        st.error("Spam")
    else:
        st.success("Not Spam")