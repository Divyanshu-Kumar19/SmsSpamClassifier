# SMS Spam Classifier

A small Streamlit application that classifies SMS messages as spam or not spam using a trained machine learning model.

## Project Structure

```text
SmsSpamClassifier/
|-- app.py
|-- model.pkl
|-- vectorizer.pkl
|-- requirements.txt
|-- notebooks/
|   `-- spamDetection.ipynb
`-- README.md
```

## Requirements

- Python 3.10 or newer
- pip

## Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the dependencies:

```powershell
python -m pip install -r requirements.txt
```

## Run the Application

```powershell
streamlit run app.py
```

Open the local URL shown by Streamlit in your browser.

## How It Works

1. The input message is converted to lowercase.
2. Tokens are cleaned and stemmed using NLTK.
3. The saved TF-IDF vectorizer converts the text into features.
4. The saved model predicts whether the message is spam.

The application downloads the required NLTK resources on startup.

## Deployment

The project can be deployed using Streamlit Community Cloud.

- Repository: `Divyanshu-Kumar19/SmsSpamClassifier`
- Branch: `main`
- Main file: `app.py`

## Training Notebook

The model training and experimentation notebook is available at `notebooks/spamDetection.ipynb`.
