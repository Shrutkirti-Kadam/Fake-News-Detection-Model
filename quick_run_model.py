import json
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import spacy

# ---------------------
# Load NLP Resources
# ---------------------
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nlp = spacy.load("en_core_web_sm")

# ---------------------
# Text Preprocessing
# ---------------------
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"\d+", "", text)
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stopwords.words("english")]
    return " ".join(tokens)

def convert_to_passive(text):
    doc = nlp(text)
    passive_sentences = []
    for sent in doc.sents:
        subject, verb, obj = None, None, None
        for token in sent:
            if "subj" in token.dep_:
                subject = token
            if "obj" in token.dep_:
                obj = token
            if token.pos_ == "VERB":
                verb = token
        if subject and verb and obj:
            passive_sentence = f"{obj.text} was {verb.lemma_}ed by {subject.text}."
            passive_sentences.append(passive_sentence)
        else:
            passive_sentences.append(sent.text)
    return " ".join(passive_sentences)

# ---------------------
# Load Tokenizer
# ---------------------
with open("tokenizer.json") as f:
    tokenizer_data = json.load(f)
tokenizer = tokenizer_from_json(tokenizer_data)

# ---------------------
# Load Fake News Detection Model
# ---------------------
model_fake = load_model("fake_news_model.h5")

# ---------------------
# Load Sentiment Model and Tokenizer
# ---------------------
sentiment_tokenizer = AutoTokenizer.from_pretrained("./distilbert_sentiment")
sentiment_model = AutoModelForSequenceClassification.from_pretrained("./distilbert_sentiment")
sentiment_pipeline = pipeline("sentiment-analysis", model=sentiment_model, tokenizer=sentiment_tokenizer)

# ---------------------
# Prediction Functions
# ---------------------
def predict_fake_news(text, max_len=100):
    text = preprocess_text(text)
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_len)
    pred = model_fake.predict(padded)
    return "True News" if pred[0][0] > 0.6 else "Fake News"

def predict_sentiment(text):
    result = sentiment_pipeline(text)[0]
    return result["label"]

# ---------------------
# User Input and Prediction
# ---------------------
user_input = input("Enter news article for classification:\n")
passive_input = convert_to_passive(user_input)

print("\n--- Results ---")
print("Fake News Detection:", predict_fake_news(passive_input))
print("Sentiment Analysis:", predict_sentiment(passive_input))
