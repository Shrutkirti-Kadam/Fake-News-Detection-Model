import pandas as pd
import numpy as np
import re
import string
import nltk
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout, LSTM, Bidirectional
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
from tensorflow.keras.callbacks import EarlyStopping
from transformers import pipeline

# Download NLTK resources and SpaCy model
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nlp = spacy.load('en_core_web_sm')

# Preprocess text: lowercase, remove digits/punctuations, lemmatize
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stopwords.words('english')]
    return " ".join(tokens)

# Convert active voice to passive voice (basic transformation)
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

# Load and label fake/true news data
fake_df = pd.read_csv("/content/fake_news_expanded.csv")
true_df = pd.read_csv("/content/true_news_expanded.csv")
fake_df["label"] = 0
true_df["label"] = 1
news_df = pd.concat([fake_df, true_df], ignore_index=True)
news_df["text"] = news_df["text"].astype(str).apply(preprocess_text)

# Tokenize and pad sequences
max_words, max_len = 5000, 100
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(news_df["text"])
sequences = tokenizer.texts_to_sequences(news_df["text"])
data = pad_sequences(sequences, maxlen=max_len)
labels = news_df["label"].values

# Train-test split and handle imbalance using SMOTE
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
smote = SMOTE()
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# CNN + LSTM model for fake news classification
model_fake = Sequential([
    Embedding(max_words, 100, input_length=max_len),
    Conv1D(256, 5, activation='relu'),
    Dropout(0.3),
    Conv1D(256, 5, activation='relu'),
    Dropout(0.3),
    Bidirectional(LSTM(128, return_sequences=True)),
    GlobalMaxPooling1D(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])
model_fake.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train model with early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
print("Training Fake News Model...")
model_fake.fit(X_train_resampled, y_train_resampled, batch_size=32, epochs=20,
               validation_data=(X_test, y_test), callbacks=[early_stop])

# Evaluate the model
y_pred_fake = (model_fake.predict(X_test) > 0.6).astype(int)
print("Accuracy:", accuracy_score(y_test, y_pred_fake))
print("Precision:", precision_score(y_test, y_pred_fake))
print("Recall:", recall_score(y_test, y_pred_fake))
print("F1 Score:", f1_score(y_test, y_pred_fake))

# Predict fake/true news label
def predict_fake_news(text):
    text = preprocess_text(text)
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_len)
    pred = model_fake.predict(padded)
    return "True News" if pred > 0.6 else "Fake News"

# Sentiment analysis using HuggingFace DistilBERT
sentiment_model = pipeline("sentiment-analysis")
def predict_sentiment(text):
    result = sentiment_model(text)[0]
    return result["label"]

# Accept user input and run predictions
user_text = input("\nEnter news article for classification: ")
passive_text = convert_to_passive(user_text)
print("\n--- Predictions ---")
print("Fake News Detection:", predict_fake_news(passive_text))
print("Sentiment Analysis:", predict_sentiment(passive_text))
