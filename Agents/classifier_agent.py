89% of storage used … If you run out of space, you can't save to Drive or use Gmail. Get 30 GB of storage for MYR 3.50 MYR 0.90/month for 3 months.
import re 
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import nltk

stop_words = set(stopwords.words("english"))
stemmer = SnowballStemmer("english")

import joblib
# Random forest model 
rf = joblib.load("trained_models/random_forest_model.pkl")
vectorizer = joblib.load("trained_models/vectorizer.pkl")
nzv = joblib.load("trained_models/variance_threshold.pkl")

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z]", " ", text)  # remove numbers & symbols
    tokens = text.split()
    tokens = [w for w in tokens if w not in stop_words]
    tokens = [stemmer.stem(w) for w in tokens]
    return " ".join(tokens)

def ticket_classification(ticket_description):
    """Classify a customer support ticket using OpenAI's API"""
    cleaned_ticket = clean_text(ticket_description)
    matrix = vectorizer.transform([cleaned_ticket])      # Bag-of-words features
    cleaned_matrix = nzv.transform(matrix) 
    predicted_department = rf.predict(cleaned_matrix)
    return predicted_department[0]
