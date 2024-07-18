from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
import pickle
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
import nltk

# Initialize the app
app = FastAPI()

# Load the pre-trained models and vectorizer
with open('Models/countVectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('Models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('Models/model_xgb.pkl', 'rb') as f:
    model_xgb = pickle.load(f)


# Define a request body model
class Review(BaseModel):
    review: str


# Preprocess the input review
def preprocess_review(review):
    from nltk.stem.porter import PorterStemmer
    import re
    from nltk.corpus import stopwords
    nltk.download('stopwords')
    STOPWORDS = set(stopwords.words('english'))

    stemmer = PorterStemmer()
    review = re.sub('[^a-zA-Z]', ' ', review)
    review = review.lower().split()
    review = [stemmer.stem(word) for word in review if not word in STOPWORDS]
    review = ' '.join(review)
    return review


# Define a route for the root URL
@app.get("/")
def read_root():
    return {"message": "Welcome to the Sentiment Analysis API"}


# Define a route for sentiment prediction
@app.post("/predict")
def predict_sentiment(review: Review):
    processed_review = preprocess_review(review.review)
    vectorized_review = vectorizer.transform([processed_review]).toarray()
    scaled_review = scaler.transform(vectorized_review)
    prediction = model_xgb.predict(scaled_review)
    return {"prediction": int(prediction[0])}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)