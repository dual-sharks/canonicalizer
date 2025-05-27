# adapters/sentiment_model.py
from transformers import pipeline

class SentimentModel:
    def __init__(self, model="distilbert-base-uncased-finetuned-sst-2-english"):
        self.model = pipeline("sentiment-analysis", model=model)

    def predict(self, text: str) -> str:
        return self.model(text[:512])[0]['label'].lower()
