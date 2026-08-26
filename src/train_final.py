import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load training data

X_train = pd.read_csv("data/X_train.csv")
y_train = pd.read_csv("data/y_train.csv")

# Prepare data

# Merge X and y using ticket id
data = X_train.merge(y_train, on="id")

texts = data["text"]
labels = data["category_truth"]

# Custom stop words

custom_stop_words = [
    "ticket",
    "id",
    "support",
    "received",
    "name",
    "company",
    "location",
    "address"
]

# Build final pipeline

pipeline = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            stop_words=custom_stop_words,
            ngram_range=(1, 2)
        )
    ),

    (
        "classifier",
        LogisticRegression(
            C=10,
            class_weight="balanced",
            max_iter=1000
        )
    )
])

# Train final model

pipeline.fit(texts, labels)

print("Final model trained successfully.")

# Save final model

joblib.dump(
    pipeline,
    "models/ticket_classifier.joblib"
)

print("Model saved to models/ticket_classifier.joblib")