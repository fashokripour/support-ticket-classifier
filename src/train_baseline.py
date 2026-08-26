import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


# Load data
X_train = pd.read_csv("./data/X_train.csv")
X_test = pd.read_csv("./data/X_test.csv")

y_train = pd.read_csv("./data/y_train.csv")
y_test = pd.read_csv("./data/y_test.csv")


# Extract text and labels
train_texts = X_train["text"]
test_texts = X_test["text"]

train_labels = y_train["category_truth"]
test_labels = y_test["category_truth"]

# Convert text to Bag of Words

custom_stop_words = list(ENGLISH_STOP_WORDS) + [
    "ticket",
    "id",
    "support",
    "received",
    "name",
    "location",
    "company",
    "address"
]

vectorizer = CountVectorizer(
    stop_words=custom_stop_words,
    ngram_range=(1, 2)
)

X_train_bow = vectorizer.fit_transform(train_texts)
X_test_bow = vectorizer.transform(test_texts)


print("Vocabulary size:", len(vectorizer.vocabulary_))
print("Train matrix shape:", X_train_bow.shape)
print("Test matrix shape:", X_test_bow.shape)


# Train model
model = MultinomialNB()
model.fit(X_train_bow, train_labels)

import numpy as np

feature_names = vectorizer.get_feature_names_out()

for class_index, class_name in enumerate(model.classes_):
    top_indices = np.argsort(
        model.feature_log_prob_[class_index]
    )[-15:][::-1]

    top_words = feature_names[top_indices]

    print(f"\n{class_name}:")
    print(", ".join(top_words))

# Predict
predictions = model.predict(X_test_bow)

# Evaluation
accuracy = accuracy_score(test_labels, predictions)

print("\nAccuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(test_labels, predictions))
