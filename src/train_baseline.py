import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfVectorizer,
    ENGLISH_STOP_WORDS
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from sklearn.linear_model import LogisticRegression

# Load original training data

X_data = pd.read_csv("./data/X_train.csv")
y_data = pd.read_csv("./data/y_train.csv")

texts = X_data["text"]
labels = y_data["category_truth"]


print("Total training samples:", len(texts))


# Create train / validation split

X_train, X_val, y_train, y_val = train_test_split(
    texts,
    labels,
    test_size=0.20,
    random_state=42,
    stratify=labels
)

print("Train samples:", len(X_train))
print("Validation samples:", len(X_val))

# Stop words

custom_stop_words = list(ENGLISH_STOP_WORDS) + [
    "ticket",
    "id",
    "support",
    "received",
    "name",
    "location",
    "company",
    "address",
]

# Vectorizer

vectorizer = TfidfVectorizer(
    stop_words=custom_stop_words,
    ngram_range=(1, 2)
)


X_train_vec = vectorizer.fit_transform(X_train)

X_val_vec = vectorizer.transform(X_val)

print("Number of features:", len(vectorizer.get_feature_names_out()))

# Train model

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    C = 10
)

model.fit(X_train_vec, y_train)

# Validation prediction

val_predictions = model.predict(X_val_vec)

# Validation metrics

accuracy = accuracy_score(y_val, val_predictions)

macro_f1 = f1_score(
    y_val,
    val_predictions,
    average="macro"
)

weighted_f1 = f1_score(
    y_val,
    val_predictions,
    average="weighted"
)

print("\nValidation Results")
print("-" * 50)

print(f"Accuracy:     {accuracy:.4f}")
print(f"Macro F1:     {macro_f1:.4f}")
print(f"Weighted F1:  {weighted_f1:.4f}")

print("\nClassification Report:")
print(classification_report(y_val, val_predictions))


# Confusion Matrix

labels_order = model.classes_

cm = confusion_matrix(
    y_val,
    val_predictions,
    labels=labels_order
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=labels_order
)

disp.plot(xticks_rotation=45)

plt.tight_layout()
plt.show()
    
# errors = pd.DataFrame({
#     "text": X_val,
#     "actual": y_val,
#     "predicted": val_predictions
# })

# o365_errors = errors[
#     (errors["actual"] == "O365") &
#     (errors["predicted"] == "Support general")
# ]

# for i, text in enumerate(o365_errors["text"], start=1):
#     print(f"\n{i}. {' '.join(str(text).split())}")