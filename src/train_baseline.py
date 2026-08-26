import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


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
vectorizer = CountVectorizer()

X_train_bow = vectorizer.fit_transform(train_texts)
X_test_bow = vectorizer.transform(test_texts)


print("Vocabulary size:", len(vectorizer.vocabulary_))
print("Train matrix shape:", X_train_bow.shape)
print("Test matrix shape:", X_test_bow.shape)


# Train model
model = MultinomialNB()
model.fit(X_train_bow, train_labels)


# Predict
predictions = model.predict(X_test_bow)

# Evaluation
accuracy = accuracy_score(test_labels, predictions)

print("\nAccuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(test_labels, predictions))

# labels = model.classes_

# cm = confusion_matrix(
#     test_labels,
#     predictions,
#     labels=labels
# )

# print("\nConfusion Matrix:")
# print(cm)

# disp = ConfusionMatrixDisplay(
#     confusion_matrix=cm,
#     display_labels=labels
# )

# disp.plot(xticks_rotation=45)
# plt.tight_layout()
# plt.show()

errors = pd.DataFrame({
    "text": test_texts,
    "actual": test_labels,
    "predicted": predictions
})

cs_as_o365 = errors[
    (errors["actual"] == "Computer-Services") &
    (errors["predicted"] == "O365")
]

import re

def clean_text(text):
    text = re.sub(r"\s+", " ", str(text))
    return text.strip()

cs_as_o365 = cs_as_o365.copy()
cs_as_o365["clean_text"] = cs_as_o365["text"].apply(clean_text)

for i, text in enumerate(cs_as_o365["text"].head(10), start=1):
    text = re.sub(r"\s+", " ", text).strip()
    print(f"\n{i}. {text}")

# print(ad_as_general[["text"]].to_string())