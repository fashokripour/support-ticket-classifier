import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# Load final model

model = joblib.load("models/ticket_classifier.joblib")

# Load test data

X_test = pd.read_csv("data/X_test.csv")
y_test = pd.read_csv("data/y_test.csv")

# Merge using id
test_data = X_test.merge(y_test, on="id")

test_texts = test_data["text"]
test_labels = test_data["category_truth"]

# Predict

predictions = model.predict(test_texts)

# Metrics

accuracy = accuracy_score(
    test_labels,
    predictions
)

macro_f1 = f1_score(
    test_labels,
    predictions,
    average="macro"
)

weighted_f1 = f1_score(
    test_labels,
    predictions,
    average="weighted"
)


print("\nFinal Test Results")
print("-" * 50)

print(f"Accuracy:    {accuracy:.4f}")
print(f"Macro F1:    {macro_f1:.4f}")
print(f"Weighted F1: {weighted_f1:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        test_labels,
        predictions
    )
)


# Confusion Matrix

labels_order = model.classes_

cm = confusion_matrix(
    test_labels,
    predictions,
    labels=labels_order
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=labels_order
)

disp.plot(
    xticks_rotation=45
)

plt.tight_layout()
plt.show()