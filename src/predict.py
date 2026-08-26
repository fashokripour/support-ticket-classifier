import joblib
import numpy as np


model = joblib.load("models/ticket_classifier.joblib")


samples = [
    "I cannot access the shared folder",
    "Outlook email is not working",
    "Please install printer driver",
    "Reset password for new user account"
]


predictions = model.predict(samples)
probabilities = model.predict_proba(samples)


for text, prediction, probs in zip(samples, predictions, probabilities):

    confidence = np.max(probs)

    print("-" * 70)
    print("Ticket:", text)
    print("Predicted category:", prediction)
    print(f"Confidence: {confidence:.2%}")