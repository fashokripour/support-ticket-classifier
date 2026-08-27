import joblib
import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Support Ticket Classifier",
    page_icon="🎫",
    layout="centered"
)


@st.cache_resource
def load_model():
    return joblib.load("models/ticket_classifier.joblib")


model = load_model()


st.title("Support Ticket Classifier")

st.write(
    "Enter an IT support ticket below. "
    "The model will predict the most appropriate support category "
    "and show the probability assigned to each category."
)


ticket_text = st.text_area(
    "Support ticket",
    placeholder="Example: I cannot access my shared folder...",
    height=150
)


if st.button("Classify Ticket"):

    if not ticket_text.strip():

        st.warning("Please enter a support ticket.")

    else:

        # Final predicted class
        prediction = model.predict([ticket_text])[0]

        # Probability of all classes
        probabilities = model.predict_proba([ticket_text])[0]

        # Class names
        classes = model.classes_

        # Highest probability
        confidence = np.max(probabilities)


        st.success(
            f"Predicted category: **{prediction}**"
        )

        st.metric(
            label="Model confidence",
            value=f"{confidence:.2%}"
        )

        if confidence < 0.50:
            st.warning(
                "Low-confidence prediction. "
                "This ticket may require manual review."
            )

        # Build a table containing every class probability
        probability_df = pd.DataFrame({
            "Category": classes,
            "Probability": probabilities
        })

        # Sort from highest probability to lowest
        probability_df = probability_df.sort_values(
            by="Probability",
            ascending=False
        )

        # Convert probability from 0-1 to percentage
        probability_df["Probability"] = (
            probability_df["Probability"] * 100
        )


        st.subheader("Category probabilities")

        st.dataframe(
            probability_df,
            hide_index=True,
            use_container_width=True
        )