
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Load the trained machine learning model
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "Risk_model1.joblib"

st.set_page_config(
    page_title="Healthcare Outcome Prediction",
    page_icon="🏥",
    layout="centered"
)

st.title("🏥 Healthcare Outcome Prediction")

st.write(
    "Enter patient details to predict the possibility "
    "of an adverse healthcare outcome."
)

if not MODEL_PATH.exists():
    st.error("Risk_model1.joblib was not found in the project folder.")

else:
    try:
        model = joblib.load(MODEL_PATH)

        st.subheader("Patient Information")

        age = st.number_input(
            "Age (years)",
            min_value=0,
            max_value=120,
            value=45
        )

        length_of_stay = st.number_input(
            "Length of Stay (days)",
            min_value=0,
            value=5
        )

        treatment_cost = st.number_input(
            "Treatment Cost (₹)",
            min_value=0.0,
            value=5000.0,
            step=500.0
        )

        abnormal_lab_count = st.number_input(
            "Abnormal Lab Count",
            min_value=0,
            value=1
        )

        if st.button("Predict Outcome", use_container_width=True):

            input_data = pd.DataFrame(
                [[age, length_of_stay, treatment_cost, abnormal_lab_count]],
                columns=[
                    "Age",
                    "LengthOfStay",
                    "TreatmentCost",
                    "AbnormalLabCount"
                ]
            )

            prediction = model.predict(input_data)[0]

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(input_data)[0]
                classes = list(model.classes_)

                if 1 in classes:
                    adverse_index = classes.index(1)
                    probability = float(probabilities[adverse_index])

                    st.metric(
                        "Estimated adverse-outcome probability",
                        f"{probability:.1%}"
                    )
                else:
                    probability = None

            else:
                probability = None

            if int(prediction) == 1:
                st.error("⚠️ Model Prediction: ADVERSE OUTCOME")
            else:
                st.success("✅ Model Prediction: RECOVERY")

            if probability is not None:
                st.progress(probability)

            st.caption(
                "This is an educational prediction based on the project's "
                "training data. It is not a clinical diagnosis or a "
                "validated medical risk assessment."
            )

    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        st.info(
            "Check that Risk_model1.joblib is available and compatible "
            "with the four input features."
        )
