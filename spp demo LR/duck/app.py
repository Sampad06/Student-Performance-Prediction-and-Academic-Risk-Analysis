import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ============================================================
# Student Performance Prediction — Linear Regression Demo
# ============================================================

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="centered"
)

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "linear_regression_final.joblib"
)

st.title("All Student Performance Prediction")
st.caption("Linear Regression Demo — Final Examination Marks")

st.markdown(
    """
    Enter the student's academic and learning information below.
    The trained Linear Regression pipeline will predict the final
    examination marks.
    """
)

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "Model file not found. Place 'linear_regression_final.joblib' "
            "in the same folder as app.py."
        )
    return joblib.load(MODEL_PATH)


# ------------------------------------------------------------
# Load model
# ------------------------------------------------------------

try:
    model = load_model()
except Exception as exc:
    st.error(f"Could not load the trained model: {exc}")
    st.stop()


# ------------------------------------------------------------
# Student inputs
# ------------------------------------------------------------

st.subheader("Student Information")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input(
        "Age",
        min_value=17,
        max_value=25,
        value=20,
        step=1
    )

    gender = st.selectbox(
        "Gender",
        ["Male", "Female", "Other"]
    )

    attendance = st.slider(
        "Attendance (%)",
        min_value=0.0,
        max_value=100.0,
        value=80.0,
        step=0.5
    )

    study_hours = st.number_input(
        "Study Hours per Week",
        min_value=0.0,
        max_value=60.0,
        value=20.0,
        step=0.5
    )

with col2:
    previous_grade = st.number_input(
        "Previous Grade",
        min_value=0.0,
        max_value=100.0,
        value=65.0,
        step=0.5
    )

    assignment_marks = st.number_input(
        "Assignment Marks",
        min_value=0.0,
        max_value=100.0,
        value=65.0,
        step=0.5
    )

    internal_marks = st.number_input(
        "Internal Assessment Marks",
        min_value=0.0,
        max_value=100.0,
        value=65.0,
        step=0.5
    )

    online_classes = st.selectbox(
        "Online Classes",
        ["Yes", "No"]
    )

extracurricular = st.slider(
    "Number of Extracurricular Activities",
    min_value=0,
    max_value=5,
    value=1,
    step=1
)

parental_support = st.selectbox(
    "Parental Support",
    ["Low", "Medium", "High"]
)


# ------------------------------------------------------------
# Prediction
# ------------------------------------------------------------

if st.button("Predict Final Marks", type="primary", use_container_width=True):

    # Feature engineering must match the training notebook.
    attendance_study_interaction = (
        attendance * study_hours
    )

    input_data = pd.DataFrame({
        "Age": [age],
        "Gender": [gender],
        "AttendancePct": [attendance],
        "StudyHoursPerWeek": [study_hours],
        "PreviousGrade": [previous_grade],
        "AssignmentMarks": [assignment_marks],
        "InternalAssessmentMarks": [internal_marks],
        "OnlineClasses": [online_classes],
        "ExtracurricularActivities": [extracurricular],
        "ParentalSupport": [parental_support],
        "AttendanceStudyInteraction": [
            attendance_study_interaction
        ]
    })

    try:
        prediction = float(model.predict(input_data)[0])
        prediction = float(np.clip(prediction, 0, 100))

        st.divider()
        st.subheader("Prediction Result")

        result_col1, result_col2 = st.columns(2)

        with result_col1:
            st.metric(
                "Predicted Final Marks",
                f"{prediction:.2f} / 100"
            )

        with result_col2:
            if prediction >= 40:
                st.metric("Projected Result", "PASS")
            else:
                st.metric("Projected Result", "FAIL")

        # Rule-based interpretation for demo purposes.
        # This is NOT Logistic Regression risk prediction.
        if prediction < 40:
            risk = "High"
        elif prediction < 60:
            risk = "Medium"
        else:
            risk = "Low"

        st.info(
            f"Academic Risk (rule-based from predicted marks): **{risk}**"
        )

        with st.expander("View model input"):
            display_input = input_data.copy()
            display_input["AttendanceStudyInteraction"] = round(
                display_input["AttendanceStudyInteraction"], 2
            )
            st.dataframe(
                display_input,
                use_container_width=True,
                hide_index=True
            )

    except Exception as exc:
        st.error(f"Prediction failed: {exc}")


st.divider()

st.caption(
    "Note: This demo uses a synthetic student-performance dataset. "
    "The risk label shown here is a simple rule based on the predicted "
    "mark; the Logistic Regression model for Pass/Fail will be added later."
)
