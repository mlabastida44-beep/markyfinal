# app.py
import os
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Student Predict", page_icon="🎓", layout="wide")

if not (os.path.exists("student_model.pkl") and os.path.exists("student_data.csv")):
    st.error("student_model.pkl or student_data.csv not found. "
             "Run `python train_model.py` first, in this same folder.")
    st.stop()


@st.cache_resource
def load_model():
    return joblib.load("student_model.pkl")


model = load_model()
data = pd.read_csv("student_data.csv")
required_columns = [c for c in data.columns if c != "G3"]

st.title("🎓 Student Performance Prediction")
st.caption("Classroom prototype for academic risk exploration")

c1, c2, c3 = st.columns(3)
c1.metric("Records", len(data))
c2.metric("Average final grade", round(float(data["G3"].mean()), 2))
c3.metric("Historical at-risk records", int((data["G3"] < 10).sum()))

st.subheader("Dataset preview")
st.dataframe(data.head(10))

st.subheader("Predict uploaded student records")
st.write("Upload a CSV containing the same predictor columns as the training data.")

template = data[required_columns].head(5)
st.download_button(
    "Download sample input CSV",
    template.to_csv(index=False),
    "sample_prediction_input.csv",
    "text/csv",
)

uploaded = st.file_uploader("Choose student CSV", type=["csv"])

if uploaded is not None:
    try:
        new_data = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Could not read the CSV file: {e}")
        st.stop()

    missing = [c for c in required_columns if c not in new_data.columns]
    if missing:
        st.error("Missing required columns: " + ", ".join(missing))
    else:
        try:
            predictions = model.predict(new_data[required_columns])
        except Exception as e:
            st.error(f"Prediction failed. Check the values in your CSV: {e}")
            st.stop()

        results = new_data.copy()
        results["Prediction"] = ["AT RISK" if v == 1 else "NOT AT RISK"
                                 for v in predictions]

        st.subheader("Prediction results")
        st.dataframe(results)
        st.metric("Predicted at-risk records", int((predictions == 1).sum()))
        st.download_button(
            "Download prediction results",
            results.to_csv(index=False),
            "student_predictions.csv",
            "text/csv",
        )

st.caption("Responsible use: predictions are not diagnoses and must not be the "
           "sole basis for real student decisions.")
