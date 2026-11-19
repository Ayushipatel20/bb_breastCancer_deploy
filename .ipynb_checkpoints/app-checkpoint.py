import streamlit as st
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>

/* Main application background */

.stApp {
    background: linear-gradient(
        135deg,
        #0F172A,
        #172554,
        #164E63
    );
    color: #F8FAFC;
}

/* Main heading */

.main-title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    color: #FFFFFF;
    padding: 15px;
}

/* Subtitle */

.subtitle {
    text-align: center;
    font-size: 17px;
    color: #BAE6FD;
    margin-bottom: 30px;
}

/* Input labels */

label {
    color: #F8FAFC !important;
    font-weight: 600 !important;
}

/* Input fields */

.stNumberInput input,
.stSelectbox div[data-baseweb="select"] {
    border-radius: 10px;
}

/* Buttons */

.stButton > button {
    background: linear-gradient(
        90deg,
        #0D9488,
        #06B6D4
    );

    color: white;
    border: none;
    border-radius: 12px;
    padding: 12px 25px;
    font-weight: bold;
    transition: all 0.3s ease;
}

/* Button hover animation */

.stButton > button:hover {
    transform: scale(1.04);
    box-shadow: 0 5px 20px rgba(6,182,212,0.4);
}

/* Cards */

.custom-card {
    background: rgba(255,255,255,0.10);
    border: 1px solid rgba(255,255,255,0.20);
    border-radius: 16px;
    padding: 25px;
    margin: 12px 0;
    transition: transform 0.3s ease;
}

.custom-card:hover {
    transform: translateY(-5px);
}

/* Result heading */

.result-title {
    color: #67E8F9;
    font-size: 25px;
    font-weight: bold;
}

/* Footer */

.footer {
    text-align: center;
    color: #CBD5E1;
    padding: 25px;
    font-size: 13px;
}

</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="main-title">❤️ Heart Disease Prediction System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-Powered Healthcare Prediction & Explanation Dashboard'
    '</div>',
    unsafe_allow_html=True
)

@st.cache_data
def load_data():
    return pd.read_csv("heart.csv")


df = load_data()

X = df.drop("target", axis=1)
y = df["target"]

@st.cache_resource
def train_model():
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X, y)

    return model


model = train_model()

st.markdown("## 🧑‍⚕️ Patient Information")

patient_data = {}

col1, col2 = st.columns(2)

for i, feature in enumerate(X.columns):

    column = col1 if i % 2 == 0 else col2

    with column:

        minimum = float(X[feature].min())
        maximum = float(X[feature].max())
        default = float(X[feature].median())

        patient_data[feature] = st.number_input(
            feature.replace("_", " ").title(),
            min_value=minimum,
            max_value=maximum,
            value=default,
            key=feature
        )


# Step 9: NLG Explanation Function

def generate_explanation(prediction, confidence):

    if confidence < 60:
        level = "Low"
        message = (
            "The model is uncertain about this prediction. "
            "Further clinical evaluation is important."
        )

    elif confidence < 80:
        level = "Moderate"
        message = (
            "The model shows moderate confidence. "
            "Some uncertainty remains in the prediction."
        )

    else:
        level = "High"
        message = (
            "The model shows a strong prediction probability "
            "based on patterns learned from the training data."
        )

    result = (
        "Higher likelihood of heart disease"
        if prediction == 1
        else "Lower likelihood of heart disease"
    )

    return result, level, message


# Step 10: Prediction Button

if st.button("🔍 Predict Heart Disease", use_container_width=True):

    input_df = pd.DataFrame(
        [patient_data],
        columns=X.columns
    )

    prediction = model.predict(input_df)[0]

    probabilities = model.predict_proba(input_df)[0]

    confidence = max(probabilities) * 100

    # Display prediction

    st.markdown("### ❤️ Prediction Result")

    if prediction == 1:
        st.error("Higher likelihood of heart disease")
    else:
        st.success("Lower likelihood of heart disease")

    st.metric(
        "Model Prediction Confidence",
        f"{confidence:.2f}%"
    )

    st.progress(confidence / 100)

    # Generate NLG explanation

    result, level, message = generate_explanation(
        prediction,
        confidence
    )

    # Display explanation

    st.markdown("### 🧠 AI Prediction Explanation")

    st.info(
        f"Result: {result}\n\n"
        f"Confidence Level: {level}\n\n"
        f"Explanation: {message}"
    )

    st.caption(
        "This is an educational prediction, "
        "not a medical diagnosis."
    )

st.markdown("""
<div class="footer">
    ❤️ Heart Disease Prediction System
    <br>
    Developed using Python, Machine Learning & Streamlit
    <br>
    For educational purposes only.
</div>
""", unsafe_allow_html=True)

