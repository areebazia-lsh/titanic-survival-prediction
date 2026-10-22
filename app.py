
import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import svm

st.set_page_config(page_title="Titanic Survival Prediction", page_icon="🚢")

st.title("Titanic Passenger Survival Prediction")
st.write("Enter passenger details to predict the model's result.")

@st.cache_resource
def train_model():
    data = pd.read_csv("sample-data.csv")
    features = ["PClass", "Gender", "Sibling", "Embarked"]
    encoders = {}

    for col in features:
        encoders[col] = LabelEncoder()
        data[col] = encoders[col].fit_transform(data[col].astype(str))

    target_encoder = LabelEncoder()
    y = target_encoder.fit_transform(data["Survived"].astype(str))

    model = svm.SVC()
    model.fit(data[features], y)

    return model, encoders, target_encoder, features

try:
    model, encoders, target_encoder, features = train_model()

    pclass = st.selectbox("Passenger Class", ["First", "Second", "Third"])
    gender = st.selectbox("Gender", ["Female", "Male"])
    sibling = st.selectbox("Number of Siblings", ["Zero", "One", "Two", "Three"])
    embarked = st.selectbox(
        "Port of Embarkation",
        ["Southampton", "Cherbourg", "Queenstown"]
    )

    if st.button("Predict Survival"):
        values = {
            "PClass": pclass,
            "Gender": gender,
            "Sibling": sibling,
            "Embarked": embarked
        }

        row = pd.DataFrame([values])

        for col in features:
            row[col] = encoders[col].transform(row[col].astype(str))

        prediction = model.predict(row[features])[0]
        result = target_encoder.inverse_transform([prediction])[0]

        if result == "Yes":
            st.success("Prediction: SURVIVED")
        else:
            st.error("Prediction: NOT SURVIVED")

    st.caption(
        "Educational demonstration using a small sample dataset. "
        "Predictions are not historically reliable."
    )

except Exception as e:
    st.error(f"Could not load or train the model: {e}")
    st.info("Check that sample-data.csv is uploaded with the correct column names.")
