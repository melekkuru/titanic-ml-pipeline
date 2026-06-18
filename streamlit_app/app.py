import streamlit as st
import requests

API_URL = "http://localhost:8000/predict"

st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢")

st.title("🚢 Titanic Survival Predictor")
st.write("Enter passenger details to predict survival probability.")

with st.form("prediction_form"):
    pclass = st.selectbox("Passenger Class", [1, 2, 3], index=2)
    sex = st.selectbox("Sex", ["male", "female"])
    age = st.slider("Age", 0, 100, 22)
    sibsp = st.number_input("Siblings/Spouses Aboard (SibSp)", min_value=0, value=0)
    parch = st.number_input("Parents/Children Aboard (Parch)", min_value=0, value=0)
    fare = st.number_input("Fare", min_value=0.0, value=7.25)
    embarked = st.selectbox("Port of Embarkation", ["S", "C", "Q"])

    submitted = st.form_submit_button("Predict")

if submitted:
    payload = {
        "pclass": pclass,
        "sex": sex,
        "age": age,
        "sibsp": sibsp,
        "parch": parch,
        "fare": fare,
        "embarked": embarked
    }

    try:
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()
        result = response.json()

        survived = result["survived"]
        probability = result["survival_probability"]

        if survived == 1:
            st.success(f"✅ Prediction: SURVIVED ({probability:.2%} probability)")
        else:
            st.error(f"❌ Prediction: DID NOT SURVIVE ({probability:.2%} probability)")

    except requests.exceptions.ConnectionError:
        st.warning("⚠️ Cannot connect to the API. Make sure the Docker container is running: `docker-compose up -d`")
    except Exception as e:
        st.error(f"An error occurred: {e}")