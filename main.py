import streamlit as st
import pandas as pd
import joblib


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Product Taken Prediction",
    layout="centered"
)

st.title("🧳 Product Taken Prediction App")
st.write(
    "Enter customer details to predict whether the product will be taken or not."
)

st.markdown("---")


# ============================================================
# LOAD MODEL AND PREPROCESSOR
# ============================================================

@st.cache_resource
def load_files():

    loaded_model = joblib.load("model.joblib")
    loaded_preprocessor = joblib.load("preprocessor.joblib")

    # If model.joblib contains a dictionary
    if isinstance(loaded_model, dict):

        # Most likely case:
        # {"XGB": trained_xgb_model}
        if "XGB" in loaded_model:
            loaded_model = loaded_model["XGB"]

        # Other common names
        elif "model" in loaded_model:
            loaded_model = loaded_model["model"]

        elif "best_model" in loaded_model:
            loaded_model = loaded_model["best_model"]

        else:
            # Try to find an object that has predict()
            found_model = None

            for value in loaded_model.values():
                if hasattr(value, "predict"):
                    found_model = value
                    break

            if found_model is None:
                raise ValueError(
                    "model.joblib contains a dictionary, but no model "
                    "with a predict() method was found."
                )

            loaded_model = found_model

    # If the loaded object is a tuple/list and the first
    # element is the actual trained model
    if isinstance(loaded_model, (tuple, list)):

        for value in loaded_model:
            if hasattr(value, "predict"):
                loaded_model = value
                break

    # Final check
    if not hasattr(loaded_model, "predict"):
        raise ValueError(
            f"The loaded model does not have a predict() method. "
            f"Loaded type: {type(loaded_model)}"
        )

    return loaded_model, loaded_preprocessor


try:
    model, preprocessor = load_files()

except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()


# ============================================================
# INPUT FIELDS
# ============================================================

Age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=30
)

TypeofContact = st.selectbox(
    "Type of Contact",
    ["Self Enquiry", "Company Invited"]
)

CityTier = st.selectbox(
    "City Tier",
    [1, 2, 3]
)

DurationOfPitch = st.number_input(
    "Duration Of Pitch (minutes)",
    min_value=0,
    max_value=300,
    value=30
)

Occupation = st.selectbox(
    "Occupation",
    [
        "Salaried",
        "Small Business",
        "Large Business",
        "Free Lancer"
    ]
)

Gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

NumberOfFollowups = st.number_input(
    "Number Of Followups",
    min_value=0,
    max_value=20,
    value=2
)

ProductPitched = st.selectbox(
    "Product Pitched",
    [
        "Basic",
        "Standard",
        "Deluxe",
        "Super Deluxe",
        "King"
    ]
)

PreferredPropertyStar = st.selectbox(
    "Preferred Property Star",
    [3, 4, 5]
)

MaritalStatus = st.selectbox(
    "Marital Status",
    [
        "Unmarried",
        "Married",
        "Divorced"
    ]
)

NumberOfTrips = st.number_input(
    "Number Of Trips",
    min_value=0,
    max_value=50,
    value=2
)

Passport = st.selectbox(
    "Passport",
    [0, 1]
)

PitchSatisfactionScore = st.slider(
    "Pitch Satisfaction Score",
    min_value=1,
    max_value=5,
    value=3
)

OwnCar = st.selectbox(
    "Own Car",
    [0, 1]
)

Designation = st.selectbox(
    "Designation",
    [
        "Executive",
        "Manager",
        "Senior Manager",
        "AVP",
        "VP"
    ]
)

MonthlyIncome = st.number_input(
    "Monthly Income",
    min_value=0,
    value=50000
)

TotalVisiting = st.number_input(
    "Total Visiting",
    min_value=0,
    max_value=50,
    value=2
)


# ============================================================
# PREDICTION
# ============================================================

if st.button("🔍 Predict"):

    # Create input DataFrame
    input_data = pd.DataFrame({
        "Age": [Age],
        "TypeofContact": [TypeofContact],
        "CityTier": [CityTier],
        "DurationOfPitch": [DurationOfPitch],
        "Occupation": [Occupation],
        "Gender": [Gender],
        "NumberOfFollowups": [NumberOfFollowups],
        "ProductPitched": [ProductPitched],
        "PreferredPropertyStar": [PreferredPropertyStar],
        "MaritalStatus": [MaritalStatus],
        "NumberOfTrips": [NumberOfTrips],
        "Passport": [Passport],
        "PitchSatisfactionScore": [PitchSatisfactionScore],
        "OwnCar": [OwnCar],
        "Designation": [Designation],
        "MonthlyIncome": [MonthlyIncome],
        "TotalVisiting": [TotalVisiting]
    })

    try:

        # ----------------------------------------------------
        # Apply preprocessing
        # ----------------------------------------------------

        X_processed = preprocessor.transform(input_data)

        # ----------------------------------------------------
        # Make prediction
        # ----------------------------------------------------

        prediction = model.predict(X_processed)[0]

        # ----------------------------------------------------
        # Probability
        # ----------------------------------------------------

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(X_processed)[0]

            # Binary classification
            if len(probabilities) == 2:
                prob = probabilities[1]

                st.info(
                    f"📊 Probability of Product Taken: "
                    f"**{prob:.2%}**"
                )

        # ----------------------------------------------------
        # Display result
        # ----------------------------------------------------

        if prediction == 1:
            st.success(
                "✅ Product **WILL BE TAKEN**"
            )

        elif prediction == 0:
            st.error(
                "❌ Product **WILL NOT BE TAKEN**"
            )

        else:
            # In case your model returns strings
            st.success(
                f"🎯 Prediction: **{prediction}**"
            )

    except Exception as e:

        st.error(
            f"❌ Prediction error: {e}"
        )

        st.write("Input data:")
        st.dataframe(input_data)


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "ML Model Prediction using Streamlit"
)