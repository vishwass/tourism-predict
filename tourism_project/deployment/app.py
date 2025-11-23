import streamlit as st
import pandas as pd
from huggingface_hub import  hf_hub_download
import joblib

repo_id = "Cruise949/tourism-predict"
repo_type = "model"
model_path = "model.joblib"
modeljoblib = hf_hub_download(repo_id=repo_id, filename=model_path, repo_type=repo_type)
model = joblib.load(modeljoblib)

st.title("Tourism Package Prediction")
st.write(" This app is a internal tool for 'Visit With Us' company employees to understand customer choices in terms of package purchase.")
st.write("Kindly enter customer details")

Age = st.number_input("Age", min_value=18, max_value=100, value=30)
TypeofContact = st.selectbox("Type of Contact", ["Company Invited", "Self Inquiry"])
CityTier = st.selectbox("City Tier", ["Tier 1", "Tier 2", "Tier 3"])
Occupation = st.selectbox("Occupation", ["Salaried", "Self Employed", "Business Owner", "Student", "Retired", "Housewife"])
Gender = st.selectbox("Gender", ["Male", "Female"])
NumberOfPersonVisiting = st.number_input("Number of People Visiting", min_value=1, max_value=10, value=2)
PreferredPropertyStar = st.selectbox("Preferred Property Star", [1, 2, 3, 4, 5])
MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
NumberOfTrips = st.number_input("Number of Trips", min_value=1, max_value=10, value=2)
Passport = st.selectbox("Passport", ["Yes", "No"])
ProductPitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "King","Super deluxe"])
OwnCar = st.selectbox("Own Car", ["Yes", "No"])
NumberOfChildrenVisiting = st.number_input("Number of Children Visiting", min_value=0, max_value=10, value=0)
Designation = st.selectbox("Designation", ["Executive", "Managerial", "Professional", "Other"])
MonthlyIncome = st.number_input("Monthly Income", min_value=0, max_value=1000000, value=50000)
DurationOfPitch = st.number_input("Duration of Pitch", min_value=1, max_value=100, value = 15)
NumberOfFollowups = st.number_input("Number of Followups", min_value=0, max_value=100, value = 1)
PitchSatisfactionScore = st.number_input("Pitch Satisfaction Score", min_value=1, max_value=5, value = 5)


data = pd.DataFrame({
    'Age': [Age],
    'TypeofContact': TypeofContact ,
    'CityTier': CityTier,
    'Occupation': Occupation,
    'Gender': Gender,
    'NumberOfPersonVisiting': NumberOfPersonVisiting,
    'PreferredPropertyStar': PreferredPropertyStar,
    'MaritalStatus': MaritalStatus,
    'NumberOfTrips': NumberOfTrips,
    'Passport': Passport,
    'ProductPitched': ProductPitched,
    'OwnCar': OwnCar,
    'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
    'Designation': Designation,
    'MonthlyIncome': MonthlyIncome,
    'DurationOfPitch': DurationOfPitch,
    'NumberOfFollowups': NumberOfFollowups,
    'PitchSatisfactionScore': PitchSatisfactionScore
})

if st.button("Predict"):
  prediction = model.predict(data)[0,1]
  pred =  "purchase" if (prediction == 1 ) else "not purchase"
  st.write("Based on the prediction, the customer will", pred,"the travel product")








