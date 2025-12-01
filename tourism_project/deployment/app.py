import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the trained model
model_path = hf_hub_download(repo_id="jackfroooot/tourism-package-prediction-proj", filename="best_tourism_pkg_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI
st.title("Tourism Package Purchase Prediction")
st.write("""
This application predicts whether a customer **purchases** the newly introduced Wellness Tourism Package before contacting them.  It is based on characteristics such as customer City, Age, Income, Trips etc. Please enter the customer details below to get a purchase prediction.
""")

# User input
age		         = 		st.number_input("age", min_value=5, max_value=90, step=1, value=30)
type_of_contact	 = 		st.selectbox("Type of Contact", ["Company Invited", "Self Inquiry"])
city_tier		 = 		st.selectbox("City Category", ["Tier_1", "Tier_2", "Tier_3"]) 
occupation		 = 		st.selectbox("Customer Occupation", ["Salaried", "Freelancer"]) 
gender		         = 		st.selectbox("Customer Gender", ["Male", "Female"]) 
persons_visiting	 = 		st.number_input("Number Of Person Visiting", min_value=0, max_value=50, value=2, step=1)
pref_prop_star		 = 		st.number_input("Preferred Hotel Rating", min_value=1, max_value=5, value=4, step=1) 
marital		         = 		st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
num_of_trips		 = 		st.number_input("Avg Annual Trips", min_value=1, max_value=25, value=4, step=1) 
passport		     = 		st.selectbox("Holds Passport", ["Yes", "No"])
own_car	        	 = 		st.selectbox("Owns Car", ["Yes", "No"])
num_of_child_visit   = 		st.number_input("No. of Children above 5 accompanying", min_value=0, max_value=20, value=4, step=1) 
designation		     = 		st.selectbox("Designation", ["AVP", "Executive", "Manager", "Senior Manager", "VP"])
monthly_income		 = 		st.number_input("Gross monthly income", min_value=1, max_value=100000, value=3000, step=1) 
pitch_score		     = 		st.number_input("Satisfaction with the sales pitch", min_value=1, max_value=5, value=3, step=1) 
prod_pitched		 = 		st.selectbox("Type of Product Pitched", ["Basic", "Deluxe", "King", "Standard", "Super Deluxe"])
num_of_followups	 = 		st.number_input("No. of follow-ups after pitch", min_value=1, max_value=15, value=3, step=1) 
duration_of_pitch	 = 		st.number_input("Duration of sales pitch", min_value=1, max_value=100, value=30, step=1) 

# Assemble input into DataFrame
input_data = pd.DataFrame([{
  'Age'		         :     age,
  'TypeofContact'	 :     type_of_contact,
  'CityTier'		 :     (1 if city_tier=='Tier_1' else 2 if city_tier=='Tier_2' else 3),
  'Occupation'		 :     occupation,
  'Gender'		     :     gender,
  'NumberOfPersonVisiting' :     persons_visiting,
  'PreferredPropertyStar'	 :     pref_prop_star,
  'MaritalStatus'		     :     marital,
  'NumberOfTrips'	 :     num_of_trips,
  'Passport'		 :     (1 if passport=='Yes' else 0),
  'OwnCar'	         :     (1 if own_car=='Yes' else 0),
  'NumberOfChildrenVisiting'   :     num_of_child_visit,
  'Designation'		 :     designation,
  'MonthlyIncome'	 :     monthly_income,
  'PitchSatisfactionScore'		 :     pitch_score,
  'ProductPitched'	 :     prod_pitched,
  'NumberOfFollowups'	 :     num_of_followups,
  'DurationOfPitch'	 :     duration_of_pitch
}])

# Predict button
if st.button("Predict Purchase"):
    prediction = model.predict(input_data)
    st.subheader("Prediction Result:")
    st.success(f"Prediction (score={prediction}) : {'**Will Purchase**' if prediction[0]==1 else '**Will Not Purchase**'}")
