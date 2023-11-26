# Import libraries
import streamlit as st
import time
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score



# Build the sidebar of the web app which will help us navigate through the different sections of the entire application
rad=st.sidebar.radio("Navigation Menu",["Diabetes", "Data"])

#----------
# Home Page 
#----------

# Displays all the available disease prediction options in the web app
if rad=="Home":
    st.title("Diabetes Predictor")
    # st.image("Medical Prediction Home Page.jpg")
    st.text("Diabetes Prediction based on the parameters below")
  
#--------------------
# Diabetes Prediction
#--------------------

# Loading the Diabetes dataset
df2 = pd.read_csv('Resources/diabetes_prediction_modified.csv')

st.markdown("Please wait while this page loads. This may take a few minutes.")

progress_bar = st.progress(0)
for perc_completed in range(100):
    time.sleep(1.2)
    progress_bar.progress(perc_completed+1)
#cleaning the data by dropping the target column and dividing the data as features(x2) & target(y2)
# Separate the data into labels and features

# Separate the y variable, the labels
y2 = df2['diabetes']
# Separate the X variable, the features
x2 = df2.drop(columns=['diabetes'])

# Perform train-test split on the data
x2_train,x2_test,y2_train,y2_test=train_test_split(x2,y2,test_size=0.2,random_state=0)

# Create the model
model2=RandomForestClassifier(n_estimators=800, random_state=42)
# Fit the model with train data (x2_train & y2_train)
model2.fit(x2_train,y2_train)

#-------------
#Diabetes Page
#-------------

# Set up the Diabetes Predictor page
if rad=="Diabetes":
    st.title("Diabetes Predictor")
    st.header("Check If You May Be Susceptible to Developing Diabetes")
    st.header("Check If You May Be Susceptible to Developing Diabetes")
    st.write("All The Values Should Be In Range Mentioned")
    # Set up features as input -> Blood Glucose, HbA1c, Body Mass Index, Age, Smoking History, Heart Disease, Hypertension and Gender.
    # Set a minimum value & maximum value range and step=1 for the user to enter a value.
    # If the user enters a value which is not in the range an alert message will pop up.
    blood_glucose_level=st.number_input("Enter Your Blood Glucose Level (0-300)",0,300,1)
    HbA1c_level=st.number_input("Enter Your HbA1c Level (0-9)",0,9,1)
    bmi=st.number_input("Enter Your Body Mass Index/BMI Value (0-60)",0,60,1)
    age=st.number_input("Enter Your Age (20-80)",min_value=20,max_value=80,step=1)
    smoking_history=st.number_input("Enter Your Smoking History 1=never 2=smoked or currently smoking)",0,2,1)
    heart_disease=st.number_input("Enter 1 for Heart Disease or 0 if no Heart Disease",0,1,1)
    hypertension=st.number_input("Enter 1 for Hypertension or 0 if no Hypertension",0,1,1)
    gender=st.number_input("Enter your Gender (0 for Male or 1 for Female)",0,1,1)

    # Predict the diabetes state by passing the 8 features to the model
   
    feature_names = ['gender','age','hypertension','heart_disease','smoking_history','bmi','HbA1c_level','blood_glucose_level']


# Create a DataFrame for user input with feature names
    user_input = pd.DataFrame([[gender, age, hypertension, heart_disease, smoking_history, bmi, HbA1c_level, blood_glucose_level]], columns=feature_names)
    prediction2 = model2.predict(user_input)[0]
    # Display the results
    if st.button("Predict"):
        if prediction2==1:
            st.warning("You may have an increased probability of developing Diabetes, please check with your Doctor")
        elif prediction2==0:
            st.success("You have a low probability of developing Diabetes")

if rad=="Data":
    # Display information about the loaded data
    st.text("Loaded Data:")
    st.markdown("Please wait while this page loads.  This may take a few minutes.")
    st.write("Data Shape:", df2.shape)
    y2_pred = model2.predict(x2_test)
    accuracy = accuracy_score(y2_test, y2_pred)
    st.write("Model Accuracy: {:.0f}%".format(accuracy * 100))
    st.write("First 100 Rows:")
    st.write(df2.head(100))
    
    

                                        


