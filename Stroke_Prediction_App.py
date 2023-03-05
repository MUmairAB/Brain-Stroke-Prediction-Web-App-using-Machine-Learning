# -*- coding: utf-8 -*-
"""
This code creates an app that predicts the brain stroke using Machine Learning. The App is deployed on Streamlit Community Cloud.
It takes 7 characteristics as input and predicts whether you are vulnerable to brain stroke or not.

It is created by Umair Akram
"""

# Import necessary libraries
import numpy as np
import pickle
import streamlit as st

# Loading pre-trained model
model = pickle.load(open('trained_model.sav','rb'))

def stroke_classifier(user_input):
    
    # Convert 1D array to 2D features matrix
    X = np.array(user_input).reshape(1,-1)
    pred = model.predict(X)[0]
    if pred == 0:
        return("Congratulation! your brain stroke test result is negative")
    else:
        return("Unfortunately! your brain stroke test result is positive. Kindly consult a doctor!")

#################### Define main function ########################

def main():
    
    st.title('Brain Stroke Prediction App')
    st.write("Created by: [Umair Akram](https://www.linkedin.com/in/m-umair01/)")
    st.write("The Machine Learning model is a pipeline of Standard Scaler and XGBoost Classifier. Its code and other interesting projects are available on my [website](https://mumairab.github.io/)")
    h1 = "This App uses Machine Learning to predict whether you are vulnerable to Brain Stroke or not!"
    st.subheader(h1)
    h2 = "Enter the following values to know the status of your health"
    st.write(h2)
    
    ############### Taking user input ####################
    
    #                       Gender                       #
    st.write("1. Select your Gender:")
    gend = st.selectbox(label="", 
                        options=('Male', 'Female','Other'),
                        key=11132)
    st.write('Gender: ', gend)
    if gend == 'Male':
        gender = 0
    elif gend == 'Female':
        gender = 1
    else:
        gender = -1
        
    #                       Age                         #
    st.write("2. Enter your age:")
    age = st.number_input(label='',
                          min_value=1,
                          max_value=100,
                          )
    st.write('Age: ',age)

    #                       Hypertension                #    
    st.write("3. Do you suffer from hypertension?")
    hyper = st.radio(label="",
                     options=['Yes', 'No'],
                     key=1234)
 
    st.write("Hypertension (Y/N): ", hyper)
    if hyper == 'Yes':
        hypertension = 1
    else:
        hypertension = 0
        
    #                    Heart Disease                  #
    st.write("4. Do you have any heart disease?")
    heart = st.radio(label="",
                     options=['Yes', 'No'],
                     key=5678)
 
    st.write("Heart Disease (Y/N): ", heart)
    if heart == 'Yes':
        heart_disease = 1
    else:
        heart_disease = 0
    
    #                    Work Type                     #
    st.write("5. What type of work you do?")
    work = st.selectbox(label="",
          options=['Private Job', 'Self Employed','Govt. Job','Never Worked','You are a Child'],
          key=222)
 
    st.write("Work Type: ", work)
    if work == 'Private Job':
        work_type = 0
    elif work == 'Self Employed':
        work_type = 1
    elif work == 'Govt. Job':
        work_type = 2
    elif work == 'Never Worked':
        work_type = -2
    elif work == 'You are a Child':
        work_type = -1

    #                    Average Glucose               #
    st.write("6. Enter your average blood glucose level:")
    glucose = st.number_input(label='',
                          min_value=40.0,
                          max_value=400.0,
                          step=1.,
                          format="%.2f",
                          value=40.0)
    st.write("Blood Glucose Level: ",glucose)
    
    #                    BMI                           #
    st.write("7. Enter your Body mass index (BMI):")
    bmi = st.number_input(label='',
                          min_value=10.0,
                          max_value=100.0,
                          step=1.,
                          format="%.2f",
                          value=10.0)
    st.write("Body Mass Index (BMI): ",bmi)


    ########   Converting the user input to an array   ############
    
    arr = [gender,age,hypertension,heart_disease,work_type,glucose,bmi]
    
    ###################  Making Predictions   #####################
    
    strk = ''
    
    if st.button(label='Show the Brain Stroke Results'):
        strk = stroke_classifier(arr)
        
    st.success(strk)


if __name__ == '__main__':
    main()
