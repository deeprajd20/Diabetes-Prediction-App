# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 12:01:19 2024

@author: RAJ
"""

import numpy as np 
import pickle 
import streamlit as st


loaded_model = pickle.load(open('C:\\Users\\RAJ\\Model_Deployment\\trained_model.pkl','rb'))

def Diabetes_or_not2(input_data):
    array_data = np.asarray(input_data)
    input_data_reshaped = array_data.reshape(1,-1)
    
    predic = loaded_model.predict(input_data_reshaped)
    if predic == 0:
        return 'The Person is Not Diabetic'
    else:
        return 'The Person is Diabetic'
        

def main():
    #giving title
    st.title('Diabetes Prediction Web App')
    
    #getting the input data from the user
    
    Pregnancies = st.text_input('Enter your pregnancies:')
    Glucose= st.text_input('Enter your Glucose:')
    BloodPressure= st.text_input('Enter your BP:')
    SkinThickness= st.text_input('Enter your Skin Thick:')
    Insulin= st.text_input('Enter your Insuline:')
    BMI= st.text_input('Enter your BMI:')
    DiabetesPedigreeFunction= st.text_input('Enter your DPF:')
    Age= st.text_input('Enter your Age:')
    
    # output
    diagnosis = ''
    
    # creating a button for prediction
    
    if st.button('Test Results'):
        diagnosis = Diabetes_or_not2([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
        
        
    st.success(diagnosis)
    
    
    
if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
    
    
    
    