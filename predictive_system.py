# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 11:54:52 2024

@author: RAJ
"""


import numpy as np
import pickle

# loading the saved model
loaded_model = pickle.load(open('C:\\Users\\RAJ\\Model_Deployment\\trained_model.pkl','rb'))

def Diabetes_or_not2(input_data):
    array_data = np.asarray(input_data)
    input_data_reshaped = array_data.reshape(1,-1)
    
    predic = loaded_model.predict(input_data_reshaped)
    if predic == 0:
        print('The Person is Not Diabetic')
    else:
        print('The Person is Diabetic')
        
#Diabetes_or_not2(input_data=(0,137,40,35,168,43.1,2.288,33))