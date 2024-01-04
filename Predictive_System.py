# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pickle

# Loading the saved model
loaded_model = pickle.load(open('trained_model (1).sav', 'rb'))

input_data=(2,197,70,45,543,30.5,0.158,53)

# Changing the input_data to numpy array
input_data_as_numpy_array=np.array(input_data)

# Reshape the array as we are predicting for one instance
input_data_reshape=input_data_as_numpy_array.reshape(1,-1)

prediction=loaded_model.predict(input_data_reshape)
print(prediction)

if (prediction[0]==0):
    print("Person is not Diabestes")
else:
    print("Person is Diabetes")