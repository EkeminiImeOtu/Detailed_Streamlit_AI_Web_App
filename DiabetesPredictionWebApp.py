import numpy as np
import pickle
import streamlit as st

# Loading the saved model
loaded_model = pickle.load(open('diabetes_model.sav', 'rb'))

# Creating a function for Prediction
def diabetes_prediction(input_data):
    # Changing the input_data to a numpy array
    input_data_as_numpy_array = np.asarray(input_data, dtype=np.float32)
    # Reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = loaded_model.predict(input_data_reshaped)
    if prediction[0] == 0:
        return 'The person is not diabetic'
    else:
        return 'The person is diabetic'

def main():
    # Giving a title
    st.title('Diabetes Prediction Web App')
    
    # Dataset information
    st.write("""
    This dataset is originally from the National Institute of Diabetes and Digestive and Kidney Diseases. 
    The objective of the dataset is to diagnostically predict whether or not a patient has diabetes, based 
    on certain diagnostic measurements included in the dataset. Several constraints were placed on the 
    selection of these instances from a larger database. In particular, all patients here are females at 
    least 21 years old of Pima Indian heritage.
    """)
    
    # Getting the input data from the user
    Pregnancies = st.text_input('Number of Pregnancies', type='numeric', help='Enter the number of times the person has been pregnant.')
    Glucose = st.text_input('Glucose Level', type='numeric', help='Enter the plasma glucose concentration over 2 hours in an oral glucose tolerance test.')
    BloodPressure = st.text_input('Blood Pressure', type='numeric', help='Enter the diastolic blood pressure in mm Hg.')
    SkinThickness = st.text_input('Skin Thickness', type='numeric', help='Enter the triceps skin fold thickness in mm.')
    Insulin = st.text_input('Insulin Level', type='numeric', help='Enter the 2-hour serum insulin level in mu U/ml.')
    BMI = st.text_input('BMI', type='numeric', help='Enter the body mass index (weight in kg / (height in m)^2).')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function', type='numeric', help='Enter the diabetes pedigree function value.')
    Age = st.text_input('Age', type='numeric', help='Enter the age of the person in years.')
    
    # Code for Prediction
    diagnosis = ''
    
    # Creating a button for Prediction
    if st.button('Diabetes Test Result'):
        # Check if all inputs are valid numbers
        if all(isinstance(x, (int, float)) for x in [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]):
            diagnosis = diabetes_prediction([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])
        else:
            diagnosis = 'Please enter valid numeric values for all inputs'
        
    st.success(diagnosis)

if __name__ == '__main__':
    main()
