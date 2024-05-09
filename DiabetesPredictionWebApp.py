import numpy as np
import pickle
import streamlit as st

# loading the saved model
loaded_model = pickle.load(open('diabetes_model.sav', 'rb'))

# creating a function for Prediction
def diabetes_prediction(input_data):
    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)
    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    prediction = loaded_model.predict(input_data_reshaped)
    if prediction[0] == 0:
        return 'The person is not diabetic'
    else:
        return 'The person is diabetic'

def main():
    # giving a title
    st.title('Diabetes Prediction Web App')
    
    # Dataset information
    st.write("""
    The dataset used to build this artificial intelligence model is originally from the National Institute of Diabetes and Digestive and Kidney Diseases. 
    Its objective is to predict whether or not a patient has diabetes, based on certain diagnostic measurements included in the dataset. 
    Several constraints were placed on the selection of these instances from a larger database. 
    In particular, all patients here are females at least 21 years old of Pima Indian heritage.
    """)
    
    # getting the input data from the user
    Pregnancies = st.text_input('Number of Pregnancies', help='Enter the number of times the person has been pregnant.')
    Glucose = st.text_input('Glucose Level', help='Enter the plasma glucose concentration over 2 hours in an oral glucose tolerance test.')
    BloodPressure = st.text_input('Blood Pressure', help='Enter the diastolic blood pressure in mm Hg.')
    SkinThickness = st.text_input('Skin Thickness', help='Enter the triceps skin fold thickness in mm.')
    Insulin = st.text_input('Insulin Level', help='Enter the 2-hour serum insulin level in mu U/ml.')
    BMI = st.text_input('BMI', help='Enter the body mass index (weight in kg / (height in m)^2).')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function', help='Enter the diabetes pedigree function value.')
    Age = st.text_input('Age', help='Enter the age of the person in years.')
    
    # code for Prediction
    diagnosis = ''
    
    # creating a button for Prediction
    if st.button('Diabetes Test Result'):
        # Check if all inputs are valid numbers
        try:
            Pregnancies = float(Pregnancies)
            Glucose = float(Glucose)
            BloodPressure = float(BloodPressure)
            SkinThickness = float(SkinThickness)
            Insulin = float(Insulin)
            BMI = float(BMI)
            DiabetesPedigreeFunction = float(DiabetesPedigreeFunction)
            Age = float(Age)
            
            diagnosis = diabetes_prediction([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])
        except ValueError:
            diagnosis = 'Please enter valid numeric values for all inputs'
        
    st.success(diagnosis)

if __name__ == '__main__':
    main()
