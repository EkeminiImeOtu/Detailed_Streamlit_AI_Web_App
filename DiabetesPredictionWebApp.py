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
    
    # dataset information
    st.markdown("""The dataset used to build this artificial intelligence model is originally from the National Institute of Diabetes and Digestive and Kidney Diseases. 
    Its objective is to predict whether or not a patient has diabetes, based on certain diagnostic measurements included in the dataset. 
    Several constraints were placed on the selection of these instances from a larger database. 
    In particular, all patients here are females at least 21 years old of Pima Indian heritage.""")
    
    # getting the input data from the user
    Pregnancies = st.text_input('Number of Pregnancies', type='numeric', help='Enter the number of times the person has been pregnant.')
    Glucose = st.text_input('Glucose Level', type='numeric', help='Enter the plasma glucose concentration over 2 hours in an oral glucose tolerance test.')
    BloodPressure = st.text_input('Blood Pressure value', type='numeric', help='Enter the diastolic blood pressure (mm Hg).')
    SkinThickness = st.text_input('Skin Thickness value', type='numeric', help='Enter the triceps skin fold thickness (mm).')
    Insulin = st.text_input('Insulin Level', type='numeric', help='Enter the 2-Hour serum insulin (mu U/ml).')
    BMI = st.text_input('BMI value', type='numeric', help='Enter the body mass index (weight in kg/(height in m)^2).')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value', type='numeric', help='Enter the diabetes pedigree function value (a function which scores likelihood of diabetes based on family history).')
    Age = st.text_input('Age of the Person', type='numeric', help='Enter the age in years.')
    
    # code for Prediction
    diagnosis = ''
    
    # creating a button for Prediction
    if st.button('Diabetes Test Result'):
        diagnosis = diabetes_prediction([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])
    
    st.success(diagnosis)

if __name__ == '__main__':
    main()
