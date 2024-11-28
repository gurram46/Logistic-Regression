import streamlit as st
import joblib

st.title('Logistic Regression Model Deployment')

# Load the trained model
try:
    model = joblib.load(r'C:\Users\sande\OneDrive\Desktop\Documents\Streamlit_Logistic_Regression\logistic_model (1).pkl')
    st.write("Model loaded successfully.")
    st.write(f"The model expects {model.n_features_in_} features.")
except Exception as e:
    st.write(f"Error loading the model: {e}")

# Get user inputs
num_features = model.n_features_in_

inputs = []
for i in range(num_features):
    inputs.append(st.number_input(f'Enter feature {i+1} value'))

# Make prediction
if st.button('Predict'):
    try:
        prediction = model.predict([inputs])
        st.write(f'Prediction: {prediction[0]}')
    except Exception as e:
        st.write(f"Error making prediction: {e}")
