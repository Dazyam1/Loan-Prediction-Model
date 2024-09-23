import streamlit as st
import pandas as pd
import pickle
import joblib

# Load the pre-trained model using pickle
model_path = 'C:/Users/user/Documents/Data Science/Models/Loan/Loan Prediction Model 1.sav'
with open(model_path, 'rb') as file:
    model = joblib.load(file)

# Initialize the input fields in Streamlit
st.title('Loan Status Prediction')
st.write("Provide your details to predict the likelihood of loan approval.")

# Create a sidebar for navigation and input fields
st.sidebar.title("Input Features")
st.sidebar.write("Fill in the details below:")

# Input fields for the loan prediction model
gender = st.sidebar.selectbox('Gender', options=[1, 0], index=0)  # 1: Male, 0: Female
married = st.sidebar.selectbox('Married', options=[1, 0], index=0)  # 1: Yes, 0: No
dependents = st.sidebar.selectbox('Dependents', options=[0, 1, 2, 4], index=0)  # 0, 1, 2, 4 dependents
education = st.sidebar.selectbox('Education', options=[1, 0], index=0)  # 1: Graduate, 0: Not Graduate
self_employed = st.sidebar.selectbox('Self Employed', options=[0, 1], index=0)  # 0: No, 1: Yes
applicant_income = st.sidebar.slider('Applicant Income', min_value=0, max_value=100000, value=5000)
coapplicant_income = st.sidebar.slider('Coapplicant Income', min_value=0, max_value=50000, value=0)
loan_amount = st.sidebar.slider('Loan Amount (in thousands)', min_value=0.0, max_value=1000.0, value=200.0)
loan_amount_term = st.sidebar.slider('Loan Amount Term (in months)', min_value=0, max_value=480, value=360)
credit_history = st.sidebar.selectbox('Credit History', options=[1, 0], index=0)  # 1: Yes, 0: No
property_area = st.sidebar.selectbox('Property Area', options=[0, 1, 2], index=0)  # 0: Rural, 1: Semiurban, 2: Urban

# Prepare input data for prediction
input_data = pd.DataFrame({
    'Gender': [gender],
    'Married': [married],
    'Dependents': [dependents],
    'Education': [education],
    'Self_Employed': [self_employed],
    'ApplicantIncome': [applicant_income],
    'CoapplicantIncome': [coapplicant_income],
    'LoanAmount': [loan_amount],
    'Loan_Amount_Term': [loan_amount_term],
    'Credit_History': [credit_history],
    'Property_Area': [property_area]
})

# Predict Button
if st.button('Predict'):
    try:
        # Make the prediction
        prediction = model.predict(input_data)

        # Display the prediction with increased font size
        st.subheader('Prediction:')
        if prediction[0] == 1:
            st.markdown('<p style="font-size:24px; color:green;">Loan Approved</p>', unsafe_allow_html=True)
            st.balloons()
        else:
            st.markdown('<p style="font-size:24px; color:red;">Loan Not Approved</p>', unsafe_allow_html=True)

            # Recommendations based on prediction
            st.subheader("Recommendations")
            st.write("""
            - **Check Credit History**: Ensure your credit history is accurate and up-to-date.
            - **Review Loan Amount**: Consider applying for a smaller loan amount.
            - **Reassess Income**: Ensure you have sufficient income or include a co-applicant with a higher income.
            - **Consult with the Bank**: Speak with your bank or financial advisor for more guidance.
            """)

        # Visualization of input data
        st.subheader("Input Data Visualization")
        input_df = pd.DataFrame([input_data.iloc[0]], columns=input_data.columns)
        st.bar_chart(input_df.T)

    except Exception as e:
        st.write(f"Error: {e}")
