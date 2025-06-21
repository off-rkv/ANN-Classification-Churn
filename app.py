import streamlit as st
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder

# Load the trained model
model=tf.keras.models.load_model('model.h5')


# Load the encoder
with open('label_encoder_gender.pkl', 'rb') as encoder_file:
    label_encoder_gender = pickle.load(encoder_file)

with open('onehot_encoder_geo.pkl', 'rb') as encoder_file:
    onehot_encoder_geo = pickle.load(encoder_file)

# Load the scaler
with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

##Streamlit app
st.title('Customer Churn Prediction')

# User input fields
geography = st.selectbox('Geography',['France','Germany','Spain'])
#geography = st.selectbox('Geography', onehot_encoder_geo.categories[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', min_value=18, max_value=100)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure (years)',0,20)
num_of_products = st.number_input('Number of Products', min_value=1, max_value=4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# Prepare input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()

geo_encoded_df=pd.DataFrame(geo_encoded,columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

input_data=pd.concat([input_data.reset_index(drop=True),geo_encoded_df],axis=1)

input_data_scaled=scaler.transform(input_data)

prediction=model.predict(input_data_scaled)
prediction_prob=float(prediction[0][0])

st.metric(label="Churn Probability", value=f"{prediction_prob*100:.2f} %")

if prediction_prob > 0.5:
    st.warning("⚠️ The customer is likely to churn.")
else:
    st.success("✅ The customer is not likely to churn.")
