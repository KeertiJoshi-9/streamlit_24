import streamlit as st
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import time



st.set_page_config(page_title="Jamboree Education", 
                   page_icon="https://www.creativefabrica.com/wp-content/uploads/2020/02/10/Education-Logo-Graphics-1-10.jpg",
                   layout="wide")
st.markdown(
    """
    <style>
    .main {
    background-color:#f0f8ff;
    }
    <\style>
    """ , unsafe_allow_html=True
)

st.title("Jamboree Education Institute - Admission Chance Predictor")
st.subheader("A Linear regression case study")

st.text("This interactive tool will help you predict the chance of getting admission to Ivy league schools based on your inputs")
st.text("The default values are filled in, please enter your own input to check for your score and other details")
st.write("______________________________________________________________________________________________")
st.write(" ")

col1, col2 = st.columns(2)

with col1:
    st.image("jamboree_ad.jpg")

with col2:
    cgpa = st.number_input(
        label = 'CGPA',
        min_value= 0.0,
        max_value=10.0,
        value=9.0,
        step = 0.1,
        help = 'Cumulative Grade Point Average (CGPA) should be between 0 and 10',
        placeholder = "Enter your CGPA here"
    )

    gre = st.number_input(
        label = 'GRE Score',
        min_value= 0,
        max_value=340,
        value=300,
        step = 1,
        help = 'Graduate Record Examination (GRE) score should be between 0 and 340',
        placeholder = "Enter your GRE score here"
    )
    toefl_score = st.number_input(
        label = 'TOEFL Score',
        min_value= 0,
        max_value=120,
        value=100,
        step = 1,
        help = 'Test of English as a Foreign Language (TOEFL) score should be between 0 and 120',
        placeholder = "Enter your TOEFL Score here"
    )
    urating = st.radio(
        label = 'University Rating',
        options=[0,1,2,3,4,5],
        help="Rate your university on a scale of 0 to 5"
    )
    experience_research = st.radio(
        label = 'Do you have any Research Experience?',
        options=['Yes', 'No']
    )
    experience_research = 1 if experience_research=='Yes' else 0

    lor=st.slider(
        label="Letter of Recommendation Strength",
        min_value=0.0,
        max_value=5.0,
        value=4.0,
        step=0.5,
        help="Rate the strength of your Letter of Recommendations from 0 to 5"
    )

col1, col2 = st.columns(2)

with col1:
    st.image("jamboree_ad1.jpg")

with col2:
    if st.button("Know your Chances", type='primary'):
        input_data = np.array([[gre, toefl_score, urating, 5, lor, cgpa, experience_research]])

        with open("scale_jamboree.pkl", 'rb') as f:
            scaler = pickle.load(f)
        scaled_input = scaler.transform(input_data)
        print(scaled_input)

        scaled_input = scaled_input.reshape(-1,1)
        scaled_input= np.append(scaled_input[:3], scaled_input[4:]).reshape(1,-1)

        with open("sk_model_jamboree.pkl", 'rb') as f1:
            sk_model = pickle.load(f1)
        with st.spinner("Predicting..."):
            time.sleep(1)
        
        pred = sk_model.predict(scaled_input)
        if pred<0:
            pred = [0]
        elif pred > 1.0:
            pred = [1.0]
        
        percentage_pred = np.round(pred[0]*100,2)
        if percentage_pred > 85.0:
            st.markdown(f"<h3 style='color:#046e45; font-weight:bold;'> There is a {percentage_pred} chance of admission", unsafe_allow_html=True)
        elif percentage_pred > 75.0 and percentage_pred < 85.0 :
            st.markdown(f"<h3 style='color:#c49212; font-weight:bold;'> There is a {percentage_pred} chance of admission", unsafe_allow_html=True)
        else:
            st.markdown(f"<h3 style='color:#a3170a; font-weight:bold;'> There is a {percentage_pred} chance of admission", unsafe_allow_html=True)
        

