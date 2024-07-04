import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image

# Load your trained model and scalers
model = joblib.load('EDA\Flight\model.pkl')
scaler = joblib.load('D:\ML\EDA\Flight\scaler.pkl')
y_scaler = joblib.load('D:\ML\EDA\Flight\y_scaler.pkl')

# Preprocess function
def preprocess_data(df):
    df['stops'] = df['stops'].map({'zero': 0, 'one': 1, 'two_or_more': 2})
    df['class'] = df['class'].map({'Economy': 0, 'Business': 1})
    df['duration'] = df['duration'].astype(int).round()
    df['departure_time'] = df['departure_time'].map({'Early_Morning': 1, 'Morning': 2, 'Afternoon': 3, 'Evening': 4, 'Night': 5, 'Late_Night': 6})
    df['arrival_time'] = df['arrival_time'].map({'Early_Morning': 1, 'Morning': 2, 'Afternoon': 3, 'Evening': 4, 'Night': 5, 'Late_Night': 6})
    return df

# Streamlit app
st.set_page_config(page_title="Flight Price Prediction", page_icon="✈️", layout="wide")

# Add a background image
def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://images.unsplash.com/photo-1512446733611-9099a758e9a0");
             background-size: cover;
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

add_bg_from_url()

# Title and description
st.title('Flight Ticket Price Prediction')
st.markdown("""
    <div style="background-color:rgba(255, 255, 255, 0.8); padding: 10px; border-radius: 10px;">
        <h2 style="color: #333;">Enter the details of your flight to get the predicted price.</h2>
    </div>
    """, unsafe_allow_html=True)

# Collect user inputs
with st.form(key='flight_form'):
    col1, col2 = st.columns(2)
    with col1:
        airline = st.selectbox('Airline', ['SpiceJet', 'AirAsia', 'Vistara', 'GO_FIRST', 'Indigo', 'Air_India'])
        source_city = st.selectbox('Source City', ['Delhi', 'Mumbai', 'Bangalore', 'Kolkata', 'Hyderabad', 'Chennai'])
        departure_time = st.selectbox('Departure Time', ['Early_Morning', 'Morning', 'Afternoon', 'Evening', 'Night', 'Late_Night'])
        stops = st.selectbox('Stops', ['zero', 'one', 'two_or_more'])
    with col2:
        destination_city = st.selectbox('Destination City', ['Mumbai', 'Bangalore', 'Kolkata', 'Hyderabad', 'Chennai', 'Delhi'])
        arrival_time = st.selectbox('Arrival Time', ['Early_Morning', 'Morning', 'Afternoon', 'Evening', 'Night', 'Late_Night'])
        flight_class = st.selectbox('Class', ['Economy', 'Business'])
        duration = st.slider('Duration (hours)', 0, 50, 1)
        days_left = st.slider('Days Left for Departure', 0, 50, 1)
    
    submit_button = st.form_submit_button(label='Predict')

# Create a DataFrame from user inputs
if submit_button:
    user_input = pd.DataFrame({
        'class': [flight_class],
        'duration': [duration],
        'stops': [stops],
        'departure_time': [departure_time],
        'arrival_time': [arrival_time],
        'days_left': [days_left]
    })

    # Preprocess the user input
    user_input = preprocess_data(user_input)

    # Scale the user input
    user_input_scaled = scaler.transform(user_input)

    # Predict the price using the model
    predicted_price = model.predict(user_input_scaled)

    # Rescale the predicted price to the original price
    predicted_price_rescaled = y_scaler.inverse_transform(predicted_price.reshape(-1, 1))

    # Display the predicted price
    st.markdown(f"""
        <div style="background-color:rgba(255, 255, 255, 0.8); padding: 20px; border-radius: 10px; text-align: center;">
            <h3 style="color: #333;">Predicted Price: ₹{predicted_price_rescaled[0][0]:.2f}</h3>
        </div>
        """, unsafe_allow_html=True)