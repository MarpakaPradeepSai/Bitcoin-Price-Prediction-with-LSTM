import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Load the Bitcoin model
model_file = 'LSTM_Bitcoin_5_1(98831.51).h5'
try:
    model = load_model(model_file)
    st.success("Bitcoin price prediction model loaded successfully.")
except Exception as e:
    st.error(f"Error loading Bitcoin model: {e}")

# Function to get Bitcoin data
def get_bitcoin_data(ticker='BTC-INR'):
    data = yf.download(ticker, period='max')
    return data

# Function to generate a list of business days
def generate_business_days(start_date, num_days):
    return pd.bdate_range(start=start_date, periods=num_days).tolist()

# Function to make predictions for business days
def predict_next_business_days(model, data, look_back=5, days=5):
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)
    
    last_sequence = data_scaled[-look_back:]
    predictions = []

    for _ in range(days):
        X_input = np.reshape(last_sequence, (1, look_back, 1))
        prediction = model.predict(X_input)
        predictions.append(prediction[0, 0])
        
        last_sequence = np.append(last_sequence[1:], prediction, axis=0)
    
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    return predictions

# Streamlit app layout
st.set_page_config(page_title="Bitcoin Price Predictor", page_icon="💰", layout="wide")

# Header
st.markdown("<h1 style='text-align: center; font-size: 50px;'>Bitcoin Price Predictor 💰📈</h1>", unsafe_allow_html=True)

# Sidebar for inputs
st.sidebar.header("User Inputs")
num_days = st.sidebar.slider("Select number of business days to forecast", min_value=1, max_value=30, value=5)
st.sidebar.markdown("<p style='color: grey;'>Adjust the slider to choose how many days ahead you'd like to predict Bitcoin prices.</p>", unsafe_allow_html=True)

# Display current date
current_date = datetime.now().strftime('%Y-%m-%d')
st.markdown(f"**Current Date:** {current_date}")

# Button to predict Bitcoin prices
if st.button(f'Predict Next {num_days} Days Bitcoin Prices', key='predict-button'):
    bitcoin_data = get_bitcoin_data()
    close_prices = bitcoin_data['Close'].values.reshape(-1, 1)
    dates = bitcoin_data.index

    st.markdown("### Historical Data for Bitcoin (BTC-INR)")
    st.dataframe(bitcoin_data.style.highlight_max(axis=0), height=400)

    look_back = 5
    predictions = predict_next_business_days(model, close_prices, look_back=look_back, days=num_days)
    
    last_date = dates[-1]
    prediction_dates = generate_business_days(last_date + timedelta(days=1), num_days)

    # Plot historical and predicted prices
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(dates, close_prices, label='Historical Prices', color='blue')
    ax.plot(prediction_dates, predictions, label='Predicted Prices', linestyle='--', color='orange')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price (INR)')
    ax.set_title('Bitcoin Prices (BTC-INR)', fontsize=16, fontweight='bold')
    ax.legend()
    ax.grid()

    st.pyplot(fig)

    # Show predictions in a table format
    prediction_df = pd.DataFrame({
        'Date': prediction_dates,
        'Predicted Price (INR)': predictions.flatten()
    })
    st.markdown(f"##### Predicted Bitcoin Prices for the Next {num_days} Business Days")
    st.dataframe(prediction_df.style.highlight_max(axis=0), width=600)

# Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("---")
st.markdown("<h6 style='text-align: center;'>Created with ❤️ by Your Name</h6>", unsafe_allow_html=True)
