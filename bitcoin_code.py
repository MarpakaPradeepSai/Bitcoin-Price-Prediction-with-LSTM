import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Custom CSS to style the button
st.markdown("""
    <style>
        .stButton > button {
            background-color: #0066ff;
            color: white;
        }
        .stButton > button:hover {
            background-color: #0052cc;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# Load the Bitcoin model
model_file = 'LSTM_Bitcoin_5_1(98831.51).h5'
try:
    model = load_model(model_file)
except Exception as e:
    st.error(f"Error loading Bitcoin model: {e}")

# Function to get Bitcoin data
def get_bitcoin_data(ticker='BTC-INR'):
    data = yf.download(ticker, period='max')
    return data

# Function to make predictions for days
def predict_next_days(model, data, look_back=5, days=5):
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)
    
    last_sequence = data_scaled[-look_back:]
    predictions = []

    for _ in range(days):
        X_input = np.reshape(last_sequence, (1, look_back, 1))
        prediction = model.predict(X_input)
        predictions.append(prediction[0, 0])
        
        # Update the sequence for the next prediction
        last_sequence = np.append(last_sequence[1:], prediction, axis=0)
    
    # Inverse transform the predictions to the original scale
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    return predictions

# Streamlit app layout
st.markdown("<h1 style='text-align: center; font-size: 50px;'>Bitcoin Price Predictor 📈📉</h1>", unsafe_allow_html=True)

# User input for number of days to forecast
num_days = st.slider("Select number of days to forecast", min_value=1, max_value=30, value=5)

# Display current date
current_date = datetime.now().strftime('%Y-%m-%d')
st.write(f"Current Date: {current_date}")

# Button to predict Bitcoin prices
if st.button(f'Predict Next {num_days} Days Bitcoin Prices'):
    # Load Bitcoin data
    bitcoin_data = get_bitcoin_data()
    close_prices = bitcoin_data['Close'].values.reshape(-1, 1)
    dates = bitcoin_data.index

    # Display the historical data
    st.markdown(f"### Historical Data for Bitcoin (BTC-INR)")
    st.dataframe(bitcoin_data, height=400, width=1000)

    # Predict the next num_days
    look_back = 5
    predictions = predict_next_days(model, close_prices, look_back=look_back, days=num_days)
    
    # Create dates for the predictions
    last_date = dates[-1]
    prediction_dates = [last_date + timedelta(days=i) for i in range(1, num_days + 1)]

    # Prepare data for plotting the historical and predicted prices
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(dates, close_prices, label='Historical Prices', color='blue')
    ax.plot(prediction_dates, predictions, label='Predicted Prices', linestyle='--', color='red')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price (INR)')
    ax.set_title('Bitcoin Prices (BTC-INR)', fontsize=16, fontweight='bold')
    ax.legend()
    ax.grid()

    st.pyplot(fig)

    # Plot only the predicted Bitcoin prices
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.plot(prediction_dates, predictions, marker='o', color='blue')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Predicted Price (INR)')
    ax2.set_title(f'Predicted Bitcoin Prices for the Next {num_days} Days', fontsize=16, fontweight='bold')
    
    # Use DayLocator to specify spacing of tick marks and set the format for the date labels
    ax2.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    
    plt.xticks(rotation=45)
    
    st.pyplot(fig2)
    
    # Show predictions in a table format
    prediction_df = pd.DataFrame({
        'Date': [d.strftime('%Y-%m-%d') for d in prediction_dates],  # Format the date
        'Predicted Price (INR)': predictions.flatten()
    })
    st.markdown(f"##### Predicted Bitcoin Prices for the Next {num_days} Days")
    st.dataframe(prediction_df, width=600)

# Disclaimer
st.markdown("""
<div style='text-align: center;'>
    <h4 style='color: red;'>Disclaimer:</h4>
    <p>This prediction model is for informational purposes only and should not be used as a basis for making financial decisions.</p>
</div>
""", unsafe_allow_html=True)
