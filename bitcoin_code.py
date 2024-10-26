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
            color: white !important;
        }
        .stButton > button:hover {
            background-color: #0052cc;
            color: white !important;
        }
        .stButton > button:active, .stButton > button:focus {
            background-color: #0052cc;
            color: white !important;
        }
        /* This ensures the text stays white even after clicking */
        .stButton > button p {
            color: white !important;
        }
    </style>
""", unsafe_allow_html=True)

[Rest of your code remains exactly the same...]
