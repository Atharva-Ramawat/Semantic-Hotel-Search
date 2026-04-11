# frontend_streamlit/data_loader.py
import pandas as pd
import streamlit as st
import os

# Safely locate the CSV file one directory above this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, '..', 'hotels.csv')

@st.cache_data
def load_data():
    try:
        df = pd.read_csv(CSV_PATH, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(CSV_PATH, encoding='latin1')
    
    df = df.sample(n=10000, random_state=42)
    df.columns = df.columns.str.strip()
    df = df.fillna('')
    
    df['Search_Text'] = df['HotelName'].astype(str) + " " + \
                        df['Description'].astype(str) + " " + \
                        df['HotelFacilities'].astype(str) + " " + \
                        df['cityName'].astype(str) + " " + \
                        df['countyName'].astype(str)
    return df