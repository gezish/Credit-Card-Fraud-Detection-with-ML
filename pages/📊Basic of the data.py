from io import StringIO
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score




# Load your dataset
@st.cache_data
def load_data():
    # Load your dataset
    data = pd.read_csv('dataset/credit_card.csv')
    return data

def main():
    st.title('Credit Card Fraud Detection using ML Model')

    # Load data
    data_load_state = st.text('Loading data...')
    data = load_data()
    data_load_state.text('Loading data... done!')
    
    # Display basic info about the loaded data
    st.subheader('Basic Info of Loaded Data')
    st.write("Number of rows & columns", data.shape)
    st.subheader('Data Info')
    
    st.subheader('Data Info')
    buffer = StringIO()
    data.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)  

    st.write(data.describe())
