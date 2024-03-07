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

# Train your RandomForestClassifier model
@st.cache_data
def train_model(data):
    # Define features and target
    X = data.drop(columns=['target_column'])
    y = data['target_column']

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train RandomForestClassifier model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    return model, X_test, y_test

# Streamlit app
def main():
    st.title('Credit Card Fraud Detection using ML Model')

    # Load data
    data_load_state = st.text('Loading data...')
    data = load_data()
    data_load_state.text('Loading data... done!')
    
    # Display basic info about the loaded data
    st.subheader('Basic Info of Loaded Data')
    st.write("Number of rows & columns:", data.shape)   
    st.subheader('Data Info')
    buffer = StringIO()
    data.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)  

# Example of sidebar usage
    """st.sidebar.header("Descriptive statistics of a DF")
    st.sidebar.slider("Slider", 0, 100, 50)"""

    st.write(data.describe())

    # Train model
    model_train_state = st.text('Training model...')
    model, X_test, y_test = train_model(data)
    model_train_state.text('Training model... done!')

    # Display dataset
    st.subheader('Dataset')
    st.write(data)

    # Display model evaluation
    st.subheader('Model Evaluation')
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.write('Accuracy:', accuracy)

    # Add more features as needed

if __name__ == '__main__':
    main()