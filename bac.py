import pandas as pd
import os, sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import streamlit as st

import seaborn as sns
import matplotlib.pyplot as plt
import joblib
st.set_option('deprecation.showPyplotGlobalUse', False)
import plotly.express as px
from io import StringIO

from sklearn.ensemble import RandomForestClassifier
sys.path.append(os.path.abspath(os.path.join('../scripts')))
#mod = CreditCardFraudDetection()
from sklearn.metrics import accuracy_score, confusion_matrix
@st.cache_data
def load_data():
    data = pd.read_csv('./dataset/credit_card.csv')
    return data
@st.cache_data
def load_enc_data():
    data = pd.read_csv('./dataset/encoded_data.csv')
    return data
## Create a function to bin timestamps into categories
def quantitize(string):
    time_hour = int(string[:2])
    if time_hour < 6:
        return 0
    elif 6 <= time_hour < 12:
        return 1
    elif 12 <= time_hour <18:
        return 2
    else:
        return 3

def main():
    st.sidebar.title('Stages')
    
    selected_section = st.sidebar.radio('Go to', ['Home', 'Load Data',
                                                  'Average Values Spend Per Fraud',
                                                  'Total Fraud and Non-fraud Spend',
                                                  'Feature Selection',
                                                  'Train-Test Split',
                                                  
                                                  ])
    st.title('Classification of Credit Card Fraud with Machine Learning')
    
    if selected_section == 'Home':
        st.markdown('#### Business Need')
        st.markdown(
    '''
         A a Machine Learning Engineer at a financial services startup company, where you have been tasked with developing a machine learning classifier to identify fraudulent credit card transactions. 
         The company thinks that creating a reliable fraud detection system will allow them to provide a Software as a Service platform in the future to detect fraud at banking institutions. 
         A potential banking partner has provided the company with a dataset of credit card transactions that have been labeled as either fraudulent or non-fraudulent.
         Your boss wants you to build a classifier that can accurately identify fraudulent transactions.
    ''')
    elif selected_section == 'Load Data':
        st.header('Data Loading and Overview:')
        data = load_data()
        st.markdown('#### Data Info')
        buffer = StringIO()
        data.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)
        st.write(data)
        st.markdown('#### Descriptive statistics info')
        st.write(data.describe())
        
    elif selected_section == 'Average Values Spend Per Fraud':
        data = load_data()
        st.markdown('#### Check Average Values Spend Per Fraud')
        st.write('1. We can use groupby() to separate based on the is_fraud column categories and then calculate the mean and median values:')
        st.markdown("**Median Spend per Fraud Category:**")
        st.write(data.groupby("is_fraud")["amt"].median())
        
        st.markdown("**Mean Spend per Fraud Category:**")
        st.write(data.groupby("is_fraud")["amt"].mean())
        
    elif selected_section == 'Total Fraud and Non-fraud Spend':
        data = load_data()
        st.markdown('#### Calculate Total Fraud and Non-fraud Spend on a Specific Credit Card')
        st.markdown('**Sum of legitimate purchases:**')
        
        st.write(data[(data["cc_num"] == 344709867813900) & (data["is_fraud"] == 0)]["amt"].sum())
        st.markdown('**Sum of fraud purchases:**')
        st.write(data[(data["cc_num"] == 344709867813900) & (data["is_fraud"] == 1)]["amt"].sum())
        
        st.write("**Both operations:**", data[(data["cc_num"] == 344709867813900)].groupby('is_fraud').sum()["amt"])
     
    elif selected_section == 'Feature Selection': 
        data = load_data()
        st.markdown('#### Clean Data Columns for further analysis:')
        st.markdown('* Drop some columns which do not actually hold information relevant to the transaction being fraudulent.')
        st.markdown('- Drop the columns date portion of the timestamp')
        
        data = data.drop(["unix_time", "trans_num"], axis=1)
        
        st.markdown('We create some new features to indicate a specific category for time of day (to tell if the transaction occurred within a specific block of time in a day).')
        st.markdown('To do this, clean the transaction time column (trans_date_trans_time) by binning the trans_date_trans_time column into 4 categories:')
        st.markdown('- Category 1: 00:00:00 to 05:59:59')
        st.markdown('- Category 2: 06:00:00 to 11:59:59')

        st.markdown('- Category 3: 12:00:00 to 17:59:59')

        st.markdown('- Category 4: 8:00:00 to 23:59:59')
        ## Create a function to bin timestamps into categories
        ## create the encoder
        encoder = LabelEncoder()
        data["trans_date_trans_time"] = data["trans_date_trans_time"].apply(lambda x: x.split(" ")[1])
        
        data["trans_date_trans_time"] = data["trans_date_trans_time"].apply(quantitize)
        if st.checkbox('See each cluster'):
            st.write(st.write(data["trans_date_trans_time"].value_counts()))
        
        if st.checkbox('Visual Correlation of Data Features'): 
            st.markdown('#### Visual Correlation Matrix of Data Features')
            plt.figure(figsize=(15,10),dpi=150)
            sns.heatmap(data.corr(numeric_only=True),vmin=0,vmax=1,cmap="viridis")
            st.pyplot()
            st.write('* We can notice that how the amount spent, and long and lat location information are by far the most correlated features to fraud')
        
        ### Get the categorical features with Pandas

        categorical_features = data.select_dtypes(include=['object']).columns
        ### Apply fit_transform to create the encoded category data columns
        data_encoded = data.copy()
        data_encoded[categorical_features] = data_encoded[categorical_features].apply(encoder.fit_transform) 
       # Display dataframe
        data_encoded.shape
        if st.checkbox("Encoded Categorical Data Features"):
            buffer = StringIO()
            data_encoded.info(buf=buffer)
            s = buffer.getvalue()
            st.text(s)
            
            st.write('Data Shape:', data_encoded.shape)
            if st.button('Save The Data to CSV'):
                data_encoded.to_csv('./dataset/encoded_data.csv', index=False)
                st.write("Data saved to CSV fil Successfully.")
    elif selected_section=='Train-Test Split':
        
        data_encoded = load_enc_data()
        ### Separate Features and Label
        data_encoded, labels = data_encoded.drop("is_fraud", axis=1), data_encoded["is_fraud"]                
        ### Perform the split
        st.markdown('#### Split result')
        X_train, X_test, y_train, y_test = train_test_split(data_encoded, labels, test_size = 0.1, random_state = 42)
        st.write("X_train:", X_train.shape) 
        st.write("X_test:", X_test.shape)
        st.write("y_train:", y_train.shape)
        st.write("y_test:", y_test.shape)
        if st.checkbox('Train a Random Forest Classifier'):
            classifier = RandomForestClassifier()
            classifier.fit(X_train, y_train)
            st.markdown('Training completed use the **Save The Model** Button to save it')
            if st.button('Save The Model'):
                joblib.dump(classifier, "./models/model.pkl")
                st.write("Model saved Successfully.")
                
        if st.checkbox('Evaluate the Model on the Test Set'):
            y_pred = classifier.predict(X_test)
            st.write('Accuracy:', accuracy_score(y_test, y_pred))
            ## Create the confusion matrix

            st.write('**Confusion Matric**:', confusion_matrix(y_test, y_pred))
            
        st.markdown("Observation from the above result:") 
        st.write("**From the Accuracy**:") 
        st.markdown("- We can understand that our model can correctly predict **58,020** data values from the given 59073 dataset i.e, 98%")    

        
        
        
if __name__ == '__main__':
    main()