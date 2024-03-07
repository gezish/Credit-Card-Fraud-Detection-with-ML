import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from io import StringIO

# Load your dataset
@st.cache_data
def load_data():
    # Load your dataset
    data = pd.read_csv('dataset/credit_card.csv')
    return data
@st.cache_data
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
def clean():
    # Load your dataset
    clean_d = pd.read_csv('dataset/credit_card.csv')
    clean_d["trans_date_trans_time"] = clean_d["trans_date_trans_time"].apply(lambda x: x.split(" ")[1])
    clean_d["trans_date_trans_time"] = clean_d["trans_date_trans_time"].apply(quantitize)
    return clean_d
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
def main(data = load_data()):
    st.sidebar.title('Navigation')
    selected_section = st.sidebar.radio('Go to', ['Home', 'Basic Data Information','Check Average Values Spend Per Fraud','Total Fraud and Non-fraud Spend on a Specific Credit Card',
                                                  'Cleaned Data Columns for further analysis','Model Evaluation'])

    if selected_section == 'Home':
        st.title('Classification of Credit Card Fraud with Machine Learning')
        st.write("**Informations about the project:-**")
        st.markdown("* Main objective is to Build a machine learning model to detect fraudulent credit card transactions.") 
        st.markdown("* The goal is to create a Software as a Service platform for banks..") 
        st.markdown("* A bank has provided a dataset labeled with fraudulent and legitimate transactions.") 
        st.markdown("* Your objective is to build a model to accurately identify fraud for a proof of concept.") 
        
    elif selected_section == 'Basic Data Information':
        st.title('Basic Info of Loaded Data')
        
        data = load_data()
        st.subheader('Basic Info of Loaded Data')
        st.write("Number of rows & columns:", data.shape)   
        st.subheader('Data Info')
        #data = load_data()
        buffer = StringIO()
        data.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)
        st.subheader('Descriptive Statistics')
        st.write(data.describe())
        
    elif selected_section == 'Check Average Values Spend Per Fraud':
        data = load_data()
       # st.write(data.describe())
        st.title("Average Values Spend Per Fraud:")
        st.markdown("**Median Spend per Fraud Category:**")
        st.write(data.groupby("is_fraud")["amt"].median())
        st.write('\n')
        st.markdown("**Mean Spend per Fraud Category:**")
        st.write(data.groupby("is_fraud")["amt"].mean())
        
    elif selected_section == 'Total Fraud and Non-fraud Spend on a Specific Credit Card':
        data = load_data()
       # st.write(data.describe())
        st.title("Total Fraud and Non-fraud Spend on a Specific Credit Card:")
        st.markdown("**Filter and calculate sum on legitimate purchases**")
        st.write(data[(data["cc_num"] == 344709867813900) & (data["is_fraud"] == 0)]["amt"].sum())
        st.write('\n')
        st.markdown("**Filter and calculate on fraud purchases:**")
        st.write(data[(data["cc_num"] == 344709867813900) & (data["is_fraud"] == 1)]["amt"].sum())  
         
          
    elif selected_section == 'Cleaned Data Columns for further analysis':
        clean_d = clean()
       # st.write(data.describe())
        st.title("Total Fraud and Non-fraud Spend on a Specific Credit Card:")
        st.markdown("**Drop some columns which do not actually hold information relevant to the transaction being fraudulent.**")
        st.write("Create some new features to indicate a specific category for time of day (to tell if the transaction occurred within a specific block of time in a day) To do this, clean the transaction time column (trans_date_trans_time) by binning the trans_date_trans_time column into 4 categories:")
        st.markdown("* Category 1: 00:00:00 to 05:59:59")
        st.markdown("* Category 2: 06:00:00 to 11:59:59")
        st.markdown("* Category 3: 12:00:00 to 17:59:59")
        st.markdown("* Category 4: 8:00:00 to 23:59:59")
        #st.write(data["trans_date_trans_time"] = data["trans_date_trans_time"].apply(lambda x: x.split(" ")[1]))
        
        #st.write(data["trans_date_trans_time"] = data["trans_date_trans_time"].apply(quantitize))       
        st.markdown("**Final results**")  
        st.write(clean_d["trans_date_trans_time"].value_counts())  
          
    elif selected_section == 'Model Evaluation':
        st.title('Model Evaluation')
        data = load_data()
        model, X_test, y_test = train_model(data)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        st.write('Accuracy:', accuracy)

if __name__ == '__main__':
    main()
