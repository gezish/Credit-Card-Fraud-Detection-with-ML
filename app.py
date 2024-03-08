import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
from io import StringIO
st.set_option('deprecation.showPyplotGlobalUse', False)


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
    selected_section = st.sidebar.radio('Go to', ['Home', 'Basic Data Information',
                                                  'Check Average Values Spend Per Fraud',
                                                  'Total Fraud and Non-fraud Spend on a Specific Credit Card',
                                                  'Cleaned Data Columns for further analysis',
                                                  'Visual Correlation Matrix of Data Features',
                                                  'Train, test & Evaluate'
                                                  ])

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
        
    elif selected_section == 'Visual Correlation Matrix of Data Features':
        data = load_data()
       # st.write(data.describe())
        st.title("Correlation Matrix of Data Features:")
        plt.figure(figsize=(15,10),dpi=150)
        sns.heatmap(data.corr(numeric_only=True), vmin=0,vmax=1,cmap="viridis")
        st.pyplot()
        
        
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
        st.markdown("**Final results**")  
        st.write(clean_d["trans_date_trans_time"].value_counts())
    
    elif selected_section == 'Train, test & Evaluate':
        data = load_data()
        data = data.drop(["unix_time", "trans_num"], axis=1)
        st.title("Train-Test Split, Train, test & Evaluate The Model")
        st.markdown("## 1. Features Encoding ")
        st.write("* Using the Pandas DataFrame and Scikit-Learn, we use Label Encoding to encode the categorical features in the DataFrame.")
        encoder = LabelEncoder()
        categorical_features = data.select_dtypes(include=['object']).columns
        ### Apply fit_transform to create the encoded category data columns
        data_encoded = data.copy()
        data_encoded[categorical_features] = data_encoded[categorical_features].apply(encoder.fit_transform)
        st.write(data_encoded.shape)
        
        st.markdown("## 2. Train-Test(10% ) spliting Result:**")
        data_encoded, labels = data_encoded.drop("is_fraud", axis=1), data_encoded["is_fraud"]
        X_train, X_test, y_train, y_test = train_test_split(data_encoded, labels, test_size = 0.1, random_state = 42)
        st.write("X_train:", X_train.shape) 
        st.write("X_test:", X_test.shape)
        st.write("y_train:", y_train.shape)
        st.write("y_test:", y_test.shape)
           
        st.markdown("## 3. Train With Random Forest Classifier Model:")
        st.write("Random Forest Classifier training result:")
        st.write(" -Using Scikit-Learn create and train a random forest classifier on the training data set")
        classifier = RandomForestClassifier(class_weight='balanced')
        st.write(classifier.fit(X_train, y_train))              
        
        st.markdown("## 4. Model Evaluation Result:")
        #st.title('Model Evaluation')
        preds = classifier.predict(X_test)
        ## Calculate the accuracy
        st.write("Accuracy Score of the Model:", accuracy_score(preds, y_test))
        
        st.write("Confusion matrix result:", confusion_matrix(y_test, preds))

if __name__ == '__main__':
    main()
