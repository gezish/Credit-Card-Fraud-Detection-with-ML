import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib

class CreditCardFraudDetection:
    def init(self, file_path):
        self.data = pd.read_csv(file_path)
        self.encoder = LabelEncoder()
        self.classifier = RandomForestClassifier(class_weight='balanced')

    def explore_data(self):
        print(self.data.describe())
        print(self.data.info())

    def calculate_spend_per_fraud_category(self):
        print("Median Spend per Fraud Category:")
        print(self.data.groupby("is_fraud")["amt"].median())
        print('\n')
        print("Mean Spend per Fraud Category:")
        print(self.data.groupby("is_fraud")["amt"].mean())

    def calculate_total_spend_per_credit_card(self, credit_card_num):
        total_legitimate_spend = self.data[(self.data["cc_num"] == credit_card_num) & (self.data["is_fraud"] == 0)]["amt"].sum()
        total_fraudulent_spend = self.data[(self.data["cc_num"] == credit_card_num) & (self.data["is_fraud"] == 1)]["amt"].sum()
        return total_legitimate_spend, total_fraudulent_spend

    def clean_data(self):
        self.data = self.data.drop(["unix_time", "trans_num"], axis=1)
        self.data["trans_date_trans_time"] = self.data["trans_date_trans_time"].apply(lambda x: int(x.split(" ")[1][:2]) // 6)
        print(self.data["trans_date_trans_time"].value_counts())

    def create_correlation_matrix(self):
        plt.figure(figsize=(15, 10), dpi=150)
        sns.heatmap(self.data.corr(), vmin=0, vmax=1, cmap="viridis")

    def encode_categorical_features(self):
        categorical_features = self.data.select_dtypes(include=['object']).columns
        self.data[categorical_features] = self.data[categorical_features].apply(self.encoder.fit_transform)

    def train_model(self):
        data_encoded, labels = self.data.drop("is_fraud", axis=1), self.data["is_fraud"]
        X_train, X_test, y_train, y_test = train_test_split(data_encoded, labels, test_size=0.1, random_state=42)
        self.classifier.fit(X_train, y_train)
        joblib.dump(self.classifier, "../models")

    def evaluate_model(self):
        preds = self.classifier.predict(X_test)
        accuracy = accuracy_score(preds, y_test)
        confusion_mat = confusion_matrix(y_test, preds)
        return accuracy, confusion_mat

# Example Usage:
file_path = '../dataset/credit_card.csv'
fraud_detection = CreditCardFraudDetection(file_path)
fraud_detection.explore_data()
fraud_detection.calculate_spend_per_fraud_category()
total_legitimate_spend, total_fraudulent_spend = fraud_detection.calculate_total_spend_per_credit_card(344709867813900)
fraud_detection.clean_data()
fraud_detection.create_correlation_matrix()
fraud_detection.encode_categorical_features()
fraud_detection.train_model()
accuracy, confusion_matrix = fraud_detection.evaluate_model()