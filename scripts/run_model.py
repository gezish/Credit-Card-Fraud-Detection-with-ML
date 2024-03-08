import modeling as md



file_path = '../dataset/credit_card.csv'
fraud_detection = md.CreditCardFraudDetection(file_path)
fraud_detection.explore_data()
fraud_detection.calculate_spend_per_fraud_category()
total_legitimate_spend, total_fraudulent_spend = fraud_detection.calculate_total_spend_per_credit_card(344709867813900)
fraud_detection.clean_data()
fraud_detection.create_correlation_matrix()
fraud_detection.encode_categorical_features()
fraud_detection.train_model()
accuracy, confusion_matrix = md.evaluate_model()