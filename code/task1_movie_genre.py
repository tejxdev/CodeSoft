import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
import os

# Define paths
train_path = "/home/ubuntu/task_folder/task1/Genre Classification Dataset/train_data.txt"
test_path = "/home/ubuntu/task_folder/task1/Genre Classification Dataset/test_data.txt"
test_solution_path = "/home/ubuntu/task_folder/task1/Genre Classification Dataset/test_data_solution.txt"

# Load data
print("Loading data...")
train_data = pd.read_csv(train_path, sep=" ::: ", engine="python", names=["ID", "TITLE", "GENRE", "DESCRIPTION"])
test_data = pd.read_csv(test_path, sep=" ::: ", engine="python", names=["ID", "TITLE", "DESCRIPTION"])
test_solution = pd.read_csv(test_solution_path, sep=" ::: ", engine="python", names=["ID", "TITLE", "GENRE", "DESCRIPTION"])

# Preprocessing
print("Preprocessing data...")
train_data['DESCRIPTION'] = train_data['DESCRIPTION'].str.lower()
test_data['DESCRIPTION'] = test_data['DESCRIPTION'].str.lower()

# Feature Extraction
print("Vectorizing text data...")
tfidf = TfidfVectorizer(stop_words='english', max_features=10000)
X_train = tfidf.fit_transform(train_data['DESCRIPTION'])
X_test = tfidf.transform(test_data['DESCRIPTION'])

y_train = train_data['GENRE']
y_test = test_solution['GENRE']

# Model Training
print("Training Multinomial Naive Bayes model...")
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

# Predictions
print("Making predictions...")
y_pred = nb_model.predict(X_test)

# Evaluation
print("\nEvaluation Results:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# Save sample predictions
output_df = test_data[['TITLE']].copy()
output_df['PREDICTED_GENRE'] = y_pred
output_df['ACTUAL_GENRE'] = y_test
output_df.head(20).to_csv("/home/ubuntu/task_folder/task1_results_sample.csv", index=False)
print("\nSample results saved to task1_results_sample.csv")
