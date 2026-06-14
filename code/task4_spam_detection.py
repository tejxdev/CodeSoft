import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import os

# Define path
data_path = "/home/ubuntu/task_folder/task4/spam.csv"

print("Loading data...")
# The CSV seems to have some encoding issues and extra columns
df = pd.read_csv(data_path, encoding='latin-1')
df = df.iloc[:, :2]
df.columns = ['label', 'message']

# Data Preprocessing
print("Preprocessing data...")
df['label'] = df['label'].map({'ham': 0, 'spam': 1})
df['message'] = df['message'].str.lower()

# Feature Extraction
print("Vectorizing text data...")
tfidf = TfidfVectorizer(stop_words='english')
X = tfidf.fit_transform(df['message'])
y = df['label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
print("Training SVM model...")
svm_model = SVC(kernel='linear', C=1.0)
svm_model.fit(X_train, y_train)

# Predictions
print("Making predictions...")
y_pred = svm_model.predict(X_test)

# Evaluation
print("\nEvaluation Results:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save sample results
sample_df = pd.DataFrame({
    'Message': df.iloc[y_test.index]['message'].values[:20],
    'Actual': y_test.values[:20],
    'Predicted': y_pred[:20]
})
sample_df['Actual'] = sample_df['Actual'].map({0: 'ham', 1: 'spam'})
sample_df['Predicted'] = sample_df['Predicted'].map({0: 'ham', 1: 'spam'})
sample_df.to_csv("/home/ubuntu/task_folder/task4_results_sample.csv", index=False)
print("\nSample results saved to task4_results_sample.csv")
