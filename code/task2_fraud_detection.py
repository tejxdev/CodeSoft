import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import os

# Define paths
train_path = "/home/ubuntu/task_folder/task2/fraudTrain.csv"
test_path = "/home/ubuntu/task_folder/task2/fraudTest.csv"

print("Loading data...")
# Loading a subset for speed if files are very large, but let's try full first
train_df = pd.read_csv(train_path, index_col=0)
test_df = pd.read_csv(test_path, index_col=0)

# Feature selection & Preprocessing
print("Preprocessing data...")
# We'll use a subset of features for simplicity: category, amt, gender, city_pop
features = ['category', 'amt', 'gender', 'city_pop']
target = 'is_fraud'

X_train = train_df[features].copy()
y_train = train_df[target]
X_test = test_df[features].copy()
y_test = test_df[target]

# Encode categorical variables
le = LabelEncoder()
for col in ['category', 'gender']:
    X_train[col] = le.fit_transform(X_train[col])
    X_test[col] = le.transform(X_test[col])

# Since the dataset is likely highly imbalanced, let's use a subset or handle it
print(f"Fraud ratio in train: {y_train.mean():.4f}")

# Training a Random Forest Classifier
print("Training Random Forest model (using a subset for efficiency)...")
# Using a smaller sample of the training data to ensure it finishes quickly in this environment
train_sample = train_df.sample(n=100000, random_state=42)
X_train_sample = train_sample[features].copy()
y_train_sample = train_sample[target]

for col in ['category', 'gender']:
    X_train_sample[col] = le.fit_transform(X_train_sample[col])

rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train_sample, y_train_sample)

# Predictions
print("Making predictions...")
y_pred = rf_model.predict(X_test)

# Evaluation
print("\nEvaluation Results:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save results
output_sample = test_df.head(20).copy()
output_sample['PREDICTED_FRAUD'] = y_pred[:20]
output_sample.to_csv("/home/ubuntu/task_folder/task2_results_sample.csv")
print("\nSample results saved to task2_results_sample.csv")
