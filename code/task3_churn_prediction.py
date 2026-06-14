import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score
import os

# Define path
data_path = "/home/ubuntu/task_folder/task3/Churn_Modelling.csv"

print("Loading data...")
df = pd.read_csv(data_path)

# Data Preprocessing
print("Preprocessing data...")
# Drop irrelevant columns
df = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)

# Encode categorical variables
le = LabelEncoder()
df['Geography'] = le.fit_transform(df['Geography'])
df['Gender'] = le.fit_transform(df['Gender'])

# Define features and target
X = df.drop('Exited', axis=1)
y = df['Exited']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model Training
print("Training Gradient Boosting model...")
gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gb_model.fit(X_train, y_train)

# Predictions
print("Making predictions...")
y_pred = gb_model.predict(X_test)

# Evaluation
print("\nEvaluation Results:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save sample results
results_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}).head(20)
results_df.to_csv("/home/ubuntu/task_folder/task3_results_sample.csv", index=False)
print("\nSample results saved to task3_results_sample.csv")
