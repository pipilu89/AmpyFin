import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score
from helper_files.client_helper import strategies
import warnings

warnings.filterwarnings('ignore')

# Load the data
trades_data = pd.read_csv('./results/10year_sp500_trades.csv')

# Create the 'return' column efficiently
trades_data['return'] = (trades_data['ratio'] > 1).astype(int)

# Create 'strategy_index' column efficiently
strategy_to_integer = {strategy.__name__: i for i, strategy in enumerate(strategies)}
trades_data['strategy_index'] = trades_data['strategy'].map(strategy_to_integer)

# Features and target variable
X = trades_data[['strategy_index', 'current_vix']]
y = trades_data['return']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize RandomForestClassifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1) #added n_jobs

# Fit the classifier to the training data
rf_classifier.fit(X_train, y_train)

# Make predictions
y_pred = rf_classifier.predict(X_test)

# Calculate and print metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print("\nClassification Report:\n", classification_rep)

# Sample prediction
sample = X_test.iloc[[0]]

prediction = rf_classifier.predict(sample)

print(f"\nSample: \n{sample}")
print(f"\nSample Values: {sample.iloc[0].to_dict()}")
print(f"Predicted Return: {prediction[0]}")
