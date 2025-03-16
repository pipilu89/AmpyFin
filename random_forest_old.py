from venv import logger
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score
from helper_files.client_helper import strategies
import warnings

warnings.filterwarnings('ignore')

def train_random_forest_classifier(trades_data):
  # Load the data
  # trades_data = pd.read_csv('./results/10year_sp500_trades.csv')

  # Create the 'return' column efficiently
  # trades_data['return'] = (trades_data['ratio'] > 1).astype(int)
  trades_data['return'] = np.where(trades_data['ratio'] > 1, 1, 0)

  # Create 'strategy_index' column efficiently
  # strategy_to_integer = {strategy.__name__: i for i, strategy in enumerate(strategies)}
  # trades_data['strategy_index'] = trades_data['strategy'].map(strategy_to_integer)

  # print(trades_data)

  # Features and target variable
  # X = trades_data[['strategy_index', 'current_vix']]
  X = trades_data[['current_vix']]
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

  # print(f"Accuracy: {accuracy:.2f}")
  # print(f"Precision: {precision:.2f}")
  # print(f"Recall: {recall:.2f}")
  # print("\nClassification Report:\n", classification_rep)
  return rf_classifier, accuracy, precision, recall



def predict_random_forest_classifier(rf_classifier, sample_df):
  # Sample prediction
  # sample = X_test.iloc[[0]]

  prediction = rf_classifier.predict(sample_df)

  # print(f"\nSample: \n{sample}")
  # print(f"\nSample Values: {sample.iloc[0].to_dict()}")
  # print(f"Predicted Return: {prediction[0]}")

  return prediction[0]

if __name__ == "__main__":
  # train model
  trades_data_df = pd.read_csv('./results/10year_sp500_trades.csv')
      
  rf_classifiers = {}
  trades_data_df = pd.read_csv('./results/10year_sp500_trades.csv')

  if 'strategy' not in trades_data_df.columns:
      raise ValueError("The DataFrame must contain a 'strategy' column.")

  min_rows = 5
  strategy_counts = trades_data_df['strategy'].value_counts()
  strategies_with_enough_data = strategy_counts[strategy_counts >= min_rows].index.tolist()
    
  print(f"{strategies_with_enough_data = }")

  for strategy in strategies:
      strategy_name = strategy.__name__
      if strategy_name in strategies_with_enough_data:
        rf_classifier, accuracy, precision, recall = train_random_forest_classifier(trades_data_df[trades_data_df['strategy'] == strategy_name])
        rf_classifiers[strategy_name] = rf_classifier
        # rf_classifiers[strategy_name]['rf_classifier'] = rf_classifier
        # rf_classifiers[strategy_name]['accuracy'] = accuracy
        # rf_classifiers[strategy_name]['precision'] = precision
        # rf_classifiers[strategy_name]['recall'] = recall
      else:
        #  logger.warning(f"strategy {strategy_name} does not have enough data")
        ...

  print(rf_classifiers)

  # print(trades_data_df[trades_data_df['strategy'] == 'HT_TRENDLINE_indicator'])


  # rf_classifier, accuracy, precision, recall = train_random_forest_classifier(trades_data_df)
  # print(f"{rf_classifier = }")

  # Method 1: Using a dictionary and specifying the index
  data = {'strategy_index': [100], 'current_vix': [34.27]}
  data = {'current_vix': [34.27]}
  sample_df = pd.DataFrame(data, index=[0])
  print("\n", sample_df)

  # Method 2: Using a list of lists (or tuples)
  # data = [[8, 14.27]]
  # sample_df = pd.DataFrame(data, columns=['strategy_index', 'current_vix'])

  # Method 3: Using a list of dictionaries
  # data = [{'strategy_index': 8, 'current_vix': 14.27}]
  # sample_df = pd.DataFrame(data)

  # Make prediction
  for strategy in strategies_with_enough_data:
      strategy_name = strategy
      prediction = predict_random_forest_classifier(rf_classifiers[strategy_name], sample_df)
      print(f"\nFinal Prediction {strategy_name}: {prediction}, {accuracy = }, {precision = }, {recall = }")