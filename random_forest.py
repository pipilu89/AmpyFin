import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score
from helper_files.client_helper import strategies
import warnings

warnings.filterwarnings('ignore')

def train_random_forest_classifier(trades_data):
    """
    Trains a RandomForestClassifier on the given trades data.

    Args:
        trades_data (pd.DataFrame): DataFrame containing trade data with 'ratio', 'current_vix', and other features.

    Returns:
        tuple: A tuple containing the trained RandomForestClassifier, accuracy, precision, and recall.
    """
    # Create the 'return' column where 1 = +ve return, 0 = neutral or -ve.
    trades_data['return'] = np.where(trades_data['ratio'] > 1, 1, 0)

    # Features and target variable
    X = trades_data[['current_vix']]  # Using only 'current_vix' as a feature
    y = trades_data['return']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize RandomForestClassifier
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)  # added n_jobs

    # Fit the classifier to the training data
    rf_classifier.fit(X_train, y_train)

    # Make predictions
    y_pred = rf_classifier.predict(X_test)

    # Calculate and print metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    # classification_rep = classification_report(y_test, y_pred)

    return rf_classifier, accuracy, precision, recall


def predict_random_forest_classifier(rf_classifier, sample_df):
    """
    Predicts the return using the given RandomForestClassifier and sample data.

    Args:
        rf_classifier (RandomForestClassifier): The trained RandomForestClassifier.
        sample_df (pd.DataFrame): DataFrame containing the sample data for prediction.

    Returns:
        int: The predicted return (0 or 1). 1 is positive return, 0 is -ve return.
    """
    prediction = rf_classifier.predict(sample_df)
    return prediction[0]


def train_and_store_classifiers(trades_data_df, logger):
    """
    Trains a RandomForestClassifier for each strategy and stores them in a dictionary.

    Args:
        trades_data_df (pd.DataFrame): DataFrame containing trade data with 'strategy', 'ratio', 'current_vix', etc.

    Returns:
        dict: A dictionary where keys are strategy names and values are dictionaries containing the trained
              RandomForestClassifier, accuracy, precision, and recall for that strategy.
    """
    rf_classifiers = {}

    if 'strategy' not in trades_data_df.columns:
        raise ValueError("The DataFrame must contain a 'strategy' column.")

    min_rows = 5
    strategy_counts = trades_data_df['strategy'].value_counts()
    strategies_with_enough_data = strategy_counts[strategy_counts >= min_rows].index.tolist()

    logger.info(f"{strategies_with_enough_data = }")

    for strategy_name in strategies_with_enough_data:
        try:
            logger.info(f"Training classifier for strategy {strategy_name}")
            strategy_data = trades_data_df[trades_data_df['strategy'] == strategy_name]
            logger.info(f"{strategy_data.head()}")
            rf_classifier, accuracy, precision, recall = train_random_forest_classifier(strategy_data)
            rf_classifiers[strategy_name] = {
                'rf_classifier': rf_classifier,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall
            }
            logger.info(f"Classifier for strategy {strategy_name} trained successfully.")
        except Exception as e:
            print(f"Error training classifier for strategy {strategy_name}: {e}")

    return rf_classifiers, strategies_with_enough_data


if __name__ == "__main__":
    # Load the trades data
    trades_data_df = pd.read_csv('./results/10year_sp500_trades.csv')

    # Train and store classifiers
    trained_classifiers = train_and_store_classifiers(trades_data_df)

    print(f"{trained_classifiers}")

    # Example usage: Make a prediction for a specific strategy
    sample_data = {'current_vix': [34.27]}
    sample_df = pd.DataFrame(sample_data, index=[0])

    for strategy_name, classifier_data in trained_classifiers.items():
        rf_classifier = classifier_data['rf_classifier']
        accuracy = classifier_data['accuracy']
        precision = classifier_data['precision']
        recall = classifier_data['recall']

        prediction = predict_random_forest_classifier(rf_classifier, sample_df)
        print(f"\nFinal Prediction for {strategy_name}: {prediction}, accuracy = {accuracy:.2f}, precision = {precision:.2f}, recall = {recall:.2f}")
