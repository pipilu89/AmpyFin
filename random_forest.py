import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_score,
    recall_score,
)
from helper_files.client_helper import strategies
import warnings

warnings.filterwarnings("ignore")


def train_random_forest_classifier(trades_data):
    """
    Trains a RandomForestClassifier on the given trades data.

    Args:
        trades_data (pd.DataFrame): DataFrame containing trade data with 'ratio', 'current_vix', and other features.

    Returns:
        tuple: A tuple containing the trained RandomForestClassifier, accuracy, precision, and recall.
    """
    # Create the 'return' column where 1 = +ve return, 0 = neutral or -ve.
    trades_data["return"] = np.where(trades_data["ratio"] > 1, 1, 0)

    # Features and target variable
    X = trades_data[
        ["^VIX", "One_day_spy_return"]
    ]  # Using only 'current_vix' as a feature
    y = trades_data["return"]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Initialize RandomForestClassifier
    rf_classifier = RandomForestClassifier(
        n_estimators=100, random_state=42, n_jobs=-1
    )  # added n_jobs

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


def train_random_forest_classifier_RandomizedSearchCV(
    trades_data,
    param_dist={"n_estimators": randint(50, 500), "max_depth": randint(1, 20)},
):
    """
    Trains a RandomForestClassifier on the given trades data.

    Args:
        trades_data (pd.DataFrame): DataFrame containing trade data with 'ratio', 'current_vix', and other features.

    Returns:
        tuple: A tuple containing the trained RandomForestClassifier, accuracy, precision, and recall.
    """
    # Create the 'return' column where 1 = +ve return, 0 = neutral or -ve.
    trades_data["return"] = np.where(trades_data["ratio"] > 1, 1, 0)

    # Features and target variable
    X = trades_data[
        ["^VIX", "One_day_spy_return"]
    ]  # Using only 'current_vix' as a feature
    y = trades_data["return"]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Hyperparameter Tuning

    # Create a random forest classifier
    rf = RandomForestClassifier()

    # Use random search to find the best hyperparameters
    rand_search = RandomizedSearchCV(
        rf, param_distributions=param_dist, n_iter=10, cv=5, n_jobs=-1
    )

    # Fit the random search object to the data
    rand_search.fit(X_train, y_train)

    # Create a variable for the best model
    best_rf = rand_search.best_estimator_

    # Print the best hyperparameters
    print("Best hyperparameters:", rand_search.best_params_)

    # Generate predictions with the best model
    y_pred = best_rf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    return best_rf, accuracy, precision, recall


def predict_random_forest_classifier(rf_classifier, sample_df):
    """
    Predicts the return using the given RandomForestClassifier and sample data.

    Args:
        rf_classifier (RandomForestClassifier): The trained RandomForestClassifier.
        sample_df (pd.DataFrame): DataFrame containing the sample data for prediction.
                                  It should contain the features the model was trained on.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: An array of predicted classes (0 or 1) for each sample.
            - np.ndarray: An array of probabilities of predicting class 1 for each sample.
    """
    # Ensure sample_df contains the necessary feature columns used during training
    required_features = rf_classifier.feature_names_in_
    if not all(feature in sample_df.columns for feature in required_features):
        missing = set(required_features) - set(sample_df.columns)
        raise ValueError(f"Input DataFrame is missing required features: {missing}")

    # Get predictions for all samples directly
    predictions = rf_classifier.predict(
        sample_df[required_features]
    )  # Use only required features

    # Get probabilities for all samples
    probabilities = rf_classifier.predict_proba(
        sample_df[required_features]
    )  # Use only required features

    # Extract probability of class 1 for all samples
    if probabilities.shape[1] == 2:
        # Find the column index corresponding to class 1
        try:
            class_1_index = np.where(rf_classifier.classes_ == 1)[0][0]
            prob_class_1 = probabilities[:, class_1_index]
        except IndexError:
            # Handle case where class 1 was not present during training (should be caught by shape[1]==1 ideally)
            prob_class_1 = np.zeros(probabilities.shape[0])

    elif probabilities.shape[1] == 1:
        # Handle single-class case (model always predicts the same class)
        known_class = rf_classifier.classes_[0]
        if known_class == 1:
            # The single column is the probability of class 1
            prob_class_1 = probabilities[:, 0]
        else:  # known_class == 0
            # The single column is the probability of class 0, so prob_class_1 is 0 for all samples
            prob_class_1 = np.zeros(probabilities.shape[0])
    else:
        # Unexpected shape
        raise ValueError(
            f"Unexpected shape for probabilities array: {probabilities.shape}"
        )

    return predictions, prob_class_1  # Return arrays


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

    if "strategy" not in trades_data_df.columns:
        raise ValueError("The DataFrame must contain a 'strategy' column.")

    min_rows = 5
    strategy_counts = trades_data_df["strategy"].value_counts()
    strategies_with_enough_data = strategy_counts[
        strategy_counts >= min_rows
    ].index.tolist()

    logger.info(f"{len(strategies_with_enough_data) = }")
    logger.info(f"{strategies_with_enough_data = }")

    for strategy_name in strategies_with_enough_data:
        try:
            strategy_data = trades_data_df[trades_data_df["strategy"] == strategy_name]
            logger.info(
                f"Training classifier for strategy {strategy_name}. {len(strategy_data) = }"
            )
            # logger.info(f"{strategy_data.head()}")
            rf_classifier, accuracy, precision, recall = train_random_forest_classifier(
                strategy_data
            )
            rf_classifiers[strategy_name] = {
                "rf_classifier": rf_classifier,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
            }
            logger.info(
                f"Classifier for strategy {strategy_name} trained successfully."
            )
        except Exception as e:
            logger.error(f"Error training classifier for strategy {strategy_name}: {e}")

    return rf_classifiers, strategies_with_enough_data


import logging

if __name__ == "__main__":
    # Basic logger setup for standalone execution
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Load the trades data
    trades_data_df = pd.read_csv("./results/10year_sp500_trades.csv")

    # Train and store classifiers
    trained_classifiers, _ = train_and_store_classifiers(
        trades_data_df, logger
    )  # Added logger and ignored second return value

    # logger.info(f"{trained_classifiers}")

    # Example usage: Make a prediction for a specific strategy
    sample_data = {"current_vix": [34.27]}
    sample_df = pd.DataFrame(sample_data, index=[0])

    for strategy_name, classifier_data in trained_classifiers.items():
        rf_classifier = classifier_data["rf_classifier"]
        accuracy = classifier_data["accuracy"]
        precision = classifier_data["precision"]
        recall = classifier_data["recall"]

        prediction, probability = predict_random_forest_classifier(
            rf_classifier, sample_df
        )
        # logger.info(
        #     f"\nFinal Prediction for {strategy_name}: {prediction} (Prob: {probability:.2f}), accuracy = {accuracy:.2f}, precision = {precision:.2f}, recall = {recall:.2f}"
        # )
