import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
import warnings
warnings.filterwarnings('ignore')

# Corrected URL for the dataset
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
titanic_data = pd.read_csv(url)
# titanic_data = pd.read_excel('random-trade-data.xlsx', index_col=0)
titanic_data = pd.read_excel('random-trade-data.xlsx')
# titanic_data

# Drop rows with missing 'Survived' values
# titanic_data = titanic_data.dropna(subset=['Survived'])

# Features and target variable
# X = titanic_data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]
X = titanic_data[['strategy', 'vix', 'sp500']]
y = titanic_data['return']

# Encode 'Sex' column
# X.loc[:, 'Sex'] = X['Sex'].map({'female': 0, 'male': 1})

# Fill missing 'Age' values with the median
# X.loc[:, 'Age'].fillna(X['Age'].median(), inplace=True)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize RandomForestClassifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Fit the classifier to the training data
rf_classifier.fit(X_train, y_train)

# Make predictions
y_pred = rf_classifier.predict(X_test)

# Calculate accuracy and classification report
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# Print the results
print(f"Accuracy: {accuracy:.2f}")
print(f"precision: {precision:.2f}")
print("Recall:", recall)
print("\nClassification Report:\n", classification_rep)

# Sample prediction
# sample = X_test.iloc[0:1]  # Keep as DataFrame to match model input format
data = {'row_1': ["0","50","0"]}
sample = pd.DataFrame.from_dict(data, orient='index', columns=['strategy', 'vix', 'sp500'])

prediction = rf_classifier.predict(sample)
print(f"\nsample {sample}")
# print(f"{prediction = }")

# Retrieve and display the sample
sample_dict = sample.iloc[0].to_dict()
print(f"\nSample Passenger: {sample_dict}")
# print(f"Predicted return: {'Survived' if prediction[0] == 1 else 'Did Not Survive'}")
print(f"Predicted return: {prediction[0] = }")
