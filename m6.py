import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
df = pd.read_csv(url, header=None, names=columns)

# Data preprocessing
df = df.drop_duplicates()
df = df.dropna()
df.ffill(inplace=True)
# Define features and target variable
X = df.drop(columns=['Outcome'])
y = df['Outcome']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define the parameter grid for Decision Tree
param_grid = {
    'max_depth': np.arange(1, 20)
}

# Create a GridSearchCV object
grid_search = GridSearchCV(estimator=DecisionTreeClassifier(random_state=42), param_grid=param_grid, cv=5, scoring='accuracy')

# Fit the grid search to the data
grid_search.fit(X_train, y_train)

# Get the best parameters and the best score
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print("Best Parameters:", best_params)
print("Best Cross-Validation Accuracy:", best_score)

# Train the Decision Tree with the best parameters
clf = DecisionTreeClassifier(**best_params, random_state=42)
clf.fit(X_train, y_train)

# Predict on the test set
y_pred = clf.predict(X_test)

# Calculate the accuracy score
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy with best params: {accuracy:.2f}")

# Print the classification report
report = classification_report(y_test, y_pred, target_names=["No Diabetes", "Diabetes"])
print("Classification Report:")
print(report)

# Print the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)


# Now apply Bagging with the tuned Decision Tree
bagging_clf = BaggingClassifier(estimator=clf, n_estimators=100, random_state=42, n_jobs=-1)
bagging_clf.fit(X_train, y_train)
y_pred_bagging = bagging_clf.predict(X_test)

# Evaluate Bagging Classifier with Decision Tree
print("\nBagging with Decision Tree Performance:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_bagging):.2f}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_bagging))
print("Classification Report:")
print(classification_report(y_test, y_pred_bagging))

# Predefined user input for prediction
user_input = {
    'Pregnancies': 2,
    'Glucose': 85,
    'BloodPressure': 75,
    'SkinThickness': 30,
    'Insulin': 90,
    'BMI': 28.1,
    'DiabetesPedigreeFunction': 0.5,
    'Age': 25
}

# Convert user input to DataFrame
user_df = pd.DataFrame([user_input])6

# Ensure user_df has the same columns as X
user_df = user_df.reindex(columns=X.columns, fill_value=0)

# Predict based on user input with consistent column names
user_prediction = clf.predict(user_df)
print("Decision Tree Prediction:", "Diabetes" if user_prediction[0] == 1 else "No Diabetes")

user_prediction_bagging = bagging_clf.predict(user_df)
print("Bagging Classifier Prediction:", "Diabetes" if user_prediction_bagging[0] == 1 else "No Diabetes")
