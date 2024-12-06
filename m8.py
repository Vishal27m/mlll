import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
df = pd.read_csv(url, header=None, names=columns)

# Data preprocessing
df = df.drop_duplicates()
df = df.dropna()

# Split features and target
X = df.drop(columns=['Outcome'])
y = df['Outcome']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define the parameter grid for Random Forest
param_grid = {
    'n_estimators': [50, 100, 150, 200],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# Create a GridSearchCV object
grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42), param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)

# Fit the grid search to the data
grid_search.fit(X_train, y_train)

# Get the best parameters and the best score
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print("Best Parameters:", best_params)
print("Best Cross-Validation Accuracy:", best_score)

# Train the Random Forest with the best parameters
rf_clf = RandomForestClassifier(**best_params, random_state=42)
rf_clf.fit(X_train, y_train)

# Predict on the test set
y_pred_rf = rf_clf.predict(X_test)

# Calculate the accuracy score
accuracy = accuracy_score(y_test, y_pred_rf)
print(f"Accuracy: {accuracy:.2f}")

# Print the classification report
report = classification_report(y_test, y_pred_rf, target_names=["No Diabetes", "Diabetes"])
print("Classification Report:")
print(report)

# Print the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_rf)
print("Confusion Matrix:")
print(conf_matrix)

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

user_df = pd.DataFrame([user_input])
user_df = user_df.reindex(columns=X.columns, fill_value=0)

# Predict based on user input
user_prediction_rf = rf_clf.predict(user_df)
print("Random Forest Classifier Prediction:", "Diabetes" if user_prediction_rf[0] == 1 else "No Diabetes")
