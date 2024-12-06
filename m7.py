import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb

# Load the dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
df = pd.read_csv(url, header=None, names=columns)

# Data preprocessing
df = df.drop_duplicates()
df = df.dropna()

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
print(f"Accuracy: {accuracy:.2f}")

# Print the classification report
report = classification_report(y_test, y_pred, target_names=["No Diabetes", "Diabetes"])
print("Classification Report:")
print(report)

# Print the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# 1. AdaBoost with Decision Tree
ada_clf = AdaBoostClassifier(estimator=clf, n_estimators=100, random_state=42)
ada_clf.fit(X_train, y_train)
y_pred_ada = ada_clf.predict(X_test)

print("\nAdaBoost with Decision Tree Performance:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_ada):.2f}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_ada))
print("Classification Report:")
print(classification_report(y_test, y_pred_ada))

# 2. Gradient Boosting
gb_clf = GradientBoostingClassifier(n_estimators=100, max_depth=best_params['max_depth'], random_state=42)
gb_clf.fit(X_train, y_train)
y_pred_gb = gb_clf.predict(X_test)

print("\nGradient Boosting Classifier Performance:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_gb):.2f}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_gb))
print("Classification Report:")
print(classification_report(y_test, y_pred_gb))

# 3. XGBoost
xgb_clf = xgb.XGBClassifier(n_estimators=100, max_depth=best_params['max_depth'], random_state=42)
xgb_clf.fit(X_train, y_train)
y_pred_xgb = xgb_clf.predict(X_test)

print("\nXGBoost Classifier Performance:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_xgb):.2f}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_xgb))
print("Classification Report:")
print(classification_report(y_test, y_pred_xgb))


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
user_prediction_ada = ada_clf.predict(user_df)
print("\nAdaBoost Classifier Prediction:", "Diabetes" if user_prediction_ada[0] == 1 else "No Diabetes")

user_prediction_gb = gb_clf.predict(user_df)
print("Gradient Boosting Classifier Prediction:", "Diabetes" if user_prediction_gb[0] == 1 else "No Diabetes")

user_prediction_xgb = xgb_clf.predict(user_df)
print("XGBoost Classifier Prediction:", "Diabetes" if user_prediction_xgb[0] == 1 else "No Diabetes")
