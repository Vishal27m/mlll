# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load the dataset from the URL
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
data = pd.read_csv(url, header=None, names=columns)


imputer = SimpleImputer(strategy='mean')
data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

X = data_imputed.drop('Outcome', axis=1)
y = data_imputed['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


k_values = range(1, 11)
accuracies = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    cv_scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy')
    accuracies.append(cv_scores.mean())

# Plotting the results
plt.plot(k_values, accuracies, marker='o')
plt.title('KNN Classifier: Varying Number of Neighbors')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Cross-Validation Accuracy')
plt.xticks(k_values)
plt.grid(True)
plt.show()

# Determine the best k value based on the highest accuracy
best_k = k_values[np.argmax(accuracies)]
best_accuracy = max(accuracies)

print(f"The best k value is {best_k} with an accuracy of {best_accuracy:.4f}")

# Train the KNN model with the best k value
knn_best = KNeighborsClassifier(n_neighbors=best_k)
knn_best.fit(X_train, y_train)

# Predict on the test set
y_pred = knn_best.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of KNN with k={best_k}: {accuracy:.4f}")

# Function to get user input for prediction
def get_user_input():
    print("Enter the following values for prediction:")
    Pregnancies = float(input("Pregnancies: "))
    Glucose = float(input("Glucose: "))
    BloodPressure = float(input("BloodPressure: "))
    SkinThickness = float(input("SkinThickness: "))
    Insulin = float(input("Insulin: "))
    BMI = float(input("BMI: "))
    DiabetesPedigreeFunction = float(input("DiabetesPedigreeFunction: "))
    Age = float(input("Age: "))

    return [[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]]

new_data_point = get_user_input()

new_data_point_df = pd.DataFrame(new_data_point, columns=X.columns)

predicted_class = knn_best.predict(new_data_point_df)
print(f"The predicted class for the new data point is: {predicted_class[0]}")

# Visualize the original data and the new data point
plt.figure(figsize=(10, 6))

# Plotting the original data points
for class_value in np.unique(y):
    plt.scatter(X['Glucose'][y == class_value],
                X['BMI'][y == class_value],
                label=f'Class {class_value}', alpha=0.6)

# Plotting the new data point
plt.scatter(new_data_point_df['Glucose'], new_data_point_df['BMI'], color='red', marker='x', s=100, label='New Data Point')

# Adding labels and title
plt.xlabel('Glucose')
plt.ylabel('BMI')
plt.title('KNN Classification with New Data Point')
plt.legend()
plt.grid(True)
plt.show()
