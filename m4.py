import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
data = pd.read_csv(url, header=None, names=columns)

# Number of observations in each class
class_counts = data['Outcome'].value_counts()
print("\nClass distribution:\n", class_counts)

# Determine if it's a binary or multiclass classifier
if len(class_counts) == 2:
    print("\nThe classifier is binary.")
else:
    print("\nThe classifier is multiclass.")

# Check data types to identify categorical vs continuous features
print("\nData types:\n", data.dtypes)

# Preprocessing
# Continuous features
continuous_features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

# Split dataset into features and target
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Gaussian Naive Bayes for continuous features
scaler = StandardScaler()
X_train_cont = scaler.fit_transform(X_train[continuous_features])
X_test_cont = scaler.transform(X_test[continuous_features])

gnb = GaussianNB()
gnb.fit(X_train_cont, y_train)
y_pred_gnb = gnb.predict(X_test_cont)

print("\nGaussian Naive Bayes (Continuous Features)")
print("Accuracy:", accuracy_score(y_test, y_pred_gnb))
print(classification_report(y_test, y_pred_gnb))

# Multinomial Naive Bayes for all features
mnb = MultinomialNB()
mnb.fit(X_train, y_train)
y_pred_mnb = mnb.predict(X_test)

print("\nMultinomial Naive Bayes (Categorical Features)")
print("Accuracy:", accuracy_score(y_test, y_pred_mnb))
print(classification_report(y_test, y_pred_mnb))

# Bernoulli Naive Bayes for all features, interpreting them as binary
bnb = BernoulliNB()
bnb.fit(X_train, y_train)
y_pred_bnb = bnb.predict(X_test)

print("\nBernoulli Naive Bayes (Binary Features)")
print("Accuracy:", accuracy_score(y_test, y_pred_bnb))
print(classification_report(y_test, y_pred_bnb))

# Calculate the correlation matrix
correlation_matrix = data.corr()

# Plot the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title("Correlation Matrix of Dataset")
plt.show()

# Select features with correlation values closest to 0 with the target variable
selected_features = ['Glucose','SkinThickness', 'Pregnancies', 'DiabetesPedigreeFunction']

# Split dataset into selected features and target
X_selected = data[selected_features]
y_selected = data['Outcome']

# Split into train and test sets for selected features
X_train_sel, X_test_sel, y_train_sel, y_test_sel = train_test_split(X_selected, y_selected, test_size=0.3, random_state=42)

# Gaussian Naive Bayes for selected features
scaler_sel = StandardScaler()
X_train_sel_scaled = scaler_sel.fit_transform(X_train_sel)
X_test_sel_scaled = scaler_sel.transform(X_test_sel)

gnb_sel = GaussianNB()
gnb_sel.fit(X_train_sel_scaled, y_train_sel)
y_pred_gnb_sel = gnb_sel.predict(X_test_sel_scaled)

print("\nGaussian Naive Bayes (Selected Continuous Features)")
print("Accuracy:", accuracy_score(y_test_sel, y_pred_gnb_sel))
print(classification_report(y_test_sel, y_pred_gnb_sel))

# Confusion matrix for Gaussian Naive Bayes (Selected Features)
conf_matrix_sel = confusion_matrix(y_test_sel, y_pred_gnb_sel)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix_sel, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Gaussian Naive Bayes (Selected Features)")
plt.show()

# Get user input for selected features
user_input = []

print("Enter values for the following features:")
for feature in selected_features:
    value = float(input(feature + ": "))
    user_input.append(value)

# Convert user input to numpy array and reshape
user_input = np.array(user_input).reshape(1, -1)

# Scale user input
user_input_scaled = scaler_sel.transform(user_input)

# Predict outcome using Gaussian Naive Bayes
predicted_outcome = gnb_sel.predict(user_input_scaled)

print("Predicted outcome:", predicted_outcome)
