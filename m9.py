import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFECV, VarianceThreshold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt
import numpy as np


# Load the Pima Indians Diabetes dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
data = pd.read_csv(url, header=None, names=columns)

X = data.drop('Outcome', axis=1)  
y = data['Outcome']  


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model 1: Train with all features

model_all = DecisionTreeClassifier(random_state=42)
model_all.fit(X_train, y_train)
y_pred_all = model_all.predict(X_test)

accuracy_all = accuracy_score(y_test, y_pred_all)
print("Accuracy (All Features): ", accuracy_all)

print("\n")

# filter1: Variance Threshold
thresholds = np.linspace(0.01, 0.5, 50)  

best_threshold = None
best_accuracy = 0

for threshold in thresholds:
    var_thresh = VarianceThreshold(threshold=threshold)
    X_train_var = var_thresh.fit_transform(X_train)
    X_test_var = var_thresh.transform(X_test)
    
    model_var_thresh = DecisionTreeClassifier(random_state=42)
    model_var_thresh.fit(X_train_var, y_train)
    
    y_pred_var_thresh = model_var_thresh.predict(X_test_var)
    accuracy_var_thresh = accuracy_score(y_test, y_pred_var_thresh)
     
    if accuracy_var_thresh > best_accuracy:
        best_accuracy = accuracy_var_thresh
        best_threshold = threshold

print(f"Best Threshold: {best_threshold:.2f}")
print(f"Best Accuracy: {best_accuracy}")
print("\n")


# filter2: Correlation-Based Feature Selection 

correlation_matrix = X_train.corr()  
correlation_threshold = 0.4
corr_features = set()

for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > correlation_threshold:
            corr_features.add(correlation_matrix.columns[i])

X_train_corr = X_train.drop(columns=list(corr_features))
X_test_corr = X_test.drop(columns=list(corr_features))

X_train_corr_scaled = scaler.fit_transform(X_train_corr)
X_test_corr_scaled = scaler.transform(X_test_corr)

model_corr = DecisionTreeClassifier(random_state=42)
model_corr.fit(X_train_corr_scaled, y_train)
y_pred_corr = model_corr.predict(X_test_corr_scaled)

selected_features_corr = X_train_corr.columns
print("Selected Features (Correlation-Based):", selected_features_corr)

accuracy_corr = accuracy_score(y_test, y_pred_corr)
print("Accuracy (Correlation-Based Selected Features): ", accuracy_corr)

print("\n")


# wrapping: Recursive Feature Elimination (RFE)
model_rfe = DecisionTreeClassifier(random_state=42)
rfe = RFE(estimator=model_rfe, n_features_to_select=5)
rfe.fit(X_train, y_train)

X_train_rfe = rfe.transform(X_train)
X_test_rfe = rfe.transform(X_test)

model_rfe.fit(X_train_rfe, y_train)
y_pred_rfe = model_rfe.predict(X_test_rfe)

selected_features_rfe = X.columns[rfe.get_support()]
print("Selected Features (RFE):", selected_features_rfe)

accuracy_rfe = accuracy_score(y_test, y_pred_rfe)
print("Accuracy (RFE Selected Features): ", accuracy_rfe)

print("\n")


# Model 5: Lasso Regularization
lasso = LassoCV(cv=5, random_state=42)
lasso.fit(X_train_scaled, y_train)

lasso_coefficients = lasso.coef_
important_features_lasso = [i for i, coef in enumerate(lasso_coefficients) if coef != 0]

X_train_lasso = X_train_scaled[:, important_features_lasso]
X_test_lasso = X_test_scaled[:, important_features_lasso]

model_lasso = DecisionTreeClassifier(random_state=42)
model_lasso.fit(X_train_lasso, y_train)
y_pred_lasso = model_lasso.predict(X_test_lasso)

selected_features_lasso = X.columns[important_features_lasso]
print("Selected Features (Lasso Regularization):", selected_features_lasso)


accuracy_lasso = accuracy_score(y_test, y_pred_lasso)
print("Accuracy (Lasso Regularization - Selected Features): ", accuracy_lasso)

print("\n")

