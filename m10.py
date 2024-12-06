import pandas as pd
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
import numpy as np

# Load the Pima Indians Diabetes dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
df = pd.read_csv(url, header=None, names=columns)

# Split into features and target manually
X = df.drop(columns=['Outcome'])  
y = df['Outcome']  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initial Decision Tree
initial_dt = DecisionTreeClassifier(random_state=42)
initial_dt.fit(X_train, y_train)

y_pred_initial = initial_dt.predict(X_test)
accuracy_initial_dt = accuracy_score(y_test, y_pred_initial)
print(f"Initial Decision Tree Accuracy : {accuracy_initial_dt}")

# PCA pipeline
pca_pipeline = Pipeline([
    ('pca', PCA()),  
    ('dt', DecisionTreeClassifier(random_state=42)) 
])

pca_param_grid = {
    'pca__n_components': [2, 3, 4, 5, 6, 7],  
    'dt__max_depth': [2, 5, 7, 10, None]  
}

pca_grid_search = GridSearchCV(pca_pipeline, pca_param_grid, cv=5, scoring='accuracy')
pca_grid_search.fit(X_train, y_train)

best_pca_components = pca_grid_search.best_params_['pca__n_components']
best_pca_accuracy = pca_grid_search.best_score_

print(f"Best PCA n_components: {best_pca_components}")

# Factor Analysis pipeline
fa_pipeline = Pipeline([
    ('fa', FactorAnalysis()),  
    ('dt', DecisionTreeClassifier(random_state=42))  
])

fa_param_grid = {
    'fa__n_components': [2, 3, 4, 5, 6, 7], 
    'dt__max_depth': [2, 5, 7, 10, None]  
}

fa_grid_search = GridSearchCV(fa_pipeline, fa_param_grid, cv=5, scoring='accuracy')
fa_grid_search.fit(X_train, y_train)

best_fa_components = fa_grid_search.best_params_['fa__n_components']
best_fa_accuracy = fa_grid_search.best_score_

print(f"Best FA n_components: {best_fa_components}")

# PCA transformation and evaluation
pca = PCA(n_components=best_pca_components)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

dt_pca = DecisionTreeClassifier(random_state=42)
dt_pca.fit(X_train_pca, y_train)
y_pred_pca = dt_pca.predict(X_test_pca)
accuracy_pca = accuracy_score(y_test, y_pred_pca)
print("PCA Accuracy on test set:", accuracy_pca)

# Get the top contributing feature names from PCA
pca_components_sorted = np.argsort(np.abs(pca.components_), axis=1)[:, ::-1]
best_pca_features = [columns[i] for i in pca_components_sorted[0][:best_pca_components]]
print("\nBest PCA Features based on component weights:", best_pca_features)

# Confusion matrix for PCA
conf_matrix_pca = confusion_matrix(y_test, y_pred_pca)
print("\nConfusion Matrix for PCA:")
print(conf_matrix_pca)

# Factor Analysis transformation and evaluation
fa = FactorAnalysis(n_components=best_fa_components)
X_train_fa = fa.fit_transform(X_train)
X_test_fa = fa.transform(X_test)

dt_fa = DecisionTreeClassifier(random_state=42)
dt_fa.fit(X_train_fa, y_train)
y_pred_fa = dt_fa.predict(X_test_fa)
accuracy_fa = accuracy_score(y_test, y_pred_fa)
print("FA Accuracy on test set:", accuracy_fa)

# Get the top contributing feature names from FA
fa_components_sorted = np.argsort(np.abs(fa.components_), axis=1)[:, ::-1]
best_fa_features = [columns[i] for i in fa_components_sorted[0][:best_fa_components]]
print("\nBest FA Features based on component weights:", best_fa_features)

# Confusion matrix for FA
conf_matrix_fa = confusion_matrix(y_test, y_pred_fa)
print("\nConfusion Matrix for FA:")
print(conf_matrix_fa)

if accuracy_initial_dt > max(accuracy_pca, accuracy_fa):
    print(f"Initial DecisionTree performs better with an accuracy of: {accuracy_initial_dt}")
elif accuracy_pca > accuracy_fa:
    print(f"PCA performs better with an accuracy of: {accuracy_pca}")
else:
    print(f"FA performs better with an accuracy of: {accuracy_fa}")
