import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn import tree
import matplotlib.pyplot as plt

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
data = pd.read_csv(url, header=None, names=columns)

X = data.drop('Outcome', axis=1)
y = data['Outcome']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

clf = DecisionTreeClassifier(max_depth=2, random_state=42)
clf.fit(X_train, y_train)


y_pred = clf.predict(X_test)

print("Initial Accuracy:", accuracy_score(y_test, y_pred))
print("Initial Classification Report:\n", classification_report(y_test, y_pred))

plt.figure(figsize=(20,12))
tree.plot_tree(clf, feature_names=X.columns, class_names=['No Diabetes', 'Diabetes'], filled=True, rounded=True, fontsize=8)
plt.title("Initial Decision Tree")
plt.savefig("initial_decision_tree.png")  
plt.show()


param_grid = {
    'max_depth': np.arange(1, 20)
}
grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5, n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)


best_params = grid_search.best_params_
print("Best parameters:", best_params)

clf_optimized = DecisionTreeClassifier(**best_params, random_state=42)
clf_optimized.fit(X_train, y_train)

y_pred_optimized = clf_optimized.predict(X_test)

print("Optimized Accuracy:", accuracy_score(y_test, y_pred_optimized))
print("Optimized Classification Report:\n", classification_report(y_test, y_pred_optimized))

plt.figure(figsize=(20,12))
tree.plot_tree(clf_optimized, feature_names=X.columns, class_names=['No Diabetes', 'Diabetes'], filled=True, rounded=True, fontsize=8)
plt.title("Optimized Decision Tree")
plt.savefig("optimized_decision_tree.png")  
plt.show()

# Confusion Matrix for the optimized model
cm = confusion_matrix(y_test, y_pred_optimized)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Diabetes', 'Diabetes'])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix - Optimized Model")
plt.show()

mean_test_scores = grid_search.cv_results_['mean_test_score']
param_depths = [params['max_depth'] for params in grid_search.cv_results_['params']]

for depth, score in zip(param_depths, mean_test_scores):
    print(f"Max Depth: {depth}, Mean Test Score: {score:.4f}")

depth_scores = {depth: mean_test_scores[param_depths.index(depth)] for depth in set(param_depths)}
depth_values = sorted(depth_scores.keys())
scores = [depth_scores[depth] for depth in depth_values]

plt.figure(figsize=(10, 6))
plt.plot(depth_values, scores, marker='o', linestyle='-', color='b')
plt.xlabel('Max Depth')
plt.ylabel('Mean Test Score (Accuracy)')
plt.title('Effect of Max Depth on Model Accuracy')
plt.xticks(depth_values)  
plt.grid(True)
plt.show()
