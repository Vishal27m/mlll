import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn import tree
import matplotlib.pyplot as plt

# Load the Wine dataset
wine = load_wine()
X = wine.data
y = wine.target

# Preprocess the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Create and train the initial Decision Tree model
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Make predictions with the initial model
y_pred = clf.predict(X_test)

# Evaluate the initial model
print("Initial Accuracy:", accuracy_score(y_test, y_pred))
print("Initial Classification Report:\n", classification_report(y_test, y_pred))

# Plot the initial Decision Tree
plt.figure(figsize=(20,10))
tree.plot_tree(clf, feature_names=wine.feature_names, class_names=wine.target_names, filled=True, rounded=True, fontsize=10)
plt.title("Initial Decision Tree")
plt.show()

# Hyperparameter tuning with GridSearchCV
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5, n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

# Get the best parameters and train the optimized model
best_params = grid_search.best_params_
print("Best parameters:", best_params)

clf_optimized = DecisionTreeClassifier(**best_params, random_state=42)
clf_optimized.fit(X_train, y_train)

# Make predictions with the optimized model
y_pred_optimized = clf_optimized.predict(X_test)

# Evaluate the optimized model
print("Optimized Accuracy:", accuracy_score(y_test, y_pred_optimized))
print("Optimized Classification Report:\n", classification_report(y_test, y_pred_optimized))

# Plot the optimized Decision Tree
plt.figure(figsize=(20,10))
tree.plot_tree(clf_optimized, feature_names=wine.feature_names, class_names=wine.target_names, filled=True, rounded=True, fontsize=10)
plt.title("Optimized Decision Tree")
plt.show()

# Confusion Matrix for the optimized model
cm = confusion_matrix(y_test, y_pred_optimized)
print(cm)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=wine.target_names)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix - Optimized Model")
plt.show()

# Extract mean test scores for each max_depth
mean_test_scores = grid_search.cv_results_['mean_test_score']
param_depths = [params['max_depth'] for params in grid_search.cv_results_['params']]

# Print out mean test scores and corresponding max_depth values
for depth, score in zip(param_depths, mean_test_scores):
    print(f"Max Depth: {depth}, Mean Test Score: {score}")

# Separate the depths and calculate mean scores
depth_scores = {}
for depth in set(param_depths):
    if depth is not None:
        scores = [mean_test_scores[i] for i in range(len(param_depths)) if param_depths[i] == depth]
        depth_scores[depth] = np.mean(scores)

# Extract and sort depths and scores for plotting
depth_values = sorted(depth_scores.keys())
scores = [depth_scores[depth] for depth in depth_values]

# Plot the accuracy for different max_depth values
plt.figure(figsize=(10, 6))
plt.plot(depth_values, scores, marker='o', linestyle='-', color='b')
plt.xlabel('Max Depth')
plt.ylabel('Mean Test Score (Accuracy)')
plt.title('Effect of Max Depth on Model Accuracy')
plt.xticks(depth_values, [str(depth) if depth is not None else 'None' for depth in depth_values])  # Customize x-ticks
plt.grid(True)
plt.show()
