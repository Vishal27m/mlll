import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier,BaggingClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.model_selection import train_test_split,GridSearchCV

df=pd.read_csv("C:/Users/visha/Downloads/CAT 1 Breast cancer dataset - breast-cancer.csv")

#preprocessing
df['diagnosis']=df['diagnosis'].map({'M':1,'B':0})
imputer=SimpleImputer(strategy='mean')
df_imputed=pd.DataFrame(imputer.fit_transform(df),columns=df.columns)

X=df_imputed.drop('diagnosis',axis=1)
y=df_imputed['diagnosis']

#test and train 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# parameter grid 
param_grid ={
    'max_depth': np.arange(1, 20)
}

# GridSearch
grid_search = GridSearchCV(estimator=DecisionTreeClassifier(random_state=42), param_grid=param_grid, cv=5, scoring='accuracy')

grid_search.fit(X_train, y_train)

# best params
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print("Best Parameters:", best_params)
print("Best Cross-Validation Accuracy:", best_score)

# train the Decision Tree with the best parameters
clf = DecisionTreeClassifier(**best_params, random_state=42)
clf.fit(X_train, y_train)

# Predict 
y_pred = clf.predict(X_test)

# accuracy score
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy with best parameters: {accuracy:.2f}")

# classification report
report = classification_report(y_test, y_pred, target_names=["M", "B"])
print("Classification Report:")
print(report)

#confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Bagging classifier
bagging_clf = BaggingClassifier(estimator=clf, n_estimators=100, random_state=42, n_jobs=-1)
bagging_clf.fit(X_train, y_train)
y_pred_bagging = bagging_clf.predict(X_test)

print("\nBagging with Decision Tree Performance:")
accuracy=

#random forest 
rf_clf = RandomForestClassifier(**best_params, random_state=42)
rf_clf.fit(X_train, y_train)
y_pred_rf = rf_clf.predict(X_test)

print("\nRandom Forest Classifier Perfomance:")

# accuracy score for random
accuracy = accuracy_score(y_test, y_pred_rf)
print(f"Accuracy: {accuracy:.2f}")

# classification report for random 
report = classification_report(y_test, y_pred_rf, target_names=["M", "B"])
print("Classification Report:")
print(report)

# confusion matrix for random
conf_matrix = confusion_matrix(y_test, y_pred_rf)
print("Confusion Matrix for Random:")
print(conf_matrix)
