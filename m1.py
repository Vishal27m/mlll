import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error,accuracy_score
import numpy as np


class Layoff:

    def __init__(self):
        self.data = None
        self.encoded_data = None

    def preprocess(self):
        self.data = pd.read_csv("C:/Machine learning/Student_performance_data _.csv")

        # Missing values
        self.data = self.data.dropna()

        # Categorical features
        categorical_features = self.data.select_dtypes(include=['object']).columns
        self.encoded_data = pd.get_dummies(self.data, columns=categorical_features)

        # Correlation heatmap
        corr_matrix = self.encoded_data.corr()
        plt.figure(figsize=(14, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
        plt.title('Correlation Heatmap of Student Performance Data')
        plt.show()

        # Linear regression
        features = ['Absences']
        target = 'GPA'
        X = self.encoded_data[features]
        y = self.encoded_data[target]

        # Training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Linear regression model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)

        # Metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        print(f'Simple Linear Regression - MAE: {mae}, MSE: {mse}, RMSE: {rmse}, R2: {r2}')

        # Plot
        plt.figure(figsize=(12, 8))
        plt.plot(range(50), y_test[:50], color='blue', label='Actual GPA')
        plt.plot(range(50), y_pred[:50], color='red', linestyle='dashed', label='Predicted GPA')
        plt.title('Simple Linear Regression (First 50 Data Points)')
        plt.xlabel('Data Point Index')
        plt.ylabel('GPA')
        plt.legend()
        plt.show()

    def multiple_linear_regression(self):
        features = ['Absences', 'GradeClass', 'ParentalSupport']
        target = 'GPA'
        X = self.encoded_data[features]
        y = self.encoded_data[target]
         # Training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Linear regression model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)

        # Metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        print(f'Multiple Linear Regression - MAE: {mae}, MSE: {mse}, RMSE: {rmse}, R2: {r2}')

        # Plot for the last fold (as an example)
        plt.figure(figsize=(12, 8))
        plt.plot(range(50), y_test[:50], color='blue', label='Actual GPA')
        plt.plot(range(50), y_pred[:50], color='red', linestyle='dashed', label='Predicted GPA')
        plt.title('Multiple Linear Regression (Last Fold - First 50 Data Points)')
        plt.xlabel('Data Point Index')
        plt.ylabel('GPA')
        plt.legend()
        plt.show()

    def polynomial_regression(self):
        features = ['Absences', 'GradeClass', 'ParentalSupport']
        target = 'GPA'
        X = self.encoded_data[features]
        y = self.encoded_data[target]

        # Training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Polynomial Regression model
        poly_features = PolynomialFeatures(degree=2)
        X_train_poly = poly_features.fit_transform(X_train)
        X_test_poly = poly_features.transform(X_test)

        model = LinearRegression()
        model.fit(X_train_poly, y_train)

        # Predict
        y_pred = model.predict(X_test_poly)

        # Metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        print(f'Polynomial Regression - MAE: {mae}, MSE: {mse}, RMSE: {rmse}, R2: {r2}')

        # Plot
        plt.figure(figsize=(12, 8))
        plt.plot(range(50), y_test[:50], color='blue', label='Actual GPA')
        plt.plot(range(50), y_pred[:50], color='red', linestyle='dashed', label='Predicted GPA')
        plt.title('Polynomial Regression (Test Data Points)')
        plt.xlabel('Data Point Index')
        plt.ylabel('GPA')
        plt.legend()
        plt.show()

    def polynomial_regression_kfold(self, degree=2):
        features = ['Absences', 'GradeClass', 'ParentalSupport']
        target = 'GPA'
        X = self.encoded_data[features]
        y = self.encoded_data[target]

        accuracy = []
        max_acc = []
        mean_acc = []
        std_acc = []

        for i in range(2, 11):   
            kf = KFold(n_splits=i, shuffle=True, random_state=42)
            fold_accuracy = []

            for train_index, test_index in kf.split(X):
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]

                # Polynomial Regression
                poly_features = PolynomialFeatures(degree=degree)
                X_train_poly = poly_features.fit_transform(X_train)
                X_test_poly = poly_features.transform(X_test)

                model = LinearRegression()
                model.fit(X_train_poly, y_train)
                y_pred = model.predict(X_test_poly)

                # Convert regression output to classification by rounding
                y_test_class = np.round(y_test).astype(int)
                y_pred_class = np.round(y_pred).astype(int)

                ac = accuracy_score(y_test_class, y_pred_class)
                fold_accuracy.append(ac)

            accuracy.append(fold_accuracy)
            max_acc.append(max(fold_accuracy))
            mean_acc.append(np.mean(fold_accuracy))
            std_acc.append(np.std(fold_accuracy))
            r2 = r2_score(y_test, y_pred)

            print(f"\nK-Folds: {i}, Accuracy: {fold_accuracy}")
            print(f"K-Folds: {i}, Accuracy_mean: {np.mean(fold_accuracy)}")
            print(f"K-Folds: {i}, Accuracy_std: {np.std(fold_accuracy)}")
            print(f"K-Folds: {i}, Max_Accuracy: {max(fold_accuracy)}")
            print(f"K-Folds: {i}, r2_score: {r2}")



        # Plotting Accuracy
        plt.figure(figsize=(12, 8))
        plt.plot(range(2, 11), max_acc, linestyle='-', marker='o', label="Max Accuracy", color='red')
        plt.plot(range(2, 11), mean_acc, linestyle='-', marker='^', label="Mean Accuracy", color='blue')
        plt.plot(range(2, 11), std_acc, linestyle='-', marker='*', label="Std Dev of Accuracy", color='green')
        plt.xlabel('K-Fold')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('Accuracy of Polynomial Regression with K-Fold Cross-Validation')
        plt.grid(True)
        min_accuracy = min(min(max_acc), min(mean_acc)) * 0.9
        max_accuracy = max(max(max_acc), max(mean_acc)) * 1.1
        plt.ylim([min_accuracy, max_accuracy])

        plt.show()
        kf = KFold(n_splits=10, shuffle=True, random_state=42)
            
        for train_index, test_index in kf.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                
            poly_features = PolynomialFeatures(degree=2)
            X_train_poly = poly_features.fit_transform(X_train)
            X_test_poly = poly_features.transform(X_test)

            model = LinearRegression()
            model.fit(X_train_poly, y_train)

            # Predict
            y_pred = model.predict(X_test_poly)

        # Metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        print(f'Polynomial Regression  for K-fold- MAE: {mae}, MSE: {mse}, RMSE: {rmse}, R2: {r2}')




def main():
    l = Layoff()
    l.preprocess()
    l.multiple_linear_regression()
    l.polynomial_regression()
    l.polynomial_regression_kfold()


if __name__ == "__main__":
    main()
