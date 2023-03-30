import numpy as np
import pandas as pd

from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelBinarizer, StandardScaler
from sklearn.linear_model import LinearRegression

import warnings
warnings.simplefilter(action='ignore', category=Warning)

from datasets import datasets
import matplotlib.pyplot as plt

#all_datasets = ["letter", "mushroom", "bank", "credit"]
all_datasets = ["bank"]
all_missingness = [30]

def linearRegression(X_train, X_test, y_train, y_test):
    # Create a LinearRegression object
    lr = LinearRegression()

    # Fit the model to the training data
    lr.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = lr.predict(X_test)

    # Calculate the mean squared error
    mse = mean_squared_error(y_test, y_pred)

    return mse

def kNeighborsClassifier(X, y, X_train, X_test, y_train, y_test):
   # Scale the features using StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Find best k value
    k_values = [i for i in range (1,31)]
    scores = []

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        score = cross_val_score(knn, X, y, cv=5)
        scores.append(np.mean(score))

    best_index = np.argmax(scores)
    best_k = k_values[best_index]

    # Create classifier
    knn = KNeighborsClassifier(n_neighbors=best_k)
    knn.fit(X_train, y_train)

    # Predict
    y_pred = knn.predict(X_test)

    # Convert string targets to binary form
    lb = LabelBinarizer()
    y_test_binary = lb.fit_transform(y_test)
    y_pred_binary = lb.transform(y_pred)

    # Evaluate 
    accuracy = accuracy_score(y_test_binary, y_pred_binary)
    auroc = roc_auc_score(y_test_binary, y_pred_binary)

    return accuracy, auroc

def main():
    results = []

    for data_name in all_datasets:
       for miss_rate in all_missingness:
            filename_data_w_removed_missingness = 'ohe_preprocessed_data/one_hot_test_data_missing_rows_deleted/one_hot_{}_test_{}.csv'.format(data_name, miss_rate)
            data_w_removed_missingness = pd.read_csv(filename_data_w_removed_missingness)

            if data_w_removed_missingness.empty:
                break

            # Split the data into features (X) and target (y)
            X = data_w_removed_missingness.iloc[:, :-1]
            y = data_w_removed_missingness.iloc[:, -1]

            # Split the data into training and test sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                        
            if datasets[data_name]["classification"]["model"] == KNeighborsClassifier:
                accuracy, auroc = kNeighborsClassifier(X, y, X_train, X_test, y_train, y_test)
                results.append({'dataset': data_name + str(miss_rate), 'scores':{'accuracy': str(accuracy), 'auroc': str(auroc)}})
            elif datasets[data_name]["classification"]["model"] == LinearRegression:
                mse = linearRegression(X_train, X_test, y_train, y_test)
                results.append({'dataset': data_name + str(miss_rate), 'scores':{'mse': str(mse)}})
            print(f"{data_name} with {miss_rate}% missingness, predicted")
    return results

if __name__ == '__main__':
  results = main()
  for item in results:
    print('Dataset:', item['dataset'])
    print('Scores:', item['scores'])


