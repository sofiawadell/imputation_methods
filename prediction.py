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

all_datasets = ["mushroom", "letter", "bank", "credit"]
# all_datasets = ["news"]
all_missingness = [10]

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

def main(method, ctgan):
    results = []

    for data_name in all_datasets:
       for miss_rate in all_missingness:
            if ctgan == "":
                filename_imputed_data = 'imputed_data/no_ctgan/imputed_{}_test_data/imputed_{}_{}_test_{}.csv'.format(method,method,data_name, miss_rate)
                imputed_data_wo_target = pd.read_csv(filename_imputed_data)
            else:
                filename_imputed_data = 'imputed_data/ctgan{}/imputed_{}_test_data/imputed_{}_{}_test_{}_ctgan{}.csv'.format(ctgan,method,method,data_name, miss_rate, ctgan)
                imputed_data_wo_target = pd.read_csv(filename_imputed_data)
            
            # Load original data to extract target column 
            filename_original_data = 'ohe_preprocessed_data/one_hot_test_data/one_hot_{}_test.csv'.format(data_name)
            original_data = pd.read_csv(filename_original_data)
            
            # Split the data into features (X) and target (y)
            X = imputed_data_wo_target
            y = original_data[datasets[data_name]["target"]]

            # Split the data into training and test sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                        
            if datasets[data_name]["classification"]["model"] == KNeighborsClassifier:
                accuracy, auroc = kNeighborsClassifier(X, y, X_train, X_test, y_train, y_test)
                results.append({'dataset': data_name + str(miss_rate), 'scores':{'accuracy': str(accuracy), 'auroc': str(auroc)}})
            elif datasets[data_name]["classification"]["model"] == LinearRegression:
                mse = linearRegression(X_train, X_test, y_train, y_test)
                results.append({'dataset': data_name + str(miss_rate), 'scores':{'mse': str(mse)}})
            print(f"{data_name} with {miss_rate}% missingness & ctgan: {ctgan}% predicted")

    return results

if __name__ == '__main__':
  method = "missforest"
  ctgan = "100"    
  results = main(method, ctgan)
  print('Method: ', method)
  for item in results:    
    print('Dataset:', item['dataset'])
    print('Scores:', item['scores'])


