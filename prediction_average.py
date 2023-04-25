import os
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=Warning)

import pandas as pd

from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelBinarizer, StandardScaler
from sklearn.linear_model import LinearRegression

from datasets import datasets
import matplotlib.pyplot as plt
from utils import find_average_and_st_dev

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

def kNeighborsClassifier(X, y, X_train, X_test, y_train, y_test, data_name):
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
    
    class_case = datasets[data_name]['classification']['class-case']
    
    if class_case == 'binary':
        auroc = roc_auc_score(y_test_binary, y_pred_binary)
    elif class_case == 'multiclass':
        auroc = roc_auc_score(y_test_binary, y_pred_binary, multi_class='ovr', average='macro')

    return accuracy, auroc

def main():
    df_results = pd.DataFrame(columns=['Dataset', 'Missing%', 'Additional CTGAN data%', 'Average Accuracy',
                    'St Dev Accuracy', 'Average AUROC', 'St Dev AUROC', 'Average MSE', 'St Dev MSE'])

    for extra_amount in all_extra_amount:
        for data_name in all_datasets:
            for miss_rate in all_missingness:
                all_accuracy_result = []
                all_auroc_result = []
                all_mse_result = []

                for i in range(1, no_datasets+1):
                    print(f"Data: {data_name}, miss rate: {miss_rate}, ctgan: {extra_amount}, data set: {i}")
                    filename_original_data = 'ohe_preprocessed_data/one_hot_test_data/one_hot_{}_test.csv'.format(data_name)
                    original_data = pd.read_csv(filename_original_data)
                    
                    if extra_amount == "":
                        filename_imputed_data = 'imputed_data/no_ctgan/imputed_{}_test_data/{}/imputed_{}_{}_test_{}_{}.csv'.format(method, data_name, method, data_name, miss_rate, i)
                    else:
                        filename_imputed_data = 'imputed_data/ctgan{}/imputed_{}_test_data/{}/imputed_{}_{}_test_{}_ctgan{}_{}.csv'.format(extra_amount, method, data_name, method, data_name, miss_rate, extra_amount, i)
                    
                    # Check if the file exists
                    if not os.path.isfile(filename_imputed_data):
                        continue
                
                    imputed_data_wo_target = pd.read_csv(filename_imputed_data)

                    # Split the data into features (X) and target (y)
                    X = imputed_data_wo_target
                    y = original_data[datasets[data_name]["target"]]

                    # Split the data into training and test sets
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                                
                    if datasets[data_name]['classification']['model'] == KNeighborsClassifier:
                        accuracy, auroc = kNeighborsClassifier(X, y, X_train, X_test, y_train, y_test, data_name)
                        all_accuracy_result.append(accuracy)
                        all_auroc_result.append(auroc)
                    elif datasets[data_name]['classification']['model'] == LinearRegression:
                        mse = linearRegression(X_train, X_test, y_train, y_test)
                        all_mse_result.append(mse)

                average_accuracy, st_dev_accuracy = find_average_and_st_dev(all_accuracy_result)
                average_auroc, st_dev_auroc= find_average_and_st_dev(all_auroc_result)
                average_mse, st_dev_mse = find_average_and_st_dev(all_mse_result)

                results = {'Dataset': data_name, 'Missing%': miss_rate, 'Additional CTGAN data%': extra_amount, 'Average Accuracy': average_accuracy,
                    'St Dev Accuracy': st_dev_accuracy, 'Average AUROC': average_auroc, 'St Dev AUROC': st_dev_auroc, 'Average MSE': average_mse, 'St Dev MSE': st_dev_mse}
                df_results = df_results.append(results, ignore_index=True)

    return df_results

if __name__ == '__main__':
    all_datasets = [ "mushroom", "letter", "bank", "credit", "news"]
    all_missingness = [10, 30, 50]
    all_extra_amount = [""]
    no_datasets = 10
    method = "missforest"
  
    df_results = main()
    # df_results.to_csv('results/prediction_average_missforest_noctgan.csv', index=False)


