import pandas as pd
import numpy as np

from data_loader import data_loader_ohe
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, KFold

def knn_impute(train_data_x_np, train_miss_data_x_np, test_data_x_np, test_miss_data_x_np):

    # Define a range of values for the number of neighbors
    k_values = range(1, 3)

    # Define the number of folds for cross-validation
    n_folds = 3

    # Initialize a dictionary to store the RMSE scores for each k value
    rmse_scores = {}

    # Loop over the k values and perform cross-validation
    for k in k_values:
        # Initialize a kNN imputer with k neighbors
        imputer = KNNImputer(n_neighbors=k)
        
        # Use cross-validation to evaluate the imputer
        cv = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        # Calculate the RMSE between the missing and non-missing datasets for each fold
        fold_rmses = []
        print(k)

        for train_index, test_index in cv.split(train_miss_data_x_np):
            
            # Split the data into training and test sets
            X_train, X_test = train_miss_data_x_np[train_index], train_miss_data_x_np[test_index]
            y_test = train_data_x_np[test_index]
            
            # Impute the missing values in the test set using the training set
            imputer.fit(X_train)
            X_test_imputed = imputer.transform(X_test)
            
            # Calculate the RMSE between the imputed test set and the non-missing test set
            fold_rmses.append(np.sqrt(mean_squared_error(y_test, X_test_imputed)))
            print(np.sqrt(mean_squared_error(y_test, X_test_imputed)))
        
        # Calculate the mean RMSE across all folds for this k value
        rmse_scores[k] = np.mean(fold_rmses)

    # Print the RMSE scores for each k value
    for k, rmse in rmse_scores.items():
        print(f"k={k}: RMSE={rmse:.3f}")

    # Select the k value with the lowest RMSE score
    best_k = min(rmse_scores, key=rmse_scores.get)
    print(f"Best k value: {best_k}")

    # Impute the missing values using the best k value
    imputer = KNNImputer(n_neighbors=best_k)
    train_imputed_norm_data_x = imputer.fit_transform(train_miss_data_x_np)
    test_imputed_norm_data_x = imputer.transform(test_miss_data_x_np)
    
    return test_imputed_norm_data_x, train_imputed_norm_data_x
