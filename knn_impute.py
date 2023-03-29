import pandas as pd
import numpy as np
import time

from sklearn.impute import KNNImputer

''' Description

Imputes a data set with missing data using KNNImputer from SciKit Learn

'''

def knn_impute(train_miss_norm_data_x_np, test_miss_norm_data_x_np, best_k):

    ''' Imputes a normalized training and testing data set that contains missingness using kNNimputer with best_k neighbors. 
    Fits the model on a train data set and imputes a training and a testing data set. 
  
    Args:
    - train_miss_norm_data_x_np: Normalized data set containing missingness in a Numpy matrix
    - test_miss_norm_data_x_np: Normalized data set containing missingness in a Numpy matrix
    - best_k: Optimal number of neighbors for the specific data set
    
    Returns:
    - train_imp_norm_data_x_np: Training data set imputed using kNNImputer with best_k neighbors
    - test_imp_norm_data_x_np: Testing data set imputed using kNNImputer with best_k neighbors
    '''
    start_time = time.time()

    # Impute the missing values using kNNImputer with the best_k value
    imputer = KNNImputer(n_neighbors=best_k)

    # Fit a model using train data and impute train data 
    train_imp_norm_data_x_np = imputer.fit_transform(train_miss_norm_data_x_np)

    print("Training done")
    
    # Impute test data set
    test_imp_norm_data_x_np = imputer.transform(test_miss_norm_data_x_np)

    # End timer
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"Elapsed time: {elapsed_time} seconds")
    
    return train_imp_norm_data_x_np, test_imp_norm_data_x_np
