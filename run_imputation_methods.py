import numpy as np

from data_loader import data_loader_factor
from data_loader import data_loader_ohe
from knn_optimize_params import optimize_params
from missForest_impute import missForest_impute
from mice_impute import mice_impute
from knn_impute import knn_impute
from utils import normalization


def run_mf(data_name, miss_rate):
    
    # Load OHE data
    train_data_x, train_miss_data_x, test_data_x, test_miss_data_x = data_loader_ohe(data_name, miss_rate) 

    # Define mask matrix
    mask_train = 1-np.isnan(train_miss_data_x)
    mask_test = 1-np.isnan(test_miss_data_x) 

    # Normalize all data sets using the norm_parameters 
    data = [train_data_x,train_miss_data_x, test_data_x, test_miss_data_x]
    train_data_norm_x, train_miss_data_norm_x, test_data_norm_x, test_miss_data_norm_x = [normalization(d,norm_parameters) for d in data]

    # Impute data using missForest
    test_imputed_norm_data_x, train_imputed_norm_data_x = missForest_impute(train_miss_data_norm_x, test_miss_data_norm_x)

    # Renormalize data sets

    return train_data_norm_x, train_imputed_norm_data_x, test_data_norm_x, test_imputed_norm_data_x, mask_train, mask_test, norm_params  


def run_MICE(data_name, miss_rate):

    # Load data without, introduce missingness & dummy encode 
    train_data_x, train_miss_data_x, test_data_x, test_miss_data_x = data_loader_factor(data_name, miss_rate) 
    
    # Define mask matrix 
    mask_train = 1-np.isnan(train_miss_data_x)
    mask_test = 1-np.isnan(test_miss_data_x) 

    # Normalize the numerical values of all data sets 
    data = [train_data_x,train_miss_data_x, test_data_x, test_miss_data_x]
    train_data_norm_x, train_miss_data_norm_x, test_data_norm_x, test_miss_data_norm_x = [normalization(d,data_name,norm_params) for d in data]
    
    # Transform using MICE
    test_imputed_norm_data_x, train_imputed_norm_data_x = mice_impute(train_miss_data_x, test_miss_data_x)

    # Renormalize data sets

    return train_data_norm_x, train_imputed_norm_data_x, test_data_norm_x, test_imputed_norm_data_x, mask_train, mask_test, norm_params 


def run_kNN(data_name, miss_rate, best_k):
    
    # Optimize knn parameters (if not already done)
    if best_k is None:
        best_k = optimize_params(data_name)
    else:
        best_k = best_k

    # Load OHE data
    train_data_x, train_miss_data_x, test_data_x, test_miss_data_x = data_loader_ohe(data_name, miss_rate) 

    # Define mask matrix
    mask_train = 1-np.isnan(train_miss_data_x)
    mask_test = 1-np.isnan(test_miss_data_x) 

    # Normalize the numerical values of all data sets 
    data = [train_data_x,train_miss_data_x, test_data_x, test_miss_data_x]
    train_data_norm_x, train_miss_data_norm_x, test_data_norm_x, test_miss_data_norm_x = [normalization(d,norm_parameters) for d in data]

    # Transform to numpy arrays
    train_data_x_np, train_miss_data_x_np, test_data_x_np, test_miss_data_x_np = train_data_x.values, train_miss_data_x.values, test_data_x.values, test_miss_data_x.values

    # Impute data using kNN
    test_imputed_norm_data_x, train_imputed_norm_data_x = knn_impute(train_data_x_np, train_miss_data_x_np, test_data_x_np, test_miss_data_x_np, best_k)

    # Renormalize data sets


    return train_data_norm_x, train_imputed_norm_data_x, test_data_norm_x, test_imputed_norm_data_x, mask_train, mask_test, norm_params 


def run_median_mode(data_name, miss_rate):
    # Load data, introduce missingness & dummy encode 
    train_data_x, train_miss_data_x, test_data_x, test_miss_data_x = data_loader_factor(data_name, miss_rate) 

    # Define mask matrix
    mask_train = 1-np.isnan(train_miss_data_x)
    mask_test = 1-np.isnan(test_miss_data_x) 

    # Normalize the numerical values of all data sets 
    data = [train_data_x,train_miss_data_x, test_data_x, test_miss_data_x]
    train_data_norm_x, train_miss_data_norm_x, test_data_norm_x, test_miss_data_norm_x = [normalization(d,norm_parameters) for d in data]
    
    # Impute using median/mode strategy 
    test_imputed_norm_data_x, train_imputed_norm_data_x = median_mode_impute(train_miss_data_x, test_miss_data_x)

    # Renormalize data sets

    return train_data_norm_x, train_imputed_norm_data_x, test_data_norm_x, test_imputed_norm_data_x, mask_train, mask_test, norm_params 
