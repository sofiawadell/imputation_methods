import numpy as np
import pandas as pd

from data_loader import data_loader_factor
from data_loader import data_loader_factor_wo_target

from knn_optimize_params import optimize_params
from missForest_impute import missForest_impute
from knn_impute import knn_impute

from utils import normalize_numeric
from utils import renormalize_numeric


def run_mf(data_name, miss_rate):
    
    # Load OHE data
    train_data_x, train_miss_data_x, test_data_x, test_miss_data_x = data_loader_factor_wo_target(data_name, miss_rate) 

    # Define mask matrix
    mask_train = pd.DataFrame(1-np.isnan(train_miss_data_x.values))
    mask_test = pd.DataFrame(1-np.isnan(test_miss_data_x.values)) 

    # Normalize the test data set using the norm_parameters from the training data with missingness
    train_miss_norm_data_x, norm_params_train_miss_data = normalize_numeric(train_miss_data_x, data_name)
    test_miss_norm_data_x, _ = normalize_numeric(test_miss_data_x, data_name, norm_params_train_miss_data)

    # Impute data using missForest
    test_imp_norm_data_x_np, train_imp_norm_data_x_np = missForest_impute(train_miss_norm_data_x, test_miss_norm_data_x)
    
    # Transform to pandas array to keep the column names 
    train_imp_norm_data_x = pd.DataFrame(train_imp_norm_data_x_np, columns=test_miss_data_x.columns) 
    test_imp_norm_data_x = pd.DataFrame(test_imp_norm_data_x_np, columns=test_miss_data_x.columns) 

    # Renormalize data sets
    train_imp_data_x = renormalize_numeric(train_imp_norm_data_x, norm_params_train_miss_data, data_name)
    test_imp_data_x = renormalize_numeric(test_imp_norm_data_x, norm_params_train_miss_data, data_name)

    return train_imp_data_x, test_imp_data_x, train_data_x, test_data_x, mask_train, mask_test  

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


    return train_imp_data_x, test_imp_data_x, train_data_x, test_data_x, mask_train, mask_test


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

    return train_imp_data_x, test_imp_data_x, train_data_x, test_data_x, mask_train, mask_test 
