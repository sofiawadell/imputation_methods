import pandas as pd

from data_loader import data_loader
from data_loader import data_loader_ohe
from data_loader import data_loader_full

from run_imputation_methods import run_mf
from run_imputation_methods import run_MICE
from run_imputation_methods import run_kNN
from run_imputation_methods import run_median_mode

from missForest_impute import missForest_impute
from mice_impute import mice_impute
# from utils import normalize_num_data

def main (data_name, miss_rate, method):

    # TBU: Method which ensures arguments are correct
    data_name = data_name
    miss_rate = miss_rate #0.1, 0.3, 0.5 and 0.7 

    # Find max/min values for entire data set to enable normalization for numerical variables 
    full_data = data_loader_full(data_name, miss_rate)
    # num_norm_data, norm_parameters = normalize_num_data(full_data)
    norm_parameters = 1
    
    # Choose method for imputation
    if method == "missForest": # Impute missing data for test and training data for MissForest
        train_data_norm_x, train_imputed_norm_data_x, test_data_norm_x, test_imputed_norm_data_x = run_mf(data_name, miss_rate, norm_parameters)
    elif method == "MICE": # Impute using MICE 
        train_data_norm_x, train_imputed_norm_data_x, test_data_norm_x, test_imputed_norm_data_x = run_MICE(data_name, miss_rate, norm_parameters)
    elif method == "kNN": # Impute using kNN 
        train_data_norm_x, train_imputed_norm_data_x, test_data_norm_x, test_imputed_norm_data_x = run_kNN(data_name, miss_rate, norm_parameters)
    elif method == "median/mode":
        train_data_norm_x, train_imputed_norm_data_x, test_data_norm_x, test_imputed_norm_data_x = run_median_mode(data_name, miss_rate, norm_parameters)
    else:
        ValueError("Method not found")

    # Calculate RMSE for numerical data
    rmse_num = 1

    # Calculate RMSE for categorical data 
    rmse_cat = 1

    # Calculate PFC for categorical data 
    pfc_categorical = 1

    # Save imputed
    print(pd.DataFrame(test_imputed_norm_data_x))

    return pd.DataFrame(test_imputed_norm_data_x), rmse_num, rmse_cat, pfc_categorical

test_imputed_norm_data_x, rmse_num, rmse_cat, pfc_categorical = main("credit", 0.1, "kNN")

