import pandas as pd

from data_loader import data_loader
from data_loader import data_loader_ohe
from data_loader import data_loader_full

from run_imputation_methods import run_mf
from run_imputation_methods import run_MICE
from run_imputation_methods import run_kNN
from run_imputation_methods import run_median_mode

from utils import normalization

def main (data_name, miss_rate, method, best_k = None):

    # TBU: Method which ensures arguments are correct
    data_name = data_name
    miss_rate = miss_rate #0.1, 0.3, 0.5 and 0.7 
    
    # Choose method for imputation
    if method == "missForest": # Impute missing data for test and training data for MissForest
        train_data_x, train_imputed_data_x, test_data_x, test_imputed_data_x, mask_train, mask_test = run_mf(data_name, miss_rate)
    elif method == "kNN": # Impute using kNN 
        train_data_x, train_imputed_data_x, test_data_x, test_imputed_data_x, mask_train, mask_test = run_kNN(data_name, miss_rate, best_k)
    elif method == "median/mode":
        train_data_x, train_imputed_data_x, test_data_x, test_imputed_data_x, mask_train, mask_test = run_median_mode(data_name, miss_rate)
    else:
        ValueError("Method not found")

    # Calculate RMSE for numerical data
    rmse_num = 1

    # Calculate RMSE for categorical data 
    rmse_cat = 1

    # Calculate PFC for categorical data 
    pfc_categorical = 1

    # Save imputed
    print(pd.DataFrame(train_imputed_data_x))

    return pd.DataFrame(train_imputed_data_x), rmse_num, rmse_cat, pfc_categorical

# Run main code for kNN, missForest and median/mode imputation
train_imputed_data_x, rmse_num, rmse_cat, pfc_categorical = main("credit", 0.1, "kNN")

