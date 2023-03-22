import pandas as pd

from data_loader import data_loader
from data_loader import data_loader_factor_wo_target
from data_loader import data_loader_full

from run_imputation_methods import run_mf
from run_imputation_methods import run_kNN
from run_imputation_methods import run_median_mode

from utils import normalize_numeric
from utils import rmse_num_loss
from utils import rmse_cat_loss
from utils import m_rmse_loss
from utils import pfc

def main (data_name, miss_rate, method, best_k = None):

    # TBU: Method which ensures arguments are correct
    data_name = data_name
    miss_rate = miss_rate 
    
    # Choose method for imputation
    if method == "missforest": # Impute missing data for test and training data for MissForest
        train_imp_data_x, test_imp_data_x, train_data_x, test_data_x, mask_train, mask_test  = run_mf(data_name, miss_rate)
    elif method == "knn": # Impute using kNN 
        train_imp_data_x, test_imp_data_x, train_data_x, test_data_x, mask_train, mask_test  = run_kNN(data_name, miss_rate, best_k)
    elif method == "median_mode":
        train_imp_data_x, test_imp_data_x, train_data_x, test_data_x, mask_train, mask_test  = run_median_mode(data_name, miss_rate)
    else:
        ValueError("Method not found")

    # Save renormalized imputed data 
    missingness = int(miss_rate*100)

    filename_train_imp= 'imputed_{}_train_data/imputed_{}_{}_train_{}.csv'.format(method, method, data_name, missingness)
    train_imp_data_x.to_csv(filename_train_imp, index=False)

    filename_test_imp = 'imputed_{}_test_data/imputed_{}_{}_test_{}.csv'.format(method, method, data_name, missingness)
    test_imp_data_x.to_csv(filename_test_imp, index=False)

    # Normalize the imputed data set using the full data set 
    train_full_data_norm_x, norm_params_full_data_train = normalize_numeric(train_data_x, data_name)
    test_full_data_norm_x, _ = normalize_numeric(test_data_x, data_name, norm_params_full_data_train)
    test_imp_data_norm_x, _ = normalize_numeric(test_imp_data_x, data_name, norm_params_full_data_train)   

    # Calculate RMSE for numerical data
    rmse_num = rmse_num_loss(test_full_data_norm_x, test_imp_data_norm_x, mask_test, data_name)

    # Calculate RMSE for numerical and categorical data 
    rmse_cat = rmse_cat_loss(test_full_data_norm_x, test_imp_data_norm_x, mask_test, data_name)
    m_rmse = m_rmse_loss(rmse_num, rmse_cat)

    # Calculate PFC for categorical data 
    pfc_value = pfc(test_full_data_norm_x, test_imp_data_norm_x, mask_test, data_name)

    return rmse_num, m_rmse, pfc_value, rmse_cat

# Run main code for kNN, missForest and median/mode imputation
rmse_num, m_rmse, pfc_value, rmse_cat = main("credit", 0.1, "missforest")

print(rmse_num)
print(rmse_cat)
print(m_rmse)
print(pfc_value)