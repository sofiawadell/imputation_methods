import pandas as pd

from run_imputation_methods import run_mf
from run_imputation_methods import run_kNN
from run_imputation_methods import run_median_mode

from utils import normalize_numeric
from utils import rmse_num_loss
from utils import rmse_cat_loss
from utils import m_rmse_loss
from utils import pfc

''' Description

Imputes data set using the imputation methods MissForest, kNN or median/mode imputation and prints the result

'''

def main (data_name, miss_rate, method, ctgan, best_k = None):

    ''' Runs a chosen imputation method and evaluates the result
  
    Args:
    - data_name: mushroom, letter, bank, credit or news
    - miss_rate: the probability of missing components (0.1, 0.3 or 0.5)
    - method: missforest, knn or median_mode
    - ctgan: '50', '100' or "", determines if ctgan increased training data should be used or not.
        "50": Training data is increased by 50%
        "100": Training data is increased by 100% 
        "". CTGAN is not used for increasing training data
    
    Returns:
    - rmse_num: RMSE between the original and the imputed data set for the numerical variables 
    - m_rmse: Modified RMSE between the original and the imputed data set for all variables 
    - rmse_cat: RMSE between the original and the imputed data set for the categorical variables 
    - pfc_value: Share of percently falsely classified imputed values for the categorical variables, when comparing to the original data

    '''
    
    # Choose method for imputation
    if method == "missforest": # Impute missing data for test and training data for MissForest
        train_imp_data_x, test_imp_data_x, train_data_x, test_data_x, mask_train, mask_test  = run_mf(data_name, miss_rate, ctgan)
    elif method == "knn": # Impute using kNN 
        train_imp_data_x, test_imp_data_x, train_data_x, test_data_x, mask_train, mask_test  = run_kNN(data_name, miss_rate, best_k, ctgan)
    elif method == "median_mode":
        train_imp_data_x, test_imp_data_x, train_data_x, test_data_x, mask_train, mask_test  = run_median_mode(data_name, miss_rate, ctgan)
    else:
        ValueError("Method not found")

    # Save renormalized imputed data 
    missingness = int(miss_rate*100)

    # Save imputed data based on ctgan variable 
    if ctgan == "50":
        filename_test_imp_ctgan50 = 'imputed_data/ctgan50/imputed_{}_test_data/imputed_{}_{}_test_{}_ctgan50.csv'.format(method, method, data_name, missingness)
        test_imp_data_x.to_csv(filename_test_imp_ctgan50, index=False)
    elif ctgan == "100":
        filename_test_imp_ctgan100 = 'imputed_data/ctgan100/imputed_{}_test_data/imputed_{}_{}_test_{}_ctgan100.csv'.format(method, method, data_name, missingness)
        test_imp_data_x.to_csv(filename_test_imp_ctgan100, index=False)
    else: 
        filename_train_imp = 'imputed_data/no_ctgan/imputed_{}_train_data/imputed_{}_{}_train_{}.csv'.format(method, method, data_name, missingness)
        train_imp_data_x.to_csv(filename_train_imp, index=False)

        filename_test_imp = 'imputed_data/no_ctgan/imputed_{}_test_data/imputed_{}_{}_test_{}.csv'.format(method, method, data_name, missingness)
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
miss = [0.1, 0.3, 0.5]
# dat =["mushroom", "letter", "bank", "credit"]
# dat =["mushroom", "letter"]
dat = ["letter"]
ctgan = ["", "50"]
for m in miss:
    for c in ctgan:
        for d in dat:
            rmse_num, m_rmse, pfc_value, rmse_cat = main(d, m, "knn", c)
            # Print results
            print(f"Dataset: {d}, Missingness: {int(m*100)}%, ctgan: {c}%")
            print(f"Numerical RMSE: {rmse_num}")
            print(f"Categorical RMSE: {rmse_cat}")
            print(f"Modified RMSE: {m_rmse}")
            print(f"PFC: {pfc_value}")