import numpy as np

from data_loader import data_loader_ohe_wo_target
from data_loader import data_loader_norm_factor_mice_imputed_data
from data_loader import data_loader_factor_wo_target
from data_loader import data_loader_factor_wo_target_ctgan50
from data_loader import data_loader_factor_wo_target_ctgan100
from data_loader import data_loader_norm_factor_mice_imputed_data_ctgan
from data_loader import data_loader_ohe_wo_target_ctgan50
from data_loader import data_loader_ohe_wo_target_ctgan100

from factor_encode_to_ohe import factor_encode_to_ohe

from utils import normalize_numeric
from utils import renormalize_numeric
from utils import rmse_num_loss
from utils import rmse_cat_loss
from utils import m_rmse_loss
from utils import pfc

''' Description

Prepares a data set with missingness for imputation in R using MICE. Evaluates the results after the imputation and prints RMSE and PFC.

'''

#### Parameters to adjust
data_name = "mushroom" # "mushroom", "letter", "bank", "credit" or "news"
miss_rate = 0.3 # 0.1, 0.3 or 0.5
ctgan = "100"   #"50", "100" or ""
state = "after imputation" #"before imputation" or "after imputation"

def before_imputation(data_name, miss_rate, ctgan):
    ''' Prepares a data set with missing values for imputation using MICE in R. 
    Loads factor encoded data set without target cariable, normalizes the train and test data using parameters from the train data set.
  
    Args:
    - data_name: mushroom, letter, bank, credit or news
    - miss_rate: the probability of missing components (0.1, 0.3 or 0.5)
    - ctgan: '50', '100' or None, determines if ctgan increased training data should be used or not.
        "50": Training data is increased by 50%
        "100": Training data is increased by 100% 
        "": CTGAN is not used for increasing training data
    
    Returns:
    - train_miss_factor_data_norm_x: Normalized factor encoded train data set with missing values
    - test_miss_factor_data_norm_x: Normalized factor encoded test data set with missing values
    - train_factor_data_x: Full factor encoded train data set without missing values
    - test_factor_data_x: Full factor encoded test data set without missing values
    - norm_params_train_miss_data: Normalizing parameters from train data set with missing values, which have been used to normalize the train and test data set with missing values 
    '''

    # Load factor encoded data without target column
    if ctgan == "50":
        train_factor_data_x, train_factor_miss_data_x, test_factor_data_x, test_factor_miss_data_x = data_loader_factor_wo_target_ctgan50(data_name, miss_rate) 
    elif ctgan == "100":
        train_factor_data_x, train_factor_miss_data_x, test_factor_data_x, test_factor_miss_data_x = data_loader_factor_wo_target_ctgan100(data_name, miss_rate) 
    else:
        train_factor_data_x, train_factor_miss_data_x, test_factor_data_x, test_factor_miss_data_x = data_loader_factor_wo_target(data_name, miss_rate) 

    # Normalize training missing data 
    train_miss_factor_data_norm_x, norm_params_train_miss_data = normalize_numeric(train_factor_miss_data_x, data_name)

    # Normalize testing missing data using parameters from training missing data
    test_miss_factor_data_norm_x, _ = normalize_numeric(test_factor_miss_data_x, data_name, norm_params_train_miss_data)

    return train_miss_factor_data_norm_x, test_miss_factor_data_norm_x, train_factor_data_x, test_factor_data_x, norm_params_train_miss_data

def after_imputation(train_factor_data_x, test_factor_data_x, norm_params_train_miss_data, data_name, miss_rate, ctgan):
    ''' Handles factor encoded data sets (train and test) which have been imputed in R using MICE. Transform the data sets to one hot encoded data sets, renormalizes and saves them to csv files. 
    Normalizes the imputed test data set using the full train data set and evaluates the quality of the imputation using RMSE and PFC. 
  
    Args:
    - data_name: mushroom, letter, bank, credit or news
    - miss_rate: the probability of missing components (0.1, 0.3 or 0.5)
    - ctgan: '50', '100' or None, determines if ctgan increased training data should be used or not.
        "50": Training data is increased by 50%
        "100": Training data is increased by 100% 
        "": CTGAN is not used for increasing training data
    
    Returns:
    - rmse_num: RMSE between the original and the imputed data set for the numerical variables 
    - m_rmse: Modified RMSE between the original and the imputed data set for all variables 
    - rmse_cat: RMSE between the original and the imputed data set for the categorical variables 
    - pfc_value: Share of percently falsely classified imputed values for the categorical variables, when comparing to the original data

    '''

    # Import imputed data from R 
    if ctgan == "":
        train_imp_factor_norm_data_x, test_imp_factor_norm_data_x = data_loader_norm_factor_mice_imputed_data(data_name, miss_rate)
    else:
        train_imp_factor_norm_data_x, test_imp_factor_norm_data_x = data_loader_norm_factor_mice_imputed_data_ctgan(data_name, miss_rate, ctgan)

    # Transform to OHE data 
    train_imp_norm_ohe_data_x, test_imp_norm_ohe_data_x = factor_encode_to_ohe(train_imp_factor_norm_data_x, test_imp_factor_norm_data_x, train_factor_data_x, test_factor_data_x, data_name)

    # Renormalize data using norm_params
    train_imp_ohe_data_x = renormalize_numeric(train_imp_norm_ohe_data_x, norm_params_train_miss_data, data_name)
    test_imp_ohe_data_x = renormalize_numeric(test_imp_norm_ohe_data_x, norm_params_train_miss_data, data_name)

    # Save renormalized imputed data 
    missingness = int(miss_rate*100)
    if ctgan == "":
        filename_train_imp_mice = 'imputed_data/no_ctgan/imputed_mice_train_data/imputed_mice_{}_train_{}.csv'.format(data_name, missingness)
        train_imp_ohe_data_x.to_csv(filename_train_imp_mice, index=False)

        filename_test_imp_mice = 'imputed_data/no_ctgan/imputed_mice_test_data/imputed_mice_{}_test_{}.csv'.format(data_name, missingness)
        test_imp_ohe_data_x.to_csv(filename_test_imp_mice, index=False)
    else:
        filename_train_imp_mice = 'imputed_data/ctgan{}/imputed_mice_train_data/imputed_mice_{}_train_{}_ctgan{}.csv'.format(ctgan,data_name, missingness,ctgan)
        train_imp_ohe_data_x.to_csv(filename_train_imp_mice, index=False)

        filename_test_imp_mice = 'imputed_data/ctgan{}/imputed_mice_test_data/imputed_mice_{}_test_{}_ctgan{}.csv'.format(ctgan,data_name, missingness,ctgan)
        test_imp_ohe_data_x.to_csv(filename_test_imp_mice, index=False)

    # Load full OHE training and testing data sets
    if ctgan == "50":
        train_ohe_data_x, train_ohe_miss_data_x, test_ohe_data_x, test_ohe_miss_data_x = data_loader_ohe_wo_target_ctgan50(data_name, miss_rate) 
    elif ctgan == "100":
        train_ohe_data_x, train_ohe_miss_data_x, test_ohe_data_x, test_ohe_miss_data_x = data_loader_ohe_wo_target_ctgan100(data_name, miss_rate) 
    else:
        train_ohe_data_x, train_ohe_miss_data_x, test_ohe_data_x, test_ohe_miss_data_x = data_loader_ohe_wo_target(data_name, miss_rate) 

    # Define mask matrix using OHE data
    mask_train = 1-np.isnan(train_ohe_miss_data_x)
    mask_test = 1-np.isnan(test_ohe_miss_data_x) 

    # Normalize full training data and save parameters 
    train_full_data_norm_x, norm_params_full_data_train = normalize_numeric(train_ohe_data_x, data_name)

    # Normalize imputed test data and full test data sets using full training data params 
    test_full_data_norm_x, _ = normalize_numeric(test_ohe_data_x, data_name, norm_params_full_data_train)
    test_imp_data_norm_x, _ = normalize_numeric(test_imp_ohe_data_x, data_name, norm_params_full_data_train)

    # Calculate RMSE on normalized data and numerical columns 
    rmse_num = rmse_num_loss(test_full_data_norm_x, test_imp_data_norm_x, mask_test, data_name)

    # Calculate modified RMSE
    rmse_cat = rmse_cat_loss(test_full_data_norm_x, test_imp_data_norm_x, mask_test, data_name)
    m_rmse = m_rmse_loss(rmse_num, rmse_cat)

    # Caluclate PFC on normalized data and categorical columns 
    pfc_value = pfc(test_full_data_norm_x, test_imp_data_norm_x, mask_test, data_name)

    return rmse_num, rmse_cat, m_rmse, pfc_value 

if state == "before imputation":
    # Run before_imputation to get normalized factor encoded data 
    train_miss_factor_data_norm_x, test_miss_factor_data_norm_x, train_factor_data_x, test_factor_data_x, norm_params_train_miss_data = before_imputation(data_name, miss_rate, ctgan)

    # Export normalized factor data sets to R
    missingness = int(miss_rate*100)
    if ctgan == "":
        filename_train_norm = 'factor_preprocessed_data/data_for_mice/norm_factor_train_data_wo_target_for_mice/norm_factor_for_mice_{}_train_{}.csv'.format(data_name, missingness)
        train_miss_factor_data_norm_x.to_csv(filename_train_norm, index=False)

        filename_test_norm = 'factor_preprocessed_data/data_for_mice/norm_factor_test_data_wo_target_for_mice/norm_factor_for_mice_{}_test_{}.csv'.format(data_name, missingness)
        test_miss_factor_data_norm_x.to_csv(filename_test_norm, index=False)
    else:
        filename_train_norm = 'factor_preprocessed_data/data_for_mice/norm_factor_train_data_wo_target_for_mice_ctgan{}/norm_factor_for_mice_{}_train_{}_ctgan{}.csv'.format(ctgan,data_name, missingness, ctgan)
        train_miss_factor_data_norm_x.to_csv(filename_train_norm, index=False)

        filename_test_norm = 'factor_preprocessed_data/data_for_mice/norm_factor_test_data_wo_target_for_mice_ctgan{}/norm_factor_for_mice_{}_test_{}_ctgan{}.csv'.format(ctgan,data_name, missingness, ctgan)
        test_miss_factor_data_norm_x.to_csv(filename_test_norm, index=False)

elif state == "after imputation":
    # Find norm parameters 
    train_miss_factor_data_norm_x, test_miss_factor_data_norm_x, train_factor_data_x, test_factor_data_x, norm_params_train_miss_data = before_imputation(data_name, miss_rate, ctgan)

    # Import imputed data sets from R and find RMSE and pfc for testing data 
    rmse_num, rmse_cat, m_rmse, pfc_value = after_imputation(train_factor_data_x, test_factor_data_x, norm_params_train_miss_data, data_name, miss_rate, ctgan)

    # Print
    print(f"Numerical RMSE: {rmse_num}")
    print(f"Categorical RMSE: {rmse_cat}")
    print(f"Modified RMSE: {m_rmse}")
    print(f"PFC: {pfc_value}")

else:
    ValueError("State must be either 'before imputation' or 'after imputation'")

