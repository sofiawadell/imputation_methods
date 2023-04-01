import numpy as np
import pandas as pd

from data_loader import data_loader_factor_wo_target
from data_loader import data_loader_ohe_wo_target

from data_loader import data_loader_factor_wo_target_ctgan50
from data_loader import data_loader_factor_wo_target_ctgan100
from data_loader import data_loader_ohe_wo_target_ctgan50
from data_loader import data_loader_ohe_wo_target_ctgan100

from missforest_impute import missforest_impute
from knn_impute import knn_impute
from median_mode_impute import median_mode_impute

from utils import normalize_numeric
from utils import renormalize_numeric
from utils import rounding_discrete

''' Description

Methods for normalizing, imputing and renormalizing data set using the imputation methods MissForest, kNN or median/mode imputation

'''

def run_mf(data_name, miss_rate, ctgan):
    ''' Prepares a data set for being imputed with the MissForest method by loading and normalizing it, calls the method to initiate imputation 
    and renormalizes the data set after the imputation.
  
    Args:
    - data_name: mushroom, letter, bank, credit or news
    - miss_rate: the probability of missing components (0.1, 0.3 or 0.5)
    - ctgan: '50', '100' or None, determines if ctgan increased training data should be used or not.
        "50": Training data is increased by 50%
        "100": Training data is increased by 100% 
        None. CTGAN is not used for increasing training data
    
    Returns:
    - train_imp_data_x: Imputed one hot encoded renormalized train data set
    - test_imp_data_x: Imputed one hot encoded renormalized test data set
    - train_data_x: Full one hot encoded train data set
    - test_data_x: Full one hot encoded test data set
    - mask_train: Mask matrix indicating which values that were missing before imputation in train data set
    - mask_test: Mask matrix indicating which values that were missing before imputation in test data set

    '''
    
    # Load OHE data without target column depending on ctgan variable
    if ctgan == "50":
        train_data_x, train_miss_data_x, test_data_x, test_miss_data_x = data_loader_ohe_wo_target_ctgan50(data_name, miss_rate) 
    elif ctgan == "100":
        train_data_x, train_miss_data_x, test_data_x, test_miss_data_x = data_loader_ohe_wo_target_ctgan100(data_name, miss_rate) 
    else:            
        train_data_x, train_miss_data_x, test_data_x, test_miss_data_x = data_loader_ohe_wo_target(data_name, miss_rate) 

    # Define mask matrix
    mask_train = pd.DataFrame(1-np.isnan(train_miss_data_x.values))
    mask_test = pd.DataFrame(1-np.isnan(test_miss_data_x.values)) 

    # Normalize the test data set using the norm_parameters from the training data with missingness
    train_miss_norm_data_x, norm_params_train_miss_data = normalize_numeric(train_miss_data_x, data_name)
    test_miss_norm_data_x, _ = normalize_numeric(test_miss_data_x, data_name, norm_params_train_miss_data)

    # Impute data using missForest
    test_imp_norm_data_x_np, train_imp_norm_data_x_np = missforest_impute(train_miss_norm_data_x, test_miss_norm_data_x, data_name)
    
    # Transform to pandas array to keep the column names 
    train_imp_norm_data_x = pd.DataFrame(train_imp_norm_data_x_np, columns=test_miss_data_x.columns) 
    test_imp_norm_data_x = pd.DataFrame(test_imp_norm_data_x_np, columns=test_miss_data_x.columns) 

    # Renormalize data sets
    train_imp_data_x = renormalize_numeric(train_imp_norm_data_x, norm_params_train_miss_data, data_name)
    test_imp_data_x = renormalize_numeric(test_imp_norm_data_x, norm_params_train_miss_data, data_name)

    return train_imp_data_x, test_imp_data_x, train_data_x, test_data_x, mask_train, mask_test  

def run_kNN(data_name, miss_rate, best_k, ctgan):

    ''' Prepares a data set for being imputed with the SciKitLearn kNNImputer method by loading and normalizing it, calls the method to initiate imputation 
    and renormalizes the data set after the imputation.
  
    Args:
    - data_name: mushroom, letter, bank, credit or news
    - miss_rate: the probability of missing components (0.1, 0.3 or 0.5)
    - best_k: Optimal number of neighbors for the data set data_name
    - ctgan: '50', '100' or None, determines if ctgan increased training data should be used or not.
        "50": Training data is increased by 50%
        "100": Training data is increased by 100% 
        None. CTGAN is not used for increasing training data
    
    Returns:
    - train_imp_data_x: Imputed one hot encoded renormalized train data set
    - test_imp_data_x: Imputed one hot encoded renormalized test data set
    - train_data_x: Full one hot encoded train data set
    - test_data_x: Full one hot encoded test data set
    - mask_train: Mask matrix indicating which values that were missing before imputation in train data set
    - mask_test: Mask matrix indicating which values that were missing before imputation in test data set

    '''

    # Load OHE data without target column depending on ctgan variable
    if ctgan == "50":
        train_data_x, train_miss_data_x, test_data_x, test_miss_data_x = data_loader_ohe_wo_target_ctgan50(data_name, miss_rate) 
    elif ctgan == "100":
        train_data_x, train_miss_data_x, test_data_x, test_miss_data_x = data_loader_ohe_wo_target_ctgan100(data_name, miss_rate) 
    else:            
        train_data_x, train_miss_data_x, test_data_x, test_miss_data_x = data_loader_ohe_wo_target(data_name, miss_rate) 

    # Define mask matrix
    mask_train = 1-np.isnan(train_miss_data_x)
    mask_test = 1-np.isnan(test_miss_data_x) 

    # Normalize the test data set using the norm_parameters from the training data with missingness
    train_miss_norm_data_x, norm_params_train_miss_data = normalize_numeric(train_miss_data_x, data_name)
    test_miss_norm_data_x, _ = normalize_numeric(test_miss_data_x, data_name, norm_params_train_miss_data)

    # Transform to numpy arrays
    train_miss_norm_data_x_np, test_miss_norm_data_x_np = train_miss_norm_data_x.values, test_miss_norm_data_x.values

    # Impute data using kNN
    train_imp_norm_data_x_np, test_imp_norm_data_x_np = knn_impute(train_miss_norm_data_x_np, test_miss_norm_data_x_np, best_k)

    # Round categorical variables 
    train_imp_norm_data_x_np = rounding_discrete(train_imp_norm_data_x_np, train_miss_norm_data_x_np, data_name)
    test_imp_norm_data_x_np = rounding_discrete(test_imp_norm_data_x_np, test_miss_norm_data_x_np, data_name)

    # Transform to pandas array to keep the column names 
    train_imp_norm_data_x = pd.DataFrame(train_imp_norm_data_x_np, columns=test_miss_data_x.columns) 
    test_imp_norm_data_x = pd.DataFrame(test_imp_norm_data_x_np, columns=test_miss_data_x.columns) 


    # Renormalize data sets
    train_imp_data_x = renormalize_numeric(train_imp_norm_data_x, norm_params_train_miss_data, data_name)
    test_imp_data_x = renormalize_numeric(test_imp_norm_data_x, norm_params_train_miss_data, data_name)

    return train_imp_data_x, test_imp_data_x, train_data_x, test_data_x, mask_train, mask_test


def run_median_mode(data_name, miss_rate, ctgan):

    ''' Prepares a data set for being imputed with the median/mode method (where numerical variables are imputed by median imputation 
    and categorical variables are imputed by mode imputation) by loading and normalizing it, calls the method to initiate imputation 
    and renormalizes the data set after the imputation.
  
    Args:
    - data_name: mushroom, letter, bank, credit or news
    - miss_rate: the probability of missing components (0.1, 0.3 or 0.5)
    - ctgan: '50', '100' or None, determines if ctgan increased training data should be used or not.
        "50": Training data is increased by 50%
        "100": Training data is increased by 100% 
        None. CTGAN is not used for increasing training data
    
    Returns:
    - train_imp_data_x: Imputed one hot encoded renormalized train data set
    - test_imp_data_x: Imputed one hot encoded renormalized test data set
    - train_data_x: Full one hot encoded train data set
    - test_data_x: Full one hot encoded test data set
    - mask_train: Mask matrix indicating which values that were missing before imputation in train data set
    - mask_test: Mask matrix indicating which values that were missing before imputation in test data set

    '''

    # Load data without target column, depending on ctgan variable
    if ctgan == "50":
        train_factor_data_x, train_factor_miss_data_x, test_factor_data_x, test_factor_miss_data_x = data_loader_factor_wo_target_ctgan50(data_name, miss_rate) 
        train_data_ohe_x, train_ohe_miss_data_x, test_data_ohe_x, test_ohe_miss_data_x = data_loader_ohe_wo_target_ctgan50(data_name, miss_rate) 
    elif ctgan == "100":
        train_factor_data_x, train_factor_miss_data_x, test_factor_data_x, test_factor_miss_data_x = data_loader_factor_wo_target_ctgan100(data_name, miss_rate) 
        train_data_ohe_x, train_ohe_miss_data_x, test_data_ohe_x, test_ohe_miss_data_x = data_loader_ohe_wo_target_ctgan100(data_name, miss_rate) 
    else:
        train_factor_data_x, train_factor_miss_data_x, test_factor_data_x, test_factor_miss_data_x = data_loader_factor_wo_target(data_name, miss_rate) 
        train_data_ohe_x, train_ohe_miss_data_x, test_data_ohe_x, test_ohe_miss_data_x = data_loader_ohe_wo_target(data_name, miss_rate) 

    # Define mask matrix using OHE data
    mask_train = 1-np.isnan(train_ohe_miss_data_x)
    mask_test = 1-np.isnan(test_ohe_miss_data_x) 

    # Normalize the test data set using the norm_parameters from the training data with missingness
    train_factor_miss_norm_data_x, norm_params_train_miss_data = normalize_numeric(train_factor_miss_data_x, data_name)
    test_factor_miss_norm_data_x, _ = normalize_numeric(test_factor_miss_data_x, data_name, norm_params_train_miss_data)
    
    # Impute using median/mode strategy and transform to OHE data
    train_imputed_norm_ohe_data_x, test_imputed_norm_ohe_data_x = median_mode_impute(train_factor_miss_norm_data_x, test_factor_miss_norm_data_x, train_factor_data_x, test_factor_data_x, data_name)

    # Renormalize data sets
    train_imp_data_x = renormalize_numeric(train_imputed_norm_ohe_data_x, norm_params_train_miss_data, data_name)
    test_imp_data_x = renormalize_numeric(test_imputed_norm_ohe_data_x, norm_params_train_miss_data, data_name)

    return train_imp_data_x, test_imp_data_x, train_data_ohe_x, test_data_ohe_x, mask_train, mask_test 
