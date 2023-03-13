import numpy as np
from data_loader import data_loader_factor_wo_target

data_name = "credit"
miss_rate = 10

def before_imputation(data_name, miss_rate):
    # Load data without target column, introduce missingness & dummy encode 
    train_data_x, train_miss_data_x, test_data_x, test_miss_data_x = data_loader_factor_wo_target(data_name, miss_rate) 

    # Define mask matrix 
    mask_train = 1-np.isnan(train_miss_data_x)
    mask_test = 1-np.isnan(test_miss_data_x) 

    # Normalize training missing data 
    norm_params = 1
    train_miss_data_norm_x = 1

    # Normalize testing missing data
    test_miss_data_norm_x = 2

    return train_miss_data_norm_x, test_miss_data_norm_x, norm_params, mask_train, mask_test


def after_imputation(norm_params, data_name, miss_rate, train_imp_data_norm_x, test_imp_data_norm_x):
    
    # Renormalize data using norm_params
    train_imp_data_x = None
    test_imp_data_x = None

    # Load full training and testing data sets 


    # Normalize full training data


    # Normalize imputed data and full data sets using training data params 


    # Calculate RMSE on normalized data and numerical columns 

    # Caluclate RMSE on normalized data and categorical columns 

    return 1 


# Run before_imputation to get normalized data and export data sets to R

# Import imputed data sets from R 

# Run after imputation and 