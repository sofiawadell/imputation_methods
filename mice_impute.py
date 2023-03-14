import numpy as np
from data_loader import data_loader_factor_wo_target
from utils import normalize_numeric

data_name = "credit"
miss_rate = 0.1

def before_imputation(data_name, miss_rate):
    # Load factor encoded data without target column 
    train_data_x, train_miss_data_x, test_data_x, test_miss_data_x = data_loader_factor_wo_target(data_name, miss_rate) 

    # Define mask matrix 
    mask_train = 1-np.isnan(train_miss_data_x)
    mask_test = 1-np.isnan(test_miss_data_x) 

    # Normalize training missing data 
    train_miss_data_norm_x, norm_params = normalize_numeric(train_miss_data_x, data_name)

    # Normalize testing missing data
    test_miss_data_norm_x, _ = normalize_numeric(test_miss_data_x, data_name, norm_params)

    return train_miss_data_norm_x, test_miss_data_norm_x, norm_params, mask_train, mask_test


def after_imputation(norm_params, data_name, miss_rate, train_imp_data_norm_x, test_imp_data_norm_x):
    
    # Renormalize data using norm_params
    train_imp_data_x = None
    test_imp_data_x = None

    # Load full training and testing data sets 


    # Normalize full training data


    # Normalize imputed data and full data sets using training data params 


    # Calculate RMSE on normalized data and numerical columns 

    # Caluclate PFC on normalized data and categorical columns 
    pfc = 1
    return pfc 


# Run before_imputation to get normalized data 
train_miss_data_norm_x, test_miss_data_norm_x, norm_params, mask_train, mask_test = before_imputation(data_name, miss_rate)

# Export data sets to R
missingness = int(miss_rate*100)
filename_train_norm = 'norm_factor_train_data_wo_target/norm_factor_encode_{}_train_{}.csv'.format(data_name, missingness)
train_miss_data_norm_x.to_csv(filename_train_norm, index=False)

filename_test_norm = 'norm_factor_test_data_wo_target/norm_factor_encode_{}_test_{}.csv'.format(data_name, missingness)
test_miss_data_norm_x.to_csv(filename_test_norm, index=False)

# Import imputed data sets from R 

# Run after imputation and find RMSE and pfc for testing data 