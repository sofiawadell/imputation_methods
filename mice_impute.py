import numpy as np
from data_loader import data_loader_factor_wo_target
from data_loader import data_loader_mice_imputed_data
from utils import normalize_numeric
from utils import renormalize_numeric

# Parameters to adjust
data_name = "bank"
miss_rate = 0.1
state = "before imputation"

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

def after_imputation(norm_params, data_name, miss_rate):
    
    # Save normalizing parameters for training data with missingness
    norm_params_train_miss_data = norm_params

    # Import data from R 
    train_imp_norm_data_x, test_imp_norm_data_x = data_loader_mice_imputed_data(data_name, miss_rate)

    # Renormalize data using norm_params
    train_imp_data_x = renormalize_numeric(train_imp_norm_data_x, norm_params_train_miss_data, data_name)
    test_imp_data_x = renormalize_numeric(test_imp_norm_data_x, norm_params_train_miss_data, data_name)

    # Save renormalized imputed data 
    missingness = int(miss_rate*100)

    filename_train_imp_mice = 'imputed_mice_train_data/imputed_mice_{}_train_{}.csv'.format(data_name, missingness)
    train_imp_data_x.to_csv(filename_train_imp_mice, index=False)

    filename_test_imp_mice = 'imputed_mice_test_data/imputed_mice_{}_test_{}.csv'.format(data_name, missingness)
    test_imp_data_x.to_csv(filename_test_imp_mice, index=False)

    # Load full training and testing data sets 
    train_data_x, train_miss_data_x, test_data_x, test_miss_data_x = data_loader_factor_wo_target(data_name, miss_rate) 

    # Normalize full training data and save parameters 
    train_full_data_norm_x, norm_params_full_data_train = normalize_numeric(train_data_x, data_name)

    # Normalize imputed test data and full test data sets using full training data params 
    test_full_data_norm_x, _ = normalize_numeric(test_data_x, data_name, norm_params_full_data_train)
    test_imp_data_norm_x, _ = normalize_numeric(test_imp_data_x, data_name, norm_params_full_data_train)

    # Calculate RMSE on normalized data and numerical columns 

    # Caluclate PFC on normalized data and categorical columns 
    pfc = 1
    
    return train_imp_data_x, test_imp_data_x, rmse, pfc 

if state == "before imputation":
    # Run before_imputation to get normalized data 
    train_miss_data_norm_x, test_miss_data_norm_x, norm_params, mask_train, mask_test = before_imputation(data_name, miss_rate)

    # Export data sets to R
    missingness = int(miss_rate*100)
    filename_train_norm = 'norm_factor_train_data_wo_target/norm_factor_encode_{}_train_{}.csv'.format(data_name, missingness)
    train_miss_data_norm_x.to_csv(filename_train_norm, index=False)

    filename_test_norm = 'norm_factor_test_data_wo_target/norm_factor_encode_{}_test_{}.csv'.format(data_name, missingness)
    test_miss_data_norm_x.to_csv(filename_test_norm, index=False)

elif state == "after imputation":
    # Find mask matrix and norm parameters 
    train_miss_data_norm_x, test_miss_data_norm_x, norm_params, mask_train, mask_test = before_imputation(data_name, miss_rate)

    # Import imputed data sets from R and find RMSE and pfc for testing data 
    train_imp_data_x, test_imp_data_x, rmse, pfc = after_imputation(norm_params, data_name, miss_rate)

    # Export non-normalized 

else:
    ValueError("State must be either 'before imputation' or 'after imputation'")

