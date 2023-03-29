
from datasets import datasets
from sklearn.impute import SimpleImputer
from factor_encode_to_ohe import factor_encode_to_ohe

import pandas as pd 
import numpy as np
import time

def median_mode_impute(train_factor_miss_norm_data_x, test_factor_miss_norm_data_x, train_data_x, test_data_x, data_name):

    ''' Imputes a data set with missing values using median imputation for the numerical variables and mode imputation for the categorical variables. Prints the elapsed time of the imputation.
  
    Args:
    - train_factor_miss_norm_data_x: Normalized factor encoded train data with missing components represented as np.nan
    - test_factor_miss_norm_data_x: Normalized factor encoded train data with missing components represented as np.nan
    - train_data_x: Full train data
    - test_data_x: Full test data
    - data_name: mushroom, letter, bank, credit or news
    
    Returns:
    - train_imputed_norm_ohe_data_x: Imputed one hot encoded train data 
    - test_imputed_norm_ohe_data_x: Imputed one hot encoded test data 

    '''

    N_num_cols = len(datasets[data_name]["num_cols"])
    N_cat_cols = len(datasets[data_name]["cat_cols"])
    
    # Start timer
    start_time = time.time() 

    # Define the imputers for numeric and categorical columns
    numeric_imputer = SimpleImputer(strategy='median')
    categorical_imputer = SimpleImputer(strategy='most_frequent')

    # Create a copy for imputing
    train_imputed_norm_data_x = train_factor_miss_norm_data_x.copy()
    test_imputed_norm_data_x = test_factor_miss_norm_data_x.copy()

    if(N_num_cols != 0):
        # Impute the missing values for the numeric columns
        numeric_cols = train_imputed_norm_data_x.columns[:N_num_cols]
        numeric_imputer.fit(train_imputed_norm_data_x[numeric_cols])
        train_imputed_norm_data_x[numeric_cols] = numeric_imputer.transform(train_imputed_norm_data_x[numeric_cols])
        test_imputed_norm_data_x[numeric_cols] = numeric_imputer.transform(test_imputed_norm_data_x[numeric_cols])

    if(N_cat_cols != 0):
        # Impute the missing values for the categorical columns
        categorical_cols = train_imputed_norm_data_x.columns[N_num_cols:]
        categorical_imputer.fit(train_imputed_norm_data_x[categorical_cols])
        train_imputed_norm_data_x[categorical_cols] = categorical_imputer.transform(train_imputed_norm_data_x[categorical_cols])
        test_imputed_norm_data_x[categorical_cols] = categorical_imputer.transform(test_imputed_norm_data_x[categorical_cols])

    # End timer and print result
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")

    # OHE data
    train_imputed_norm_ohe_data_x, test_imputed_norm_ohe_data_x = factor_encode_to_ohe(train_imputed_norm_data_x, test_imputed_norm_data_x, train_data_x, test_data_x, data_name)
    
    return train_imputed_norm_ohe_data_x, test_imputed_norm_ohe_data_x
