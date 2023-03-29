
import sklearn.neighbors._base
import sys
import time
import pandas as pd
from datasets import datasets

from missingpy import MissForest

''' Description

Imputes data set using the imputation methods MissForest, kNN or median/mode imputation and prints the result

'''


def missforest_impute(train_miss_norm_data_x, test_miss_norm_data_x, data_name):

    ''' Imputes a data set with missing values using the MissForest imputation method. Prints the elapsed time of the imputation.
  
    Args:
    - train_miss_norm_data_x: Normalized one hot encoded train data with missing components represented as np.nan
    - test_miss_norm_data_x: Normalized one hot encoded train data with missing components represented as np.nan
    - data_name: mushroom, letter, bank, credit or news
    
    Returns:
    - test_imp_norm_data_x: Imputed one hot encoded train data 
    - train_imp_norm_data_x: Imputed one hot encoded test data 

    '''

    # Required to use missingpys
    sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base

    # Define categorical columns
    _, dim_train = train_miss_norm_data_x.shape
    N_num_cols = len(datasets[data_name]["num_cols"])
    cat_index = list(range(N_num_cols,dim_train))

    # Case if no categorical columns
    if len(cat_index) == 0:
     cat_index = None

    # Begin timer
    start_time = time.time()

    # Create a MissForest model and train it - no tuning required 
    imputer = MissForest(random_state = 0, verbose = 0)    
    tmp = imputer.fit(train_miss_norm_data_x, cat_vars = cat_index)
    train_imp_norm_data_x = tmp.transform(train_miss_norm_data_x)

    print("Training done")
    
    # Impute test data using final model 
    test_imp_norm_data_x = imputer.transform(test_miss_norm_data_x)

    # End timer
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"Elapsed time: {elapsed_time} seconds")
    
    return test_imp_norm_data_x, train_imp_norm_data_x