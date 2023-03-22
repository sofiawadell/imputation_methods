
import sklearn.neighbors._base
import sys
import pandas as pd
from datasets import datasets

from missingpy import MissForest

def missForest_impute(train_miss_norm_data_x, test_miss_norm_data_x, data_name):

    # Required to use missingpys
    sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base

    # Define categorical columns
    _, dim_train = train_miss_norm_data_x.shape
    N_num_cols = len(datasets[data_name]["num_cols"])
    cat_index = list(range(N_num_cols,dim_train))

    # Case if no categorical columns
    if len(cat_index) == 0:
     cat_index = None

    # Create a MissForest model and train it - no tuning required 
    imputer = MissForest(random_state = 0, verbose = 0)    
    tmp = imputer.fit(train_miss_norm_data_x, cat_vars = cat_index)
    train_imp_norm_data_x = tmp.transform(train_miss_norm_data_x)

    print("Training done")
    
    # Impute test data using final model 
    test_imp_norm_data_x = imputer.transform(test_miss_norm_data_x)
    
    return test_imp_norm_data_x, train_imp_norm_data_x