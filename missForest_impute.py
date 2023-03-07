
import sklearn.neighbors._base
import sys

from missingpy import MissForest


def missForest_impute(train_miss_data_x, test_miss_data_x):
    
    # Required to use missingpys
    sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base

    # Define categorical columns
    _, dim_train = train_miss_data_x.shape
    cat_index = list(range(13,dim_train))

    # Create a MissForest model and train it - no tuning required 
    imputer = MissForest(random_state=0,max_iter = 2, verbose = 0)    
    tmp = imputer.fit(train_miss_data_x, cat_vars = cat_index)
    train_imputed_data_x_mf = tmp.transform(train_miss_data_x)
    

    # Impute test data using final model 
    test_imputed_data_x_mf = imputer.transform(test_miss_data_x)
    
    return test_imputed_data_x_mf, train_imputed_data_x_mf