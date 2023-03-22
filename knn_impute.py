import pandas as pd
import numpy as np

from data_loader import data_loader_factor_wo_target
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, KFold

def knn_impute(train_miss_norm_data_x_np, test_miss_norm_data_x_np, best_k):

    # Impute the missing values using the best_k value
    imputer = KNNImputer(n_neighbors=best_k)
    train_imp_norm_data_x_np = imputer.fit_transform(train_miss_norm_data_x_np)
    test_imp_norm_data_x_np = imputer.transform(test_miss_norm_data_x_np)
    
    return train_imp_norm_data_x_np, test_imp_norm_data_x_np
