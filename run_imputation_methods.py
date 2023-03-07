from data_loader import data_loader
from data_loader import data_loader_ohe
from missForest_impute import missForest_impute
from mice_impute import mice_impute
from knn_impute import knn_impute
import numpy as np

# from utils import normalize_num_data


def run_mf(data_name, miss_rate, norm_parameters):
    # Load OHE data
    train_data_x, train_miss_data_x, test_data_x, test_miss_data_x = data_loader_ohe(data_name, miss_rate) 

    # Find mask matrix 


    # Normalize all data sets using the norm_parameters 
    data = [train_data_x,train_miss_data_x, test_data_x, test_miss_data_x]
    train_data_norm_x, train_miss_data_norm_x, test_data_norm_x, test_miss_data_norm_x = [normalize_num_data(d,norm_parameters) for d in data]

    # Impute data using missForest
    test_imputed_norm_data_x, train_imputed_norm_data_x = missForest_impute(train_miss_data_norm_x, test_miss_data_norm_x)

    return train_data_norm_x, train_imputed_norm_data_x, test_data_norm_x, test_imputed_norm_data_x

def run_MICE(data_name, miss_rate, norm_parameters):

    # Load data, introduce missingness & dummy encode 
    train_data_x, train_miss_data_x, test_data_x, test_miss_data_x = data_loader(data_name, miss_rate) 
    
    # TBU: Dummy encode data

    
    # Find mask matrix 


    # Normalize the numerical values of all data sets 
    data = [train_data_x,train_miss_data_x, test_data_x, test_miss_data_x]
    train_data_norm_x, train_miss_data_norm_x, test_data_norm_x, test_miss_data_norm_x = [normalize_num_data(d,norm_parameters) for d in data]
    
    # Transform using MICE
    test_imputed_norm_data_x, train_imputed_norm_data_x = mice_impute(train_miss_data_x, test_miss_data_x)

    return train_data_norm_x, train_imputed_norm_data_x, test_data_norm_x, test_imputed_norm_data_x

def run_kNN(data_name, miss_rate, norm_parameters):
    # Load OHE data
    train_data_x, train_miss_data_x, test_data_x, test_miss_data_x = data_loader_ohe(data_name, miss_rate) 

    # Define mask matrix
    train_data_m = 1-np.isnan(train_data_x)
    test_data_m = 1-np.isnan(test_data_x)

    # # Normalize the numerical values of all data sets 
    # data = [train_data_x,train_miss_data_x, test_data_x, test_miss_data_x]
    # train_data_norm_x, train_miss_data_norm_x, test_data_norm_x, test_miss_data_norm_x = [normalize_num_data(d,norm_parameters) for d in data]

    # Transform to numpy arrays
    train_data_x_np, train_miss_data_x_np, test_data_x_np, test_miss_data_x_np = train_data_x.values, train_miss_data_x.values, test_data_x.values, test_miss_data_x.values

    # Impute data using kNN
    test_imputed_norm_data_x, train_imputed_norm_data_x = knn_impute(train_data_x_np, train_miss_data_x_np, test_data_x_np, test_miss_data_x_np)

    return train_data_x, train_imputed_norm_data_x, test_data_x, test_imputed_norm_data_x

def run_median_mode(data_name, miss_rate, norm_parameters):
    # Load data, introduce missingness & dummy encode 
    train_data_x, train_miss_data_x, test_data_x, test_miss_data_x = data_loader(data_name, miss_rate) 
    # TBU: Dummy encode data

    # Find mask matrix 


    # Normalize the numerical values of all data sets 
    data = [train_data_x,train_miss_data_x, test_data_x, test_miss_data_x]
    train_data_norm_x, train_miss_data_norm_x, test_data_norm_x, test_miss_data_norm_x = [normalize_num_data(d,norm_parameters) for d in data]
    
    # Impute using median/mode strategy 
    test_imputed_norm_data_x, train_imputed_norm_data_x = median_mode_impute(train_miss_data_x, test_miss_data_x)

    return train_data_norm_x, train_imputed_norm_data_x, test_data_norm_x, test_imputed_norm_data_x

# def save_imputed_data():
    # missingness = 10
    # dataset = "credit"
    # mode = "train"
    # train_imputed_data_x_mf_pd = pd.DataFrame(train_imputed_data_x_mf)
    # save_filename_missing = '{}{}_{}{}_{}_{}.csv'.format('one_hot_', mode, 'data/one_hot_', dataset, mode, missingness)
    # train_imputed_data_x_mf_pd.to_csv(save_filename_missing, index=False)