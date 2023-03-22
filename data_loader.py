
# Import datasets
from datasets import datasets

# Necessary packages
import numpy as np
import pandas as pd



def data_loader_full(data_name, miss_rate):
## Load training data
    if data_name in datasets.keys():
        full_data = pd.read_csv('original_data/'+data_name+'.csv')    
    else:
        ValueError("Dataset not found")            
    return full_data



def data_loader(data_name, miss_rate):
  '''Loads datasets and introduce missingness.
  
  Args:
    - data_name: 
    - miss_rate: the probability of missing components
    
  Returns:
    train_data_x: original data
    train_miss_data_x: data with missing values
    # train_data_m: indicator matrix for missing components

    test_data_x: original data
    test_miss_data_x: data with missing values
    # test_data_m: indicator matrix for missing components
  '''
  missingness_str = str(int(miss_rate*100))

  ## Load training data
  if data_name in datasets.keys():
    train_data_x = pd.read_csv('train_data/'+data_name+'_train.csv')    
  else:
    ValueError("Dataset not found")
  
  # Load train missing data with correct missingness
  if data_name in datasets.keys():   
    train_miss_data_x = pd.read_csv('train_data/'+data_name+'_train_'+missingness_str+'.csv')
  else:
    ValueError("Dataset not found")

  ## Load test data
  if data_name in datasets.keys():
    test_data_x = pd.read_csv('test_data/'+data_name+'_test.csv') 
  else:
    ValueError("Dataset not found")

# Load test missing data with correct missingness
  if data_name in datasets.keys():   
    test_miss_data_x = pd.read_csv('test_data/'+data_name+'_test_'+missingness_str+'.csv') 
  else:
    ValueError("Dataset not found")
      
  return train_data_x, train_miss_data_x, test_data_x, test_miss_data_x

def data_loader_ohe(data_name, miss_rate):
    missingness_str = str(int(miss_rate*100))

    ## Load training data
    if data_name in datasets.keys():
        train_data_x = pd.read_csv('one_hot_train_data/one_hot_'+data_name+'_train.csv')    
    else:
        ValueError("Dataset not found")
    
    # Load train missing data with correct missingness
    if data_name in datasets.keys():   
        train_miss_data_x = pd.read_csv('one_hot_train_data/one_hot_'+data_name+'_train_'+missingness_str+'.csv')
    else:
        ValueError("Dataset not found")

    ## Load test data
    if data_name in datasets.keys():
        test_data_x = pd.read_csv('one_hot_test_data/one_hot_'+data_name+'_test.csv') 
    else:
        ValueError("Dataset not found")

    # Load test missing data with correct missingness
    if data_name in datasets.keys():   
        test_miss_data_x = pd.read_csv('one_hot_test_data/one_hot_'+data_name+'_test_'+missingness_str+'.csv') 
    else:
        ValueError("Dataset not found")
        
    return train_data_x, train_miss_data_x, test_data_x, test_miss_data_x

def data_loader_ohe_wo_target(data_name, miss_rate):
    missingness_str = str(int(miss_rate*100))

    ## Load training data
    if data_name in datasets.keys():
        train_data_x = pd.read_csv('one_hot_train_data_wo_target/one_hot_'+data_name+'_train.csv')    
    else:
        ValueError("Dataset not found")
    
    # Load train missing data with correct missingness
    if data_name in datasets.keys():   
        train_miss_data_x = pd.read_csv('one_hot_train_data_wo_target/one_hot_'+data_name+'_train_'+missingness_str+'.csv')
    else:
        ValueError("Dataset not found")

    ## Load test data
    if data_name in datasets.keys():
        test_data_x = pd.read_csv('one_hot_test_data_wo_target/one_hot_'+data_name+'_test.csv') 
    else:
        ValueError("Dataset not found")

    # Load test missing data with correct missingness
    if data_name in datasets.keys():   
        test_miss_data_x = pd.read_csv('one_hot_test_data_wo_target/one_hot_'+data_name+'_test_'+missingness_str+'.csv') 
    else:
        ValueError("Dataset not found")
        
    return train_data_x, train_miss_data_x, test_data_x, test_miss_data_x

def data_loader_ohe_for_mice_wo_target(data_name, miss_rate):
    missingness_str = str(int(miss_rate*100))

    ## Load training data
    if data_name in datasets.keys():
        train_data_x = pd.read_csv('one_hot_train_data_wo_target_for_mice/one_hot_for_mice_'+data_name+'_train.csv')    
    else:
        ValueError("Dataset not found")
    
    # Load train missing data with correct missingness
    if data_name in datasets.keys():   
        train_miss_data_x = pd.read_csv('one_hot_train_data_wo_target_for_mice/one_hot_for_mice_'+data_name+'_train_'+missingness_str+'.csv')
    else:
        ValueError("Dataset not found")

    ## Load test data
    if data_name in datasets.keys():
        test_data_x = pd.read_csv('one_hot_test_data_wo_target_for_mice/one_hot_for_mice_'+data_name+'_test.csv') 
    else:
        ValueError("Dataset not found")

    # Load test missing data with correct missingness
    if data_name in datasets.keys():   
        test_miss_data_x = pd.read_csv('one_hot_test_data_wo_target_for_mice/one_hot_for_mice_'+data_name+'_test_'+missingness_str+'.csv') 
    else:
        ValueError("Dataset not found")
        
    return train_data_x, train_miss_data_x, test_data_x, test_miss_data_x    

def data_loader_factor(data_name, miss_rate):
    missingness_str = str(int(miss_rate*100))

    ## Load training data
    if data_name in datasets.keys():
        train_data_x = pd.read_csv('factor_train_data/factor_encode_'+data_name+'_train.csv')    
    else:
        ValueError("Dataset not found")
    
    # Load train missing data with correct missingness
    if data_name in datasets.keys():   
        train_miss_data_x = pd.read_csv('factor_train_data/factor_encode_'+data_name+'_train_'+missingness_str+'.csv')
    else:
        ValueError("Dataset not found")

    ## Load test data
    if data_name in datasets.keys():
        test_data_x = pd.read_csv('factor_test_data/factor_encode_'+data_name+'_test.csv') 
    else:
        ValueError("Dataset not found")

    # Load test missing data with correct missingness
    if data_name in datasets.keys():   
        test_miss_data_x = pd.read_csv('factor_test_data/factor_encode_'+data_name+'_test_'+missingness_str+'.csv') 
    else:
        ValueError("Dataset not found")
        
    return train_data_x, train_miss_data_x, test_data_x, test_miss_data_x    



def data_loader_factor_wo_target(data_name, miss_rate):
    missingness_str = str(int(miss_rate*100))

    ## Load training data
    if data_name in datasets.keys():
        train_data_x = pd.read_csv('factor_train_data_wo_target/factor_encode_'+data_name+'_train.csv')    
    else:
        ValueError("Dataset not found")
    
    # Load train missing data with correct missingness
    if data_name in datasets.keys():   
        train_miss_data_x = pd.read_csv('factor_train_data_wo_target/factor_encode_'+data_name+'_train_'+missingness_str+'.csv')
    else:
        ValueError("Dataset not found")

    ## Load test data
    if data_name in datasets.keys():
        test_data_x = pd.read_csv('factor_test_data_wo_target/factor_encode_'+data_name+'_test.csv') 
    else:
        ValueError("Dataset not found")

    # Load test missing data with correct missingness
    if data_name in datasets.keys():   
        test_miss_data_x = pd.read_csv('factor_test_data_wo_target/factor_encode_'+data_name+'_test_'+missingness_str+'.csv') 
    else:
        ValueError("Dataset not found")
        
    return train_data_x, train_miss_data_x, test_data_x, test_miss_data_x    

def data_loader_norm_mice_imputed_data(data_name, miss_rate):
    missingness_str = str(int(miss_rate*100))
    
    # Load normalized imputed train missing data with correct missingness
    if data_name in datasets.keys():   
        train_imp_norm_data_x = pd.read_csv('norm_imputed_mice_train_data/norm_imputed_mice_'+data_name+'_train_'+missingness_str+'.csv')
    else:
        ValueError("Dataset not found")

    # Load normalized imputed test missing data with correct missingness
    if data_name in datasets.keys():   
        test_imp_norm_data_x = pd.read_csv('norm_imputed_mice_test_data/norm_imputed_mice_'+data_name+'_test_'+missingness_str+'.csv') 
    else:
        ValueError("Dataset not found")
        
    return train_imp_norm_data_x, test_imp_norm_data_x