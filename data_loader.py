
# Imports
from datasets import datasets
import numpy as np
import pandas as pd

'''
Description: Data loader for UCI mushroom, letter, bank, credit and news datasets.

'''


def data_loader(data_name, miss_rate):
  
    '''Loads training and test datasets with and without missingness.
  
    Args:
    - data_name: mushroom, letter, bank, credit or news
    - miss_rate: the probability of missing components (0.1, 0.3 or 0.5)
    
    Returns:
    train_data_x: original data
    train_miss_data_x: data with missing values

    test_data_x: original data
    test_miss_data_x: data with missing values
    '''

    # Define missingness in percentages
    missingness_str = str(int(miss_rate*100))

    ## Load training data
    if data_name in datasets.keys():
        train_data_x = pd.read_csv('train_test_split_data/train_data/'+data_name+'_train.csv')    
    else:
        ValueError("Dataset not found")
    
    # Load train missing data with missingness
    if data_name in datasets.keys():   
        train_miss_data_x = pd.read_csv('train_test_split_data/train_data/'+data_name+'_train_'+missingness_str+'.csv')
    else:
        ValueError("Dataset not found")

    ## Load test data
    if data_name in datasets.keys():
        test_data_x = pd.read_csv('train_test_split_data/test_data/'+data_name+'_test.csv') 
    else:
        ValueError("Dataset not found")

    # Load test missing data with missingness
    if data_name in datasets.keys():   
        test_miss_data_x = pd.read_csv('train_test_split_data/test_data/'+data_name+'_test_'+missingness_str+'.csv') 
    else:
        ValueError("Dataset not found")
        
    return train_data_x, train_miss_data_x, test_data_x, test_miss_data_x


def data_loader_ohe(data_name, miss_rate):

    '''Loads training and test datasets where the categorical columns are one hot encoded, with and without missingness.
  
    Args:
    - data_name: mushroom, letter, bank, credit or news
    - miss_rate: the probability of missing components (0.1, 0.3 or 0.5)
    
    Returns:
    train_ohe_data_x: one hot encoded original data 
    train_ohe_miss_data_x: one hot encoded data with missing values 

    test_ohe_data_x: one hot encoded original data 
    test_ohe_miss_data_x: one hot encoded data with missing values 
    '''
    
    # Define missingness in percentages
    missingness_str = str(int(miss_rate*100))

    ## Load training data
    if data_name in datasets.keys():
        train_ohe_data_x = pd.read_csv('ohe_preprocessed_data/one_hot_train_data/one_hot_'+data_name+'_train.csv')    
    else:
        ValueError("Dataset not found")
    
    # Load train missing data with missingness
    if data_name in datasets.keys():   
        train_ohe_miss_data_x = pd.read_csv('ohe_preprocessed_data/one_hot_train_data/one_hot_'+data_name+'_train_'+missingness_str+'.csv')
    else:
        ValueError("Dataset not found")

    ## Load test data
    if data_name in datasets.keys():
        test_ohe_data_x = pd.read_csv('ohe_preprocessed_data/one_hot_test_data/one_hot_'+data_name+'_test.csv') 
    else:
        ValueError("Dataset not found")

    # Load test missing data with missingness
    if data_name in datasets.keys():   
        test_ohe_miss_data_x = pd.read_csv('ohe_preprocessed_data/one_hot_test_data/one_hot_'+data_name+'_test_'+missingness_str+'.csv') 
    else:
        ValueError("Dataset not found")
        
    return train_ohe_data_x, train_ohe_miss_data_x, test_ohe_data_x, test_ohe_miss_data_x

def data_loader_ohe_wo_target(data_name, miss_rate):

    '''Loads training and test datasets where the categorical columns are one hot encoded and the target column is removed, with and without missingness.
  
    Args:
    - data_name: mushroom, letter, bank, credit or news
    - miss_rate: the probability of missing components (0.1, 0.3 or 0.5)
    
    Returns:
    train_ohe_data_x: one hot encoded original data without target column
    train_ohe_miss_data_x: one hot encoded data with missing values without target column

    test_ohe_data_x: one hot encoded original data without target column
    test_ohe_miss_data_x: one hot encoded data with missing values without target column
    '''

    # Define missingness in percentages      
    missingness_str = str(int(miss_rate*100))

    ## Load training data
    if data_name in datasets.keys():
        train_ohe_data_x = pd.read_csv('ohe_preprocessed_data/one_hot_train_data_wo_target/one_hot_'+data_name+'_train.csv')    
    else:
        ValueError("Dataset not found")
    
    # Load train missing data with missingness
    if data_name in datasets.keys():   
        train_ohe_miss_data_x = pd.read_csv('ohe_preprocessed_data/one_hot_train_data_wo_target/one_hot_'+data_name+'_train_'+missingness_str+'.csv')
    else:
        ValueError("Dataset not found")

    ## Load test data
    if data_name in datasets.keys():
        test_ohe_data_x = pd.read_csv('ohe_preprocessed_data/one_hot_test_data_wo_target/one_hot_'+data_name+'_test.csv') 
    else:
        ValueError("Dataset not found")

    # Load test missing data with missingness
    if data_name in datasets.keys():   
        test_ohe_miss_data_x = pd.read_csv('ohe_preprocessed_data/one_hot_test_data_wo_target/one_hot_'+data_name+'_test_'+missingness_str+'.csv') 
    else:
        ValueError("Dataset not found")
        
    return train_ohe_data_x, train_ohe_miss_data_x, test_ohe_data_x, test_ohe_miss_data_x


def data_loader_factor(data_name, miss_rate):

    '''Loads training and test datasets where the categorical columns are factor (also known as label) encoded, with and without missingness.
  
    Args:
    - data_name: mushroom, letter, bank, credit or news
    - miss_rate: the probability of missing components (0.1, 0.3 or 0.5)
    
    Returns:
    train_factor_data_x: factor encoded original data
    train_factor_miss_data_x: factor encoded data with missing values

    test_factor_data_x: factor encoded original data
    test_factor_miss_data_x: factor encoded data with missing values
    '''

    # Define missingness in percentages   
    missingness_str = str(int(miss_rate*100))

    ## Load training data
    if data_name in datasets.keys():
        train_factor_data_x = pd.read_csv('factor_preprocessed_data/factor_train_data/factor_encode_'+data_name+'_train.csv')    
    else:
        ValueError("Dataset not found")
    
    # Load train missing data with missingness
    if data_name in datasets.keys():   
        train_factor_miss_data_x = pd.read_csv('factor_preprocessed_data/factor_train_data/factor_encode_'+data_name+'_train_'+missingness_str+'.csv')
    else:
        ValueError("Dataset not found")

    ## Load test data
    if data_name in datasets.keys():
        test_factor_data_x = pd.read_csv('factor_preprocessed_data/factor_test_data/factor_encode_'+data_name+'_test.csv') 
    else:
        ValueError("Dataset not found")

    # Load test missing data with missingness
    if data_name in datasets.keys():   
        test_factor_miss_data_x = pd.read_csv('factor_preprocessed_data/factor_test_data/factor_encode_'+data_name+'_test_'+missingness_str+'.csv') 
    else:
        ValueError("Dataset not found")
        
    return train_factor_data_x, train_factor_miss_data_x, test_factor_data_x, test_factor_miss_data_x    


def data_loader_factor_wo_target(data_name, miss_rate):

    '''Loads training and test datasets where the categorical columns are factor (also known as label) encoded and where the target variable is removed, with and without missingness.
  
    Args:
    - data_name: mushroom, letter, bank, credit or news
    - miss_rate: the probability of missing components (0.1, 0.3 or 0.5)
    
    Returns:
    train_factor_data_x: factor encoded original data without target column
    train_factor_miss_data_x: factor encoded data with missing values without target column

    test_factor_data_x: factor encoded original data without target column
    test_factor_miss_data_x: factor encoded data with missing values without target column
    '''

    # Define missingness in percentages
    missingness_str = str(int(miss_rate*100))

    ## Load training data
    if data_name in datasets.keys():
        train_factor_data_x = pd.read_csv('factor_preprocessed_data/factor_train_data_wo_target/factor_encode_'+data_name+'_train.csv')    
    else:
        ValueError("Dataset not found")
    
    # Load train missing data with correct missingness
    if data_name in datasets.keys():   
        train_factor_miss_data_x = pd.read_csv('factor_preprocessed_data/factor_train_data_wo_target/factor_encode_'+data_name+'_train_'+missingness_str+'.csv')
    else:
        ValueError("Dataset not found")

    ## Load test data
    if data_name in datasets.keys():
        test_factor_data_x = pd.read_csv('factor_preprocessed_data/factor_test_data_wo_target/factor_encode_'+data_name+'_test.csv') 
    else:
        ValueError("Dataset not found")

    # Load test missing data with correct missingness
    if data_name in datasets.keys():   
        test_factor_miss_data_x = pd.read_csv('factor_preprocessed_data/factor_test_data_wo_target/factor_encode_'+data_name+'_test_'+missingness_str+'.csv') 
    else:
        ValueError("Dataset not found")
        
    return train_factor_data_x, train_factor_miss_data_x, test_factor_data_x, test_factor_miss_data_x    


def data_loader_norm_factor_mice_imputed_data(data_name, miss_rate):

    '''Loads normalized training and test datasets where the categorical columns are factor encoded (also known as label encoded), and where the target variable is removed, that have been imputed using MICE method. 
  
    Args:
    - data_name: mushroom, letter, bank, credit or news
    - miss_rate: the probability of missing components (0.1, 0.3 or 0.5) before the data was imputed with MICE
    
    Returns:
    - train_imp_factor_norm_data_x: normalized MICE imputed training data 
    - test_imp_factor_norm_data_x: normalized MICE imputed test data 

    '''

    # Define missingness in percentages
    missingness_str = str(int(miss_rate*100))
    
    # Load normalized MICE imputed train missing data
    if data_name in datasets.keys():   
        train_imp_factor_norm_data_x = pd.read_csv('imputed_data/no_ctgan/norm_factor_imputed_mice_train_data/norm_factor_imputed_mice_'+data_name+'_train_'+missingness_str+'.csv')
    else:
        ValueError("Dataset not found")

    # Load normalized MICE imputed test missing data 
    if data_name in datasets.keys():   
        test_imp_factor_norm_data_x = pd.read_csv('imputed_data/no_ctgan/norm_factor_imputed_mice_test_data/norm_factor_imputed_mice_'+data_name+'_test_'+missingness_str+'.csv') 
    else:
        ValueError("Dataset not found")
        
    return train_imp_factor_norm_data_x, test_imp_factor_norm_data_x


def data_loader_ohe_wo_target_ctgan50(data_name, miss_rate):

    '''Loads training and test datasets where the categorical columns are one hot encoded and the target column is removed, with and without missingness. 
    The train data with missing values have been increased with 50% using CTGAN
  
    Args:
    - data_name: mushroom, letter, bank, credit or news
    - miss_rate: the probability of missing components (0.1, 0.3 or 0.5)
    
    Returns:
    train_ohe_data_x: one hot encoded original data without target column
    train_ohe_miss_data_x: one hot encoded data with missing values without target column where the data amount has been increased by 50% using ctgan

    test_ohe_data_x: one hot encoded original data without target column
    test_ohe_miss_data_x: one hot encoded data with missing values without target column
    '''

    # Define missingness in percentages      
    missingness_str = str(int(miss_rate*100))

    ## Load training data
    if data_name in datasets.keys():
        train_ohe_data_x = pd.read_csv('ohe_preprocessed_data/one_hot_train_data_wo_target/one_hot_'+data_name+'_train.csv')    
    else:
        ValueError("Dataset not found")
    
    # Load train missing data with missingness
    if data_name in datasets.keys():   
        train_ohe_miss_data_x = pd.read_csv('ohe_preprocessed_data/one_hot_train_data_wo_target_extra_50/one_hot_'+data_name+'_train_'+missingness_str+'_extra_50.csv')
    else:
        ValueError("Dataset not found")

    ## Load test data
    if data_name in datasets.keys():
        test_ohe_data_x = pd.read_csv('ohe_preprocessed_data/one_hot_test_data_wo_target/one_hot_'+data_name+'_test.csv') 
    else:
        ValueError("Dataset not found")

    # Load test missing data with missingness
    if data_name in datasets.keys():   
        test_ohe_miss_data_x = pd.read_csv('ohe_preprocessed_data/one_hot_test_data_wo_target/one_hot_'+data_name+'_test_'+missingness_str+'.csv') 
    else:
        ValueError("Dataset not found")
        
    return train_ohe_data_x, train_ohe_miss_data_x, test_ohe_data_x, test_ohe_miss_data_x

def data_loader_ohe_wo_target_ctgan100(data_name, miss_rate):

    '''Loads training and test datasets where the categorical columns are one hot encoded and the target column is removed, with and without missingness. 
    The train data with missing values have been increased with 100% using CTGAN
  
    Args:
    - data_name: mushroom, letter, bank, credit or news
    - miss_rate: the probability of missing components (0.1, 0.3 or 0.5)
    
    Returns:
    train_ohe_data_x: one hot encoded original data without target column
    train_ohe_miss_data_x: one hot encoded data with missing values without target column where the data amount has been increased by 100% using ctgan

    test_ohe_data_x: one hot encoded original data without target column
    test_ohe_miss_data_x: one hot encoded data with missing values without target column
    '''

    # Define missingness in percentages      
    missingness_str = str(int(miss_rate*100))

    ## Load training data
    if data_name in datasets.keys():
        train_ohe_data_x = pd.read_csv('ohe_preprocessed_data/one_hot_train_data_wo_target/one_hot_'+data_name+'_train.csv')    
    else:
        ValueError("Dataset not found")
    
    # Load train missing data with missingness
    if data_name in datasets.keys():   
        train_ohe_miss_data_x = pd.read_csv('ohe_preprocessed_data/one_hot_train_data_wo_target_extra_100/one_hot_'+data_name+'_train_'+missingness_str+'_extra_100.csv')
    else:
        ValueError("Dataset not found")

    ## Load test data
    if data_name in datasets.keys():
        test_ohe_data_x = pd.read_csv('ohe_preprocessed_data/one_hot_test_data_wo_target/one_hot_'+data_name+'_test.csv') 
    else:
        ValueError("Dataset not found")

    # Load test missing data with missingness
    if data_name in datasets.keys():   
        test_ohe_miss_data_x = pd.read_csv('ohe_preprocessed_data/one_hot_test_data_wo_target/one_hot_'+data_name+'_test_'+missingness_str+'.csv') 
    else:
        ValueError("Dataset not found")
        
    return train_ohe_data_x, train_ohe_miss_data_x, test_ohe_data_x, test_ohe_miss_data_x

def data_loader_factor_wo_target_ctgan50(data_name, miss_rate):

    '''Loads training and test datasets where the categorical columns are factor (also known as label) encoded and where the target variable is removed, with and without missingness.
    The training data with missing values have been increased using CTGAN. 
  
    Args:
    - data_name: mushroom, letter, bank, credit or news
    - miss_rate: the probability of missing components (0.1, 0.3 or 0.5)
    
    Returns:
    train_factor_data_x: factor encoded original data without target column
    train_factor_miss_data_x: factor encoded data with missing values without target column, where the data amount has been increased by 50% using ctgan

    test_factor_data_x: factor encoded original data without target column
    test_factor_miss_data_x: factor encoded data with missing values without target column
    '''

    # Define missingness in percentages
    missingness_str = str(int(miss_rate*100))

    ## Load training data
    if data_name in datasets.keys():
        train_factor_data_x = pd.read_csv('factor_preprocessed_data/factor_train_data_wo_target/factor_encode_'+data_name+'_train.csv')    
    else:
        ValueError("Dataset not found")
    
    # Load train missing data with correct missingness
    if data_name in datasets.keys():   
        train_factor_miss_data_x = pd.read_csv('factor_preprocessed_data/factor_train_data_wo_target_extra_50/factor_encode_'+data_name+'_train_'+missingness_str+'_extra_50.csv')
    else:
        ValueError("Dataset not found")

    ## Load test data
    if data_name in datasets.keys():
        test_factor_data_x = pd.read_csv('factor_preprocessed_data/factor_test_data_wo_target/factor_encode_'+data_name+'_test.csv') 
    else:
        ValueError("Dataset not found")

    # Load test missing data with correct missingness
    if data_name in datasets.keys():   
        test_factor_miss_data_x = pd.read_csv('factor_preprocessed_data/factor_test_data_wo_target/factor_encode_'+data_name+'_test_'+missingness_str+'.csv') 
    else:
        ValueError("Dataset not found")
        
    return train_factor_data_x, train_factor_miss_data_x, test_factor_data_x, test_factor_miss_data_x 

def data_loader_factor_wo_target_ctgan100(data_name, miss_rate):

    '''Loads training and test datasets where the categorical columns are factor (also known as label) encoded and where the target variable is removed, with and without missingness.
    The training data with missing values have been increased with 100% using CTGAN. 
  
    Args:
    - data_name: mushroom, letter, bank, credit or news
    - miss_rate: the probability of missing components (0.1, 0.3 or 0.5)
    
    Returns:
    train_factor_data_x: factor encoded original data without target column
    train_factor_miss_data_x: factor encoded data with missing values without target column, where the data amount has been increased by 100% using ctgan

    test_factor_data_x: factor encoded original data without target column
    test_factor_miss_data_x: factor encoded data with missing values without target column
    '''

    # Define missingness in percentages
    missingness_str = str(int(miss_rate*100))

    ## Load training data
    if data_name in datasets.keys():
        train_factor_data_x = pd.read_csv('factor_preprocessed_data/factor_train_data_wo_target/factor_encode_'+data_name+'_train.csv')    
    else:
        ValueError("Dataset not found")
    
    # Load train missing data with correct missingness
    if data_name in datasets.keys():   
        train_factor_miss_data_x = pd.read_csv('factor_preprocessed_data/factor_train_data_wo_target_extra_100/factor_encode_'+data_name+'_train_'+missingness_str+'_extra_100.csv')
    else:
        ValueError("Dataset not found")

    ## Load test data
    if data_name in datasets.keys():
        test_factor_data_x = pd.read_csv('factor_preprocessed_data/factor_test_data_wo_target/factor_encode_'+data_name+'_test.csv') 
    else:
        ValueError("Dataset not found")

    # Load test missing data with correct missingness
    if data_name in datasets.keys():   
        test_factor_miss_data_x = pd.read_csv('factor_preprocessed_data/factor_test_data_wo_target/factor_encode_'+data_name+'_test_'+missingness_str+'.csv') 
    else:
        ValueError("Dataset not found")
        
    return train_factor_data_x, train_factor_miss_data_x, test_factor_data_x, test_factor_miss_data_x 

def data_loader_norm_factor_mice_imputed_data_ctgan(data_name, miss_rate, ctgan):
    '''Loads normalized training and test datasets where the categorical columns are factor encoded (also known as label encoded), and where the target variable is removed, that have been imputed using MICE method. 
    The training data have been increased using ctgan.
  
    Args:
    - data_name: mushroom, letter, bank, credit or news
    - miss_rate: the probability of missing components (0.1, 0.3 or 0.5) before the data was imputed with MICE
    - ctgan: '50', '100' or None, determines how much % the training data is increased with
        "50": Training data is increased by 50%
        "100": Training data is increased by 100% 
    
    Returns:
    - train_imp_factor_norm_data_x: normalized MICE imputed training data 
    - test_imp_factor_norm_data_x: normalized MICE imputed test data 

    '''

    # Define missingness in percentages
    missingness_str = str(int(miss_rate*100))
    
    # Load normalized MICE imputed train missing data
    if data_name in datasets.keys():   
        train_imp_factor_norm_data_x = pd.read_csv('imputed_data/ctgan'+ctgan+'/norm_factor_imputed_mice_train_data_ctgan'+ctgan+'/norm_factor_imputed_mice_'+data_name+'_train_'+missingness_str+'_ctgan'+ctgan+'.csv')
    else:
        ValueError("Dataset not found")

    # Load normalized MICE imputed test missing data 
    if data_name in datasets.keys():   
        test_imp_factor_norm_data_x = pd.read_csv('imputed_data/ctgan'+ctgan+'/norm_factor_imputed_mice_test_data_ctgan'+ctgan+'/norm_factor_imputed_mice_'+data_name+'_test_'+missingness_str+'_ctgan'+ctgan+'.csv')
    else:
        ValueError("Dataset not found")
        
    return train_imp_factor_norm_data_x, test_imp_factor_norm_data_x