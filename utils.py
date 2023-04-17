
import numpy as np
import itertools
import pandas as pd
from datasets import datasets
import sklearn
from sklearn import metrics
from sklearn.metrics import accuracy_score

'''Description: Utility functions for imputation methods MICE, MissForest, kNNmputer and 

(1) normalize_numeric: MinMax Normalizer for numeric columns
(2) renormalize_numeric: Recover the numerical columns data from normalized data
(3) rmse_num_loss: Evaluate numerical columns of imputed data in terms of RMSE
(4) rmse_cat_loss: Evaluate one hot encoded categorical columns of imputed data in terms of RMSE
(5) m_rmse_loss: Evaluate imputed data in terms of modified RMSE
(6) pfc: Calculateds the percently falsely classified imputed values in the categorical columns

'''

def normalize_numeric(data, data_name, parameters=None):
  '''Normalize the numeric columns of a data set in [0, 1] range.
  
  Args:
    - data: original data in Pandas dataframe
    - data_name: mushroom, letter, bank, credit or news
    - parameters: parameters to use for normalization (optional)
  
  Returns:
    - norm_data_pd: full data set where the numeric columns are normalized
    - norm_parameters: min_val, max_val for each numerical feature for renormalization
  '''

  # Transform dataframe to numpy array
  data_np = data.values

  num_cols = datasets[data_name]["num_cols"]
  nbr_of_num_cols = len(num_cols)
  norm_data = data_np.copy().astype(float)

  if parameters is None:

    # MixMax normalization
    min_val = np.zeros(nbr_of_num_cols)
    max_val = np.zeros(nbr_of_num_cols)

    # For each dimension
    for i in range(nbr_of_num_cols):
      min_val[i] = np.nanmin(norm_data[:,i])
      max_val[i] = np.nanmax(norm_data[:,i])
      norm_data[:,i] = norm_data[:,i] - min_val[i]
      norm_data[:,i] = norm_data[:,i] / (max_val[i] - min_val[i] + 1e-6)   
        
    # Return norm_parameters for renormalization
    norm_parameters = {'min_val': min_val,
                        'max_val': max_val}
    
    norm_data_pd = pd.DataFrame(norm_data, columns=data.columns) 

  else:
    min_val = parameters['min_val']
    max_val = parameters['max_val']

    # For each dimension
    for i in range(nbr_of_num_cols):
        norm_data[:,i] = norm_data[:,i] - min_val[i]
        norm_data[:,i] = norm_data[:,i] / (max_val[i] - min_val[i] + 1e-6)  
    
    norm_parameters = parameters 
    norm_data_pd = pd.DataFrame(norm_data, columns=data.columns) 

  return norm_data_pd, norm_parameters


def renormalize_numeric (norm_data, norm_parameters, data_name):
  '''Renormalize numeric columns from [0, 1] range to the original range.
  
  Args:
    - norm_data: normalized data in Pandas dataframe
    - norm_parameters: min_val, max_val for each numerical feature for renormalization
    - data_name: mushroom, letter, bank, credit or news
  
  Returns:
    - renorm_data_pd: renormalized original data in a Pandas dataframe
  '''
  
  # Renormalize data 
  data_norm_np = norm_data.values #Transform to numpy

  min_val = norm_parameters['min_val']
  max_val = norm_parameters['max_val']

  num_cols = datasets[data_name]["num_cols"]
  nbr_of_num_cols = len(num_cols)
  renorm_data = data_norm_np.copy()

  for i in range(nbr_of_num_cols):
      renorm_data[:,i] = renorm_data[:,i] * (max_val[i] - min_val[i] + 1e-6)   
      renorm_data[:,i] = renorm_data[:,i] + min_val[i]

  # Change to pandas 
  renorm_data_pd = pd.DataFrame(renorm_data, columns=norm_data.columns) 
    
  return renorm_data_pd

def rmse_num_loss(ori_data_norm, imputed_data_norm, data_m, data_name):
  '''Compute RMSE loss between normalized ori_data and imputed_data for numerical variables
  
  Args:
    - ori_data_norm: normalized original data without missing values
    - imputed_data_norm: normalized imputed data
    - data_m: indicator matrix for missingness in the imputed data
    - data_name: mushroom, letter, bank, credit or news
    
  Returns:
    - rmse_num: Root Mean Squared Error
  '''

  # Find number of numerical columns
  N_num_cols = len(datasets[data_name]["num_cols"])

  # Ensure data set is in Numpy 
  ori_data_norm_np = ori_data_norm.values 
  imputed_data_norm_np = imputed_data_norm.values 
  data_m_np = data_m.values

  if N_num_cols == 0:
    return None
  else: 
    
    # Extract only the numerical columns
    ori_data_norm_num = ori_data_norm_np[:, :N_num_cols]
    imputed_data_norm_num = imputed_data_norm_np[:, :N_num_cols]
    data_m_num = data_m_np[:, :N_num_cols]
    
    # Calculate RMSE numerical   
    nominator = np.sum(((1-data_m_num) * ori_data_norm_num - (1-data_m_num) * imputed_data_norm_num)**2)
    denominator = np.sum(1-data_m_num)
    
    rmse_num = np.sqrt(nominator/float(denominator))
    
    return rmse_num

def rmse_cat_loss(ori_data, imputed_data, data_m, data_name):
  '''Compute RMSE loss between ori_data and imputed_data for categorical variables
  
  Args:
    - ori_data: original data without missing values
    - imputed_data: imputed data
    - data_m: indicator matrix for missingness
    - data_name: mushroom, letter, bank, credit or news
    
  Returns:
    - rmse_cat: Root Mean Squared Error
  '''

  # Find number of columns
  N_num_cols = len(datasets[data_name]["num_cols"])   # Find number of numerical columns
  N_cat_cols = len(datasets[data_name]["cat_cols"])   # Find number of categorical columns
  
  # Ensure data set is in Numpy 
  ori_data_np = ori_data.values 
  imputed_data_np = imputed_data.values 
  data_m_np = data_m.values

  if N_cat_cols == 0:
    return None
  else:

    # Extract only the categorical columns
    ori_data_cat = ori_data_np[:, N_num_cols:]
    imputed_data_cat = imputed_data_np[:, N_num_cols:]
    data_m_cat = data_m_np[:, N_num_cols:]
    
    # RMSE categorical  
    nominator = np.sum(((1-data_m_cat) * ori_data_cat - (1-data_m_cat) * imputed_data_cat)**2)
    denominator = np.sum(1-data_m_cat)
    
    rmse_cat = np.sqrt(nominator/float(denominator))
    
    return rmse_cat
  
def m_rmse_loss(rmse_num, rmse_cat):
  '''Compute mRMSE loss between ori_data and imputed_data
  
  Args:
    - rmse_num: RMSE for numerical columns
    - rmse_cat: RMSE for categorical columns
    
  Returns:
    - m_rmse: modified Root Mean Squared Error
  '''
  if rmse_cat == None: 
    rmse_cat = 0
  if rmse_num == None:
    rmse_num = 0
  
  m_rmse = np.sqrt((rmse_num**2) + (rmse_cat**2))
    
  return m_rmse

def pfc(ori_data, imputed_data, data_m, data_name): # No taking into consideration category belonging now, to be fixed
  '''Compute PFC between ori_data and imputed_data
  
  Args:
    - ori_data: original data without missing values
    - imputed_data: imputed data
    - data_m: indicator matrix for missingness
    - data_name: mushroom, letter, bank, credit or news
    
  Returns:
    - pfc: Proportion Falsely Classified
  '''
  # Find number of columns
  # Find number of columns
  N_num_cols = len(datasets[data_name]["num_cols"])   # Find number of numerical columns
  N_cat_cols = len(datasets[data_name]["cat_cols"])   # Find number of categorical columns
  
  # Ensure data set is in Numpy 
  ori_data_np = ori_data.values 
  imputed_data_np = imputed_data.values 
  data_m_np = data_m.values

  if N_cat_cols == 0:
    return None
  else: 
    # Extract only the categorical columns
    ori_data_cat = ori_data_np[:, N_num_cols:]
    imputed_data_cat = imputed_data_np[:, N_num_cols:]
    data_m_cat = data_m_np[:, N_num_cols:]

    data_m_bool = ~data_m_cat.astype(bool) # True indicates missing value (=0), False indicates non-missing value (=1)

    N_missing = np.count_nonzero(data_m_cat == 0) # 0 = missing value
    N_correct = np.sum(ori_data_cat[data_m_bool] == imputed_data_cat[data_m_bool])

    # Calculate PFC
    pfc = (1 - (N_correct/N_missing))*100 # Number of incorrect / Number total missing
    
    return pfc
  

def rounding_discrete(imputed_data, data_x, data_name):
  '''Round imputed data for categorical variables. 
  Ensure to only get one "1" per categorical feature.

  Args:
    - imputed_data: imputed data
    - data_x: original data with missing values
    
  Returns:
    - rounded_data: rounded imputed data
  '''
  no, dim = data_x.shape
  rounded_data = imputed_data.copy()

  n_num_cols = len(datasets[data_name]["num_cols"])
  cat_cols = datasets[data_name]["cat_cols"]

  if (len(cat_cols) == 0):
    return rounded_data

  # Add the start indexes for each categorical feature
  cat_cols_start_indexes = cat_cols.copy()
  cumulative_sums = [0] + list(itertools.accumulate(cat_cols_start_indexes.values()))
  start_indexes = [x + n_num_cols for x in cumulative_sums[:-1]]

  for i, feature_name in enumerate(cat_cols_start_indexes.keys()):
    start_index = start_indexes[i]
    cat_cols_start_indexes[feature_name] = start_index

  # Loop through each value in the matrix
  row = 0
  while row < no:
      col = n_num_cols
      while col < dim:
          # check if the current value is NaN
          if np.isnan(data_x[row, col]):        
              for feature_name, index_value in cat_cols_start_indexes.items():
                if index_value == col: # We found the correct feature
                  n_categories = cat_cols[feature_name] 
                  break

              # Extract the current value and the next n_categories-1 values
              values = imputed_data[row, col:col+n_categories]
              
              # Find the index of the maximum value
              max_index = np.argmax(values)

              # Set the maximum value to 1 and the rest to 0 in rounded_data
              rounded_data[row, col:col+n_categories] = 0
              rounded_data[row, col+max_index] = 1
              
              # skip the next n_categories values
              col += n_categories - 1
          col += 1
      row += 1
        
  return rounded_data

def find_average_and_st_dev(values):
  '''Finding the average and standard deviation along a vector of values.
  
  Args:
    - values: vector of values
    
  Returns:
    - average_value
    - st_dev
  '''
  if all(x is None for x in values):
    return None, None
  
  average_value = np.mean(values)
  st_dev = np.std(values)

  return average_value, st_dev

def round_if_not_none(x):
    '''Round if not none
    
    Args:
      - x: value
      
    Returns:
      - rounded_value'''
    if x is not None:
        return round(x, 4)
    return None