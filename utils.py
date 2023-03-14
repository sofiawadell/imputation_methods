
import numpy as np
import pandas as pd
from datasets import datasets

def normalization (data, parameters=None):
  '''Normalize data in [0, 1] range.
  
  Args:
    - data: original data
  
  Returns:
    - norm_data: normalized data
    - norm_parameters: min_val, max_val for each feature for renormalization
  '''

  # Parameters
  _, dim = data.shape
  norm_data = data.copy()
  
  if parameters is None:
  
    # MixMax normalization
    min_val = np.zeros(dim)
    max_val = np.zeros(dim)
    
    # For each dimension
    for i in range(dim):
      min_val[i] = np.nanmin(norm_data[:,i])
      norm_data[:,i] = norm_data[:,i] - np.nanmin(norm_data[:,i])
      max_val[i] = np.nanmax(norm_data[:,i])
      norm_data[:,i] = norm_data[:,i] / (np.nanmax(norm_data[:,i]) - min_val[i] + 1e-6)   
      
    # Return norm_parameters for renormalization
    norm_parameters = {'min_val': min_val,
                       'max_val': max_val}

  else:
    min_val = parameters['min_val']
    max_val = parameters['max_val']
    
    # For each dimension
    for i in range(dim):
      norm_data[:,i] = norm_data[:,i] - min_val[i]
      norm_data[:,i] = norm_data[:,i] / (max_val[i] - min_val[i] + 1e-6)  
      
    norm_parameters = parameters    
      
  return norm_data, norm_parameters

def normalize_numeric(data, data_name, parameters=None):
  '''Normalize the numeric columns of a data set in [0, 1] range.
  
  Args:
    - data: original data
    - data_name: name of data set
  
  Returns:
    - norm_data: full data set where the numeric columns are normalized
    - norm_parameters: min_val, max_val for each numerical feature for renormalization
  '''

  # Transform dataframe to numpy array
  data_np = data.values

  num_cols = datasets[data_name]["num_cols"]
  nbr_of_num_cols = len(num_cols)
  parameters = None

  norm_data = data_np.copy().astype(float)

  if parameters is None:

    # MixMax normalization
    min_val = np.zeros(nbr_of_num_cols)
    max_val = np.zeros(nbr_of_num_cols)

    # For each dimension
    for i in range(nbr_of_num_cols):
        min_val[i] = np.nanmin(norm_data[:,i])
        norm_data[:,i] = norm_data[:,i] - np.nanmin(norm_data[:,i])
        max_val[i] = np.nanmax(norm_data[:,i])
        norm_data[:,i] = norm_data[:,i] / (np.nanmax(norm_data[:,i]) - min_val[i])   
        
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
        norm_data[:,i] = norm_data[:,i] / (max_val[i] - min_val[i])  
    
    norm_parameters = parameters 
    norm_data_pd = pd.DataFrame(norm_data, columns=data.columns) 

  return norm_data_pd, norm_parameters

def rmse_num_loss(ori_data, imputed_data, data_m, data_name, norm_params):
  '''Compute RMSE loss between ori_data and imputed_data for numerical variables
  
  Args:
    - ori_data: original data without missing values
    - imputed_data: imputed data
    - data_m: indicator matrix for missingness
    
  Returns:
    - rmse_num: Root Mean Squared Error
  '''

  # Find number of numerical columns
  N_num_cols = len(datasets[data_name]["num_cols"])

  if N_num_cols == 0:
    return None
  else: 
    # Normalize with norm_params
    ori_data_norm, _ = normalization(ori_data, norm_params)
    imputed_data_norm, _ = normalization(imputed_data, norm_params) 

    # Extract only the numerical columns
    ori_data_norm_num = ori_data_norm[:, :N_num_cols]
    imputed_data_norm_num = imputed_data_norm[:, :N_num_cols]
    data_m_num = data_m[:, :N_num_cols]
    
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
    
  Returns:
    - rmse_cat: Root Mean Squared Error
  '''

  # Find number of columns
  N_num_cols = len(datasets[data_name]["num_cols"])   # Find number of numerical columns
  N_cat_cols = len(datasets[data_name]["cat_cols"])   # Find number of categorical columns
  
  if N_cat_cols == 0:
    return None
  else:
    # Extract only the categorical columns
    ori_data_cat = ori_data[:, N_num_cols:]
    imputed_data_cat = imputed_data[:, N_num_cols:]
    data_m_cat = data_m[:, N_num_cols:]
    
    # RMSE categorical  
    nominator = np.sum(((1-data_m_cat) * ori_data_cat - (1-data_m_cat) * imputed_data_cat)**2)
    denominator = np.sum(1-data_m_cat)
    
    rmse_cat = np.sqrt(nominator/float(denominator))
    
    return rmse_cat
  
def m_rmse_loss(rmse_num, rmse_cat):
  '''Compute mRMSE loss between ori_data and imputed_data
  
  Args:
    - ori_data: original data without missing values
    - imputed_data: imputed data
    - data_m: indicator matrix for missingness
    
  Returns:
    - m_rmse: modified Root Mean Squared Error
  '''
  if rmse_cat == None: 
    rmse_cat = 0
  if rmse_num == None:
    rmse_num = 0
  
  m_rmse = np.sqrt((rmse_num**2) + (rmse_cat**2))
    
  return m_rmse