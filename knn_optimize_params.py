import numpy as np
import pandas as pd 

from sklearn.impute import KNNImputer
from sklearn.model_selection import KFold

from data_loader import data_loader_ohe_wo_target

from utils import normalize_numeric
from utils import rmse_num_loss
from utils import rmse_cat_loss
from utils import m_rmse_loss

''' Description

Determines the optimal number of neighbors for kNNImputer for a specific a data set across all levels of missingnesses, based on modified rmse

'''

# Input parameters
data_name = "news" 
k_values = range(59,34)
# k_values = []
# for i in range(7,50):
#     k_values.append(i*3)

def calculate_RMSE(data_name, miss_rate, k):

    ''' Calculates the RMSE for kNNImputer for a specific a data set with missingness, based on modified rmse
  
    Args:
    - data_name: mushroom, letter, bank, credit or news
    - miss_rate: the probability of missing components (0.1, 0.3 or 0.5)
    - k: Number of neighbors 
    
    Returns:
    - rmse_score: Modified RMSE for the dataset data_name, with missingness miss_rate and k neighbors

    '''
    
    # Load OHE data without target column (only training data is used)
    train_data_x, train_miss_data_x, _, _ = data_loader_ohe_wo_target(data_name, miss_rate) 

    # Normalize the missing training data set using the parameters from the full data set 
    train_norm_data_x, norm_params_train_full_data = normalize_numeric(train_data_x, data_name)
    train_miss_norm_data_x, _ = normalize_numeric(train_miss_data_x, data_name, norm_params_train_full_data)

    # Transform training data and training missing data to numpy arrays
    train_norm_data_x_np, train_miss_norm_data_x_np = train_norm_data_x.values, train_miss_norm_data_x.values

    # Define the number of folds for cross-validation
    n_folds = 5
    
    # Initialize a dictionary to store the modified RMSE scores for the k value
    rmse_score = 0

    # Initialize a kNN imputer with k neighbors
    imputer = KNNImputer(n_neighbors=k)
    
    # Use cross-validation to evaluate the imputer
    cv = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # Placeholders to calculate the modified RMSE between the missing and non-missing datasets for each fold
    fold_m_rmses = []

    for train_index, test_index in cv.split(train_miss_norm_data_x_np):
        
        # Split the original train data into (new) training and test sets in order to perform cross validation
        miss_data_train, miss_data_test = train_miss_norm_data_x_np[train_index], train_miss_norm_data_x_np[test_index]
        full_data_test = train_norm_data_x_np[test_index]

        # Find the mask matrix for the test data set
        mask_test = pd.DataFrame(1-np.isnan(miss_data_test))
        
        # Impute the missing values in the test set using the model fitted on the training set
        imputer.fit(miss_data_train)
        miss_data_test_imputed = imputer.transform(miss_data_test)
        
        # Calculate the modified RMSE between the imputed test set and the non-missing test set
        rmse_num = rmse_num_loss(pd.DataFrame(full_data_test), pd.DataFrame(miss_data_test_imputed), mask_test, data_name)
        rmse_cat = rmse_cat_loss(pd.DataFrame(full_data_test), pd.DataFrame(miss_data_test_imputed), mask_test, data_name)
        m_rmse = m_rmse_loss(rmse_num, rmse_cat)

        # Save and print the modified rmse
        fold_m_rmses.append(m_rmse)
    
    # Calculate the mean RMSE across all folds for this k value
    rmse_score = np.mean(fold_m_rmses)
    print(f"Missingness {int(miss_rate*100)}%: RMSE {rmse_score}")
    return rmse_score


def optimize_parameters(data_name, k_values):
    ''' Determines the optmal number of neighbors for kNNImputer for a specific a data set, based on modified rmse
  
    Args:
    - data_name: mushroom, letter, bank, credit or news
    - k_values: Possible range of neighbors 
    
    Returns:
    - best_k: Optimal number of neighbors based on modified RMSE

    '''

    # Initialize dictionaries to store RMSE values for each data set
    rmse_data_set_10 = {}
    rmse_data_set_30 = {}
    rmse_data_set_50 = {}

    # Initialize dictionary to store average RMSE values
    avg_rmse = {}

    # Iterate over RMSE dictionaries and calculate average RMSE for each k value
    for k in k_values:
        print(f"k={k}")
        rmse_data_set_10[k] = calculate_RMSE(data_name, 0.1, k)
        rmse_data_set_30[k] = calculate_RMSE(data_name, 0.3, k)
        rmse_data_set_50[k] = calculate_RMSE(data_name, 0.5, k)
        avg_rmse[k] = (rmse_data_set_10[k] + rmse_data_set_30[k] + rmse_data_set_50[k]) / 3
        print(f"Average RMSE={avg_rmse[k]:.5f}")

    # Summary: print the mean RMSE scores for each k value
    for k, rmse in avg_rmse.items():
        print(f"k={k}: Average RMSE={rmse:.5f}")

    # Select the k value with the lowest RMSE score
    best_k = min(avg_rmse, key=avg_rmse.get)
    print(f"Best k value: {best_k}")
    return best_k

best_k = optimize_parameters(data_name, k_values)