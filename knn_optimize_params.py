import numpy as np
import pandas as pd 

from sklearn.impute import KNNImputer
from sklearn.model_selection import cross_val_score, KFold

from data_loader import data_loader_ohe_wo_target

from utils import normalize_numeric
from utils import rmse_num_loss
from utils import rmse_cat_loss
from utils import m_rmse_loss

def optimize_params(data_name, miss_rate):

    # Load OHE data without target column (only training data is used)
    train_data_x, train_miss_data_x, test_data_x, test_miss_data_x = data_loader_ohe_wo_target(data_name, miss_rate) 

    # Normalize the missing training data set using the parameters from the full data set 
    train_norm_data_x, norm_params_train_full_data = normalize_numeric(train_data_x, data_name)
    train_miss_norm_data_x, _ = normalize_numeric(train_miss_data_x, data_name, norm_params_train_full_data)

    # Transform to numpy arrays
    train_norm_data_x_np, train_miss_norm_data_x_np = train_norm_data_x.values, train_miss_norm_data_x.values

    # Define a range of values for the number of neighbors
    k_values = range(1, 3)

    # Define the number of folds for cross-validation
    n_folds = 3
    
    # Initialize a dictionary to store the RMSE scores for each k value
    rmse_scores = {}

     # Loop over the k values and perform cross-validation
    for k in k_values:
        # Initialize a kNN imputer with k neighbors
        imputer = KNNImputer(n_neighbors=k)
        
        # Use cross-validation to evaluate the imputer
        cv = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        # Calculate the RMSE between the missing and non-missing datasets for each fold
        fold_m_rmses = []
        print(k)

        for train_index, test_index in cv.split(train_miss_norm_data_x_np):
            
            # Split the data into training and test sets
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

            fold_m_rmses.append(m_rmse)
            print(m_rmse)
        
        # Calculate the mean RMSE across all folds for this k value
        rmse_scores[k] = np.mean(fold_m_rmses)
    
    # Print the RMSE scores for each k value
    for k, rmse in rmse_scores.items():
        print(f"k={k}: RMSE={rmse:.3f}")

    # Select the k value with the lowest RMSE score
    best_k = min(rmse_scores, key=rmse_scores.get)
    print(f"Best k value: {best_k}")

    return best_k

# Run code
optimize_params("bank", 0.1) 