from data_loader import data_loader_factor_wo_target
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, KFold

def optimize_params(data_name):

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
        fold_rmses = []
        print(k)

        for train_index, test_index in cv.split(train_miss_data_x_np):
            
            # Split the data into training and test sets
            X_train, X_test = train_miss_data_x_np[train_index], train_miss_data_x_np[test_index]
            y_test = train_data_x_np[test_index]
            
            # Impute the missing values in the test set using the training set
            imputer.fit(X_train)
            X_test_imputed = imputer.transform(X_test)
            
            # Calculate the RMSE between the imputed test set and the non-missing test set
            cat_rmse = np.sqrt(mean_squared_error(y_test, X_test_imputed))
            num_rmse = 1

            fold_rmses.append(np.sqrt(cat_rmse**2 + num_rmse**2))

            print(np.sqrt(mean_squared_error(y_test, X_test_imputed)))
        
        # Calculate the mean RMSE across all folds for this k value
        rmse_scores[k] = np.mean(fold_rmses)


    best_k = 2
    return best_k
