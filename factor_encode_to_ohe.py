
from datasets import datasets
import pandas as pd
import numpy as np

'''
Description: Converts categorical columns of a data set to one hot encoded columns.
'''

def factor_encode_to_ohe(train_imputed_factor_norm_data_x, test_imputed_factor_norm_data_x, train_factor_data_x, test_factor_data_x, data_name):

    '''Transform factor encoded training and testing data sets which have been imputed using an imputation method to one hot encoded data sets.
  
    Args:
    - train_imputed_factor_norm_data_x: Normalized factor encoded train data set which originally contained missingness, but have been imputed using an imputation method
    - test_imputed_factor_norm_data_x: Normalized factor encoded test data set which originally contained missingness, but have been imputed using an imputation method
    - train_factor_data_x: Full factor encoded train data set (original, before missingness is introduced)
    - test_factor_data_x: Full factor encoded test data set (original, before missingness is introduced)
    - data_name: mushroom, letter, bank, credit or news
    
    Returns:
    - train_imputed_norm_ohe_data_x: Transformed one hot encoded train data set 
    - test_imputed_norm_ohe_data_x: Transformed one hot encoded test data set 
    '''

    # Find categorical columns of the data set
    cat_cols = datasets[data_name]["cat_cols"]

    # Merge training and testing data for the imputed and the original data
    full_data_imp = pd.concat([train_imputed_factor_norm_data_x, test_imputed_factor_norm_data_x], axis=0)
    full_data_original = pd.concat([train_factor_data_x, test_factor_data_x], axis=0)

    # Create copy of dataframes
    df_full_data_imp = full_data_imp.copy()
    df_full_data_original = full_data_original.copy()

    # Loop through each categorical column and apply one-hot encoding
    for col in cat_cols:

        # Adjusts column names to factor encoded names
        col = col + '_encoded'

        # Get unique categories for original data set, and the categories missing in the imputed data set
        categories = full_data_original[col].unique()
        missing_categories = set(categories) - set(full_data_imp[col].unique())

        # Perform one-hot encoding on the column, specifying the column order and feature name prefix
        prefix = col
        encoded_col_imp = pd.get_dummies(full_data_imp[col], prefix=prefix, columns=categories)
        encoded_col_complete = pd.get_dummies(full_data_original[col], prefix=prefix, columns=categories)

        # Add new columns to imputed dataset for any missing categories
        for category in missing_categories:
            prefix = col + '_'
            new_col = pd.Series([0] * len(full_data_original))
            new_col.name = prefix + str(category)
            encoded_col_imp[new_col.name] = new_col
                        
        # Add the encoded column(s) to the new dataframe
        df_full_data_original = pd.concat([df_full_data_original, encoded_col_complete], axis=1)
        df_full_data_imp  = pd.concat([df_full_data_imp, encoded_col_imp], axis=1)

        # Remove the original factor encoded columns from the new dataframe
        df_full_data_original.drop(col, axis=1, inplace=True)
        df_full_data_imp.drop(col, axis=1, inplace=True)

    # Split back into training and test
    train_imputed_norm_ohe_data_x, test_imputed_norm_ohe_data_x = np.vsplit(df_full_data_imp, [len(train_imputed_factor_norm_data_x)])
    
    return train_imputed_norm_ohe_data_x, test_imputed_norm_ohe_data_x